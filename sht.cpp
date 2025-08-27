#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <sys/mman.h>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <memory>
#include <limits>
#include <utility>
#include <cstring>
#include <cassert>

// --- Portable Endian Swap ---
// Use compiler built-ins for performance and portability (GCC, Clang, MSVC)
#if defined(_MSC_VER)
#include <stdlib.h>
#define PORTABLE_BSWAP64(x) _byteswap_uint64(x)
#else
#define PORTABLE_BSWAP64(x) __builtin_bswap64(x)
#endif

// Check endianness at compile time
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ || defined(__BIG_ENDIAN__)
    inline uint64_t portable_htobe64(uint64_t x) { return x; }
#else
    inline uint64_t portable_htobe64(uint64_t x) { return PORTABLE_BSWAP64(x); }
#endif


// --- Tagged Index Implementation ---
// We use the top 2 bits of the 64-bit slot for tags.
// 00: Empty
// 01: Pointer to a Node
// 10: Pointer to a StaxRecord (leaf)
// 11: Inline key (for 1D data, not used in this impl.)
namespace Tag {
    constexpr uint64_t MASK = 3ULL << 62;
    constexpr uint64_t EMPTY = 0ULL << 62;
    constexpr uint64_t NODE = 1ULL << 62;
    constexpr uint64_t RECORD = 2ULL << 62;
    constexpr uint64_t OFFSET_MASK = ~(MASK);
};

// --- Heuristics Enum ---
enum class SplittingHeuristic {
    LEXICOGRAPHICAL,
    K_DIMENSIONAL_CYCLIC
};


// --- Data Structures ---

// Represents a leaf in the tree, storing the actual data point and an optional value.
// To be allocated by the MemoryManager.
struct StaxRecord {
    uint32_t value_len;
    // For simplicity in this benchmark, value is not stored.
    // In a real scenario, a value would be here or a pointer to it.
    uint64_t coords[]; // Flexible array member for k-dimensional data

    static size_t get_size(uint32_t dim) {
        return sizeof(StaxRecord) + sizeof(uint64_t) * dim;
    }
};

// The core node structure. A 16-way fanout using nibbles.
struct Node {
    // 16 * 8 = 128 bytes for children pointers
    std::atomic<uint64_t> slots[16];

    // 16 bytes for metadata
    union {
        // For LEXICOGRAPHICAL heuristic (path compression)
        struct {
            uint32_t test_idx; // The nibble index to test
            uint32_t r_offset; // Offset to a representative record
        } lexico;

        // For K_DIMENSIONAL_CYCLIC heuristic
        struct {
            uint32_t depth; // The depth of the node, to determine split_dim = depth % D
            uint32_t r_offset; // Offset to a representative record
        } k_cyclic;
    } meta;

    Node() {
        memset(&meta, 0, sizeof(meta));
        for(size_t i = 0; i < 16; ++i) {
            slots[i].store(0, std::memory_order_relaxed);
        }
    }
};
static_assert(sizeof(Node) == 136, "Node should be 136 bytes");


// A simple memory manager using a single mmap-ed region.
class MemoryManager {
public:
    explicit MemoryManager(size_t size) : allocation_size(size) {
        base_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base_ptr == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap memory");
        }
        // Use atomic to ensure thread-safe allocations
        current_offset.store(0);
        end_offset = size;
        allocated_bytes = 0;
    }

    ~MemoryManager() {
        if (base_ptr != MAP_FAILED) {
            munmap(base_ptr, allocation_size);
        }
    }

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // Thread-safe allocation
    uint64_t alloc(size_t size) {
        size_t aligned_size = (size + 7) & ~7;
        uint64_t offset = current_offset.fetch_add(aligned_size, std::memory_order_relaxed);
        if (offset + aligned_size > end_offset) {
            current_offset.fetch_sub(aligned_size, std::memory_order_relaxed); // Revert
            throw std::bad_alloc();
        }
        void* mem = get_ptr(offset);
        memset(mem, 0, aligned_size);
        allocated_bytes += aligned_size;
        return offset;
    }

    void* get_ptr(uint64_t offset) const {
        return static_cast<uint8_t*>(base_ptr) + offset;
    }

    size_t get_allocated_size() const {
        return allocated_bytes;
    }

private:
    void* base_ptr = nullptr;
    size_t allocation_size = 0;
    std::atomic<uint64_t> current_offset;
    uint64_t end_offset;
    std::atomic<size_t> allocated_bytes;
};


// The main data structure class.
// This is the "Lazy Radix Tree" that will be enhanced.
class LayeredSlotMap {
public:
    LayeredSlotMap(MemoryManager& mem_manager, uint32_t dimensionality, SplittingHeuristic heuristic)
        : mem(mem_manager), dim(dimensionality), heuristic_(heuristic)
    {
        uint64_t root_node_offset = mem.alloc(sizeof(Node));
        root_ptr.store(root_node_offset | Tag::NODE);
    }

    void insert(const uint64_t* coords, uint32_t value_len, const void* value) {
        if (heuristic_ == SplittingHeuristic::LEXICOGRAPHICAL) {
            insert_lex(coords, value_len, value);
        } else {
            insert_kd(coords, value_len, value);
        }
    }

    bool get(const uint64_t* coords, std::string& value) {
        if (heuristic_ == SplittingHeuristic::LEXICOGRAPHICAL) {
            return get_lex(coords, value);
        } else {
            return get_kd(coords, value);
        }
    }

    void scan(const uint64_t* start_coords, const uint64_t* end_coords, std::vector<StaxRecord*>& results) const {
        // For now, only implemented for LEXICOGRAPHICAL
        if (heuristic_ != SplittingHeuristic::LEXICOGRAPHICAL) return;

        struct ScanFrame {
            uint64_t tagged_off;
            bool lower_tight;
            bool upper_tight;
        };

        std::vector<ScanFrame> stack;
        stack.push_back({root_ptr.load(std::memory_order_acquire), true, true});

        while(!stack.empty()) {
            ScanFrame frame = stack.back();
            stack.pop_back();

            uint64_t tag = frame.tagged_off & Tag::MASK;
            uint64_t offset = frame.tagged_off & Tag::OFFSET_MASK;

            if (tag == Tag::EMPTY) continue;

            if (tag == Tag::RECORD) {
                StaxRecord* rec = static_cast<StaxRecord*>(mem.get_ptr(offset));
                if ((!frame.lower_tight || memcmp(rec->coords, start_coords, dim * sizeof(uint64_t)) >= 0) &&
                    (!frame.upper_tight || memcmp(rec->coords, end_coords, dim * sizeof(uint64_t)) <= 0)) {
                    results.push_back(rec);
                }
                continue;
            }

            if (tag == Tag::NODE) {
                Node* node = static_cast<Node*>(mem.get_ptr(offset));
                uint32_t test_idx = node->meta.lexico.test_idx;
                int start_nibble = frame.lower_tight ? get_nibble_from_coords(start_coords, test_idx) : 0;
                int end_nibble = frame.upper_tight ? get_nibble_from_coords(end_coords, test_idx) : 15;

                for (int i = start_nibble; i <= end_nibble; ++i) {
                    uint64_t child_tagged_off = node->slots[i].load(std::memory_order_acquire);
                    if (child_tagged_off == 0) continue;

                    bool next_lower_tight = frame.lower_tight && (i == start_nibble);
                    bool next_upper_tight = frame.upper_tight && (i == end_nibble);
                    stack.push_back({child_tagged_off, next_lower_tight, next_upper_tight});
                }
            }
        }
    }

    size_t get_mem_usage() const {
        return mem.get_allocated_size();
    }

private:
    // --- Helper functions for nibble-based traversal ---
    static int get_nibble(uint64_t key, uint32_t nibble_idx) {
        // For k-d splitting, we operate on the value directly.
        // For lexicographical, we assume key is already big-endian.
        const uint32_t shift = 60 - (nibble_idx * 4);
        return (key >> shift) & 0x0F;
    }

    static int get_nibble_from_coords(const uint64_t* coords, uint32_t nibble_idx) {
        // Assumes keys are in big-endian format for lexicographical comparison
        const size_t byte_idx = nibble_idx / 2;
        const uint8_t byte = reinterpret_cast<const uint8_t*>(coords)[byte_idx];
        const uint32_t shift_amount = (1 - (nibble_idx & 1)) * 4;
        return (byte >> shift_amount) & 0x0F;
    }

    static int find_first_differing_nibble(const uint64_t* key1, const uint64_t* key2, uint32_t dim) {
        const char* k1 = reinterpret_cast<const char*>(key1);
        const char* k2 = reinterpret_cast<const char*>(key2);
        const size_t len = dim * sizeof(uint64_t);
        size_t byte_idx = 0;

        while (byte_idx < len && k1[byte_idx] == k2[byte_idx]) {
            byte_idx++;
        }

        if (byte_idx == len) return -1; // Keys are identical

        uint8_t b1 = k1[byte_idx];
        uint8_t b2 = k2[byte_idx];
        if ((b1 >> 4) != (b2 >> 4)) {
            return byte_idx * 2;
        }
        return byte_idx * 2 + 1;
    }


    void insert_kd(const uint64_t* coords, uint32_t value_len, const void* value) {
    restart_kd:
        std::atomic<uint64_t>* parent_slot = &root_ptr;
        uint64_t current_tagged_off = root_ptr.load(std::memory_order_acquire);
        uint32_t depth = 0;

        while (true) {
            uint64_t tag = current_tagged_off & Tag::MASK;
            uint64_t offset = current_tagged_off & Tag::OFFSET_MASK;

            if (tag == Tag::EMPTY) {
                size_t record_size = StaxRecord::get_size(dim);
                uint64_t new_rec_off = mem.alloc(record_size);
                StaxRecord* new_rec = static_cast<StaxRecord*>(mem.get_ptr(new_rec_off));
                memcpy(new_rec->coords, coords, dim * sizeof(uint64_t));
                new_rec->value_len = value_len;

                uint64_t new_slot_val = new_rec_off | Tag::RECORD;
                uint64_t expected_slot_val = 0;
                if (parent_slot->compare_exchange_strong(expected_slot_val, new_slot_val, std::memory_order_release, std::memory_order_relaxed)) {
                    return;
                }
                goto restart_kd;
            }

            if (tag == Tag::RECORD) {
                StaxRecord* existing_rec = static_cast<StaxRecord*>(mem.get_ptr(offset));
                if (memcmp(coords, existing_rec->coords, dim * sizeof(uint64_t)) == 0) {
                    return; // Key exists
                }

                size_t record_size = StaxRecord::get_size(dim);
                uint64_t new_rec_off = mem.alloc(record_size);
                StaxRecord* new_rec = static_cast<StaxRecord*>(mem.get_ptr(new_rec_off));
                memcpy(new_rec->coords, coords, dim * sizeof(uint64_t));

                uint32_t split_depth = depth;
                uint64_t top_node_off = mem.alloc(sizeof(Node));
                Node* parent_node = new (mem.get_ptr(top_node_off)) Node();
                parent_node->meta.k_cyclic.depth = split_depth;
                parent_node->meta.k_cyclic.r_offset = new_rec_off;

                while (true) {
                    uint32_t split_dim = split_depth % dim;
                    uint32_t chunk_idx = split_depth / dim;
                    if (chunk_idx >= 16) throw std::runtime_error("Max depth exceeded");

                    int new_nibble = get_nibble(coords[split_dim], chunk_idx);
                    int old_nibble = get_nibble(existing_rec->coords[split_dim], chunk_idx);

                    if (new_nibble != old_nibble) {
                        parent_node->slots[new_nibble].store(new_rec_off | Tag::RECORD, std::memory_order_relaxed);
                        parent_node->slots[old_nibble].store(current_tagged_off, std::memory_order_relaxed);
                        break;
                    }

                    uint64_t intermediate_node_off = mem.alloc(sizeof(Node));
                    Node* intermediate_node = new (mem.get_ptr(intermediate_node_off)) Node();
                    parent_node->slots[new_nibble].store(intermediate_node_off | Tag::NODE, std::memory_order_relaxed);

                    parent_node = intermediate_node;
                    split_depth++;
                    parent_node->meta.k_cyclic.depth = split_depth;
                    parent_node->meta.k_cyclic.r_offset = new_rec_off;
                }

                if (parent_slot->compare_exchange_strong(current_tagged_off, top_node_off | Tag::NODE, std::memory_order_release, std::memory_order_relaxed)) {
                    return;
                }
                goto restart_kd;
            }

            if (tag == Tag::NODE) {
                Node* node = static_cast<Node*>(mem.get_ptr(offset));
                depth = node->meta.k_cyclic.depth;
                uint32_t split_dim = depth % dim;
                uint32_t chunk_idx = depth / dim;
                if (chunk_idx >= 16) throw std::runtime_error("Max depth exceeded on traversal");
                int nibble = get_nibble(coords[split_dim], chunk_idx);
                parent_slot = &node->slots[nibble];
                current_tagged_off = parent_slot->load(std::memory_order_acquire);
                depth++;
            }
        }
    }

    bool get_kd(const uint64_t* coords, std::string& value) {
        uint64_t current_tagged_off = root_ptr.load(std::memory_order_acquire);
        uint32_t depth = 0;
        while (true) {
            uint64_t tag = current_tagged_off & Tag::MASK;
            uint64_t offset = current_tagged_off & Tag::OFFSET_MASK;

            if (tag == Tag::EMPTY) return false;
            if (tag == Tag::RECORD) {
                StaxRecord* rec = static_cast<StaxRecord*>(mem.get_ptr(offset));
                return memcmp(coords, rec->coords, dim * sizeof(uint64_t)) == 0;
            }
            if (tag == Tag::NODE) {
                Node* node = static_cast<Node*>(mem.get_ptr(offset));
                depth = node->meta.k_cyclic.depth;
                uint32_t split_dim = depth % dim;
                uint32_t chunk_idx = depth / dim;
                 if (chunk_idx >= 16) return false;
                int nibble = get_nibble(coords[split_dim], chunk_idx);
                current_tagged_off = node->slots[nibble].load(std::memory_order_acquire);
                depth++;
            }
        }
        return false;
    }

    void insert_lex(const uint64_t* coords, uint32_t value_len, const void* value) {
    restart_lex:
        std::atomic<uint64_t>* parent_slot = &root_ptr;
        uint64_t current_tagged_off = root_ptr.load(std::memory_order_acquire);

        while (true) {
            uint64_t tag = current_tagged_off & Tag::MASK;
            uint64_t offset = current_tagged_off & Tag::OFFSET_MASK;

            if (tag == Tag::EMPTY) {
                size_t record_size = StaxRecord::get_size(dim);
                uint64_t new_rec_off = mem.alloc(record_size);
                StaxRecord* new_rec = static_cast<StaxRecord*>(mem.get_ptr(new_rec_off));
                memcpy(new_rec->coords, coords, dim * sizeof(uint64_t));
                new_rec->value_len = value_len;

                uint64_t new_slot_val = new_rec_off | Tag::RECORD;
                uint64_t expected_slot_val = 0;
                if (parent_slot->compare_exchange_strong(expected_slot_val, new_slot_val, std::memory_order_release, std::memory_order_relaxed)) {
                    return;
                }
                goto restart_lex;
            }

            if (tag == Tag::RECORD) {
                StaxRecord* existing_rec = static_cast<StaxRecord*>(mem.get_ptr(offset));
                int diff_nib_idx = find_first_differing_nibble(coords, existing_rec->coords, dim);
                if (diff_nib_idx == -1) return;

                size_t record_size = StaxRecord::get_size(dim);
                uint64_t new_rec_off = mem.alloc(record_size);
                StaxRecord* new_rec = static_cast<StaxRecord*>(mem.get_ptr(new_rec_off));
                memcpy(new_rec->coords, coords, dim * sizeof(uint64_t));
                new_rec->value_len = value_len;

                uint64_t new_node_off = mem.alloc(sizeof(Node));
                Node* new_node = new (mem.get_ptr(new_node_off)) Node();
                new_node->meta.lexico.test_idx = diff_nib_idx;
                new_node->meta.lexico.r_offset = new_rec_off;

                int new_key_nibble = get_nibble_from_coords(coords, diff_nib_idx);
                int existing_key_nibble = get_nibble_from_coords(existing_rec->coords, diff_nib_idx);

                new_node->slots[new_key_nibble].store(new_rec_off | Tag::RECORD, std::memory_order_relaxed);
                new_node->slots[existing_key_nibble].store(current_tagged_off, std::memory_order_relaxed);

                uint64_t new_node_tagged_off = new_node_off | Tag::NODE;
                if (parent_slot->compare_exchange_strong(current_tagged_off, new_node_tagged_off, std::memory_order_release, std::memory_order_relaxed)) {
                    return;
                }
                goto restart_lex;
            }

            if (tag == Tag::NODE) {
                Node* current_node = static_cast<Node*>(mem.get_ptr(offset));
                uint32_t test_idx = current_node->meta.lexico.test_idx;
                StaxRecord* rep_rec = static_cast<StaxRecord*>(mem.get_ptr(current_node->meta.lexico.r_offset));
                int diff_nib_idx = find_first_differing_nibble(coords, rep_rec->coords, dim);

                if (diff_nib_idx != -1 && (uint32_t)diff_nib_idx < test_idx) {
                    size_t record_size = StaxRecord::get_size(dim);
                    uint64_t new_rec_off = mem.alloc(record_size);
                    StaxRecord* new_rec = static_cast<StaxRecord*>(mem.get_ptr(new_rec_off));
                    memcpy(new_rec->coords, coords, dim * sizeof(uint64_t));
                    new_rec->value_len = value_len;

                    uint64_t new_node_off = mem.alloc(sizeof(Node));
                    Node* new_node = new (mem.get_ptr(new_node_off)) Node();
                    new_node->meta.lexico.test_idx = diff_nib_idx;
                    new_node->meta.lexico.r_offset = new_rec_off;

                    int new_key_nibble = get_nibble_from_coords(coords, diff_nib_idx);
                    int existing_key_nibble = get_nibble_from_coords(rep_rec->coords, diff_nib_idx);

                    new_node->slots[new_key_nibble].store(new_rec_off | Tag::RECORD, std::memory_order_relaxed);
                    new_node->slots[existing_key_nibble].store(current_tagged_off, std::memory_order_relaxed);

                    uint64_t new_node_tagged_off = new_node_off | Tag::NODE;
                    if (parent_slot->compare_exchange_strong(current_tagged_off, new_node_tagged_off, std::memory_order_release, std::memory_order_relaxed)) {
                        return;
                    }
                    goto restart_lex;
                } else {
                    int nibble = get_nibble_from_coords(coords, test_idx);
                    parent_slot = &current_node->slots[nibble];
                    current_tagged_off = parent_slot->load(std::memory_order_acquire);
                }
            }
        }
    }

    bool get_lex(const uint64_t* coords, std::string& value) {
        uint64_t current_tagged_off = root_ptr.load(std::memory_order_acquire);
        while(true) {
            uint64_t tag = current_tagged_off & Tag::MASK;
            uint64_t offset = current_tagged_off & Tag::OFFSET_MASK;

            if (tag == Tag::EMPTY) return false;

            if (tag == Tag::RECORD) {
                StaxRecord* rec = static_cast<StaxRecord*>(mem.get_ptr(offset));
                return memcmp(coords, rec->coords, dim * sizeof(uint64_t)) == 0;
            }

            if (tag == Tag::NODE) {
                Node* node = static_cast<Node*>(mem.get_ptr(offset));
                uint32_t test_idx = node->meta.lexico.test_idx;
                int nibble = get_nibble_from_coords(coords, test_idx);
                current_tagged_off = node->slots[nibble].load(std::memory_order_acquire);
            }
        }
        return false;
    }

    MemoryManager& mem;
    std::atomic<uint64_t> root_ptr;
    const uint32_t dim;
    const SplittingHeuristic heuristic_;

    // Helper functions to be added here
};

void run_lex_benchmark_1d(size_t num_keys, const std::string& key_type) {
    const size_t SHT_MEM_SIZE = num_keys * 200; // Estimate memory usage
    MemoryManager mem(SHT_MEM_SIZE);
    LayeredSlotMap lsm(mem, 1, SplittingHeuristic::LEXICOGRAPHICAL);

    std::cout << "\n\n--- Benchmark: 1D Lexicographical (Path-Compressed) ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    std::vector<uint64_t> keys(num_keys);
    std::iota(keys.begin(), keys.end(), 1);

    if (key_type == "Random") {
        std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::shuffle(keys.begin(), keys.end(), rng);
    }

    // --- Insert Benchmark ---
    std::cout << "\n--- INSERTION ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) {
        uint64_t be_key = portable_htobe64(key);
        lsm.insert(&be_key, 0, nullptr);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_insert = (double)duration.count() / num_keys;
    std::cout << "[LayeredSlotMap]      Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_insert << " ns/insert" << std::endl;

    // --- Get Benchmark ---
    std::cout << "\n--- LOOKUP ---" << std::endl;
    if (key_type == "Random") {
        std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::shuffle(keys.begin(), keys.end(), rng);
    }
    size_t found_count = 0;
    std::string dummy_val;
    start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) {
        uint64_t be_key = portable_htobe64(key);
        if (lsm.get(&be_key, dummy_val)) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_get = (double)duration.count() / num_keys;
    std::cout << "[LayeredSlotMap]      Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_get << " ns/get" << std::endl;
    if (found_count != num_keys) {
        std::cerr << "1D GET VERIFICATION FAILED! Found " << found_count << "/" << num_keys << std::endl;
    } else {
        std::cout << "1D GET VERIFICATION PASSED!" << std::endl;
    }

    // --- Scan Benchmark ---
    const size_t SCAN_SIZE = 1000;
    if (num_keys > SCAN_SIZE && key_type == "Sequential") {
        std::cout << "\n--- RANGE SCAN ---" << std::endl;
        uint64_t scan_start_key = (num_keys / 2);
        uint64_t scan_end_key = scan_start_key + SCAN_SIZE - 1;
        std::vector<StaxRecord*> scan_results;
        scan_results.reserve(SCAN_SIZE);

        uint64_t be_scan_start = portable_htobe64(scan_start_key);
        uint64_t be_scan_end = portable_htobe64(scan_end_key);

        start_time = std::chrono::high_resolution_clock::now();
        lsm.scan(&be_scan_start, &be_scan_end, scan_results);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double ns_per_scan_key = scan_results.empty() ? 0 : (double)duration.count() / scan_results.size();

        std::cout << "[LayeredSlotMap]      Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_scan_key << " ns/key (for a scan of " << scan_results.size() << " keys)" << std::endl;
        if (scan_results.size() != SCAN_SIZE) {
            std::cerr << "SCAN VERIFICATION FAILED! Found " << scan_results.size() << "/" << SCAN_SIZE << std::endl;
        } else {
            std::cout << "SCAN VERIFICATION PASSED!" << std::endl;
        }
    }

    // --- MEMORY USAGE ---
    std::cout << "\n--- MEMORY USAGE ---" << std::endl;
    size_t mem_used = lsm.get_mem_usage();
    std::cout << "[LayeredSlotMap]      Total (Actual):    " << mem_used / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[LayeredSlotMap]      Per Key (Actual):  " << (double)mem_used / num_keys << " bytes/key" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
}

void run_kd_benchmark(size_t num_points, uint32_t dimensions) {
    const size_t SHT_MEM_SIZE = num_points * (dimensions * 8 + 200); // Estimate
    MemoryManager mem(SHT_MEM_SIZE);
    LayeredSlotMap lsm(mem, dimensions, SplittingHeuristic::K_DIMENSIONAL_CYCLIC);

    std::cout << "\n\n--- Benchmark: " << dimensions << "D K-Dimensional Cyclic ---" << std::endl;
    std::cout << "--- Configuration: " << num_points << " points, Random distribution ---" << std::endl;

    std::vector<std::vector<uint64_t>> points(num_points, std::vector<uint64_t>(dimensions));
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < num_points; ++i) {
        for (uint32_t d = 0; d < dimensions; ++d) {
            points[i][d] = dist(rng);
        }
    }

    // --- Insert Benchmark ---
    std::cout << "\n--- " << dimensions << "D INSERTION ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const auto& p : points) {
        lsm.insert(p.data(), 0, nullptr);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_insert = (double)duration.count() / num_points;
    std::cout << "[LayeredSlotMap]      Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_insert << " ns/insert" << std::endl;

    // --- Get Benchmark ---
    std::cout << "\n--- " << dimensions << "D LOOKUP ---" << std::endl;
    size_t found_count = 0;
    std::string dummy_val;
    start_time = std::chrono::high_resolution_clock::now();
    for (const auto& p : points) {
        if (lsm.get(p.data(), dummy_val)) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_get = (double)duration.count() / num_points;
    std::cout << "[LayeredSlotMap]      Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_get << " ns/get" << std::endl;
    if (found_count != num_points) {
        std::cerr << "KD GET VERIFICATION FAILED! Found " << found_count << "/" << num_points << std::endl;
    } else {
        std::cout << "KD GET VERIFICATION PASSED!" << std::endl;
    }
    // --- MEMORY USAGE ---
    std::cout << "\n--- MEMORY USAGE ---" << std::endl;
    size_t mem_used = lsm.get_mem_usage();
    std::cout << "[LayeredSlotMap]      Total (Actual):    " << mem_used / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[LayeredSlotMap]      Per Key (Actual):  " << (double)mem_used / num_points << " bytes/key" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
}

void run_multithread_benchmark(size_t num_keys, int num_threads) {
    const size_t SHT_MEM_SIZE = num_keys * 200;
    MemoryManager mem(SHT_MEM_SIZE);
    LayeredSlotMap lsm(mem, 1, SplittingHeuristic::LEXICOGRAPHICAL);

    std::cout << "\n\n--- Benchmark: Multithreaded Lexicographical ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, Random distribution, " << num_threads << " threads ---" << std::endl;

    std::vector<uint64_t> all_keys(num_keys);
    std::iota(all_keys.begin(), all_keys.end(), 1);
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::shuffle(all_keys.begin(), all_keys.end(), rng);

    std::vector<std::vector<uint64_t>> thread_keys(num_threads);
    for (size_t i = 0; i < num_keys; ++i) {
        thread_keys[i % num_threads].push_back(all_keys[i]);
    }

    // --- Concurrent Insert Benchmark ---
    std::cout << "\n--- CONCURRENT INSERTION ---" << std::endl;
    std::vector<std::thread> threads;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (uint64_t key : thread_keys[i]) {
                uint64_t be_key = portable_htobe64(key);
                lsm.insert(&be_key, 0, nullptr);
            }
        });
    }
    for (auto& t : threads) { t.join(); }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double inserts_per_sec = (double)num_keys / (duration.count() / 1000.0);
    std::cout << "[LayeredSlotMap]    Total Time: " << duration.count() << " ms" << std::endl;
    std::cout << "[LayeredSlotMap]    Throughput: " << std::fixed << std::setprecision(0) << inserts_per_sec << " inserts/sec" << std::endl;

    // --- Concurrent Get Benchmark ---
    std::cout << "\n--- CONCURRENT LOOKUP ---" << std::endl;
    threads.clear();
    std::atomic<size_t> total_found = 0;
    start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t found_count = 0;
            std::string dummy_val;
            for (uint64_t key : thread_keys[i]) {
                uint64_t be_key = portable_htobe64(key);
                if (lsm.get(&be_key, dummy_val)) {
                    found_count++;
                }
            }
            total_found += found_count;
        });
    }
    for (auto& t : threads) { t.join(); }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double lookups_per_sec = (double)num_keys / (duration.count() / 1000.0);
    std::cout << "[LayeredSlotMap]    Total Time: " << duration.count() << " ms" << std::endl;
    std::cout << "[LayeredSlotMap]    Throughput: " << std::fixed << std::setprecision(0) << lookups_per_sec << " lookups/sec" << std::endl;
    if (total_found != num_keys) {
        std::cerr << "MT GET VERIFICATION FAILED! Found " << total_found << "/" << num_keys << std::endl;
    } else {
        std::cout << "MT GET VERIFICATION PASSED!" << std::endl;
    }
    // --- MEMORY USAGE ---
    std::cout << "\n--- MEMORY USAGE ---" << std::endl;
    size_t mem_used = lsm.get_mem_usage();
    std::cout << "[LayeredSlotMap]      Total (Actual):    " << mem_used / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[LayeredSlotMap]      Per Key (Actual):  " << (double)mem_used / num_keys << " bytes/key" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
}


int main() {
    try {
        const size_t ONE_MILLION = 1000000;
        const size_t HALF_MILLION = 500000;
        const int num_threads = std::thread::hardware_concurrency();

        run_lex_benchmark_1d(ONE_MILLION, "Random");
        run_lex_benchmark_1d(ONE_MILLION, "Sequential");
        run_kd_benchmark(HALF_MILLION, 4);
        run_kd_benchmark(HALF_MILLION / 2, 8);
        run_multithread_benchmark(ONE_MILLION, num_threads);

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
