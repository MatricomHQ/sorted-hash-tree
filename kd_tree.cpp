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
#include <optional>
#include <cmath>
#include <array>

// --- Constants ---
const uint64_t POINTER_TAG = 1ULL << 63;
const uint64_t OFFSET_MASK = ~(POINTER_TAG);

// --- Fixed Optimal Geometry ---
const unsigned int BITS_PER_LEVEL = 16;
const uint64_t VIRTUAL_NODE_SLOTS = 1ULL << BITS_PER_LEVEL;
const uint64_t LEVEL_INDEX_MASK = VIRTUAL_NODE_SLOTS - 1;
// const unsigned int MAX_DEPTH = (63 + BITS_PER_LEVEL - 1) / BITS_PER_LEVEL; // Will be redefined in the class
const unsigned int SLOTS_PER_PAGE = 256;
const uint64_t PAGES_PER_NODE = VIRTUAL_NODE_SLOTS / SLOTS_PER_PAGE;

// --- NEW PAYLOAD STRUCT ---
template<unsigned int K>
struct SlotPayload {
    std::array<uint64_t, K> key;
    uint64_t value; // We will apply the POINTER_TAG to this field
};

// Memory Manager
class MemoryManager {
public:
    explicit MemoryManager(size_t size) : allocation_size(size) {
        base_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base_ptr == MAP_FAILED) { throw std::runtime_error("Failed to mmap memory"); }
        current_ptr = static_cast<uint8_t*>(base_ptr);
        end_ptr = current_ptr + size;
    }
    ~MemoryManager() { if (base_ptr != MAP_FAILED) { munmap(base_ptr, allocation_size); } }
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    void* alloc(size_t size) {
        size_t aligned_size = (size + 15) & ~15; // Align to 16 bytes
        if (current_ptr + aligned_size > end_ptr) { throw std::bad_alloc(); }
        void* mem = current_ptr;
        current_ptr += aligned_size;
        std::fill(static_cast<uint8_t*>(mem), static_cast<uint8_t*>(mem) + aligned_size, 0);
        return mem;
    }
    uint64_t get_offset(const void* ptr) const { return static_cast<const uint8_t*>(ptr) - static_cast<uint8_t*>(base_ptr); }
    void* get_ptr(uint64_t offset) const { return static_cast<uint8_t*>(base_ptr) + offset; }
    size_t get_allocated_size() const {
        return static_cast<uint8_t*>(current_ptr) - static_cast<uint8_t*>(base_ptr);
    }
private:
    void* base_ptr = nullptr;
    uint8_t* current_ptr = nullptr;
    uint8_t* end_ptr = nullptr;
    size_t allocation_size = 0;
};

// --- Node Structures using the new Payload ---
template<unsigned int K>
struct Page { SlotPayload<K> slots[SLOTS_PER_PAGE]; };

template<unsigned int K>
struct PagedNode { Page<K>** pages; };

template<unsigned int K>
class KeyValueRadixTree {
private:
    static constexpr unsigned int BITS_PER_DIM_PER_LEVEL = BITS_PER_LEVEL;
    static constexpr unsigned int LEVELS_PER_DIM = (63 + BITS_PER_DIM_PER_LEVEL - 1) / BITS_PER_DIM_PER_LEVEL;
    static constexpr unsigned int MAX_DEPTH = K * LEVELS_PER_DIM;

    MemoryManager mem;
    uint64_t root_node_offset;

    void* new_node() {
        PagedNode<K>* node = static_cast<PagedNode<K>*>(mem.alloc(sizeof(PagedNode<K>)));
        node->pages = static_cast<Page<K>**>(mem.alloc(sizeof(Page<K>*) * PAGES_PER_NODE));
        return node;
    }

    inline SlotPayload<K>* get_slot_ptr(PagedNode<K>* node, uint64_t index_in_node, bool allocate_on_demand) {
        uint64_t page_idx = index_in_node >> 8;
        uint64_t slot_idx = index_in_node & 255;
        if (node->pages[page_idx] == nullptr) {
            if (allocate_on_demand) {
                node->pages[page_idx] = static_cast<Page<K>*>(mem.alloc(sizeof(Page<K>)));
            } else { return nullptr; }
        }
        return &node->pages[page_idx]->slots[slot_idx];
    }

    bool is_key_zero(const std::array<uint64_t, K>& key) const {
        for (unsigned int i = 0; i < K; ++i) {
            if (key[i] != 0) return false;
        }
        return true;
    }

public:
    explicit KeyValueRadixTree(size_t size) : mem(size) {
        root_node_offset = mem.get_offset(new_node());
    }

    void insert(const std::array<uint64_t, K>& key, uint64_t value) {
        if (value & POINTER_TAG) return; // Values cannot have the tag bit

        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            PagedNode<K>* current_node = static_cast<PagedNode<K>*>(mem.get_ptr(current_node_offset));

            const unsigned int dim_idx = depth % K;
            const unsigned int level_in_dim = depth / K;
            const int shift = 64 - (level_in_dim + 1) * BITS_PER_DIM_PER_LEVEL;

            uint64_t index_in_node = (key[dim_idx] >> shift) & LEVEL_INDEX_MASK;
            SlotPayload<K>* slot_ptr = get_slot_ptr(current_node, index_in_node, true);
            SlotPayload<K>& slot = *slot_ptr;

            if (slot.value == 0 && is_key_zero(slot.key)) { // Empty slot
                slot.key = key;
                slot.value = value;
                return;
            }
            if (slot.value & POINTER_TAG) {
                current_node_offset = slot.value & OFFSET_MASK;
                __builtin_prefetch(mem.get_ptr(current_node_offset));
                continue;
            }

            if (slot.key == key) { // Update existing key
                slot.value = value;
                return;
            }

            // Collision with an existing key
            SlotPayload<K> existing_payload = slot;

            void* new_node_ptr = new_node();
            slot.value = mem.get_offset(new_node_ptr) | POINTER_TAG;
            slot.key = {}; // Clear key field for pointers

            PagedNode<K>* child_node = static_cast<PagedNode<K>*>(new_node_ptr);
            unsigned int next_depth = depth + 1;

            while (next_depth < MAX_DEPTH) {
                const unsigned int next_dim_idx = next_depth % K;
                const unsigned int next_level_in_dim = next_depth / K;
                const int next_shift = 64 - (next_level_in_dim + 1) * BITS_PER_DIM_PER_LEVEL;

                uint64_t index_existing = (existing_payload.key[next_dim_idx] >> next_shift) & LEVEL_INDEX_MASK;
                uint64_t index_new = (key[next_dim_idx] >> next_shift) & LEVEL_INDEX_MASK;

                if (index_existing != index_new) {
                    *get_slot_ptr(child_node, index_existing, true) = existing_payload;

                    SlotPayload<K> new_payload;
                    new_payload.key = key;
                    new_payload.value = value;
                    *get_slot_ptr(child_node, index_new, true) = new_payload;
                    return;
                }

                void* intermediate_node_ptr = new_node();
                SlotPayload<K>* intermediate_slot = get_slot_ptr(child_node, index_existing, true);
                intermediate_slot->value = mem.get_offset(intermediate_node_ptr) | POINTER_TAG;
                intermediate_slot->key = {};
                child_node = static_cast<PagedNode<K>*>(intermediate_node_ptr);
                next_depth++;
            }
        }
    }

    std::optional<uint64_t> get(const std::array<uint64_t, K>& key) {
        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            PagedNode<K>* current_node = static_cast<PagedNode<K>*>(mem.get_ptr(current_node_offset));

            const unsigned int dim_idx = depth % K;
            const unsigned int level_in_dim = depth / K;
            const int shift = 64 - (level_in_dim + 1) * BITS_PER_DIM_PER_LEVEL;

            uint64_t index_in_node = (key[dim_idx] >> shift) & LEVEL_INDEX_MASK;
            SlotPayload<K>* slot_ptr = get_slot_ptr(current_node, index_in_node, false);
            if (slot_ptr == nullptr) return std::nullopt;

            SlotPayload<K>& slot = *slot_ptr;

            if (slot.value & POINTER_TAG) {
                current_node_offset = slot.value & OFFSET_MASK;
                continue;
            }

            if (slot.key == key) {
                 if (slot.value == 0 && is_key_zero(slot.key)) return std::nullopt; // Empty slot
                 return slot.value;
            }

            return std::nullopt; // Found a non-matching key or an empty slot
        }
        return std::nullopt;
    }

    size_t get_mem_usage() const { return mem.get_allocated_size(); }
};

template<unsigned int K>
void run_benchmark(size_t num_keys, const std::string& key_type) {
    const size_t SHT_MEM_SIZE = 3000000000;

    std::cout << "\n\n--- Benchmark: " << K << "D Key-Value Radix Tree ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    std::cout << "Preparing keys..." << std::endl;
    std::vector<std::array<uint64_t, K>> keys(num_keys);
    if (num_keys > 0) {
        for (size_t i = 0; i < num_keys; ++i) {
            for (unsigned int d = 0; d < K; ++d) {
                keys[i][d] = i + 1 + d; // Simple way to generate unique k-d keys
            }
        }
    }

    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    if (key_type == "Random") std::shuffle(keys.begin(), keys.end(), rng);

    std::cout << "\n--- INSERTION ---" << std::endl;
    KeyValueRadixTree<K> sht(SHT_MEM_SIZE);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) { sht.insert(key, key[0] + 1); } // value is key[0] + 1
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_insert_sht = num_keys > 0 ? (double)duration.count() / num_keys : 0;
    std::cout << "[KeyValueRadixTree]   Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_insert_sht << " ns/insert" << std::endl;

    std::cout << "\n--- HIT LATENCY (LOOKUP) ---" << std::endl;
    if (key_type == "Random") std::shuffle(keys.begin(), keys.end(), rng);

    size_t found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) { if (sht.get(key)) { found_count++; } }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_get_sht = num_keys > 0 ? (double)duration.count() / num_keys : 0;
    std::cout << "[KeyValueRadixTree]   Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_get_sht << " ns/get" << std::endl;
    if (found_count != num_keys) std::cerr << "RADIX GET VERIFICATION FAILED! Found " << found_count << "/" << num_keys << std::endl;

    if (num_keys > 0) {
        std::cout << "\n--- MISS LATENCY (LOOKUP) ---" << std::endl;
        std::vector<std::array<uint64_t, K>> missing_keys = keys;
        uint64_t offset = num_keys * 2;
        for(size_t i = 0; i < num_keys; ++i) {
            for (unsigned int d = 0; d < K; ++d) {
                missing_keys[i][d] += offset;
            }
        }
        std::shuffle(missing_keys.begin(), missing_keys.end(), rng);

        size_t miss_found_count = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (const auto& key : missing_keys) { if (sht.get(key)) { miss_found_count++; } }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double ns_per_miss_sht = (double)duration.count() / num_keys;
        std::cout << "[KeyValueRadixTree]   Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_miss_sht << " ns/get" << std::endl;
        if (miss_found_count != 0) std::cerr << "RADIX MISS VERIFICATION FAILED! Found " << miss_found_count << " non-existent keys" << std::endl;
    }

    std::cout << "\n--- MEMORY USAGE ---" << std::endl;
    size_t mem_used_sht = sht.get_mem_usage();
    std::cout << "[KeyValueRadixTree]   Total (Actual):    " << mem_used_sht / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[KeyValueRadixTree]   Per Key (Actual):  " << (num_keys > 0 ? (double)mem_used_sht / num_keys : 0) << " bytes/key" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
}

int main() {
    try {
        const unsigned int K = 3; // Number of dimensions
        const size_t K10 = 10000;
        const size_t M1 = 1000000;
        const size_t M20 = 20000000;

        run_benchmark<K>(100, "Random");
        run_benchmark<K>(K10, "Random");
        run_benchmark<K>(M1, "Random");
        // Reduce count for quicker test, original M20 might be slow with larger payloads
        run_benchmark<K>(5000000, "Random");

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
