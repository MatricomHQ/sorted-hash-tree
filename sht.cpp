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

// --- Constants for the data structure ---
const uint64_t POINTER_TAG = 1ULL << 63;
const uint64_t OFFSET_MASK = ~(POINTER_TAG);
const unsigned int BITS_PER_LEVEL = 3;
const unsigned int NODE_SLOTS = 1 << BITS_PER_LEVEL;
const uint64_t LEVEL_INDEX_MASK = NODE_SLOTS - 1;
const unsigned int MAX_DEPTH = (63 + BITS_PER_LEVEL - 1) / BITS_PER_LEVEL;


// A simple memory manager using a single mmap-ed region.
class MemoryManager {
public:
    explicit MemoryManager(size_t size) : allocation_size(size) {
        base_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base_ptr == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap memory");
        }
        current_ptr = static_cast<uint8_t*>(base_ptr);
        end_ptr = current_ptr + size;
        allocated_bytes = 0;
    }

    ~MemoryManager() {
        if (base_ptr != MAP_FAILED) {
            munmap(base_ptr, allocation_size);
        }
    }

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    void* alloc(size_t size) {
        size_t aligned_size = (size + 7) & ~7;
        if (current_ptr + aligned_size > end_ptr) {
            throw std::bad_alloc();
        }
        void* mem = current_ptr;
        current_ptr += aligned_size;
        allocated_bytes += aligned_size;
        std::fill(static_cast<uint64_t*>(mem), static_cast<uint64_t*>(mem) + aligned_size / sizeof(uint64_t), 0);
        return mem;
    }

    uint64_t get_offset(const void* ptr) const {
        return static_cast<const uint8_t*>(ptr) - static_cast<uint8_t*>(base_ptr);
    }

    void* get_ptr(uint64_t offset) const {
        return static_cast<uint8_t*>(base_ptr) + offset;
    }

    size_t get_allocated_size() const {
        return allocated_bytes;
    }

private:
    void* base_ptr = nullptr;
    uint8_t* current_ptr = nullptr;
    uint8_t* end_ptr = nullptr;
    size_t allocation_size = 0;
    size_t allocated_bytes = 0;
};


// The core node structure. It's exactly 64 bytes (a cache line).
struct Node {
    uint64_t slots[NODE_SLOTS];
};


// The main data structure class.
class LayeredSlotMap {
public:
    explicit LayeredSlotMap(size_t size) : mem(size) {
        void* root_node_ptr = mem.alloc(sizeof(Node));
        root_node_offset = mem.get_offset(root_node_ptr);
    }

    void insert(uint64_t key) {
        if (key & POINTER_TAG) return;

        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            Node* current_node = static_cast<Node*>(mem.get_ptr(current_node_offset));
            int shift = 63 - (depth + 1) * BITS_PER_LEVEL;
            uint64_t index = (key >> shift) & LEVEL_INDEX_MASK;

            uint64_t& slot = current_node->slots[index];

            if (slot == 0) {
                slot = key;
                return;
            }
            if (slot & POINTER_TAG) {
                current_node_offset = slot & OFFSET_MASK;
                continue;
            }
            uint64_t existing_key = slot;
            if (existing_key == key) return;

            void* new_node_ptr = mem.alloc(sizeof(Node));
            uint64_t new_node_offset = mem.get_offset(new_node_ptr);
            slot = new_node_offset | POINTER_TAG;
            current_node = static_cast<Node*>(new_node_ptr);
            unsigned int next_depth = depth + 1;

            while (next_depth < MAX_DEPTH) {
                int next_shift = 63 - (next_depth + 1) * BITS_PER_LEVEL;
                uint64_t index_existing = (existing_key >> next_shift) & LEVEL_INDEX_MASK;
                uint64_t index_new = (key >> next_shift) & LEVEL_INDEX_MASK;

                if (index_existing != index_new) {
                    current_node->slots[index_existing] = existing_key;
                    current_node->slots[index_new] = key;
                    return;
                }
                void* intermediate_node_ptr = mem.alloc(sizeof(Node));
                uint64_t intermediate_node_offset = mem.get_offset(intermediate_node_ptr);
                current_node->slots[index_existing] = intermediate_node_offset | POINTER_TAG;
                current_node = static_cast<Node*>(intermediate_node_ptr);
                next_depth++;
            }
        }
    }

    bool get(uint64_t key) {
        if (key & POINTER_TAG) return false;
        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            Node* current_node = static_cast<Node*>(mem.get_ptr(current_node_offset));
            int shift = 63 - (depth + 1) * BITS_PER_LEVEL;
            uint64_t index = (key >> shift) & LEVEL_INDEX_MASK;
            uint64_t slot = current_node->slots[index];

            if (slot == 0) return false;
            if (slot & POINTER_TAG) {
                current_node_offset = slot & OFFSET_MASK;
                continue;
            }
            return slot == key;
        }
        return false;
    }

    std::vector<uint64_t> scan(uint64_t start, uint64_t end) {
        std::vector<uint64_t> results;
        if (start > end) return results;
        results.reserve(1024); // Avoid reallocations for typical scan sizes
        scan_recursive(root_node_offset, 0, 0, start, end, results);
        return results;
    }

    size_t get_mem_usage() const {
        return mem.get_allocated_size();
    }

private:
    MemoryManager mem;
    uint64_t root_node_offset;

    void scan_recursive(uint64_t node_offset, int depth, uint64_t prefix, uint64_t start, uint64_t end, std::vector<uint64_t>& results) {
        if (depth >= MAX_DEPTH) return;
        Node* node = static_cast<Node*>(mem.get_ptr(node_offset));
        int shift = 63 - (depth + 1) * BITS_PER_LEVEL;

        for (int i = 0; i < NODE_SLOTS; ++i) {
            uint64_t slot = node->slots[i];
            if (slot == 0) continue;

            uint64_t child_prefix = prefix | ((uint64_t)i << shift);
            uint64_t lower_bound = child_prefix;
            uint64_t upper_bound_mask = (shift < 63) ? (1ULL << shift) - 1 : UINT64_MAX;
            uint64_t upper_bound = child_prefix | upper_bound_mask;

            if (lower_bound > end || upper_bound < start) {
                continue; // Prune this entire branch
            }

            if (slot & POINTER_TAG) {
                scan_recursive(slot & OFFSET_MASK, depth + 1, child_prefix, start, end, results);
            } else {
                uint64_t key = slot;
                if (key >= start && key <= end) {
                    results.push_back(key);
                }
            }
        }
    }
};

void run_benchmark(size_t num_keys, const std::string& key_type) {
    // Adjust memory based on key count. Allocate 40 bytes per key as a safe upper bound for this structure.
    const size_t MEM_SIZE = num_keys * 40;
    LayeredSlotMap sht(MEM_SIZE);

    std::cout << "\n\n--- Layered Slot Map (SHT) Benchmark ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    std::cout << "Preparing keys..." << std::endl;
    std::vector<uint64_t> keys(num_keys);
    std::iota(keys.begin(), keys.end(), 1);

    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    if (key_type == "Random") {
        std::shuffle(keys.begin(), keys.end(), rng);
    }

    // --- Insert Benchmark ---
    std::cout << "\n--- Inserting " << num_keys << " keys ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) {
        sht.insert(key);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_insert = (double)duration.count() / num_keys;
    std::cout << "Total time: " << duration.count() / 1e6 << " ms" << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(2) << ns_per_insert << " ns/insert" << std::endl;

    // --- Get Benchmark ---
    std::cout << "\n--- Looking up " << num_keys << " existing keys ---" << std::endl;
    if (key_type == "Random") {
        std::shuffle(keys.begin(), keys.end(), rng);
    }
    size_t found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) {
        if (sht.get(key)) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_get = (double)duration.count() / num_keys;
    std::cout << "Total time: " << duration.count() / 1e6 << " ms" << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(2) << ns_per_get << " ns/get" << std::endl;
    std::cout << "Verification: " << found_count << " of " << num_keys << " keys found." << std::endl;
    if (found_count != num_keys) {
        std::cerr << "ERROR: Not all inserted keys were found!" << std::endl;
    }

    // --- Scan Benchmark ---
    const size_t SCAN_SIZE = 1000;
    if (num_keys > SCAN_SIZE) {
        std::uniform_int_distribution<uint64_t> dist(1, num_keys - SCAN_SIZE);
        uint64_t scan_start = dist(rng);
        uint64_t scan_end = scan_start + SCAN_SIZE - 1;

        std::cout << "\n--- Benchmarking Range Scan (1000 keys) ---" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        std::vector<uint64_t> scan_results = sht.scan(scan_start, scan_end);
        end_time = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double ns_per_scan_key = scan_results.empty() ? 0 : (double)duration.count() / scan_results.size();

        std::cout << "Total time: " << duration.count() / 1e6 << " ms" << std::endl;
        std::cout << "Verification: Found " << scan_results.size() << " keys." << std::endl;
        std::cout << "Average latency: " << std::fixed << std::setprecision(2) << ns_per_scan_key << " ns/key" << std::endl;

        if (scan_results.size() != SCAN_SIZE) {
            std::cerr << "ERROR: Scan did not return the expected number of keys (" << SCAN_SIZE << ")" << std::endl;
        }
    }


    // --- Memory Usage ---
    std::cout << "\n--- Memory Usage ---" << std::endl;
    size_t mem_used = sht.get_mem_usage();
    std::cout << "Total memory allocated for nodes: " << mem_used / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Memory per key: " << (double)mem_used / num_keys << " bytes/key" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
}

int main() {
    try {
        const size_t ONE_MILLION = 1000000;
        const size_t TEN_MILLION = 10000000;
        const size_t TWENTY_MILLION = 20000000;

        run_benchmark(ONE_MILLION, "Random");
        run_benchmark(ONE_MILLION, "Sequential");

        run_benchmark(TEN_MILLION, "Random");
        run_benchmark(TEN_MILLION, "Sequential");

        run_benchmark(TWENTY_MILLION, "Random");
        run_benchmark(TWENTY_MILLION, "Sequential");

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
