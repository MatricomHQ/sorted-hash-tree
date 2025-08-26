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

// --- Constants ---
const uint64_t POINTER_TAG = 1ULL << 63;
const uint64_t OFFSET_MASK = ~(POINTER_TAG);

// --- Fixed Optimal Geometry ---
const unsigned int BITS_PER_LEVEL = 16;
const uint64_t VIRTUAL_NODE_SLOTS = 1ULL << BITS_PER_LEVEL;
const uint64_t LEVEL_INDEX_MASK = VIRTUAL_NODE_SLOTS - 1;
const unsigned int MAX_DEPTH = (63 + BITS_PER_LEVEL - 1) / BITS_PER_LEVEL;
const unsigned int SLOTS_PER_PAGE = 256;
const uint64_t PAGES_PER_NODE = VIRTUAL_NODE_SLOTS / SLOTS_PER_PAGE;

// --- NEW PAYLOAD STRUCT ---
struct SlotPayload {
    uint64_t key;
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
        size_t aligned_size = (size + 15) & ~15; // Align to 16 bytes for SlotPayload
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
struct Page { SlotPayload slots[SLOTS_PER_PAGE]; };
struct PagedNode { Page** pages; };

class KeyValueRadixTree {
private:
    MemoryManager mem;
    uint64_t root_node_offset;

    void* new_node() {
        PagedNode* node = static_cast<PagedNode*>(mem.alloc(sizeof(PagedNode)));
        node->pages = static_cast<Page**>(mem.alloc(sizeof(Page*) * PAGES_PER_NODE));
        return node;
    }

    SlotPayload* get_slot_ptr(PagedNode* node, uint64_t index_in_node, bool allocate_on_demand) {
        uint64_t page_idx = index_in_node / SLOTS_PER_PAGE;
        uint64_t slot_idx = index_in_node % SLOTS_PER_PAGE;
        if (node->pages[page_idx] == nullptr) {
            if (allocate_on_demand) {
                node->pages[page_idx] = static_cast<Page*>(mem.alloc(sizeof(Page)));
            } else { return nullptr; }
        }
        return &node->pages[page_idx]->slots[slot_idx];
    }

public:
    explicit KeyValueRadixTree(size_t size) : mem(size) {
        root_node_offset = mem.get_offset(new_node());
    }

    void insert(uint64_t key, uint64_t value) {
        if (value & POINTER_TAG) return; // Values cannot have the tag bit

        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            PagedNode* current_node = static_cast<PagedNode*>(mem.get_ptr(current_node_offset));
            int shift = 64 - (depth + 1) * BITS_PER_LEVEL;
            if (shift < 0) shift = 0;
            uint64_t index_in_node = (key >> shift) & LEVEL_INDEX_MASK;
            SlotPayload* slot_ptr = get_slot_ptr(current_node, index_in_node, true);
            SlotPayload& slot = *slot_ptr;

            if (slot.value == 0 && slot.key == 0) { // Empty slot
                slot.key = key;
                slot.value = value;
                return;
            }
            if (slot.value & POINTER_TAG) {
                current_node_offset = slot.value & OFFSET_MASK;
                continue;
            }

            if (slot.key == key) { // Update existing key
                slot.value = value;
                return;
            }

            // Collision with an existing key
            SlotPayload existing_payload = slot;

            void* new_node_ptr = new_node();
            slot.value = mem.get_offset(new_node_ptr) | POINTER_TAG;
            slot.key = 0; // Clear key field for pointers

            PagedNode* child_node = static_cast<PagedNode*>(new_node_ptr);
            unsigned int next_depth = depth + 1;

            while (next_depth < MAX_DEPTH) {
                int next_shift = 64 - (next_depth + 1) * BITS_PER_LEVEL;
                if (next_shift < 0) next_shift = 0;
                uint64_t index_existing = (existing_payload.key >> next_shift) & LEVEL_INDEX_MASK;
                uint64_t index_new = (key >> next_shift) & LEVEL_INDEX_MASK;

                if (index_existing != index_new) {
                    *get_slot_ptr(child_node, index_existing, true) = existing_payload;
                    *get_slot_ptr(child_node, index_new, true) = {key, value};
                    return;
                }

                void* intermediate_node_ptr = new_node();
                SlotPayload* intermediate_slot = get_slot_ptr(child_node, index_existing, true);
                intermediate_slot->value = mem.get_offset(intermediate_node_ptr) | POINTER_TAG;
                intermediate_slot->key = 0;
                child_node = static_cast<PagedNode*>(intermediate_node_ptr);
                next_depth++;
            }
        }
    }

    std::optional<uint64_t> get(uint64_t key) {
        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            PagedNode* current_node = static_cast<PagedNode*>(mem.get_ptr(current_node_offset));
            int shift = 64 - (depth + 1) * BITS_PER_LEVEL;
            if (shift < 0) shift = 0;
            uint64_t index_in_node = (key >> shift) & LEVEL_INDEX_MASK;
            SlotPayload* slot_ptr = get_slot_ptr(current_node, index_in_node, false);
            if (slot_ptr == nullptr) return std::nullopt;

            SlotPayload& slot = *slot_ptr;

            if (slot.value & POINTER_TAG) {
                current_node_offset = slot.value & OFFSET_MASK;
                continue;
            }

            if (slot.key == key) {
                 if (slot.value == 0 && slot.key == 0) return std::nullopt; // Empty slot
                 return slot.value;
            }

            return std::nullopt; // Found a non-matching key or an empty slot
        }
        return std::nullopt;
    }

    size_t get_mem_usage() const { return mem.get_allocated_size(); }
};

void run_benchmark(size_t num_keys, const std::string& key_type) {
    const size_t SHT_MEM_SIZE = 3000000000;

    std::cout << "\n\n--- Benchmark: Key-Value Radix Tree vs std::unordered_map ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    std::cout << "Preparing keys..." << std::endl;
    std::vector<uint64_t> keys(num_keys);
    if (num_keys > 0) std::iota(keys.begin(), keys.end(), 1);

    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    if (key_type == "Random") std::shuffle(keys.begin(), keys.end(), rng);

    std::cout << "\n--- INSERTION ---" << std::endl;
    KeyValueRadixTree sht(SHT_MEM_SIZE);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) { sht.insert(key, key + 1); } // value is key + 1
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_insert_sht = num_keys > 0 ? (double)duration.count() / num_keys : 0;
    std::cout << "[KeyValueRadixTree]   Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_insert_sht << " ns/insert" << std::endl;

    std::unordered_map<uint64_t, uint64_t> umap;
    if (num_keys > 0) umap.reserve(num_keys);
    start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) { umap.insert({key, key + 1}); }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_insert_umap = num_keys > 0 ? (double)duration.count() / num_keys : 0;
    std::cout << "[std::unordered_map]    Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_insert_umap << " ns/insert" << std::endl;

    std::cout << "\n--- HIT LATENCY (LOOKUP) ---" << std::endl;
    if (key_type == "Random") std::shuffle(keys.begin(), keys.end(), rng);

    size_t found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) { if (sht.get(key)) { found_count++; } }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_get_sht = num_keys > 0 ? (double)duration.count() / num_keys : 0;
    std::cout << "[KeyValueRadixTree]   Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_get_sht << " ns/get" << std::endl;
    if (found_count != num_keys) std::cerr << "RADIX GET VERIFICATION FAILED! Found " << found_count << "/" << num_keys << std::endl;

    found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (uint64_t key : keys) { if (umap.find(key) != umap.end()) { found_count++; } }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_get_umap = num_keys > 0 ? (double)duration.count() / num_keys : 0;
    std::cout << "[std::unordered_map]    Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_get_umap << " ns/get" << std::endl;
    if (found_count != num_keys) std::cerr << "UMAP GET VERIFICATION FAILED! Found " << found_count << "/" << num_keys << std::endl;

    if (num_keys > 0) {
        std::cout << "\n--- MISS LATENCY (LOOKUP) ---" << std::endl;
        std::vector<uint64_t> missing_keys(num_keys);
        uint64_t offset = num_keys * 2;
        for(size_t i = 0; i < num_keys; ++i) { missing_keys[i] = keys[i] + offset; }
        std::shuffle(missing_keys.begin(), missing_keys.end(), rng);

        size_t miss_found_count = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (uint64_t key : missing_keys) { if (sht.get(key)) { miss_found_count++; } }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double ns_per_miss_sht = (double)duration.count() / num_keys;
        std::cout << "[KeyValueRadixTree]   Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_miss_sht << " ns/get" << std::endl;
        if (miss_found_count != 0) std::cerr << "RADIX MISS VERIFICATION FAILED! Found " << miss_found_count << " non-existent keys" << std::endl;

        miss_found_count = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (uint64_t key : missing_keys) { if (umap.find(key) != umap.end()) { miss_found_count++; } }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        double ns_per_miss_umap = (double)duration.count() / num_keys;
        std::cout << "[std::unordered_map]    Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_miss_umap << " ns/get" << std::endl;
        if (miss_found_count != 0) std::cerr << "UMAP MISS VERIFICATION FAILED! Found " << miss_found_count << " non-existent keys" << std::endl;
    }

    std::cout << "\n--- MEMORY USAGE ---" << std::endl;
    size_t mem_used_sht = sht.get_mem_usage();
    std::cout << "[KeyValueRadixTree]   Total (Actual):    " << mem_used_sht / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[KeyValueRadixTree]   Per Key (Actual):  " << (num_keys > 0 ? (double)mem_used_sht / num_keys : 0) << " bytes/key" << std::endl;

    size_t node_size = sizeof(uint64_t) * 2 + sizeof(void*);
    size_t mem_used_umap = (umap.size() * node_size) + (umap.bucket_count() * sizeof(void*));
    std::cout << "[std::unordered_map]    Total (Estimated): " << mem_used_umap / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[std::unordered_map]    Per Key (Estimated): " << (num_keys > 0 ? (double)mem_used_umap / num_keys : 0) << " bytes/key" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
}

int main() {
    try {
        const size_t K10 = 10000;
        const size_t M1 = 1000000;
        const size_t M20 = 20000000;

        run_benchmark(100, "Random");
        run_benchmark(K10, "Random");
        run_benchmark(M1, "Random");
        run_benchmark(M20, "Random");

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
