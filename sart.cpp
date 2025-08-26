#include <iostream>
#include <vector>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <random>
#include <unordered_map>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h> // AVX2 for x86
#elif defined(__aarch64__)
    #include <arm_neon.h> // NEON for ARM64
#endif

// Type definitions for clarity and portability
using u64 = uint64_t;
// An offset from the start of the mmap'd region. Using 32-bit offsets halves pointer
// sizes and is sufficient for up to 4GB of memory.
using NodeOffset = uint32_t;

// --- SART Data Structures and Constants ---

// We use the lowest 2 bits of the NodeOffset to store a tag.
const u64 TAG_MASK = 0x3;
const u64 OFFSET_MASK = ~TAG_MASK;
const u64 LEAF_NODE_TAG = 0x0;
const u64 INTERNAL_NODE_TAG = 0x1;

// Constants for the tree structure.
const size_t ROOT_BITS = 16;
const size_t ROOT_TABLE_SIZE = 1 << ROOT_BITS; // 2^16 = 65536
const size_t INTERNAL_NODE_BITS = 5; // Wider nodes for a shallower tree
const size_t INTERNAL_NODE_TABLE_SIZE = 1 << INTERNAL_NODE_BITS; // 2^5 = 32
const size_t LEAF_NODE_CAPACITY = 15; // Larger leaves to improve data density

// A Leaf Node holds a small, sorted array of keys.
// Its size is 128 bytes: 1 (count) + 7 (padding) + 15*8 (keys) = 128 bytes.
struct LeafNode {
    uint8_t count;
    uint8_t padding[7]; // Ensure alignment and size
    u64 keys[LEAF_NODE_CAPACITY];
};

// An Internal Node acts as a dispatch table, pointing to other nodes.
// Its size is 128 bytes: 32 * 4 (offsets).
struct InternalNode {
    NodeOffset children[INTERNAL_NODE_TABLE_SIZE];
};


// The SART (Sorted Array Radix Trie) class
class SART {
public:
    // Constructor: Initializes the memory map and sets up the header and root table.
    SART(size_t size_in_gb) {
        if (size_in_gb == 0) {
            throw std::invalid_argument("SART size must be at least 1 GB.");
        }
        uint64_t capacity_64 = (uint64_t)size_in_gb * 1024 * 1024 * 1024;
        if (capacity_64 > UINT32_MAX) {
            throw std::invalid_argument("SART capacity cannot exceed 4GB with 32-bit offsets.");
        }
        capacity_ = capacity_64;

        mem_base_ = mmap(NULL, capacity_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem_base_ == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap memory for SART.");
        }

        header_ = static_cast<Header*>(mem_base_);
        header_->memory_used = sizeof(Header);
        header_->capacity = capacity_;

        header_->root_table_offset = header_->memory_used;
        root_table_ = static_cast<NodeOffset*>(allocate(sizeof(NodeOffset) * ROOT_TABLE_SIZE));
        std::fill(root_table_, root_table_ + ROOT_TABLE_SIZE, 0);

        std::cout << "SART successfully initialized with " << size_in_gb << " GB of memory." << std::endl;
    }

    // Destructor: Releases the memory map.
    ~SART() {
        if (mem_base_ != nullptr) {
            munmap(mem_base_, capacity_);
            std::cout << "SART memory released." << std::endl;
        }
    }

    size_t get_memory_usage() const {
        return header_->memory_used;
    }

    // Public get method
    bool get(u64 key) const {
        const size_t root_idx = key >> (64 - ROOT_BITS);
        NodeOffset node_offset = root_table_[root_idx];
        if (node_offset == 0) return false;

        int depth = 0;
        while (true) {
            u64 tag = node_offset & TAG_MASK;
            void* node_ptr = ptr_from_offset(node_offset);

            if (tag == LEAF_NODE_TAG) {
                const LeafNode* leaf = static_cast<const LeafNode*>(node_ptr);
                return find_in_leaf_simd(key, leaf);
            } else if (tag == INTERNAL_NODE_TAG) {
                const InternalNode* internal = static_cast<const InternalNode*>(node_ptr);
                size_t child_idx = (key >> (64 - ROOT_BITS - ((depth + 1) * INTERNAL_NODE_BITS))) & (INTERNAL_NODE_TABLE_SIZE - 1);
                node_offset = internal->children[child_idx];
                if (node_offset == 0) return false;
                depth++;
            } else {
                // Should not happen
                return false;
            }
        }
    }

    // Public insert method
    void insert(u64 key) {
        size_t root_idx = key >> (64 - ROOT_BITS);
        NodeOffset* parent_link = &root_table_[root_idx];
        if (*parent_link == 0) {
            // Case 1: Empty root slot, create a new leaf.
            LeafNode* leaf = static_cast<LeafNode*>(allocate(sizeof(LeafNode)));
            leaf->count = 1;
            leaf->keys[0] = key;
            *parent_link = offset_from_ptr(leaf) | LEAF_NODE_TAG;
            return;
        }

        int depth = 0;
        while (true) {
            u64 tag = *parent_link & TAG_MASK;
            void* node_ptr = ptr_from_offset(*parent_link);

            if (tag == LEAF_NODE_TAG) {
                LeafNode* leaf = static_cast<LeafNode*>(node_ptr);

                // Use non-const iterators as we intend to modify the leaf.
                u64* start = leaf->keys;
                u64* end = leaf->keys + leaf->count;
                u64* pos = std::lower_bound(start, end, key);

                if (pos != end && *pos == key) {
                    return; // Key already exists, do nothing.
                }

                if (leaf->count < LEAF_NODE_CAPACITY) {
                    // Case 2: Leaf has space.
                    // Shift elements to the right to make space for the new key.
                    std::move_backward(pos, end, end + 1);
                    *pos = key;
                    leaf->count++;
                } else {
                    // Case 3: Leaf is full, needs to split.
                    split_leaf_node(leaf, key, parent_link, depth);
                }
                return;
            } else if (tag == INTERNAL_NODE_TAG) {
                InternalNode* internal = static_cast<InternalNode*>(node_ptr);
                size_t child_idx = (key >> (64 - ROOT_BITS - ((depth + 1) * INTERNAL_NODE_BITS))) & (INTERNAL_NODE_TABLE_SIZE - 1);
                parent_link = &internal->children[child_idx];
                if (*parent_link == 0) {
                    // Create a new leaf under this internal node
                    LeafNode* leaf = static_cast<LeafNode*>(allocate(sizeof(LeafNode)));
                    leaf->count = 1;
                    leaf->keys[0] = key;
                    *parent_link = offset_from_ptr(leaf) | LEAF_NODE_TAG;
                    return;
                }
                depth++;
            } else {
                return; // Should not happen
            }
        }
    }


private:
    // The Header resides at the beginning of the mmap'd region.
    struct Header {
        u64 capacity;
        u64 memory_used;
        NodeOffset root_table_offset;
    };

    void* mem_base_ = nullptr;
    Header* header_ = nullptr;
    size_t capacity_ = 0;
    NodeOffset* root_table_ = nullptr; // Direct pointer to the root table for fast access

    // --- Memory Management ---

    void* allocate(size_t size) {
        size_t aligned_size = (size + 7) & ~7;
        if (header_->memory_used + aligned_size > header_->capacity) throw std::bad_alloc();
        void* ptr = static_cast<char*>(mem_base_) + header_->memory_used;
        header_->memory_used += aligned_size;
        return ptr;
    }

    void* ptr_from_offset(NodeOffset offset) const {
        if (offset == 0) return nullptr;
        return static_cast<void*>(static_cast<char*>(mem_base_) + (offset & OFFSET_MASK));
    }

    NodeOffset offset_from_ptr(void* ptr) const {
        if (ptr == nullptr) return 0;
        return static_cast<char*>(ptr) - static_cast<char*>(mem_base_);
    }

    public:
    std::vector<u64> scan(u64 start_key, u64 end_key) const {
        std::vector<u64> results;
        if (end_key < start_key) {
            return results;
        }

        size_t start_root_idx = start_key >> (64 - ROOT_BITS);
        size_t end_root_idx = end_key >> (64 - ROOT_BITS);

        for (size_t i = start_root_idx; i <= end_root_idx; ++i) {
            if (root_table_[i] != 0) {
                scan_recursive(root_table_[i], start_key, end_key, 0, results);
            }
        }
        // The results may not be perfectly sorted between root table entries if a split caused
        // keys to be distributed widely. A final sort is the most robust solution.
        std::sort(results.begin(), results.end());
        return results;
    }
    private:

    // --- Core Logic ---

    // Find a key in a leaf node using architecture-specific SIMD instructions for high performance.
    bool find_in_leaf_simd(u64 key, const LeafNode* leaf) const {
        #if defined(__x86_64__) || defined(_M_X64)
            if (leaf->count == 0) return false;
            const __m256i search_key_vec = _mm256_set1_epi64x(key);
            size_t i = 0;
            for (; i + 3 < leaf->count; i += 4) {
                __m256i keys_vec = _mm256_loadu_si256((const __m256i*)&leaf->keys[i]);
                if (_mm256_movemask_epi8(_mm256_cmpeq_epi64(search_key_vec, keys_vec)) != 0) {
                    return true;
                }
            }
            for (; i < leaf->count; ++i) {
                if (leaf->keys[i] == key) return true;
            }
            return false;
        #elif defined(__aarch64__)
            if (leaf->count == 0) return false;
            const uint64x2_t search_key_vec = vdupq_n_u64(key);
            size_t i = 0;
            for (; i + 1 < leaf->count; i += 2) {
                uint64x2_t keys_vec = vld1q_u64(&leaf->keys[i]);
                uint64x2_t cmp_res = vceqq_u64(search_key_vec, keys_vec);
                if (vmaxvq_u64(cmp_res) != 0) {
                    return true;
                }
            }
            for (; i < leaf->count; ++i) {
                if (leaf->keys[i] == key) return true;
            }
            return false;
        #else
            // Fallback for non-SIMD systems.
            const u64* result = std::lower_bound(leaf->keys, leaf->keys + leaf->count, key);
            return (result != leaf->keys + leaf->count && *result == key);
        #endif
    }

    void split_leaf_node(LeafNode* leaf, u64 new_key, NodeOffset* parent_link, int depth) {
        // Collect all keys that were in the full leaf, plus the new key.
        std::vector<u64> all_keys;
        all_keys.assign(leaf->keys, leaf->keys + leaf->count);
        all_keys.push_back(new_key);

        // Create the new internal node that will replace the old leaf.
        InternalNode* new_internal = static_cast<InternalNode*>(allocate(sizeof(InternalNode)));
        std::fill(new_internal->children, new_internal->children + INTERNAL_NODE_TABLE_SIZE, 0);

        // Point the parent link to the new internal node. This is critical.
        // From now on, any traversal will see the internal node instead of the old leaf.
        *parent_link = offset_from_ptr(new_internal) | INTERNAL_NODE_TAG;

        // Re-insert all the keys. The main insert() function is the single source of truth
        // for insertion logic. It will see the new InternalNode and create the necessary
        // new leaf nodes underneath it, and handle recursive splits if necessary.
        for (u64 key : all_keys) {
            insert(key);
        }
    }

    void scan_recursive(NodeOffset node_offset, u64 start_key, u64 end_key, int depth, std::vector<u64>& results) const {
        if (node_offset == 0) return;

        u64 tag = node_offset & TAG_MASK;
        void* node_ptr = ptr_from_offset(node_offset);

        if (tag == LEAF_NODE_TAG) {
            const LeafNode* leaf = static_cast<const LeafNode*>(node_ptr);
            for (int i = 0; i < leaf->count; ++i) {
                u64 key = leaf->keys[i];
                if (key >= start_key && key <= end_key) {
                    results.push_back(key);
                }
            }
        } else if (tag == INTERNAL_NODE_TAG) {
            const InternalNode* internal = static_cast<const InternalNode*>(node_ptr);
            for (int i = 0; i < INTERNAL_NODE_TABLE_SIZE; ++i) {
                // Future optimization: Prune branches that are provably outside the key range.
                // For now, we traverse all existing children. The sorted nature of the traversal
                // means the final result vector will be sorted.
                scan_recursive(internal->children[i], start_key, end_key, depth + 1, results);
            }
        }
    }
};


void run_benchmark(const std::string& benchmark_name, std::vector<u64>& keys, size_t mem_gb) {
    const size_t num_keys = keys.size();
    SART tree(mem_gb);

    std::cout << "\n\n=================================================" << std::endl;
    std::cout << "Starting Benchmark: " << benchmark_name << std::endl;
    std::cout << "=================================================\n" << std::endl;

    // --- Insertion Benchmark ---
    std::cout << "--- Benchmarking Insertions ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const u64 key : keys) {
        tree.insert(key);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double avg_insert_ns = static_cast<double>(duration.count()) / num_keys;
    std::cout << "Inserted " << num_keys << " keys in " << duration.count() / 1e9 << " seconds." << std::endl;
    std::cout << "Average insert latency: " << avg_insert_ns << " ns/key" << std::endl;
    double memory_mb = static_cast<double>(tree.get_memory_usage()) / (1024 * 1024);
    std::cout << "Total memory used: " << memory_mb << " MB" << std::endl;

    // --- Get Benchmark & Verification ---
    std::cout << "\n--- SART Get and Verifying ---" << std::endl;
    size_t not_found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (const u64 key : keys) {
        if (!tree.get(key)) {
            not_found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double avg_get_ns = static_cast<double>(duration.count()) / num_keys;
    std::cout << "Looked up " << num_keys << " keys in " << duration.count() / 1e9 << " seconds." << std::endl;
    std::cout << "Average get latency: " << avg_get_ns << " ns/key" << std::endl;
    if (not_found_count > 0) {
        std::cerr << "ERROR: " << not_found_count << " (SART) keys were not found after insertion!" << std::endl;
    } else {
        std::cout << "Verification: OK, all SART keys found." << std::endl;
    }

    // --- std::unordered_map Benchmark ---
    std::cout << "\n--- std::unordered_map Benchmark ---" << std::endl;
    std::unordered_map<u64, bool> map;
    start_time = std::chrono::high_resolution_clock::now();
    for (const u64 key : keys) {
        map.insert({key, true});
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double avg_map_insert_ns = static_cast<double>(duration.count()) / num_keys;
    std::cout << "Average insert latency: " << avg_map_insert_ns << " ns/key" << std::endl;

    not_found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (const u64 key : keys) {
        if (map.find(key) == map.end()) {
            not_found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double avg_map_get_ns = static_cast<double>(duration.count()) / num_keys;
    std::cout << "Average get latency: " << avg_map_get_ns << " ns/key" << std::endl;
     if (not_found_count > 0) {
        std::cerr << "ERROR: " << not_found_count << " (map) keys were not found after insertion!" << std::endl;
    } else {
        std::cout << "Verification: OK, all map keys found." << std::endl;
    }
    std::cout << "(std::unordered_map does not support ordered range scans)" << std::endl;


    // --- Range Scan Demonstration ---
    std::cout << "\n--- SART Complex Range Scans ---" << std::endl;
    std::sort(keys.begin(), keys.end());

    auto print_scan_results = [](const std::vector<u64>& results, const std::string& name, auto duration) {
        std::cout << "\n-- Scan Test: " << name << " --" << std::endl;
        double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
        std::cout << "Found " << results.size() << " keys in " << duration_ms << " ms." << std::endl;
        if (!results.empty()) {
            double ns_per_key = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() / (double)results.size();
            std::cout << "Latency: " << ns_per_key << " ns/key" << std::endl;
            std::cout << "First 5: ";
            for(size_t i=0; i < std::min((size_t)5, results.size()); ++i) std::cout << results[i] << " ";
            std::cout << std::endl;
            std::cout << "Last 5: ";
            for(size_t i=std::max((size_t)0, results.size() - 5); i < results.size(); ++i) std::cout << results[i] << " ";
            std::cout << std::endl;
        }
        if (!std::is_sorted(results.begin(), results.end())) {
            std::cerr << "ERROR: " << name << " results are not sorted!" << std::endl;
        } else {
            std::cout << "Verification: OK, sorted." << std::endl;
        }
    };

    auto time_and_run_scan = [&](const std::string& name, u64 start_key, u64 end_key) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto results = tree.scan(start_key, end_key);
        auto end_time = std::chrono::high_resolution_clock::now();
        print_scan_results(results, name, end_time - start_time);
        return results;
    };

    time_and_run_scan("Beginning", keys[0], keys[100]);
    time_and_run_scan("Middle", keys[num_keys / 2], keys[num_keys / 2] + 1000);
    time_and_run_scan("Large/Dense", keys[num_keys / 4], keys[num_keys / 4 + num_keys / 100]);

    u64 sparse_start = keys[num_keys / 3] + 1;
    u64 sparse_end = keys[num_keys / 3 + 1] - 1;
    if (sparse_start <= sparse_end) {
        auto sparse_results = time_and_run_scan("Sparse/Empty", sparse_start, sparse_end);
        if (!sparse_results.empty()) {
            std::cerr << "ERROR: Sparse scan should be empty but found " << sparse_results.size() << " keys." << std::endl;
        } else {
            std::cout << "Verification: OK, empty as expected." << std::endl;
        }
    }

    time_and_run_scan("End", keys[num_keys - 101], keys[num_keys - 1]);
}

int main() {
    try {
        std::vector<size_t> benchmark_sizes = {1000000, 10000000, 20000000};

        for (size_t num_keys : benchmark_sizes) {
            // Determine memory allocation based on benchmark size.
            size_t mem_gb = 2;
            if (num_keys >= 20000000) {
                mem_gb = 3;
            } else if (num_keys >= 10000000) {
                mem_gb = 2;
            }

            // --- Sequential Benchmark ---
            std::cout << "\n\n--- Generating " << num_keys << " sequential 64-bit keys... ---" << std::endl;
            std::vector<u64> sequential_keys(num_keys);
            for (size_t i = 0; i < num_keys; ++i) {
                sequential_keys[i] = i;
            }
            std::cout << "Key generation complete." << std::endl;
            run_benchmark("Sequential Keys", sequential_keys, mem_gb);

            // --- Random Benchmark ---
            std::cout << "\n\n--- Generating " << num_keys << " random 64-bit keys... ---" << std::endl;
            std::vector<u64> random_keys(num_keys);
            std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            for (size_t i = 0; i < num_keys; ++i) {
                random_keys[i] = rng();
            }
            std::cout << "Key generation complete." << std::endl;
            run_benchmark("Random Keys", random_keys, mem_gb);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
