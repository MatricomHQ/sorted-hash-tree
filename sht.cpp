#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <thread>
#include <memory>
#include <stdexcept>
#include <limits>
#include <utility>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <mutex>
#include <unordered_map>

#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// --- Foundational Data Structures (Adapted from Stax) ---

class ValueStore;

struct Record {
    uint32_t value_len;
    uint32_t dim;
    uint64_t value_or_offset;
    uint64_t coords[];

    static constexpr uint32_t INLINE_FLAG = 1U << 31;
    bool is_value_inlined() const { return (dim & INLINE_FLAG) != 0; }
    void set_inlined(bool is_inlined) { dim = (dim & ~INLINE_FLAG) | (is_inlined ? INLINE_FLAG : 0); }
    static size_t get_size(uint32_t d) { return sizeof(Record) + sizeof(uint64_t) * d; }
};

class ValueStore {
    std::atomic<uint64_t> next_offset_;
    uint8_t* value_pool_;
    static constexpr uint64_t MAX_VALUE_BYTES = 1024 * 1024 * 128; // 128MB
public:
    ValueStore() : next_offset_(1) {
        value_pool_ = (uint8_t*)mmap(nullptr, MAX_VALUE_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (value_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for value store");
    }
    ~ValueStore() { munmap(value_pool_, MAX_VALUE_BYTES); }
    uint64_t allocate_value(const void* data, uint32_t size) {
        size_t padded_size = (size + 7) & ~7;
        uint64_t offset = next_offset_.fetch_add(padded_size, std::memory_order_relaxed);
        if (offset + padded_size > MAX_VALUE_BYTES) throw std::runtime_error("Value pool exhausted");
        memcpy(value_pool_ + offset, data, size);
        return offset;
    }
};

namespace TaggedIndex {
    static constexpr uint32_t TAG_BIT = 1U << 31;
    static constexpr uint32_t INDEX_MASK = ~TAG_BIT;
    inline bool is_leaf(uint32_t idx) { return (idx & TAG_BIT) != 0; }
    inline bool is_node(uint32_t idx) { return (idx & TAG_BIT) == 0 && idx != 0; }
    inline uint32_t get_index(uint32_t idx) { return idx & INDEX_MASK; }
    inline uint32_t make_leaf_idx(uint32_t rec_idx) { return rec_idx | TAG_BIT; }
    inline uint32_t make_node_idx(uint32_t node_idx) { return node_idx; }
};

// --- Paged Node Structures ---

template<size_t SLOTS>
struct NodePage {
    std::atomic<uint32_t> children[SLOTS];
    NodePage() {
        for (size_t i = 0; i < SLOTS; ++i) children[i].store(0, std::memory_order_relaxed);
    }
};

// The FANOUT is fixed at 256 for the byte-based K-D cyclic heuristic.
constexpr size_t FANOUT = 256;
// SLOTS_PER_PAGE is specified as 16 in the requirements.
constexpr size_t SLOTS_PER_PAGE = 16;

struct RadixNode {
    uint32_t depth; // Node's distance from the root, determines split dimension.
    uint32_t representative_record_idx; // Optional, for tie-breaking and queries.

    using PageType = NodePage<SLOTS_PER_PAGE>;
    static constexpr size_t PAGES_PER_NODE = (FANOUT + SLOTS_PER_PAGE - 1) / SLOTS_PER_PAGE;

    // Children pointers are stored in pages since FANOUT > SLOTS_PER_PAGE.
    std::atomic<PageType*> children_or_pages[PAGES_PER_NODE];

    RadixNode() : depth(0), representative_record_idx(0) {
        for (size_t i = 0; i < PAGES_PER_NODE; ++i) {
            children_or_pages[i].store(nullptr, std::memory_order_relaxed);
        }
    }
};

class PageManager; // Forward declaration

class NodeManager {
private:
    std::atomic<uint32_t> next_node_idx_;
    uint8_t* node_pool_;
    const size_t node_size_;
    static constexpr uint32_t MAX_NODES = 2 * 1024 * 1024; // Reduced from 32M
    static constexpr size_t THREAD_CACHE_SIZE = 64;
    static thread_local uint32_t node_cache_[THREAD_CACHE_SIZE];
    static thread_local int cache_ptr_;

    void refill_cache() {
        uint32_t start_idx = next_node_idx_.fetch_add(THREAD_CACHE_SIZE, std::memory_order_relaxed);
        if (start_idx + THREAD_CACHE_SIZE >= MAX_NODES) throw std::runtime_error("Node pool exhausted");
        for (size_t i = 0; i < THREAD_CACHE_SIZE; ++i) node_cache_[i] = start_idx + i;
        cache_ptr_ = THREAD_CACHE_SIZE - 1;
    }
public:
    NodeManager(size_t node_size) : next_node_idx_(1), node_size_(node_size) {
        size_t pool_size = (size_t)MAX_NODES * node_size_;
        node_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (node_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for node pool");
    }
    ~NodeManager() { munmap(node_pool_, (size_t)MAX_NODES * node_size_); }
    template<typename TNode> TNode* get_node(uint32_t idx) { return reinterpret_cast<TNode*>(node_pool_ + (size_t)idx * node_size_); }
    template<typename TNode> uint32_t allocate_node() {
        if (cache_ptr_ < 0) refill_cache();
        uint32_t new_idx = node_cache_[cache_ptr_--];
        new (get_node<TNode>(new_idx)) TNode();
        return new_idx;
    }
    size_t get_mem_usage() const { return (size_t)next_node_idx_.load() * node_size_; }
};
thread_local uint32_t NodeManager::node_cache_[NodeManager::THREAD_CACHE_SIZE];
thread_local int NodeManager::cache_ptr_ = -1;

class PageManager {
private:
    std::atomic<uint32_t> next_page_idx_;
    uint8_t* page_pool_;
    const size_t page_size_;
    static constexpr uint32_t MAX_PAGES = 32 * 1024 * 1024;
    static constexpr size_t THREAD_CACHE_SIZE = 64;
    static thread_local uint32_t page_cache_[THREAD_CACHE_SIZE];
    static thread_local int cache_ptr_;

    void refill_cache() {
        uint32_t start_idx = next_page_idx_.fetch_add(THREAD_CACHE_SIZE, std::memory_order_relaxed);
        if (start_idx + THREAD_CACHE_SIZE >= MAX_PAGES) throw std::runtime_error("Page pool exhausted");
        for (size_t i = 0; i < THREAD_CACHE_SIZE; ++i) page_cache_[i] = start_idx + i;
        cache_ptr_ = THREAD_CACHE_SIZE - 1;
    }

public:
    PageManager(size_t page_size) : next_page_idx_(1), page_size_(page_size) {
        size_t pool_size = (size_t)MAX_PAGES * page_size_;
        page_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (page_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for page pool");
    }
    ~PageManager() { munmap(page_pool_, (size_t)MAX_PAGES * page_size_); }

    template<typename TPage> TPage* get_page(uint32_t idx) { return reinterpret_cast<TPage*>(page_pool_ + (size_t)idx * page_size_); }
    template<typename TPage> uint32_t allocate_page() {
        if (cache_ptr_ < 0) refill_cache();
        uint32_t new_idx = page_cache_[cache_ptr_--];
        new (get_page<TPage>(new_idx)) TPage();
        return new_idx; // Note: page index is not tagged
    }
     template<typename TPage> TPage* allocate_page_and_get_ptr() {
        if (cache_ptr_ < 0) refill_cache();
        uint32_t new_idx = page_cache_[cache_ptr_--];
        TPage* page = get_page<TPage>(new_idx);
        new (page) TPage();
        return page;
    }
    size_t get_mem_usage() const { return (size_t)next_page_idx_.load() * page_size_; }
};
thread_local uint32_t PageManager::page_cache_[PageManager::THREAD_CACHE_SIZE];
thread_local int PageManager::cache_ptr_ = -1;


class RecordManager {
private:
    std::atomic<uint32_t> next_record_idx_;
    uint8_t* record_pool_;
    ValueStore* value_store_;
    const uint32_t dim_;
    const size_t record_size_with_coords_;
    static constexpr uint32_t MAX_RECORDS = 16 * 1024 * 1024;
    static constexpr size_t THREAD_CACHE_SIZE = 64;
    static thread_local uint32_t record_cache_[THREAD_CACHE_SIZE];
    static thread_local int cache_ptr_;

    void refill_cache() {
        uint32_t start_idx = next_record_idx_.fetch_add(THREAD_CACHE_SIZE, std::memory_order_relaxed);
        if (start_idx + THREAD_CACHE_SIZE >= MAX_RECORDS) throw std::runtime_error("Record pool exhausted");
        for (size_t i = 0; i < THREAD_CACHE_SIZE; ++i) record_cache_[i] = start_idx + i;
        cache_ptr_ = THREAD_CACHE_SIZE - 1;
    }
public:
    RecordManager(ValueStore* vs, uint32_t d) : next_record_idx_(1), value_store_(vs), dim_(d), record_size_with_coords_(Record::get_size(d)) {
        size_t pool_size = (size_t)MAX_RECORDS * record_size_with_coords_;
        record_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (record_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for record pool");
    }
    ~RecordManager() { munmap(record_pool_, (size_t)MAX_RECORDS * record_size_with_coords_); }
    Record* get_record(uint32_t idx) { return reinterpret_cast<Record*>(record_pool_ + (size_t)idx * record_size_with_coords_); }
    uint32_t allocate_record(const uint64_t* coords, uint64_t value) {
        if (cache_ptr_ < 0) refill_cache();
        uint32_t new_idx = record_cache_[cache_ptr_--];
        Record* rec = get_record(new_idx);
        rec->dim = dim_;
        rec->value_len = sizeof(value);
        rec->value_or_offset = value;
        rec->set_inlined(true);
        memcpy(rec->coords, coords, sizeof(uint64_t) * dim_);
        return new_idx;
    }
    size_t get_mem_usage() const { return (size_t)next_record_idx_.load() * record_size_with_coords_; }
};
thread_local uint32_t RecordManager::record_cache_[RecordManager::THREAD_CACHE_SIZE];
thread_local int RecordManager::cache_ptr_ = -1;

class KeyValueRadixTree {
    using NodeType = RadixNode;

    NodeManager& node_manager_;
    RecordManager& record_manager_;
    PageManager& page_manager_;
    std::atomic<uint32_t>& root_ptr_;
    const uint32_t dim_;

    // K-D Cyclic Heuristic: extract the correct byte from the key based on depth.
    static inline uint8_t get_key_byte_by_depth(const uint64_t* coords, uint32_t depth, uint32_t d) {
        // This is the K-D cyclic heuristic.
        // Since D (dim) is fixed at 8, a power of two, we can optimize the modulo and division
        // into fast bitwise operations for a significant performance gain in this critical function.
        uint32_t split_dim = depth & 7;    // depth % 8
        uint32_t byte_in_dim = depth >> 3; // depth / 8

        // A depth can go from 0 up to D*8 - 1 (0-63).
        // If byte_in_dim is >= 8, the key has been fully processed.
        if (byte_in_dim >= 8) {
            return 0;
        }
        // Extract the byte from the correct dimension's coordinate.
        // Bytes are indexed from most significant (0) to least significant (7).
        return (coords[split_dim] >> (56 - (byte_in_dim * 8))) & 0xFF;
    }

public:
    KeyValueRadixTree(NodeManager& nm, RecordManager& rm, PageManager& pm, std::atomic<uint32_t>& root, uint32_t d)
        : node_manager_(nm), record_manager_(rm), page_manager_(pm), root_ptr_(root), dim_(d) {}

    void insert(const uint64_t* coords, uint64_t value) {
    restart:
        std::atomic<uint32_t>* parent_slot = &root_ptr_;
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);
        uint32_t depth = 0; // The depth we are currently inspecting in the key.

        while (TaggedIndex::is_node(current_idx)) {
            NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(current_idx));

            // A node's depth indicates the byte of the key to test. By setting our local depth
            // to the node's depth, we handle compressed paths where intermediate nodes are skipped.
            depth = node->depth;

            uint8_t frag = get_key_byte_by_depth(coords, depth, dim_);
            parent_slot = get_child_slot<true>(node, frag);
            if (!parent_slot) goto restart; // Contention on page creation, retry.

            current_idx = parent_slot->load(std::memory_order_acquire);
            depth++; // After this decision, we move to the next depth for the next level.
        }

        // Case 1: Found an empty slot. We can insert a new leaf directly.
        if (current_idx == 0) {
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_release, std::memory_order_relaxed)) {
                return; // Success
            }
            goto restart; // CAS failed, another thread interfered. Retry.
        }

        // Case 2: Found a leaf node. This is a collision and requires a split.
        if (TaggedIndex::is_leaf(current_idx)) {
            Record* existing_rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));

            // The path was common up until `depth`. Find the first differing byte from this point.
            uint32_t split_depth = depth;
            const uint32_t max_depth = dim_ * 8;

            while (split_depth < max_depth && get_key_byte_by_depth(coords, split_depth, dim_) == get_key_byte_by_depth(existing_rec->coords, split_depth, dim_)) {
                split_depth++;
            }

            // If keys are fully identical, the insert is a no-op.
            // (An alternative would be to update the value.)
            if (split_depth == max_depth) {
                return;
            }

            // Create the new node where the two keys diverge.
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            uint32_t split_node_idx = node_manager_.allocate_node<NodeType>();
            NodeType* split_node = node_manager_.get_node<NodeType>(split_node_idx);
            split_node->depth = split_depth;
            split_node->representative_record_idx = new_rec_idx;

            uint8_t new_key_frag = get_key_byte_by_depth(coords, split_depth, dim_);
            uint8_t existing_key_frag = get_key_byte_by_depth(existing_rec->coords, split_depth, dim_);

            get_child_slot<true>(split_node, new_key_frag)->store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);
            get_child_slot<true>(split_node, existing_key_frag)->store(current_idx, std::memory_order_relaxed);

            // Create a chain of intermediate, single-child nodes for the shared prefix.
            uint32_t child_idx_to_link = TaggedIndex::make_node_idx(split_node_idx);
            for (int d = split_depth - 1; d >= (int)depth; --d) {
                uint32_t intermediate_node_idx = node_manager_.allocate_node<NodeType>();
                NodeType* intermediate_node = node_manager_.get_node<NodeType>(intermediate_node_idx);
                intermediate_node->depth = d;
                intermediate_node->representative_record_idx = new_rec_idx;

                uint8_t frag = get_key_byte_by_depth(coords, d, dim_);
                get_child_slot<true>(intermediate_node, frag)->store(child_idx_to_link, std::memory_order_relaxed);
                child_idx_to_link = TaggedIndex::make_node_idx(intermediate_node_idx);
            }

            // Atomically swap the original leaf with our new subtree.
            if (parent_slot->compare_exchange_strong(current_idx, child_idx_to_link, std::memory_order_release, std::memory_order_relaxed)) {
                return; // Success
            }
            goto restart; // CAS failed, another thread interfered. Retry.
        }
    }

    struct QueryBox {
        const uint64_t* min_coords;
        const uint64_t* max_coords;
    };

    std::vector<Record*> box_query(const QueryBox& query) {
        std::vector<Record*> results;
        box_query_recursive(root_ptr_.load(std::memory_order_acquire), query, results);
        return results;
    }

private:
    template<bool create_if_missing>
    std::atomic<uint32_t>* get_child_slot(NodeType* node, int frag) {
        size_t page_idx = frag / SLOTS_PER_PAGE;
        size_t slot_idx_in_page = frag % SLOTS_PER_PAGE;

        // This is the paged implementation, as FANOUT is fixed to 256.
        typename NodeType::PageType* page = node->children_or_pages[page_idx].load(std::memory_order_acquire);
        if (page == nullptr) {
            if constexpr (create_if_missing) {
                typename NodeType::PageType* new_page = page_manager_.template allocate_page_and_get_ptr<typename NodeType::PageType>();
                if (node->children_or_pages[page_idx].compare_exchange_strong(page, new_page, std::memory_order_release, std::memory_order_relaxed)) {
                    page = new_page;
                } else {
                    // Another thread beat us, leak our page for now.
                    page = node->children_or_pages[page_idx].load(std::memory_order_acquire);
                }
            } else {
                return nullptr; // Page does not exist, and we are not creating it.
            }
        }
        return &page->children[slot_idx_in_page];
    }

    void box_query_recursive(uint32_t node_idx, const QueryBox& query, std::vector<Record*>& results) {
        if (!node_idx) return;

        // Case 1: Leaf node. Perform full dimensional check.
        if (TaggedIndex::is_leaf(node_idx)) {
            Record* rec = record_manager_.get_record(TaggedIndex::get_index(node_idx));
            if (!rec) return;

            bool in_box = true;
            for (uint32_t i = 0; i < dim_; ++i) {
                if (rec->coords[i] < query.min_coords[i] || rec->coords[i] > query.max_coords[i]) {
                    in_box = false;
                    break;
                }
            }
            if (in_box) {
                results.push_back(rec);
            }
            return;
        }

        // Case 2: Internal node. Prune children based on the node's split dimension.
        NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(node_idx));
        uint32_t depth = node->depth;
        if (depth >= dim_ * 8) return; // Beyond max key depth

        uint32_t split_dim = depth % dim_;
        uint32_t byte_in_dim = depth / dim_;

        uint64_t q_min_coord = query.min_coords[split_dim];
        uint64_t q_max_coord = query.max_coords[split_dim];

        uint8_t start_frag = 0;
        uint8_t end_frag = 255;

        // Create a mask to compare prefixes (all bytes before the current one).
        uint64_t prefix_mask = (byte_in_dim == 0) ? 0 : (-1ULL << (64 - (byte_in_dim * 8)));

        // If the prefixes of the min and max query bounds are the same for this dimension,
        // we can narrow the search range for the current byte (fragment). Otherwise, we must scan all children.
        if ((q_min_coord & prefix_mask) == (q_max_coord & prefix_mask)) {
            start_frag = (q_min_coord >> (56 - (byte_in_dim * 8))) & 0xFF;
            end_frag = (q_max_coord >> (56 - (byte_in_dim * 8))) & 0xFF;
        }

        for (int frag = start_frag; frag <= end_frag; ++frag) {
            std::atomic<uint32_t>* child_slot = get_child_slot<false>(node, frag);
            if (!child_slot) continue;

            uint32_t child_idx = child_slot->load(std::memory_order_acquire);
            if (child_idx) {
                // The query box is not modified for the recursive call.
                // Pruning is stateless and only depends on the node's depth.
                box_query_recursive(child_idx, query, results);
            }
        }
    }
};

// =================================================================================
// --- "THE GAUNTLET" BENCHMARK SUITE ---
// =================================================================================

const int DIMS = 8;
struct Key { uint64_t coords[DIMS]; };

// --- Query Box Generation Helpers ---

// Generates a query box for a point lookup
KeyValueRadixTree::QueryBox create_point_query(const Key& key, Key& storage) {
    storage = key;
    return {storage.coords, storage.coords};
}

// --- Benchmark Implementations ---

void benchmark_unordered_map(const std::vector<Key>& keys, const std::vector<Key>& miss_keys) {
    std::cout << "\n--- std::unordered_map Benchmark (baseline) ---" << std::endl;
    std::unordered_map<std::string, uint64_t> umap;
    size_t num_keys = keys.size();

    // Insertion
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) {
        std::string key_str(reinterpret_cast<const char*>(keys[i].coords), DIMS * sizeof(uint64_t));
        umap[key_str] = i;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "  Avg Latency (Insert): " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/insert" << std::endl;

    // Hit Latency
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) {
        std::string key_str(reinterpret_cast<const char*>(keys[i].coords), DIMS * sizeof(uint64_t));
        volatile auto it = umap.find(key_str);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "  Avg Latency (Get Hit):  " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;

    // Miss Latency
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) {
        std::string key_str(reinterpret_cast<const char*>(miss_keys[i].coords), DIMS * sizeof(uint64_t));
        volatile auto it = umap.find(key_str);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "  Avg Latency (Get Miss): " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;
}

void benchmark_insert(KeyValueRadixTree& tree, const std::vector<Key>& keys, int num_threads) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    size_t num_keys = keys.size();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t start = i * (num_keys / num_threads);
            size_t end = (i == num_threads - 1) ? num_keys : start + (num_keys / num_threads);
            for (size_t j = start; j < end; ++j) {
                tree.insert(keys[j].coords, j);
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double throughput = (duration.count() > 0) ? (double)num_keys / duration.count() * 1000000.0 : 0;

    std::cout << "  Threads: " << std::setw(2) << num_threads
              << " | Total Time: " << std::setw(8) << duration.count() << " us"
              << " | Throughput: " << std::fixed << std::setprecision(2) << throughput << " ops/sec" << std::endl;
}

void benchmark_point_get(KeyValueRadixTree& tree, const std::vector<Key>& keys_to_find, const std::string& name, size_t expected_results_per_key) {
    std::cout << "\n--- " << name << " ---" << std::endl;
    size_t num_to_find = keys_to_find.size();
    std::vector<Key> storage(num_to_find);
    size_t found_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_to_find; ++i) {
        auto query = create_point_query(keys_to_find[i], storage[i]);
        auto results = tree.box_query(query);
        if (results.size() == expected_results_per_key) {
            found_count++;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "  Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_to_find << " ns/get" << std::endl;
    std::cout << "  Verification: " << (found_count == num_to_find ? "SUCCESS" : "FAILED") << " (Correctly verified " << found_count << "/" << num_to_find << " queries)" << std::endl;
}

void benchmark_box_query(KeyValueRadixTree& tree, const std::string& name, const KeyValueRadixTree::QueryBox& query, size_t expected_min_results) {
     std::cout << "\n--- " << name << " ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto results = tree.box_query(query);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    bool success = results.size() >= expected_min_results;

    std::cout << "  Query Time: " << duration.count() << " us" << std::endl;
    std::cout << "  Results Found: " << results.size() << " (Expected >=" << expected_min_results << ")" << std::endl;
    std::cout << "  Verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
}


int main(int argc, char* argv[]) {
    try {
        const size_t NUM_KEYS = 1000000;

        std::cout << "--- CYCLOPS Engine Benchmark ---" << std::endl;
        std::cout << "--- Fixed 8-dimensional keys (uint64_t[8]) ---" << std::endl;
        std::cout << "--- FANOUT=256, SLOTS_PER_PAGE=16 ---" << std::endl;
        std::cout << "--- " << NUM_KEYS << " random keys ---" << std::endl;

        // --- Key Generation ---
        std::cout << "\nGenerating " << NUM_KEYS << " random keys..." << std::endl;
        std::vector<Key> keys(NUM_KEYS);
        std::vector<Key> miss_keys(NUM_KEYS);
        std::mt19937_64 rng(12345);
        for(size_t i = 0; i < NUM_KEYS; ++i) {
            for (int d = 0; d < DIMS; ++d) {
                keys[i].coords[d] = rng();
                miss_keys[i].coords[d] = rng();
            }
        }
        std::cout << "Key generation complete." << std::endl;

        // --- Run baseline benchmark before CYCLOPS engine ---
        benchmark_unordered_map(keys, miss_keys);

        // --- Scalability & Insertion Benchmark ---
        std::cout << "\n--- CYCLOPS Engine: Scalability_Insert ---" << std::endl;
        std::vector<int> thread_counts = {1, 2, 4, 8};
        for (int num_threads : thread_counts) {
            ValueStore vs;
            NodeManager nm(sizeof(RadixNode));
            RecordManager rm(&vs, DIMS);
            PageManager pm(sizeof(NodePage<SLOTS_PER_PAGE>));
            std::atomic<uint32_t> root_ptr(0);
            KeyValueRadixTree tree(nm, rm, pm, root_ptr, DIMS);
            benchmark_insert(tree, keys, num_threads);
        }

        // --- Populate a single tree for all query benchmarks ---
        std::cout << "\nPopulating main tree for querying with 8 threads..." << std::endl;
        ValueStore final_vs;
        NodeManager final_nm(sizeof(RadixNode));
        RecordManager final_rm(&final_vs, DIMS);
        PageManager final_pm(sizeof(NodePage<SLOTS_PER_PAGE>));
        std::atomic<uint32_t> final_root_ptr(0);
        KeyValueRadixTree final_tree(final_nm, final_rm, final_pm, final_root_ptr, DIMS);
        benchmark_insert(final_tree, keys, 8);

        // --- Memory Usage ---
        std::cout << "\n--- Memory_Per_Key ---" << std::endl;
        size_t total_mem = final_nm.get_mem_usage() + final_rm.get_mem_usage() + final_pm.get_mem_usage();
        std::cout << "  Total Memory: " << std::fixed << std::setprecision(2) << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Per Key: " << std::fixed << std::setprecision(2) << (double)total_mem / NUM_KEYS << " bytes/key" << std::endl;

        // --- Point Query Performance ---
        std::cout << "\n--- Point Performance ---";
        benchmark_point_get(final_tree, keys, "Point_Get_Hit", 1);
        benchmark_point_get(final_tree, miss_keys, "Point_Get_Miss", 0);

        // --- Spatial Query Performance ---
        std::cout << "\n--- Spatial Query Performance (Correctness Guaranteed) ---";
        Key min_key_storage, max_key_storage;

        auto create_box_around_key = [&](const Key& center_key, const std::vector<double>& percentages) {
            for (int i = 0; i < DIMS; ++i) {
                uint64_t range_span = percentages[i] * std::numeric_limits<uint64_t>::max();
                uint64_t center = center_key.coords[i];
                min_key_storage.coords[i] = (center < range_span / 2) ? 0 : center - range_span / 2;
                max_key_storage.coords[i] = (center > std::numeric_limits<uint64_t>::max() - range_span / 2) ? std::numeric_limits<uint64_t>::max() : center + range_span / 2;
            }
            return KeyValueRadixTree::QueryBox{min_key_storage.coords, max_key_storage.coords};
        };

        Key& center_key_for_spatial = keys[rng() % NUM_KEYS];
        benchmark_box_query(final_tree, "Wide_Box_Query", create_box_around_key(center_key_for_spatial, std::vector<double>(DIMS, 0.25)), 1);
        benchmark_box_query(final_tree, "Narrow_Box_Query", create_box_around_key(center_key_for_spatial, std::vector<double>(DIMS, 0.001)), 1);

        std::vector<double> elongated_dims;
        for(int i=0; i<DIMS; ++i) elongated_dims.push_back(i % 2 == 0 ? 0.50 : 0.001);
        benchmark_box_query(final_tree, "Elongated_Box_Query", create_box_around_key(center_key_for_spatial, elongated_dims), 1);

        // --- Lexicographical Emulation & Hybrid Performance ---
        std::cout << "\n--- Lexicographical Emulation & Hybrid Performance (Correctness Guaranteed) ---";
        Key min_storage, max_storage;

        for(int i=0; i<DIMS; ++i) {
            min_storage.coords[i] = keys[rng() % NUM_KEYS].coords[i];
            max_storage.coords[i] = keys[rng() % NUM_KEYS].coords[i];
            if (min_storage.coords[i] > max_storage.coords[i]) std::swap(min_storage.coords[i], max_storage.coords[i]);
        }
        // For a random lexicographical scan, we can't guarantee results, so we expect >= 0.
        benchmark_box_query(final_tree, "Lex_Scan_Emulation", {min_storage.coords, max_storage.coords}, 0);

        // --- Hybrid Queries centered around a known key ---
        Key& center_key_for_hybrid = keys[rng() % NUM_KEYS];
        Key prefix_min, prefix_max;
        for(int i=0; i<DIMS; ++i) {
            if (i < 4) { // Prefix part
                prefix_min.coords[i] = center_key_for_hybrid.coords[i];
                prefix_max.coords[i] = center_key_for_hybrid.coords[i];
            } else { // Narrow box part centered on the key
                 uint64_t range_span = 0.001 * std::numeric_limits<uint64_t>::max();
                uint64_t center = center_key_for_hybrid.coords[i];
                prefix_min.coords[i] = (center < range_span / 2) ? 0 : center - range_span / 2;
                prefix_max.coords[i] = (center > std::numeric_limits<uint64_t>::max() - range_span / 2) ? std::numeric_limits<uint64_t>::max() : center + range_span / 2;
            }
        }
        benchmark_box_query(final_tree, "Prefix_Plus_Narrow_Box", {prefix_min.coords, prefix_max.coords}, 1);

        Key suffix_min, suffix_max;
        for(int i=0; i<DIMS; ++i) {
            if (i < 4) { // Wide box part centered on the key
                uint64_t range_span = 0.25 * std::numeric_limits<uint64_t>::max();
                uint64_t center = center_key_for_hybrid.coords[i];
                suffix_min.coords[i] = (center < range_span / 2) ? 0 : center - range_span / 2;
                suffix_max.coords[i] = (center > std::numeric_limits<uint64_t>::max() - range_span / 2) ? std::numeric_limits<uint64_t>::max() : center + range_span / 2;
            } else { // Suffix part
                suffix_min.coords[i] = center_key_for_hybrid.coords[i];
                suffix_max.coords[i] = center_key_for_hybrid.coords[i];
            }
        }
        benchmark_box_query(final_tree, "Wide_Box_Plus_Suffix", {suffix_min.coords, suffix_max.coords}, 1);

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
