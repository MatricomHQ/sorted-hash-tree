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

template<size_t FANOUT, size_t SLOTS_PER_PAGE = 16, size_t PAGED_THRESHOLD = 16>
struct RadixNode {
    uint32_t depth; // The node's distance in traversal steps from the root.
    uint32_t representative_record_idx;

    // Conditional compilation for node structure
    static constexpr bool IS_PAGED = FANOUT >= PAGED_THRESHOLD;

    using PageType = NodePage<SLOTS_PER_PAGE>;
    static constexpr size_t PAGES_PER_NODE = IS_PAGED ? (FANOUT + SLOTS_PER_PAGE - 1) / SLOTS_PER_PAGE : 0;

    // The actual storage for children pointers
    std::conditional_t<
        IS_PAGED,
        std::atomic<PageType*>[PAGES_PER_NODE],
        std::atomic<uint32_t>[FANOUT]
    > children_or_pages;

    RadixNode() : depth(0), representative_record_idx(0) {
        if constexpr (IS_PAGED) {
            for (size_t i = 0; i < PAGES_PER_NODE; ++i) {
                children_or_pages[i].store(nullptr, std::memory_order_relaxed);
            }
        } else {
            for (size_t i = 0; i < FANOUT; ++i) {
                children_or_pages[i].store(0, std::memory_order_relaxed);
            }
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

const int MAX_DIMS = 8;
struct Key { uint64_t coords[MAX_DIMS]; };
struct QueryBox { uint64_t min_coords[MAX_DIMS]; uint64_t max_coords[MAX_DIMS]; };

template<size_t FANOUT, size_t SLOTS_PER_PAGE = 16>
class KeyValueRadixTree {
    using NodeType = RadixNode<FANOUT, SLOTS_PER_PAGE>;

    NodeManager& node_manager_;
    RecordManager& record_manager_;
    PageManager& page_manager_;
    std::atomic<uint32_t>& root_ptr_;
    const uint32_t dim_;

    // K-D Cyclic Heuristic for selecting the key fragment (byte)
    static inline int get_key_fragment(const uint64_t* coords, int depth, int d) {
        if (d <= 0) return 0;

        if constexpr (FANOUT == 256) { // Byte-based
            int split_dim = depth % d;
            int chunk_in_dim = depth / d;

            // Each dimension is one uint64_t, which has 8 bytes.
            if (chunk_in_dim >= 8) {
                return -1; // Past the end of the key.
            }
            // Extract the byte from the coordinate of the chosen dimension.
            return (coords[split_dim] >> (56 - (chunk_in_dim * 8))) & 0xFF;
        } else { // FANOUT 16 (Nibble-based) - Not required by spec, but kept for completeness
            int split_dim = depth % d;
            int chunk_in_dim = depth / d;
             if (chunk_in_dim >= 16) {
                return -1; // Past the end of the key
            }
            return (coords[split_dim] >> (60 - (chunk_in_dim * 4))) & 0x0F;
        }
    }

public:
    KeyValueRadixTree(NodeManager& nm, RecordManager& rm, PageManager& pm, std::atomic<uint32_t>& root, uint32_t d)
        : node_manager_(nm), record_manager_(rm), page_manager_(pm), root_ptr_(root), dim_(d) {}

    void insert(const uint64_t* coords, uint64_t value) {
    restart:
        std::atomic<uint32_t>* parent_slot = &root_ptr_;
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);
        uint32_t traversal_depth = 0;

        while (TaggedIndex::is_node(current_idx)) {
            NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(current_idx));
            traversal_depth = node->depth;

            int frag = get_key_fragment(coords, traversal_depth, dim_);
            if (frag == -1) { return; /* Key is a prefix of an existing key, or too long. */ }

            parent_slot = get_child_slot<true>(node, frag);
            if (!parent_slot) goto restart; // Contention on page creation

            current_idx = parent_slot->load(std::memory_order_acquire);
            traversal_depth++;
        }

        // Case 1: Traversal led to an empty slot. Insert a new leaf.
        if (current_idx == 0) {
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_release, std::memory_order_relaxed)) {
                return;
            }
            goto restart; // CAS failed, another thread intervened.
        }

        // Case 2: Traversal led to a leaf. This is a collision, so we must split.
        if (TaggedIndex::is_leaf(current_idx)) {
            uint32_t existing_rec_idx = TaggedIndex::get_index(current_idx);
            Record* existing_rec = record_manager_.get_record(existing_rec_idx);

            // If keys are identical, we are done.
            if (memcmp(coords, existing_rec->coords, dim_ * sizeof(uint64_t)) == 0) {
                return; // Key already exists.
            }

            // Find the first depth at which the keys' fragments differ.
            uint32_t diff_depth = traversal_depth;
            int new_key_frag, existing_key_frag;
            while (true) {
                new_key_frag = get_key_fragment(coords, diff_depth, dim_);
                existing_key_frag = get_key_fragment(existing_rec->coords, diff_depth, dim_);
                if (new_key_frag != existing_key_frag || new_key_frag == -1) {
                    break; // Found the point of divergence or end of key.
                }
                diff_depth++;
            }

            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);

            // Create the new node where the paths will diverge.
            uint32_t split_node_idx = node_manager_.allocate_node<NodeType>();
            NodeType* split_node = node_manager_.get_node<NodeType>(split_node_idx);
            split_node->depth = diff_depth;
            split_node->representative_record_idx = new_rec_idx;

            // Add the new and existing records as leaves of the new split node.
            std::atomic<uint32_t>* new_child_slot = get_child_slot<true>(split_node, new_key_frag);
            new_child_slot->store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);

            std::atomic<uint32_t>* existing_child_slot = get_child_slot<true>(split_node, existing_key_frag);
            existing_child_slot->store(TaggedIndex::make_leaf_idx(existing_rec_idx), std::memory_order_relaxed);

            // If the keys differed at a greater depth than the current traversal,
            // we must create a chain of intermediate single-child nodes.
            uint32_t top_of_chain_idx = TaggedIndex::make_node_idx(split_node_idx);
            for (int d = diff_depth - 1; d >= (int)traversal_depth; --d) {
                uint32_t intermediate_node_idx = node_manager_.allocate_node<NodeType>();
                NodeType* intermediate_node = node_manager_.get_node<NodeType>(intermediate_node_idx);
                intermediate_node->depth = d;
                intermediate_node->representative_record_idx = new_rec_idx;

                int common_frag = get_key_fragment(coords, d, dim_);
                std::atomic<uint32_t>* child_slot = get_child_slot<true>(intermediate_node, common_frag);
                child_slot->store(top_of_chain_idx, std::memory_order_release);

                top_of_chain_idx = TaggedIndex::make_node_idx(intermediate_node_idx);
            }

            // Atomically swap the old leaf with the new subtree.
            if (parent_slot->compare_exchange_strong(current_idx, top_of_chain_idx, std::memory_order_release, std::memory_order_relaxed)) {
                return;
            }
            goto restart; // CAS failed, another thread changed the tree.
        }
    }

    bool get(const uint64_t* coords) {
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);
        while (TaggedIndex::is_node(current_idx)) {
            NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(current_idx));
            int frag = get_key_fragment(coords, node->depth, dim_);
             if (frag == -1) { // Key is shorter than path, so can't exist.
                return false;
            }

            std::atomic<uint32_t>* child_slot = get_child_slot<false>(node, frag);
            if (!child_slot) return false; // Page not allocated, so key can't exist

            current_idx = child_slot->load(std::memory_order_acquire);
        }
        if (TaggedIndex::is_leaf(current_idx)) {
            Record* rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));
            if (!rec) return false;
            return memcmp(rec->coords, coords, dim_ * sizeof(uint64_t)) == 0;
        }
        return false;
    }

    std::vector<Record*> box_query(const QueryBox& box) {
        std::vector<Record*> results;
        bool tight_mins[MAX_DIMS], tight_maxs[MAX_DIMS];
        for(uint32_t i=0; i<dim_; ++i) {
            tight_mins[i] = true;
            tight_maxs[i] = true;
        }
        box_query_recursive(root_ptr_.load(std::memory_order_acquire), 0, box, tight_mins, tight_maxs, results);
        return results;
    }

private:
    template<bool create_if_missing>
    std::atomic<uint32_t>* get_child_slot(NodeType* node, int frag) {
        if (frag < 0) return nullptr;
        if constexpr (NodeType::IS_PAGED) {
            size_t page_idx = frag / SLOTS_PER_PAGE;
            size_t slot_idx_in_page = frag % SLOTS_PER_PAGE;

            typename NodeType::PageType* page = node->children_or_pages[page_idx].load(std::memory_order_acquire);
            if (page == nullptr) {
                if constexpr (create_if_missing) {
                    typename NodeType::PageType* new_page = page_manager_.template allocate_page_and_get_ptr<typename NodeType::PageType>();
                    if (node->children_or_pages[page_idx].compare_exchange_strong(page, new_page, std::memory_order_release, std::memory_order_relaxed)) {
                        page = new_page;
                    } else {
                        page = node->children_or_pages[page_idx].load(std::memory_order_acquire);
                    }
                } else {
                    return nullptr;
                }
            }
            return &page->children[slot_idx_in_page];
        } else {
            return &node->children_or_pages[frag];
        }
    }

    void box_query_recursive(uint32_t node_idx, int parent_depth, const QueryBox& box, bool tight_mins[], bool tight_maxs[], std::vector<Record*>& results) {
        if (!node_idx) return;

        if (TaggedIndex::is_leaf(node_idx)) {
            Record* rec = record_manager_.get_record(TaggedIndex::get_index(node_idx));
            if (rec) {
                bool is_inside = true;
                for (uint32_t d = 0; d < dim_; ++d) {
                    if (rec->coords[d] < box.min_coords[d] || rec->coords[d] > box.max_coords[d]) {
                        is_inside = false;
                        break;
                    }
                }
                if (is_inside) {
                    results.push_back(rec);
                }
            }
            return;
        }

        NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(node_idx));

        auto get_byte = [](uint64_t val, int chunk) { return (val >> (56 - (chunk * 8))) & 0xFF; };

        // --- Path Compression Pruning ---
        // This is the crucial fix. Check the compressed path from the parent's depth to this node's depth.
        Record* rep_rec = record_manager_.get_record(node->representative_record_idx);
        if (rep_rec) {
            for (int d = parent_depth; d < node->depth; ++d) {
                int check_dim = d % dim_;
                int chunk_in_dim = d / dim_;
                if (chunk_in_dim >= 8) continue;

                int rep_frag = get_byte(rep_rec->coords[check_dim], chunk_in_dim);

                if (tight_mins[check_dim]) {
                    int min_frag = get_byte(box.min_coords[check_dim], chunk_in_dim);
                    if (rep_frag < min_frag) return;
                }
                if (tight_maxs[check_dim]) {
                    int max_frag = get_byte(box.max_coords[check_dim], chunk_in_dim);
                    if (rep_frag > max_frag) return;
                }
            }
        }

        int depth = node->depth;
        int split_dim = depth % dim_;
        int chunk_in_dim = depth / dim_;

        if (chunk_in_dim >= 8) { return; }

        int start_frag = tight_mins[split_dim] ? get_byte(box.min_coords[split_dim], chunk_in_dim) : 0;
        int end_frag   = tight_maxs[split_dim] ? get_byte(box.max_coords[split_dim], chunk_in_dim) : 255;

        for (int i = start_frag; i <= end_frag; ++i) {
            std::atomic<uint32_t>* child_slot = get_child_slot<false>(node, i);
            if (!child_slot) continue;
            uint32_t child_idx = child_slot->load(std::memory_order_acquire);
            if (!child_idx) continue;

            bool next_tight_mins[MAX_DIMS];
            bool next_tight_maxs[MAX_DIMS];
            memcpy(next_tight_mins, tight_mins, dim_ * sizeof(bool));
            memcpy(next_tight_maxs, tight_maxs, dim_ * sizeof(bool));

            next_tight_mins[split_dim] = tight_mins[split_dim] && (i == start_frag);
            next_tight_maxs[split_dim] = tight_maxs[split_dim] && (i == end_frag);

            box_query_recursive(child_idx, depth + 1, box, next_tight_mins, next_tight_maxs, results);
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        const size_t NUM_KEYS = 1000000;
        const int DIMS = 8;
        const size_t FANOUT = 256;
        const size_t SLOTS_PER_PAGE = 16;

        using TreeType = KeyValueRadixTree<FANOUT, SLOTS_PER_PAGE>;
        using NodeType = RadixNode<FANOUT, SLOTS_PER_PAGE>;
        using PageType = NodePage<SLOTS_PER_PAGE>;

        // --- Prepare Keys ---
        std::cout << "--- CYCLOPS Benchmark Gauntlet ---" << std::endl;
        std::cout << "--- Dims=" << DIMS << ", Keys=" << NUM_KEYS << ", FANOUT=" << FANOUT << ", SLOTS_PER_PAGE=" << SLOTS_PER_PAGE << " ---" << std::endl;
        std::cout << "\nPreparing " << NUM_KEYS << " random keys..." << std::endl;
        auto keys = std::make_unique<std::vector<Key>>(NUM_KEYS);
        auto miss_keys = std::make_unique<std::vector<Key>>(NUM_KEYS);
        std::mt19937_64 rng(12345);
        for(size_t i = 0; i < NUM_KEYS; ++i) {
            for (int d = 0; d < DIMS; ++d) {
                (*keys)[i].coords[d] = rng();
                (*miss_keys)[i].coords[d] = rng();
            }
        }

        // --- Unordered Map Benchmark (Baseline) ---
        std::cout << "\n--- std::unordered_map Benchmark (Baseline) ---" << std::endl;
        std::unordered_map<std::string, uint64_t> umap;
        auto start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_KEYS; ++i) {
            std::string key_str(reinterpret_cast<const char*>((*keys)[i].coords), DIMS * sizeof(uint64_t));
            umap[key_str] = i;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "unordered_map Insert: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_KEYS; ++i) {
            std::string key_str(reinterpret_cast<const char*>((*keys)[i].coords), DIMS * sizeof(uint64_t));
            volatile auto it = umap.find(key_str);
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "unordered_map Get (Hit): " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << std::endl;

        // --- Memory and Scalability Benchmark ---
        std::cout << "\n--- Memory & Insertion Scalability ---" << std::endl;
        std::vector<int> thread_counts = {1, 2, 4, 8};
        if (argc > 1) {
            thread_counts.clear();
            for(int i = 1; i < argc; ++i) thread_counts.push_back(std::stoi(argv[i]));
        }

        for (int num_threads : thread_counts) {
            ValueStore vs;
            NodeManager nm(sizeof(NodeType));
            RecordManager rm(&vs, DIMS);
            PageManager pm(sizeof(PageType));
            std::atomic<uint32_t> root_ptr(0);
            TreeType temp_tree(nm, rm, pm, root_ptr, DIMS);

            auto start_time_insert = std::chrono::high_resolution_clock::now();
            std::vector<std::thread> threads;
            for (int i = 0; i < num_threads; ++i) {
                threads.emplace_back([&, i]() {
                    size_t start = i * (NUM_KEYS / num_threads);
                    size_t end = (i == num_threads - 1) ? NUM_KEYS : start + (num_threads / num_threads);
                    for (size_t j = start; j < end; ++j) temp_tree.insert((*keys)[j].coords, j);
                });
            }
            for (auto& t : threads) t.join();
            auto end_time_insert = std::chrono::high_resolution_clock::now();
            auto duration_insert = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_insert - start_time_insert);
            double throughput = (duration_insert.count() > 0) ? (double)NUM_KEYS / duration_insert.count() * 1000.0 : 0;
            std::cout << "Scalability_Insert " << num_threads << " Threads: " << std::fixed << std::setprecision(0) << throughput << " ops/sec" << std::endl;
        }

        // --- Build Main Tree for Querying ---
        std::cout << "\nBuilding main tree for query tests with 8 threads..." << std::endl;
        ValueStore vs_main;
        NodeManager nm_main(sizeof(NodeType));
        RecordManager rm_main(&vs_main, DIMS);
        PageManager pm_main(sizeof(PageType));
        std::atomic<uint32_t> root_ptr_main(0);
        TreeType tree(nm_main, rm_main, pm_main, root_ptr_main, DIMS);

        auto build_start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> build_threads;
        int build_thread_count = 8;
        for (int i = 0; i < build_thread_count; ++i) {
           build_threads.emplace_back([&, i]() {
               size_t start = i * (NUM_KEYS / build_thread_count);
               size_t end = (i == build_thread_count - 1) ? NUM_KEYS : start + (NUM_KEYS / build_thread_count);
               for (size_t j = start; j < end; ++j) tree.insert((*keys)[j].coords, j);
           });
        }
        for(auto& t : build_threads) t.join();
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
        std::cout << "Tree built in " << build_duration.count() << " ms." << std::endl;
        size_t total_mem = nm_main.get_mem_usage() + rm_main.get_mem_usage() + pm_main.get_mem_usage();
        std::cout << "Memory_Per_Key: " << std::fixed << std::setprecision(2) << (double)total_mem / NUM_KEYS << " bytes/key" << std::endl;

        // --- Query Performance Benchmarks ---
        std::cout << "\n--- Point Performance ---" << std::endl;
        size_t found_count = 0;
        start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_KEYS; ++i) {
            if(tree.get((*keys)[i].coords)) found_count++;
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "Point_Get_Hit: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << std::endl;
        if (found_count != NUM_KEYS) {
             std::cout << "  Verification FAILED: Found " << found_count << "/" << NUM_KEYS << " keys." << std::endl;
        } else {
             std::cout << "  Verification SUCCESS: Found all " << NUM_KEYS << " keys." << std::endl;
        }

        start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_KEYS; ++i) { tree.get((*miss_keys)[i].coords); }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "Point_Get_Miss: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << std::endl;


        std::cout << "\n--- Spatial Query Performance ---" << std::endl;
        QueryBox box;
        Key query_center_key = (*keys)[NUM_KEYS / 4]; // Pick a key guaranteed to be in the set
        const uint64_t WIDE_BOX_DELTA = std::numeric_limits<uint64_t>::max() / 8; // 25% of total space
        const uint64_t NARROW_BOX_DELTA = std::numeric_limits<uint64_t>::max() / 2000; // 0.1% of total space

        // Wide Box (25% of space)
        for(int d=0; d<DIMS; ++d) {
            uint64_t c = query_center_key.coords[d];
            box.min_coords[d] = (c > WIDE_BOX_DELTA) ? c - WIDE_BOX_DELTA : 0;
            box.max_coords[d] = (c < std::numeric_limits<uint64_t>::max() - WIDE_BOX_DELTA) ? c + WIDE_BOX_DELTA : std::numeric_limits<uint64_t>::max();
        }
        start_time = std::chrono::high_resolution_clock::now();
        auto wide_results = tree.box_query(box);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Wide_Box_Query (25%): " << duration.count() << " us (" << wide_results.size() << " results)" << std::endl;

        // Narrow Box (0.1% of space)
        for(int d=0; d<DIMS; ++d) {
            uint64_t c = query_center_key.coords[d];
            box.min_coords[d] = (c > NARROW_BOX_DELTA) ? c - NARROW_BOX_DELTA : 0;
            box.max_coords[d] = (c < std::numeric_limits<uint64_t>::max() - NARROW_BOX_DELTA) ? c + NARROW_BOX_DELTA : std::numeric_limits<uint64_t>::max();
        }
        start_time = std::chrono::high_resolution_clock::now();
        auto narrow_results = tree.box_query(box);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Narrow_Box_Query (0.1%): " << duration.count() << " us (" << narrow_results.size() << " results)" << std::endl;

        // Elongated Box
        for(int d=0; d<DIMS; ++d) {
            uint64_t c = query_center_key.coords[d];
            if (d % 2 == 0) { // Wide (50%)
                 const uint64_t ELONGATED_WIDE_DELTA = std::numeric_limits<uint64_t>::max() / 4;
                 box.min_coords[d] = (c > ELONGATED_WIDE_DELTA) ? c - ELONGATED_WIDE_DELTA : 0;
                 box.max_coords[d] = (c < std::numeric_limits<uint64_t>::max() - ELONGATED_WIDE_DELTA) ? c + ELONGATED_WIDE_DELTA : std::numeric_limits<uint64_t>::max();
            } else { // Narrow (0.1%)
                box.min_coords[d] = (c > NARROW_BOX_DELTA) ? c - NARROW_BOX_DELTA : 0;
                box.max_coords[d] = (c < std::numeric_limits<uint64_t>::max() - NARROW_BOX_DELTA) ? c + NARROW_BOX_DELTA : std::numeric_limits<uint64_t>::max();
            }
        }
        start_time = std::chrono::high_resolution_clock::now();
        auto elongated_results = tree.box_query(box);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Elongated_Box_Query: " << duration.count() << " us (" << elongated_results.size() << " results)" << std::endl;


        std::cout << "\n--- Lexicographical Emulation & Hybrid Performance ---" << std::endl;
        // Lex_Scan_Emulation - Create a valid box around a range of lexicographically sorted keys
        std::sort(keys->begin(), keys->end(), [](const Key& a, const Key& b){
            return std::memcmp(a.coords, b.coords, sizeof(Key)) < 0;
        });
        size_t scan_start_idx = NUM_KEYS / 2;
        size_t scan_count = 1000;
        // To create a valid box, we must find the min/max for each dimension within the scan range
        for(int d=0; d<DIMS; ++d) {
            box.min_coords[d] = (*keys)[scan_start_idx].coords[d];
            box.max_coords[d] = (*keys)[scan_start_idx].coords[d];
        }
        for(size_t i = scan_start_idx + 1; i < scan_start_idx + scan_count; ++i) {
            for(int d=0; d<DIMS; ++d) {
                if ((*keys)[i].coords[d] < box.min_coords[d]) box.min_coords[d] = (*keys)[i].coords[d];
                if ((*keys)[i].coords[d] > box.max_coords[d]) box.max_coords[d] = (*keys)[i].coords[d];
            }
        }
        start_time = std::chrono::high_resolution_clock::now();
        auto lex_results = tree.box_query(box);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Lex_Scan_Emulation: " << duration.count() << " us (" << lex_results.size() << " results)" << std::endl;

        // Prefix_Plus_Narrow_Box
        Key center_key = (*keys)[NUM_KEYS/2];
        for(int d=0; d<DIMS; ++d) {
            if (d < 4) { // Exact match prefix
                box.min_coords[d] = center_key.coords[d];
                box.max_coords[d] = center_key.coords[d];
            } else { // Narrow box
                uint64_t c = center_key.coords[d];
                box.min_coords[d] = (c > NARROW_BOX_DELTA) ? c - NARROW_BOX_DELTA : 0;
                box.max_coords[d] = (c < std::numeric_limits<uint64_t>::max() - NARROW_BOX_DELTA) ? c + NARROW_BOX_DELTA : std::numeric_limits<uint64_t>::max();
            }
        }
        start_time = std::chrono::high_resolution_clock::now();
        auto prefix_narrow_results = tree.box_query(box);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Prefix_Plus_Narrow_Box: " << duration.count() << " us (" << prefix_narrow_results.size() << " results)" << std::endl;

        // Wide_Box_Plus_Suffix
        for(int d=0; d<DIMS; ++d) {
            if (d < 4) { // Wide box
                uint64_t c = center_key.coords[d];
                box.min_coords[d] = (c > WIDE_BOX_DELTA) ? c - WIDE_BOX_DELTA : 0;
                box.max_coords[d] = (c < std::numeric_limits<uint64_t>::max() - WIDE_BOX_DELTA) ? c + WIDE_BOX_DELTA : std::numeric_limits<uint64_t>::max();
            } else { // Exact match suffix
                box.min_coords[d] = center_key.coords[d];
                box.max_coords[d] = center_key.coords[d];
            }
        }
        start_time = std::chrono::high_resolution_clock::now();
        auto wide_suffix_results = tree.box_query(box);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Wide_Box_Plus_Suffix: " << duration.count() << " us (" << wide_suffix_results.size() << " results)" << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
