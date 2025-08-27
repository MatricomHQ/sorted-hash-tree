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
    uint32_t test_nibble_idx;
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

    RadixNode() : test_nibble_idx(0), representative_record_idx(0) {
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

template<size_t FANOUT, size_t SLOTS_PER_PAGE = 16>
class KeyValueRadixTree {
    using NodeType = RadixNode<FANOUT, SLOTS_PER_PAGE>;

    NodeManager& node_manager_;
    RecordManager& record_manager_;
    PageManager& page_manager_;
    std::atomic<uint32_t>& root_ptr_;
    const uint32_t dim_;

    static inline int get_key_fragment(const uint64_t* coords, int frag_idx, int d) {
        if constexpr (FANOUT == 16) { // Nibble-based
            int max_frags = d * 16;
            if (frag_idx < 0 || frag_idx >= max_frags) return 0;
            int dim_idx = frag_idx / 16;
            int nibble_in_dim = frag_idx % 16;
            return (coords[dim_idx] >> (60 - (nibble_in_dim * 4))) & 0x0F;
        } else if constexpr (FANOUT == 256) { // Byte-based
            int max_frags = d * 8;
            if (frag_idx < 0 || frag_idx >= max_frags) return 0;
            int dim_idx = frag_idx / 8;
            int byte_in_dim = frag_idx % 8;
            return (coords[dim_idx] >> (56 - (byte_in_dim * 8))) & 0xFF;
        }
        return 0; // Should not happen with supported fanouts
    }

    static int find_first_differing_fragment(const uint64_t* k1, const uint64_t* k2, int d, int start_frag = 0) {
        int max_frags = d * (FANOUT == 16 ? 16 : 8);
        for (int i = start_frag; i < max_frags; ++i) {
            if (get_key_fragment(k1, i, d) != get_key_fragment(k2, i, d)) return i;
        }
        return -1;
    }

public:
    KeyValueRadixTree(NodeManager& nm, RecordManager& rm, PageManager& pm, std::atomic<uint32_t>& root, uint32_t d)
        : node_manager_(nm), record_manager_(rm), page_manager_(pm), root_ptr_(root), dim_(d) {}

    void insert(const uint64_t* coords, uint64_t value) {
    restart:
        std::atomic<uint32_t>* parent_slot = &root_ptr_;
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);

        while(TaggedIndex::is_node(current_idx)) {
            NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(current_idx));
            Record* rep_rec = record_manager_.get_record(node->representative_record_idx);

            int diff_idx = find_first_differing_fragment(coords, rep_rec->coords, dim_);

            if (diff_idx != -1 && (uint32_t)diff_idx < node->test_nibble_idx) {
                uint32_t new_node_idx = node_manager_.allocate_node<NodeType>();
                NodeType* new_node = node_manager_.get_node<NodeType>(new_node_idx);
                uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
                new_node->test_nibble_idx = diff_idx;
                new_node->representative_record_idx = new_rec_idx;

                int new_key_frag = get_key_fragment(coords, diff_idx, dim_);
                int existing_key_frag = get_key_fragment(rep_rec->coords, diff_idx, dim_);

                std::atomic<uint32_t>* new_child_slot = get_child_slot<true>(new_node, new_key_frag);
                new_child_slot->store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);

                std::atomic<uint32_t>* existing_child_slot = get_child_slot<true>(new_node, existing_key_frag);
                existing_child_slot->store(current_idx, std::memory_order_relaxed);

                if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart;
            }
            int frag = get_key_fragment(coords, node->test_nibble_idx, dim_);
            parent_slot = get_child_slot<true>(node, frag);
            if (!parent_slot) goto restart; // Contention on page creation
            current_idx = parent_slot->load(std::memory_order_acquire);
        }

        if (current_idx == 0) {
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart;
        }

        if (TaggedIndex::is_leaf(current_idx)) {
            Record* existing_rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));
            int diff_idx = find_first_differing_fragment(coords, existing_rec->coords, dim_);
            if (diff_idx == -1) return;

            uint32_t new_node_idx = node_manager_.allocate_node<NodeType>();
            NodeType* new_node = node_manager_.get_node<NodeType>(new_node_idx);
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            new_node->test_nibble_idx = diff_idx;
            new_node->representative_record_idx = new_rec_idx;

            int new_key_frag = get_key_fragment(coords, diff_idx, dim_);
            int existing_key_frag = get_key_fragment(existing_rec->coords, diff_idx, dim_);

            std::atomic<uint32_t>* new_child_slot = get_child_slot<true>(new_node, new_key_frag);
            new_child_slot->store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);

            std::atomic<uint32_t>* existing_child_slot = get_child_slot<true>(new_node, existing_key_frag);
            existing_child_slot->store(current_idx, std::memory_order_relaxed);

            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart;
        }
    }

    bool get(const uint64_t* coords) {
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);
        while (TaggedIndex::is_node(current_idx)) {
            NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(current_idx));
            int frag = get_key_fragment(coords, node->test_nibble_idx, dim_);

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

    std::vector<Record*> scan(const uint64_t* start_key, const uint64_t* end_key) {
        std::vector<Record*> results;
        scan_recursive(root_ptr_.load(std::memory_order_acquire), start_key, end_key, true, true, results);
        return results;
    }

private:
    template<bool create_if_missing>
    std::atomic<uint32_t>* get_child_slot(NodeType* node, int frag) {
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
                        // Another thread beat us, leak our page for now.
                        page = node->children_or_pages[page_idx].load(std::memory_order_acquire);
                    }
                } else {
                    return nullptr; // Page does not exist, and we are not creating it.
                }
            }
            return &page->children[slot_idx_in_page];
        } else {
            return &node->children_or_pages[frag];
        }
    }

    void scan_recursive(uint32_t node_idx, const uint64_t* start_key, const uint64_t* end_key, bool tight_lower, bool tight_upper, std::vector<Record*>& results) {
        if (!node_idx) return;

        if (TaggedIndex::is_leaf(node_idx)) {
            Record* rec = record_manager_.get_record(TaggedIndex::get_index(node_idx));
            if (rec) {
                if ((!tight_lower || memcmp(rec->coords, start_key, dim_ * sizeof(uint64_t)) >= 0) &&
                    (!tight_upper || memcmp(rec->coords, end_key, dim_ * sizeof(uint64_t)) <= 0)) {
                    results.push_back(rec);
                }
            }
            return;
        }

        NodeType* node = node_manager_.get_node<NodeType>(TaggedIndex::get_index(node_idx));
        int start_frag = tight_lower ? get_key_fragment(start_key, node->test_nibble_idx, dim_) : 0;
        int end_frag = tight_upper ? get_key_fragment(end_key, node->test_nibble_idx, dim_) : FANOUT - 1;

        for (int i = start_frag; i <= end_frag; ++i) {
            std::atomic<uint32_t>* child_slot = get_child_slot<false>(node, i);
            if (!child_slot) continue;

            uint32_t child_idx = child_slot->load(std::memory_order_acquire);
            if (child_idx) {
                bool next_tight_lower = tight_lower && (i == start_frag);
                bool next_tight_upper = tight_upper && (i == end_frag);
                scan_recursive(child_idx, start_key, end_key, next_tight_lower, next_tight_upper, results);
            }
        }
    }
};

const int MAX_DIMS = 8;
struct Key { uint64_t coords[MAX_DIMS]; };

template<size_t FANOUT, size_t SLOTS_PER_PAGE>
void run_benchmark(size_t num_keys, int num_threads, int dimensionality, const std::string& key_type) {
    std::cout << "\n--- Benchmark: " << dimensionality << "D Key-Value Radix Tree (" << (dimensionality*8) << " bytes), FANOUT=" << FANOUT << ", SLOTS_PER_PAGE=" << SLOTS_PER_PAGE << " ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    using TreeType = KeyValueRadixTree<FANOUT, SLOTS_PER_PAGE>;
    using NodeType = RadixNode<FANOUT, SLOTS_PER_PAGE>;
    using PageType = NodePage<SLOTS_PER_PAGE>;

    ValueStore vs;
    NodeManager nm(sizeof(NodeType));
    RecordManager rm(&vs, dimensionality);
    PageManager pm(sizeof(PageType));
    std::atomic<uint32_t> root_ptr(0);
    TreeType tree(nm, rm, pm, root_ptr, dimensionality);

    std::cout << "Preparing keys..." << std::endl;
    std::vector<Key> keys(num_keys);
    std::vector<Key> miss_keys(num_keys);

    if (key_type == "Random") {
        std::mt19937_64 rng(12345);
        for(size_t i = 0; i < num_keys; ++i) {
            for (int d = 0; d < dimensionality; ++d) {
                keys[i].coords[d] = rng();
                miss_keys[i].coords[d] = rng();
            }
        }
    } else { // Sequential
        for(size_t i = 0; i < num_keys; ++i) {
            for (int d = 0; d < dimensionality; ++d) {
                keys[i].coords[d] = i;
                miss_keys[i].coords[d] = i + num_keys;
            }
        }
    }

    std::cout << "\n--- INSERTION ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t start = i * (num_keys / num_threads);
            size_t end = (i == num_threads - 1) ? num_keys : start + (num_keys / num_threads);
            for (size_t j = start; j < end; ++j) tree.insert(keys[j].coords, j);
        });
    }
    for (auto& t : threads) t.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[KeyValueRadixTree] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/insert" << std::endl;

    std::cout << "\n--- HIT LATENCY (LOOKUP) ---" << std::endl;
    size_t found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) {
        if (tree.get(keys[i].coords)) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[KeyValueRadixTree] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;
    if (found_count == num_keys) {
        std::cout << "  Verification: SUCCESS" << std::endl;
    } else {
        std::cout << "  Verification: FAILED (Found " << found_count << "/" << num_keys << ")" << std::endl;
    }

    std::cout << "\n--- MISS LATENCY (LOOKUP) ---" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) { tree.get(miss_keys[i].coords); }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[KeyValueRadixTree] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;

    std::cout << "\n--- MEMORY USAGE ---" << std::endl;
    size_t total_mem = nm.get_mem_usage() + rm.get_mem_usage();
    std::cout << "[KeyValueRadixTree] Total (Actual): " << std::fixed << std::setprecision(2) << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[KeyValueRadixTree] Per Key (Actual): " << std::fixed << std::setprecision(2) << (double)total_mem / num_keys << " bytes/key" << std::endl;

    // --- Unordered Map Benchmark ---
    if (FANOUT == 16) { // Only run this once to avoid redundant output
        std::cout << "\n--- std::unordered_map Benchmark (string key) ---" << std::endl;
        std::unordered_map<std::string, uint64_t> umap;

        // Insertion
        start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            std::string key_str(reinterpret_cast<const char*>(keys[i].coords), dimensionality * sizeof(uint64_t));
            umap[key_str] = i;
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "[unordered_map] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/insert" << std::endl;

        // Hit Latency
        start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            std::string key_str(reinterpret_cast<const char*>(keys[i].coords), dimensionality * sizeof(uint64_t));
            volatile auto it = umap.find(key_str);
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "[unordered_map] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;

        // Miss Latency
        start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            std::string key_str(reinterpret_cast<const char*>(miss_keys[i].coords), dimensionality * sizeof(uint64_t));
            volatile auto it = umap.find(key_str);
        }
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        std::cout << "[unordered_map] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;
    }


    if (key_type == "Sequential" && num_keys > 1000) {
        std::cout << "\n--- RANGE SCAN ---" << std::endl;
        const size_t scan_size = 1000;
        size_t start_idx = num_keys / 4;
        const uint64_t* start_key = keys[start_idx].coords;
        const uint64_t* end_key = keys[start_idx + scan_size - 1].coords;

        start_time = std::chrono::high_resolution_clock::now();
        std::vector<Record*> scan_results = tree.scan(start_key, end_key);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

        if (!scan_results.empty()) {
            std::cout << "[KeyValueRadixTree] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / scan_results.size() << " ns/key (" << scan_results.size() << " keys)" << std::endl;
        } else {
             std::cout << "[KeyValueRadixTree] Scan found no results." << std::endl;
        }
        if (scan_results.size() != scan_size) {
            std::cerr << "  SCAN VERIFICATION FAILED! Expected " << scan_size << ", got " << scan_results.size() << std::endl;
        }
    }
}

template<size_t FANOUT, size_t SLOTS_PER_PAGE>
void run_benchmark(size_t num_keys, int num_threads, int dimensionality, const std::string& key_type);

int main(int argc, char* argv[]) {
    try {
        int num_threads = std::thread::hardware_concurrency();
        if (argc > 1) {
            num_threads = std::stoi(argv[1]);
        }
        std::cout << "--- Running with " << num_threads << " threads ---" << std::endl;

        const size_t LARGE_KEY_COUNT = 1000000;

        std::vector<int> dims_to_test = {1, 3, 8};
        std::vector<std::string> key_types_to_test = {"Random", "Sequential"};
        std::vector<size_t> slots_to_test = {4, 8, 16};

        for (int dims : dims_to_test) {
            for (const auto& key_type : key_types_to_test) {
                // Run baseline FANOUT=16
                run_benchmark<16, 16>(LARGE_KEY_COUNT, num_threads, dims, key_type);

                // Test different slot sizes for FANOUT=256
                for (size_t slots : slots_to_test) {
                    if (slots == 4) run_benchmark<256, 4>(LARGE_KEY_COUNT, num_threads, dims, key_type);
                    if (slots == 8) run_benchmark<256, 8>(LARGE_KEY_COUNT, num_threads, dims, key_type);
                    if (slots == 16) run_benchmark<256, 16>(LARGE_KEY_COUNT, num_threads, dims, key_type);
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
