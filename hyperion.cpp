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
#include <optional>
#include <tuple>
#include <set>

#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// --- Type Aliases ---
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;

// --- Foundational Data Structures ---

struct Record {
    u32 value_len;
    u32 dim;
    u64 value_or_offset;
    u64 coords[];

    static constexpr u32 INLINE_FLAG = 1U << 31;
    bool is_value_inlined() const { return (dim & INLINE_FLAG) != 0; }
    void set_inlined(bool is_inlined) { dim = (dim & ~INLINE_FLAG) | (is_inlined ? INLINE_FLAG : 0); }
    static size_t get_size(u32 d) { return sizeof(Record) + sizeof(u64) * d; }
};

namespace TaggedIndex {
    static constexpr u32 NODE_TAG = 0b00;
    static constexpr u32 LEAF_TAG = 0b10;
    static constexpr u32 BUCKET_TAG = 0b11;
    static constexpr u32 TAG_MASK = 0b11 << 30;
    static constexpr u32 INDEX_MASK = ~TAG_MASK;
    inline u32 get_tag(u32 idx) { return (idx >> 30); }
    inline u32 get_index(u32 idx) { return idx & INDEX_MASK; }
    inline bool is_leaf(u32 idx) { return get_tag(idx) == LEAF_TAG; }
    inline bool is_bucket(u32 idx) { return get_tag(idx) == BUCKET_TAG; }
    inline bool is_node(u32 idx) { return get_tag(idx) == NODE_TAG && idx != 0; }
    inline u32 make_leaf_idx(u32 rec_idx) { return (rec_idx & INDEX_MASK) | (LEAF_TAG << 30); }
    inline u32 make_bucket_idx(u32 bucket_idx) { return (bucket_idx & INDEX_MASK) | (BUCKET_TAG << 30); }
    inline u32 make_node_idx(u32 node_idx) { return (node_idx & INDEX_MASK); }
};

template<size_t SLOTS> struct NodePage { std::atomic<u32> children[SLOTS]; NodePage() { for (size_t i = 0; i < SLOTS; ++i) children[i].store(0, std::memory_order_relaxed); } };

template<size_t FANOUT, size_t SLOTS_PER_PAGE = 16, size_t PAGED_THRESHOLD = 16>
struct RadixNode {
    u32 test_nibble_idx = 0;
    u32 representative_record_idx = 0;
    static constexpr bool IS_PAGED = FANOUT >= PAGED_THRESHOLD;
    using PageType = NodePage<SLOTS_PER_PAGE>;
    static constexpr size_t PAGES_PER_NODE = IS_PAGED ? (FANOUT + SLOTS_PER_PAGE - 1) / SLOTS_PER_PAGE : 0;
    std::conditional_t<IS_PAGED, std::atomic<PageType*>[PAGES_PER_NODE], std::atomic<u32>[FANOUT]> children_or_pages;
    RadixNode() {
        if constexpr (IS_PAGED) for (size_t i = 0; i < PAGES_PER_NODE; ++i) children_or_pages[i].store(nullptr, std::memory_order_relaxed);
        else for (size_t i = 0; i < FANOUT; ++i) children_or_pages[i].store(0, std::memory_order_relaxed);
    }
};

// --- Memory Management ---
class NodeManager {
    std::atomic<u32> next_node_idx_{1};
    uint8_t* node_pool_;
    const size_t node_size_;
    static constexpr u32 MAX_NODES = 8 * 1024 * 1024;
public:
    NodeManager(size_t node_size) : node_size_(node_size) {
        node_pool_ = (uint8_t*)mmap(nullptr, (size_t)MAX_NODES * node_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (node_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for node pool");
    }
    ~NodeManager() { munmap(node_pool_, (size_t)MAX_NODES * node_size_); }
    template<typename TNode> TNode* get_node(u32 idx) { return reinterpret_cast<TNode*>(node_pool_ + (size_t)idx * node_size_); }
    template<typename TNode> u32 allocate_node() {
        u32 idx = next_node_idx_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_NODES) throw std::runtime_error("Node pool exhausted");
        new (get_node<TNode>(idx)) TNode(); return idx;
    }
    size_t get_mem_usage() const { return (size_t)next_node_idx_.load() * node_size_; }
};

class PageManager {
    std::atomic<u32> next_page_idx_{1};
    uint8_t* page_pool_;
    const size_t page_size_;
    static constexpr u32 MAX_PAGES = 32 * 1024 * 1024;
public:
    PageManager(size_t page_size) : page_size_(page_size) {
        page_pool_ = (uint8_t*)mmap(nullptr, (size_t)MAX_PAGES * page_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (page_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for page pool");
    }
    ~PageManager() { munmap(page_pool_, (size_t)MAX_PAGES * page_size_); }
    template<typename TPage> TPage* get_page(u32 idx) { return reinterpret_cast<TPage*>(page_pool_ + (size_t)idx * page_size_); }
    template<typename TPage> TPage* allocate_page_and_get_ptr() {
        u32 idx = next_page_idx_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_PAGES) throw std::runtime_error("Page pool exhausted");
        TPage* page = get_page<TPage>(idx); new (page) TPage(); return page;
    }
    size_t get_mem_usage() const { return (size_t)next_page_idx_.load() * page_size_; }
};

template<size_t BUCKET_SIZE> struct LeafBucket { std::atomic<uint8_t> count{0}; u32 record_indices[BUCKET_SIZE]{0}; bool is_full() const { return count.load(std::memory_order_relaxed) >= BUCKET_SIZE; }
    bool append(u32 record_idx) {
        uint8_t c = count.load(std::memory_order_acquire);
        while (c < BUCKET_SIZE) if (count.compare_exchange_weak(c, c + 1, std::memory_order_release, std::memory_order_relaxed)) { record_indices[c] = record_idx; return true; }
        return false;
    }
};
class LeafBucketManager {
    std::atomic<u32> next_bucket_idx_{1};
    uint8_t* bucket_pool_;
    const size_t bucket_size_;
    static constexpr u32 MAX_BUCKETS = 8 * 1024 * 1024;
public:
    LeafBucketManager(size_t bucket_size) : bucket_size_(bucket_size) {
        bucket_pool_ = (uint8_t*)mmap(nullptr, (size_t)MAX_BUCKETS * bucket_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (bucket_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for bucket pool");
    }
    ~LeafBucketManager() { munmap(bucket_pool_, (size_t)MAX_BUCKETS * bucket_size_); }
    template<typename TBucket> TBucket* get_bucket(u32 idx) { return reinterpret_cast<TBucket*>(bucket_pool_ + (size_t)idx * bucket_size_); }
    template<typename TBucket> u32 allocate_bucket() {
        u32 idx = next_bucket_idx_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_BUCKETS) throw std::runtime_error("LeafBucket pool exhausted");
        new (get_bucket<TBucket>(idx)) TBucket(); return idx;
    }
    size_t get_mem_usage() const { return (size_t)next_bucket_idx_.load() * bucket_size_; }
};

class RecordManager {
    std::atomic<u32> next_record_idx_{1};
    uint8_t* record_pool_;
    const size_t record_size_with_coords_;
    static constexpr u32 MAX_RECORDS = 32 * 1024 * 1024;
public:
    RecordManager(u32 d) : record_size_with_coords_(Record::get_size(d)) {
        record_pool_ = (uint8_t*)mmap(nullptr, (size_t)MAX_RECORDS * record_size_with_coords_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (record_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for record pool");
    }
    ~RecordManager() { munmap(record_pool_, (size_t)MAX_RECORDS * record_size_with_coords_); }
    Record* get_record(u32 idx) { return reinterpret_cast<Record*>(record_pool_ + (size_t)idx * record_size_with_coords_); }
    u32 allocate_record(const u64* coords, u32 dim, u64 value, u32 value_len) {
        u32 idx = next_record_idx_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_RECORDS) throw std::runtime_error("Record pool exhausted");
        Record* rec = get_record(idx);
        rec->dim = dim; rec->value_len = value_len;
        rec->value_or_offset = value;
        rec->set_inlined(true);
        memcpy(rec->coords, coords, sizeof(u64) * dim);
        return idx;
    }
    size_t get_mem_usage() const { return (size_t)next_record_idx_.load() * record_size_with_coords_; }
};

struct MemoryContext {
    std::unique_ptr<NodeManager> nm;
    std::unique_ptr<RecordManager> rm;
    std::unique_ptr<PageManager> pm;
    std::unique_ptr<LeafBucketManager> bm;
    MemoryContext(size_t node_size, size_t page_size, size_t bucket_size, u32 dimensionality) {
        nm = std::make_unique<NodeManager>(node_size);
        rm = std::make_unique<RecordManager>(dimensionality);
        pm = std::make_unique<PageManager>(page_size);
        bm = std::make_unique<LeafBucketManager>(bucket_size);
    }
    size_t get_mem_usage() const {
        return nm->get_mem_usage() + rm->get_mem_usage() + bm->get_mem_usage() + pm->get_mem_usage();
    }
};

// --- Radix Tree Core ---
template<size_t FANOUT, size_t SLOTS_PER_PAGE, size_t LEAF_BUCKET_SIZE, typename ValueType>
class KeyValueRadixTree {
    using NodeType = RadixNode<FANOUT, SLOTS_PER_PAGE>;
    using BucketType = LeafBucket<LEAF_BUCKET_SIZE>;
    NodeManager& node_manager_; RecordManager& record_manager_; PageManager& page_manager_; LeafBucketManager& bucket_manager_;
    std::atomic<u32> root_ptr_{0};
    const u32 dim_;

    static inline int get_key_fragment(const u64* coords, int frag_idx, int d) {
        if constexpr (FANOUT == 16) { int max_frags = d * 16; if (frag_idx < 0 || frag_idx >= max_frags) return 0; return (coords[frag_idx / 16] >> (60 - ((frag_idx % 16) * 4))) & 0x0F; }
        else if constexpr (FANOUT == 256) { int max_frags = d * 8; if (frag_idx < 0 || frag_idx >= max_frags) return 0; return (coords[frag_idx / 8] >> (56 - ((frag_idx % 8) * 8))) & 0xFF; }
        return 0;
    }
    static int find_first_differing_fragment(const u64* k1, const u64* k2, int d, int start_frag = 0) {
        int max_frags = d * (FANOUT == 16 ? 16 : 8);
        for (int i = start_frag; i < max_frags; ++i) if (get_key_fragment(k1, i, d) != get_key_fragment(k2, i, d)) return i;
        return -1;
    }
    static int find_first_differing_fragment_multi(const std::vector<const u64*>& keys, int d) {
        if (keys.size() < 2) return -1;
        int max_frags = d * (FANOUT == 16 ? 16 : 8);
        for (int frag_idx = 0; frag_idx < max_frags; ++frag_idx) {
            int first_frag = get_key_fragment(keys[0], frag_idx, d);
            for (size_t i = 1; i < keys.size(); ++i) if (get_key_fragment(keys[i], frag_idx, d) != first_frag) return frag_idx;
        }
        return -1;
    }
    void insert_into_new_node(NodeType* node, u32 record_idx, const u64* key, int diff_idx, u32 existing_tagged_idx) {
        const u64* existing_coords = nullptr;
        if (TaggedIndex::is_leaf(existing_tagged_idx)) existing_coords = record_manager_.get_record(TaggedIndex::get_index(existing_tagged_idx))->coords;
        else if (TaggedIndex::is_node(existing_tagged_idx)) existing_coords = record_manager_.get_record(node_manager_.template get_node<NodeType>(TaggedIndex::get_index(existing_tagged_idx))->representative_record_idx)->coords;
        else return;
        get_child_slot<true>(node, get_key_fragment(key, diff_idx, dim_))->store(TaggedIndex::make_leaf_idx(record_idx), std::memory_order_relaxed);
        get_child_slot<true>(node, get_key_fragment(existing_coords, diff_idx, dim_))->store(existing_tagged_idx, std::memory_order_relaxed);
    }
public:
    KeyValueRadixTree(MemoryContext& ctx, u32 d) : node_manager_(*ctx.nm), record_manager_(*ctx.rm), page_manager_(*ctx.pm), bucket_manager_(*ctx.bm), dim_(d) {}

    void insert(const u64* coords, ValueType value) {
        static_assert(sizeof(ValueType) <= sizeof(u64)); u64 val_u64 = 0; memcpy(&val_u64, &value, sizeof(ValueType));
        insert_recursive(coords, val_u64, &root_ptr_);
    }
    void insert_recursive(const u64* coords, u64 value, std::atomic<u32>* parent_slot) {
    restart:
        u32 current_idx = parent_slot->load(std::memory_order_acquire);
        while(TaggedIndex::is_node(current_idx)) {
            NodeType* node = node_manager_.template get_node<NodeType>(TaggedIndex::get_index(current_idx));
            Record* rep_rec = record_manager_.get_record(node->representative_record_idx);
            int diff_idx = find_first_differing_fragment(coords, rep_rec->coords, dim_);
            if (diff_idx != -1 && (u32)diff_idx < node->test_nibble_idx) {
                u32 new_rec_idx = record_manager_.allocate_record(coords, dim_, value, sizeof(ValueType));
                u32 new_node_idx = node_manager_.template allocate_node<NodeType>();
                NodeType* new_node = node_manager_.template get_node<NodeType>(new_node_idx);
                new_node->test_nibble_idx = diff_idx; new_node->representative_record_idx = new_rec_idx;
                insert_into_new_node(new_node, new_rec_idx, coords, diff_idx, current_idx);
                if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart;
            }
            int frag = get_key_fragment(coords, node->test_nibble_idx, dim_);
            parent_slot = get_child_slot<true>(node, frag); if (!parent_slot) goto restart;
            current_idx = parent_slot->load(std::memory_order_acquire);
        }
        if (current_idx == 0) {
            u32 new_rec_idx = record_manager_.allocate_record(coords, dim_, value, sizeof(ValueType));
            u32 new_tagged_idx = (LEAF_BUCKET_SIZE > 1) ? TaggedIndex::make_bucket_idx(bucket_manager_.template allocate_bucket<BucketType>()) : TaggedIndex::make_leaf_idx(new_rec_idx);
            if constexpr (LEAF_BUCKET_SIZE > 1) bucket_manager_.template get_bucket<BucketType>(TaggedIndex::get_index(new_tagged_idx))->append(new_rec_idx);
            if (parent_slot->compare_exchange_strong(current_idx, new_tagged_idx, std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart;
        }
        if (TaggedIndex::is_leaf(current_idx)) {
            Record* existing_rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));
            if (memcmp(coords, existing_rec->coords, dim_ * sizeof(u64)) == 0) return;
            u32 new_rec_idx = record_manager_.allocate_record(coords, dim_, value, sizeof(ValueType));
            if constexpr (LEAF_BUCKET_SIZE > 1) {
                u32 bucket_idx = bucket_manager_.template allocate_bucket<BucketType>();
                BucketType* bucket = bucket_manager_.template get_bucket<BucketType>(bucket_idx);
                bucket->append(TaggedIndex::get_index(current_idx)); bucket->append(new_rec_idx);
                if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_bucket_idx(bucket_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            } else {
                int diff_idx = find_first_differing_fragment(coords, existing_rec->coords, dim_);
                u32 new_node_idx = node_manager_.template allocate_node<NodeType>();
                NodeType* new_node = node_manager_.template get_node<NodeType>(new_node_idx);
                new_node->test_nibble_idx = diff_idx; new_node->representative_record_idx = new_rec_idx;
                insert_into_new_node(new_node, new_rec_idx, coords, diff_idx, current_idx);
                if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            }
            goto restart;
        }
        if (TaggedIndex::is_bucket(current_idx)) {
            BucketType* bucket = bucket_manager_.template get_bucket<BucketType>(TaggedIndex::get_index(current_idx));
            for (uint8_t i = 0; i < bucket->count.load(std::memory_order_relaxed); ++i) if (memcmp(coords, record_manager_.get_record(bucket->record_indices[i])->coords, dim_ * sizeof(u64)) == 0) return;
            if (!bucket->is_full()) { if (bucket->append(record_manager_.allocate_record(coords, dim_, value, sizeof(ValueType)))) return; goto restart; }
            else {
                u32 new_rec_idx = record_manager_.allocate_record(coords, dim_, value, sizeof(ValueType));
                std::vector<const u64*> keys_in_bucket;
                keys_in_bucket.reserve(LEAF_BUCKET_SIZE + 1);
                keys_in_bucket.push_back(coords);
                for(uint8_t i = 0; i < bucket->count.load(std::memory_order_relaxed); ++i) keys_in_bucket.push_back(record_manager_.get_record(bucket->record_indices[i])->coords);
                int diff_idx = find_first_differing_fragment_multi(keys_in_bucket, dim_);
                if (diff_idx == -1) goto restart;
                u32 new_node_idx = node_manager_.template allocate_node<NodeType>();
                NodeType* new_node = node_manager_.template get_node<NodeType>(new_node_idx);
                new_node->test_nibble_idx = diff_idx; new_node->representative_record_idx = new_rec_idx;
                insert_recursive(coords, value, get_child_slot<true>(new_node, get_key_fragment(coords, diff_idx, dim_)));
                for (uint8_t i = 0; i < bucket->count.load(std::memory_order_relaxed); ++i) {
                    Record* reinsert_rec = record_manager_.get_record(bucket->record_indices[i]);
                    insert_recursive(reinsert_rec->coords, reinsert_rec->value_or_offset, get_child_slot<true>(new_node, get_key_fragment(reinsert_rec->coords, diff_idx, dim_)));
                }
                if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart;
            }
        }
    }
    std::optional<ValueType> get(const u64* coords) {
        u32 current_idx = root_ptr_.load(std::memory_order_acquire);
        while (TaggedIndex::is_node(current_idx)) {
            NodeType* node = node_manager_.template get_node<NodeType>(TaggedIndex::get_index(current_idx));
            std::atomic<u32>* child_slot = get_child_slot<false>(node, get_key_fragment(coords, node->test_nibble_idx, dim_));
            if (!child_slot) return std::nullopt;
            current_idx = child_slot->load(std::memory_order_acquire);
        }
        if (TaggedIndex::is_leaf(current_idx)) {
            Record* rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));
            if (rec && memcmp(rec->coords, coords, dim_ * sizeof(u64)) == 0) { ValueType val; memcpy(&val, &rec->value_or_offset, sizeof(ValueType)); return val; }
        } else if (TaggedIndex::is_bucket(current_idx)) {
            BucketType* bucket = bucket_manager_.template get_bucket<BucketType>(TaggedIndex::get_index(current_idx));
            for (uint8_t i = 0; i < bucket->count.load(std::memory_order_acquire); ++i) {
                Record* rec = record_manager_.get_record(bucket->record_indices[i]);
                if (memcmp(rec->coords, coords, dim_ * sizeof(u64)) == 0) { ValueType val; memcpy(&val, &rec->value_or_offset, sizeof(ValueType)); return val; }
            }
        }
        return std::nullopt;
    }
private:
    template<bool create_if_missing>
    std::atomic<u32>* get_child_slot(NodeType* node, int frag) {
        if constexpr (NodeType::IS_PAGED) {
            size_t page_idx = frag / SLOTS_PER_PAGE;
            typename NodeType::PageType* page = node->children_or_pages[page_idx].load(std::memory_order_acquire);
            if (page == nullptr) {
                if constexpr (!create_if_missing) return nullptr;
                typename NodeType::PageType* new_page = page_manager_.template allocate_page_and_get_ptr<typename NodeType::PageType>();
                if (!node->children_or_pages[page_idx].compare_exchange_strong(page, new_page, std::memory_order_release, std::memory_order_relaxed)) {}
                page = node->children_or_pages[page_idx].load(std::memory_order_acquire);
            }
            return &page->children[frag % SLOTS_PER_PAGE];
        } else return &node->children_or_pages[frag];
    }
};

template<size_t LEAF_BUCKET_SIZE>
void run_benchmark(int dimensionality, const std::string& key_type) {
    const size_t NUM_KEYS = 2000000;
    std::cout << "\n--- Flat Radix Tree Benchmark (DIM=" << dimensionality << ", BUCKET=" << LEAF_BUCKET_SIZE << ", " << key_type << ") ---" << std::endl;

    using TreeType = KeyValueRadixTree<256, 16, LEAF_BUCKET_SIZE, u64>;
    using NodeType = RadixNode<256, 16>;
    using PageType = NodePage<16>;
    using BucketType = LeafBucket<LEAF_BUCKET_SIZE>;

    MemoryContext ctx(sizeof(NodeType), sizeof(PageType), sizeof(BucketType), dimensionality);
    TreeType tree(ctx, dimensionality);

    std::vector<std::vector<u64>> keys(NUM_KEYS, std::vector<u64>(dimensionality));
    std::mt19937_64 rng(12345);

    if (key_type == "Random") {
        for (size_t i = 0; i < NUM_KEYS; ++i) {
            for(int d = 0; d < dimensionality; ++d) keys[i][d] = rng();
        }
    } else if (key_type == "Sequential") {
        for (size_t i = 0; i < NUM_KEYS; ++i) {
            for(int d = 0; d < dimensionality; ++d) keys[i][d] = i;
        }
    } else if (key_type == "Clustered") {
        u64 prefix = rng();
        for (size_t i = 0; i < NUM_KEYS; ++i) {
            keys[i][0] = prefix;
            for(int d = 1; d < dimensionality; ++d) keys[i][d] = rng();
        }
    }


    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        tree.insert(key.data(), (u64)key.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "[RadixTree] Insertion: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << std::endl;

    size_t found_count = 0;
    start = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        if (tree.get(key.data())) found_count++;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "[RadixTree] Lookup: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << " (Found " << found_count << "/" << NUM_KEYS << ")" << std::endl;

    size_t total_mem = ctx.get_mem_usage();
    std::cout << "[RadixTree] Memory: " << std::fixed << std::setprecision(2) << total_mem / (1024.0 * 1024.0) << " MB (" << (double)total_mem / NUM_KEYS << " bytes/key)" << std::endl;


    // --- std::unordered_map Benchmark ---
    std::unordered_map<std::string, u64> map;
    start = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        map[std::string((char*)key.data(), dimensionality * sizeof(u64))] = (u64)key.data();
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "[std::unordered_map] Insertion: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << std::endl;

    found_count = 0;
    start = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        if (map.count(std::string((char*)key.data(), dimensionality * sizeof(u64)))) found_count++;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "[std::unordered_map] Lookup: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << " (Found " << found_count << "/" << NUM_KEYS << ")" << std::endl;

    size_t key_size = dimensionality * sizeof(u64);
    size_t string_obj_size = sizeof(std::string);
    size_t node_overhead = sizeof(void*) * 2; // Approximate overhead per node
    size_t map_mem = NUM_KEYS * (key_size + sizeof(u64) + string_obj_size + node_overhead);
    std::cout << "[std::unordered_map] Memory (est.): " << std::fixed << std::setprecision(2) << map_mem / (1024.0 * 1024.0) << " MB (" << (double)map_mem / NUM_KEYS << " bytes/key)" << std::endl;

}

int main() {
    try {
        std::vector<int> dims_to_test = {1, 3, 4, 8};
        std::vector<std::string> key_types_to_test = {"Random", "Sequential", "Clustered"};
        std::vector<size_t> bucket_sizes_to_test = {1, 2, 4};

        for (int dims : dims_to_test) {
            for (const auto& key_type : key_types_to_test) {
                if (dims == 1 && key_type == "Clustered") continue;
                for (size_t bucket_size : bucket_sizes_to_test) {
                    if (bucket_size == 1) run_benchmark<1>(dims, key_type);
                    else if (bucket_size == 2) run_benchmark<2>(dims, key_type);
                    else if (bucket_size == 4) run_benchmark<4>(dims, key_type);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl; return 1;
    }
    return 0;
}
