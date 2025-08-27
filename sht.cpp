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

#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// --- Foundational Data Structures (Adapted from Stax) ---

class ValueStore;
template<size_t FANOUT, size_t SLOTS_PER_PAGE> struct RadixNode;

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

template<size_t SLOTS_PER_PAGE>
struct NodePage {
    std::atomic<uint32_t> children[SLOTS_PER_PAGE];
    NodePage() {
        for(size_t i = 0; i < SLOTS_PER_PAGE; ++i) children[i].store(0, std::memory_order_relaxed);
    }
};

template<size_t FANOUT, size_t SLOTS_PER_PAGE>
struct RadixNode {
    uint32_t test_nibble_idx;
    uint32_t representative_record_idx;

    struct PagedLayout {
        static constexpr size_t PAGES_PER_NODE = FANOUT / SLOTS_PER_PAGE;
        std::atomic<NodePage<SLOTS_PER_PAGE>*> pages[PAGES_PER_NODE];
    };

    struct NonPagedLayout {
        std::atomic<uint32_t> children[FANOUT];
    };

    using Layout = typename std::conditional<(FANOUT >= SLOTS_PER_PAGE), PagedLayout, NonPagedLayout>::type;
    Layout layout;

    RadixNode() : test_nibble_idx(0), representative_record_idx(0) {
        if constexpr (FANOUT >= SLOTS_PER_PAGE) {
            for(size_t i = 0; i < PagedLayout::PAGES_PER_NODE; ++i) layout.pages[i].store(nullptr, std::memory_order_relaxed);
        } else {
            for(size_t i = 0; i < FANOUT; ++i) layout.children[i].store(0, std::memory_order_relaxed);
        }
    }
};

template<size_t FANOUT, size_t SLOTS_PER_PAGE>
class NodeManager {
private:
    std::atomic<uint32_t> next_node_idx_;
    uint8_t* node_pool_;
    static constexpr uint32_t MAX_NODES = 32 * 1024 * 1024;
    static constexpr size_t NODE_SIZE = sizeof(RadixNode<FANOUT, SLOTS_PER_PAGE>);
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
    NodeManager() : next_node_idx_(1) {
        size_t pool_size = (size_t)MAX_NODES * NODE_SIZE;
        node_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (node_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for node pool");
    }
    ~NodeManager() { munmap(node_pool_, (size_t)MAX_NODES * NODE_SIZE); }
    RadixNode<FANOUT, SLOTS_PER_PAGE>* get_node(uint32_t idx) { return reinterpret_cast<RadixNode<FANOUT, SLOTS_PER_PAGE>*>(node_pool_ + (size_t)idx * NODE_SIZE); }
    uint32_t allocate_node() {
        if (cache_ptr_ < 0) refill_cache();
        uint32_t new_idx = node_cache_[cache_ptr_--];
        new (get_node(new_idx)) RadixNode<FANOUT, SLOTS_PER_PAGE>();
        return new_idx;
    }
    size_t get_mem_usage() const { return (size_t)next_node_idx_.load() * NODE_SIZE; }
};
template<size_t FANOUT, size_t SLOTS_PER_PAGE>
thread_local uint32_t NodeManager<FANOUT, SLOTS_PER_PAGE>::node_cache_[NodeManager<FANOUT, SLOTS_PER_PAGE>::THREAD_CACHE_SIZE];
template<size_t FANOUT, size_t SLOTS_PER_PAGE>
thread_local int NodeManager<FANOUT, SLOTS_PER_PAGE>::cache_ptr_ = -1;


template<size_t SLOTS_PER_PAGE>
class PageManager {
private:
    std::atomic<NodePage<SLOTS_PER_PAGE>*> next_page_ptr_;
    uint8_t* page_pool_;
    static constexpr size_t PAGE_SIZE = sizeof(NodePage<SLOTS_PER_PAGE>);
    static constexpr uint32_t MAX_PAGES = 16 * 1024 * 1024;
    static constexpr size_t THREAD_CACHE_SIZE = 64;
    static thread_local NodePage<SLOTS_PER_PAGE>* page_cache_[THREAD_CACHE_SIZE];
    static thread_local int cache_ptr_;
    std::mutex pool_mutex_;

    void refill_cache() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        uint8_t* start_ptr = (uint8_t*)next_page_ptr_.load(std::memory_order_relaxed);
        size_t bytes_to_alloc = THREAD_CACHE_SIZE * PAGE_SIZE;
        if (start_ptr + bytes_to_alloc > page_pool_ + (size_t)MAX_PAGES * PAGE_SIZE) {
             throw std::runtime_error("Page pool exhausted");
        }
        next_page_ptr_.store((NodePage<SLOTS_PER_PAGE>*)(start_ptr + bytes_to_alloc), std::memory_order_relaxed);
        for (size_t i = 0; i < THREAD_CACHE_SIZE; ++i) {
            page_cache_[i] = new (start_ptr + i * PAGE_SIZE) NodePage<SLOTS_PER_PAGE>();
        }
        cache_ptr_ = THREAD_CACHE_SIZE - 1;
    }

public:
    PageManager() {
        size_t pool_size = (size_t)MAX_PAGES * PAGE_SIZE;
        page_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (page_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for page pool");
        next_page_ptr_.store((NodePage<SLOTS_PER_PAGE>*)page_pool_);
    }

    ~PageManager() {
        munmap(page_pool_, (size_t)MAX_PAGES * PAGE_SIZE);
    }

    NodePage<SLOTS_PER_PAGE>* allocate_page() {
        if (cache_ptr_ < 0) refill_cache();
        NodePage<SLOTS_PER_PAGE>* new_page = page_cache_[cache_ptr_--];
        // Already zero-initialized in refill_cache
        return new_page;
    }
};
template<size_t SLOTS_PER_PAGE>
thread_local NodePage<SLOTS_PER_PAGE>* PageManager<SLOTS_PER_PAGE>::page_cache_[PageManager<SLOTS_PER_PAGE>::THREAD_CACHE_SIZE];
template<size_t SLOTS_PER_PAGE>
thread_local int PageManager<SLOTS_PER_PAGE>::cache_ptr_ = -1;


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

template<size_t FANOUT, size_t SLOTS_PER_PAGE>
class KeyValueRadixTree {
    NodeManager<FANOUT, SLOTS_PER_PAGE>& node_manager_;
    PageManager<SLOTS_PER_PAGE>& page_manager_;
    RecordManager& record_manager_;
    std::atomic<uint32_t>& root_ptr_;
    const uint32_t dim_;

    std::atomic<uint32_t>* get_child_slot(RadixNode<FANOUT, SLOTS_PER_PAGE>* node, int child_index) {
        if constexpr (FANOUT < SLOTS_PER_PAGE) {
            return &node->layout.children[child_index];
        } else {
            const size_t page_index = child_index / SLOTS_PER_PAGE;
            const size_t slot_index_within_page = child_index % SLOTS_PER_PAGE;

            NodePage<SLOTS_PER_PAGE>* page = node->layout.pages[page_index].load(std::memory_order_acquire);
            if (page == nullptr) {
                NodePage<SLOTS_PER_PAGE>* new_page = page_manager_.allocate_page();
                if (!node->layout.pages[page_index].compare_exchange_strong(page, new_page, std::memory_order_release, std::memory_order_acquire)) {
                    // Another thread won the race, the page it allocated is now in `page`.
                    // The page we allocated (`new_page`) will be recycled by the thread-local cache.
                } else {
                    // We successfully installed our new page.
                    page = new_page;
                }
            }
            return &page->children[slot_index_within_page];
        }
    }

    static inline int get_nibble(const uint64_t* coords, int nibble_idx, int d) {
        int max_nibbles = d * 16;
        if (nibble_idx < 0 || nibble_idx >= max_nibbles) return 0;
        int dim_idx = nibble_idx / 16;
        int nibble_in_dim = nibble_idx % 16;
        return (coords[dim_idx] >> (60 - (nibble_in_dim * 4))) & 0x0F;
    }

    static inline int get_byte(const uint64_t* coords, int byte_idx, int d) {
        int max_bytes = d * 8;
        if (byte_idx < 0 || byte_idx >= max_bytes) return 0;
        int dim_idx = byte_idx / 8;
        int byte_in_dim = byte_idx % 8;
        return (coords[dim_idx] >> (56 - (byte_in_dim * 8))) & 0xFF;
    }

    static int find_first_differing_key_part(const uint64_t* k1, const uint64_t* k2, int d, int start_idx = 0) {
        if constexpr (FANOUT == 256) {
            int max_bytes = d * 8;
            for (int i = start_idx; i < max_bytes; ++i) {
                if (get_byte(k1, i, d) != get_byte(k2, i, d)) return i;
            }
        } else {
            int max_nibbles = d * 16;
            for (int i = start_idx; i < max_nibbles; ++i) {
                if (get_nibble(k1, i, d) != get_nibble(k2, i, d)) return i;
            }
        }
        return -1;
    }

    static int get_key_part(const uint64_t* coords, int idx, int d) {
        if constexpr (FANOUT == 256) {
            return get_byte(coords, idx, d);
        } else {
            return get_nibble(coords, idx, d);
        }
    }

public:
    KeyValueRadixTree(NodeManager<FANOUT, SLOTS_PER_PAGE>& nm, PageManager<SLOTS_PER_PAGE>& pm, RecordManager& rm, std::atomic<uint32_t>& root, uint32_t d)
        : node_manager_(nm), page_manager_(pm), record_manager_(rm), root_ptr_(root), dim_(d) {}

    void insert(const uint64_t* coords, uint64_t value) {
    restart:
        std::atomic<uint32_t>* parent_slot = &root_ptr_;
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);

        while(TaggedIndex::is_node(current_idx)) {
            RadixNode<FANOUT, SLOTS_PER_PAGE>* node = node_manager_.get_node(TaggedIndex::get_index(current_idx));
            Record* rep_rec = record_manager_.get_record(node->representative_record_idx);

            int diff_idx = find_first_differing_key_part(coords, rep_rec->coords, dim_);

            if (diff_idx != -1 && (uint32_t)diff_idx < node->test_nibble_idx) {
                uint32_t new_node_idx = node_manager_.allocate_node();
                RadixNode<FANOUT, SLOTS_PER_PAGE>* new_node = node_manager_.get_node(new_node_idx);
                uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
                new_node->test_nibble_idx = diff_idx;
                new_node->representative_record_idx = new_rec_idx;

                int new_key_part = get_key_part(coords, diff_idx, dim_);
                int existing_key_part = get_key_part(rep_rec->coords, diff_idx, dim_);

                get_child_slot(new_node, new_key_part)->store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);
                get_child_slot(new_node, existing_key_part)->store(current_idx, std::memory_order_relaxed);

                if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart;
            }
            int key_part = get_key_part(coords, node->test_nibble_idx, dim_);
            parent_slot = get_child_slot(node, key_part);
            current_idx = parent_slot->load(std::memory_order_acquire);
        }

        if (current_idx == 0) {
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart;
        }

        if (TaggedIndex::is_leaf(current_idx)) {
            Record* existing_rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));
            int diff_idx = find_first_differing_key_part(coords, existing_rec->coords, dim_);
            if (diff_idx == -1) return;

            uint32_t new_node_idx = node_manager_.allocate_node();
            RadixNode<FANOUT, SLOTS_PER_PAGE>* new_node = node_manager_.get_node(new_node_idx);
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            new_node->test_nibble_idx = diff_idx;
            new_node->representative_record_idx = new_rec_idx;

            int new_key_part = get_key_part(coords, diff_idx, dim_);
            int existing_key_part = get_key_part(existing_rec->coords, diff_idx, dim_);
            get_child_slot(new_node, new_key_part)->store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);
            get_child_slot(new_node, existing_key_part)->store(current_idx, std::memory_order_relaxed);

            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart;
        }
    }

    bool get(const uint64_t* coords) {
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);
        while (TaggedIndex::is_node(current_idx)) {
            RadixNode<FANOUT, SLOTS_PER_PAGE>* node = node_manager_.get_node(TaggedIndex::get_index(current_idx));
            int key_part = get_key_part(coords, node->test_nibble_idx, dim_);
            current_idx = get_child_slot(node, key_part)->load(std::memory_order_acquire);
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

        RadixNode<FANOUT, SLOTS_PER_PAGE>* node = node_manager_.get_node(TaggedIndex::get_index(node_idx));
        int start_part = tight_lower ? get_key_part(start_key, node->test_nibble_idx, dim_) : 0;
        int end_part = tight_upper ? get_key_part(end_key, node->test_nibble_idx, dim_) : (FANOUT - 1);

        for (int i = start_part; i <= end_part; ++i) {
            uint32_t child_idx = get_child_slot(node, i)->load(std::memory_order_acquire);
            if (child_idx) {
                bool next_tight_lower = tight_lower && (i == start_part);
                bool next_tight_upper = tight_upper && (i == end_part);
                scan_recursive(child_idx, start_key, end_key, next_tight_lower, next_tight_upper, results);
            }
        }
    }
};

#include <unordered_map>

const int MAX_DIMS = 8;
struct Key { uint64_t coords[MAX_DIMS]; };

struct KeyEq {
    int dim;
    bool operator()(const Key& a, const Key& b) const {
        return memcmp(a.coords, b.coords, dim * sizeof(uint64_t)) == 0;
    }
};

struct KeyHash {
    int dim;
    std::size_t operator()(const Key& key) const {
        size_t h = 0;
        // Simple hash combination
        for (int i = 0; i < dim; ++i) {
            h ^= std::hash<uint64_t>{}(key.coords[i]) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

void run_unordered_map_benchmark(size_t num_keys, int num_threads, int dimensionality, const std::string& key_type, const std::vector<Key>& keys, const std::vector<Key>& miss_keys) {
    std::cout << "\n--- Benchmark: std::unordered_map (" << (dimensionality*8) << " bytes) ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    std::unordered_map<Key, uint64_t, KeyHash, KeyEq> map(num_keys, KeyHash{dimensionality}, KeyEq{dimensionality});
    std::mutex map_mutex;

    std::cout << "\n--- INSERTION ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t start = i * (num_keys / num_threads);
            size_t end = (i == num_threads - 1) ? num_keys : start + (num_keys / num_threads);
            for (size_t j = start; j < end; ++j) {
                std::lock_guard<std::mutex> lock(map_mutex);
                map[keys[j]] = j;
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[std::unordered_map] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/insert" << std::endl;

    std::cout << "\n--- HIT LATENCY (LOOKUP) ---" << std::endl;
    size_t found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) {
        if (map.count(keys[i])) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[std::unordered_map] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;
    if (found_count == num_keys) {
        std::cout << "  Verification: SUCCESS" << std::endl;
    } else {
        std::cout << "  Verification: FAILED (Found " << found_count << "/" << num_keys << ")" << std::endl;
    }

    std::cout << "\n--- MISS LATENCY (LOOKUP) ---" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) { map.count(miss_keys[i]); }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[std::unordered_map] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;
}


template<size_t FANOUT, size_t SLOTS_PER_PAGE>
void run_benchmark(size_t num_keys, int num_threads, int dimensionality, const std::string& key_type, const std::vector<Key>& keys, const std::vector<Key>& miss_keys) {
    std::cout << "\n--- Benchmark: " << dimensionality << "D Key-Value Radix Tree (" << (dimensionality*8) << " bytes) ---" << std::endl;
    std::cout << "--- FANOUT=" << FANOUT << ", SLOTS_PER_PAGE=" << SLOTS_PER_PAGE << " ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    ValueStore vs;
    NodeManager<FANOUT, SLOTS_PER_PAGE> nm;
    PageManager<SLOTS_PER_PAGE> pm;
    RecordManager rm(&vs, dimensionality);
    std::atomic<uint32_t> root_ptr(0);
    KeyValueRadixTree<FANOUT, SLOTS_PER_PAGE> tree(nm, pm, rm, root_ptr, dimensionality);

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

int main() {
    try {
        const int NUM_THREADS = std::thread::hardware_concurrency();
        const size_t LARGE_KEY_COUNT = 1000000;

        std::vector<int> dims_to_test = {1, 3, 8};
        std::vector<std::string> key_types_to_test = {"Random", "Sequential"};

        for (int dims : dims_to_test) {
            for (const auto& key_type : key_types_to_test) {
                std::cout << "\n========================================================" << std::endl;
                std::cout << "          " << dims << "D / " << key_type << " Keys" << std::endl;
                std::cout << "========================================================" << std::endl;

                std::vector<Key> keys(LARGE_KEY_COUNT);
                std::vector<Key> miss_keys(LARGE_KEY_COUNT);

                if (key_type == "Random") {
                    std::mt19937_64 rng(12345);
                    for(size_t i = 0; i < LARGE_KEY_COUNT; ++i) {
                        for (int d = 0; d < dims; ++d) {
                            keys[i].coords[d] = rng();
                            miss_keys[i].coords[d] = rng();
                        }
                    }
                } else { // Sequential
                    for(size_t i = 0; i < LARGE_KEY_COUNT; ++i) {
                        for (int d = 0; d < dims; ++d) {
                            keys[i].coords[d] = i;
                            miss_keys[i].coords[d] = i + LARGE_KEY_COUNT;
                        }
                    }
                }

                // Test with non-paged layout
                run_benchmark<16, 32>(LARGE_KEY_COUNT, NUM_THREADS, dims, key_type, keys, miss_keys);
                // Test with paged layout (FANOUT 16)
                run_benchmark<16, 16>(LARGE_KEY_COUNT, NUM_THREADS, dims, key_type, keys, miss_keys);
                 // Test with paged layout (FANOUT 256)
                run_benchmark<256, 16>(LARGE_KEY_COUNT, NUM_THREADS, dims, key_type, keys, miss_keys);
                // Test with std::unordered_map
                run_unordered_map_benchmark(LARGE_KEY_COUNT, NUM_THREADS, dims, key_type, keys, miss_keys);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
