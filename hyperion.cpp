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
    u64 value;
    u64 coords[];
    static size_t get_size(u32 d) { return sizeof(Record) + sizeof(u64) * d; }
};

namespace TaggedIndex {
    static constexpr u32 NODE256_TAG = 0b00;
    static constexpr u32 NODE16_TAG  = 0b01;
    static constexpr u32 LEAF_TAG    = 0b10;

    static constexpr u32 TAG_MASK    = 0b11 << 30;
    static constexpr u32 INDEX_MASK  = ~TAG_MASK;

    inline u32 get_tag(u32 ptr) { return (ptr >> 30); }
    inline u32 get_index(u32 ptr) { return ptr & INDEX_MASK; }

    inline bool is_leaf(u32 ptr) { return get_tag(ptr) == LEAF_TAG; }
    inline bool is_node16(u32 ptr) { return get_tag(ptr) == NODE16_TAG; }
    inline bool is_node256(u32 ptr) { return get_tag(ptr) == NODE256_TAG && ptr != 0; }

    inline u32 make_leaf_idx(u32 rec_idx) { return (rec_idx & INDEX_MASK) | (LEAF_TAG << 30); }
    inline u32 make_node16_idx(u32 node_idx) { return (node_idx & INDEX_MASK) | (NODE16_TAG << 30); }
    inline u32 make_node256_idx(u32 node_idx) { return (node_idx & INDEX_MASK) | (NODE256_TAG << 30); }
};

template<size_t SIZE> struct Node16 {
    std::atomic<uint8_t> count{0};
    uint8_t keys[SIZE];
    std::atomic<u32> children[SIZE];
    Node16() { for(size_t i=0; i<SIZE; ++i) { keys[i] = 0; children[i].store(0, std::memory_order_relaxed); }}
    bool is_full() const { return count.load(std::memory_order_relaxed) >= SIZE; }
};

struct Node256 {
    std::atomic<u32> children[256];
    Node256() { for (size_t i = 0; i < 256; ++i) children[i].store(0, std::memory_order_relaxed); }
};

// --- Memory Management ---
template<typename T>
class Manager {
    std::atomic<u32> next_idx_{1};
    uint8_t* pool_;
    const size_t max_nodes_;
public:
    Manager(size_t max_nodes) : max_nodes_(max_nodes) {
        size_t pool_size = max_nodes_ * sizeof(T);
        pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pool_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    }
    ~Manager() { munmap(pool_, max_nodes_ * sizeof(T)); }
    T* get_node(u32 idx) { return reinterpret_cast<T*>(pool_ + (size_t)idx * sizeof(T)); }
    u32 allocate_node() {
        u32 idx = next_idx_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= max_nodes_) throw std::runtime_error("Pool exhausted");
        new (get_node(idx)) T(); return idx;
    }
    size_t get_mem_usage() const { return (size_t)next_idx_.load() * sizeof(T); }
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
    u32 allocate_record(const u64* coords, u32 dim, u64 value) {
        u32 idx = next_record_idx_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_RECORDS) throw std::runtime_error("Record pool exhausted");
        Record* rec = get_record(idx);
        rec->value = value;
        memcpy(rec->coords, coords, sizeof(u64) * dim);
        return idx;
    }
    size_t get_mem_usage() const { return (size_t)next_record_idx_.load() * record_size_with_coords_; }
};


struct Box {
    const u64* lower_bounds;
    const u64* upper_bounds;
    u32 dim;
};

struct MemoryContext {
    std::unique_ptr<Manager<Node16<4>>> nm16_4;
    std::unique_ptr<Manager<Node16<16>>> nm16_16;
    std::unique_ptr<Manager<Node256>> nm256;
    std::unique_ptr<RecordManager> rm;
    MemoryContext(u32 dimensionality) {
        nm16_4 = std::make_unique<Manager<Node16<4>>>(16 * 1024 * 1024);
        nm16_16 = std::make_unique<Manager<Node16<16>>>(16 * 1024 * 1024);
        nm256 = std::make_unique<Manager<Node256>>(2 * 1024 * 1024);
        rm = std::make_unique<RecordManager>(dimensionality);
    }
};

// --- Hyperion Tree ---
template<size_t NODE16_SIZE, typename ValueType>
class HyperionTree {
    using Node16Type = Node16<NODE16_SIZE>;

    Manager<Node16<NODE16_SIZE>>* nm16_;
    Manager<Node256>& nm256_;
    RecordManager& rm_;

    std::atomic<u32> root_ptr_{0};
    const u32 dim_;

    static inline uint8_t get_key_fragment(const u64* key, int depth, u32 dim) {
        const u32 current_dim = depth % dim;
        const u32 byte_idx_in_dim = depth / dim;
        if (byte_idx_in_dim >= sizeof(u64)) return 0;
        const u32 byte_shift = (7 - (byte_idx_in_dim % sizeof(u64))) * 8;
        return (key[current_dim] >> byte_shift) & 0xFF;
    }

    void insert_recursive(const u64* key, u64 value, std::atomic<u32>* parent_slot, int depth) {
        u32 leaf_ptr = 0;

        while(true) {
            u32 current_tagged_ptr = parent_slot->load(std::memory_order_acquire);

            if (current_tagged_ptr == 0) {
                if (leaf_ptr == 0) {
                    u32 new_rec_idx = rm_.allocate_record(key, dim_, value);
                    leaf_ptr = TaggedIndex::make_leaf_idx(new_rec_idx);
                }
                if(parent_slot->compare_exchange_strong(current_tagged_ptr, leaf_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
                else continue;
            }

            if (TaggedIndex::is_leaf(current_tagged_ptr)) {
                u32 existing_rec_idx = TaggedIndex::get_index(current_tagged_ptr);
                Record* existing_rec = rm_.get_record(existing_rec_idx);

                bool are_equal = true;
                for(u32 i = 0; i < dim_; ++i) {
                    if (key[i] != existing_rec->coords[i]) {
                        are_equal = false;
                        break;
                    }
                }
                if (are_equal) return;

                if (leaf_ptr == 0) {
                    u32 new_rec_idx = rm_.allocate_record(key, dim_, value);
                    leaf_ptr = TaggedIndex::make_leaf_idx(new_rec_idx);
                }

                u32 node16_idx = nm16_->allocate_node();
                Node16Type* node = nm16_->get_node(node16_idx);

                int existing_frag = get_key_fragment(existing_rec->coords, depth, dim_);
                int new_frag = get_key_fragment(key, depth, dim_);

                if (existing_frag != new_frag) {
                    node->keys[0] = std::min(existing_frag, new_frag);
                    node->keys[1] = std::max(existing_frag, new_frag);
                    node->children[0].store(existing_frag < new_frag ? current_tagged_ptr : leaf_ptr, std::memory_order_relaxed);
                    node->children[1].store(existing_frag < new_frag ? leaf_ptr : current_tagged_ptr, std::memory_order_relaxed);
                    node->count.store(2, std::memory_order_relaxed);
                    if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node16_idx(node16_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                    else continue;
                } else {
                    node->keys[0] = new_frag;
                    node->count.store(1, std::memory_order_relaxed);
                    if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node16_idx(node16_idx), std::memory_order_release, std::memory_order_relaxed)) {
                        insert_recursive(existing_rec->coords, existing_rec->value, &node->children[0], depth + 1);
                        insert_recursive(key, value, &node->children[0], depth + 1);
                        return;
                    } else continue;
                }
            }

            if (TaggedIndex::is_node16(current_tagged_ptr)) {
                Node16Type* node = nm16_->get_node(TaggedIndex::get_index(current_tagged_ptr));
                uint8_t frag = get_key_fragment(key, depth, dim_);
                int index = -1;
                uint8_t count = node->count.load(std::memory_order_relaxed);

                #if defined(__x86_64__) || defined(_M_X64)
                if constexpr (NODE16_SIZE >= 16) {
                    __m128i keys_vec = _mm_loadu_si128((const __m128i*)node->keys);
                    __m128i frag_vec = _mm_set1_epi8(frag);
                    __m128i cmp_mask_vec = _mm_cmpeq_epi8(keys_vec, frag_vec);
                    int mask = _mm_movemask_epi8(cmp_mask_vec);
                    mask &= (1 << count) - 1;
                    if (mask > 0) {
                        #if defined(__GNUC__) || defined(__clang__)
                        index = __builtin_ctz(mask);
                        #else
                        unsigned long bsf_index;
                        _BitScanForward(&bsf_index, mask);
                        index = bsf_index;
                        #endif
                    }
                } else {
                    for(uint8_t i=0; i<count; ++i) {
                        if(node->keys[i] == frag) {
                            index = i;
                            break;
                        }
                    }
                }
                #else
                for(uint8_t i=0; i<count; ++i) {
                    if(node->keys[i] == frag) {
                        index = i;
                        break;
                    }
                }
                #endif

                if (index != -1) {
                    parent_slot = &node->children[index];
                    depth++;
                    goto next_level;
                }

                if (!node->is_full()) {
                    if (leaf_ptr == 0) {
                        u32 new_rec_idx = rm_.allocate_record(key, dim_, value);
                        leaf_ptr = TaggedIndex::make_leaf_idx(new_rec_idx);
                    }
                    uint8_t c = node->count.fetch_add(1, std::memory_order_relaxed);
                    node->keys[c] = frag;
                    node->children[c].store(leaf_ptr, std::memory_order_release);
                    return;
                } else {
                    u32 new_node256_idx = nm256_.allocate_node();
                    Node256* new_node256 = nm256_.get_node(new_node256_idx);
                    for(uint8_t i=0; i<count; ++i) {
                        new_node256->children[node->keys[i]].store(node->children[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
                    }
                    if (leaf_ptr == 0) {
                        u32 new_rec_idx = rm_.allocate_record(key, dim_, value);
                        leaf_ptr = TaggedIndex::make_leaf_idx(new_rec_idx);
                    }
                    new_node256->children[frag].store(leaf_ptr, std::memory_order_relaxed);
                    if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node256_idx(new_node256_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                    else continue;
                }
            }

            if (TaggedIndex::is_node256(current_tagged_ptr)) {
                Node256* node = nm256_.get_node(TaggedIndex::get_index(current_tagged_ptr));
                parent_slot = &node->children[get_key_fragment(key, depth, dim_)];
                depth++;
                continue;
            }
            next_level:;
        }
    }

public:
    HyperionTree(MemoryContext& ctx, u32 d) : nm256_(*ctx.nm256), rm_(*ctx.rm), dim_(d) {
        if constexpr (NODE16_SIZE == 4) nm16_ = ctx.nm16_4.get();
        else if constexpr (NODE16_SIZE == 16) nm16_ = ctx.nm16_16.get();
    }

    void insert(const u64* key, ValueType value) {
        u64 val_u64 = 0;
        memcpy(&val_u64, &value, sizeof(ValueType));
        insert_recursive(key, val_u64, &root_ptr_, 0);
    }

    std::optional<ValueType> get(const u64* key) {
        int depth = 0;
        u32 current_tagged_ptr = root_ptr_.load(std::memory_order_acquire);

        while(true) {
            if(current_tagged_ptr == 0) return std::nullopt;

            if(TaggedIndex::is_leaf(current_tagged_ptr)) {
                Record* rec = rm_.get_record(TaggedIndex::get_index(current_tagged_ptr));
                bool are_equal = true;
                for(u32 i = 0; i < dim_; ++i) {
                    if (key[i] != rec->coords[i]) {
                        are_equal = false;
                        break;
                    }
                }
                if (are_equal) {
                    ValueType val;
                    memcpy(&val, &rec->value, sizeof(ValueType));
                    return val;
                }
                return std::nullopt;
            }

            uint8_t frag = get_key_fragment(key, depth, dim_);
            if(TaggedIndex::is_node16(current_tagged_ptr)) {
                Node16Type* node = nm16_->get_node(TaggedIndex::get_index(current_tagged_ptr));
                #if defined(__x86_64__) || defined(_M_X64)
                if constexpr (NODE16_SIZE >= 16) {
                    __m128i keys_vec = _mm_loadu_si128((const __m128i*)node->keys);
                    __m128i frag_vec = _mm_set1_epi8(frag);
                    __m128i cmp_mask_vec = _mm_cmpeq_epi8(keys_vec, frag_vec);
                    int mask = _mm_movemask_epi8(cmp_mask_vec);
                    uint8_t count = node->count.load(std::memory_order_relaxed);
                    mask &= (1 << count) - 1;
                    if (mask > 0) {
                        #if defined(__GNUC__) || defined(__clang__)
                        int index = __builtin_ctz(mask);
                        #else
                        unsigned long index;
                        _BitScanForward(&index, mask);
                        #endif
                        current_tagged_ptr = node->children[index].load(std::memory_order_acquire);
                        depth++;
                        goto next_level;
                    }
                    return std::nullopt;
                }
                #endif
                uint8_t count = node->count.load(std::memory_order_relaxed);
                for(uint8_t i=0; i<count; ++i) {
                    if(node->keys[i] == frag) {
                        current_tagged_ptr = node->children[i].load(std::memory_order_acquire);
                        depth++;
                        goto next_level;
                    }
                }
                return std::nullopt;
            }

            if(TaggedIndex::is_node256(current_tagged_ptr)) {
                Node256* node = nm256_.get_node(TaggedIndex::get_index(current_tagged_ptr));
                current_tagged_ptr = node->children[frag].load(std::memory_order_acquire);
                depth++;
                continue;
            }
            next_level:;
        }
        return std::nullopt;
    }

    void box_query(const Box& query_box, std::vector<ValueType>& results) {
        box_query_recursive(root_ptr_.load(std::memory_order_acquire), query_box, 0, results);
    }

private:
    void box_query_recursive(u32 current_tagged_ptr, const Box& query_box, int depth, std::vector<ValueType>& results) {
        if (current_tagged_ptr == 0) {
            return;
        }

        if (TaggedIndex::is_leaf(current_tagged_ptr)) {
            Record* rec = rm_.get_record(TaggedIndex::get_index(current_tagged_ptr));
            bool is_inside = true;
            for (u32 i = 0; i < dim_; ++i) {
                if (rec->coords[i] < query_box.lower_bounds[i] || rec->coords[i] > query_box.upper_bounds[i]) {
                    is_inside = false;
                    break;
                }
            }
            if (is_inside) {
                ValueType val;
                memcpy(&val, &rec->value, sizeof(ValueType));
                results.push_back(val);
            }
            return;
        }

        const u32 current_dim = depth % dim_;
        const u32 byte_idx_in_dim = depth / dim_;

        if (byte_idx_in_dim >= sizeof(u64)) {
            return;
        }

        const u32 byte_shift = (7 - (byte_idx_in_dim % sizeof(u64))) * 8;
        u64 prefix_mask = (byte_shift == 56) ? 0 : (((u64)-1) << (byte_shift + 8));

        uint8_t start_frag = 0;
        uint8_t end_frag = 255;

        if ((query_box.lower_bounds[current_dim] & prefix_mask) == (query_box.upper_bounds[current_dim] & prefix_mask)) {
            start_frag = (query_box.lower_bounds[current_dim] >> byte_shift) & 0xFF;
            end_frag = (query_box.upper_bounds[current_dim] >> byte_shift) & 0xFF;
        }

        if (TaggedIndex::is_node16(current_tagged_ptr)) {
            Node16Type* node = nm16_->get_node(TaggedIndex::get_index(current_tagged_ptr));
            uint8_t count = node->count.load(std::memory_order_relaxed);
            for (uint8_t i = 0; i < count; ++i) {
                uint8_t frag = node->keys[i];
                if (frag >= start_frag && frag <= end_frag) {
                    box_query_recursive(node->children[i].load(std::memory_order_acquire), query_box, depth + 1, results);
                }
            }
        } else if (TaggedIndex::is_node256(current_tagged_ptr)) {
            Node256* node = nm256_.get_node(TaggedIndex::get_index(current_tagged_ptr));
            for (int i = start_frag; i <= end_frag; ++i) {
                u32 child_ptr = node->children[i].load(std::memory_order_acquire);
                if (child_ptr != 0) {
                    box_query_recursive(child_ptr, query_box, depth + 1, results);
                }
            }
        }
    }
};

// --- Utility Functions ---
void print_header() {
    std::cout << std::left << std::setw(25) << "Query Name" << " | "
              << std::right << std::setw(10) << "Items" << " | "
              << std::right << std::setw(15) << "Latency (ns)" << " | "
              << std::right << std::setw(20) << "Amortized ns/Item" << " | "
              << std::right << std::setw(15) << "Ops/sec" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
}

void print_query_stats(const std::string& name, size_t items, long long latency_ns) {
    double amortized_ns = (items > 0) ? (double)latency_ns / items : latency_ns;
    double ops_sec = (latency_ns > 0) ? 1e9 / latency_ns : 0;

    std::cout << std::left << std::setw(25) << name << " | "
              << std::right << std::setw(10) << items << " | "
              << std::right << std::setw(15) << latency_ns << " | "
              << std::right << std::setw(20) << std::fixed << std::setprecision(2) << amortized_ns << " | "
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << ops_sec << std::endl;
}


// --- Benchmark Suite ---
template<size_t NODE16_SIZE>
void run_hyperion_benchmarks(u32 dim, size_t num_keys) {
    using TreeType = HyperionTree<NODE16_SIZE, u64>;
    using ValueType = u64;

    std::cout << "\n--- Hyperion Benchmark (DIM=" << dim << ", NODE16_SIZE=" << NODE16_SIZE << ", N=" << num_keys << ") ---\n" << std::endl;

    MemoryContext ctx(dim);
    TreeType tree(ctx, dim);

    std::vector<std::vector<u64>> keys(num_keys, std::vector<u64>(dim));
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<u64> dist(0, std::numeric_limits<u64>::max());

    for (auto& key : keys) {
        for (u32 d = 0; d < dim; ++d) {
            key[d] = dist(rng);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) {
        tree.insert(keys[i].data(), i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Total Insertion Time: " << duration.count() / 1e6 << " ms (" << (double)duration.count() / num_keys << " ns/op)\n" << std::endl;

    // --- Query Performance ---
    print_header();
    std::vector<ValueType> results;
    results.reserve(num_keys);

    const u64 MAX_COORD = std::numeric_limits<u64>::max();
    const int ITERS = 20; // Reduced iterations for faster benchmark run
    const size_t MIN_ITEMS = 1000;


    // Point Query
    {
        long long total_latency = 0;
        size_t total_items = 0;
        const int POINT_ITERS = 1000;
        for (int i = 0; i < POINT_ITERS; ++i) {
            results.clear();
            std::vector<u64> query_key = keys[(num_keys / 2 + i * 7) % num_keys];
            Box query_box = {query_key.data(), query_key.data(), dim};
            auto s = std::chrono::high_resolution_clock::now();
            tree.box_query(query_box, results);
            auto e = std::chrono::high_resolution_clock::now();
            total_latency += std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
            total_items += results.size();
        }
        print_query_stats("Point Query", total_items / POINT_ITERS, total_latency / POINT_ITERS);
    }

    // Narrow Box Query (adaptive)
    {
        long long total_latency = 0;
        size_t total_items = 0;
        for (int i = 0; i < ITERS; ++i) {
            u64 range = 1024; // Start small
            size_t items_found = 0;
            std::chrono::nanoseconds latency(0);

            while(items_found < MIN_ITEMS) {
                results.clear();
                std::vector<u64> lower = keys[(num_keys / 2 + i * 7) % num_keys];
                std::vector<u64> upper = lower;
                for(u32 d=0; d<dim; ++d) upper[d] = (lower[d] > MAX_COORD - range) ? MAX_COORD : lower[d] + range;
                Box query_box = {lower.data(), upper.data(), dim};

                auto s = std::chrono::high_resolution_clock::now();
                tree.box_query(query_box, results);
                auto e = std::chrono::high_resolution_clock::now();
                latency = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s);

                items_found = results.size();
                if (range > MAX_COORD / 2) { if (items_found == 0) items_found = 1; break; }
                if (items_found < MIN_ITEMS) range *= 2;
            }
            total_latency += latency.count();
            total_items += items_found;
        }
        print_query_stats("Narrow Box Query", total_items / ITERS, total_latency / ITERS);
    }

    // Elongated Box Query (adaptive)
    {
        long long total_latency = 0;
        size_t total_items = 0;
        for (int i = 0; i < ITERS; ++i) {
            u64 long_range = 1024;
            size_t items_found = 0;
            std::chrono::nanoseconds latency(0);

            while(items_found < MIN_ITEMS) {
                results.clear();
                u64 short_range = long_range / 100;
                if (short_range == 0) short_range = 1;
                std::vector<u64> lower = keys[(num_keys / 3 + i * 7) % num_keys];
                std::vector<u64> upper = lower;
                upper[0] = (lower[0] > MAX_COORD - long_range) ? MAX_COORD : lower[0] + long_range;
                for(u32 d=1; d<dim; ++d) upper[d] = (lower[d] > MAX_COORD - short_range) ? MAX_COORD : lower[d] + short_range;
                Box query_box = {lower.data(), upper.data(), dim};

                auto s = std::chrono::high_resolution_clock::now();
                tree.box_query(query_box, results);
                auto e = std::chrono::high_resolution_clock::now();
                latency = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s);

                items_found = results.size();
                if (long_range > MAX_COORD / 2) { if (items_found == 0) items_found = 1; break; }
                if (items_found < MIN_ITEMS) {
                    long_range *= 2;
                }
            }
            total_latency += latency.count();
            total_items += items_found;
        }
        print_query_stats("Elongated Box", total_items / ITERS, total_latency / ITERS);
    }

    // Full Axis Scan (adaptive)
    if (dim > 1) {
        long long total_latency = 0;
        size_t total_items = 0;
        for (int i = 0; i < ITERS; ++i) {
            u64 range = 0; // Start with a perfect slice
            size_t items_found = 0;
            std::chrono::nanoseconds latency(0);

            while(items_found < MIN_ITEMS) {
                results.clear();
                std::vector<u64> lower(dim, 0);
                std::vector<u64> upper(dim, MAX_COORD);
                u64 center = keys[(num_keys/4 + i * 7) % num_keys][0];
                lower[0] = (center > range) ? center - range : 0;
                upper[0] = (center < MAX_COORD - range) ? center + range : MAX_COORD;
                Box query_box = {lower.data(), upper.data(), dim};

                auto s = std::chrono::high_resolution_clock::now();
                tree.box_query(query_box, results);
                auto e = std::chrono::high_resolution_clock::now();
                latency = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s);

                items_found = results.size();
                if (range > MAX_COORD / 4) { if (items_found == 0) items_found = 1; break; }
                if (items_found < MIN_ITEMS) range = (range == 0) ? 1 : range * 2;
            }
            total_latency += latency.count();
            total_items += items_found;
        }
        print_query_stats("Full Axis Scan", total_items / ITERS, total_latency / ITERS);
    }

    // Partial Axis Scan (adaptive)
    if (dim > 1) {
        long long total_latency = 0;
        size_t total_items = 0;
        for (int i = 0; i < ITERS; ++i) {
            u64 range = 1024;
            size_t items_found = 0;
            std::chrono::nanoseconds latency(0);

            while(items_found < MIN_ITEMS) {
                results.clear();
                std::vector<u64> lower(dim, 0);
                std::vector<u64> upper(dim, MAX_COORD);
                lower[0] = keys[(num_keys/5 + i * 7) % num_keys][0];
                upper[0] = lower[0] > MAX_COORD - range ? MAX_COORD : lower[0] + range;
                Box query_box = {lower.data(), upper.data(), dim};

                auto s = std::chrono::high_resolution_clock::now();
                tree.box_query(query_box, results);
                auto e = std::chrono::high_resolution_clock::now();
                latency = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s);

                items_found = results.size();
                if (range > MAX_COORD / 2) { if (items_found == 0) items_found = 1; break; }
                if (items_found < MIN_ITEMS) range *= 2;
            }

            total_latency += latency.count();
            total_items += items_found;
        }
        print_query_stats("Partial Axis Scan", total_items / ITERS, total_latency / ITERS);
    }

    // Cross-Axis Scan (adaptive)
    if (dim > 2) {
        long long total_latency = 0;
        size_t total_items = 0;
        for (int i = 0; i < ITERS; ++i) {
            u64 range = 0;
            size_t items_found = 0;
            std::chrono::nanoseconds latency(0);

            while(items_found < MIN_ITEMS) {
                results.clear();
                std::vector<u64> lower(dim, 0);
                std::vector<u64> upper(dim, MAX_COORD);
                u64 center0 = keys[(num_keys/6 + i * 7) % num_keys][0];
                u64 center1 = keys[(num_keys/6 + i * 7) % num_keys][1];
                lower[0] = (center0 > range) ? center0 - range : 0;
                upper[0] = (center0 < MAX_COORD - range) ? center0 + range : MAX_COORD;
                lower[1] = (center1 > range) ? center1 - range : 0;
                upper[1] = (center1 < MAX_COORD - range) ? center1 + range : MAX_COORD;
                Box query_box = {lower.data(), upper.data(), dim};

                auto s = std::chrono::high_resolution_clock::now();
                tree.box_query(query_box, results);
                auto e = std::chrono::high_resolution_clock::now();
                latency = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s);

                items_found = results.size();
                if (range > MAX_COORD / 4) { if (items_found == 0) items_found = 1; break; }
                if (items_found < MIN_ITEMS) range = (range == 0) ? 1 : range * 2;
            }
            total_latency += latency.count();
            total_items += items_found;
        }
        print_query_stats("Cross-Axis Scan", total_items / ITERS, total_latency / ITERS);
    }

    // Diagonal Sliver (adaptive)
    {
        long long total_latency = 0;
        size_t total_items = 0;
        for (int i = 0; i < ITERS; ++i) {
            u64 range = 1;
            size_t items_found = 0;
            std::chrono::nanoseconds latency(0);

            while(items_found < MIN_ITEMS) {
                results.clear();
                std::vector<u64> lower(dim);
                std::vector<u64> upper(dim);
                u64 center = keys[(num_keys/7 + i*7) % num_keys][0];
                for(u32 d=0; d<dim; ++d) {
                    lower[d] = (center > range) ? center - range : 0;
                    upper[d] = (center < MAX_COORD - range) ? center + range : MAX_COORD;
                }
                Box query_box = {lower.data(), upper.data(), dim};

                auto s = std::chrono::high_resolution_clock::now();
                tree.box_query(query_box, results);
                auto e = std::chrono::high_resolution_clock::now();
                latency = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s);

                items_found = results.size();
                if (range > MAX_COORD / 4) { if (items_found == 0) items_found = 1; break; }
                if (items_found < MIN_ITEMS) range *= 2;
            }
            total_latency += latency.count();
            total_items += items_found;
        }
        print_query_stats("Diagonal Sliver", total_items / ITERS, total_latency / ITERS);
    }

    // High-Bit Query (no change needed, already finds many items)
    {
        results.clear();
        std::vector<u64> lower(dim, 0);
        std::vector<u64> upper(dim, MAX_COORD);
        lower[0] = 1ULL << 63;
        Box query_box = {lower.data(), upper.data(), dim};
        auto s = std::chrono::high_resolution_clock::now();
        tree.box_query(query_box, results);
        auto e = std::chrono::high_resolution_clock::now();
        print_query_stats("High-Bit Query", results.size(), std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count());
    }

    // Checkerboard Query (no change needed, already finds many items)
    {
        long long total_latency = 0;
        size_t total_items = 0;
        const int C_ITERS = 10; // 10x10x... checkerboard
        u64 range = MAX_COORD / (C_ITERS * 2);

        auto s = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < C_ITERS; ++i) {
            for (int j = 0; j < C_ITERS; ++j) {
                if ((i+j) % 2 == 0) {
                    std::vector<ValueType> temp_results;
                    std::vector<u64> lower(dim, 0);
                    std::vector<u64> upper(dim, 0);
                    lower[0] = i * 2 * range;
                    upper[0] = lower[0] + range;
                    if (dim > 1) {
                        lower[1] = j * 2 * range;
                        upper[1] = lower[1] + range;
                    }
                    for(u32 d=2; d<dim; ++d) {
                        upper[d] = MAX_COORD;
                    }
                    Box query_box = {lower.data(), upper.data(), dim};
                    tree.box_query(query_box, temp_results);
                    total_items += temp_results.size();
                }
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        total_latency = std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
        print_query_stats("Checkerboard Query", total_items, total_latency);
    }
}

int main() {
    try {
        run_hyperion_benchmarks<4>(3, 1000000);
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
