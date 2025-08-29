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

// --- Adaptive Radix Tree ---
template<size_t NODE16_SIZE, typename ValueType>
class ART {
    using Node16Type = Node16<NODE16_SIZE>;

    Manager<Node16<NODE16_SIZE>>* nm16_;
    Manager<Node256>& nm256_;
    RecordManager& rm_;

    std::atomic<u32> root_ptr_{0};
    const u32 dim_;

    static inline uint8_t get_key_fragment(const u64* key, int depth) {
        if((size_t)depth >= sizeof(u64) * key[0]) return 0;
        return ((uint8_t*)key)[depth];
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
                if (memcmp(key, existing_rec->coords, dim_ * sizeof(u64)) == 0) return;

                if (leaf_ptr == 0) {
                    u32 new_rec_idx = rm_.allocate_record(key, dim_, value);
                    leaf_ptr = TaggedIndex::make_leaf_idx(new_rec_idx);
                }

                u32 node16_idx = nm16_->allocate_node();
                Node16Type* node = nm16_->get_node(node16_idx);

                int existing_frag = get_key_fragment(existing_rec->coords, depth);
                int new_frag = get_key_fragment(key, depth);

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
                uint8_t frag = get_key_fragment(key, depth);

                uint8_t count = node->count.load(std::memory_order_relaxed);
                for(uint8_t i=0; i<count; ++i) {
                    if(node->keys[i] == frag) {
                        parent_slot = &node->children[i];
                        depth++;
                        goto next_level;
                    }
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
                parent_slot = &node->children[get_key_fragment(key, depth)];
                depth++;
                continue;
            }
            next_level:;
        }
    }

public:
    ART(MemoryContext& ctx, u32 d) : nm256_(*ctx.nm256), rm_(*ctx.rm), dim_(d) {
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
                if(memcmp(key, rec->coords, dim_ * sizeof(u64)) == 0) {
                    ValueType val;
                    memcpy(&val, &rec->value, sizeof(ValueType));
                    return val;
                }
                return std::nullopt;
            }

            uint8_t frag = get_key_fragment(key, depth);
            if(TaggedIndex::is_node16(current_tagged_ptr)) {
                Node16Type* node = nm16_->get_node(TaggedIndex::get_index(current_tagged_ptr));
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
};

template<size_t NODE16_SIZE>
void run_benchmark(int dimensionality, const std::string& key_type) {
    const size_t NUM_KEYS = 2000000;
    std::cout << "\n--- ART Benchmark (DIM=" << dimensionality << ", NODE16_SIZE=" << NODE16_SIZE << ", " << key_type << ") ---" << std::endl;

    using TreeType = ART<NODE16_SIZE, u64>;

    MemoryContext ctx(dimensionality);
    TreeType tree(ctx, dimensionality);

    std::vector<std::vector<u64>> keys(NUM_KEYS, std::vector<u64>(dimensionality));
    std::mt19937_64 rng(12345);

    if (key_type == "Random") for (auto& key : keys) for(int d=0; d<dimensionality; ++d) key[d] = rng();
    else if (key_type == "Sequential") for (size_t i=0; i<NUM_KEYS; ++i) for(int d=0; d<dimensionality; ++d) keys[i][d] = i;
    else if (key_type == "Clustered") {
        u64 prefix = rng();
        for (auto& key : keys) { key[0] = prefix; for(int d=1; d<dimensionality; ++d) key[d] = rng(); }
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) tree.insert(key.data(), (u64)key.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "[ART] Insertion: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << std::endl;

    size_t found_count = 0;
    start = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) if(tree.get(key.data())) found_count++;
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "[ART] Lookup: " << std::fixed << std::setprecision(2) << (double)duration.count() / NUM_KEYS << " ns/op" << " (Found " << found_count << "/" << NUM_KEYS << ")" << std::endl;

    size_t total_mem = ctx.nm16_4->get_mem_usage() + ctx.nm16_16->get_mem_usage() + ctx.nm256->get_mem_usage() + ctx.rm->get_mem_usage();
    std::cout << "[ART] Memory: " << std::fixed << std::setprecision(2) << total_mem / (1024.0 * 1024.0) << " MB (" << (double)total_mem / NUM_KEYS << " bytes/key)" << std::endl;

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
    size_t node_overhead = sizeof(void*) * 2;
    size_t map_mem = NUM_KEYS * (key_size + sizeof(u64) + string_obj_size + node_overhead);
    std::cout << "[std::unordered_map] Memory (est.): " << std::fixed << std::setprecision(2) << map_mem / (1024.0 * 1024.0) << " MB (" << (double)map_mem / NUM_KEYS << " bytes/key)" << std::endl;
}

int main() {
    try {
        std::vector<int> dims_to_test = {1, 8};
        std::vector<std::string> key_types_to_test = {"Random", "Sequential"};
        std::vector<size_t> node16_sizes_to_test = {4, 16};

        for (int dims : dims_to_test) {
            for (const auto& key_type : key_types_to_test) {
                if (dims == 1 && key_type == "Clustered") continue;
                for (size_t n16_size : node16_sizes_to_test) {
                    if (n16_size == 4) run_benchmark<4>(dims, key_type);
                    else if (n16_size == 16) run_benchmark<16>(dims, key_type);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl; return 1;
    }
    return 0;
}
