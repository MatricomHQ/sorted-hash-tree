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
template<size_t FANOUT> struct RadixNode;

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

template<size_t FANOUT>
struct RadixNode {
    uint32_t test_nibble_idx;
    uint32_t representative_record_idx;
    std::atomic<uint32_t> children[FANOUT];

    RadixNode() : test_nibble_idx(0), representative_record_idx(0) {
        for(size_t i = 0; i < FANOUT; ++i) children[i].store(0, std::memory_order_relaxed);
    }
};

class NodeManager {
private:
    std::atomic<uint32_t> next_node_idx_;
    uint8_t* node_pool_;
    static constexpr uint32_t MAX_NODES = 32 * 1024 * 1024;
    static constexpr size_t NODE_SIZE = sizeof(RadixNode<16>);
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
    template<size_t FANOUT> RadixNode<FANOUT>* get_node(uint32_t idx) { return reinterpret_cast<RadixNode<FANOUT>*>(node_pool_ + (size_t)idx * NODE_SIZE); }
    template<size_t FANOUT> uint32_t allocate_node() {
        if (cache_ptr_ < 0) refill_cache();
        uint32_t new_idx = node_cache_[cache_ptr_--];
        new (get_node<FANOUT>(new_idx)) RadixNode<FANOUT>();
        return new_idx;
    }
    size_t get_mem_usage() const { return (size_t)next_node_idx_.load() * NODE_SIZE; }
};
thread_local uint32_t NodeManager::node_cache_[NodeManager::THREAD_CACHE_SIZE];
thread_local int NodeManager::cache_ptr_ = -1;

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
    static constexpr size_t FANOUT = 16;
    NodeManager& node_manager_;
    RecordManager& record_manager_;
    std::atomic<uint32_t>& root_ptr_;
    const uint32_t dim_;

    static inline int get_nibble(const uint64_t* coords, int nibble_idx, int d) {
        int max_nibbles = d * 16;
        if (nibble_idx < 0 || nibble_idx >= max_nibbles) return 0;
        int dim_idx = nibble_idx / 16;
        int nibble_in_dim = nibble_idx % 16;
        return (coords[dim_idx] >> (60 - (nibble_in_dim * 4))) & 0x0F;
    }

    static int find_first_differing_nibble(const uint64_t* k1, const uint64_t* k2, int d, int start_nibble = 0) {
        int max_nibbles = d * 16;
        for (int i = start_nibble; i < max_nibbles; ++i) {
            if (get_nibble(k1, i, d) != get_nibble(k2, i, d)) return i;
        }
        return -1;
    }

public:
    KeyValueRadixTree(NodeManager& nm, RecordManager& rm, std::atomic<uint32_t>& root, uint32_t d)
        : node_manager_(nm), record_manager_(rm), root_ptr_(root), dim_(d) {}

    void insert(const uint64_t* coords, uint64_t value) {
    restart:
        std::atomic<uint32_t>* parent_slot = &root_ptr_;
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);

        while(TaggedIndex::is_node(current_idx)) {
            RadixNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_idx));
            Record* rep_rec = record_manager_.get_record(node->representative_record_idx);

            int diff_idx = find_first_differing_nibble(coords, rep_rec->coords, dim_);

            if (diff_idx != -1 && (uint32_t)diff_idx < node->test_nibble_idx) {
                uint32_t new_node_idx = node_manager_.allocate_node<FANOUT>();
                RadixNode<FANOUT>* new_node = node_manager_.get_node<FANOUT>(new_node_idx);
                uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
                new_node->test_nibble_idx = diff_idx;
                new_node->representative_record_idx = new_rec_idx;

                int new_key_nibble = get_nibble(coords, diff_idx, dim_);
                int existing_key_nibble = get_nibble(rep_rec->coords, diff_idx, dim_);

                new_node->children[new_key_nibble].store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);
                new_node->children[existing_key_nibble].store(current_idx, std::memory_order_relaxed);

                if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart;
            }
            int nibble = get_nibble(coords, node->test_nibble_idx, dim_);
            parent_slot = &node->children[nibble];
            current_idx = parent_slot->load(std::memory_order_acquire);
        }

        if (current_idx == 0) {
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart;
        }

        if (TaggedIndex::is_leaf(current_idx)) {
            Record* existing_rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));
            int diff_idx = find_first_differing_nibble(coords, existing_rec->coords, dim_);
            if (diff_idx == -1) return;

            uint32_t new_node_idx = node_manager_.allocate_node<FANOUT>();
            RadixNode<FANOUT>* new_node = node_manager_.get_node<FANOUT>(new_node_idx);
            uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
            new_node->test_nibble_idx = diff_idx;
            new_node->representative_record_idx = new_rec_idx;

            int new_key_nibble = get_nibble(coords, diff_idx, dim_);
            int existing_key_nibble = get_nibble(existing_rec->coords, diff_idx, dim_);

            new_node->children[new_key_nibble].store(TaggedIndex::make_leaf_idx(new_rec_idx), std::memory_order_relaxed);
            new_node->children[existing_key_nibble].store(current_idx, std::memory_order_relaxed);

            if (parent_slot->compare_exchange_strong(current_idx, TaggedIndex::make_node_idx(new_node_idx), std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart;
        }
    }

    bool get(const uint64_t* coords) {
        uint32_t current_idx = root_ptr_.load(std::memory_order_acquire);
        while (TaggedIndex::is_node(current_idx)) {
            RadixNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_idx));
            int nibble = get_nibble(coords, node->test_nibble_idx, dim_);
            current_idx = node->children[nibble].load(std::memory_order_acquire);
        }
        if (TaggedIndex::is_leaf(current_idx)) {
            Record* rec = record_manager_.get_record(TaggedIndex::get_index(current_idx));
            if (!rec) return false;
            return memcmp(rec->coords, coords, dim_ * sizeof(uint64_t)) == 0;
        }
        return false;
    }
};

const int MAX_DIMS = 8;
struct Key { uint64_t coords[MAX_DIMS]; };

void run_benchmark(size_t num_keys, int num_threads, int dimensionality, const std::string& key_type) {
    std::cout << "\n--- Benchmark: " << dimensionality << "D Key-Value Radix Tree (" << (dimensionality*8) << " bytes) ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution ---" << std::endl;

    ValueStore vs;
    NodeManager nm;
    RecordManager rm(&vs, dimensionality);
    std::atomic<uint32_t> root_ptr(0);
    KeyValueRadixTree tree(nm, rm, root_ptr, dimensionality);

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
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) { tree.get(keys[i].coords); }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[KeyValueRadixTree] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;

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
}

int main() {
    try {
        const int NUM_THREADS = std::thread::hardware_concurrency();
        const size_t LARGE_KEY_COUNT = 1000000;

        std::vector<int> dims_to_test = {1, 3, 8};
        std::vector<std::string> key_types_to_test = {"Random", "Sequential"};

        for (int dims : dims_to_test) {
            for (const auto& key_type : key_types_to_test) {
                run_benchmark(LARGE_KEY_COUNT, NUM_THREADS, dims, key_type);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
