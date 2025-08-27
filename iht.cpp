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

namespace TaggedPtr {
    static constexpr uint32_t LEAF_TAG = 1U << 31;
    static constexpr uint32_t INDEX_MASK = ~(LEAF_TAG);

    inline bool is_leaf(uint32_t ptr) { return (ptr & LEAF_TAG) != 0; }
    inline bool is_node(uint32_t ptr) { return ptr != 0 && !is_leaf(ptr); }
    inline uint32_t get_leaf_index(uint32_t ptr) { return ptr & INDEX_MASK; }
    inline uint32_t make_leaf_ptr(uint32_t record_idx) { return record_idx | LEAF_TAG; }

    // For node pointers, we need to encode the level (fanout tier)
    static constexpr uint32_t LEVEL_SHIFT = 28;
    static constexpr uint32_t LEVEL_MASK = 0b111 << LEVEL_SHIFT; // 3 bits for level (0-6)
    static constexpr uint32_t NODE_INDEX_MASK = ~(LEAF_TAG | LEVEL_MASK);

    inline uint32_t make_node_ptr(uint32_t block_idx, int level) {
        return (block_idx & NODE_INDEX_MASK) | (level << LEVEL_SHIFT);
    }
    inline int get_level(uint32_t ptr) { return (ptr & LEVEL_MASK) >> LEVEL_SHIFT; }
    inline uint32_t get_node_index(uint32_t ptr) { return ptr & NODE_INDEX_MASK; }
};

// --- IHT Constants & Block Allocator ---

namespace IHT {
    static constexpr int NUM_LEVELS = 8;
    static constexpr int FANOUT_SEQUENCE[NUM_LEVELS] = {256, 256, 256, 256, 256, 256, 256, 256};
    static constexpr int BITS_PER_LEVEL[NUM_LEVELS] = {8, 8, 8, 8, 8, 8, 8, 8};
    static constexpr int BITS_BEFORE_LEVEL[NUM_LEVELS] = {0, 8, 16, 24, 32, 40, 48, 56};
};

class BlockAllocator {
    std::vector<uint8_t*> block_pools_;
    std::vector<std::atomic<uint32_t>> next_block_idxs_;
    std::vector<size_t> block_sizes_bytes_;
    static constexpr uint32_t POOL_SIZE_IN_BLOCKS = 1 * 1024 * 1024;

public:
    BlockAllocator() : next_block_idxs_(IHT::NUM_LEVELS) {
        block_pools_.resize(IHT::NUM_LEVELS);
        block_sizes_bytes_.resize(IHT::NUM_LEVELS);

        for (int i = 0; i < IHT::NUM_LEVELS; ++i) {
            size_t fanout = IHT::FANOUT_SEQUENCE[i];
            size_t block_size = fanout * sizeof(std::atomic<uint32_t>);
            block_sizes_bytes_[i] = block_size;

            size_t pool_size_bytes = POOL_SIZE_IN_BLOCKS * block_size;
            block_pools_[i] = (uint8_t*)mmap(nullptr, pool_size_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (block_pools_[i] == MAP_FAILED) throw std::runtime_error("mmap failed for block pool");

            next_block_idxs_[i].store(1, std::memory_order_relaxed); // Start at 1, 0 is invalid
        }
    }

    ~BlockAllocator() {
        for (int i = 0; i < IHT::NUM_LEVELS; ++i) {
            size_t pool_size_bytes = POOL_SIZE_IN_BLOCKS * block_sizes_bytes_[i];
            munmap(block_pools_[i], pool_size_bytes);
        }
    }

    uint32_t allocate_block(int level) {
        uint32_t block_idx = next_block_idxs_[level].fetch_add(1, std::memory_order_relaxed);
        if (block_idx >= POOL_SIZE_IN_BLOCKS) {
            throw std::runtime_error("Block pool exhausted for level " + std::to_string(level));
        }
        // Zero-initialize the block
        std::atomic<uint32_t>* block_ptr = get_block_ptr(level, block_idx);
        std::memset(block_ptr, 0, block_sizes_bytes_[level]);
        return block_idx;
    }

    std::atomic<uint32_t>* get_block_ptr(int level, uint32_t block_idx) const {
        return reinterpret_cast<std::atomic<uint32_t>*>(block_pools_[level] + block_idx * block_sizes_bytes_[level]);
    }

    std::atomic<uint32_t>* get_block_ptr(uint32_t tagged_node_ptr) const {
        int level = TaggedPtr::get_level(tagged_node_ptr);
        uint32_t index = TaggedPtr::get_node_index(tagged_node_ptr);
        return get_block_ptr(level, index);
    }

    size_t get_mem_usage() const {
        size_t total_bytes = 0;
        for (int i = 0; i < IHT::NUM_LEVELS; ++i) {
            total_bytes += (next_block_idxs_[i].load() - 1) * block_sizes_bytes_[i];
        }
        return total_bytes;
    }

    std::pair<size_t, size_t> get_density_stats() const {
        size_t total_slots = 0;
        size_t filled_slots = 0;
        for (int i = 0; i < IHT::NUM_LEVELS; ++i) {
            size_t fanout = IHT::FANOUT_SEQUENCE[i];
            uint32_t allocated_blocks = next_block_idxs_[i].load() - 1;
            total_slots += allocated_blocks * fanout;

            for (uint32_t j = 1; j <= allocated_blocks; ++j) {
                std::atomic<uint32_t>* block = get_block_ptr(i, j);
                for (size_t k = 0; k < fanout; ++k) {
                    if (block[k].load(std::memory_order_relaxed) != 0) {
                        filled_slots++;
                    }
                }
            }
        }
        return {filled_slots, total_slots};
    }
};

class RecordManager; // Forward declaration

class ImplicitHierarchicalTrie {
private:
    std::atomic<uint32_t> root_ptr_;
    BlockAllocator& block_allocator_;
    RecordManager& record_manager_;
    const uint32_t dim_;

public:
    struct TraversalContext;

    ImplicitHierarchicalTrie(BlockAllocator& ba, RecordManager& rm, uint32_t d);

    TraversalContext find_slot_context(const uint64_t* key_coords);
    bool get(const uint64_t* coords);
    void insert(const uint64_t* coords, uint64_t value);
};

struct ImplicitHierarchicalTrie::TraversalContext {
    std::atomic<uint32_t>* final_slot;
    int chunk_depth;
    int level;
    uint64_t chunk;
};

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

// --- IHT Method Implementations ---

ImplicitHierarchicalTrie::ImplicitHierarchicalTrie(BlockAllocator& ba, RecordManager& rm, uint32_t d)
    : root_ptr_(0), block_allocator_(ba), record_manager_(rm), dim_(d) {}

ImplicitHierarchicalTrie::TraversalContext ImplicitHierarchicalTrie::find_slot_context(const uint64_t* key_coords) {
    std::atomic<uint32_t>* current_slot = &root_ptr_;
    int chunk_depth = 0;
    int num_chunks = dim_;

restart_chunk:
    uint64_t chunk = key_coords[chunk_depth];
    int level = 0;

    while (level < IHT::NUM_LEVELS) {
        uint32_t ptr = current_slot->load(std::memory_order_acquire);

        if (!TaggedPtr::is_node(ptr)) {
            return {current_slot, chunk_depth, level, chunk};
        }

        int node_level = TaggedPtr::get_level(ptr);
        if (node_level != level) {
            return {current_slot, chunk_depth, level, chunk};
        }

        std::atomic<uint32_t>* block_addr = block_allocator_.get_block_ptr(ptr);
        int fanout = IHT::FANOUT_SEQUENCE[level];
        int bits_consumed = IHT::BITS_BEFORE_LEVEL[level];
        int bits_for_this_level = IHT::BITS_PER_LEVEL[level];
        int shift = 64 - bits_consumed - bits_for_this_level;
        int index = (chunk >> shift) & (fanout - 1);

        current_slot = &block_addr[index];
        level++;
    }

    uint32_t final_ptr = current_slot->load(std::memory_order_acquire);
    if (TaggedPtr::is_node(final_ptr)) {
        chunk_depth++;
        if (chunk_depth < num_chunks) {
            current_slot = block_allocator_.get_block_ptr(final_ptr);
            goto restart_chunk;
        }
    }

    return {current_slot, chunk_depth, level, chunk};
}

bool ImplicitHierarchicalTrie::get(const uint64_t* coords) {
    TraversalContext ctx = find_slot_context(coords);
    uint32_t ptr = ctx.final_slot->load(std::memory_order_acquire);

    if (TaggedPtr::is_leaf(ptr)) {
        uint32_t rec_idx = TaggedPtr::get_leaf_index(ptr);
        Record* rec = record_manager_.get_record(rec_idx);
        return memcmp(rec->coords, coords, dim_ * sizeof(uint64_t)) == 0;
    }
    return false;
}

void ImplicitHierarchicalTrie::insert(const uint64_t* coords, uint64_t value) {
restart:
    TraversalContext ctx = find_slot_context(coords);
    std::atomic<uint32_t>* slot = ctx.final_slot;
    uint32_t current_val = slot->load(std::memory_order_acquire);

    if (current_val == 0) { // Case 1: Empty slot
        uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
        uint32_t new_leaf_ptr = TaggedPtr::make_leaf_ptr(new_rec_idx);
        if (slot->compare_exchange_strong(current_val, new_leaf_ptr, std::memory_order_release, std::memory_order_relaxed)) {
            return;
        }
        goto restart;
    }

    if (TaggedPtr::is_leaf(current_val)) { // Case 2: Leaf in slot
        uint32_t old_rec_idx = TaggedPtr::get_leaf_index(current_val);
        Record* old_rec = record_manager_.get_record(old_rec_idx);

        if (memcmp(old_rec->coords, coords, dim_ * sizeof(uint64_t)) == 0) {
            return; // Key already exists
        }

        // Conflict detected. Start the expansion process.
        int conflict_level = ctx.level;

        // Allocate the first new block. The pointer to this block is the one we will
        // ultimately try to CAS into the trie.
        uint32_t head_block_idx = block_allocator_.allocate_block(conflict_level);
        uint32_t node_ptr_to_install = TaggedPtr::make_node_ptr(head_block_idx, conflict_level);

        std::atomic<uint32_t>* current_block_ptr = block_allocator_.get_block_ptr(conflict_level, head_block_idx);
        uint64_t old_key_chunk = old_rec->coords[ctx.chunk_depth];

        // Loop to create a chain of nodes as long as the keys' indices collide.
        while (true) {
            int bits_consumed = IHT::BITS_BEFORE_LEVEL[conflict_level];
            int bits_for_this_level = IHT::BITS_PER_LEVEL[conflict_level];
            int shift = 64 - bits_consumed - bits_for_this_level;
            int fanout = IHT::FANOUT_SEQUENCE[conflict_level];

            int new_key_idx = (ctx.chunk >> shift) & (fanout - 1);
            int old_key_idx = (old_key_chunk >> shift) & (fanout - 1);

            if (new_key_idx != old_key_idx) {
                // Found a level where the keys diverge. Populate this node and we're done expanding.
                uint32_t new_rec_idx = record_manager_.allocate_record(coords, value);
                uint32_t new_leaf_ptr = TaggedPtr::make_leaf_ptr(new_rec_idx);
                current_block_ptr[new_key_idx].store(new_leaf_ptr, std::memory_order_relaxed);
                current_block_ptr[old_key_idx].store(current_val, std::memory_order_relaxed);
                break; // Exit the expansion loop
            }

            // Keys still collide at this level. We must go deeper.
            conflict_level++;
            if (conflict_level >= IHT::NUM_LEVELS) {
                // This should only happen if the 64-bit chunks are identical.
                // A full implementation would need to handle this by starting a new sub-trie
                // for the next chunk (fractal expansion). For this benchmark, we'll error out.
                throw std::runtime_error("Key chunks are identical, requires fractal expansion not yet implemented.");
            }

            // Allocate a child node for the next level and link to it from the current node.
            uint32_t child_block_idx = block_allocator_.allocate_block(conflict_level);
            uint32_t child_node_ptr = TaggedPtr::make_node_ptr(child_block_idx, conflict_level);
            current_block_ptr[new_key_idx].store(child_node_ptr, std::memory_order_relaxed);

            // The child block is now the one we need to populate in the next iteration.
            current_block_ptr = block_allocator_.get_block_ptr(conflict_level, child_block_idx);
        }

        // After the expansion loop, try to CAS the original slot with the head of our new node chain.
        if (slot->compare_exchange_strong(current_val, node_ptr_to_install, std::memory_order_release, std::memory_order_relaxed)) {
            return; // Success
        }
        goto restart; // Contention
    }
}

const int MAX_DIMS = 8;
struct Key { uint64_t coords[MAX_DIMS]; };


void run_benchmark(size_t num_keys, int num_threads, int dimensionality, const std::string& key_type) {
    std::cout << "\n--- Benchmark: " << dimensionality << "D Implicit Hierarchical Trie (" << (dimensionality*8) << " bytes) ---" << std::endl;
    std::cout << "--- Configuration: " << num_keys << " keys, " << key_type << " distribution, " << num_threads << " threads ---" << std::endl;

    ValueStore vs;
    BlockAllocator ba;
    RecordManager rm(&vs, dimensionality);
    ImplicitHierarchicalTrie tree(ba, rm, dimensionality);

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
        uint64_t counter = 0;
        for(size_t i = 0; i < num_keys; ++i) {
            keys[i].coords[0] = ++counter;
            miss_keys[i].coords[0] = counter + num_keys;
            for (int d = 1; d < dimensionality; ++d) {
                keys[i].coords[d] = keys[i].coords[0];
                miss_keys[i].coords[d] = miss_keys[i].coords[0];
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
            for (size_t j = start; j < end; ++j) {
                tree.insert(keys[j].coords, j);
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[IHT] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/insert" << std::endl;

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
    std::cout << "[IHT] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;
    if (found_count == num_keys) {
        std::cout << "  Verification: SUCCESS" << std::endl;
    } else {
        std::cout << "  Verification: FAILED (Found " << found_count << "/" << num_keys << ")" << std::endl;
    }

    std::cout << "\n--- MISS LATENCY (LOOKUP) ---" << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_keys; ++i) {
        tree.get(miss_keys[i].coords);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "[IHT] Avg Latency: " << std::fixed << std::setprecision(2) << (double)duration.count() / num_keys << " ns/get" << std::endl;

    // --- Unordered Map Benchmark ---
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

    // --- MEMORY USAGE & DENSITY ---
    std::cout << "\n--- IHT MEMORY USAGE & DENSITY ---" << std::endl;
    size_t block_mem = ba.get_mem_usage();
    size_t record_mem = rm.get_mem_usage();
    size_t total_mem = block_mem + record_mem;
    std::cout << "Block Allocator Memory: " << std::fixed << std::setprecision(2) << block_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Record Manager Memory:  " << std::fixed << std::setprecision(2) << record_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total Memory (IHT):     " << std::fixed << std::setprecision(2) << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Average Bytes per Key:  " << std::fixed << std::setprecision(2) << (double)total_mem / num_keys << " bytes/key" << std::endl;

    auto density_stats = ba.get_density_stats();
    if (density_stats.second > 0) {
        double density = (double)density_stats.first / density_stats.second * 100.0;
        std::cout << "Pointer Block Density:  " << std::fixed << std::setprecision(2) << density << "% (" << density_stats.first << " / " << density_stats.second << " slots)" << std::endl;
    } else {
        std::cout << "Pointer Block Density:  N/A (no blocks allocated)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        int num_threads = std::thread::hardware_concurrency();
        if (argc > 1) {
            num_threads = std::stoi(argv[1]);
        }
        std::cout << "--- Running with " << num_threads << " threads ---" << std::endl;

        const size_t LARGE_KEY_COUNT = 1000000;

        std::vector<int> dims_to_test = {1, 2, 8};
        std::vector<std::string> key_types_to_test = {"Random", "Sequential"};

        for (int dims : dims_to_test) {
            for (const auto& key_type : key_types_to_test) {
                 run_benchmark(LARGE_KEY_COUNT, num_threads, dims, key_type);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
