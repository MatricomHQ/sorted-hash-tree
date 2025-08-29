//
// Self-contained benchmark for HyperionTree
//

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
#include <optional>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


// --- Type Aliases & Constants ---
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;


// =================================================================================================
// --- HyperionTree Implementation ---
// =================================================================================================
namespace Hyperion {

struct Record {
    u64 value;
    u64 coords[];
    static size_t get_size(u32 d) { return sizeof(Record) + sizeof(u64) * d; }
};

namespace TaggedIndex {
    static constexpr u32 NODE256_TAG = 0b00, NODE4_TAG = 0b01, LEAF_TAG = 0b10;
    static constexpr u32 TAG_MASK = 0b11 << 30, INDEX_MASK = ~TAG_MASK;
    inline u32 get_tag(u32 ptr) { return (ptr >> 30); }
    inline u32 get_index(u32 ptr) { return ptr & INDEX_MASK; }
    inline bool is_leaf(u32 ptr) { return get_tag(ptr) == LEAF_TAG; }
    inline bool is_node4(u32 ptr) { return get_tag(ptr) == NODE4_TAG; }
    inline bool is_node256(u32 ptr) { return get_tag(ptr) == NODE256_TAG && ptr != 0; }
    inline u32 make_leaf_idx(u32 rec_idx) { return (rec_idx & INDEX_MASK) | (LEAF_TAG << 30); }
    inline u32 make_node4_idx(u32 node_idx) { return (node_idx & INDEX_MASK) | (NODE4_TAG << 30); }
    inline u32 make_node256_idx(u32 node_idx) { return (node_idx & INDEX_MASK) | (NODE256_TAG << 30); }
};

struct Node4 {
    std::atomic<uint8_t> count{0};
    uint8_t keys[4];
    std::atomic<u32> children[4];
    Node4() { for(size_t i=0; i<4; ++i) { keys[i] = 0; children[i].store(0, std::memory_order_relaxed); }}
    bool is_full() const { return count.load(std::memory_order_relaxed) >= 4; }
};

struct Node256 {
    std::atomic<u32> children[256];
    Node256() { for (size_t i = 0; i < 256; ++i) children[i].store(0, std::memory_order_relaxed); }
};

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
    size_t get_memory_usage() const {
        return (size_t)next_idx_.load(std::memory_order_relaxed) * sizeof(T);
    }
};

class RecordManager {
    std::atomic<u32> next_record_idx_{1};
    uint8_t* record_pool_;
    const size_t record_size_with_coords_;
    static constexpr u32 MAX_RECORDS = 11 * 1024 * 1024;
public:
    RecordManager(u32 d) : record_size_with_coords_(Record::get_size(d)) {
        size_t pool_size = (size_t)MAX_RECORDS * record_size_with_coords_;
        record_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
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
    size_t get_memory_usage() const {
        return (size_t)next_record_idx_.load(std::memory_order_relaxed) * record_size_with_coords_;
    }
};

struct MemoryContext {
    std::unique_ptr<Manager<Node4>> nm4;
    std::unique_ptr<Manager<Node256>> nm256;
    std::unique_ptr<RecordManager> rm;
    MemoryContext(u32 dimensionality) {
        nm4 = std::make_unique<Manager<Node4>>(64 * 1024 * 1024);
        nm256 = std::make_unique<Manager<Node256>>(4 * 1024 * 1024);
        rm = std::make_unique<RecordManager>(dimensionality);
    }
    size_t get_total_memory_usage() const {
        return nm4->get_memory_usage() +
               nm256->get_memory_usage() +
               rm->get_memory_usage();
    }
};

class HyperionTree {
private:
    Manager<Node4>& nm4_;
    Manager<Node256>& nm256_;
    RecordManager& rm_;
    std::atomic<u32> root_ptr_{0};
    const u32 dim_;

    static inline uint8_t get_key_fragment(const u64* key, int depth) {
        const u32 u64_idx = depth / sizeof(u64);
        const u32 byte_shift = (7 - (depth % sizeof(u64))) * 8;
        return (key[u64_idx] >> byte_shift) & 0xFF;
    }

    void insert_recursive(const u64* key, u64 rec_idx, std::atomic<u32>* parent_slot, int depth) {
        u32 leaf_ptr = 0;
        while(true) {
            u32 current_tagged_ptr = parent_slot->load(std::memory_order_acquire);
            if (current_tagged_ptr == 0) {
                if (leaf_ptr == 0) leaf_ptr = TaggedIndex::make_leaf_idx(rec_idx);
                if(parent_slot->compare_exchange_strong(current_tagged_ptr, leaf_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
                else continue;
            }
            if (TaggedIndex::is_leaf(current_tagged_ptr)) {
                u32 existing_rec_idx = TaggedIndex::get_index(current_tagged_ptr);
                Record* existing_rec = rm_.get_record(existing_rec_idx);
                if (memcmp(key, existing_rec->coords, dim_ * sizeof(u64)) == 0) return;

                u32 node4_idx = nm4_.allocate_node();
                Node4* node = nm4_.get_node(node4_idx);
                int existing_frag = get_key_fragment(existing_rec->coords, depth);
                int new_frag = get_key_fragment(key, depth);

                if (existing_frag != new_frag) {
                    node->keys[0] = std::min(existing_frag, new_frag);
                    node->keys[1] = std::max(existing_frag, new_frag);
                    if (leaf_ptr == 0) leaf_ptr = TaggedIndex::make_leaf_idx(rec_idx);
                    node->children[0].store(existing_frag < new_frag ? current_tagged_ptr : leaf_ptr, std::memory_order_relaxed);
                    node->children[1].store(existing_frag < new_frag ? leaf_ptr : current_tagged_ptr, std::memory_order_relaxed);
                    node->count.store(2, std::memory_order_relaxed);
                    if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node4_idx(node4_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                    else continue;
                } else {
                    node->keys[0] = new_frag;
                    node->count.store(1, std::memory_order_relaxed);
                    if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node4_idx(node4_idx), std::memory_order_release, std::memory_order_relaxed)) {
                        insert_recursive(existing_rec->coords, existing_rec_idx, &node->children[0], depth + 1);
                        insert_recursive(key, rec_idx, &node->children[0], depth + 1);
                        return;
                    } else continue;
                }
            }
            if (TaggedIndex::is_node4(current_tagged_ptr)) {
                Node4* node = nm4_.get_node(TaggedIndex::get_index(current_tagged_ptr));
                uint8_t frag = get_key_fragment(key, depth);
                int index = -1;
                uint8_t count = node->count.load(std::memory_order_relaxed);
                for(uint8_t i=0; i<count; ++i) { if(node->keys[i] == frag) { index = i; break; } }

                if (index != -1) {
                    parent_slot = &node->children[index];
                    depth++;
                    continue;
                }

                if (!node->is_full()) {
                    if (leaf_ptr == 0) leaf_ptr = TaggedIndex::make_leaf_idx(rec_idx);
                    uint8_t c = node->count.fetch_add(1, std::memory_order_release);
                    if (c < 4) {
                        node->keys[c] = frag;
                        node->children[c].store(leaf_ptr, std::memory_order_release);
                        return;
                    } else {
                        node->count.fetch_sub(1, std::memory_order_release);
                    }
                }

                u32 new_node256_idx = nm256_.allocate_node();
                Node256* new_node256 = nm256_.get_node(new_node256_idx);
                count = node->count.load(std::memory_order_relaxed);
                for(uint8_t i=0; i<count; ++i) {
                    u32 child_ptr = node->children[i].load(std::memory_order_acquire);
                    if (child_ptr != 0) {
                        new_node256->children[node->keys[i]].store(child_ptr, std::memory_order_relaxed);
                    }
                }
                if (leaf_ptr == 0) leaf_ptr = TaggedIndex::make_leaf_idx(rec_idx);
                new_node256->children[frag].store(leaf_ptr, std::memory_order_relaxed);
                if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node256_idx(new_node256_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                else continue;
            }
            if (TaggedIndex::is_node256(current_tagged_ptr)) {
                Node256* node = nm256_.get_node(TaggedIndex::get_index(current_tagged_ptr));
                parent_slot = &node->children[get_key_fragment(key, depth)];
                depth++;
                continue;
            }
        }
    }

public:
    HyperionTree(MemoryContext& ctx, u32 d) :
        nm4_(*ctx.nm4),
        nm256_(*ctx.nm256),
        rm_(*ctx.rm),
        dim_(d) {}

    void insert(const u64* key, u64 value) {
        u32 rec_idx = rm_.allocate_record(key, dim_, value);
        insert_recursive(key, rec_idx, &root_ptr_, 0);
    }

    std::optional<u64> get(const u64* key) {
        int depth = 0;
        u32 current_tagged_ptr = root_ptr_.load(std::memory_order_acquire);
        while(true) {
            if(current_tagged_ptr == 0) return std::nullopt;
            if(TaggedIndex::is_leaf(current_tagged_ptr)) {
                Record* rec = rm_.get_record(TaggedIndex::get_index(current_tagged_ptr));
                if (memcmp(key, rec->coords, dim_ * sizeof(u64)) == 0) return rec->value;
                return std::nullopt;
            }
            uint8_t frag = get_key_fragment(key, depth);
            if(TaggedIndex::is_node4(current_tagged_ptr)) {
                Node4* node = nm4_.get_node(TaggedIndex::get_index(current_tagged_ptr));
                uint8_t count = node->count.load(std::memory_order_relaxed);
                for(uint8_t i=0; i<count; ++i) {
                    u32 child_ptr = node->children[i].load(std::memory_order_acquire);
                    if (child_ptr != 0 && node->keys[i] == frag) {
                        current_tagged_ptr = child_ptr;
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

} // namespace Hyperion


// =================================================================================================
// --- Benchmark Implementation ---
// =================================================================================================

void print_stats(const std::string& tree_name, const std::string& op_name, long long duration_ns, size_t num_keys, int key_size_bytes, size_t total_mem_bytes = 0) {
    double duration_s = duration_ns / 1e9;
    double ns_per_op = (double)duration_ns / num_keys;
    double mops = num_keys / duration_s / 1e6;
    double total_mb = (double)num_keys * (key_size_bytes + sizeof(u64)) / (1024 * 1024);
    double mb_per_s = total_mb / duration_s;

    std::cout << std::left << std::setw(22) << tree_name
              << std::left << std::setw(20) << op_name
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << ns_per_op << " ns/op |"
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << mops << " Mops/s |"
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << mb_per_s << " MB/s";

    if (total_mem_bytes > 0) {
        double mem_mb = total_mem_bytes / (1024.0 * 1024.0);
        double bytes_per_key = (double)total_mem_bytes / num_keys;
        std::cout << " |" << std::right << std::setw(10) << std::fixed << std::setprecision(2) << mem_mb << " MB |"
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << bytes_per_key << " B/key";
    }
    std::cout << std::endl;
}

void run_benchmarks(const std::string& key_type, int key_size_bytes) {
    const size_t num_keys = 10000000;
    const int dim = key_size_bytes / 8;

    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "--- Benchmark: " << key_type << " keys, " << key_size_bytes << " bytes "
              << "(N=" << num_keys / 1000000 << "M) ---" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::left << std::setw(22) << "Tree"
              << std::left << std::setw(20) << "Operation"
              << std::right << std::setw(15) << "Latency |"
              << std::right << std::setw(15) << "Throughput |"
              << std::right << std::setw(15) << "Bandwidth |"
              << std::right << std::setw(13) << "Total Mem |"
              << std::right << std::setw(13) << "Mem/Key"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    std::vector<std::vector<u64>> keys(num_keys, std::vector<u64>(dim));
    std::mt19937_64 rng(12345);
    if (key_type == "Random") {
        for (auto& key : keys) {
            for (int d = 0; d < dim; ++d) key[d] = rng();
        }
    } else {
        for (size_t i = 0; i < num_keys; ++i) {
            keys[i].assign(dim, 0);
            keys[i][0] = i;
        }
    }

    // --- HyperionTree Benchmark ---
    {
        Hyperion::MemoryContext ctx(dim);
        Hyperion::HyperionTree tree(ctx, dim);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) tree.insert(keys[i].data(), i);
        auto end = std::chrono::high_resolution_clock::now();
        size_t total_memory = ctx.get_total_memory_usage();
        print_stats("HyperionTree", "Insert", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, total_memory);

        size_t found_count = 0;
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            if (tree.get(keys[i].data())) found_count++;
        }
        end = std::chrono::high_resolution_clock::now();
        print_stats("HyperionTree", "Get", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes);
        if (found_count != num_keys) std::cerr << "HyperionTree found " << found_count << "/" << num_keys << std::endl;
    }
}

void run_concurrent_benchmark(const std::string& key_type, int key_size_bytes, int num_threads) {
    const size_t num_keys_per_thread = 1000000;
    const size_t num_keys = num_keys_per_thread * num_threads;
    const int dim = key_size_bytes / 8;

    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "--- Concurrent Benchmark: " << num_threads << " threads, " << key_type << " keys, "
              << key_size_bytes << " bytes (N=" << num_keys / 1000000 << "M) ---" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::left << std::setw(22) << "Tree"
              << std::left << std::setw(20) << "Operation"
              << std::right << std::setw(15) << "Latency |"
              << std::right << std::setw(15) << "Throughput |"
              << std::right << std::setw(15) << "Bandwidth |"
              << std::right << std::setw(13) << "Total Mem |"
              << std::right << std::setw(13) << "Mem/Key"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    std::vector<std::vector<u64>> keys(num_keys, std::vector<u64>(dim));
    std::mt19937_64 rng(12345);
    if (key_type == "Random") {
        for (auto& key : keys) {
            for (int d = 0; d < dim; ++d) key[d] = rng();
        }
    } else { // Sequential
        for (size_t i = 0; i < num_keys; ++i) {
            keys[i].assign(dim, 0);
            keys[i][0] = i;
        }
    }

    Hyperion::MemoryContext ctx(dim);
    Hyperion::HyperionTree tree(ctx, dim);
    std::vector<std::thread> threads;

    // --- Concurrent Insert ---
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t start_idx = i * num_keys_per_thread;
            size_t end_idx = start_idx + num_keys_per_thread;
            for (size_t j = start_idx; j < end_idx; ++j) {
                tree.insert(keys[j].data(), j);
            }
        });
    }
    for (auto& t : threads) t.join();
    auto end = std::chrono::high_resolution_clock::now();
    size_t total_memory = ctx.get_total_memory_usage();
    print_stats("HyperionTree", "Concurrent Insert", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, total_memory);

    threads.clear();

    // --- Concurrent Verify ---
    std::atomic<size_t> verified_count{0};
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t start_idx = i * num_keys_per_thread;
            size_t end_idx = start_idx + num_keys_per_thread;
            for (size_t j = start_idx; j < end_idx; ++j) {
                auto val = tree.get(keys[j].data());
                if (val && *val == j) {
                    verified_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& t : threads) t.join();
    end = std::chrono::high_resolution_clock::now();
    print_stats("HyperionTree", "Concurrent Verify", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes);

    if (verified_count.load() != num_keys) {
        std::cerr << "ERROR: Concurrent verification FAILED. Verified " << verified_count.load() << "/" << num_keys << " keys." << std::endl;
    } else {
        std::cout << "SUCCESS: Concurrent verification PASSED." << std::endl;
    }
}

void run_long_running_noisy_benchmark(const std::string& key_type, int key_size_bytes, int num_threads, int duration_s) {
    const size_t num_keys_per_thread = 4000000; // More than enough for 10s
    const size_t num_keys = num_keys_per_thread * num_threads;
    const int dim = key_size_bytes / 8;

    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "--- Long-Running Noisy Benchmark: " << num_threads << " threads, " << key_type << " keys, "
              << key_size_bytes << " bytes, " << duration_s << "s ---" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    std::vector<std::vector<u64>> keys(num_keys, std::vector<u64>(dim));
    std::mt19937_64 rng(12345);
    // Only random keys make sense for this noisy benchmark
    for (auto& key : keys) {
        for (int d = 0; d < dim; ++d) key[d] = rng();
    }

    Hyperion::MemoryContext ctx(dim);
    Hyperion::HyperionTree tree(ctx, dim);
    std::vector<std::thread> threads;
    std::atomic<bool> time_is_up{false};
    std::vector<size_t> inserts_per_thread(num_threads, 0);
    std::atomic<size_t> total_reads{0};

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t my_inserts = 0;
            size_t my_reads = 0;
            size_t start_idx = i * num_keys_per_thread;
            size_t end_idx = start_idx + num_keys_per_thread;
            std::mt19937_64 thread_rng(i);
            std::uniform_int_distribution<size_t> dist(0, num_keys - 1);

            bool write_pool_exhausted = false;
            while (!time_is_up.load(std::memory_order_acquire)) {
                // Perform one write
                if (!write_pool_exhausted && (start_idx + my_inserts < end_idx)) {
                    try {
                        tree.insert(keys[start_idx + my_inserts].data(), start_idx + my_inserts);
                        my_inserts++;
                    } catch (const std::runtime_error& e) {
                        write_pool_exhausted = true;
                    }
                }

                // Perform one read
                tree.get(keys[dist(thread_rng)].data());
                my_reads++;
            }
            inserts_per_thread[i] = my_inserts;
            total_reads.fetch_add(my_reads, std::memory_order_relaxed);
        });
    }

    std::this_thread::sleep_for(std::chrono::seconds(duration_s));
    time_is_up.store(true, std::memory_order_release);

    for (auto& t : threads) t.join();
    auto end_time = std::chrono::high_resolution_clock::now();

    size_t total_inserts = 0;
    for(size_t count : inserts_per_thread) total_inserts += count;

    double duration = std::chrono::duration<double>(end_time - start_time).count();
    double write_mops = total_inserts / duration / 1e6;
    double read_mops = total_reads.load() / duration / 1e6;

    std::cout << "  Run duration: " << std::fixed << std::setprecision(2) << duration << "s" << std::endl;
    std::cout << "  Write throughput: " << std::fixed << std::setprecision(2) << write_mops << " Mops/s" << std::endl;
    std::cout << "  Read throughput: " << std::fixed << std::setprecision(2) << read_mops << " Mops/s" << std::endl;
    std::cout << "  Total writes: " << total_inserts << std::endl;
    std::cout << "  Total reads: " << total_reads.load() << std::endl;

    // --- Verification ---
    std::cout << "Verifying all " << total_inserts << " inserted keys..." << std::endl;
    std::atomic<size_t> verified_count{0};
    threads.clear();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t start_idx = i * num_keys_per_thread;
            size_t num_thread_inserts = inserts_per_thread[i];
            for (size_t j = 0; j < num_thread_inserts; ++j) {
                size_t key_idx = start_idx + j;
                auto val = tree.get(keys[key_idx].data());
                if (val && *val == key_idx) {
                    verified_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) t.join();

    if (verified_count.load() != total_inserts) {
        std::cerr << "ERROR: Long-running verification FAILED. Verified " << verified_count.load() << "/" << total_inserts << " keys." << std::endl;
    } else {
        std::cout << "SUCCESS: Long-running verification PASSED." << std::endl;
    }
}

int main() {
    try {
        run_benchmarks("Sequential", 8);
        run_benchmarks("Random", 8);
        run_benchmarks("Sequential", 16);
        run_benchmarks("Random", 16);
        run_benchmarks("Sequential", 32);
        run_benchmarks("Random", 32);

        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;

        run_concurrent_benchmark("Random", 8, num_threads);
        run_concurrent_benchmark("Random", 16, num_threads);
        run_concurrent_benchmark("Random", 32, num_threads);

        run_long_running_noisy_benchmark("Random", 16, num_threads, 10);

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
