//
// Self-contained benchmark for comparing HyperionTree vs StaxTree vs std::unordered_map
//
// This file combines:
// 1. The user-provided HyperionTree implementation.
// 2. An extracted, minimal version of the StaxDimensionStore and its dependencies from the project.
// 3. A new benchmark runner to perform a head-to-head comparison for key-value workloads.
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

#if defined(_MSC_VER)
#define STAX_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STAX_ALWAYS_INLINE __attribute__((always_inline))
#else
#define STAX_ALWAYS_INLINE inline
#endif


// =================================================================================================
// --- StaxTree (StaxDimensionStore) Implementation ---
// This section contains the necessary components for StaxDimensionStore, adapted for this benchmark.
// =================================================================================================
namespace Stax {

// --- Stax Memory Management ---
constexpr uint64_t DB_MAX_VIRTUAL_SIZE = 20ULL * 1024 * 1024 * 1024; // 20 GB for benchmark

struct FileHeader {
    std::atomic<uint64_t> global_alloc_offset;
    std::atomic<uint64_t> root_ptr;
};

class StaxAllocator {
private:
    FileHeader* file_header_ = nullptr;
    uint8_t* mmap_base_addr_ = nullptr;
public:
    StaxAllocator(FileHeader* file_header, uint8_t* mmap_base_addr)
        : file_header_(file_header), mmap_base_addr_(mmap_base_addr) {}

    uint64_t allocate(size_t size, size_t alignment = 8) {
        if (!file_header_) throw std::runtime_error("Cannot allocate chunk: file header is null.");
        if ((alignment & (alignment - 1)) != 0) throw std::invalid_argument("Alignment must be a power of two.");
        const uint64_t alignment_mask = alignment - 1;
        uint64_t current_offset = file_header_->global_alloc_offset.load(std::memory_order_relaxed);
        while (true) {
            uint64_t aligned_offset = (current_offset + alignment_mask) & ~alignment_mask;
            uint64_t next_offset = aligned_offset + size;
            if (next_offset > DB_MAX_VIRTUAL_SIZE) throw std::runtime_error("Database out of space.");
            if (file_header_->global_alloc_offset.compare_exchange_weak(current_offset, next_offset, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return aligned_offset;
            }
        }
    }
    template<typename T>
    T* get_ptr(uint64_t offset) const {
        if (offset == 0) return nullptr;
        return reinterpret_cast<T*>(mmap_base_addr_ + offset);
    }
};

class ThreadLocalAllocator {
private:
    StaxAllocator& global_allocator_;
    uint64_t arena_offset_ = 0;
    uint64_t current_alloc_ptr_ = 0;
    uint64_t arena_end_ptr_ = 0;
    static constexpr size_t ARENA_SIZE = 64 * 1024;
    void request_new_arena() {
        arena_offset_ = global_allocator_.allocate(ARENA_SIZE, ARENA_SIZE);
        current_alloc_ptr_ = arena_offset_;
        arena_end_ptr_ = arena_offset_ + ARENA_SIZE;
    }
public:
    ThreadLocalAllocator(StaxAllocator& global_allocator) : global_allocator_(global_allocator) { request_new_arena(); }
    uint64_t allocate(size_t size, size_t alignment = 8) {
        const uint64_t alignment_mask = alignment - 1;
        uint64_t aligned_ptr = (current_alloc_ptr_ + alignment_mask) & ~alignment_mask;
        if (aligned_ptr + size > arena_end_ptr_) {
            if (size > ARENA_SIZE) return global_allocator_.allocate(size, alignment);
            request_new_arena();
            aligned_ptr = (current_alloc_ptr_ + alignment_mask) & ~alignment_mask;
            if (aligned_ptr + size > arena_end_ptr_) return global_allocator_.allocate(size, alignment);
        }
        current_alloc_ptr_ = aligned_ptr + size;
        return aligned_ptr;
    }
};

// --- Stax Core Data Structures ---
struct StaxRecord {
    uint32_t dimensionality;
    uint32_t value_size;
    uint64_t coords[];
    static size_t required_size(uint32_t dim, uint32_t value_size) { return sizeof(StaxRecord) + sizeof(uint64_t) * dim + value_size; }
    void* get_value_ptr() { return (void*)(coords + dimensionality); }
};

template<size_t FANOUT> struct UniformNode {
    std::atomic<uint64_t> children[FANOUT];
    uint64_t representative_leaf_offset;
    UniformNode() : representative_leaf_offset(0) {
        for(size_t i = 0; i < FANOUT; ++i) children[i].store(0, std::memory_order_relaxed);
    }
};

struct FatPointer {
    enum Tag : uint64_t { EMPTY = 0, LEAF_PTR = 1, NODE_PTR = 2 };
    static constexpr uint64_t TAG_BITS = 3, DIM_BITS = 5, CHUNK_BITS = 8, OFFSET_BITS = 48;
    static constexpr uint64_t TAG_SHIFT = 0, DIM_SHIFT = TAG_BITS, CHUNK_SHIFT = DIM_SHIFT + DIM_BITS, OFFSET_SHIFT = CHUNK_SHIFT + CHUNK_BITS;
    static constexpr uint64_t TAG_MASK = ((1ULL << TAG_BITS) - 1) << TAG_SHIFT;
    static constexpr uint64_t DIM_MASK = ((1ULL << DIM_BITS) - 1) << DIM_SHIFT;
    static constexpr uint64_t CHUNK_MASK = ((1ULL << CHUNK_BITS) - 1) << CHUNK_SHIFT;
    static constexpr uint64_t OFFSET_MASK = ((1ULL << OFFSET_BITS) - 1) << OFFSET_SHIFT;
    STAX_ALWAYS_INLINE static uint64_t encode(uint64_t offset, uint32_t test_idx, Tag tag) {
        return ((offset << OFFSET_SHIFT) & OFFSET_MASK) |
               ((static_cast<uint64_t>(test_idx) << DIM_SHIFT) & (CHUNK_MASK | DIM_MASK)) |
               (tag & TAG_MASK);
    }
    STAX_ALWAYS_INLINE static Tag get_tag(uint64_t ptr) { return static_cast<Tag>(ptr & TAG_MASK); }
    STAX_ALWAYS_INLINE static uint64_t get_offset(uint64_t ptr) { return (ptr & OFFSET_MASK) >> OFFSET_SHIFT; }
    STAX_ALWAYS_INLINE static uint32_t get_test_idx(uint64_t ptr) { return (ptr & (CHUNK_MASK | DIM_MASK)) >> DIM_SHIFT; }
};

enum class SplittingHeuristic { ADAPTIVE, LEXICOGRAPHICAL };

template<size_t FANOUT>
class StaxDimensionStore {
private:
    StaxAllocator &allocator_;
    std::atomic<uint64_t> &root_ptr_;
    uint32_t dimensionality_;
    SplittingHeuristic heuristic_;

public:
    struct TreeStats {
        uint64_t node_count = 0;
        uint64_t total_children = 0;
        uint64_t leaf_count = 0;
    };
    StaxDimensionStore(StaxAllocator &allocator, std::atomic<uint64_t> &root_ref, uint32_t dimensionality, SplittingHeuristic heuristic)
        : allocator_(allocator), root_ptr_(root_ref), dimensionality_(dimensionality), heuristic_(heuristic) {}

    void insert(ThreadLocalAllocator& local_alloc, const uint64_t* coords, const void* value_payload, size_t value_size);
    StaxRecord* get(const uint64_t* coords) const;
    TreeStats get_stats() const {
        TreeStats stats;
        get_stats_recursive(root_ptr_.load(std::memory_order_relaxed), stats);
        return stats;
    }
private:
    void get_stats_recursive(uint64_t current_ptr, TreeStats& stats) const {
        if (current_ptr == 0) return;
        FatPointer::Tag tag = FatPointer::get_tag(current_ptr);
        if (tag == FatPointer::LEAF_PTR) {
            stats.leaf_count++;
            return;
        }
        if (tag == FatPointer::NODE_PTR) {
            stats.node_count++;
            UniformNode<FANOUT>* node = allocator_.get_ptr<UniformNode<FANOUT>>(FatPointer::get_offset(current_ptr));
            for (size_t i = 0; i < FANOUT; ++i) {
                uint64_t child_ptr = node->children[i].load(std::memory_order_relaxed);
                if (child_ptr != 0) {
                    stats.total_children++;
                    get_stats_recursive(child_ptr, stats);
                }
            }
        }
    }
    uint64_t allocate_new_record(ThreadLocalAllocator& local_alloc, const uint64_t* coords, const void* value_payload, size_t value_size);
    static int find_first_differing_nibble_linear(const uint64_t* coords1, const uint64_t* coords2, uint32_t dim);
    STAX_ALWAYS_INLINE static int get_nibble_from_coords(const uint64_t* coords, uint32_t nibble_idx);
};

template<size_t FANOUT>
STAX_ALWAYS_INLINE int StaxDimensionStore<FANOUT>::get_nibble_from_coords(const uint64_t* coords, uint32_t nibble_idx) {
    const size_t byte_idx = nibble_idx / 2;
    const uint8_t byte = reinterpret_cast<const uint8_t*>(coords)[byte_idx];
    const uint32_t shift_amount = (1 - (nibble_idx & 1)) * 4;
    return (byte >> shift_amount) & 0x0F;
}

template<size_t FANOUT>
int StaxDimensionStore<FANOUT>::find_first_differing_nibble_linear(const uint64_t* coords1, const uint64_t* coords2, uint32_t dim) {
    const char* k1 = reinterpret_cast<const char*>(coords1);
    const char* k2 = reinterpret_cast<const char*>(coords2);
    const size_t len = dim * sizeof(uint64_t);
    size_t byte_idx = 0;
    while (byte_idx < len) {
        if (k1[byte_idx] != k2[byte_idx]) {
            uint8_t b1 = k1[byte_idx]; uint8_t b2 = k2[byte_idx];
            return byte_idx * 2 + ((b1 >> 4) != (b2 >> 4) ? 0 : 1);
        }
        byte_idx++;
    }
    return -1;
}

template<size_t FANOUT>
uint64_t StaxDimensionStore<FANOUT>::allocate_new_record(ThreadLocalAllocator& local_alloc, const uint64_t* coords, const void* value_payload, size_t value_size) {
    size_t record_size = StaxRecord::required_size(dimensionality_, value_size);
    uint64_t offset = local_alloc.allocate(record_size);
    StaxRecord* new_rec = allocator_.get_ptr<StaxRecord>(offset);
    new_rec->dimensionality = dimensionality_;
    new_rec->value_size = value_size;
    memcpy(new_rec->coords, coords, sizeof(uint64_t) * dimensionality_);
    memcpy(new_rec->get_value_ptr(), value_payload, value_size);
    return offset;
}

template<size_t FANOUT>
void StaxDimensionStore<FANOUT>::insert(ThreadLocalAllocator& local_alloc, const uint64_t* coords, const void* value_payload, size_t value_size) {
restart_lex:
    std::atomic<uint64_t>* cas_slot = &root_ptr_;
    uint64_t current_ptr = root_ptr_.load(std::memory_order_acquire);
    while (true) {
        if (current_ptr == 0) {
            uint64_t new_record_offset = allocate_new_record(local_alloc, coords, value_payload, value_size);
            uint64_t new_leaf_ptr = FatPointer::encode(new_record_offset, 0, FatPointer::LEAF_PTR);
            uint64_t expected_ptr = 0;
            if (cas_slot->compare_exchange_strong(expected_ptr, new_leaf_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart_lex;
        }
        FatPointer::Tag tag = FatPointer::get_tag(current_ptr);
        if (tag == FatPointer::LEAF_PTR) {
            uint64_t existing_record_offset = FatPointer::get_offset(current_ptr);
            StaxRecord* existing_record = allocator_.get_ptr<StaxRecord>(existing_record_offset);
            int test_idx = find_first_differing_nibble_linear(coords, existing_record->coords, dimensionality_);
            if (test_idx == -1) return;
            uint64_t new_node_offset = local_alloc.allocate(sizeof(UniformNode<FANOUT>), alignof(UniformNode<FANOUT>));
            UniformNode<FANOUT>* new_node = allocator_.get_ptr<UniformNode<FANOUT>>(new_node_offset);
            new (new_node) UniformNode<FANOUT>();
            uint64_t new_record_offset = allocate_new_record(local_alloc, coords, value_payload, value_size);
            uint64_t new_leaf_ptr = FatPointer::encode(new_record_offset, 0, FatPointer::LEAF_PTR);
            int new_key_nibble = get_nibble_from_coords(coords, test_idx);
            int existing_key_nibble = get_nibble_from_coords(existing_record->coords, test_idx);
            new_node->children[new_key_nibble].store(new_leaf_ptr, std::memory_order_relaxed);
            new_node->children[existing_key_nibble].store(current_ptr, std::memory_order_relaxed);
            new_node->representative_leaf_offset = new_record_offset;
            uint64_t new_internal_ptr = FatPointer::encode(new_node_offset, test_idx, FatPointer::NODE_PTR);
            if (cas_slot->compare_exchange_strong(current_ptr, new_internal_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart_lex;
        } else {
            UniformNode<FANOUT>* node = allocator_.get_ptr<UniformNode<FANOUT>>(FatPointer::get_offset(current_ptr));
            uint32_t test_idx = FatPointer::get_test_idx(current_ptr);
            StaxRecord* rep_record = allocator_.get_ptr<StaxRecord>(node->representative_leaf_offset);
            int d_idx = find_first_differing_nibble_linear(coords, rep_record->coords, dimensionality_);
            if (d_idx != -1 && static_cast<uint32_t>(d_idx) < test_idx) {
                uint64_t new_node_offset = local_alloc.allocate(sizeof(UniformNode<FANOUT>), alignof(UniformNode<FANOUT>));
                UniformNode<FANOUT>* new_node = allocator_.get_ptr<UniformNode<FANOUT>>(new_node_offset);
                new (new_node) UniformNode<FANOUT>();
                uint64_t new_record_offset = allocate_new_record(local_alloc, coords, value_payload, value_size);
                uint64_t new_leaf_ptr = FatPointer::encode(new_record_offset, 0, FatPointer::LEAF_PTR);
                int new_key_nibble = get_nibble_from_coords(coords, d_idx);
                int existing_key_nibble = get_nibble_from_coords(rep_record->coords, d_idx);
                new_node->children[new_key_nibble].store(new_leaf_ptr, std::memory_order_relaxed);
                new_node->children[existing_key_nibble].store(current_ptr, std::memory_order_relaxed);
                new_node->representative_leaf_offset = new_record_offset;
                uint64_t new_internal_ptr = FatPointer::encode(new_node_offset, d_idx, FatPointer::NODE_PTR);
                if (cas_slot->compare_exchange_strong(current_ptr, new_internal_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart_lex;
            }
            int nibble = get_nibble_from_coords(coords, test_idx);
            cas_slot = &node->children[nibble];
            current_ptr = cas_slot->load(std::memory_order_acquire);
        }
    }
}

template<size_t FANOUT>
StaxRecord* StaxDimensionStore<FANOUT>::get(const uint64_t* coords) const {
    uint64_t current_ptr = root_ptr_.load(std::memory_order_acquire);
    while (current_ptr != 0) {
        FatPointer::Tag tag = FatPointer::get_tag(current_ptr);
        if (tag == FatPointer::LEAF_PTR) {
            StaxRecord* record = allocator_.get_ptr<StaxRecord>(FatPointer::get_offset(current_ptr));
            if (memcmp(record->coords, coords, dimensionality_ * sizeof(uint64_t)) == 0) return record;
            return nullptr;
        }
        uint32_t test_idx = FatPointer::get_test_idx(current_ptr);
        int nibble = get_nibble_from_coords(coords, test_idx);
        UniformNode<FANOUT>* node = allocator_.get_ptr<UniformNode<FANOUT>>(FatPointer::get_offset(current_ptr));
        current_ptr = node->children[nibble].load(std::memory_order_acquire);
    }
    return nullptr;
}

template<size_t FANOUT>
class DimensionalDBWrapper {
private:
    int fd_ = -1;
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_;
    std::string mmap_filepath_;
    FileHeader* file_header_;
    std::unique_ptr<StaxAllocator> global_allocator_;
    std::unique_ptr<ThreadLocalAllocator> local_allocator_;
    std::unique_ptr<StaxDimensionStore<FANOUT>> tree_;
public:
    DimensionalDBWrapper(uint32_t dimensionality, SplittingHeuristic heuristic) : mmap_size_(DB_MAX_VIRTUAL_SIZE) {
        mmap_filepath_ = "/tmp/stax_dim_bench_" + std::to_string(getpid()) + ".db";
        fd_ = open(mmap_filepath_.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        if (fd_ == -1) throw std::runtime_error("Could not open mmap file for benchmark");
        if (ftruncate(fd_, mmap_size_) == -1) throw std::runtime_error("Could not truncate file");
        mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mmap_ptr_ == MAP_FAILED) throw std::runtime_error("mmap failed");
        file_header_ = new (mmap_ptr_) FileHeader();
        file_header_->global_alloc_offset.store(sizeof(FileHeader), std::memory_order_relaxed);
        file_header_->root_ptr.store(0, std::memory_order_relaxed);
        global_allocator_ = std::make_unique<StaxAllocator>(file_header_, static_cast<uint8_t*>(mmap_ptr_));
        local_allocator_ = std::make_unique<ThreadLocalAllocator>(*global_allocator_);
        tree_ = std::make_unique<StaxDimensionStore<FANOUT>>(*global_allocator_, file_header_->root_ptr, dimensionality, heuristic);
    }
    ~DimensionalDBWrapper() {
        if (mmap_ptr_ != MAP_FAILED && mmap_ptr_ != nullptr) munmap(mmap_ptr_, mmap_size_);
        if (fd_ != -1) close(fd_);
        unlink(mmap_filepath_.c_str());
    }
    StaxDimensionStore<FANOUT>* get_tree() { return tree_.get(); }
    ThreadLocalAllocator* get_local_allocator() { return local_allocator_.get(); }
    size_t get_total_memory() const { return file_header_->global_alloc_offset.load(std::memory_order_relaxed); }
};
} // namespace Stax


// =================================================================================================
// --- HyperionTree Implementation ---
// This is the user-provided tree implementation.
// =================================================================================================
namespace Hyperion {

template<typename ValueType>
struct RecordT {
    ValueType value;
    u64 coords[];
    static size_t get_size(u32 d) { return sizeof(RecordT<ValueType>) + sizeof(u64) * d; }
};
using Record = RecordT<u64>;

namespace TaggedIndex {
    static constexpr u32 NODE256_TAG = 0b00, NODE16_TAG  = 0b01, LEAF_TAG = 0b10;
    static constexpr u32 TAG_MASK = 0b11 << 30, INDEX_MASK = ~TAG_MASK;
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
    u32 get_allocated_count() const { return next_idx_.load(std::memory_order_relaxed); }
};

template<typename ValueType>
class RecordManagerT {
    std::atomic<u32> next_record_idx_{1};
    uint8_t* record_pool_;
    const size_t record_size_with_coords_;
    static constexpr u32 MAX_RECORDS = 11 * 1024 * 1024;
public:
    RecordManagerT(u32 d) : record_size_with_coords_(RecordT<ValueType>::get_size(d)) {
        size_t pool_size = (size_t)MAX_RECORDS * record_size_with_coords_;
        record_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (record_pool_ == MAP_FAILED) throw std::runtime_error("mmap failed for record pool");
    }
    ~RecordManagerT() { munmap(record_pool_, (size_t)MAX_RECORDS * record_size_with_coords_); }
    RecordT<ValueType>* get_record(u32 idx) { return reinterpret_cast<RecordT<ValueType>*>(record_pool_ + (size_t)idx * record_size_with_coords_); }
    u32 allocate_record(const u64* coords, u32 dim, ValueType value) {
        u32 idx = next_record_idx_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= MAX_RECORDS) throw std::runtime_error("Record pool exhausted");
        RecordT<ValueType>* rec = get_record(idx);
        rec->value = value;
        memcpy(rec->coords, coords, sizeof(u64) * dim);
        return idx;
    }
    u32 get_allocated_count() const { return next_record_idx_.load(std::memory_order_relaxed); }
    size_t get_record_size() const { return record_size_with_coords_; }
};
using RecordManager = RecordManagerT<u64>;

template<typename ValueType>
struct MemoryContextT {
    std::unique_ptr<Manager<Node16<16>>> nm16;
    std::unique_ptr<Manager<Node256>> nm256;
    std::unique_ptr<RecordManagerT<ValueType>> rm;
    MemoryContextT(u32 dimensionality) {
        nm16 = std::make_unique<Manager<Node16<16>>>(32 * 1024 * 1024);
        nm256 = std::make_unique<Manager<Node256>>(4 * 1024 * 1024);
        rm = std::make_unique<RecordManagerT<ValueType>>(dimensionality);
    }
};
using MemoryContext = MemoryContextT<u64>;

template<size_t NODE16_SIZE, typename ValueType>
class HyperionTree {
    using Node16Type = Node16<NODE16_SIZE>;
    Manager<Node16<NODE16_SIZE>>* nm16_;
    Manager<Node256>& nm256_;
    RecordManagerT<ValueType>& rm_;
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
                RecordT<ValueType>* existing_rec = rm_.get_record(existing_rec_idx);
                if (memcmp(key, existing_rec->coords, dim_ * sizeof(u64)) == 0) return;

                u32 node16_idx = nm16_->allocate_node();
                Node16Type* node = nm16_->get_node(node16_idx);
                int existing_frag = get_key_fragment(existing_rec->coords, depth);
                int new_frag = get_key_fragment(key, depth);

                if (existing_frag != new_frag) {
                    node->keys[0] = std::min(existing_frag, new_frag);
                    node->keys[1] = std::max(existing_frag, new_frag);
                    if (leaf_ptr == 0) leaf_ptr = TaggedIndex::make_leaf_idx(rec_idx);
                    node->children[0].store(existing_frag < new_frag ? current_tagged_ptr : leaf_ptr, std::memory_order_relaxed);
                    node->children[1].store(existing_frag < new_frag ? leaf_ptr : current_tagged_ptr, std::memory_order_relaxed);
                    node->count.store(2, std::memory_order_relaxed);
                    if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node16_idx(node16_idx), std::memory_order_release, std::memory_order_relaxed)) return;
                    else continue;
                } else {
                    node->keys[0] = new_frag;
                    node->count.store(1, std::memory_order_relaxed);
                    if(parent_slot->compare_exchange_strong(current_tagged_ptr, TaggedIndex::make_node16_idx(node16_idx), std::memory_order_release, std::memory_order_relaxed)) {
                        insert_recursive(existing_rec->coords, existing_rec_idx, &node->children[0], depth + 1);
                        insert_recursive(key, rec_idx, &node->children[0], depth + 1);
                        return;
                    } else continue;
                }
            }
            if (TaggedIndex::is_node16(current_tagged_ptr)) {
                Node16Type* node = nm16_->get_node(TaggedIndex::get_index(current_tagged_ptr));
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
                    if (leaf_ptr == 0) leaf_ptr = TaggedIndex::make_leaf_idx(rec_idx);
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
        }
    }
public:
    HyperionTree(MemoryContextT<ValueType>& ctx, u32 d) : nm256_(*ctx.nm256), rm_(*ctx.rm), dim_(d) {
        nm16_ = (Manager<Node16<16>>*)ctx.nm16.get();
    }
    void insert(const u64* key, ValueType value) {
        u32 rec_idx = rm_.allocate_record(key, dim_, value);
        insert_recursive(key, rec_idx, &root_ptr_, 0);
    }
    std::optional<ValueType> get(const u64* key) {
        int depth = 0;
        u32 current_tagged_ptr = root_ptr_.load(std::memory_order_acquire);
        while(true) {
            if(current_tagged_ptr == 0) return std::nullopt;
            if(TaggedIndex::is_leaf(current_tagged_ptr)) {
                RecordT<ValueType>* rec = rm_.get_record(TaggedIndex::get_index(current_tagged_ptr));
                if (memcmp(key, rec->coords, dim_ * sizeof(u64)) == 0) return rec->value;
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
} // namespace Hyperion



// =================================================================================================
// --- Benchmark Implementation ---
// =================================================================================================

void print_stats(const std::string& tree_name, const std::string& op_name, long long duration_ns, size_t num_keys, int key_size_bytes, int value_size_bytes) {
    double duration_s = duration_ns / 1e9;
    double ns_per_op = (double)duration_ns / num_keys;
    double mops = num_keys / duration_s / 1e6;
    double total_mb = (double)num_keys * (key_size_bytes + value_size_bytes) / (1024 * 1024);
    double mb_per_s = total_mb / duration_s;

    std::cout << std::left << std::setw(22) << tree_name
              << std::left << std::setw(10) << op_name
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << ns_per_op << " ns/op |"
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << mops << " Mops/s |"
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << mb_per_s << " MB/s"
              << std::endl;
}

void print_hyperion_stats(Hyperion::MemoryContextT<u64>& ctx) {
    auto& nm16 = *ctx.nm16;
    auto& nm256 = *ctx.nm256;
    auto& rm = *ctx.rm;

    u32 node16_count = nm16.get_allocated_count() > 1 ? nm16.get_allocated_count() - 1 : 0;
    u32 node256_count = nm256.get_allocated_count() > 1 ? nm256.get_allocated_count() - 1 : 0;
    u32 record_count = rm.get_allocated_count() > 1 ? rm.get_allocated_count() - 1 : 0;

    size_t node16_mem = (size_t)node16_count * sizeof(Hyperion::Node16<16>);
    size_t node256_mem = (size_t)node256_count * sizeof(Hyperion::Node256);
    size_t record_mem = (size_t)record_count * rm.get_record_size();
    size_t total_mem_mb = (node16_mem + node256_mem + record_mem) / (1024 * 1024);

    double node16_density = 0.0;
    if (node16_count > 0) {
        uint64_t node16_total_children = 0;
        for (u32 i = 1; i <= node16_count; ++i) {
            node16_total_children += nm16.get_node(i)->count.load(std::memory_order_relaxed);
        }
        node16_density = (double)node16_total_children / ((uint64_t)node16_count * 16) * 100.0;
    }

    double node256_density = 0.0;
    if (node256_count > 0) {
        uint64_t node256_total_children = 0;
        for (u32 i = 1; i <= node256_count; ++i) {
            Hyperion::Node256* node = nm256.get_node(i);
            for(int j=0; j<256; ++j) {
                if (node->children[j].load(std::memory_order_relaxed) != 0) {
                    node256_total_children++;
                }
            }
        }
        node256_density = (double)node256_total_children / ((uint64_t)node256_count * 256) * 100.0;
    }

    std::cout << "  \\_ Stats Table " << std::string(55, '-') << std::endl;
    std::cout << "  | " << std::left << std::setw(15) << "Metric" << " | " << std::setw(40) << "Value" << " |" << std::endl;
    std::cout << "  |-----------------|------------------------------------------|" << std::endl;
    std::cout << "  | " << std::left << std::setw(15) << "Memory Usage" << " | " << std::setw(40) << (std::to_string(total_mem_mb) + " MB") << " |" << std::endl;
    std::cout << "  | " << std::left << std::setw(15) << "Records" << " | " << std::setw(40) << record_count << " |" << std::endl;

    std::stringstream n16_stats;
    n16_stats << node16_count << " nodes (" << std::fixed << std::setprecision(2) << node16_density << "% full)";
    std::cout << "  | " << std::left << std::setw(15) << "Node16" << " | " << std::setw(40) << n16_stats.str() << " |" << std::endl;

    std::stringstream n256_stats;
    n256_stats << node256_count << " nodes (" << std::fixed << std::setprecision(2) << node256_density << "% full)";
    std::cout << "  | " << std::left << std::setw(15) << "Node256" << " | " << std::setw(40) << n256_stats.str() << " |" << std::endl;
    std::cout << "  " << std::string(60, '-') << std::endl;
}

void print_stax_stats(Stax::DimensionalDBWrapper<16>& db) {
    size_t total_mem_mb = db.get_total_memory() / (1024 * 1024);
    auto stats = db.get_tree()->get_stats();
    double density = 0.0;
    if (stats.node_count > 0) {
        density = (double)stats.total_children / ((uint64_t)stats.node_count * 16) * 100.0;
    }

    std::cout << "  \\_ Stats Table " << std::string(55, '-') << std::endl;
    std::cout << "  | " << std::left << std::setw(15) << "Metric" << " | " << std::setw(40) << "Value" << " |" << std::endl;
    std::cout << "  |-----------------|------------------------------------------|" << std::endl;
    std::cout << "  | " << std::left << std::setw(15) << "Memory Usage" << " | " << std::setw(40) << (std::to_string(total_mem_mb) + " MB") << " |" << std::endl;
    std::cout << "  | " << std::left << std::setw(15) << "Leafs" << " | " << std::setw(40) << stats.leaf_count << " |" << std::endl;

    std::stringstream node_stats;
    node_stats << stats.node_count << " nodes (" << std::fixed << std::setprecision(2) << density << "% full)";
    std::cout << "  | " << std::left << std::setw(15) << "Nodes" << " | " << std::setw(40) << node_stats.str() << " |" << std::endl;
    std::cout << "  " << std::string(60, '-') << std::endl;
}

void run_benchmarks(const std::string& key_type, int key_size_bytes) {
    const size_t num_keys = 1000000;
    const int dim = key_size_bytes / 8;

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "--- Benchmark: " << key_type << " keys, " << key_size_bytes << " bytes "
              << "(N=" << num_keys / 1000000 << "M) ---" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(22) << "Tree"
              << std::left << std::setw(10) << "Operation"
              << std::right << std::setw(15) << "Latency |"
              << std::right << std::setw(15) << "Throughput |"
              << std::right << std::setw(15) << "Bandwidth"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

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
        Hyperion::HyperionTree<16, u64> tree(ctx, dim);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) tree.insert(keys[i].data(), i);
        auto end = std::chrono::high_resolution_clock::now();
        print_stats("HyperionTree", "Insert", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, sizeof(u64));

        size_t found_count = 0;
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            if (tree.get(keys[i].data())) found_count++;
        }
        end = std::chrono::high_resolution_clock::now();
        print_stats("HyperionTree", "Get", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, sizeof(u64));
        if (found_count != num_keys) std::cerr << "HyperionTree found " << found_count << "/" << num_keys << std::endl;
        print_hyperion_stats(ctx);
    }

    // --- StaxTree (Original) Benchmark ---
    {
        Stax::DimensionalDBWrapper<16> db(dim, Stax::SplittingHeuristic::LEXICOGRAPHICAL);
        auto tree = db.get_tree();
        auto local_alloc = db.get_local_allocator();
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            u64 value = i;
            tree->insert(*local_alloc, keys[i].data(), &value, sizeof(value));
        }
        auto end = std::chrono::high_resolution_clock::now();
        print_stats("StaxTree (Original)", "Insert", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, sizeof(u64));

        size_t found_count = 0;
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            if (tree->get(keys[i].data())) found_count++;
        }
        end = std::chrono::high_resolution_clock::now();
        print_stats("StaxTree (Original)", "Get", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, sizeof(u64));
        if (found_count != num_keys) std::cerr << "StaxTree (Original) found " << found_count << "/" << num_keys << std::endl;
        print_stax_stats(db);
    }

}

template<typename ValueType>
void run_single_throughput_test(size_t num_keys) {
    const int key_size_bytes = 8;
    const int value_size_bytes = sizeof(ValueType);
    const int dim = key_size_bytes / 8;

    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "--- Throughput Test: " << value_size_bytes << "-byte values "
              << "(N=" << num_keys / 1000000 << "M) ---" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<std::vector<u64>> keys(num_keys, std::vector<u64>(dim));
    std::vector<ValueType> values(num_keys);
    std::mt19937_64 rng(12345);
    for (size_t i = 0; i < num_keys; ++i) {
        keys[i][0] = rng();
        // Fill value with some data
        uint8_t* val_ptr = reinterpret_cast<uint8_t*>(&values[i]);
        for(size_t j = 0; j < value_size_bytes; ++j) {
            val_ptr[j] = (uint8_t)i;
        }
    }

    // --- HyperionTree Benchmark ---
    {
        Hyperion::MemoryContextT<ValueType> ctx(dim);
        Hyperion::HyperionTree<16, ValueType> tree(ctx, dim);
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) tree.insert(keys[i].data(), values[i]);
        auto end = std::chrono::high_resolution_clock::now();
        print_stats("HyperionTree", "Insert", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, value_size_bytes);

        size_t found_count = 0;
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            if (tree.get(keys[i].data())) found_count++;
        }
        end = std::chrono::high_resolution_clock::now();
        print_stats("HyperionTree", "Get", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, value_size_bytes);
        if (found_count != num_keys) std::cerr << "HyperionTree found " << found_count << "/" << num_keys << std::endl;
    }

    // --- StaxTree (Original) Benchmark ---
    {
        Stax::DimensionalDBWrapper<16> db(dim, Stax::SplittingHeuristic::LEXICOGRAPHICAL);
        auto tree = db.get_tree();
        auto local_alloc = db.get_local_allocator();
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            tree->insert(*local_alloc, keys[i].data(), &values[i], value_size_bytes);
        }
        auto end = std::chrono::high_resolution_clock::now();
        print_stats("StaxTree (Original)", "Insert", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, value_size_bytes);

        size_t found_count = 0;
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_keys; ++i) {
            if (tree->get(keys[i].data())) found_count++;
        }
        end = std::chrono::high_resolution_clock::now();
        print_stats("StaxTree (Original)", "Get", std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(), num_keys, key_size_bytes, value_size_bytes);
        if (found_count != num_keys) std::cerr << "StaxTree (Original) found " << found_count << "/" << num_keys << std::endl;
    }
}


void run_throughput_benchmarks() {
    const size_t num_keys = 1000000;
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "======           THROUGHPUT BENCHMARKS (8-byte keys)           ======" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    struct Payload8 { uint8_t data[8]; };
    struct Payload32 { uint8_t data[32]; };
    struct Payload128 { uint8_t data[128]; };

    run_single_throughput_test<Payload8>(num_keys);
    run_single_throughput_test<Payload32>(num_keys);
    run_single_throughput_test<Payload128>(num_keys);
}

int main() {
    try {
        run_benchmarks("Sequential", 8);
        run_benchmarks("Random", 8);
        run_benchmarks("Sequential", 16);
        run_benchmarks("Random", 16);
        run_benchmarks("Sequential", 32);
        run_benchmarks("Random", 32);
        run_throughput_benchmarks();
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
