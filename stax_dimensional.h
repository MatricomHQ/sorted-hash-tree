#ifndef STAX_DIMENSIONAL_H
#define STAX_DIMENSIONAL_H

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
#include <map>
#include <unordered_map>
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// --- SIMD Intrinsics Headers ---
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386)
#include <immintrin.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif


// =================================================================================================
// --- Mocked/Stubbed Dependencies ---
// =================================================================================================
#if defined(WASM_BUILD)
constexpr uint64_t DB_MAX_VIRTUAL_SIZE = 256ULL * 1024 * 1024; // 256 MB for WASM
#else
constexpr uint64_t DB_MAX_VIRTUAL_SIZE = 10ULL * 1024 * 1024 * 1024; // 10 GB
#endif
struct FileHeader {
    std::atomic<uint64_t> global_alloc_offset;
    std::atomic<uint64_t> root_ptr;
};


#if defined(_MSC_VER)
#define STAX_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define STAX_ALWAYS_INLINE __attribute__((always_inline))
#else
#define STAX_ALWAYS_INLINE inline
#endif

// =================================================================================================
// --- Memory Managers (NodeManager & RecordManager) ---
// =================================================================================================

// Forward declaration
template<size_t FANOUT> struct UniformNode;
class ValueStore;

// Represents a leaf in the tree, storing the actual data point and payload.
struct StaxRecord {
    // uint64_t value_payload;
    uint32_t flags;
    uint32_t value_len;
    uint64_t value_or_offset;
    uint64_t coords[]; // Flexible array member

    static constexpr uint32_t INLINE_FLAG = 1U << 31;

    bool is_value_inlined() const {
        return (flags & INLINE_FLAG) != 0;
    }

    void set_inlined(bool is_inlined) {
        flags = (is_inlined ? INLINE_FLAG : 0);
    }

    uint32_t get_value_len() const {
        return value_len;
    }

    std::string_view get_value(const ValueStore* value_store) const;


    const uint64_t* get_coords() const { return coords; }
    uint64_t* get_coords() { return coords; }

    static size_t get_size(uint32_t dim) {
        return sizeof(StaxRecord) + sizeof(uint64_t) * dim;
    }
};

class ValueStore {
private:
    std::atomic<uint64_t> next_offset_;
    uint8_t* value_pool_;
    // Set a practical limit for benchmarks. 4GB should be sufficient.
    static constexpr uint64_t MAX_VALUE_BYTES = 4ULL * 1024 * 1024 * 1024;

public:
    ValueStore() : next_offset_(1) { // Offset 0 is reserved
        value_pool_ = (uint8_t*)mmap(nullptr, MAX_VALUE_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (value_pool_ == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap value pool.");
        }
    }

    ~ValueStore() {
        munmap(value_pool_, MAX_VALUE_BYTES);
    }

    const uint8_t* get_value_ptr(uint64_t offset) const {
        if (offset == 0 || offset >= next_offset_.load(std::memory_order_relaxed)) {
            return nullptr;
        }
        return value_pool_ + offset;
    }

    uint64_t allocate_value(const void* data, uint32_t size) {
        size_t padded_size = (size + 7) & ~7; // Pad to 8-byte boundary
        uint64_t offset = next_offset_.fetch_add(padded_size, std::memory_order_relaxed);

        if (offset + padded_size > MAX_VALUE_BYTES) {
            next_offset_.fetch_sub(padded_size, std::memory_order_relaxed); // Revert
            throw std::runtime_error("Value pool exhausted.");
        }

        memcpy(value_pool_ + offset, data, size);
        return offset;
    }

    size_t get_allocated_size() const {
        return next_offset_.load(std::memory_order_relaxed);
    }
};


inline std::string_view StaxRecord::get_value(const ValueStore* value_store) const {
    if (is_value_inlined()) {
        return std::string_view(reinterpret_cast<const char*>(&value_or_offset), value_len);
    } else {
        const uint8_t* data_ptr = value_store->get_value_ptr(value_or_offset);
        if (data_ptr) {
            return std::string_view(reinterpret_cast<const char*>(data_ptr), value_len);
        }
        return std::string_view(); // Return empty view if offset is invalid
    }
}


class NodeManager {
private:
    std::atomic<uint32_t> next_node_idx_;
    uint8_t* node_pool_;
    // Set a practical limit for benchmarks instead of the theoretical max. 16M nodes = ~1.5GB
    static constexpr uint32_t MAX_NODES = 16 * 1024 * 1024;
    static constexpr size_t NODE_SIZE = 96; // As per the new design

public:
    NodeManager() : next_node_idx_(1) { // Index 0 is reserved (null)
        size_t pool_size = (size_t)MAX_NODES * NODE_SIZE;
        node_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (node_pool_ == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap node pool.");
        }
    }

    ~NodeManager() {
        munmap(node_pool_, (size_t)MAX_NODES * NODE_SIZE);
    }

    template<size_t FANOUT>
    UniformNode<FANOUT>* get_node(uint32_t index) const {
        if (index == 0 || index >= next_node_idx_.load(std::memory_order_relaxed)) return nullptr;
        return reinterpret_cast<UniformNode<FANOUT>*>(node_pool_ + (size_t)index * NODE_SIZE);
    }

    template<size_t FANOUT>
    uint32_t allocate_node() {
        uint32_t new_idx = next_node_idx_.fetch_add(1, std::memory_order_relaxed);
        if (new_idx >= MAX_NODES) {
            next_node_idx_.fetch_sub(1, std::memory_order_relaxed); // Revert
            throw std::runtime_error("Node pool exhausted.");
        }
        UniformNode<FANOUT>* node = get_node<FANOUT>(new_idx);
        new (node) UniformNode<FANOUT>(); // Placement new to construct
        return new_idx;
    }

    size_t get_allocated_size() const {
        return (size_t)next_node_idx_.load(std::memory_order_relaxed) * NODE_SIZE;
    }
};


class RecordManager {
private:
    std::atomic<uint32_t> next_record_idx_;
    uint8_t* record_pool_;
    ValueStore* value_store_;
    const uint32_t dimensionality_;
    const size_t record_size_;
    // Set a practical limit for benchmarks. 1GB should be sufficient.
    static constexpr uint64_t MAX_RECORDS_BYTES = 1ULL * 1024 * 1024 * 1024;
    static constexpr uint32_t MAX_RECORDS = 32 * 1024 * 1024; // 32M records


public:
    RecordManager(ValueStore* value_store, uint32_t dimensionality)
        : next_record_idx_(1),
          value_store_(value_store),
          dimensionality_(dimensionality),
          record_size_((StaxRecord::get_size(dimensionality) + 7) & ~7) // 8-byte alignment
    {
        uint64_t pool_size = (uint64_t)MAX_RECORDS * record_size_;
        if (pool_size > MAX_RECORDS_BYTES) {
             pool_size = MAX_RECORDS_BYTES;
        }
        record_pool_ = (uint8_t*)mmap(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (record_pool_ == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap record pool.");
        }
    }

    ~RecordManager() {
        uint64_t pool_size = (uint64_t)MAX_RECORDS * record_size_;
        if (pool_size > MAX_RECORDS_BYTES) {
             pool_size = MAX_RECORDS_BYTES;
        }
        munmap(record_pool_, pool_size);
    }

    StaxRecord* get_record(uint32_t index) const {
        if (index == 0 || index >= next_record_idx_.load(std::memory_order_relaxed)) return nullptr;
        return reinterpret_cast<StaxRecord*>(record_pool_ + (uint64_t)index * record_size_);
    }

    uint32_t allocate_record(const uint64_t* coords, const void* value, uint32_t value_len) {
        uint32_t index = next_record_idx_.fetch_add(1, std::memory_order_relaxed);

        if (((uint64_t)index * record_size_) + record_size_ > MAX_RECORDS_BYTES) {
            next_record_idx_.fetch_sub(1, std::memory_order_relaxed);
            throw std::runtime_error("Record pool exhausted.");
        }

        StaxRecord* new_rec = get_record(index);
        new_rec->value_len = value_len;
        memcpy(new_rec->coords, coords, sizeof(uint64_t) * dimensionality_);

        if (value_len <= 8) {
            new_rec->set_inlined(true); // Inlined
            memset(&new_rec->value_or_offset, 0, sizeof(uint64_t));
            memcpy(&new_rec->value_or_offset, value, value_len);
        } else {
            new_rec->set_inlined(false); // External
            new_rec->value_or_offset = value_store_->allocate_value(value, value_len);
        }

        return index;
    }

    size_t get_allocated_size() const {
        return (size_t)next_record_idx_.load(std::memory_order_relaxed) * record_size_;
    }
};

// =================================================================================================
// --- Core N-Dimensional Data Structures ---
// =================================================================================================
struct Range {
    uint64_t min;
    uint64_t max;
};

// =================================================================================================
// --- Tagged Index Implementation ---
// =================================================================================================
// We use a 32-bit integer for indices. The MSB is a tag.
// [ 1-bit Tag | 31-bit Index/Offset ]
// Tag = 1 -> Leaf (Record) -> Index is an index into RecordManager
// Tag = 0 -> Internal Node -> Index is an index into NodeManager
namespace TaggedIndex {
    static constexpr uint32_t TAG_BIT = 1U << 31;
    static constexpr uint32_t INDEX_MASK = (1U << 31) - 1;

    inline bool is_leaf(uint32_t tagged_idx) {
        return (tagged_idx & TAG_BIT) != 0;
    }

    inline bool is_node(uint32_t tagged_idx) {
        return (tagged_idx & TAG_BIT) == 0 && tagged_idx != 0;
    }

    inline uint32_t get_index(uint32_t tagged_idx) {
        return tagged_idx & INDEX_MASK;
    }

    inline uint32_t make_leaf_idx(uint32_t record_idx) {
        return record_idx | TAG_BIT;
    }

    inline uint32_t make_node_idx(uint32_t node_idx) {
        // Node indices don't need the tag bit set, and we assume node_idx doesn't have the MSB set.
        return node_idx;
    }
};

// The number of skip-pointer levels.
constexpr int NUM_LEVELS = 4;

template<size_t FANOUT>
struct UniformNode {
    // 64 bytes for children indices
    std::atomic<uint32_t> children[FANOUT];

    // 16 bytes for skip pointers
    std::atomic<uint32_t> skip_pointers[NUM_LEVELS];

    // 16 bytes for metadata
    union {
        // For ADAPTIVE heuristic
        struct {
            uint32_t split_dim;
            uint32_t representative_record_idx; // Untagged index to a representative record for distance calcs
            uint32_t split_chunk;
            uint8_t level;
            uint8_t padding[3]; // unused
        } adaptive;

        // For LEXICOGRAPHICAL heuristic
        struct {
            uint32_t test_idx;
            // This is an index into the RecordManager pool, NOT a tagged index.
            uint32_t representative_record_idx;
        } lexico;

        // For K_DIMENSIONAL_CYCLIC heuristic
        struct {
            uint32_t depth; // The depth of the node, to determine split_dim = depth % D
            uint32_t representative_record_idx;
        } k_cyclic;
    } meta;


    UniformNode() {
        memset(&meta, 0, sizeof(meta));

        for(size_t i = 0; i < FANOUT; ++i) {
            children[i].store(0, std::memory_order_relaxed);
        }
        for(size_t i = 0; i < NUM_LEVELS; ++i) {
            skip_pointers[i].store(0, std::memory_order_relaxed);
        }
    }
};
static_assert(sizeof(UniformNode<16>) == 96, "UniformNode must be 96 bytes");


// =================================================================================================
// --- StaxDimensionStore (Generic k-d Tree Implementation) ---
// =================================================================================================

enum class SplittingHeuristic {
    ADAPTIVE,        // The default k-d tree behavior
    LEXICOGRAPHICAL, // The new 1D-emulation behavior for KV stores
    K_DIMENSIONAL_CYCLIC // The new k-d tree with cyclic dimension splitting
};

template<size_t FANOUT>
class StaxDimensionStore {
private:
    NodeManager& node_manager_;
    RecordManager& record_manager_;
    std::atomic<uint32_t>& root_ptr_;
    uint32_t dimensionality_;
    SplittingHeuristic heuristic_;
    static thread_local std::mt19937 rng_;

    uint8_t random_level() {
        constexpr double PROB_FACTOR = 0.36067; // 1.0 / log(16)
        uint8_t level = 0;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        while (dist(rng_) < PROB_FACTOR && level < NUM_LEVELS - 1) {
            level++;
        }
        return level;
    }

    uint64_t l1_dist(const uint64_t* p1, const uint64_t* p2) const {
        uint64_t total_dist = 0;
        for (uint32_t i = 0; i < dimensionality_; ++i) {
            uint64_t diff = (p1[i] > p2[i]) ? (p1[i] - p2[i]) : (p2[i] - p1[i]);
            total_dist += diff;
        }
        return total_dist;
    }

    uint32_t find_nearest_neighbor(const uint64_t* coords, uint8_t level) const {
        uint32_t current_node_idx = 0;
        uint64_t min_dist = std::numeric_limits<uint64_t>::max();

        uint32_t entry_point_idx = root_ptr_.load(std::memory_order_acquire);
        if(entry_point_idx == 0 || TaggedIndex::is_leaf(entry_point_idx)) return 0;

        for (int l = NUM_LEVELS - 1; l > level; --l) {
            bool changed = true;
            while(changed) {
                changed = false;
                UniformNode<FANOUT>* entry_node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(entry_point_idx));
                if (entry_node == nullptr || entry_node->meta.adaptive.level < l) break;

                uint32_t next_candidate_idx = entry_node->skip_pointers[l].load(std::memory_order_acquire);
                if (next_candidate_idx == 0) break;

                const StaxRecord* entry_repr = record_manager_.get_record(entry_node->meta.adaptive.representative_record_idx);
                const UniformNode<FANOUT>* next_node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(next_candidate_idx));
                if (next_node == nullptr) break;
                const StaxRecord* next_repr = record_manager_.get_record(next_node->meta.adaptive.representative_record_idx);

                if (l1_dist(coords, next_repr->get_coords()) < l1_dist(coords, entry_repr->get_coords())) {
                    entry_point_idx = next_candidate_idx;
                    changed = true;
                }
            }
        }

        current_node_idx = entry_point_idx;
        bool changed = true;
        while(changed) {
            changed = false;
            UniformNode<FANOUT>* current_node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(current_node_idx));
            if (current_node == nullptr || current_node->meta.adaptive.level < level) break;
            const StaxRecord* current_repr = record_manager_.get_record(current_node->meta.adaptive.representative_record_idx);
            min_dist = l1_dist(coords, current_repr->get_coords());

            uint32_t neighbor_idx = current_node->skip_pointers[level].load(std::memory_order_acquire);
            if (neighbor_idx != 0) {
                const UniformNode<FANOUT>* neighbor_node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(neighbor_idx));
                if (neighbor_node != nullptr) {
                    const StaxRecord* neighbor_repr = record_manager_.get_record(neighbor_node->meta.adaptive.representative_record_idx);
                    if (l1_dist(coords, neighbor_repr->get_coords()) < min_dist) {
                        current_node_idx = neighbor_idx;
                        changed = true;
                    }
                }
            }
        }

        return current_node_idx;
    }

    void wire_skip_pointers(uint32_t new_node_idx, const uint64_t* coords) {
        UniformNode<FANOUT>* new_node = node_manager_.template get_node<FANOUT>(new_node_idx);
        uint8_t node_level = new_node->meta.adaptive.level;

        for (int l = node_level; l >= 0; --l) {
            uint32_t neighbor_idx = find_nearest_neighbor(coords, l);
            if (neighbor_idx != 0 && neighbor_idx != new_node_idx) {
                new_node->skip_pointers[l].store(neighbor_idx, std::memory_order_release);
            }
        }
    }

    uint32_t find_approximate_entry_point(const uint64_t* coords) const {
        uint32_t entry_point_idx = root_ptr_.load(std::memory_order_acquire);
        if (entry_point_idx == 0) return 0;

        for (int l = NUM_LEVELS - 1; l >= 0; --l) {
            bool changed = true;
            while (changed) {
                changed = false;
                if (TaggedIndex::is_leaf(entry_point_idx)) break;

                UniformNode<FANOUT>* entry_node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(entry_point_idx));
                if (entry_node == nullptr || entry_node->meta.adaptive.level < l) break;

                uint32_t next_candidate_idx = entry_node->skip_pointers[l].load(std::memory_order_acquire);
                if (next_candidate_idx == 0) break;

                const StaxRecord* entry_repr = record_manager_.get_record(entry_node->meta.adaptive.representative_record_idx);
                const UniformNode<FANOUT>* next_node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(next_candidate_idx));
                if (next_node == nullptr) break;
                const StaxRecord* next_repr = record_manager_.get_record(next_node->meta.adaptive.representative_record_idx);

                if (l1_dist(coords, next_repr->get_coords()) < l1_dist(coords, entry_repr->get_coords())) {
                    entry_point_idx = next_candidate_idx;
                    changed = true;
                }
            }
        }
        return entry_point_idx;
    }

public:
    StaxDimensionStore(NodeManager& node_manager, RecordManager& record_manager, std::atomic<uint32_t>& root_ref, uint32_t dimensionality, SplittingHeuristic heuristic = SplittingHeuristic::ADAPTIVE)
        : node_manager_(node_manager), record_manager_(record_manager), root_ptr_(root_ref), dimensionality_(dimensionality), heuristic_(heuristic) {}

    void insert(const uint64_t* coords, std::string_view value);
    StaxRecord* get(const uint64_t* coords) const;
    uint32_t get_record_id(const uint64_t* coords) const;
    void query(const Range* ranges, std::vector<StaxRecord*>& results) const;
    void query_no_alloc(const Range* ranges, StaxRecord** results, size_t& count, size_t max_results) const;
    void multi_query(const std::vector<const Range*>& query_boxes, std::vector<StaxRecord*>& results) const;
    void range_scan_lex(const uint64_t* start_coords, const uint64_t* end_coords, std::vector<StaxRecord*>& results) const;
    std::pair<uint64_t, uint64_t> get_density_stats() const;

private:
    static void find_first_differing_chunk(const uint64_t* coords1, const uint64_t* coords2, uint32_t dim, uint32_t& dim_idx, uint32_t& chunk_idx);
    static int find_first_differing_nibble_linear(const uint64_t* coords1, const uint64_t* coords2, uint32_t dim, uint32_t prefix_match_len_bytes = 0, uint32_t max_scan_len_bytes = std::numeric_limits<uint32_t>::max());

    STAX_ALWAYS_INLINE static int get_chunk_from_coord(uint64_t coord, uint32_t chunk_idx);
    STAX_ALWAYS_INLINE static int get_nibble_from_coords(const uint64_t* coords, uint32_t nibble_idx);
};

template<size_t FANOUT>
thread_local std::mt19937 StaxDimensionStore<FANOUT>::rng_{std::random_device{}()};

template<size_t FANOUT>
STAX_ALWAYS_INLINE int StaxDimensionStore<FANOUT>::get_nibble_from_coords(const uint64_t* coords, uint32_t nibble_idx) {
    const size_t byte_idx = nibble_idx / 2;
    const uint8_t byte = reinterpret_cast<const uint8_t*>(coords)[byte_idx];
    const uint32_t shift_amount = (1 - (nibble_idx & 1)) * 4;
    return (byte >> shift_amount) & 0x0F;
}

template<size_t FANOUT>
void StaxDimensionStore<FANOUT>::range_scan_lex(const uint64_t* start_coords, const uint64_t* end_coords, std::vector<StaxRecord*>& results) const {
    struct ScanFrame {
        uint32_t node_ptr;
        bool lower_bound_tight;
        bool upper_bound_tight;
    };
    constexpr int MAX_SCAN_DEPTH = 2048;
    ScanFrame stack[MAX_SCAN_DEPTH];
    int stack_top = -1;
    uint32_t root = root_ptr_.load(std::memory_order_acquire);
    if (root != 0) {
        stack[++stack_top] = {root, true, true};
    }
    while (stack_top != -1) {
        ScanFrame frame = stack[stack_top--];
        if (TaggedIndex::is_leaf(frame.node_ptr)) {
            StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(frame.node_ptr));
            if (record) {
                // For a leaf, we do a full comparison against the range boundaries
                if ((!frame.lower_bound_tight || memcmp(record->coords, start_coords, dimensionality_ * sizeof(uint64_t)) >= 0) &&
                    (!frame.upper_bound_tight || memcmp(record->coords, end_coords, dimensionality_ * sizeof(uint64_t)) <= 0)) {
                    results.push_back(record);
                }
            }
            continue;
        }

        UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(frame.node_ptr));
        uint32_t test_idx = node->meta.lexico.test_idx;

        int start_nibble = frame.lower_bound_tight ? get_nibble_from_coords(start_coords, test_idx) : 0;
        int end_nibble = frame.upper_bound_tight ? get_nibble_from_coords(end_coords, test_idx) : 15;

        for (int nibble = end_nibble; nibble >= start_nibble; --nibble) {
            uint32_t child_ptr = node->children[nibble].load(std::memory_order_acquire);
            if (child_ptr == 0) {
                continue;
            }
            bool next_lower_tight = frame.lower_bound_tight && (nibble == start_nibble);
            bool next_upper_tight = frame.upper_bound_tight && (nibble == end_nibble);

            if (stack_top + 1 >= MAX_SCAN_DEPTH) throw std::runtime_error("Range scan stack overflow");
            stack[++stack_top] = {child_ptr, next_lower_tight, next_upper_tight};
        }
    }
}

template<size_t FANOUT>
void StaxDimensionStore<FANOUT>::multi_query(const std::vector<const Range*>& query_boxes, std::vector<StaxRecord*>& results) const {
    struct ScanFrame { uint32_t node_ptr; };

    constexpr int MAX_SCAN_DEPTH = 1024;
    ScanFrame stack[MAX_SCAN_DEPTH];
    int stack_top = -1;

    uint32_t root = root_ptr_.load(std::memory_order_acquire);
    if (root != 0) {
        stack[++stack_top] = {root};
    }

    while (stack_top != -1) {
        ScanFrame frame = stack[stack_top--];

        if (TaggedIndex::is_leaf(frame.node_ptr)) {
            StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(frame.node_ptr));
            bool all_boxes_match = true;
            for (const auto& ranges : query_boxes) {
                bool current_box_matches = true;
                for (uint32_t d = 0; d < dimensionality_; ++d) {
                    if (record->coords[d] < ranges[d].min || record->coords[d] > ranges[d].max) {
                        current_box_matches = false;
                        break;
                    }
                }
                if (!current_box_matches) {
                    all_boxes_match = false;
                    break;
                }
            }
            if (all_boxes_match) {
                results.push_back(record);
            }
            continue;
        }

        if (TaggedIndex::is_node(frame.node_ptr)) {
            UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(frame.node_ptr));
            uint32_t split_dim = node->meta.adaptive.split_dim;
            uint32_t split_chunk = node->meta.adaptive.split_chunk;

            int min_chunk = 0;
            int max_chunk = FANOUT - 1;

            for (const auto& ranges : query_boxes) {
                const Range& dim_range = ranges[split_dim];
                min_chunk = std::max(min_chunk, get_chunk_from_coord(dim_range.min, split_chunk));
                max_chunk = std::min(max_chunk, get_chunk_from_coord(dim_range.max, split_chunk));
            }

            for (int i = min_chunk; i <= max_chunk; ++i) {
                uint32_t child_ptr = node->children[i].load(std::memory_order_acquire);
                if (child_ptr != 0) {
                     if (stack_top + 1 >= MAX_SCAN_DEPTH) throw std::runtime_error("Range query stack overflow");
                     stack[++stack_top] = {child_ptr};
                }
            }
        }
    }
}

template<size_t FANOUT>
std::pair<uint64_t, uint64_t> StaxDimensionStore<FANOUT>::get_density_stats() const {
    uint64_t used_slots = 0;
    uint64_t total_slots = 0;
    uint32_t root = root_ptr_.load(std::memory_order_acquire);
    if (root == 0) {
        return {0, 0};
    }

    std::vector<uint32_t> stack;
    stack.push_back(root);

    while (!stack.empty()) {
        uint32_t current_ptr = stack.back();
        stack.pop_back();

        if (TaggedIndex::is_node(current_ptr)) {
            UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
            total_slots += FANOUT;
            for (size_t i = 0; i < FANOUT; ++i) {
                uint32_t child_ptr = node->children[i].load(std::memory_order_acquire);
                if (child_ptr != 0) {
                    used_slots++;
                    stack.push_back(child_ptr);
                }
            }
        }
    }
    return {used_slots, total_slots};
}


// =================================================================================================
template<size_t FANOUT>
void StaxDimensionStore<FANOUT>::query_no_alloc(const Range* ranges, StaxRecord** results, size_t& count, size_t max_results) const {
    struct ScanFrame { uint32_t node_ptr; };

    constexpr int MAX_SCAN_DEPTH = 1024;
    ScanFrame stack[MAX_SCAN_DEPTH];
    int stack_top = -1;

    uint32_t root = root_ptr_.load(std::memory_order_acquire);
    if (root != 0) {
        stack[++stack_top] = {root};
    }

    count = 0;
    while (stack_top != -1) {
        ScanFrame frame = stack[stack_top--];

        if (TaggedIndex::is_leaf(frame.node_ptr)) {
            StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(frame.node_ptr));
            bool all_dims_match = true;
            for (uint32_t d = 0; d < dimensionality_; ++d) {
                if (record->coords[d] < ranges[d].min || record->coords[d] > ranges[d].max) {
                    all_dims_match = false;
                    break;
                }
            }
            if (all_dims_match) {
                if (count < max_results) {
                    results[count++] = record;
                }
            }
            continue;
        }

        if (TaggedIndex::is_node(frame.node_ptr)) {
            UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(frame.node_ptr));
            uint32_t split_dim = node->meta.adaptive.split_dim;
            uint32_t split_chunk = node->meta.adaptive.split_chunk;

            const Range& dim_range = ranges[split_dim];
            int min_chunk = get_chunk_from_coord(dim_range.min, split_chunk);
            int max_chunk = get_chunk_from_coord(dim_range.max, split_chunk);

            for (int i = min_chunk; i <= max_chunk; ++i) {
                uint32_t child_ptr = node->children[i].load(std::memory_order_acquire);
                if (child_ptr) {
                     if (stack_top + 1 >= MAX_SCAN_DEPTH) throw std::runtime_error("Range query stack overflow");
                     stack[++stack_top] = {child_ptr};
                }
            }
        }
    }
}

// --- StaxDimensionStore Method Implementations ---
// =================================================================================================

template<size_t FANOUT>
int StaxDimensionStore<FANOUT>::find_first_differing_nibble_linear(const uint64_t* coords1, const uint64_t* coords2, uint32_t dim, uint32_t prefix_match_len_bytes, uint32_t max_scan_len_bytes) {
    const char* k1 = reinterpret_cast<const char*>(coords1);
    const char* k2 = reinterpret_cast<const char*>(coords2);
    const size_t len = std::min((size_t)dim * sizeof(uint64_t), (size_t)max_scan_len_bytes);
    size_t byte_idx = prefix_match_len_bytes;

#if defined(__AVX512F__) && defined(__AVX512BW__)
    while (byte_idx + 64 <= len) {
        __m512i v1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(k1 + byte_idx));
        __m512i v2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(k2 + byte_idx));
        uint64_t match_mask = _mm512_cmpeq_epi8_mask(v1, v2);
        if (match_mask != 0xFFFFFFFFFFFFFFFF) {
            byte_idx += __builtin_ctzll(~match_mask);
            goto found_diff_byte;
        }
        byte_idx += 64;
    }
#endif
#if defined(__AVX2__)
    while (byte_idx + 32 <= len) {
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k1 + byte_idx));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k2 + byte_idx));
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v2)) != (int)0xFFFFFFFF) {
            uint32_t mask = ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v2));
            byte_idx += __builtin_ctz(mask);
            goto found_diff_byte;
        }
        byte_idx += 32;
    }
#endif
#if defined(__SSE2__)
    while (byte_idx + 16 <= len) {
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k1 + byte_idx));
        __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(k2 + byte_idx));
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(v1, v2)) != (int)0xFFFF) {
            uint32_t mask = ~_mm_movemask_epi8(_mm_cmpeq_epi8(v1, v2));
            byte_idx += __builtin_ctz(mask);
            goto found_diff_byte;
        }
        byte_idx += 16;
    }
#endif

    while (byte_idx < len) {
        if (k1[byte_idx] != k2[byte_idx]) {
            goto found_diff_byte;
        }
        byte_idx++;
    }

    return -1; // Identical

found_diff_byte:
    uint8_t b1 = k1[byte_idx];
    uint8_t b2 = k2[byte_idx];
    if ((b1 >> 4) != (b2 >> 4)) {
        return byte_idx * 2;
    }
    return byte_idx * 2 + 1;
}

template<size_t FANOUT>
STAX_ALWAYS_INLINE int StaxDimensionStore<FANOUT>::get_chunk_from_coord(uint64_t coord, uint32_t chunk_idx) {
    if constexpr (FANOUT == 16) { // Nibble-based
        uint32_t shift = 60 - (chunk_idx * 4);
        return (coord >> shift) & 0x0F;
    } else if constexpr (FANOUT == 256) { // Byte-based
        uint32_t shift = 56 - (chunk_idx * 8);
        return (coord >> shift) & 0xFF;
    }
    return 0;
}

template<size_t FANOUT>
void StaxDimensionStore<FANOUT>::find_first_differing_chunk(const uint64_t* coords1, const uint64_t* coords2, uint32_t dim, uint32_t& dim_idx, uint32_t& chunk_idx) {
    constexpr int chunks_per_coord = (FANOUT == 16) ? 16 : 8;
    for (uint32_t d = 0; d < dim; ++d) {
        if (coords1[d] != coords2[d]) {
            dim_idx = d;
            for (uint32_t c = 0; c < chunks_per_coord; ++c) {
                if (get_chunk_from_coord(coords1[d], c) != get_chunk_from_coord(coords2[d], c)) {
                    chunk_idx = c;
                    return;
                }
            }
        }
    }
    dim_idx = (uint32_t)-1;
    chunk_idx = (uint32_t)-1;
}

template<size_t FANOUT>
void StaxDimensionStore<FANOUT>::insert(const uint64_t* coords, std::string_view value) {
    if (heuristic_ == SplittingHeuristic::ADAPTIVE) {
    restart_adaptive:
        std::atomic<uint32_t>* cas_slot = &root_ptr_;
        uint32_t current_ptr = root_ptr_.load(std::memory_order_acquire);
        while (true) {
            if (current_ptr == 0) {
                uint32_t new_record_idx = record_manager_.allocate_record(coords, value.data(), value.length());
                uint32_t new_leaf_ptr = TaggedIndex::make_leaf_idx(new_record_idx);
                uint32_t expected_ptr = 0;
                if (cas_slot->compare_exchange_strong(expected_ptr, new_leaf_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart_adaptive;
            }

            if (TaggedIndex::is_leaf(current_ptr)) {
                uint32_t existing_record_idx = TaggedIndex::get_index(current_ptr);
                const StaxRecord* existing_record = record_manager_.get_record(existing_record_idx);
                uint32_t split_dim, split_chunk;
                find_first_differing_chunk(coords, existing_record->coords, dimensionality_, split_dim, split_chunk);
                if (split_dim == (uint32_t)-1) return; // Points are identical

                uint32_t new_node_idx = node_manager_.template allocate_node<FANOUT>();
                UniformNode<FANOUT>* new_node = node_manager_.template get_node<FANOUT>(new_node_idx);

                uint32_t new_record_idx = record_manager_.allocate_record(coords, value.data(), value.length());

                new_node->meta.adaptive.split_dim = split_dim;
                new_node->meta.adaptive.split_chunk = split_chunk;
                new_node->meta.adaptive.level = random_level();
                new_node->meta.adaptive.representative_record_idx = new_record_idx;

                uint32_t new_leaf_ptr = TaggedIndex::make_leaf_idx(new_record_idx);

                int new_chunk = get_chunk_from_coord(coords[split_dim], split_chunk);
                int existing_chunk = get_chunk_from_coord(existing_record->coords[split_dim], split_chunk);
                if (new_chunk == existing_chunk) throw std::logic_error("Chunk indices are identical during a split operation.");

                new_node->children[new_chunk].store(new_leaf_ptr, std::memory_order_relaxed);
                new_node->children[existing_chunk].store(current_ptr, std::memory_order_relaxed);

                uint32_t new_internal_ptr = TaggedIndex::make_node_idx(new_node_idx);
                if (cas_slot->compare_exchange_strong(current_ptr, new_internal_ptr, std::memory_order_release, std::memory_order_relaxed)) {
                    wire_skip_pointers(new_node_idx, coords);
                    return;
                }
                goto restart_adaptive;

            } else if (TaggedIndex::is_node(current_ptr)) {
                const UniformNode<FANOUT>* node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
                uint32_t split_dim = node->meta.adaptive.split_dim;
                uint32_t split_chunk = node->meta.adaptive.split_chunk;
                int chunk_to_follow = get_chunk_from_coord(coords[split_dim], split_chunk);
                cas_slot = (std::atomic<uint32_t>*)&node->children[chunk_to_follow];
                current_ptr = cas_slot->load(std::memory_order_acquire);
            } else {
                throw std::logic_error("Invalid TaggedIndex state encountered during insert.");
            }
        }
        return;
    } else if (heuristic_ == SplittingHeuristic::K_DIMENSIONAL_CYCLIC) {
    restart_k_cyclic:
        std::atomic<uint32_t>* cas_slot = &root_ptr_;
        uint32_t current_ptr = root_ptr_.load(std::memory_order_acquire);
        uint32_t depth = 0;

        while (true) {
            if (current_ptr == 0) { // Empty slot, insert leaf
                uint32_t new_record_idx = record_manager_.allocate_record(coords, value.data(), value.length());
                uint32_t new_leaf_ptr = TaggedIndex::make_leaf_idx(new_record_idx);
                uint32_t expected_ptr = 0;
                if (cas_slot->compare_exchange_strong(expected_ptr, new_leaf_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart_k_cyclic; // CAS failed, restart
            }

            if (TaggedIndex::is_leaf(current_ptr)) { // Collision with a leaf, need to split
                uint32_t existing_record_idx = TaggedIndex::get_index(current_ptr);
                const StaxRecord* existing_record = record_manager_.get_record(existing_record_idx);

                if (memcmp(coords, existing_record->coords, dimensionality_ * sizeof(uint64_t)) == 0) {
                    return; // Points are identical
                }

                uint32_t new_record_idx = record_manager_.allocate_record(coords, value.data(), value.length());
                uint32_t new_leaf_ptr = TaggedIndex::make_leaf_idx(new_record_idx);

                uint32_t split_depth = depth;

                uint32_t top_node_idx = node_manager_.template allocate_node<FANOUT>();
                UniformNode<FANOUT>* parent_node = node_manager_.template get_node<FANOUT>(top_node_idx);
                parent_node->meta.k_cyclic.depth = split_depth;
                parent_node->meta.k_cyclic.representative_record_idx = new_record_idx;

                while (true) {
                    uint32_t split_dim = split_depth % dimensionality_;
                    uint32_t chunk_idx = split_depth / dimensionality_;
                    if (chunk_idx >= 16) throw std::runtime_error("K-D tree has exceeded max depth.");

                    int new_chunk = get_chunk_from_coord(coords[split_dim], chunk_idx);
                    int existing_chunk = get_chunk_from_coord(existing_record->coords[split_dim], chunk_idx);

                    if (new_chunk != existing_chunk) {
                        parent_node->children[new_chunk].store(new_leaf_ptr, std::memory_order_relaxed);
                        parent_node->children[existing_chunk].store(current_ptr, std::memory_order_relaxed);
                        break; // Split is successful
                    }

                    // Points still collide, create an intermediate node and go deeper.
                    split_depth++;
                    uint32_t intermediate_node_idx = node_manager_.template allocate_node<FANOUT>();
                    UniformNode<FANOUT>* intermediate_node = node_manager_.template get_node<FANOUT>(intermediate_node_idx);
                    intermediate_node->meta.k_cyclic.depth = split_depth;
                    intermediate_node->meta.k_cyclic.representative_record_idx = new_record_idx;

                    uint32_t intermediate_ptr = TaggedIndex::make_node_idx(intermediate_node_idx);
                    parent_node->children[new_chunk].store(intermediate_ptr, std::memory_order_relaxed);
                    parent_node = intermediate_node;
                }

                uint32_t top_node_ptr = TaggedIndex::make_node_idx(top_node_idx);
                if (cas_slot->compare_exchange_strong(current_ptr, top_node_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart_k_cyclic;

            } else { // Internal node, continue traversal
                const UniformNode<FANOUT>* node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
                depth = node->meta.k_cyclic.depth;
                uint32_t split_dim = depth % dimensionality_;
                uint32_t chunk_idx = depth / dimensionality_;
                if (chunk_idx >= 16) { // Should not happen if split logic is correct
                     throw std::runtime_error("K-D tree has exceeded max depth during traversal.");
                }
                int chunk_to_follow = get_chunk_from_coord(coords[split_dim], chunk_idx);

                cas_slot = (std::atomic<uint32_t>*)&node->children[chunk_to_follow];
                current_ptr = cas_slot->load(std::memory_order_acquire);
                depth++; // Increment depth for the next level
            }
        }
        return;
    }


    // --- LEXICOGRAPHICAL (CORRECT PATRICIA TRIE) HEURISTIC ---
restart_lex:
    std::atomic<uint32_t>* cas_slot = &root_ptr_;
    uint32_t current_ptr = root_ptr_.load(std::memory_order_acquire);
    uint32_t prefix_match_len_bytes = 0;
    while (true) {
        if (current_ptr == 0) { // Case A: Found an empty slot.
            uint32_t new_record_idx = record_manager_.allocate_record(coords, value.data(), value.length());
            uint32_t new_leaf_ptr = TaggedIndex::make_leaf_idx(new_record_idx);
            uint32_t expected_ptr = 0;
            if (cas_slot->compare_exchange_strong(expected_ptr, new_leaf_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart_lex;
        }

        if (TaggedIndex::is_leaf(current_ptr)) { // Case B: Found a leaf.
            uint32_t existing_record_idx = TaggedIndex::get_index(current_ptr);
            const StaxRecord* existing_record = record_manager_.get_record(existing_record_idx);
            int test_idx = find_first_differing_nibble_linear(coords, existing_record->coords, dimensionality_, prefix_match_len_bytes);
            if (test_idx == -1) return; // Keys are identical, update is a no-op for benchmark

            uint32_t new_node_idx = node_manager_.allocate_node<FANOUT>();
            UniformNode<FANOUT>* new_node = node_manager_.get_node<FANOUT>(new_node_idx);

            uint32_t new_record_idx = record_manager_.allocate_record(coords, value.data(), value.length());
            uint32_t new_leaf_ptr = TaggedIndex::make_leaf_idx(new_record_idx);

            new_node->meta.lexico.test_idx = test_idx;
            new_node->meta.lexico.representative_record_idx = new_record_idx;

            int new_key_nibble = get_nibble_from_coords(coords, test_idx);
            int existing_key_nibble = get_nibble_from_coords(existing_record->coords, test_idx);
            new_node->children[new_key_nibble].store(new_leaf_ptr, std::memory_order_relaxed);
            new_node->children[existing_key_nibble].store(current_ptr, std::memory_order_relaxed);

            uint32_t new_internal_ptr = TaggedIndex::make_node_idx(new_node_idx);
            if (cas_slot->compare_exchange_strong(current_ptr, new_internal_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
            goto restart_lex;

        } else { // Case C: Found an internal node.
            const UniformNode<FANOUT>* node = node_manager_.template get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
            uint32_t test_idx = node->meta.lexico.test_idx;
            const StaxRecord* rep_record = record_manager_.get_record(node->meta.lexico.representative_record_idx);

            int d_idx = find_first_differing_nibble_linear(coords, rep_record->coords, dimensionality_, prefix_match_len_bytes, (test_idx / 2) + 1);
            if (d_idx != -1 && static_cast<uint32_t>(d_idx) < test_idx) {
                // The new key diverges from the path before this node's test index.
                // We must insert a new parent node here.
                uint32_t new_node_idx = node_manager_.allocate_node<FANOUT>();
                UniformNode<FANOUT>* new_node = node_manager_.get_node<FANOUT>(new_node_idx);

                uint32_t new_record_idx = record_manager_.allocate_record(coords, value.data(), value.length());
                uint32_t new_leaf_ptr = TaggedIndex::make_leaf_idx(new_record_idx);

                new_node->meta.lexico.test_idx = d_idx;
                new_node->meta.lexico.representative_record_idx = new_record_idx;

                int new_key_nibble = get_nibble_from_coords(coords, d_idx);
                int existing_key_nibble = get_nibble_from_coords(rep_record->coords, d_idx);
                new_node->children[new_key_nibble].store(new_leaf_ptr, std::memory_order_relaxed);
                new_node->children[existing_key_nibble].store(current_ptr, std::memory_order_relaxed);

                uint32_t new_internal_ptr = TaggedIndex::make_node_idx(new_node_idx);
                if (cas_slot->compare_exchange_strong(current_ptr, new_internal_ptr, std::memory_order_release, std::memory_order_relaxed)) return;
                goto restart_lex;
            }
            prefix_match_len_bytes = test_idx / 2;
            // Continue descent
            int nibble = get_nibble_from_coords(coords, test_idx);
            cas_slot = (std::atomic<uint32_t>*)&node->children[nibble];
            current_ptr = cas_slot->load(std::memory_order_acquire);
        }
    }
}

template<size_t FANOUT>
uint32_t StaxDimensionStore<FANOUT>::get_record_id(const uint64_t* coords) const {
    if (heuristic_ == SplittingHeuristic::ADAPTIVE) {
        uint32_t current_ptr = find_approximate_entry_point(coords);
        if (current_ptr == 0) {
            current_ptr = root_ptr_.load(std::memory_order_acquire);
        }

        while (current_ptr != 0) {
            if (TaggedIndex::is_leaf(current_ptr)) {
                StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(current_ptr));
                if (memcmp(record->coords, coords, dimensionality_ * sizeof(uint64_t)) == 0) {
                    return TaggedIndex::get_index(current_ptr);
                }
                // If the entry point was wrong, we won't find it.
                // A full search would require backtracking, which is not implemented for this accelerated path.
                return 0;
            }
            if (TaggedIndex::is_node(current_ptr)) {
                const UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
                uint32_t split_dim = node->meta.adaptive.split_dim;
                uint32_t split_chunk = node->meta.adaptive.split_chunk;
                int chunk_to_follow = get_chunk_from_coord(coords[split_dim], split_chunk);
                current_ptr = node->children[chunk_to_follow].load(std::memory_order_acquire);
            } else { return 0; }
        }
        return 0; // Not found
    } else if (heuristic_ == SplittingHeuristic::K_DIMENSIONAL_CYCLIC) {
        uint32_t current_ptr = root_ptr_.load(std::memory_order_acquire);
        while (current_ptr != 0) {
            if (TaggedIndex::is_leaf(current_ptr)) {
                StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(current_ptr));
                if (memcmp(record->coords, coords, dimensionality_ * sizeof(uint64_t)) == 0) {
                    return TaggedIndex::get_index(current_ptr);
                }
                return 0; // Not found
            }
            const UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
            uint32_t depth = node->meta.k_cyclic.depth;
            uint32_t split_dim = depth % dimensionality_;
            uint32_t chunk_idx = depth / dimensionality_;
            if (chunk_idx >= 16) return 0; // Exceeded max depth
            int chunk_to_follow = get_chunk_from_coord(coords[split_dim], chunk_idx);
            current_ptr = node->children[chunk_to_follow].load(std::memory_order_acquire);
        }
        return 0; // Not found
    }

    // --- LEXICOGRAPHICAL (PATRICIA TRIE) HEURISTIC ---
    uint32_t current_ptr = root_ptr_.load(std::memory_order_acquire);
    while (current_ptr != 0) {
        if (TaggedIndex::is_leaf(current_ptr)) {
            StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(current_ptr));
            if (memcmp(record->coords, coords, dimensionality_ * sizeof(uint64_t)) == 0) {
                return TaggedIndex::get_index(current_ptr);
            }
            return 0; // Not found
        }
        const UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
        uint32_t test_idx = node->meta.lexico.test_idx;
        int nibble = get_nibble_from_coords(coords, test_idx);
        current_ptr = node->children[nibble].load(std::memory_order_acquire);
    }
    return 0; // Not found
}

template<size_t FANOUT>
StaxRecord* StaxDimensionStore<FANOUT>::get(const uint64_t* coords) const {
    if (heuristic_ == SplittingHeuristic::ADAPTIVE) {
        uint32_t current_ptr = find_approximate_entry_point(coords);
        if (current_ptr == 0) {
            current_ptr = root_ptr_.load(std::memory_order_acquire);
        }

        while (current_ptr != 0) {
            if (TaggedIndex::is_leaf(current_ptr)) {
                StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(current_ptr));
                if (memcmp(record->coords, coords, dimensionality_ * sizeof(uint64_t)) == 0) {
                    return record;
                }
                return nullptr;
            }
            if (TaggedIndex::is_node(current_ptr)) {
                const UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
                uint32_t split_dim = node->meta.adaptive.split_dim;
                uint32_t split_chunk = node->meta.adaptive.split_chunk;
                int chunk_to_follow = get_chunk_from_coord(coords[split_dim], split_chunk);
                current_ptr = node->children[chunk_to_follow].load(std::memory_order_acquire);
            } else { return nullptr; }
        }
        return nullptr;
    } else if (heuristic_ == SplittingHeuristic::K_DIMENSIONAL_CYCLIC) {
        uint32_t current_ptr = root_ptr_.load(std::memory_order_acquire);
        while (current_ptr != 0) {
            if (TaggedIndex::is_leaf(current_ptr)) {
                StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(current_ptr));
                if (memcmp(record->coords, coords, dimensionality_ * sizeof(uint64_t)) == 0) {
                    return record;
                }
                return nullptr;
            }
            const UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
            uint32_t depth = node->meta.k_cyclic.depth;
            uint32_t split_dim = depth % dimensionality_;
            uint32_t chunk_idx = depth / dimensionality_;
            if (chunk_idx >= 16) return nullptr; // Exceeded max depth
            int chunk_to_follow = get_chunk_from_coord(coords[split_dim], chunk_idx);
            current_ptr = node->children[chunk_to_follow].load(std::memory_order_acquire);
        }
        return nullptr;
    }

    // --- LEXICOGRAPHICAL (PATRICIA TRIE) HEURISTIC ---
    uint32_t current_ptr = root_ptr_.load(std::memory_order_acquire);
    while (current_ptr != 0) {
        if (TaggedIndex::is_leaf(current_ptr)) {
            StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(current_ptr));
            if (memcmp(record->coords, coords, dimensionality_ * sizeof(uint64_t)) == 0) {
                return record;
            }
            return nullptr;
        }
        const UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(current_ptr));
        uint32_t test_idx = node->meta.lexico.test_idx;
        int nibble = get_nibble_from_coords(coords, test_idx);
        current_ptr = node->children[nibble].load(std::memory_order_acquire);
    }
    return nullptr;
}

template<size_t FANOUT>
void StaxDimensionStore<FANOUT>::query(const Range* ranges, std::vector<StaxRecord*>& results) const {
    struct ScanFrame { uint32_t node_ptr; };

    constexpr int MAX_SCAN_DEPTH = 1024;
    ScanFrame stack[MAX_SCAN_DEPTH];
    int stack_top = -1;

    uint32_t root = root_ptr_.load(std::memory_order_acquire);
    if (root != 0) {
        stack[++stack_top] = {root};
    }

    while (stack_top != -1) {
        ScanFrame frame = stack[stack_top--];

        if (TaggedIndex::is_leaf(frame.node_ptr)) {
            StaxRecord* record = record_manager_.get_record(TaggedIndex::get_index(frame.node_ptr));
            bool all_dims_match = true;
            for (uint32_t d = 0; d < dimensionality_; ++d) {
                if (record->coords[d] < ranges[d].min || record->coords[d] > ranges[d].max) {
                    all_dims_match = false;
                    break;
                }
            }
            if (all_dims_match) {
                results.push_back(record);
            }
            continue;
        }

        if (TaggedIndex::is_node(frame.node_ptr)) {
            const UniformNode<FANOUT>* node = node_manager_.get_node<FANOUT>(TaggedIndex::get_index(frame.node_ptr));
            if (heuristic_ == SplittingHeuristic::ADAPTIVE) {
                uint32_t split_dim = node->meta.adaptive.split_dim;
                uint32_t split_chunk = node->meta.adaptive.split_chunk;

                const Range& dim_range = ranges[split_dim];
                int min_chunk = get_chunk_from_coord(dim_range.min, split_chunk);
                int max_chunk = get_chunk_from_coord(dim_range.max, split_chunk);

                for (int i = min_chunk; i <= max_chunk; ++i) {
                    uint32_t child_ptr = node->children[i].load(std::memory_order_acquire);
                    if (child_ptr) {
                         if (stack_top + 1 >= MAX_SCAN_DEPTH) throw std::runtime_error("Range query stack overflow");
                         stack[++stack_top] = {child_ptr};
                    }
                }
            } else if (heuristic_ == SplittingHeuristic::K_DIMENSIONAL_CYCLIC) {
                uint32_t depth = node->meta.k_cyclic.depth;
                uint32_t split_dim = depth % dimensionality_;
                uint32_t chunk_idx = depth / dimensionality_;
                if (chunk_idx >= 16) continue; // Skip nodes that are too deep

                const Range& dim_range = ranges[split_dim];
                int min_chunk = get_chunk_from_coord(dim_range.min, chunk_idx);
                int max_chunk = get_chunk_from_coord(dim_range.max, chunk_idx);

                for (int i = min_chunk; i <= max_chunk; ++i) {
                    uint32_t child_ptr = node->children[i].load(std::memory_order_acquire);
                    if (child_ptr) {
                        if (stack_top + 1 >= MAX_SCAN_DEPTH) throw std::runtime_error("Range query stack overflow");
                        stack[++stack_top] = {child_ptr};
                    }
                }
            }
        }
    }
}


#endif // STAX_DIMENSIONAL_H
