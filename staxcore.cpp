#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <sys/mman.h>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <optional>

// --- Constants for the data structure ---
const uint64_t POINTER_TAG = 1ULL << 63;
const uint64_t OFFSET_MASK = ~(POINTER_TAG);
// Each key/value pair takes 2 slots. With 8 slots total, we have 4 pairs.
// So we need 2 bits to index into the 4 pairs.
const unsigned int BITS_PER_LEVEL = 2;
const unsigned int NODE_PAIRS = 1 << BITS_PER_LEVEL; // 4 pairs
const uint64_t LEVEL_INDEX_MASK = NODE_PAIRS - 1;   // Mask for 0-3
const unsigned int MAX_DEPTH = (63 + BITS_PER_LEVEL - 1) / BITS_PER_LEVEL;


// A simple memory manager using a single mmap-ed region.
class MemoryManager {
public:
    explicit MemoryManager(size_t size) : allocation_size(size) {
        base_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base_ptr == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap memory");
        }
        current_ptr = static_cast<uint8_t*>(base_ptr);
        end_ptr = current_ptr + size;
        allocated_bytes = 0;
    }

    ~MemoryManager() {
        if (base_ptr != MAP_FAILED) {
            munmap(base_ptr, allocation_size);
        }
    }

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    void* alloc(size_t size) {
        size_t aligned_size = (size + 7) & ~7;
        if (current_ptr + aligned_size > end_ptr) {
            throw std::bad_alloc();
        }
        void* mem = current_ptr;
        current_ptr += aligned_size;
        allocated_bytes += aligned_size;
        std::fill(static_cast<uint64_t*>(mem), static_cast<uint64_t*>(mem) + aligned_size / sizeof(uint64_t), 0);
        return mem;
    }

    uint64_t get_offset(const void* ptr) const {
        return static_cast<const uint8_t*>(ptr) - static_cast<uint8_t*>(base_ptr);
    }

    void* get_ptr(uint64_t offset) const {
        return static_cast<uint8_t*>(base_ptr) + offset;
    }

    size_t get_allocated_size() const {
        return allocated_bytes;
    }

private:
    void* base_ptr = nullptr;
    uint8_t* current_ptr = nullptr;
    uint8_t* end_ptr = nullptr;
    size_t allocation_size = 0;
    size_t allocated_bytes = 0;
};


// The core node structure. It's exactly 64 bytes (a cache line).
// It holds 4 key-value pairs (8 slots total).
struct Node {
    uint64_t slots[NODE_PAIRS * 2];
};


// The generalized LayeredSlotMap, serving as a K/V store from uint64_t to ValueType.
template<typename ValueType>
class LayeredSlotMap {
public:
    explicit LayeredSlotMap(MemoryManager& mem_manager) : mem(mem_manager) {
        void* root_node_ptr = mem.alloc(sizeof(Node));
        root_node_offset = mem.get_offset(root_node_ptr);
    }

    void insert(uint64_t key, ValueType value) {
        static_assert(sizeof(ValueType) <= sizeof(uint64_t), "ValueType must fit in 64 bits");
        if (key & POINTER_TAG) return;

        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            Node* current_node = static_cast<Node*>(mem.get_ptr(current_node_offset));
            int shift = 64 - (depth + 1) * BITS_PER_LEVEL;
            uint64_t pair_index = (key >> shift) & LEVEL_INDEX_MASK;
            uint64_t key_slot_index = pair_index * 2;

            uint64_t& key_slot = current_node->slots[key_slot_index];
            uint64_t& value_slot = current_node->slots[key_slot_index + 1];

            if (key_slot == 0) {
                key_slot = key;
                memcpy(&value_slot, &value, sizeof(ValueType));
                return;
            }

            if (key_slot & POINTER_TAG) {
                current_node_offset = key_slot & OFFSET_MASK;
                continue;
            }

            if (key_slot == key) {
                memcpy(&value_slot, &value, sizeof(ValueType)); // Update
                return;
            }

            // Collision
            uint64_t existing_key = key_slot;
            ValueType existing_value;
            memcpy(&existing_value, &value_slot, sizeof(ValueType));

            void* new_node_ptr = mem.alloc(sizeof(Node));
            uint64_t new_node_offset = mem.get_offset(new_node_ptr);
            Node* new_node = static_cast<Node*>(new_node_ptr);

            key_slot = new_node_offset | POINTER_TAG;
            value_slot = 0; // Clear old value

            unsigned int next_depth = depth + 1;
            while (next_depth < MAX_DEPTH) {
                int next_shift = 64 - (next_depth + 1) * BITS_PER_LEVEL;
                uint64_t index_existing = ((existing_key >> next_shift) & LEVEL_INDEX_MASK) * 2;
                uint64_t index_new = ((key >> next_shift) & LEVEL_INDEX_MASK) * 2;

                if (index_existing != index_new) {
                    new_node->slots[index_existing] = existing_key;
                    memcpy(&new_node->slots[index_existing + 1], &existing_value, sizeof(ValueType));
                    new_node->slots[index_new] = key;
                    memcpy(&new_node->slots[index_new + 1], &value, sizeof(ValueType));
                    return;
                }

                void* intermediate_node_ptr = mem.alloc(sizeof(Node));
                uint64_t intermediate_node_offset = mem.get_offset(intermediate_node_ptr);
                new_node->slots[index_existing] = intermediate_node_offset | POINTER_TAG;
                new_node->slots[index_existing + 1] = 0;
                new_node = static_cast<Node*>(intermediate_node_ptr);
                next_depth++;
            }
        }
    }

    std::optional<ValueType> get(uint64_t key) {
        static_assert(sizeof(ValueType) <= sizeof(uint64_t), "ValueType must fit in 64 bits");
        if (key & POINTER_TAG) return std::nullopt;

        uint64_t current_node_offset = root_node_offset;
        for (unsigned int depth = 0; depth < MAX_DEPTH; ++depth) {
            Node* current_node = static_cast<Node*>(mem.get_ptr(current_node_offset));
            int shift = 64 - (depth + 1) * BITS_PER_LEVEL;
            uint64_t pair_index = (key >> shift) & LEVEL_INDEX_MASK;
            uint64_t key_slot_index = pair_index * 2;

            uint64_t key_slot = current_node->slots[key_slot_index];

            if (key_slot == 0) return std::nullopt;

            if (key_slot & POINTER_TAG) {
                current_node_offset = key_slot & OFFSET_MASK;
                continue;
            }

            if (key_slot == key) {
                uint64_t& value_slot = current_node->slots[key_slot_index + 1];
                ValueType ret;
                memcpy(&ret, &value_slot, sizeof(ValueType));
                return ret;
            }

            return std::nullopt;
        }
        return std::nullopt;
    }

    std::vector<ValueType> scan(uint64_t start, uint64_t end) {
        static_assert(sizeof(ValueType) <= sizeof(uint64_t), "ValueType must fit in 64 bits");
        std::vector<ValueType> results;
        if (start > end) return results;
        results.reserve(1024);
        scan_recursive(root_node_offset, 0, 0, start, end, results);
        return results;
    }

private:
    MemoryManager& mem;
    uint64_t root_node_offset;

    void scan_recursive(uint64_t node_offset, int depth, uint64_t prefix, uint64_t start, uint64_t end, std::vector<ValueType>& results) {
        if (depth >= MAX_DEPTH) return;
        Node* node = static_cast<Node*>(mem.get_ptr(node_offset));
        int shift = 64 - (depth + 1) * BITS_PER_LEVEL;

        for (int i = 0; i < NODE_PAIRS; ++i) {
            uint64_t key_slot_index = i * 2;
            uint64_t key_slot = node->slots[key_slot_index];
            if (key_slot == 0) continue;

            uint64_t child_prefix = prefix | ((uint64_t)i << shift);
            uint64_t lower_bound = child_prefix;
            uint64_t upper_bound_mask = (shift < 63) ? (1ULL << shift) - 1 : UINT64_MAX;
            uint64_t upper_bound = child_prefix | upper_bound_mask;

            if (lower_bound > end || upper_bound < start) {
                continue;
            }

            if (key_slot & POINTER_TAG) {
                scan_recursive(key_slot & OFFSET_MASK, depth + 1, child_prefix, start, end, results);
            } else {
                if (key_slot >= start && key_slot <= end) {
                    uint64_t& value_slot = node->slots[key_slot_index + 1];
                    ValueType val;
                    memcpy(&val, &value_slot, sizeof(ValueType));
                    results.push_back(val);
                }
            }
        }
    }
};

// --- Hyperion N-Dimensional Index ---

// Forward declarations for recursive templates
template <typename Dim, typename... RemainingDims>
class Hyperion;

template <typename Dim>
class Hyperion<Dim>;


// Primary Template: The recursive step for N dimensions.
template <typename Dim, typename... RemainingDims>
class Hyperion {
private:
    using SubHyperion = Hyperion<RemainingDims...>;
    using Map = LayeredSlotMap<uint64_t>; // ValueType is offset to a SubHyperion

    Map map;
    MemoryManager& mem;

public:
    explicit Hyperion(MemoryManager& mem_manager) : mem(mem_manager), map(mem_manager) {}

    // Recursively inserts the key parts.
    void insert(Dim key, RemainingDims... keys, uint64_t value) {
        std::optional<uint64_t> sub_map_offset = map.get(key);

        if (!sub_map_offset) {
            // Sub-map doesn't exist, create it on-demand.
            void* sub_hyperion_ptr = mem.alloc(sizeof(SubHyperion));
            new (sub_hyperion_ptr) SubHyperion(mem); // Placement new
        uint64_t new_offset = mem.get_offset(sub_hyperion_ptr);
        map.insert(key, new_offset);
        sub_map_offset = new_offset; // THE FIX: Update the local variable
        }

        SubHyperion* sub_hyperion = static_cast<SubHyperion*>(mem.get_ptr(*sub_map_offset));
        sub_hyperion->insert(keys..., value);
    }

    // Retrieves a pointer to the sub-map for the given key.
    SubHyperion* get_sub_map(Dim key) {
        auto offset = map.get(key);
        if (!offset) {
            return nullptr;
        }
        return static_cast<SubHyperion*>(mem.get_ptr(*offset));
    }

    // Recursively gets the final value.
    std::optional<uint64_t> get(Dim key, RemainingDims... keys) {
        SubHyperion* sub_map = get_sub_map(key);
        if (!sub_map) {
            return std::nullopt;
        }
        return sub_map->get(keys...);
    }
};


// Template Specialization: The base case for the final dimension.
template <typename Dim>
class Hyperion<Dim> {
private:
    using Map = LayeredSlotMap<uint64_t>; // ValueType is the final uint64_t value.
    Map map;

public:
    explicit Hyperion(MemoryManager& mem_manager) : map(mem_manager) {}

    void insert(Dim key, uint64_t value) {
        map.insert(key, value);
    }

    std::optional<uint64_t> get(Dim key) {
        return map.get(key);
    }

    std::vector<uint64_t> scan(Dim start, Dim end) {
        return map.scan(start, end);
    }
};


// Top-level wrapper class that owns the memory and provides the public API.
template <typename... Dims>
class HyperionIndex {
public:
    using Root = Hyperion<Dims...>;

    explicit HyperionIndex(size_t memory_size) : mem(memory_size), root(mem) {}

    // Inserts a multi-dimensional key and its value.
    void insert(Dims... keys, uint64_t value) {
        root.insert(keys..., value);
    }

    // Gets a value by its full multi-dimensional key.
    std::optional<uint64_t> get(Dims... keys) {
        return root.get(keys...);
    }

    // Provides access to the root for query chaining (get_sub_map).
    Root* get_root() {
        return &root;
    }

    size_t get_mem_usage() const {
        return mem.get_allocated_size();
    }

private:
    MemoryManager mem;
    Root root;
};


#include <string_view>
#include <array>
#include <cstring>

// --- Benchmarking Utilities ---

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    double stop_ns() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

void print_header(const std::string& title) {
    std::cout << "\n\n--- " << title << " ---" << std::endl;
}

void print_result(const std::string& metric, double ns_per_op, size_t mem_bytes_per_op = 0) {
    std::cout << std::left << std::setw(20) << metric << ": "
              << std::right << std::setw(8) << std::fixed << std::setprecision(2) << ns_per_op << " ns/op";
    if (mem_bytes_per_op > 0) {
        std::cout << " | " << mem_bytes_per_op << " bytes/op";
    }
    std::cout << std::endl;
}

// --- Benchmark Implementations ---

#include <set>

void benchmark_multidim(size_t num_keys, std::mt19937_64& rng) {
    print_header("Multi-Dimensional Query (3D Key)");

    HyperionIndex<uint64_t, uint64_t, uint64_t> index(num_keys * 250);
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> keys(num_keys);
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> unique_keys;
    for (size_t i = 0; i < num_keys; ++i) {
        do {
            keys[i] = {rng() & OFFSET_MASK, rng() & OFFSET_MASK, rng() & OFFSET_MASK};
        } while(unique_keys.count(keys[i]));
        unique_keys.insert(keys[i]);
    }

    Timer timer;
    timer.start();
    for (size_t i = 0; i < num_keys; ++i) {
        index.insert(std::get<0>(keys[i]), std::get<1>(keys[i]), std::get<2>(keys[i]), i);
    }
    double insert_time = timer.stop_ns();
    print_result("Insert", insert_time / num_keys, index.get_mem_usage() / num_keys);

    timer.start();
    volatile size_t found_count = 0;
    size_t failed_count = 0;
    for (size_t i = 0; i < num_keys; ++i) {
        auto val = index.get(std::get<0>(keys[i]), std::get<1>(keys[i]), std::get<2>(keys[i]));
        if (val && *val == i) {
            found_count++;
        } else {
            if (failed_count < 5) {
                 std::cerr << "  -> Get failed for 3D key. Expected " << i << ", got " << (val ? std::to_string(*val) : "null") << std::endl;
            }
            failed_count++;
        }
    }
    double get_time = timer.stop_ns();
    if (found_count != num_keys) {
        std::cerr << "3D GET VERIFICATION FAILED! (" << failed_count << " of " << num_keys << " failed)" << std::endl;
    }
    print_result("Get (Full Key)", get_time / num_keys);

    // Prefix Scan
    timer.start();
    found_count = 0;
    for (size_t i = 0; i < num_keys; ++i) {
        if (index.get_root()->get_sub_map(std::get<0>(keys[i]))) {
            found_count++;
        }
    }
    double prefix_time = timer.stop_ns();
    print_result("Prefix Scan (1/3)", prefix_time / num_keys);

    // Range Scan
    const size_t SCAN_SIZE = 1000;
    auto final_map = index.get_root()->get_sub_map(std::get<0>(keys[0]))->get_sub_map(std::get<1>(keys[0]));
    uint64_t scan_start_key = 1ULL << 50;
    for (size_t i = 0; i < SCAN_SIZE; ++i) {
        final_map->insert(scan_start_key + i, i);
    }
    timer.start();
    auto results = final_map->scan(scan_start_key, scan_start_key + SCAN_SIZE - 1);
    double scan_time = timer.stop_ns();
    if (results.size() != SCAN_SIZE) std::cerr << "RANGE SCAN VERIFICATION FAILED!" << std::endl;
    print_result("Range Scan (per key)", scan_time / results.size());
}

std::string generate_random_string(std::mt19937_64& rng) {
    std::string s(32, '\0');
    for (int i = 0; i < 4; ++i) {
        uint64_t r = rng() & OFFSET_MASK;
        memcpy(&s[i*8], &r, 8);
    }
    return s;
}

std::array<uint64_t, 4> string_to_chunks(const std::string& str) {
    std::array<uint64_t, 4> chunks;
    memcpy(chunks.data(), str.data(), 32);
    return chunks;
}

void benchmark_string_interning(size_t num_strings, std::mt19937_64& rng) {
    print_header("String Interning (32-byte strings)");
    // NOTE: 4D trie with low fanout is very memory intensive.
    // We use a larger memory multiplier and fewer strings for this test.
    const size_t string_test_size = 10000;
    HyperionIndex<uint64_t, uint64_t, uint64_t, uint64_t> index(string_test_size * 600);
    std::vector<std::string> strings(string_test_size);
    std::set<std::string> unique_strings;
    for(size_t i = 0; i < string_test_size; ++i) {
        strings[i] = generate_random_string(rng);
        while(unique_strings.count(strings[i])) {
            strings[i] = generate_random_string(rng);
        }
        unique_strings.insert(strings[i]);
    }

    Timer timer;
    timer.start();
    for (size_t i = 0; i < string_test_size; ++i) {
        auto chunks = string_to_chunks(strings[i]);
        index.insert(chunks[0], chunks[1], chunks[2], chunks[3], i);
    }
    double insert_time = timer.stop_ns();
    print_result("Intern (Insert)", insert_time / string_test_size, index.get_mem_usage() / string_test_size);

    timer.start();
    volatile size_t found_count = 0;
    size_t failed_count = 0;
    for (size_t i = 0; i < string_test_size; ++i) {
        auto chunks = string_to_chunks(strings[i]);
        auto val = index.get(chunks[0], chunks[1], chunks[2], chunks[3]);
        if (val && *val == i) {
            found_count++;
        } else {
            failed_count++;
        }
    }
    double get_time = timer.stop_ns();
    if (found_count != string_test_size) {
        std::cerr << "STRING GET VERIFICATION FAILED! (" << failed_count << " failed)" << std::endl;
    }
    print_result("Lookup", get_time / string_test_size);
}

void benchmark_graph(size_t num_triples, std::mt19937_64& rng) {
    print_header("Graph Indexing (SPO)");
    HyperionIndex<uint64_t, uint64_t, uint64_t> index(num_triples * 250);
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> triples(num_triples);
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> unique_triples;
    for (size_t i = 0; i < num_triples; ++i) {
        do {
            triples[i] = {rng() & OFFSET_MASK, rng() & OFFSET_MASK, rng() & OFFSET_MASK};
        } while(unique_triples.count(triples[i]));
        unique_triples.insert(triples[i]);
    }

    Timer timer;
    timer.start();
    for (size_t i = 0; i < num_triples; ++i) {
        index.insert(std::get<0>(triples[i]), std::get<1>(triples[i]), std::get<2>(triples[i]), 1);
    }
    double insert_time = timer.stop_ns();
    print_result("Triple Insert", insert_time / num_triples, index.get_mem_usage() / num_triples);

    timer.start();
    volatile size_t found_count = 0;
    for (size_t i = 0; i < num_triples; ++i) {
        if (index.get(std::get<0>(triples[i]), std::get<1>(triples[i]), std::get<2>(triples[i]))) {
            found_count++;
        }
    }
    double get_time = timer.stop_ns();
    if (found_count != num_triples) std::cerr << "GRAPH GET VERIFICATION FAILED!" << std::endl;
    print_result("Triple Lookup", get_time / num_triples);

    timer.start();
    volatile size_t found_count_prefix = 0;
    for (size_t i = 0; i < num_triples; ++i) {
        if (index.get_root()->get_sub_map(std::get<0>(triples[i]))) {
            found_count_prefix++;
        }
    }
    double prefix_time = timer.stop_ns();
    print_result("Query (S,?,?)", prefix_time / num_triples);
}


int main() {
    try {
        const size_t NUM_ITEMS = 100000;
        std::mt19937_64 rng(12345); // Fixed seed for reproducibility

        benchmark_multidim(NUM_ITEMS, rng);
        benchmark_string_interning(NUM_ITEMS, rng);
        benchmark_graph(NUM_ITEMS, rng);

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
