#include <iostream>
#include <vector>
#include <cstdint>
#include <sys/mman.h>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <list> // For debugging, will be removed

// --- Constants ---
namespace {
    constexpr size_t PAGE_SIZE = 4096; // 4KB pages
    constexpr size_t BITS_PER_LEVEL = 4; // Use 4 bits of the key per level
    constexpr size_t FANOUT = 1 << BITS_PER_LEVEL; // 16 children per directory node
    constexpr int MAX_DEPTH = 64 / BITS_PER_LEVEL; // Max depth of the tree

    using u64 = uint64_t;
    using i64 = int64_t;

    // Offsets are used instead of raw pointers
    using NodeOffset = u64;
}

// --- Node Structures ---

// Every page starts with this header
struct NodeHeader {
    enum class Type : uint8_t {
        Directory,
        Leaf
    };
    Type type;
    uint16_t count; // Number of elements in the node
};

// Leaf nodes store sorted key-value pairs
struct LeafNode {
    NodeHeader header;
    NodeOffset prev_leaf; // For linked list of leaves
    NodeOffset next_leaf; // For linked list of leaves
    // The rest of the page is an array of KV pairs
    // KV pairs will be stored packed right after this struct
};

// The number of KV pairs that can fit in a leaf node
constexpr size_t LEAF_CAPACITY = (PAGE_SIZE - sizeof(LeafNode)) / sizeof(std::pair<u64, u64>);

// Directory nodes store pointers to child nodes
struct DirectoryNode {
    NodeHeader header;
    NodeOffset children[FANOUT];
};


// --- Memory Manager ---
class MemoryManager {
private:
    void* memory_pool_ = nullptr;
    size_t pool_size_;
    NodeOffset next_free_page_offset_ = 0;

public:
    MemoryManager(size_t size) : pool_size_(size) {
        memory_pool_ = mmap(nullptr, pool_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory_pool_ == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap memory pool");
        }
    }

    ~MemoryManager() {
        if (memory_pool_) {
            munmap(memory_pool_, pool_size_);
        }
    }

    // Allocate a new page and return its offset
    NodeOffset allocate_page() {
        if (next_free_page_offset_ + PAGE_SIZE > pool_size_) {
            throw std::runtime_error("Memory pool exhausted");
        }
        NodeOffset offset = next_free_page_offset_;
        next_free_page_offset_ += PAGE_SIZE;
        return offset;
    }

    // Get a raw pointer from an offset
    template<typename T>
    T* get_page(NodeOffset offset) const {
        if (offset >= next_free_page_offset_) {
             return nullptr; // Or throw an error
        }
        return reinterpret_cast<T*>(static_cast<char*>(memory_pool_) + offset);
    }

    size_t get_usage() const {
        return next_free_page_offset_;
    }
};

// --- Sorted Radix Hash Tree (SRHT) ---
class SRHT {
private:
    MemoryManager mem_manager_;
    NodeOffset root_offset_;
    NodeOffset head_leaf_offset_; // Head of the leaf linked list

    // --- Helper Functions ---

    // Get a pointer to the KV-pair array within a leaf node
    std::pair<u64, u64>* get_kv_pairs(LeafNode* leaf) const {
        return reinterpret_cast<std::pair<u64, u64>*>(reinterpret_cast<char*>(leaf) + sizeof(LeafNode));
    }

    // Extract the slice of bits from a key for indexing at a given depth
    size_t get_key_slice(u64 key, int depth) const {
        int shift = 64 - (depth + 1) * BITS_PER_LEVEL;
        return (key >> shift) & (FANOUT - 1);
    }

    // Find the leaf node that should contain a given key
    NodeOffset find_leaf(u64 key, int& depth_found) const {
        depth_found = 0;
        NodeOffset current_offset = root_offset_;

        while (true) {
            NodeHeader* header = mem_manager_.get_page<NodeHeader>(current_offset);
            if (header->type == NodeHeader::Type::Leaf) {
                return current_offset;
            }

            DirectoryNode* dir_node = reinterpret_cast<DirectoryNode*>(header);
            size_t slice = get_key_slice(key, depth_found);

            NodeOffset child_offset = dir_node->children[slice];
            if (child_offset == 0) {
                // This path is empty, key does not exist.
                // For a get operation, this means not found.
                // For an insert, this is where a new leaf would be created.
                return 0;
            }

            current_offset = child_offset;
            depth_found++;
        }
    }


public:
    SRHT(size_t total_mem_size) : mem_manager_(total_mem_size), head_leaf_offset_(0) {
        // Initialize the root as a leaf node
        root_offset_ = mem_manager_.allocate_page();
        LeafNode* root = mem_manager_.get_page<LeafNode>(root_offset_);
        root->header.type = NodeHeader::Type::Leaf;
        root->header.count = 0;
        root->prev_leaf = 0;
        root->next_leaf = 0;
        head_leaf_offset_ = root_offset_;
    }

    bool get(u64 key, u64& value) const {
        int depth = 0;
        NodeOffset leaf_offset = find_leaf(key, depth);

        if (leaf_offset == 0) {
            return false; // No leaf at this path
        }

        LeafNode* leaf = mem_manager_.get_page<LeafNode>(leaf_offset);
        auto* pairs = get_kv_pairs(leaf);

        // Binary search for the key in the sorted leaf
        auto it = std::lower_bound(pairs, pairs + leaf->header.count, key,
            [](const std::pair<u64, u64>& elem, u64 k) {
                return elem.first < k;
            });

        if (it != pairs + leaf->header.count && it->first == key) {
            value = it->second;
            return true;
        }

        return false;
    }

    void insert(u64 key, u64 value) {
        for (;;) { // Loop to allow retrying after a split
            std::vector<NodeOffset> path;
            NodeOffset current_offset = root_offset_;
            int depth = 0;

            while (true) {
                path.push_back(current_offset);
                NodeHeader* header = mem_manager_.get_page<NodeHeader>(current_offset);

                if (header->type == NodeHeader::Type::Directory) {
                    DirectoryNode* dir_node = reinterpret_cast<DirectoryNode*>(header);
                    size_t slice = get_key_slice(key, depth);
                    NodeOffset child_offset = dir_node->children[slice];

                    if (child_offset == 0) {
                        // Found an empty slot. Create a new leaf and insert.
                        NodeOffset new_leaf_offset = mem_manager_.allocate_page();
                        LeafNode* leaf = mem_manager_.get_page<LeafNode>(new_leaf_offset);
                        leaf->header.type = NodeHeader::Type::Leaf;
                        leaf->header.count = 1;
                        get_kv_pairs(leaf)[0] = {key, value};

                        dir_node->children[slice] = new_leaf_offset;

                        // The path passed to link_new_leaf should not include the new leaf itself,
                        // so we pass the path to the parent directory.
                        path.pop_back();
                        link_new_leaf(new_leaf_offset, path, key);
                        return; // Done
                    }
                    current_offset = child_offset;
                    depth++;
                } else { // Leaf Node
                    LeafNode* leaf = reinterpret_cast<LeafNode*>(header);
                    auto* pairs = get_kv_pairs(leaf);
                    auto it = std::lower_bound(pairs, pairs + leaf->header.count, key,
                        [](const auto& elem, u64 k) { return elem.first < k; });

                    if (it != pairs + leaf->header.count && it->first == key) {
                        it->second = value; // Update existing key
                        return;
                    }

                    if (leaf->header.count < LEAF_CAPACITY) {
                        size_t insert_pos = std::distance(pairs, it);
                        std::move_backward(pairs + insert_pos, pairs + leaf->header.count, pairs + leaf->header.count + 1);
                        pairs[insert_pos] = {key, value};
                        leaf->header.count++;
                        return;
                    } else {
                        // Leaf is full, need to split.
                        split(path, depth);
                        goto retry_insert; // Break inner loops and retry from the top
                    }
                }
            }
        retry_insert:;
        }
    }

private:
    void split(const std::vector<NodeOffset>& path, int depth) {
        NodeOffset leaf_offset = path.back();

        LeafNode* old_leaf = mem_manager_.get_page<LeafNode>(leaf_offset);

        // Use a stack-allocated array to avoid heap allocation in the critical path.
        std::pair<u64, u64> temp_pairs[LEAF_CAPACITY];
        std::copy(get_kv_pairs(old_leaf), get_kv_pairs(old_leaf) + old_leaf->header.count, temp_pairs);
        size_t num_pairs = old_leaf->header.count;

        std::sort(temp_pairs, temp_pairs + num_pairs, [](const auto& a, const auto& b){
            return a.first < b.first;
        });

        NodeOffset old_prev_offset = old_leaf->prev_leaf;
        NodeOffset old_next_offset = old_leaf->next_leaf;

        // Overwrite the leaf page with a new directory
        DirectoryNode* new_dir = mem_manager_.get_page<DirectoryNode>(leaf_offset);
        new_dir->header.type = NodeHeader::Type::Directory;
        new_dir->header.count = 0;
        std::fill(std::begin(new_dir->children), std::end(new_dir->children), 0);

        // Redistribute pairs into new leaves
        NodeOffset current_predecessor = old_prev_offset;
        for (size_t i = 0; i < num_pairs; ++i) {
            const auto& pair = temp_pairs[i];
            size_t slice = get_key_slice(pair.first, depth);
            NodeOffset& child_offset_ref = new_dir->children[slice];

            if (child_offset_ref == 0) {
                child_offset_ref = mem_manager_.allocate_page();
                LeafNode* new_leaf = mem_manager_.get_page<LeafNode>(child_offset_ref);
                new_leaf->header.type = NodeHeader::Type::Leaf;
                new_leaf->header.count = 0;

                // Link this new leaf into the list
                new_leaf->prev_leaf = current_predecessor;
                if (current_predecessor != 0) {
                    mem_manager_.get_page<LeafNode>(current_predecessor)->next_leaf = child_offset_ref;
                } else {
                    head_leaf_offset_ = child_offset_ref;
                }
                current_predecessor = child_offset_ref;
            }
            LeafNode* child_leaf = mem_manager_.get_page<LeafNode>(child_offset_ref);
            auto* pairs = get_kv_pairs(child_leaf);
            pairs[child_leaf->header.count++] = pair;
        }

        // Link the last new leaf to the original successor
        LeafNode* last_new_leaf = mem_manager_.get_page<LeafNode>(current_predecessor);
        last_new_leaf->next_leaf = old_next_offset;
        if (old_next_offset != 0) {
            mem_manager_.get_page<LeafNode>(old_next_offset)->prev_leaf = current_predecessor;
        }
    }

    void link_new_leaf(NodeOffset new_leaf_offset, const std::vector<NodeOffset>& path, u64 key) {
        NodeOffset predecessor = find_predecessor_from_path(path, key);

        LeafNode* new_leaf = mem_manager_.get_page<LeafNode>(new_leaf_offset);
        if (predecessor == 0) { // This is the new head leaf
            new_leaf->prev_leaf = 0;
            new_leaf->next_leaf = head_leaf_offset_;
            if (head_leaf_offset_ != 0) {
                mem_manager_.get_page<LeafNode>(head_leaf_offset_)->prev_leaf = new_leaf_offset;
            }
            head_leaf_offset_ = new_leaf_offset;
        } else { // Link after the predecessor
            LeafNode* pred_node = mem_manager_.get_page<LeafNode>(predecessor);
            new_leaf->next_leaf = pred_node->next_leaf;
            new_leaf->prev_leaf = predecessor;
            if (pred_node->next_leaf != 0) {
                mem_manager_.get_page<LeafNode>(pred_node->next_leaf)->prev_leaf = new_leaf_offset;
            }
            pred_node->next_leaf = new_leaf_offset;
        }
    }

    NodeOffset find_predecessor_from_path(const std::vector<NodeOffset>& path, u64 key) {
        // Traverse backwards up the path from the parent of the new leaf
        for (int d = path.size() - 1; d >= 0; --d) {
            NodeOffset current_dir_offset = path[d];
            NodeHeader* header = mem_manager_.get_page<NodeHeader>(current_dir_offset);

            // This should always be a directory, but check for safety
            if (header->type != NodeHeader::Type::Directory) continue;

            DirectoryNode* dir = reinterpret_cast<DirectoryNode*>(header);
            size_t slice_at_this_depth = get_key_slice(key, d);

            // Look for a sibling branch to the left of the key's path
            for (int i = slice_at_this_depth - 1; i >= 0; --i) {
                if (dir->children[i] != 0) {
                    // Found a subtree to the left. The predecessor is its rightmost leaf.
                    return find_rightmost_leaf(dir->children[i]);
                }
            }
        }
        // If we exit the loop, it means the key is the smallest in the tree.
        return 0;
    }

    NodeOffset find_rightmost_leaf(NodeOffset current_offset) {
        while (true) {
            NodeHeader* header = mem_manager_.get_page<NodeHeader>(current_offset);
            if (header->type == NodeHeader::Type::Leaf) {
                return current_offset;
            }
            DirectoryNode* dir = reinterpret_cast<DirectoryNode*>(header);
            bool found_child = false;
            for (int i = FANOUT - 1; i >= 0; --i) {
                if (dir->children[i] != 0) {
                    current_offset = dir->children[i];
                    found_child = true;
                    break;
                }
            }
            if (!found_child) return 0; // Empty directory? Should not happen.
        }
    }

public:
    size_t get_memory_usage() const {
        return mem_manager_.get_usage();
    }

    void range_scan(u64 start_key, u64 end_key) const {
        // Find starting leaf: for simplicity, we scan from the head.
        // A more optimized version would traverse the tree to find the starting leaf.
        int depth = 0;
        NodeOffset start_leaf_offset = find_leaf(start_key, depth);
        if (start_leaf_offset == 0) {
            // Path doesn't exist, need to find the next leaf in sequence.
            // This is slow, but correct.
            NodeOffset current = head_leaf_offset_;
            while(current != 0) {
                LeafNode* leaf = mem_manager_.get_page<LeafNode>(current);
                if (leaf->header.count > 0) {
                    auto* pairs = get_kv_pairs(leaf);
                    if (pairs[leaf->header.count - 1].first >= start_key) {
                        start_leaf_offset = current;
                        break;
                    }
                }
                current = leaf->next_leaf;
            }
        }

        if (start_leaf_offset == 0) return; // No keys in range

        // Iterate through leaves and pairs
        NodeOffset current_offset = start_leaf_offset;
        while (current_offset != 0) {
            LeafNode* leaf = mem_manager_.get_page<LeafNode>(current_offset);
            auto* pairs = get_kv_pairs(leaf);

            for (uint16_t i = 0; i < leaf->header.count; ++i) {
                const auto& pair = pairs[i];
                if (pair.first > end_key) {
                    return; // Past the end of the range
                }
                if (pair.first >= start_key) {
                    std::cout << "  " << pair.first << " -> " << pair.second << "\n";
                }
            }
            current_offset = leaf->next_leaf;
        }
    }
};


// --- Main & Benchmark ---
#include <random>
#include <vector>
#include <string>

void run_benchmark(const std::string& key_type, std::vector<u64>& keys) {
    constexpr size_t NUM_KEYS = 1'000'000;
    constexpr size_t MEM_SIZE = 2UL * 1024 * 1024 * 1024; // 2GB

    std::cout << "\n====================================================\n";
    std::cout << "  Running Benchmark for: " << key_type << " Keys\n";
    std::cout << "====================================================\n";

    SRHT tree(MEM_SIZE);

    // --- Insertion Test ---
    std::cout << "\n--- Inserting " << NUM_KEYS << " keys ---\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const auto key : keys) {
        tree.insert(key, key); // Value is the same as key
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "Total insert time: " << duration.count() / 1e9 << " seconds\n";
    std::cout << "Average insert latency: " << duration.count() / NUM_KEYS << " ns/op\n";

    // --- Verification Test ---
    std::cout << "\n--- Verifying " << NUM_KEYS << " keys ---\n";
    start_time = std::chrono::high_resolution_clock::now();
    size_t errors = 0;
    for (const auto key : keys) {
        u64 value;
        if (!tree.get(key, value) || value != key) {
            errors++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    if (errors > 0) {
        std::cerr << "Verification FAILED with " << errors << " errors.\n";
    } else {
        std::cout << "Verification PASSED.\n";
    }
    std::cout << "Total get time: " << duration.count() / 1e9 << " seconds\n";
    std::cout << "Average get latency: " << duration.count() / NUM_KEYS << " ns/op\n";

    // --- Memory Usage ---
    std::cout << "\n--- Memory Usage ---\n";
    std::cout << "Total memory used: " << tree.get_memory_usage() / (1024.0 * 1024.0) << " MB\n";

    // --- Range Scan Demo ---
    std::cout << "\n--- Demonstrating Range Scan ---\n";
    std::sort(keys.begin(), keys.end());
    u64 start_range = keys[NUM_KEYS / 2];
    u64 end_range = keys[NUM_KEYS / 2 + 10];
    std::cout << "Scanning from " << start_range << " to " << end_range << ":\n";
    tree.range_scan(start_range, end_range);
    std::cout << "Range scan complete.\n";
}

int main() {
    std::cout << "SRHT Novel Hash/Tree Design" << std::endl;
    std::cout << "Leaf capacity: " << LEAF_CAPACITY << " pairs" << std::endl;
    constexpr size_t NUM_KEYS = 1'000'000;

    try {
        // --- Run with Sequential Keys ---
        {
            std::vector<u64> sequential_keys(NUM_KEYS);
            for (size_t i = 0; i < NUM_KEYS; ++i) {
                sequential_keys[i] = i;
            }
            run_benchmark("Sequential", sequential_keys);
        }

        // --- Run with Random Keys ---
        {
            std::vector<u64> random_keys(NUM_KEYS);
            std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
            for (size_t i = 0; i < NUM_KEYS; ++i) {
                random_keys[i] = rng();
            }
            run_benchmark("Random", random_keys);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
