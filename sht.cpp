#include "stax_dimensional.h"
#include <array>
#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <thread>
#include <iomanip>

// =================================================================================================
// --- UTILITY & DATA GENERATION ---
// =================================================================================================

// Generates 1D keys, either sequential or random
std::vector<uint64_t> generate_1d_keys(size_t num_keys, bool random) {
    std::vector<uint64_t> keys(num_keys);
    std::iota(keys.begin(), keys.end(), 1); // Start from 1 to avoid 0
    if (random) {
        std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::shuffle(keys.begin(), keys.end(), rng);
    }
    return keys;
}

// Generates n-dimensional keys
template<uint32_t D>
std::vector<std::array<uint64_t, D>> generate_nd_keys(size_t num_keys) {
    std::vector<std::array<uint64_t, D>> keys(num_keys);
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<uint64_t> dist;
    for (size_t i = 0; i < num_keys; ++i) {
        for (uint32_t d = 0; d < D; ++d) {
            keys[i][d] = dist(rng);
        }
    }
    return keys;
}

// Simple helper to print benchmark headers
void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "--- " << title << " ---\n";
    std::cout << std::string(70, '-') << std::endl;
}

// =================================================================================================
// --- 1D BENCHMARK (Lexicographical Heuristic) ---
// =================================================================================================

void run_benchmark_1d(size_t num_keys, bool is_random) {
    const uint32_t D = 1;
    const std::string key_type = is_random ? "Random" : "Sequential";
    print_header("1D Benchmark (" + std::to_string(num_keys) + " keys, " + key_type + ")");

    auto keys = generate_1d_keys(num_keys, is_random);
    std::string value_str = "test_val";

    // --- Setup StaxDimensionStore ---
    std::atomic<uint32_t> root_ptr = 0;
    NodeManager node_manager;
    ValueStore value_store;
    RecordManager record_manager(&value_store, D);
    StaxDimensionStore<16> store(node_manager, record_manager, root_ptr, D, SplittingHeuristic::LEXICOGRAPHICAL);

    // --- Insert Benchmark ---
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        store.insert(&key, value_str);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_op = (double)duration.count() / num_keys;
    std::cout << "[INSERT] Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_op << " ns/op" << std::endl;

    // --- Get Benchmark ---
    size_t found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        if (store.get(&key) != nullptr) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    ns_per_op = (double)duration.count() / num_keys;
    std::cout << "[GET]      Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_op << " ns/op" << std::endl;

    if (found_count != num_keys) {
        std::cerr << "  [ERROR] Verification failed! Found " << found_count << "/" << num_keys << " keys." << std::endl;
    } else {
        std::cout << "  [OK] Verification passed." << std::endl;
    }

    // --- Memory Usage ---
    size_t node_mem = node_manager.get_allocated_size();
    size_t record_mem = record_manager.get_allocated_size();
    size_t value_mem = value_store.get_allocated_size();
    size_t total_mem = node_mem + record_mem + value_mem;
    std::cout << "[MEMORY]   Total: " << std::fixed << std::setprecision(2) << total_mem / (1024.0 * 1024.0) << " MB"
              << " (" << (double)total_mem / num_keys << " bytes/key)" << std::endl;
}

// =================================================================================================
// --- N-D BENCHMARK (Adaptive & K-Cyclic Heuristics) ---
// =================================================================================================

template<uint32_t D>
void run_benchmark_nd(size_t num_keys, SplittingHeuristic heuristic) {
    std::string heuristic_name = (heuristic == SplittingHeuristic::ADAPTIVE) ? "ADAPTIVE" : "K_DIMENSIONAL_CYCLIC";
    print_header(std::to_string(D) + "D Benchmark (" + std::to_string(num_keys) + " keys, " + heuristic_name + ")");

    auto keys = generate_nd_keys<D>(num_keys);
    std::string value_str = "test_val";

    // --- Setup StaxDimensionStore ---
    std::atomic<uint32_t> root_ptr = 0;
    NodeManager node_manager;
    ValueStore value_store;
    RecordManager record_manager(&value_store, D);
    StaxDimensionStore<16> store(node_manager, record_manager, root_ptr, D, heuristic);

    // --- Insert Benchmark ---
    auto start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        store.insert(key.data(), value_str);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    double ns_per_op = (double)duration.count() / num_keys;
    std::cout << "[INSERT] Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_op << " ns/op" << std::endl;

    // --- Get Benchmark ---
    size_t found_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (const auto& key : keys) {
        if (store.get(key.data()) != nullptr) {
            found_count++;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    ns_per_op = (double)duration.count() / num_keys;
    std::cout << "[GET]      Avg Latency: " << std::fixed << std::setprecision(2) << ns_per_op << " ns/op" << std::endl;

    if (found_count != num_keys) {
        std::cerr << "  [ERROR] Verification failed! Found " << found_count << "/" << num_keys << " keys." << std::endl;
    } else {
        std::cout << "  [OK] Verification passed." << std::endl;
    }
}

// =================================================================================================
// --- CONCURRENCY BENCHMARK ---
// =================================================================================================

void run_benchmark_threaded(size_t num_keys, unsigned int num_threads) {
    const uint32_t D = 1;
    print_header("Concurrency Benchmark (" + std::to_string(num_keys) + " keys, " + std::to_string(num_threads) + " threads)");

    auto keys = generate_1d_keys(num_keys, true);
    std::string value_str = "test_val";

    // --- Setup StaxDimensionStore (shared across threads) ---
    std::atomic<uint32_t> root_ptr = 0;
    NodeManager node_manager;
    ValueStore value_store;
    RecordManager record_manager(&value_store, D);
    StaxDimensionStore<16> store(node_manager, record_manager, root_ptr, D, SplittingHeuristic::LEXICOGRAPHICAL);

    std::vector<std::thread> threads;
    std::vector<std::vector<uint64_t>> thread_keys(num_threads);

    // Distribute keys among threads
    for (size_t i = 0; i < num_keys; ++i) {
        thread_keys[i % num_threads].push_back(keys[i]);
    }

    // --- Insert Benchmark ---
    auto start_time = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (const auto& key : thread_keys[i]) {
                store.insert(&key, value_str);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double ops_per_sec = (double)num_keys / (duration.count() / 1000.0);
    std::cout << "[INSERT] Throughput: " << std::fixed << std::setprecision(0) << ops_per_sec << " ops/sec" << std::endl;

    // --- Get Benchmark ---
    threads.clear();
    std::atomic<size_t> total_found = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t found_count = 0;
            for (const auto& key : thread_keys[i]) {
                if (store.get(&key) != nullptr) {
                    found_count++;
                }
            }
            total_found += found_count;
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    ops_per_sec = (double)num_keys / (duration.count() / 1000.0);
    std::cout << "[GET]      Throughput: " << std::fixed << std::setprecision(0) << ops_per_sec << " ops/sec" << std::endl;

    if (total_found != num_keys) {
        std::cerr << "  [ERROR] Verification failed! Found " << total_found << "/" << num_keys << " keys." << std::endl;
    } else {
        std::cout << "  [OK] Verification passed." << std::endl;
    }
}


// =================================================================================================
// --- MAIN ---
// =================================================================================================

int main() {
    try {
        const size_t NUM_KEYS_SMALL = 100000;
        const size_t NUM_KEYS_LARGE = 1000000;
        const unsigned int NUM_THREADS = std::thread::hardware_concurrency();

        std::cout << "StaxDimensionStore Benchmark Suite" << std::endl;
        std::cout << "Detected " << NUM_THREADS << " hardware threads." << std::endl;

        // --- 1D Benchmarks ---
        run_benchmark_1d(NUM_KEYS_LARGE, false); // Sequential
        run_benchmark_1d(NUM_KEYS_LARGE, true);  // Random

        // --- N-D Benchmarks ---
        const uint32_t D = 4;
        run_benchmark_nd<D>(NUM_KEYS_SMALL, SplittingHeuristic::ADAPTIVE);
        run_benchmark_nd<D>(NUM_KEYS_SMALL, SplittingHeuristic::K_DIMENSIONAL_CYCLIC);

        // --- Concurrency Benchmarks ---
        run_benchmark_threaded(NUM_KEYS_LARGE, NUM_THREADS);

    } catch (const std::exception& e) {
        std::cerr << "\nAn error occurred: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Benchmark finished successfully." << std::endl;
    return 0;
}
