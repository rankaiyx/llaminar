#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>

#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "kernels/cpu/gemm/CPUQuantisedGemmKernel.h"
#include "utils/MPIContext.h"

using namespace llaminar2;
using namespace llaminar2::gemm;

class QuantisedGemmPackingPerf : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize random seed
        std::srand(42);
    }

    // Helper to create random data
    std::vector<uint8_t> create_random_data(size_t size)
    {
        std::vector<uint8_t> data(size);
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = static_cast<uint8_t>(std::rand() % 256);
        }
        return data;
    }

    struct PackingStats
    {
        std::string format_name;
        double mean_ms;
        double stddev_ms;
        double throughput_gbps;
    };

    template <typename TensorType, typename BlockType>
    PackingStats benchmark_packing(const std::string &format_name, size_t rows, size_t cols, int iters = 5)
    {
        size_t block_size = BlockType::BLOCK_SIZE; // e.g. 256 for Q4_K
        size_t blocks_per_row = (cols + block_size - 1) / block_size;
        size_t total_bytes = rows * blocks_per_row * sizeof(BlockType);

        // Create random data
        auto raw_data = create_random_data(total_bytes);

        // Create tensor
        TensorType tensor({rows, cols}, raw_data);

        std::vector<double> times_ms;
        times_ms.reserve(iters);

        for (int i = 0; i < iters; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();

            // Measure constructor time (which does packing)
            CPUQuantisedGemmKernel kernel(&tensor);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times_ms.push_back(ms);
        }

        // Compute stats
        double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();

        double variance = 0.0;
        for (double t : times_ms)
        {
            variance += (t - mean_ms) * (t - mean_ms);
        }
        double stddev_ms = std::sqrt(variance / times_ms.size());

        // Throughput: original tensor size / time
        // Note: This is "effective" throughput (how fast we process the weights)
        // Using raw data size as proxy for weight size
        double throughput_gbps = (total_bytes / (mean_ms / 1000.0)) / 1e9;

        return PackingStats{format_name, mean_ms, stddev_ms, throughput_gbps};
    }
};

TEST_F(QuantisedGemmPackingPerf, PackingBenchmark)
{
    // Use a reasonably large matrix: 4096 x 4096
    // 4096 * 4096 = 16M elements
    // For Q4_K (4.5 bits/element approx), ~9MB
    const size_t ROWS = 4096;
    const size_t COLS = 4096;
    const int ITERS = 10;

    std::cout << "\n=== CPUQuantisedGemmKernel Packing Benchmark ===" << std::endl;
    std::cout << "Matrix: " << ROWS << " x " << COLS << " (" << (ROWS * COLS / 1e6) << " M elements)" << std::endl;
    std::cout << "Iterations: " << ITERS << std::endl;
    std::cout << "\n";

    // Table header
    std::cout << "╔════════════════╦═══════════════════════════════╗" << std::endl;
    std::cout << "║ Format         ║ Time (ms)     │ Throughput    ║" << std::endl;
    std::cout << "╠════════════════╬═══════════════╪═══════════════╣" << std::endl;

    auto print_row = [](const PackingStats &stats)
    {
        std::cout << "║ " << std::left << std::setw(14) << stats.format_name << " ║ "
                  << std::right << std::setw(8) << std::fixed << std::setprecision(2) << stats.mean_ms
                  << " ±" << std::setw(4) << std::setprecision(2) << stats.stddev_ms << " │ "
                  << std::setw(8) << std::setprecision(2) << stats.throughput_gbps << " GB/s ║" << std::endl;
    };

    // Benchmark Q4_K
    auto stats_q4k = benchmark_packing<Q4_KTensor, Q4_KBlock>("Q4_K", ROWS, COLS, ITERS);
    print_row(stats_q4k);

    // Benchmark Q6_K
    auto stats_q6k = benchmark_packing<Q6_KTensor, Q6_KBlock>("Q6_K", ROWS, COLS, ITERS);
    print_row(stats_q6k);

    // Benchmark IQ4_XS
    auto stats_iq4xs = benchmark_packing<IQ4_XSTensor, IQ4_XSBlock>("IQ4_XS", ROWS, COLS, ITERS);
    print_row(stats_iq4xs);

    std::cout << "╚════════════════╩═══════════════╧═══════════════╝" << std::endl;
}
