/**
 * @file Perf__GemmSweep.cpp
 * @brief Parameter sweep for Q8_1 GEMM tuning
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <cstdlib>

// V2 includes
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;

struct BenchmarkConfig
{
    int seq_len;             ///< Sequence length (m dimension)
    int in_features;         ///< Input feature dimension (k dimension)
    int out_features;        ///< Output feature dimension (n dimension)
    int warmup_iters;        ///< Number of warmup iterations
    int bench_iters;         ///< Number of timed benchmark iterations per trial
    int num_trials;          ///< Number of independent trials for statistics
    std::string description; ///< Human-readable description
};

struct BenchmarkStats
{
    double mean_ms;     ///< Mean time per iteration (ms)
    double mean_gflops; ///< Mean throughput (GFLOPS)
};

class GemmSweep : public ::testing::Test
{
protected:
    int rank_ = 0;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    }

    BenchmarkStats run_benchmark(const BenchmarkConfig &config)
    {
        int M = config.seq_len;
        int K = config.in_features;
        int N = config.out_features;

        // 1. Create random weights (N x K)
        std::vector<float> weights_fp32(N * K);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &x : weights_fp32)
            x = dist(gen);

        auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

        // 2. Create kernel (this will pick up current DebugEnv settings)
        auto kernel = weights_tensor->createGemm();
        if (!kernel)
            throw std::runtime_error("Failed to create GEMM kernel");

        // 3. Create random input A (M x K)
        std::vector<float> A_fp32(M * K);
        for (auto &x : A_fp32)
            x = dist(gen);

        auto A_tensor = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
        }

        // Benchmark trials
        std::vector<double> trial_times_ms;
        for (int t = 0; t < config.num_trials; ++t)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.bench_iters; ++i)
            {
                kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();

            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            trial_times_ms.push_back(total_ms / config.bench_iters);
        }

        double sum = std::accumulate(trial_times_ms.begin(), trial_times_ms.end(), 0.0);
        double mean_ms = sum / config.num_trials;
        double ops = 2.0 * M * N * K;
        double mean_gflops = (ops / (mean_ms * 1e-3)) / 1e9;

        return {mean_ms, mean_gflops};
    }
};

TEST_F(GemmSweep, PrefetchSweep)
{
    if (rank_ == 0)
    {
        std::cout << "\n=== Prefetch Distance Sweep ===\n";
        std::cout << std::setw(10) << "Prefetch" << " | "
                  << std::setw(15) << "M=128 (GFLOPS)" << " | "
                  << std::setw(15) << "M=512 (GFLOPS)" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    std::vector<int> distances = {0, 1, 2, 4, 8};

    for (int dist : distances)
    {
        // Set env var and reload
        std::string env_val = std::to_string(dist);
        setenv("LLAMINAR_GEMM_JIT_PREFETCH_DISTANCE", env_val.c_str(), 1);
        mutableDebugEnv().gemm.reload();

        // M=128
        BenchmarkConfig config128{
            .seq_len = 128, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
        auto stats128 = run_benchmark(config128);

        // M=512
        BenchmarkConfig config512{
            .seq_len = 512, .in_features = 4096, .out_features = 4096, .warmup_iters = 2, .bench_iters = 10, .num_trials = 3, .description = ""};
        auto stats512 = run_benchmark(config512);

        if (rank_ == 0)
        {
            std::cout << std::setw(10) << dist << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats128.mean_gflops << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats512.mean_gflops << "\n";
        }
    }
}

TEST_F(GemmSweep, UnrollSweep)
{
    if (rank_ == 0)
    {
        std::cout << "\n=== Unroll N Sweep ===\n";
        std::cout << std::setw(10) << "Unroll" << " | "
                  << std::setw(15) << "M=128 (GFLOPS)" << " | "
                  << std::setw(15) << "M=512 (GFLOPS)" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    std::vector<int> unrolls = {4, 8};

    for (int unroll : unrolls)
    {
        std::string env_val = std::to_string(unroll);
        setenv("LLAMINAR_GEMM_JIT_UNROLL_N", env_val.c_str(), 1);
        mutableDebugEnv().gemm.reload();

        // M=128
        BenchmarkConfig config128{
            .seq_len = 128, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
        auto stats128 = run_benchmark(config128);

        // M=512
        BenchmarkConfig config512{
            .seq_len = 512, .in_features = 4096, .out_features = 4096, .warmup_iters = 2, .bench_iters = 10, .num_trials = 3, .description = ""};
        auto stats512 = run_benchmark(config512);

        if (rank_ == 0)
        {
            std::cout << std::setw(10) << unroll << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats128.mean_gflops << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats512.mean_gflops << "\n";
        }
    }
}

TEST_F(GemmSweep, KUnrollSweep)
{
    if (rank_ == 0)
    {
        std::cout << "\n=== K-Loop Unroll Sweep ===\n";
        std::cout << std::setw(10) << "K-Unroll" << " | "
                  << std::setw(15) << "M=128 (GFLOPS)" << " | "
                  << std::setw(15) << "M=512 (GFLOPS)" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    std::vector<int> k_unrolls = {1, 2, 4};

    for (int k_unroll : k_unrolls)
    {
        std::string env_val = std::to_string(k_unroll);
        setenv("LLAMINAR_GEMM_JIT_UNROLL_K", env_val.c_str(), 1);
        mutableDebugEnv().gemm.reload();

        BenchmarkConfig config128{
            .seq_len = 128, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
        auto stats128 = run_benchmark(config128);

        BenchmarkConfig config512{
            .seq_len = 512, .in_features = 4096, .out_features = 4096, .warmup_iters = 2, .bench_iters = 10, .num_trials = 3, .description = ""};
        auto stats512 = run_benchmark(config512);

        if (rank_ == 0)
        {
            std::cout << std::setw(10) << k_unroll << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats128.mean_gflops << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats512.mean_gflops << "\n";
        }
    }

    // Reset to default
    setenv("LLAMINAR_GEMM_JIT_UNROLL_K", "1", 1);
    mutableDebugEnv().gemm.reload();
}

TEST_F(GemmSweep, ScheduleSweep)
{
    if (rank_ == 0)
    {
        std::cout << "\n=== OMP Schedule Sweep ===\n";
        std::cout << std::setw(10) << "Schedule" << " | "
                  << std::setw(15) << "M=128 (GFLOPS)" << " | "
                  << std::setw(15) << "M=512 (GFLOPS)" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    std::vector<std::pair<std::string, int>> schedules = {{"static", 0}, {"dynamic", 1}};

    for (auto &[name, val] : schedules)
    {
        setenv("LLAMINAR_GEMM_DYNAMIC_SCHEDULE", std::to_string(val).c_str(), 1);
        mutableDebugEnv().gemm.reload();

        BenchmarkConfig config128{
            .seq_len = 128, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
        auto stats128 = run_benchmark(config128);

        BenchmarkConfig config512{
            .seq_len = 512, .in_features = 4096, .out_features = 4096, .warmup_iters = 2, .bench_iters = 10, .num_trials = 3, .description = ""};
        auto stats512 = run_benchmark(config512);

        if (rank_ == 0)
        {
            std::cout << std::setw(10) << name << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats128.mean_gflops << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats512.mean_gflops << "\n";
        }
    }

    // Reset to default (static)
    setenv("LLAMINAR_GEMM_DYNAMIC_SCHEDULE", "0", 1);
    mutableDebugEnv().gemm.reload();
}

TEST_F(GemmSweep, MBlockingSweep)
{
    if (rank_ == 0)
    {
        std::cout << "\n=== M-Blocking Sweep (B-matrix reuse) ===\n";
        std::cout << std::setw(10) << "M-Block" << " | "
                  << std::setw(15) << "M=128 (GFLOPS)" << " | "
                  << std::setw(15) << "M=512 (GFLOPS)" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    std::vector<int> m_blocks = {1, 2, 4};

    for (int m_block : m_blocks)
    {
        setenv("LLAMINAR_GEMM_JIT_M_BLOCKING", std::to_string(m_block).c_str(), 1);
        mutableDebugEnv().gemm.reload();

        BenchmarkConfig config128{
            .seq_len = 128, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
        auto stats128 = run_benchmark(config128);

        BenchmarkConfig config512{
            .seq_len = 512, .in_features = 4096, .out_features = 4096, .warmup_iters = 2, .bench_iters = 10, .num_trials = 3, .description = ""};
        auto stats512 = run_benchmark(config512);

        if (rank_ == 0)
        {
            std::cout << std::setw(10) << m_block << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats128.mean_gflops << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats512.mean_gflops << "\n";
        }
    }

    // Reset to default
    setenv("LLAMINAR_GEMM_JIT_M_BLOCKING", "1", 1);
    mutableDebugEnv().gemm.reload();
}

TEST_F(GemmSweep, NTileSweep)
{
    if (rank_ == 0)
    {
        std::cout << "\n=== N-Tile Sweep (cache blocking on N dimension) ===\n";
        std::cout << std::setw(10) << "N-Tile" << " | "
                  << std::setw(15) << "M=128 (GFLOPS)" << " | "
                  << std::setw(15) << "M=512 (GFLOPS)" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    // N-tile values: 0 = no tiling, 256/512/1024/2048 for cache blocking
    std::vector<int> n_tiles = {0, 256, 512, 1024, 2048};

    for (int n_tile : n_tiles)
    {
        setenv("LLAMINAR_GEMM_N_TILE", std::to_string(n_tile).c_str(), 1);
        mutableDebugEnv().gemm.reload();

        BenchmarkConfig config128{
            .seq_len = 128, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
        auto stats128 = run_benchmark(config128);

        BenchmarkConfig config512{
            .seq_len = 512, .in_features = 4096, .out_features = 4096, .warmup_iters = 2, .bench_iters = 10, .num_trials = 3, .description = ""};
        auto stats512 = run_benchmark(config512);

        if (rank_ == 0)
        {
            std::string n_tile_str = (n_tile == 0) ? "none" : std::to_string(n_tile);
            std::cout << std::setw(10) << n_tile_str << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats128.mean_gflops << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats512.mean_gflops << "\n";
        }
    }

    // Reset to default
    setenv("LLAMINAR_GEMM_N_TILE", "0", 1);
    mutableDebugEnv().gemm.reload();
}

TEST_F(GemmSweep, ComprehensiveSweep)
{
    // Run a comprehensive sweep across all key parameters to find optimal config
    if (rank_ == 0)
    {
        std::cout << "\n=== Comprehensive Parameter Sweep ===\n";
        std::cout << "Testing best combinations for M=512, N=4096, K=4096\n";
        std::cout << std::setw(12) << "N-Unroll" << " | "
                  << std::setw(10) << "N-Tile" << " | "
                  << std::setw(15) << "GFLOPS" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    double best_gflops = 0;
    int best_unroll = 4;
    int best_n_tile = 0;

    std::vector<int> unrolls = {4, 8};
    std::vector<int> n_tiles = {0, 512};

    for (int unroll : unrolls)
    {
        for (int n_tile : n_tiles)
        {
            setenv("LLAMINAR_GEMM_JIT_UNROLL_N", std::to_string(unroll).c_str(), 1);
            setenv("LLAMINAR_GEMM_N_TILE", std::to_string(n_tile).c_str(), 1);
            mutableDebugEnv().gemm.reload();

            BenchmarkConfig config{
                .seq_len = 512, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
            auto stats = run_benchmark(config);

            if (rank_ == 0)
            {
                std::string n_tile_str = (n_tile == 0) ? "none" : std::to_string(n_tile);
                std::cout << std::setw(12) << unroll << " | "
                          << std::setw(10) << n_tile_str << " | "
                          << std::setw(15) << std::fixed << std::setprecision(2) << stats.mean_gflops << "\n";

                if (stats.mean_gflops > best_gflops)
                {
                    best_gflops = stats.mean_gflops;
                    best_unroll = unroll;
                    best_n_tile = n_tile;
                }
            }
        }
    }

    if (rank_ == 0)
    {
        std::cout << std::string(50, '-') << "\n";
        std::cout << "Best config: N-Unroll=" << best_unroll << ", N-Tile="
                  << (best_n_tile == 0 ? "none" : std::to_string(best_n_tile))
                  << " (" << std::fixed << std::setprecision(2) << best_gflops << " GFLOPS)\n";
    }

    // Reset to defaults
    setenv("LLAMINAR_GEMM_JIT_UNROLL_N", "4", 1);
    setenv("LLAMINAR_GEMM_N_TILE", "0", 1);
    mutableDebugEnv().gemm.reload();
}

TEST_F(GemmSweep, BatchSizeSweep)
{
    // Test different batch sizes (M) with optimal config
    if (rank_ == 0)
    {
        std::cout << "\n=== Batch Size Sweep (M dimension) ===\n";
        std::cout << "Using N-Unroll=8, no N-tiling (optimal from comprehensive sweep)\n";
        std::cout << std::setw(8) << "M" << " | "
                  << std::setw(15) << "GFLOPS" << " | "
                  << std::setw(15) << "Throughput" << "\n";
        std::cout << std::string(50, '-') << "\n";
    }

    setenv("LLAMINAR_GEMM_JIT_UNROLL_N", "8", 1);
    setenv("LLAMINAR_GEMM_N_TILE", "0", 1);
    mutableDebugEnv().gemm.reload();

    std::vector<int> batch_sizes = {1, 8, 32, 64, 128, 256, 512, 1024};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config{
            .seq_len = m, .in_features = 4096, .out_features = 4096, .warmup_iters = 5, .bench_iters = 20, .num_trials = 3, .description = ""};
        auto stats = run_benchmark(config);

        if (rank_ == 0)
        {
            double throughput_tok_s = m / (stats.mean_ms / 1000.0);
            std::cout << std::setw(8) << m << " | "
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats.mean_gflops << " | "
                      << std::setw(12) << std::fixed << std::setprecision(0) << throughput_tok_s << " tok/s\n";
        }
    }

    // Reset to defaults
    setenv("LLAMINAR_GEMM_JIT_UNROLL_N", "4", 1);
    mutableDebugEnv().gemm.reload();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
