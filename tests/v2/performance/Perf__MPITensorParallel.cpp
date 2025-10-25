/**
 * @file Perf__MPITensorParallel.cpp
 * @brief Performance benchmarks for MPI tensor-parallel attention (Phase 3b)
 * @author David Sanftenberg
 *
 * Measures speedup and efficiency of MPI tensor-parallel attention
 * compared to single-rank baseline.
 */

#include <gtest/gtest.h>
#include "../../src/v2/testing/AttentionTestHarness.h"
#include "../../src/v2/testing/MockPipeline.h"
#include "../../src/v2/testing/PerfHarness.h"
#include "../../src/v2/utils/MPIContext.h"
#include <vector>
#include <iostream>

using namespace llaminar2::test;

/**
 * @brief Test fixture for MPI tensor-parallel performance
 */
class MPITensorParallelPerformance : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get MPI rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Create MPI context
        mpi_ctx_ = std::make_shared<llaminar2::MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
    }

    void TearDown() override
    {
        // Cleanup if needed
    }

    std::shared_ptr<llaminar2::MPIContext> mpi_ctx_;
    int rank_;
    int world_size_;
};

/**
 * @brief Performance test: Single token (decode phase)
 *
 * Measures performance for single-token attention (typical decode scenario).
 * Expected: Minimal speedup due to communication overhead dominating.
 */
TEST_F(MPITensorParallelPerformance, SingleToken_Performance)
{
    // Test parameters
    const int n_heads = 8;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int seq_len = 1; // Single token
    const bool causal = true;

    // Skip if not 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Only rank 0 runs baseline and reports results
    if (rank_ == 0)
    {
        // Create harness
        AttentionTestHarness harness(n_heads, n_kv_heads, head_dim, mpi_ctx_);
        BaselineRunner baseline(n_heads, n_kv_heads, head_dim);

        // Allocate tensors
        const size_t q_size = seq_len * n_heads * head_dim;
        const size_t kv_size = seq_len * n_kv_heads * head_dim;
        const size_t out_size = seq_len * n_heads * head_dim;

        std::vector<float> Q(q_size);
        std::vector<float> K(kv_size);
        std::vector<float> V(kv_size);
        std::vector<float> output(out_size);

        // Initialize test data
        utils::initializeTestData(Q.data(), q_size, 42);
        utils::initializeTestData(K.data(), kv_size, 43);
        utils::initializeTestData(V.data(), kv_size, 44);

        // Calculate FLOPs
        size_t flop_count = PerfHarness::calculateAttentionFLOPs(n_heads, seq_len, head_dim);

        // Benchmark baseline (single-rank)
        auto baseline_fn = [&]()
        {
            baseline.runAttention(Q.data(), K.data(), V.data(), output.data(), seq_len, causal);
        };

        std::cout << "\n[Rank 0] Benchmarking baseline (single-rank)..." << std::endl;
        auto baseline_result = PerfHarness::benchmark(baseline_fn, /*trials=*/5, /*warmup=*/3, flop_count, /*num_processes=*/1);
        PerfHarness::printResult(baseline_result, "Baseline (1 rank)");
    }

    // All ranks participate in MPI benchmark
    {
        AttentionTestHarness harness(n_heads, n_kv_heads, head_dim, mpi_ctx_);

        const size_t q_size = seq_len * n_heads * head_dim;
        const size_t kv_size = seq_len * n_kv_heads * head_dim;
        const size_t out_size = seq_len * n_heads * head_dim;

        std::vector<float> Q(q_size);
        std::vector<float> K(kv_size);
        std::vector<float> V(kv_size);
        std::vector<float> output(out_size);

        utils::initializeTestData(Q.data(), q_size, 42);
        utils::initializeTestData(K.data(), kv_size, 43);
        utils::initializeTestData(V.data(), kv_size, 44);

        size_t flop_count = PerfHarness::calculateAttentionFLOPs(n_heads, seq_len, head_dim);

        // Benchmark MPI tensor-parallel
        auto mpi_fn = [&]()
        {
            harness.runAttention(Q.data(), K.data(), V.data(), output.data(), seq_len, causal, /*use_mpi=*/true);
        };

        if (rank_ == 0)
        {
            std::cout << "\n[Rank 0] Benchmarking MPI tensor-parallel (2 ranks)..." << std::endl;
        }

        auto mpi_result = PerfHarness::benchmark(mpi_fn, /*trials=*/5, /*warmup=*/3, flop_count, /*num_processes=*/2);

        if (rank_ == 0)
        {
            PerfHarness::printResult(mpi_result, "MPI Tensor-Parallel (2 ranks)");

            std::cout << "\n=== Analysis ===" << std::endl;
            std::cout << "Expected: Minimal speedup for single token due to communication overhead" << std::endl;
            std::cout << "Note: Decode phase is latency-critical, not throughput-optimized" << std::endl;
        }
    }
}

/**
 * @brief Performance test: Multi-token (prefill phase)
 *
 * Measures performance for multi-token attention (prefill scenario).
 * Expected: Better speedup as computation dominates communication.
 */
TEST_F(MPITensorParallelPerformance, MultiToken_Performance)
{
    // Test parameters
    const int n_heads = 8;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int seq_len = 128; // Multi-token prefill
    const bool causal = true;

    // Skip if not 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Only rank 0 runs baseline and reports results
    if (rank_ == 0)
    {
        AttentionTestHarness harness(n_heads, n_kv_heads, head_dim, mpi_ctx_);
        BaselineRunner baseline(n_heads, n_kv_heads, head_dim);

        const size_t q_size = seq_len * n_heads * head_dim;
        const size_t kv_size = seq_len * n_kv_heads * head_dim;
        const size_t out_size = seq_len * n_heads * head_dim;

        std::vector<float> Q(q_size);
        std::vector<float> K(kv_size);
        std::vector<float> V(kv_size);
        std::vector<float> output(out_size);

        utils::initializeTestData(Q.data(), q_size, 142);
        utils::initializeTestData(K.data(), kv_size, 143);
        utils::initializeTestData(V.data(), kv_size, 144);

        size_t flop_count = PerfHarness::calculateAttentionFLOPs(n_heads, seq_len, head_dim);

        // Benchmark baseline
        auto baseline_fn = [&]()
        {
            baseline.runAttention(Q.data(), K.data(), V.data(), output.data(), seq_len, causal);
        };

        std::cout << "\n[Rank 0] Benchmarking baseline (single-rank, seq_len=" << seq_len << ")..." << std::endl;
        auto baseline_result = PerfHarness::benchmark(baseline_fn, /*trials=*/5, /*warmup=*/3, flop_count, 1);
        PerfHarness::printResult(baseline_result, "Baseline (1 rank)");
    }

    // All ranks participate in MPI benchmark
    {
        AttentionTestHarness harness(n_heads, n_kv_heads, head_dim, mpi_ctx_);

        const size_t q_size = seq_len * n_heads * head_dim;
        const size_t kv_size = seq_len * n_kv_heads * head_dim;
        const size_t out_size = seq_len * n_heads * head_dim;

        std::vector<float> Q(q_size);
        std::vector<float> K(kv_size);
        std::vector<float> V(kv_size);
        std::vector<float> output(out_size);

        utils::initializeTestData(Q.data(), q_size, 142);
        utils::initializeTestData(K.data(), kv_size, 143);
        utils::initializeTestData(V.data(), kv_size, 144);

        size_t flop_count = PerfHarness::calculateAttentionFLOPs(n_heads, seq_len, head_dim);

        // Benchmark MPI
        auto mpi_fn = [&]()
        {
            harness.runAttention(Q.data(), K.data(), V.data(), output.data(), seq_len, causal, /*use_mpi=*/true);
        };

        if (rank_ == 0)
        {
            std::cout << "\n[Rank 0] Benchmarking MPI tensor-parallel (2 ranks, seq_len=" << seq_len << ")..." << std::endl;
        }

        auto mpi_result = PerfHarness::benchmark(mpi_fn, /*trials=*/5, /*warmup=*/3, flop_count, 2);

        if (rank_ == 0)
        {
            PerfHarness::printResult(mpi_result, "MPI Tensor-Parallel (2 ranks)");

            std::cout << "\n=== Analysis ===" << std::endl;
            std::cout << "Expected: Better speedup for longer sequences (compute-bound)" << std::endl;
            std::cout << "Target: 1.5-1.8x speedup with 2 ranks (accounting for communication overhead)" << std::endl;
        }
    }
}

/**
 * @brief Performance test: Scaling analysis
 *
 * Measures how performance scales with sequence length.
 */
TEST_F(MPITensorParallelPerformance, ScalingAnalysis)
{
    // Test parameters
    const int n_heads = 8;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const bool causal = true;
    const std::vector<int> seq_lengths = {1, 16, 64, 256};

    // Skip if not 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    if (rank_ == 0)
    {
        std::cout << "\n=== Scaling Analysis (Sequence Length vs Speedup) ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Seq Len | Baseline (ms) | MPI (ms)  | Speedup | Efficiency" << std::endl;
        std::cout << "  --------|---------------|-----------|---------|------------" << std::endl;
    }

    for (int seq_len : seq_lengths)
    {
        double baseline_time = 0.0;
        double mpi_time = 0.0;

        // Baseline (rank 0 only)
        if (rank_ == 0)
        {
            BaselineRunner baseline(n_heads, n_kv_heads, head_dim);

            const size_t q_size = seq_len * n_heads * head_dim;
            const size_t kv_size = seq_len * n_kv_heads * head_dim;
            const size_t out_size = seq_len * n_heads * head_dim;

            std::vector<float> Q(q_size);
            std::vector<float> K(kv_size);
            std::vector<float> V(kv_size);
            std::vector<float> output(out_size);

            utils::initializeTestData(Q.data(), q_size, seq_len);
            utils::initializeTestData(K.data(), kv_size, seq_len + 1);
            utils::initializeTestData(V.data(), kv_size, seq_len + 2);

            auto baseline_fn = [&]()
            {
                baseline.runAttention(Q.data(), K.data(), V.data(), output.data(), seq_len, causal);
            };

            auto result = PerfHarness::benchmark(baseline_fn, /*trials=*/3, /*warmup=*/2);
            baseline_time = result.mean_ms;
        }

        // MPI (all ranks)
        {
            AttentionTestHarness harness(n_heads, n_kv_heads, head_dim, mpi_ctx_);

            const size_t q_size = seq_len * n_heads * head_dim;
            const size_t kv_size = seq_len * n_kv_heads * head_dim;
            const size_t out_size = seq_len * n_heads * head_dim;

            std::vector<float> Q(q_size);
            std::vector<float> K(kv_size);
            std::vector<float> V(kv_size);
            std::vector<float> output(out_size);

            utils::initializeTestData(Q.data(), q_size, seq_len);
            utils::initializeTestData(K.data(), kv_size, seq_len + 1);
            utils::initializeTestData(V.data(), kv_size, seq_len + 2);

            auto mpi_fn = [&]()
            {
                harness.runAttention(Q.data(), K.data(), V.data(), output.data(), seq_len, causal, /*use_mpi=*/true);
            };

            auto result = PerfHarness::benchmark(mpi_fn, /*trials=*/3, /*warmup=*/2);
            mpi_time = result.mean_ms;
        }

        // Report (rank 0 only)
        if (rank_ == 0 && baseline_time > 0.0 && mpi_time > 0.0)
        {
            double speedup = baseline_time / mpi_time;
            double efficiency = speedup / 2.0; // 2 ranks

            std::cout << "  " << std::setw(7) << seq_len
                      << " | " << std::setw(13) << baseline_time
                      << " | " << std::setw(9) << mpi_time
                      << " | " << std::setw(7) << speedup << "x"
                      << " | " << std::setw(10) << (efficiency * 100.0) << "%" << std::endl;
        }
    }

    if (rank_ == 0)
    {
        std::cout << "\nExpected trend: Speedup improves with longer sequences (less comm overhead)" << std::endl;
    }
}

/**
 * @brief Main entry point (MPI-aware)
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
