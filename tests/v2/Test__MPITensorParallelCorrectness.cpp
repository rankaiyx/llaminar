/**
 * @file Test__MPITensorParallelCorrectness.cpp
 * @brief Correctness tests for MPI tensor-parallel attention (Phase 3a)
 * @author David Sanftenberg
 *
 * These tests validate that MPI tensor-parallel attention produces
 * numerically identical results to single-rank baseline attention.
 */

#include <gtest/gtest.h>
#include "../../src/v2/testing/AttentionTestHarness.h"
#include "../../src/v2/testing/MockPipeline.h"
#include "../../src/v2/utils/MPIContext.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace llaminar2::test;

/**
 * @brief Test fixture for MPI tensor-parallel correctness
 */
class MPITensorParallelCorrectness : public ::testing::Test
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
        // Ensure all ranks synchronize before cleanup
        MPI_Barrier(MPI_COMM_WORLD);

        // Reset MPI context to avoid use-after-free during global cleanup
        mpi_ctx_.reset();

        // Final barrier to ensure clean shutdown
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::shared_ptr<llaminar2::MPIContext> mpi_ctx_;
    int rank_;
    int world_size_;
};

/**
 * @brief Test: Single token correctness (decode phase)
 *
 * Validates that tensor-parallel attention produces identical results
 * to single-rank baseline for a single token (decode scenario).
 */
TEST_F(MPITensorParallelCorrectness, SingleToken_Correctness)
{
    // Test parameters
    const int n_heads = 8;
    const int n_kv_heads = 4; // GQA: 2 queries per KV head
    const int head_dim = 64;
    const int seq_len = 1; // Single token (decode)
    const bool causal = true;
    const float tolerance = 1e-4f;

    // Skip if not 2 ranks (test requires exactly 2 ranks for tensor-parallel)
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Create test harness
    AttentionTestHarness harness(n_heads, n_kv_heads, head_dim, mpi_ctx_);

    // Allocate tensors
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    std::vector<float> Q(q_size);
    std::vector<float> K(kv_size);
    std::vector<float> V(kv_size);
    std::vector<float> output_mpi(out_size);
    std::vector<float> output_baseline(out_size);

    // Initialize test data (deterministic)
    utils::initializeTestData(Q.data(), q_size, 42);
    utils::initializeTestData(K.data(), kv_size, 43);
    utils::initializeTestData(V.data(), kv_size, 44);

    // Run MPI tensor-parallel attention
    bool mpi_success = harness.runAttention(
        Q.data(), K.data(), V.data(), output_mpi.data(),
        seq_len, causal, /*use_mpi=*/true);

    ASSERT_TRUE(mpi_success) << "MPI attention failed";

    // Run baseline single-rank attention (only on rank 0)
    if (rank_ == 0)
    {
        BaselineRunner baseline(n_heads, n_kv_heads, head_dim);
        bool baseline_success = baseline.runAttention(
            Q.data(), K.data(), V.data(), output_baseline.data(),
            seq_len, causal);

        ASSERT_TRUE(baseline_success) << "Baseline attention failed";

        // Compare outputs
        auto metrics = utils::compareTensors(
            output_mpi.data(), output_baseline.data(), out_size, tolerance);

        // Print results
        if (::testing::Test::HasFailure() || metrics.max_abs_diff > tolerance)
        {
            utils::printMetrics(metrics, "SingleToken_Correctness");
        }

        // Validate correctness
        EXPECT_TRUE(metrics.passed)
            << "Max abs diff: " << metrics.max_abs_diff
            << ", Rel L2 norm: " << metrics.rel_l2_norm
            << ", Mismatches: " << metrics.num_mismatches << "/" << metrics.total_elements;

        EXPECT_LT(metrics.max_abs_diff, tolerance)
            << "Maximum absolute difference exceeds tolerance";

        EXPECT_LT(metrics.rel_l2_norm, 0.01)
            << "Relative L2 norm too large (should be << 1%)";
    }

    // Ensure both ranks finish before test cleanup
    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test: Multi-token correctness (prefill phase)
 *
 * Validates tensor-parallel attention for longer sequences.
 */
TEST_F(MPITensorParallelCorrectness, MultiToken_Correctness)
{
    // Test parameters
    const int n_heads = 8;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int seq_len = 32; // Multi-token prefill
    const bool causal = true;
    const float tolerance = 1e-4f;

    // Skip if not 2 ranks
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Create test harness
    AttentionTestHarness harness(n_heads, n_kv_heads, head_dim, mpi_ctx_);

    // Allocate tensors
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    std::vector<float> Q(q_size);
    std::vector<float> K(kv_size);
    std::vector<float> V(kv_size);
    std::vector<float> output_mpi(out_size);
    std::vector<float> output_baseline(out_size);

    // Initialize test data
    utils::initializeTestData(Q.data(), q_size, 142);
    utils::initializeTestData(K.data(), kv_size, 143);
    utils::initializeTestData(V.data(), kv_size, 144);

    // Run MPI tensor-parallel attention
    bool mpi_success = harness.runAttention(
        Q.data(), K.data(), V.data(), output_mpi.data(),
        seq_len, causal, /*use_mpi=*/true);

    ASSERT_TRUE(mpi_success) << "MPI attention failed";

    // Verify all ranks have identical output after allreduce
    if (rank_ == 0)
    {
        std::cout << "[Rank " << rank_ << "] MPI output[0]=" << output_mpi[0]
                  << " output[100]=" << output_mpi[100]
                  << " output[1000]=" << output_mpi[1000]
                  << " output[8000]=" << output_mpi[8000]
                  << " output[" << (out_size - 1) << "]=" << output_mpi[out_size - 1] << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank_ == 1)
    {
        std::cout << "[Rank " << rank_ << "] MPI output[0]=" << output_mpi[0]
                  << " output[100]=" << output_mpi[100]
                  << " output[1000]=" << output_mpi[1000]
                  << " output[8000]=" << output_mpi[8000]
                  << " output[" << (out_size - 1) << "]=" << output_mpi[out_size - 1] << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Run baseline (rank 0 only)
    if (rank_ == 0)
    {
        BaselineRunner baseline(n_heads, n_kv_heads, head_dim);
        bool baseline_success = baseline.runAttention(
            Q.data(), K.data(), V.data(), output_baseline.data(),
            seq_len, causal);

        ASSERT_TRUE(baseline_success) << "Baseline attention failed";

        std::cout << "[Rank 0] Baseline output[0]=" << output_baseline[0]
                  << " output[100]=" << output_baseline[100]
                  << " output[1000]=" << output_baseline[1000]
                  << " output[8000]=" << output_baseline[8000]
                  << " output[" << (out_size - 1) << "]=" << output_baseline[out_size - 1] << std::endl;
        std::cout << "[Rank 0] MPI      output[0]=" << output_mpi[0]
                  << " output[100]=" << output_mpi[100]
                  << " output[1000]=" << output_mpi[1000]
                  << " output[8000]=" << output_mpi[8000]
                  << " output[" << (out_size - 1) << "]=" << output_mpi[out_size - 1] << std::endl;

        // Compare outputs
        auto metrics = utils::compareTensors(
            output_mpi.data(), output_baseline.data(), out_size, tolerance);

        // Print results
        if (::testing::Test::HasFailure() || metrics.max_abs_diff > tolerance)
        {
            utils::printMetrics(metrics, "MultiToken_Correctness");
        }

        // Validate correctness
        EXPECT_TRUE(metrics.passed)
            << "Max abs diff: " << metrics.max_abs_diff
            << ", Rel L2 norm: " << metrics.rel_l2_norm;

        EXPECT_LT(metrics.max_abs_diff, tolerance);
        EXPECT_LT(metrics.rel_l2_norm, 0.01);
    }

    // Ensure both ranks finish before test cleanup
    MPI_Barrier(MPI_COMM_WORLD);
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
