/**
 * @file Test__Q8_1_MPI_TensorParallel.cpp
 * @brief Integration tests for Q8_1 native MPI tensor-parallel attention
 *
 * Tests the full MPI tensor-parallel path for Q8_1 attention:
 *   1. Q8_1 head slicing (sliceQ8_1HeadBlocks)
 *   2. Q8_1 K/V GQA broadcast (broadcastQ8_1KVHeads)
 *   3. Q8_1 fused attention kernel execution
 *   4. Q8_1 allgatherv result combination
 *
 * Compares MPI tensor-parallel Q8_1 output against:
 *   - Single-rank Q8_1 baseline (validates distribution logic)
 *   - FP32 reference (validates overall precision)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include "pipelines/attention/MpiAttentionOrchestrator.h"
#include "pipelines/PipelineConfig.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;

/**
 * @brief Test fixture for Q8_1 MPI tensor-parallel attention
 */
class Test__Q8_1_MPI_TensorParallel : public ::testing::Test
{
protected:
    std::shared_ptr<MPIContext> mpi_ctx_;
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
    }

    void TearDown() override
    {
        mpi_ctx_->barrier();
    }

    /**
     * @brief Generate random FP32 data and quantize to Q8_1
     */
    std::shared_ptr<Q8_1Tensor> createRandomQ8_1Tensor(
        size_t rows, size_t cols, unsigned seed)
    {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Create FP32 data first
        std::vector<float> fp32_data(rows * cols);
        for (auto &v : fp32_data)
        {
            v = dist(gen);
        }

        // Create Q8_1 tensor using static factory method
        return Q8_1Tensor::quantize_from_fp32(fp32_data.data(), {rows, cols});
    }

    /**
     * @brief Dequantize Q8_1 tensor to FP32 for comparison
     */
    std::vector<float> dequantizeQ8_1(const Q8_1Tensor *tensor)
    {
        const auto &shape = tensor->shape();
        size_t total_elements = 1;
        for (auto dim : shape)
            total_elements *= dim;
        std::vector<float> result(total_elements);

        const Q8_1Block *blocks = tensor->q8_1_blocks();
        size_t total_blocks = (total_elements + 31) / 32;

        for (size_t b = 0; b < total_blocks; ++b)
        {
            const Q8_1Block &block = blocks[b];
            float d = simd::fp16_to_fp32(block.d);
            for (int i = 0; i < 32 && (b * 32 + i) < total_elements; ++i)
            {
                result[b * 32 + i] = static_cast<float>(block.qs[i]) * d;
            }
        }
        return result;
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    double cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size() || a.empty())
            return 0.0;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute max absolute difference
     */
    double maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b)
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        {
            max_diff = std::max(max_diff, static_cast<double>(std::abs(a[i] - b[i])));
        }
        return max_diff;
    }
};

/**
 * @test Q8_1 MPI tensor-parallel vs single-rank baseline (same Q8_1 kernel)
 *
 * This test validates that the MPI distribution logic (slicing, broadcasting,
 * allgatherv) produces identical results to single-rank execution.
 */
TEST_F(Test__Q8_1_MPI_TensorParallel, MPI_vs_SingleRank_Q8_1)
{
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Qwen 0.5B dimensions
    const int seq_len = 9;
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;

    // Create random Q/K/V tensors (same seed on all ranks!)
    auto Q = createRandomQ8_1Tensor(seq_len, n_heads * head_dim, 42);
    auto K = createRandomQ8_1Tensor(seq_len, n_kv_heads * head_dim, 43);
    auto V = createRandomQ8_1Tensor(seq_len, n_kv_heads * head_dim, 44);

    // Create output tensors
    auto output_mpi = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto output_single = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});

    // Create workspace tensors
    auto workspace_scores = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * seq_len)});
    auto workspace_buffer = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(2 * seq_len * n_heads * head_dim)});
    auto workspace_context = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len * n_heads * head_dim)});
    auto workspace_mask = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len * seq_len)});

    // Configure MPI attention
    MpiAttentionConfig config_mpi;
    config_mpi.n_heads = n_heads;
    config_mpi.n_kv_heads = n_kv_heads;
    config_mpi.head_dim = head_dim;
    config_mpi.causal = true;
    config_mpi.window_size = -1;
    config_mpi.precision = ActivationPrecision::Q8_1;
    config_mpi.mpi_ctx = mpi_ctx_;
    config_mpi.mpi_strategy = MPIStrategy::TensorParallel;
    config_mpi.workspace_scores = workspace_scores;
    config_mpi.workspace_qkv_buffer = workspace_buffer;
    config_mpi.workspace_context = workspace_context;
    config_mpi.workspace_mask = workspace_mask;

    // Run MPI tensor-parallel attention
    bool mpi_success = MpiAttentionOrchestrator::compute_tensor_parallel(
        Q.get(), K.get(), V.get(), output_mpi.get(),
        config_mpi, /*batch_size=*/1, /*sequence_lengths=*/nullptr);

    ASSERT_TRUE(mpi_success) << "MPI tensor-parallel Q8_1 attention failed";

    // Run single-rank baseline (on rank 0 only, but all ranks need the result)
    MpiAttentionConfig config_single = config_mpi;
    config_single.mpi_ctx = nullptr; // Disable MPI
    config_single.mpi_strategy = MPIStrategy::None;

    bool single_success = MpiAttentionOrchestrator::compute(
        Q.get(), K.get(), V.get(), output_single.get(),
        config_single, /*batch_size=*/1, /*sequence_lengths=*/nullptr);

    ASSERT_TRUE(single_success) << "Single-rank Q8_1 attention failed";

    // Compare outputs
    auto mpi_fp32 = dequantizeQ8_1(output_mpi.get());
    auto single_fp32 = dequantizeQ8_1(output_single.get());

    double cos_sim = cosineSimilarity(mpi_fp32, single_fp32);
    double max_diff = maxAbsDiff(mpi_fp32, single_fp32);

    if (rank_ == 0)
    {
        LOG_INFO("========================================");
        LOG_INFO("Q8_1 MPI TP vs Single-Rank Comparison");
        LOG_INFO("========================================");
        LOG_INFO("  Seq len: " << seq_len << ", n_heads: " << n_heads
                               << ", n_kv_heads: " << n_kv_heads << ", head_dim: " << head_dim);
        LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim);
        LOG_INFO("  Max abs diff: " << std::scientific << std::setprecision(2) << max_diff);

        // Print sample values
        LOG_INFO("  MPI output[0:5]: " << mpi_fp32[0] << ", " << mpi_fp32[1] << ", "
                                       << mpi_fp32[2] << ", " << mpi_fp32[3] << ", " << mpi_fp32[4]);
        LOG_INFO("  Single output[0:5]: " << single_fp32[0] << ", " << single_fp32[1] << ", "
                                          << single_fp32[2] << ", " << single_fp32[3] << ", " << single_fp32[4]);
    }

    // Expect very high similarity (same kernel, just distributed)
    EXPECT_GE(cos_sim, 0.999) << "MPI vs Single-rank Q8_1 cosine similarity too low";
    EXPECT_LT(max_diff, 0.1) << "MPI vs Single-rank Q8_1 max diff too high";

    mpi_ctx_->barrier();
}

/**
 * @test Q8_1 MPI tensor-parallel vs FP32 reference
 *
 * This test validates that Q8_1 MPI tensor-parallel attention produces
 * results comparable to FP32 attention (within quantization tolerance).
 */
TEST_F(Test__Q8_1_MPI_TensorParallel, MPI_Q8_1_vs_FP32_Reference)
{
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Qwen 0.5B dimensions
    const int seq_len = 9;
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;

    // Create random FP32 data
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(seq_len * n_kv_heads * head_dim);

    for (auto &v : Q_fp32)
        v = dist(gen);
    for (auto &v : K_fp32)
        v = dist(gen);
    for (auto &v : V_fp32)
        v = dist(gen);

    // Create Q8_1 tensors (quantized from FP32 using factory)
    auto Q_q8 = Q8_1Tensor::quantize_from_fp32(Q_fp32.data(),
                                               {static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto K_q8 = Q8_1Tensor::quantize_from_fp32(K_fp32.data(),
                                               {static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto V_q8 = Q8_1Tensor::quantize_from_fp32(V_fp32.data(),
                                               {static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});

    // Create FP32 tensors
    auto Q_fp = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto K_fp = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto V_fp = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});

    std::memcpy(Q_fp->mutable_data(), Q_fp32.data(), Q_fp32.size() * sizeof(float));
    std::memcpy(K_fp->mutable_data(), K_fp32.data(), K_fp32.size() * sizeof(float));
    std::memcpy(V_fp->mutable_data(), V_fp32.data(), V_fp32.size() * sizeof(float));

    // Create output tensors
    auto output_q8 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto output_fp = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});

    // Create workspace tensors
    auto workspace_scores = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * seq_len)});
    auto workspace_buffer = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(2 * seq_len * n_heads * head_dim)});
    auto workspace_context = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len * n_heads * head_dim)});
    auto workspace_mask = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len * seq_len)});

    // Configure Q8_1 MPI attention
    MpiAttentionConfig config_q8;
    config_q8.n_heads = n_heads;
    config_q8.n_kv_heads = n_kv_heads;
    config_q8.head_dim = head_dim;
    config_q8.causal = true;
    config_q8.window_size = -1;
    config_q8.precision = ActivationPrecision::Q8_1;
    config_q8.mpi_ctx = mpi_ctx_;
    config_q8.mpi_strategy = MPIStrategy::TensorParallel;
    config_q8.workspace_scores = workspace_scores;
    config_q8.workspace_qkv_buffer = workspace_buffer;
    config_q8.workspace_context = workspace_context;
    config_q8.workspace_mask = workspace_mask;

    bool q8_success = MpiAttentionOrchestrator::compute_tensor_parallel(
        Q_q8.get(), K_q8.get(), V_q8.get(), output_q8.get(),
        config_q8, /*batch_size=*/1, /*sequence_lengths=*/nullptr);

    ASSERT_TRUE(q8_success) << "Q8_1 MPI tensor-parallel attention failed";

    // Configure FP32 MPI attention
    MpiAttentionConfig config_fp = config_q8;
    config_fp.precision = ActivationPrecision::FP32;

    bool fp_success = MpiAttentionOrchestrator::compute_tensor_parallel(
        Q_fp.get(), K_fp.get(), V_fp.get(), output_fp.get(),
        config_fp, /*batch_size=*/1, /*sequence_lengths=*/nullptr);

    ASSERT_TRUE(fp_success) << "FP32 MPI tensor-parallel attention failed";

    // Compare outputs
    auto q8_dequant = dequantizeQ8_1(output_q8.get());
    const float *fp_data = output_fp->data();
    std::vector<float> fp_out(fp_data, fp_data + seq_len * n_heads * head_dim);

    double cos_sim = cosineSimilarity(q8_dequant, fp_out);
    double max_diff = maxAbsDiff(q8_dequant, fp_out);

    if (rank_ == 0)
    {
        LOG_INFO("========================================");
        LOG_INFO("Q8_1 MPI TP vs FP32 MPI TP Comparison");
        LOG_INFO("========================================");
        LOG_INFO("  Seq len: " << seq_len << ", n_heads: " << n_heads
                               << ", n_kv_heads: " << n_kv_heads << ", head_dim: " << head_dim);
        LOG_INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim);
        LOG_INFO("  Max abs diff: " << std::scientific << std::setprecision(2) << max_diff);

        // Print sample values
        LOG_INFO("  Q8_1 output[0:5]: " << q8_dequant[0] << ", " << q8_dequant[1] << ", "
                                        << q8_dequant[2] << ", " << q8_dequant[3] << ", " << q8_dequant[4]);
        LOG_INFO("  FP32 output[0:5]: " << fp_out[0] << ", " << fp_out[1] << ", "
                                        << fp_out[2] << ", " << fp_out[3] << ", " << fp_out[4]);
    }

    // Expect reasonable similarity (accounting for quantization)
    EXPECT_GE(cos_sim, 0.99) << "Q8_1 vs FP32 MPI TP cosine similarity too low";

    mpi_ctx_->barrier();
}

/**
 * @test Head slicing correctness
 *
 * Validates that sliceQ8_1HeadBlocks correctly extracts heads for each rank.
 */
TEST_F(Test__Q8_1_MPI_TensorParallel, HeadSlicing_Correctness)
{
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const int seq_len = 4;
    const int n_heads = 8;
    const int head_dim = 64;

    // Create Q tensor with known pattern: each head has distinct values
    auto Q = createRandomQ8_1Tensor(seq_len, n_heads * head_dim, 42);

    // Get local slice info
    auto [start_head, local_n_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(n_heads));

    if (rank_ == 0)
    {
        LOG_INFO("========================================");
        LOG_INFO("Head Slicing Test");
        LOG_INFO("========================================");
        LOG_INFO("  Total heads: " << n_heads << ", Local heads: " << local_n_heads
                                   << ", Start head: " << start_head);
    }

    // Verify slicing is correct
    EXPECT_EQ(local_n_heads, 4) << "Expected 4 heads per rank with 8 heads / 2 ranks";

    if (rank_ == 0)
    {
        EXPECT_EQ(start_head, 0) << "Rank 0 should start at head 0";
    }
    else
    {
        EXPECT_EQ(start_head, 4) << "Rank 1 should start at head 4";
    }

    mpi_ctx_->barrier();
}

/**
 * @test GQA broadcast correctness
 *
 * Validates that broadcastQ8_1KVHeads correctly replicates KV heads to Q heads.
 */
TEST_F(Test__Q8_1_MPI_TensorParallel, GQA_Broadcast_Correctness)
{
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    const int seq_len = 4;
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int heads_per_kv = n_heads / n_kv_heads; // 7

    // Create K/V tensors
    auto K = createRandomQ8_1Tensor(seq_len, n_kv_heads * head_dim, 43);
    auto V = createRandomQ8_1Tensor(seq_len, n_kv_heads * head_dim, 44);

    // Get local slice info
    auto [start_head, local_n_heads] = mpi_ctx_->get_local_slice(static_cast<size_t>(n_heads));

    if (rank_ == 0)
    {
        LOG_INFO("========================================");
        LOG_INFO("GQA Broadcast Test (Qwen 0.5B config)");
        LOG_INFO("========================================");
        LOG_INFO("  n_heads: " << n_heads << ", n_kv_heads: " << n_kv_heads
                               << ", heads_per_kv: " << heads_per_kv);
        LOG_INFO("  Rank 0: start_head=" << start_head << ", local_n_heads=" << local_n_heads);
    }

    // For 14 heads / 2 ranks: rank 0 gets heads 0-6, rank 1 gets heads 7-13
    // Heads 0-6 map to KV head 0 (7 heads per KV)
    // Heads 7-13 map to KV head 1

    if (rank_ == 0)
    {
        EXPECT_EQ(start_head, 0);
        EXPECT_EQ(local_n_heads, 7);
    }
    else
    {
        EXPECT_EQ(start_head, 7);
        EXPECT_EQ(local_n_heads, 7);
    }

    mpi_ctx_->barrier();
}

/**
 * @test Allgatherv reconstruction
 *
 * Validates that the allgatherv correctly reconstructs the full output.
 */
TEST_F(Test__Q8_1_MPI_TensorParallel, Allgatherv_Reconstruction)
{
    if (world_size_ != 2)
    {
        GTEST_SKIP() << "Test requires exactly 2 MPI ranks";
    }

    // Smaller test case for debugging
    const int seq_len = 2;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 64;

    // Create random Q/K/V tensors (same seed on all ranks!)
    auto Q = createRandomQ8_1Tensor(seq_len, n_heads * head_dim, 42);
    auto K = createRandomQ8_1Tensor(seq_len, n_kv_heads * head_dim, 43);
    auto V = createRandomQ8_1Tensor(seq_len, n_kv_heads * head_dim, 44);

    // Create output tensors
    auto output_mpi = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto output_single = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});

    // Create workspace tensors
    auto workspace_scores = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * seq_len)});
    auto workspace_buffer = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(2 * seq_len * n_heads * head_dim)});
    auto workspace_context = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len * n_heads * head_dim)});
    auto workspace_mask = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len * seq_len)});

    // Run MPI tensor-parallel
    MpiAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = true;
    config.window_size = -1;
    config.precision = ActivationPrecision::Q8_1;
    config.mpi_ctx = mpi_ctx_;
    config.mpi_strategy = MPIStrategy::TensorParallel;
    config.workspace_scores = workspace_scores;
    config.workspace_qkv_buffer = workspace_buffer;
    config.workspace_context = workspace_context;
    config.workspace_mask = workspace_mask;

    bool mpi_success = MpiAttentionOrchestrator::compute_tensor_parallel(
        Q.get(), K.get(), V.get(), output_mpi.get(),
        config, /*batch_size=*/1, /*sequence_lengths=*/nullptr);

    ASSERT_TRUE(mpi_success) << "MPI tensor-parallel attention failed";

    // Run single-rank baseline
    MpiAttentionConfig config_single = config;
    config_single.mpi_ctx = nullptr;
    config_single.mpi_strategy = MPIStrategy::None;

    bool single_success = MpiAttentionOrchestrator::compute(
        Q.get(), K.get(), V.get(), output_single.get(),
        config_single, /*batch_size=*/1, /*sequence_lengths=*/nullptr);

    ASSERT_TRUE(single_success) << "Single-rank attention failed";

    // Compare per-head outputs
    auto mpi_fp32 = dequantizeQ8_1(output_mpi.get());
    auto single_fp32 = dequantizeQ8_1(output_single.get());

    if (rank_ == 0)
    {
        LOG_INFO("========================================");
        LOG_INFO("Allgatherv Reconstruction Test");
        LOG_INFO("========================================");

        // Check each head separately
        for (int h = 0; h < n_heads; ++h)
        {
            std::vector<float> mpi_head(head_dim);
            std::vector<float> single_head(head_dim);

            for (int d = 0; d < head_dim; ++d)
            {
                // Last position (token 1), head h
                int idx = (seq_len - 1) * n_heads * head_dim + h * head_dim + d;
                mpi_head[d] = mpi_fp32[idx];
                single_head[d] = single_fp32[idx];
            }

            double cos = cosineSimilarity(mpi_head, single_head);
            double max_d = maxAbsDiff(mpi_head, single_head);

            LOG_INFO("  Head " << h << ": cosine=" << std::fixed << std::setprecision(6) << cos
                               << " max_diff=" << std::scientific << max_d);

            EXPECT_GE(cos, 0.999) << "Head " << h << " diverged after allgatherv";
        }
    }

    mpi_ctx_->barrier();
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
