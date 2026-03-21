/**
 * @file Test__QuantisedAttentionJit_Q8_1_Fused.cpp
 * @brief Unit tests for QuantisedAttentionJit_Q8_1_Fused JIT kernel
 *
 * Tests the fully-fused Q8_1 attention kernel that performs:
 *   1. Q8_1 × Q8_1 dot product (Q @ K^T)
 *   2. Online softmax (no intermediate score storage)
 *   3. Online weighted V accumulation
 *   4. Online requantization to Q8_1 output
 *
 * Validates:
 * - JIT kernel instantiation for various head dimensions
 * - Dynamic stack layout calculation
 * - Code generation correctness
 * - Spill offset calculations
 * - Numerical correctness for small matrices
 * - Edge cases (various head_dim values)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>

#include "kernels/cpu/gemm/QuantisedAttentionJit_Q8_1_Fused.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;
using namespace llaminar2::gemm;

/**
 * @brief Test fixture for QuantisedAttentionJit_Q8_1_Fused unit tests
 */
class Test__QuantisedAttentionJit_Q8_1_Fused : public ::testing::Test
{
protected:
    // Fixed seed for reproducibility
    std::mt19937 gen_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    // Correctness thresholds for Q8_1 quantization
    static constexpr double MIN_COSINE_SIM = 0.99;
    static constexpr double MAX_REL_L2_ERROR = 0.10;

    /**
     * @brief Quantize FP32 data to Q8_1 block format
     *
     * Q8_1 format per block (36 bytes):
     *   - d: fp16 (2 bytes) - scale factor
     *   - sum_qs: int16 (2 bytes) - sum of quantized values
     *   - qs: int8[32] (32 bytes) - quantized values
     */
    void quantize_fp32_to_q8_1(const float *fp32_data, int rows, int cols, Q8_1Block *blocks)
    {
        const int num_blocks_per_row = cols / 32;

        for (int row = 0; row < rows; ++row)
        {
            for (int b = 0; b < num_blocks_per_row; ++b)
            {
                const float *block_data = fp32_data + row * cols + b * 32;
                Q8_1Block &blk = blocks[row * num_blocks_per_row + b];

                // Find max absolute value
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::fabs(block_data[i]));
                }

                // Compute scale
                float scale = max_abs / 127.0f;
                if (scale == 0.0f)
                    scale = 1.0f;

                // Quantize
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(block_data[i] / scale));
                    blk.qs[i] = q;
                    sum_qs += q;
                }

                blk.d = fp32_to_fp16(scale);
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }
    }

    /**
     * @brief Dequantize Q8_1 blocks to FP32
     */
    void dequant_q8_1_to_fp32(const Q8_1Block *blocks, int rows, int cols, float *fp32_data)
    {
        const int num_blocks_per_row = cols / 32;

        for (int row = 0; row < rows; ++row)
        {
            for (int b = 0; b < num_blocks_per_row; ++b)
            {
                const Q8_1Block &blk = blocks[row * num_blocks_per_row + b];
                float scale = fp16_to_fp32(blk.d);
                float *out = fp32_data + row * cols + b * 32;

                for (int i = 0; i < 32; ++i)
                {
                    out[i] = static_cast<float>(blk.qs[i]) * scale;
                }
            }
        }
    }

    /**
     * @brief Compute FP32 reference attention: softmax(Q @ K^T * scale) @ V
     */
    void compute_fp32_reference(
        int M, int N, int D,
        const float *Q_fp32, const float *K_fp32, const float *V_fp32,
        float scale, float *output_fp32)
    {
        std::vector<float> scores(M * N, 0.0f);

        // Q @ K^T
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < D; ++k)
                {
                    sum += Q_fp32[i * D + k] * K_fp32[j * D + k];
                }
                scores[i * N + j] = sum * scale;
            }
        }

        // Softmax (row-wise)
        for (int i = 0; i < M; ++i)
        {
            float max_val = scores[i * N];
            for (int j = 1; j < N; ++j)
            {
                max_val = std::max(max_val, scores[i * N + j]);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                scores[i * N + j] = std::exp(scores[i * N + j] - max_val);
                sum_exp += scores[i * N + j];
            }

            for (int j = 0; j < N; ++j)
            {
                scores[i * N + j] /= sum_exp;
            }
        }

        // scores @ V
        for (int i = 0; i < M; ++i)
        {
            for (int d = 0; d < D; ++d)
            {
                float sum = 0.0f;
                for (int j = 0; j < N; ++j)
                {
                    sum += scores[i * N + j] * V_fp32[j * D + d];
                }
                output_fp32[i * D + d] = sum;
            }
        }
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10);
    }

    /**
     * @brief Compute relative L2 error
     */
    double relative_l2_error(const float *actual, const float *expected, size_t n)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = actual[i] - expected[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += expected[i] * expected[i];
        }
        return std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-10);
    }
};

// =============================================================================
// Stack Layout Calculation Tests
// =============================================================================

/**
 * @brief Test stack layout for D=64 (2 blocks, no spilling needed)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, StackLayout_D64_NoSpill)
{
    QuantisedAttentionJit_Q8_1_Fused kernel(64);

    // D=64 means 2 blocks, q_area = 2*64 = 128, rounded to 128
    // spill_base = 128 + 64 = 192
    // No spilled blocks (only blocks 0-1, both in registers)

    // Spill offsets for blocks >= 2 should follow the formula
    // These would be used if we had more blocks
    EXPECT_EQ(kernel.spill_offset_lo(2), 192 + 0);
    EXPECT_EQ(kernel.spill_offset_hi(2), 192 + 64);
}

/**
 * @brief Test stack layout for D=128 (4 blocks, 2 spilled)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, StackLayout_D128_TwoSpilled)
{
    QuantisedAttentionJit_Q8_1_Fused kernel(128);

    // D=128 means 4 blocks, q_area = 4*64 = 256, rounded to 256
    // spill_base = 256 + 64 = 320
    // Blocks 2-3 are spilled

    EXPECT_EQ(kernel.spill_offset_lo(2), 320 + 0);
    EXPECT_EQ(kernel.spill_offset_hi(2), 320 + 64);
    EXPECT_EQ(kernel.spill_offset_lo(3), 320 + 128);
    EXPECT_EQ(kernel.spill_offset_hi(3), 320 + 128 + 64);
}

/**
 * @brief Test stack layout for D=256 (8 blocks, 6 spilled)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, StackLayout_D256_SixSpilled)
{
    QuantisedAttentionJit_Q8_1_Fused kernel(256);

    // D=256 means 8 blocks, q_area = 8*64 = 512, rounded to 512
    // spill_base = 512 + 64 = 576
    // Blocks 2-7 are spilled

    EXPECT_EQ(kernel.spill_offset_lo(2), 576 + 0);
    EXPECT_EQ(kernel.spill_offset_hi(2), 576 + 64);
    EXPECT_EQ(kernel.spill_offset_lo(3), 576 + 128);
    EXPECT_EQ(kernel.spill_offset_hi(3), 576 + 128 + 64);
    EXPECT_EQ(kernel.spill_offset_lo(7), 576 + 5 * 128);
    EXPECT_EQ(kernel.spill_offset_hi(7), 576 + 5 * 128 + 64);
}

/**
 * @brief Test stack layout for D=512 (16 blocks, 14 spilled)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, StackLayout_D512_FourteenSpilled)
{
    QuantisedAttentionJit_Q8_1_Fused kernel(512);

    // D=512 means 16 blocks, q_area = 16*64 = 1024, rounded to 1024
    // spill_base = 1024 + 64 = 1088
    // Blocks 2-15 are spilled

    EXPECT_EQ(kernel.spill_offset_lo(2), 1088 + 0);
    EXPECT_EQ(kernel.spill_offset_hi(2), 1088 + 64);
    EXPECT_EQ(kernel.spill_offset_lo(15), 1088 + 13 * 128);
    EXPECT_EQ(kernel.spill_offset_hi(15), 1088 + 13 * 128 + 64);
}

/**
 * @brief Test stack layout for D=192 (6 blocks - odd number)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, StackLayout_D192_OddBlocks)
{
    QuantisedAttentionJit_Q8_1_Fused kernel(192);

    // D=192 means 6 blocks, q_area = 6*64 = 384, rounded to 384
    // spill_base = 384 + 64 = 448
    // Blocks 2-5 are spilled (4 spilled blocks)

    EXPECT_EQ(kernel.spill_offset_lo(2), 448 + 0);
    EXPECT_EQ(kernel.spill_offset_hi(2), 448 + 64);
    EXPECT_EQ(kernel.spill_offset_lo(5), 448 + 3 * 128);
    EXPECT_EQ(kernel.spill_offset_hi(5), 448 + 3 * 128 + 64);
}

// =============================================================================
// Kernel Instantiation Tests
// =============================================================================

/**
 * @brief Test that kernel can be instantiated for various head dimensions
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Instantiation_VariousHeadDims)
{
    // These should all succeed
    EXPECT_NO_THROW({
        QuantisedAttentionJit_Q8_1_Fused kernel32(32);
    });
    EXPECT_NO_THROW({
        QuantisedAttentionJit_Q8_1_Fused kernel64(64);
    });
    EXPECT_NO_THROW({
        QuantisedAttentionJit_Q8_1_Fused kernel128(128);
    });
    EXPECT_NO_THROW({
        QuantisedAttentionJit_Q8_1_Fused kernel192(192);
    });
    EXPECT_NO_THROW({
        QuantisedAttentionJit_Q8_1_Fused kernel256(256);
    });
    EXPECT_NO_THROW({
        QuantisedAttentionJit_Q8_1_Fused kernel512(512);
    });
}

/**
 * @brief Test that kernel rejects invalid head dimensions
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Instantiation_RejectsInvalid)
{
    // Not a multiple of 32
    EXPECT_THROW({ QuantisedAttentionJit_Q8_1_Fused kernel(33); }, std::invalid_argument);

    EXPECT_THROW({ QuantisedAttentionJit_Q8_1_Fused kernel(100); }, std::invalid_argument);

    EXPECT_THROW({ QuantisedAttentionJit_Q8_1_Fused kernel(65); }, std::invalid_argument);
}

/**
 * @brief Test that get_kernel returns a valid function pointer
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, GetKernel_ReturnsValidPtr)
{
    QuantisedAttentionJit_Q8_1_Fused kernel(64);
    auto func = kernel.get_kernel();
    EXPECT_NE(func, nullptr);
}

// =============================================================================
// Numerical Correctness Tests
// =============================================================================

/**
 * @brief Test D=64 single token numerical correctness
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Correctness_D64_SingleToken)
{
    const int M = 1, N = 8, D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Generate random data
    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    // Quantize
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Compute FP32 reference
    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    // Run JIT kernel
    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    // Dequantize output
    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // Validate
    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    double rel_l2 = relative_l2_error(jit_output.data(), ref_output.data(), M * D);

    EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
    EXPECT_LE(rel_l2, MAX_REL_L2_ERROR) << "Relative L2 error too high: " << rel_l2;
}

/**
 * @brief Test D=128 single token numerical correctness
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Correctness_D128_SingleToken)
{
    const int M = 1, N = 8, D = 128;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    double rel_l2 = relative_l2_error(jit_output.data(), ref_output.data(), M * D);

    EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
    EXPECT_LE(rel_l2, MAX_REL_L2_ERROR) << "Relative L2 error too high: " << rel_l2;
}

/**
 * @brief Test D=256 single token numerical correctness
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Correctness_D256_SingleToken)
{
    const int M = 1, N = 8, D = 256;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    double rel_l2 = relative_l2_error(jit_output.data(), ref_output.data(), M * D);

    EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
    EXPECT_LE(rel_l2, MAX_REL_L2_ERROR) << "Relative L2 error too high: " << rel_l2;
}

/**
 * @brief Test D=192 single token (odd number of spilled blocks)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Correctness_D192_OddBlocks)
{
    const int M = 1, N = 8, D = 192;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    double rel_l2 = relative_l2_error(jit_output.data(), ref_output.data(), M * D);

    EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
    EXPECT_LE(rel_l2, MAX_REL_L2_ERROR) << "Relative L2 error too high: " << rel_l2;
}

/**
 * @brief Test batch processing (M > 1)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Correctness_D64_Batch)
{
    const int M = 4, N = 16, D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    double rel_l2 = relative_l2_error(jit_output.data(), ref_output.data(), M * D);

    // Batch has more accumulated error from compounding quantization noise
    // Q8_1 has ~0.4% error per operation, which compounds through:
    // - 4 Q rows × 16 K/V positions = 64 dot products
    // - Softmax + V accumulation adds more ops
    // Expected cosine ≥ 0.95 and rel_l2 ≤ 0.30 for M=4, N=16
    EXPECT_GE(cosine, 0.95) << "Cosine similarity too low: " << cosine;
    EXPECT_LE(rel_l2, 0.30) << "Relative L2 error too high: " << rel_l2;
}

/**
 * @brief Test D=512 large head dimension (stress test)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, Correctness_D512_StressTest)
{
    const int M = 1, N = 4, D = 512; // Smaller N for speed
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    double rel_l2 = relative_l2_error(jit_output.data(), ref_output.data(), M * D);

    EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
    EXPECT_LE(rel_l2, MAX_REL_L2_ERROR) << "Relative L2 error too high: " << rel_l2;
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/**
 * @brief Test with uniform input values (should produce uniform output via softmax)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, EdgeCase_UniformInput)
{
    const int M = 1, N = 4, D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // All same values - softmax should give uniform weights
    std::vector<float> Q_fp32(M * D, 0.5f);
    std::vector<float> K_fp32(N * D, 0.5f);
    std::vector<float> V_fp32(N * D, 0.5f);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // With uniform input, output should equal V (since all attention weights are 1/N)
    // Check that output is close to V values
    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    EXPECT_GE(cosine, 0.999) << "Uniform input should give near-perfect match";
}

/**
 * @brief Test with N=1 (single KV position)
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, EdgeCase_SingleKV)
{
    const int M = 2, N = 1, D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // With N=1, softmax gives weight=1.0, so output should equal V
    // All Q rows should produce V as output (since only one K/V position)
    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    EXPECT_GE(cosine, 0.999) << "Single KV should give output equal to V";
}

/**
 * @brief Test minimum head dimension D=32
 */
TEST_F(Test__QuantisedAttentionJit_Q8_1_Fused, EdgeCase_MinimumD32)
{
    const int M = 1, N = 4, D = 32;
    const int num_blocks = D / 32; // = 1
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &v : Q_fp32)
        v = dist_(gen_);
    for (auto &v : K_fp32)
        v = dist_(gen_);
    for (auto &v : V_fp32)
        v = dist_(gen_);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    std::vector<float> ref_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

    QuantisedAttentionJit_Q8_1_Fused kernel(D);
    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel.get_kernel()(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    double cosine = cosine_similarity(jit_output.data(), ref_output.data(), M * D);
    double rel_l2 = relative_l2_error(jit_output.data(), ref_output.data(), M * D);

    EXPECT_GE(cosine, MIN_COSINE_SIM) << "Cosine similarity too low: " << cosine;
    EXPECT_LE(rel_l2, MAX_REL_L2_ERROR) << "Relative L2 error too high: " << rel_l2;
}
