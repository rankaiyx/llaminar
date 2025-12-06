/**
 * @file Test__CPURoPEKernelTyped.cpp
 * @brief Unit tests for CPURoPEKernelTyped with typed precision conversion
 *
 * Tests the typed RoPE kernel with FP32, BF16, FP16, and Q8_1 precision types.
 * The Q8_1 implementation uses pure-integer operations without FP32 round-trips.
 *
 * @author David Sanftenberg
 * @date 2025
 */

#include <gtest/gtest.h>

#include "v2/kernels/cpu/ops/CPURoPEKernelTyped.h"
#include "v2/kernels/cpu/primitives/RoPEPrimitives.h"
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/BlockStructures.h"

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace llaminar2
{
    namespace
    {

        /**
         * @brief Reference RoPE implementation for testing
         *
         * RoPE rotates pairs of elements: [x, y] -> [x*cos - y*sin, x*sin + y*cos]
         * Applied to first half paired with second half of each head.
         */
        void reference_rope(
            float *data,
            int seq_len,
            int n_heads,
            int head_dim,
            const int *position_ids,
            float rope_theta)
        {
            const int half_dim = head_dim / 2;
            const int head_stride = head_dim;
            const int token_stride = n_heads * head_dim;

            for (int tok = 0; tok < seq_len; ++tok)
            {
                int position = position_ids ? position_ids[tok] : tok;
                if (position < 0)
                    continue;

                for (int h = 0; h < n_heads; ++h)
                {
                    float *head_data = data + tok * token_stride + h * head_stride;

                    for (int i = 0; i < half_dim; ++i)
                    {
                        // Compute angle for this dimension
                        float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / head_dim);
                        float angle = position * freq;
                        float cos_val = std::cos(angle);
                        float sin_val = std::sin(angle);

                        // First half element pairs with second half element
                        float x = head_data[i];
                        float y = head_data[i + half_dim];

                        // RoPE rotation
                        head_data[i] = x * cos_val - y * sin_val;
                        head_data[i + half_dim] = x * sin_val + y * cos_val;
                    }
                }
            }
        }

        /**
         * @brief Generate random FP32 data
         */
        std::vector<float> generate_random_fp32(size_t count, float min_val = -2.0f, float max_val = 2.0f)
        {
            std::mt19937 rng(42); // Fixed seed for reproducibility
            std::uniform_real_distribution<float> dist(min_val, max_val);

            std::vector<float> data(count);
            for (auto &v : data)
            {
                v = dist(rng);
            }
            return data;
        }

        /**
         * @brief Compute max absolute difference between two vectors
         */
        float max_abs_diff(const float *a, const float *b, size_t count)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                float diff = std::abs(a[i] - b[i]);
                if (diff > max_diff)
                    max_diff = diff;
            }
            return max_diff;
        }

        /**
         * @brief Compute mean absolute error between two vectors
         */
        float mean_abs_error(const float *a, const float *b, size_t count)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                sum += std::abs(a[i] - b[i]);
            }
            return sum / count;
        }

        /**
         * @brief Compute cosine similarity between two vectors
         */
        float cosine_similarity(const float *a, const float *b, size_t count)
        {
            float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8f);
        }

    } // anonymous namespace

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class CPURoPEKernelTypedTest : public ::testing::Test
    {
    protected:
        static constexpr float ROPE_THETA = 10000.0f;

        // Tolerances for different precisions
        static constexpr float FP32_TOLERANCE = 1e-5f;
        static constexpr float FP32_LARGE_POS_TOLERANCE = 5e-4f; // Relaxed for large positions (>1000)
        static constexpr float BF16_TOLERANCE = 5e-2f;           // BF16 has ~3 decimal digits precision
        static constexpr float FP16_TOLERANCE = 2e-2f;           // FP16 has ~3.3 decimal digits precision
        static constexpr float Q8_1_TOLERANCE = 0.20f;           // Q8_1 is lossy (8-bit quantization + integer rotation)

        // Cosine similarity thresholds
        static constexpr float FP32_COSINE_MIN = 0.99999f;
        static constexpr float BF16_COSINE_MIN = 0.995f;
        static constexpr float FP16_COSINE_MIN = 0.997f;
        static constexpr float Q8_1_COSINE_MIN = 0.98f;

        void SetUp() override
        {
            // Default test dimensions
            seq_len_ = 4;
            n_heads_ = 8;
            n_kv_heads_ = 2;
            head_dim_ = 64; // Must be multiple of 32 for Q8_1
        }

        int seq_len_;
        int n_heads_;
        int n_kv_heads_;
        int head_dim_;
    };

    // =========================================================================
    // FP32 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTypedTest, FP32_apply_typed_matches_reference)
    {
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t k_size = seq_len_ * n_kv_heads_ * head_dim_;

        // Generate test data
        auto q_data = generate_random_fp32(q_size);
        auto k_data = generate_random_fp32(k_size);

        // Copy for reference
        std::vector<float> q_expected = q_data;
        std::vector<float> k_expected = k_data;

        // Position IDs: 0, 1, 2, 3
        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        // Compute reference
        reference_rope(q_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);
        reference_rope(k_expected.data(), seq_len_, n_kv_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Test typed kernel (operates in-place)
        CPURoPEKernelTyped<ActivationPrecision::FP32> kernel;

        // Call the kernel's apply_typed method
        ASSERT_TRUE(kernel.apply_typed(
            q_data.data(), k_data.data(),
            position_ids.data(),
            seq_len_, n_heads_, n_kv_heads_, head_dim_,
            ROPE_THETA));

        // Verify Q
        float q_max_diff = max_abs_diff(q_data.data(), q_expected.data(), q_size);
        EXPECT_LT(q_max_diff, FP32_TOLERANCE)
            << "FP32 RoPE Q max diff: " << q_max_diff;

        // Verify K
        float k_max_diff = max_abs_diff(k_data.data(), k_expected.data(), k_size);
        EXPECT_LT(k_max_diff, FP32_TOLERANCE)
            << "FP32 RoPE K max diff: " << k_max_diff;
    }

    TEST_F(CPURoPEKernelTypedTest, FP32_precision_metadata)
    {
        CPURoPEKernelTyped<ActivationPrecision::FP32> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::FP32);
        EXPECT_STREQ(kernel.precision_name(), "FP32");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 1.0f);
    }

    TEST_F(CPURoPEKernelTypedTest, FP32_single_token_single_head)
    {
        // Minimal test case: 1 token, 1 head
        int local_seq_len = 1;
        int local_n_heads = 1;
        const size_t q_size = local_seq_len * local_n_heads * head_dim_;

        auto q_data = generate_random_fp32(q_size);
        std::vector<float> q_expected = q_data;

        std::vector<int> position_ids = {5}; // Non-zero position

        reference_rope(q_expected.data(), local_seq_len, local_n_heads, head_dim_,
                       position_ids.data(), ROPE_THETA);

        CPURoPEKernelTyped<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            q_data.data(), nullptr,
            position_ids.data(),
            local_seq_len, local_n_heads, 0, head_dim_,
            ROPE_THETA));

        float max_diff = max_abs_diff(q_data.data(), q_expected.data(), q_size);
        EXPECT_LT(max_diff, FP32_TOLERANCE);
    }

    // =========================================================================
    // BF16 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTypedTest, BF16_apply_typed_with_conversion)
    {
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        // Generate FP32 test data and convert to BF16
        auto fp32_q = generate_random_fp32(q_size);

        // Convert to BF16
        std::vector<uint16_t> bf16_q(q_size);
        simd::convert_fp32_to_bf16(fp32_q.data(), bf16_q.data(), q_size);

        // Reference: convert to FP32, apply RoPE, convert back
        std::vector<float> fp32_expected(q_size);
        simd::convert_bf16_to_fp32(bf16_q.data(), fp32_expected.data(), q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE to BF16 data using the kernel
        CPURoPEKernelTyped<ActivationPrecision::BF16> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            bf16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA));

        // Convert result to FP32 for comparison
        std::vector<float> fp32_result(q_size);
        simd::convert_bf16_to_fp32(bf16_q.data(), fp32_result.data(), q_size);

        // Verify with BF16 tolerance
        float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_LT(max_diff, BF16_TOLERANCE)
            << "BF16 RoPE max diff: " << max_diff;

        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, BF16_COSINE_MIN)
            << "BF16 RoPE cosine similarity: " << cosine;
    }

    TEST_F(CPURoPEKernelTypedTest, BF16_precision_metadata)
    {
        CPURoPEKernelTyped<ActivationPrecision::BF16> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::BF16);
        EXPECT_STREQ(kernel.precision_name(), "BF16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);
    }

    // =========================================================================
    // FP16 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTypedTest, FP16_apply_typed_with_conversion)
    {
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        // Generate FP32 test data and convert to FP16
        auto fp32_q = generate_random_fp32(q_size);

        // Convert to FP16
        std::vector<uint16_t> fp16_q(q_size);
        simd::convert_fp32_to_fp16(fp32_q.data(), fp16_q.data(), q_size);

        // Reference: convert to FP32, apply RoPE
        std::vector<float> fp32_expected(q_size);
        simd::convert_fp16_to_fp32(fp16_q.data(), fp32_expected.data(), q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE to FP16 data using the kernel
        CPURoPEKernelTyped<ActivationPrecision::FP16> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            fp16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA));

        // Convert result to FP32 for comparison
        std::vector<float> fp32_result(q_size);
        simd::convert_fp16_to_fp32(fp16_q.data(), fp32_result.data(), q_size);

        // Verify with FP16 tolerance
        float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_LT(max_diff, FP16_TOLERANCE)
            << "FP16 RoPE max diff: " << max_diff;

        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, FP16_COSINE_MIN)
            << "FP16 RoPE cosine similarity: " << cosine;
    }

    TEST_F(CPURoPEKernelTypedTest, FP16_precision_metadata)
    {
        CPURoPEKernelTyped<ActivationPrecision::FP16> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::FP16);
        EXPECT_STREQ(kernel.precision_name(), "FP16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);
    }

    // =========================================================================
    // Q8_1 Specialization Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTypedTest, Q8_1_apply_typed_with_conversion)
    {
        // Q8_1 requires head_dim divisible by 32
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t num_blocks = q_size / 32;

        // Generate FP32 test data
        auto fp32_q = generate_random_fp32(q_size);

        // Convert to Q8_1
        std::vector<Q8_1Block> q8_q(num_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_q.data(), q8_q.data(), q_size);

        // Reference: dequant, apply RoPE
        std::vector<float> fp32_expected(q_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_expected.data(), q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply RoPE to Q8_1 data (pure-integer)
        primitives::apply_rope_q8_1_integer(
            q8_q.data(), nullptr,
            position_ids.data(),
            seq_len_,
            n_heads_, 0,
            head_dim_,
            ROPE_THETA);

        // Dequantize result for comparison
        std::vector<float> fp32_result(q_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_result.data(), q_size);

        // Q8_1 has more noise due to quantization + integer rotation
        // Use cosine similarity as primary metric
        float cosine = cosine_similarity(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_GT(cosine, Q8_1_COSINE_MIN)
            << "Q8_1 RoPE cosine similarity: " << cosine;

        float max_diff = max_abs_diff(fp32_result.data(), fp32_expected.data(), q_size);
        EXPECT_LT(max_diff, Q8_1_TOLERANCE)
            << "Q8_1 RoPE max diff: " << max_diff;
    }

    TEST_F(CPURoPEKernelTypedTest, Q8_1_precision_metadata)
    {
        CPURoPEKernelTyped<ActivationPrecision::Q8_1> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::Q8_1);
        EXPECT_STREQ(kernel.precision_name(), "Q8_1");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 4.0f);
    }

    TEST_F(CPURoPEKernelTypedTest, Q8_1_requires_head_dim_multiple_of_32)
    {
        // Test that Q8_1 RoPE requires head_dim divisible by 32
        // This is enforced by the apply_rope_q8_1_integer function
        int bad_head_dim = 48; // Not aligned to block boundaries properly
        const size_t q_size = seq_len_ * n_heads_ * bad_head_dim;
        const size_t num_blocks = (q_size + 31) / 32;

        std::vector<Q8_1Block> q8_q(num_blocks);
        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        // The function should handle this gracefully (either return early or succeed)
        // We just verify it doesn't crash
        primitives::apply_rope_q8_1_integer(
            q8_q.data(), nullptr,
            position_ids.data(),
            seq_len_,
            n_heads_, 0,
            bad_head_dim, // Not cleanly divisible
            ROPE_THETA);

        // No assertion - just verifying no crash
    }

    TEST_F(CPURoPEKernelTypedTest, Q8_1_with_both_Q_and_K)
    {
        // Test with both Q and K tensors
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        const size_t k_size = seq_len_ * n_kv_heads_ * head_dim_;
        const size_t q_blocks = q_size / 32;
        const size_t k_blocks = k_size / 32;

        auto fp32_q = generate_random_fp32(q_size);
        auto fp32_k = generate_random_fp32(k_size);

        std::vector<Q8_1Block> q8_q(q_blocks);
        std::vector<Q8_1Block> q8_k(k_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_q.data(), q8_q.data(), q_size);
        simd::quantize_fp32_to_q8_1_blocks(fp32_k.data(), q8_k.data(), k_size);

        // Reference
        std::vector<float> fp32_q_expected(q_size);
        std::vector<float> fp32_k_expected(k_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_q_expected.data(), q_size);
        simd::dequantize_q8_1_to_fp32(q8_k.data(), fp32_k_expected.data(), k_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        reference_rope(fp32_q_expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);
        reference_rope(fp32_k_expected.data(), seq_len_, n_kv_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        // Apply Q8_1 RoPE
        primitives::apply_rope_q8_1_integer(
            q8_q.data(), q8_k.data(),
            position_ids.data(),
            seq_len_,
            n_heads_, n_kv_heads_,
            head_dim_,
            ROPE_THETA);

        // Dequantize results
        std::vector<float> fp32_q_result(q_size);
        std::vector<float> fp32_k_result(k_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), fp32_q_result.data(), q_size);
        simd::dequantize_q8_1_to_fp32(q8_k.data(), fp32_k_result.data(), k_size);

        // Verify Q
        float q_cosine = cosine_similarity(fp32_q_result.data(), fp32_q_expected.data(), q_size);
        EXPECT_GT(q_cosine, Q8_1_COSINE_MIN)
            << "Q8_1 RoPE Q cosine similarity: " << q_cosine;

        // Verify K
        float k_cosine = cosine_similarity(fp32_k_result.data(), fp32_k_expected.data(), k_size);
        EXPECT_GT(k_cosine, Q8_1_COSINE_MIN)
            << "Q8_1 RoPE K cosine similarity: " << k_cosine;
    }

    // =========================================================================
    // Cross-Precision Consistency Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTypedTest, AllPrecisions_SameInput_SimilarOutput)
    {
        // Verify that all precisions produce similar results for the same input
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        auto fp32_original = generate_random_fp32(q_size);

        std::vector<int> position_ids(seq_len_);
        std::iota(position_ids.begin(), position_ids.end(), 0);

        // FP32 result (ground truth)
        std::vector<float> fp32_q = fp32_original;
        CPURoPEKernelTyped<ActivationPrecision::FP32> kernel_fp32;
        kernel_fp32.apply_typed(
            fp32_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);

        // BF16 result
        std::vector<uint16_t> bf16_q(q_size);
        simd::convert_fp32_to_bf16(fp32_original.data(), bf16_q.data(), q_size);
        CPURoPEKernelTyped<ActivationPrecision::BF16> kernel_bf16;
        kernel_bf16.apply_typed(
            bf16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);
        std::vector<float> bf16_result(q_size);
        simd::convert_bf16_to_fp32(bf16_q.data(), bf16_result.data(), q_size);

        // FP16 result
        std::vector<uint16_t> fp16_q(q_size);
        simd::convert_fp32_to_fp16(fp32_original.data(), fp16_q.data(), q_size);
        CPURoPEKernelTyped<ActivationPrecision::FP16> kernel_fp16;
        kernel_fp16.apply_typed(
            fp16_q.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);
        std::vector<float> fp16_result(q_size);
        simd::convert_fp16_to_fp32(fp16_q.data(), fp16_result.data(), q_size);

        // Q8_1 result
        const size_t num_blocks = q_size / 32;
        std::vector<Q8_1Block> q8_q(num_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_original.data(), q8_q.data(), q_size);
        CPURoPEKernelTyped<ActivationPrecision::Q8_1> kernel_q8;
        kernel_q8.apply_typed(
            q8_q.data(), nullptr,
            position_ids.data(),
            seq_len_,
            n_heads_, 0,
            head_dim_,
            ROPE_THETA);
        std::vector<float> q8_result(q_size);
        simd::dequantize_q8_1_to_fp32(q8_q.data(), q8_result.data(), q_size);

        // Compare all to FP32 reference
        float bf16_cosine = cosine_similarity(bf16_result.data(), fp32_q.data(), q_size);
        float fp16_cosine = cosine_similarity(fp16_result.data(), fp32_q.data(), q_size);
        float q8_cosine = cosine_similarity(q8_result.data(), fp32_q.data(), q_size);

        EXPECT_GT(bf16_cosine, BF16_COSINE_MIN)
            << "BF16 vs FP32 cosine: " << bf16_cosine;
        EXPECT_GT(fp16_cosine, FP16_COSINE_MIN)
            << "FP16 vs FP32 cosine: " << fp16_cosine;
        EXPECT_GT(q8_cosine, Q8_1_COSINE_MIN)
            << "Q8_1 vs FP32 cosine: " << q8_cosine;
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    TEST_F(CPURoPEKernelTypedTest, FP32_skip_negative_position)
    {
        // Negative position IDs should be skipped (padding tokens)
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        auto q_data = generate_random_fp32(q_size);
        std::vector<float> original = q_data;

        // Mark position 1 as padding (-1)
        std::vector<int> position_ids = {0, -1, 2, 3};

        CPURoPEKernelTyped<ActivationPrecision::FP32> kernel;
        kernel.apply_typed(
            q_data.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);

        // Position 1 should be unchanged
        const int token_stride = n_heads_ * head_dim_;
        for (int i = 0; i < token_stride; ++i)
        {
            EXPECT_FLOAT_EQ(q_data[token_stride + i], original[token_stride + i])
                << "Position 1 (padding) should be unchanged at index " << i;
        }
    }

    TEST_F(CPURoPEKernelTypedTest, FP32_different_rope_theta)
    {
        // Test with different rope_theta values
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;

        for (float theta : {1000.0f, 10000.0f, 100000.0f})
        {
            auto q_data = generate_random_fp32(q_size);
            std::vector<float> expected = q_data;

            std::vector<int> position_ids(seq_len_);
            std::iota(position_ids.begin(), position_ids.end(), 0);

            reference_rope(expected.data(), seq_len_, n_heads_, head_dim_,
                           position_ids.data(), theta);

            CPURoPEKernelTyped<ActivationPrecision::FP32> kernel;
            kernel.apply_typed(
                q_data.data(), nullptr,
                position_ids.data(),
                seq_len_, n_heads_, 0, head_dim_,
                theta);

            float max_diff = max_abs_diff(q_data.data(), expected.data(), q_size);
            EXPECT_LT(max_diff, FP32_TOLERANCE)
                << "RoPE with theta=" << theta << " max diff: " << max_diff;
        }
    }

    TEST_F(CPURoPEKernelTypedTest, FP32_large_position_ids)
    {
        // Test with large position IDs (long context)
        const size_t q_size = seq_len_ * n_heads_ * head_dim_;
        auto q_data = generate_random_fp32(q_size);
        std::vector<float> expected = q_data;

        // Large position IDs (simulating long context)
        std::vector<int> position_ids = {1000, 1001, 1002, 1003};

        reference_rope(expected.data(), seq_len_, n_heads_, head_dim_,
                       position_ids.data(), ROPE_THETA);

        CPURoPEKernelTyped<ActivationPrecision::FP32> kernel;
        kernel.apply_typed(
            q_data.data(), nullptr,
            position_ids.data(),
            seq_len_, n_heads_, 0, head_dim_,
            ROPE_THETA);

        float max_diff = max_abs_diff(q_data.data(), expected.data(), q_size);
        EXPECT_LT(max_diff, FP32_LARGE_POS_TOLERANCE)
            << "RoPE with large positions max diff: " << max_diff;
    }

} // namespace llaminar2