/**
 * @file Test__INT8AttentionKernel.cpp
 * @brief Unit tests for INT8 multi-head attention kernel
 * @author David Sanftenberg
 * @date 2025-11-06
 *
 * Tests INT8 attention with INT32 accumulators:
 * - Basic forward pass (no causal mask)
 * - Causal masking correctness
 * - Multi-head quantization accuracy vs validated FP32 reference
 * - Attention pattern validation
 * - Edge cases (single head, single sequence, large batch)
 *
 * Updated 2025-11-06: Now uses validated FP32AttentionKernel as ground truth
 * instead of buggy reference_attention_fp32() helper.
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cpu/INT8AttentionKernel.h"
#include "../../../src/v2/kernels/cpu/FP32AttentionKernel.h" // Validated FP32 reference
#include "../../../src/v2/utils/Logger.h"
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__INT8AttentionKernel : public ::testing::Test
{
protected:
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    void SetUp() override
    {
        // Setup code if needed
    }

    // Helper: Fill vector with random INT8 values
    void fill_random_int8(std::vector<int8_t> &data, int8_t min_val, int8_t max_val)
    {
        std::uniform_int_distribution<int> dist(min_val, max_val);
        for (auto &val : data)
        {
            val = static_cast<int8_t>(dist(rng_));
        }
    }

    // Helper: Fill vector with random FP32 values
    void fill_random_fp32(std::vector<float> &data, float min_val, float max_val)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (auto &val : data)
        {
            val = dist(rng_);
        }
    }

    // Helper: Quantize FP32 to INT8 with per-row scaling
    void quantize_fp32_to_int8(
        const std::vector<float> &fp32_data,
        std::vector<int8_t> &int8_data,
        std::vector<float> &row_scales,
        int num_rows, int row_size)
    {
        int8_data.resize(num_rows * row_size);
        row_scales.resize(num_rows);

        for (int i = 0; i < num_rows; ++i)
        {
            // Find max abs in row
            float max_abs = 0.0f;
            for (int j = 0; j < row_size; ++j)
            {
                max_abs = std::max(max_abs, std::abs(fp32_data[i * row_size + j]));
            }

            // Compute scale
            row_scales[i] = (max_abs > 1e-6f) ? (max_abs / 127.0f) : 1.0f;
            float inv_scale = 1.0f / row_scales[i];

            // Quantize row
            for (int j = 0; j < row_size; ++j)
            {
                float scaled = fp32_data[i * row_size + j] * inv_scale;
                int8_data[i * row_size + j] = static_cast<int8_t>(
                    std::clamp(static_cast<int>(std::round(scaled)), -127, 127));
            }
        }
    }

    // Helper: Dequantize INT8 to FP32
    void dequantize_int8_to_fp32(
        const std::vector<int8_t> &int8_data,
        const std::vector<float> &row_scales,
        std::vector<float> &fp32_data,
        int num_rows, int row_size)
    {
        fp32_data.resize(num_rows * row_size);

        for (int i = 0; i < num_rows; ++i)
        {
            float scale = row_scales[i];
            for (int j = 0; j < row_size; ++j)
            {
                fp32_data[i * row_size + j] = static_cast<float>(int8_data[i * row_size + j]) * scale;
            }
        }
    }

    // Helper: Compute relative L2 error
    float compute_relative_l2(const std::vector<float> &expected, const std::vector<float> &actual)
    {
        float sum_sq_error = 0.0f;
        float sum_sq_expected = 0.0f;

        for (size_t i = 0; i < expected.size(); ++i)
        {
            float error = actual[i] - expected[i];
            sum_sq_error += error * error;
            sum_sq_expected += expected[i] * expected[i];
        }

        if (sum_sq_expected < 1e-12f)
        {
            return std::sqrt(sum_sq_error);
        }

        return std::sqrt(sum_sq_error / sum_sq_expected);
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

/**
 * @brief Test basic forward pass without causal masking
 */
TEST_F(Test__INT8AttentionKernel, BasicForwardPass)
{
    const int batch = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    // Create random Q, K, V in INT8
    std::vector<int8_t> q_int8(batch * seq_len * d_model);
    std::vector<int8_t> k_int8(batch * seq_len * d_model);
    std::vector<int8_t> v_int8(batch * seq_len * d_model);

    fill_random_int8(q_int8, -50, 50);
    fill_random_int8(k_int8, -50, 50);
    fill_random_int8(v_int8, -50, 50);

    // Create random scales
    std::vector<float> q_row_scales(batch * seq_len);
    std::vector<float> k_row_scales(batch * seq_len);
    std::vector<float> v_row_scales(batch * seq_len);

    fill_random_fp32(q_row_scales, 0.01f, 0.05f);
    fill_random_fp32(k_row_scales, 0.01f, 0.05f);
    fill_random_fp32(v_row_scales, 0.01f, 0.05f);

    // Allocate output
    std::vector<int8_t> output_int8(batch * seq_len * d_model);
    std::vector<float> output_row_scales(batch * seq_len);

    // Run attention
    INT8AttentionKernel attn(n_heads, d_head);
    bool success = attn.forward(
        q_int8.data(), q_row_scales.data(),
        k_int8.data(), k_row_scales.data(),
        v_int8.data(), v_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len,
        false); // No causal mask

    ASSERT_TRUE(success) << "Forward pass should succeed";

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (auto val : output_int8)
    {
        if (val != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";

    // Verify scales are positive
    for (auto scale : output_row_scales)
    {
        EXPECT_GT(scale, 0.0f) << "Output scales should be positive";
    }
}

/**
 * @brief Test causal masking
 */
TEST_F(Test__INT8AttentionKernel, CausalMasking)
{
    const int batch = 1;
    const int seq_len = 4;
    const int n_heads = 1;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    // Create simple Q, K, V
    std::vector<int8_t> q_int8(batch * seq_len * d_model);
    std::vector<int8_t> k_int8(batch * seq_len * d_model);
    std::vector<int8_t> v_int8(batch * seq_len * d_model);

    // Fill with simple pattern
    for (size_t i = 0; i < q_int8.size(); ++i)
    {
        q_int8[i] = static_cast<int8_t>(10 + (i % 20));
        k_int8[i] = static_cast<int8_t>(20 + (i % 20));
        v_int8[i] = static_cast<int8_t>(30 + (i % 20));
    }

    std::vector<float> q_row_scales(batch * seq_len, 0.02f);
    std::vector<float> k_row_scales(batch * seq_len, 0.02f);
    std::vector<float> v_row_scales(batch * seq_len, 0.02f);

    // Run without causal mask
    std::vector<int8_t> output_no_mask(batch * seq_len * d_model);
    std::vector<float> output_scales_no_mask(batch * seq_len);

    INT8AttentionKernel attn(n_heads, d_head);
    bool success = attn.forward(
        q_int8.data(), q_row_scales.data(),
        k_int8.data(), k_row_scales.data(),
        v_int8.data(), v_row_scales.data(),
        output_no_mask.data(), output_scales_no_mask.data(),
        batch, seq_len,
        false);

    ASSERT_TRUE(success);

    // Run with causal mask
    std::vector<int8_t> output_with_mask(batch * seq_len * d_model);
    std::vector<float> output_scales_with_mask(batch * seq_len);

    success = attn.forward(
        q_int8.data(), q_row_scales.data(),
        k_int8.data(), k_row_scales.data(),
        v_int8.data(), v_row_scales.data(),
        output_with_mask.data(), output_scales_with_mask.data(),
        batch, seq_len,
        true); // Causal mask

    ASSERT_TRUE(success);

    // Results should be different (causal mask changes attention pattern)
    int diff_count = 0;
    for (size_t i = 0; i < output_no_mask.size(); ++i)
    {
        if (output_no_mask[i] != output_with_mask[i])
        {
            diff_count++;
        }
    }

    EXPECT_GT(diff_count, 0) << "Causal masking should change output";
}

/**
 * @brief Test accuracy vs FP32 reference
 */
TEST_F(Test__INT8AttentionKernel, AccuracyVsFP32Reference)
{
    const int batch = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    // Create FP32 Q, K, V
    std::vector<float> q_fp32(batch * seq_len * d_model);
    std::vector<float> k_fp32(batch * seq_len * d_model);
    std::vector<float> v_fp32(batch * seq_len * d_model);

    fill_random_fp32(q_fp32, -1.0f, 1.0f);
    fill_random_fp32(k_fp32, -1.0f, 1.0f);
    fill_random_fp32(v_fp32, -1.0f, 1.0f);

    // Quantize to INT8
    std::vector<int8_t> q_int8, k_int8, v_int8;
    std::vector<float> q_row_scales, k_row_scales, v_row_scales;

    quantize_fp32_to_int8(q_fp32, q_int8, q_row_scales, batch * seq_len, d_model);
    quantize_fp32_to_int8(k_fp32, k_int8, k_row_scales, batch * seq_len, d_model);
    quantize_fp32_to_int8(v_fp32, v_int8, v_row_scales, batch * seq_len, d_model);

    // Run INT8 attention
    std::vector<int8_t> output_int8(batch * seq_len * d_model);
    std::vector<float> output_row_scales(batch * seq_len);

    INT8AttentionKernel attn(n_heads, d_head);
    bool success = attn.forward(
        q_int8.data(), q_row_scales.data(),
        k_int8.data(), k_row_scales.data(),
        v_int8.data(), v_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len,
        false);

    ASSERT_TRUE(success);

    // Dequantize INT8 output
    std::vector<float> output_fp32_from_int8;
    dequantize_int8_to_fp32(output_int8, output_row_scales, output_fp32_from_int8,
                            batch * seq_len, d_model);

    // Compute FP32 reference using validated FP32AttentionKernel
    std::vector<float> output_fp32_reference(batch * seq_len * d_model);
    FP32AttentionKernel fp32_attn(n_heads, d_head); // Standard MHA (n_kv_heads defaults to n_heads)
    bool fp32_success = fp32_attn.forward(
        q_fp32.data(), k_fp32.data(), v_fp32.data(),
        output_fp32_reference.data(),
        batch, seq_len, false);
    ASSERT_TRUE(fp32_success) << "FP32 reference forward failed";

    // Debug: Print sample values from both outputs
    LOG_INFO("First 5 FP32 reference values: "
             << output_fp32_reference[0] << ", " << output_fp32_reference[1] << ", "
             << output_fp32_reference[2] << ", " << output_fp32_reference[3] << ", "
             << output_fp32_reference[4]);
    LOG_INFO("First 5 INT8→FP32 values: "
             << output_fp32_from_int8[0] << ", " << output_fp32_from_int8[1] << ", "
             << output_fp32_from_int8[2] << ", " << output_fp32_from_int8[3] << ", "
             << output_fp32_from_int8[4]);

    // Compare accuracy
    float rel_error = compute_relative_l2(output_fp32_reference, output_fp32_from_int8);

    // INT8 quantization introduces error, but should be < 5%
    EXPECT_LT(rel_error, 0.05f) << "INT8 attention should be reasonably accurate vs FP32";

    LOG_INFO("INT8 vs FP32 relative L2 error: " << rel_error);
}

/**
 * @brief Test single head (edge case)
 */
TEST_F(Test__INT8AttentionKernel, SingleHead)
{
    const int batch = 1;
    const int seq_len = 3;
    const int n_heads = 1;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    std::vector<int8_t> q_int8(batch * seq_len * d_model);
    std::vector<int8_t> k_int8(batch * seq_len * d_model);
    std::vector<int8_t> v_int8(batch * seq_len * d_model);

    fill_random_int8(q_int8, -50, 50);
    fill_random_int8(k_int8, -50, 50);
    fill_random_int8(v_int8, -50, 50);

    std::vector<float> q_row_scales(batch * seq_len, 0.03f);
    std::vector<float> k_row_scales(batch * seq_len, 0.03f);
    std::vector<float> v_row_scales(batch * seq_len, 0.03f);

    std::vector<int8_t> output_int8(batch * seq_len * d_model);
    std::vector<float> output_row_scales(batch * seq_len);

    INT8AttentionKernel attn(n_heads, d_head);
    bool success = attn.forward(
        q_int8.data(), q_row_scales.data(),
        k_int8.data(), k_row_scales.data(),
        v_int8.data(), v_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len,
        false);

    ASSERT_TRUE(success);
}

/**
 * @brief Test single sequence (edge case)
 */
TEST_F(Test__INT8AttentionKernel, SingleSequence)
{
    const int batch = 1;
    const int seq_len = 1;
    const int n_heads = 2;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    std::vector<int8_t> q_int8(batch * seq_len * d_model);
    std::vector<int8_t> k_int8(batch * seq_len * d_model);
    std::vector<int8_t> v_int8(batch * seq_len * d_model);

    fill_random_int8(q_int8, -50, 50);
    fill_random_int8(k_int8, -50, 50);
    fill_random_int8(v_int8, -50, 50);

    std::vector<float> q_row_scales(batch * seq_len, 0.03f);
    std::vector<float> k_row_scales(batch * seq_len, 0.03f);
    std::vector<float> v_row_scales(batch * seq_len, 0.03f);

    std::vector<int8_t> output_int8(batch * seq_len * d_model);
    std::vector<float> output_row_scales(batch * seq_len);

    INT8AttentionKernel attn(n_heads, d_head);
    bool success = attn.forward(
        q_int8.data(), q_row_scales.data(),
        k_int8.data(), k_row_scales.data(),
        v_int8.data(), v_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len,
        false);

    ASSERT_TRUE(success);

    // With seq_len=1, attention weights should be [1.0] (only attends to itself)
    // Output should be approximately equal to V
    std::vector<float> v_fp32;
    dequantize_int8_to_fp32(v_int8, v_row_scales, v_fp32, batch * seq_len, d_model);

    std::vector<float> output_fp32;
    dequantize_int8_to_fp32(output_int8, output_row_scales, output_fp32, batch * seq_len, d_model);

    float rel_error = compute_relative_l2(v_fp32, output_fp32);
    EXPECT_LT(rel_error, 0.1f) << "Single sequence should produce output ≈ V";
}

/**
 * @brief Test large batch
 */
TEST_F(Test__INT8AttentionKernel, LargeBatch)
{
    const int batch = 8;
    const int seq_len = 6;
    const int n_heads = 4;
    const int d_head = 16;
    const int d_model = n_heads * d_head;

    std::vector<int8_t> q_int8(batch * seq_len * d_model);
    std::vector<int8_t> k_int8(batch * seq_len * d_model);
    std::vector<int8_t> v_int8(batch * seq_len * d_model);

    fill_random_int8(q_int8, -50, 50);
    fill_random_int8(k_int8, -50, 50);
    fill_random_int8(v_int8, -50, 50);

    std::vector<float> q_row_scales(batch * seq_len);
    std::vector<float> k_row_scales(batch * seq_len);
    std::vector<float> v_row_scales(batch * seq_len);

    fill_random_fp32(q_row_scales, 0.01f, 0.05f);
    fill_random_fp32(k_row_scales, 0.01f, 0.05f);
    fill_random_fp32(v_row_scales, 0.01f, 0.05f);

    std::vector<int8_t> output_int8(batch * seq_len * d_model);
    std::vector<float> output_row_scales(batch * seq_len);

    INT8AttentionKernel attn(n_heads, d_head);
    bool success = attn.forward(
        q_int8.data(), q_row_scales.data(),
        k_int8.data(), k_row_scales.data(),
        v_int8.data(), v_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len,
        false);

    ASSERT_TRUE(success);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/**
 * @brief Test null pointer handling
 */
TEST_F(Test__INT8AttentionKernel, NullPointerHandling)
{
    const int batch = 1;
    const int seq_len = 2;
    const int n_heads = 1;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    std::vector<int8_t> q_int8(batch * seq_len * d_model, 10);
    std::vector<int8_t> k_int8(batch * seq_len * d_model, 10);
    std::vector<int8_t> v_int8(batch * seq_len * d_model, 10);
    std::vector<float> scales(batch * seq_len, 0.02f);
    std::vector<int8_t> output_int8(batch * seq_len * d_model);
    std::vector<float> output_scales(batch * seq_len);

    INT8AttentionKernel attn(n_heads, d_head);

    // Null Q
    EXPECT_FALSE(attn.forward(
        nullptr, scales.data(),
        k_int8.data(), scales.data(),
        v_int8.data(), scales.data(),
        output_int8.data(), output_scales.data(),
        batch, seq_len, false));

    // Null K
    EXPECT_FALSE(attn.forward(
        q_int8.data(), scales.data(),
        nullptr, scales.data(),
        v_int8.data(), scales.data(),
        output_int8.data(), output_scales.data(),
        batch, seq_len, false));

    // Null V
    EXPECT_FALSE(attn.forward(
        q_int8.data(), scales.data(),
        k_int8.data(), scales.data(),
        nullptr, scales.data(),
        output_int8.data(), output_scales.data(),
        batch, seq_len, false));

    // Null output
    EXPECT_FALSE(attn.forward(
        q_int8.data(), scales.data(),
        k_int8.data(), scales.data(),
        v_int8.data(), scales.data(),
        nullptr, output_scales.data(),
        batch, seq_len, false));

    // Null output scales
    EXPECT_FALSE(attn.forward(
        q_int8.data(), scales.data(),
        k_int8.data(), scales.data(),
        v_int8.data(), scales.data(),
        output_int8.data(), nullptr,
        batch, seq_len, false));
}

/**
 * @brief Test invalid dimensions
 */
TEST_F(Test__INT8AttentionKernel, InvalidDimensions)
{
    const int n_heads = 2;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    std::vector<int8_t> data(16, 10);
    std::vector<float> scales(4, 0.02f);
    std::vector<int8_t> output(16);
    std::vector<float> output_scales(4);

    INT8AttentionKernel attn(n_heads, d_head);

    // Zero batch
    EXPECT_FALSE(attn.forward(
        data.data(), scales.data(),
        data.data(), scales.data(),
        data.data(), scales.data(),
        output.data(), output_scales.data(),
        0, 2, false));

    // Zero seq_len
    EXPECT_FALSE(attn.forward(
        data.data(), scales.data(),
        data.data(), scales.data(),
        data.data(), scales.data(),
        output.data(), output_scales.data(),
        1, 0, false));

    // Negative batch
    EXPECT_FALSE(attn.forward(
        data.data(), scales.data(),
        data.data(), scales.data(),
        data.data(), scales.data(),
        output.data(), output_scales.data(),
        -1, 2, false));
}

/**
 * @brief Test invalid device (non-CPU)
 */
TEST_F(Test__INT8AttentionKernel, InvalidDevice)
{
    const int batch = 1;
    const int seq_len = 2;
    const int n_heads = 1;
    const int d_head = 8;
    const int d_model = n_heads * d_head;

    std::vector<int8_t> q_int8(batch * seq_len * d_model, 10);
    std::vector<int8_t> k_int8(batch * seq_len * d_model, 10);
    std::vector<int8_t> v_int8(batch * seq_len * d_model, 10);
    std::vector<float> scales(batch * seq_len, 0.02f);
    std::vector<int8_t> output_int8(batch * seq_len * d_model);
    std::vector<float> output_scales(batch * seq_len);

    // Create attention kernel with GPU device (should fail)
    INT8AttentionKernel attn(n_heads, d_head, 0); // device_idx=0 (GPU)

    bool success = attn.forward(
        q_int8.data(), scales.data(),
        k_int8.data(), scales.data(),
        v_int8.data(), scales.data(),
        output_int8.data(), output_scales.data(),
        batch, seq_len, false);

    EXPECT_FALSE(success) << "GPU device should be rejected";
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
