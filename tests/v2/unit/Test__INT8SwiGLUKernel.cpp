/**
 * @file Test__INT8SwiGLUKernel.cpp
 * @brief Tests for INT8 SwiGLU activation kernel
 * @author David Sanftenberg
 * @date 2025-11-06
 *
 * Test Coverage:
 *   1. BasicForwardPass - Kernel executes without crashes
 *   2. AccuracyVsFP32Reference - Quantization accuracy vs FP32 SwiGLU
 *   3. SigmoidNumericalStability - Sigmoid handles extreme values
 *   4. SiLUProperties - SiLU(0)=0, SiLU matches x*sigmoid(x)
 *   5. SingleToken - Edge case with seq_len=1
 *   6. LargeBatch - Scales to batch=8, seq_len=16
 *   7. NullPointerHandling - Proper error handling
 *   8. InvalidDimensions - Validates input dimensions
 *   9. InvalidDevice - CPU-only enforcement
 */

#include "../../../src/v2/kernels/cpu/INT8SwiGLUKernel.h"
#include "../../../src/v2/utils/Logger.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

/**
 * @brief Test fixture for INT8SwiGLUKernel
 */
class Test__INT8SwiGLUKernel : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed RNG for reproducibility
        rng_.seed(42);
    }

    std::mt19937 rng_;

    // Helper: Fill vector with random INT8 values
    void fill_random_int8(std::vector<int8_t> &vec, int min_val = -50, int max_val = 50)
    {
        std::uniform_int_distribution<int> dist(min_val, max_val);
        for (auto &val : vec)
        {
            val = static_cast<int8_t>(dist(rng_));
        }
    }

    // Helper: Fill vector with random FP32 values
    void fill_random_fp32(std::vector<float> &vec, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (auto &val : vec)
        {
            val = dist(rng_);
        }
    }

    // Helper: Quantize FP32 to INT8 with per-row scales
    void quantize_fp32_to_int8(
        const std::vector<float> &fp32,
        std::vector<int8_t> &int8,
        std::vector<float> &row_scales,
        int num_rows,
        int row_size)
    {
        int8.resize(num_rows * row_size);
        row_scales.resize(num_rows);

        for (int r = 0; r < num_rows; ++r)
        {
            // Find max abs in row
            float max_abs = 0.0f;
            for (int c = 0; c < row_size; ++c)
            {
                max_abs = std::max(max_abs, std::abs(fp32[r * row_size + c]));
            }

            // Compute scale
            float scale = (max_abs > 1e-6f) ? (max_abs / 127.0f) : 1.0f;
            row_scales[r] = scale;

            float inv_scale = 1.0f / scale;

            // Quantize row
            for (int c = 0; c < row_size; ++c)
            {
                int idx = r * row_size + c;
                float scaled = fp32[idx] * inv_scale;
                int quantized = static_cast<int>(std::round(scaled));
                int8[idx] = static_cast<int8_t>(std::clamp(quantized, -127, 127));
            }
        }
    }

    // Helper: Dequantize INT8 to FP32
    void dequantize_int8_to_fp32(
        const std::vector<int8_t> &int8,
        const std::vector<float> &row_scales,
        std::vector<float> &fp32,
        int num_rows,
        int row_size)
    {
        fp32.resize(num_rows * row_size);

        for (int r = 0; r < num_rows; ++r)
        {
            float scale = row_scales[r];
            for (int c = 0; c < row_size; ++c)
            {
                int idx = r * row_size + c;
                fp32[idx] = static_cast<float>(int8[idx]) * scale;
            }
        }
    }

    // Helper: Compute relative L2 error
    float compute_relative_l2(const std::vector<float> &expected, const std::vector<float> &actual)
    {
        if (expected.size() != actual.size())
        {
            return std::numeric_limits<float>::infinity();
        }

        float sum_sq_diff = 0.0f;
        float sum_sq_expected = 0.0f;

        for (size_t i = 0; i < expected.size(); ++i)
        {
            float diff = expected[i] - actual[i];
            sum_sq_diff += diff * diff;
            sum_sq_expected += expected[i] * expected[i];
        }

        if (sum_sq_expected < 1e-10f)
        {
            return (sum_sq_diff < 1e-10f) ? 0.0f : std::numeric_limits<float>::infinity();
        }

        return std::sqrt(sum_sq_diff / sum_sq_expected);
    }

    // Helper: FP32 reference SwiGLU implementation
    void reference_swiglu_fp32(
        const std::vector<float> &gate_fp32,
        const std::vector<float> &up_fp32,
        std::vector<float> &output_fp32)
    {
        output_fp32.resize(gate_fp32.size());

        for (size_t i = 0; i < gate_fp32.size(); ++i)
        {
            float gate = gate_fp32[i];
            float up = up_fp32[i];

            // SiLU(x) = x * sigmoid(x)
            float sigmoid_up = 1.0f / (1.0f + std::exp(-up));
            float silu_up = up * sigmoid_up;

            // SwiGLU = gate × SiLU(up)
            output_fp32[i] = gate * silu_up;
        }
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

/**
 * @brief Test basic forward pass
 */
TEST_F(Test__INT8SwiGLUKernel, BasicForwardPass)
{
    const int batch = 2;
    const int seq_len = 4;
    const int d_ff = 32;

    std::vector<int8_t> gate_int8(batch * seq_len * d_ff);
    std::vector<int8_t> up_int8(batch * seq_len * d_ff);

    fill_random_int8(gate_int8, -50, 50);
    fill_random_int8(up_int8, -50, 50);

    std::vector<float> gate_row_scales(batch * seq_len, 0.02f);
    std::vector<float> up_row_scales(batch * seq_len, 0.02f);

    std::vector<int8_t> output_int8(batch * seq_len * d_ff);
    std::vector<float> output_row_scales(batch * seq_len);

    llaminar2::INT8SwiGLUKernel swiglu;
    bool success = swiglu.forward(
        gate_int8.data(), gate_row_scales.data(),
        up_int8.data(), up_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len, d_ff);

    ASSERT_TRUE(success);

    // Verify output scales are reasonable
    for (float scale : output_row_scales)
    {
        EXPECT_GT(scale, 0.0f);
        EXPECT_LT(scale, 10.0f); // Reasonable range
    }
}

/**
 * @brief Test accuracy vs FP32 reference
 */
TEST_F(Test__INT8SwiGLUKernel, AccuracyVsFP32Reference)
{
    const int batch = 2;
    const int seq_len = 4;
    const int d_ff = 64;

    // Create FP32 gate and up
    std::vector<float> gate_fp32(batch * seq_len * d_ff);
    std::vector<float> up_fp32(batch * seq_len * d_ff);

    fill_random_fp32(gate_fp32, -1.0f, 1.0f);
    fill_random_fp32(up_fp32, -2.0f, 2.0f); // Wider range for up

    // Quantize to INT8
    std::vector<int8_t> gate_int8, up_int8;
    std::vector<float> gate_row_scales, up_row_scales;

    quantize_fp32_to_int8(gate_fp32, gate_int8, gate_row_scales, batch * seq_len, d_ff);
    quantize_fp32_to_int8(up_fp32, up_int8, up_row_scales, batch * seq_len, d_ff);

    // Run INT8 SwiGLU
    std::vector<int8_t> output_int8(batch * seq_len * d_ff);
    std::vector<float> output_row_scales(batch * seq_len);

    llaminar2::INT8SwiGLUKernel swiglu;
    bool success = swiglu.forward(
        gate_int8.data(), gate_row_scales.data(),
        up_int8.data(), up_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len, d_ff);

    ASSERT_TRUE(success);

    // Dequantize INT8 output
    std::vector<float> output_fp32_from_int8;
    dequantize_int8_to_fp32(output_int8, output_row_scales, output_fp32_from_int8,
                            batch * seq_len, d_ff);

    // Compute FP32 reference
    std::vector<float> output_fp32_reference;
    reference_swiglu_fp32(gate_fp32, up_fp32, output_fp32_reference);

    // Compare accuracy
    float rel_error = compute_relative_l2(output_fp32_reference, output_fp32_from_int8);

    // INT8 quantization introduces error, but should be < 5%
    EXPECT_LT(rel_error, 0.05f) << "INT8 SwiGLU should be reasonably accurate vs FP32";

    LOG_INFO("INT8 vs FP32 relative L2 error: " << rel_error);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

/**
 * @brief Test sigmoid numerical stability with extreme values
 */
TEST_F(Test__INT8SwiGLUKernel, SigmoidNumericalStability)
{
    const int batch = 1;
    const int seq_len = 1;
    const int d_ff = 5;

    // Test extreme values
    std::vector<float> up_fp32 = {-100.0f, -10.0f, 0.0f, 10.0f, 100.0f};

    // Quantize
    std::vector<int8_t> gate_int8(d_ff, 127); // Gate = 1.0 (max scale)
    std::vector<int8_t> up_int8;
    std::vector<float> gate_row_scales = {1.0f / 127.0f};
    std::vector<float> up_row_scales;

    quantize_fp32_to_int8(up_fp32, up_int8, up_row_scales, 1, d_ff);

    // Run SwiGLU
    std::vector<int8_t> output_int8(d_ff);
    std::vector<float> output_row_scales(1);

    llaminar2::INT8SwiGLUKernel swiglu;
    bool success = swiglu.forward(
        gate_int8.data(), gate_row_scales.data(),
        up_int8.data(), up_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len, d_ff);

    ASSERT_TRUE(success);

    // Dequantize output
    std::vector<float> output_fp32;
    dequantize_int8_to_fp32(output_int8, output_row_scales, output_fp32, 1, d_ff);

    // Verify results are finite and reasonable
    for (float val : output_fp32)
    {
        EXPECT_TRUE(std::isfinite(val)) << "Output should be finite (no NaN/Inf)";
    }

    // For large positive up, SiLU(up) ≈ up, so output ≈ gate × up
    // For large negative up, SiLU(up) ≈ 0, so output ≈ 0
    EXPECT_NEAR(output_fp32[0], 0.0f, 0.1f) << "SiLU(-100) should be ≈ 0";
    EXPECT_GT(output_fp32[4], 50.0f) << "SiLU(100) should be large";
}

/**
 * @brief Test SiLU properties
 */
TEST_F(Test__INT8SwiGLUKernel, SiLUProperties)
{
    const int batch = 1;
    const int seq_len = 1;
    const int d_ff = 3;

    // Test SiLU(0) = 0
    std::vector<float> up_fp32 = {0.0f, 1.0f, -1.0f};

    std::vector<int8_t> gate_int8(d_ff, 127); // Gate = 1.0
    std::vector<int8_t> up_int8;
    std::vector<float> gate_row_scales = {1.0f / 127.0f};
    std::vector<float> up_row_scales;

    quantize_fp32_to_int8(up_fp32, up_int8, up_row_scales, 1, d_ff);

    std::vector<int8_t> output_int8(d_ff);
    std::vector<float> output_row_scales(1);

    llaminar2::INT8SwiGLUKernel swiglu;
    swiglu.forward(
        gate_int8.data(), gate_row_scales.data(),
        up_int8.data(), up_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len, d_ff);

    std::vector<float> output_fp32;
    dequantize_int8_to_fp32(output_int8, output_row_scales, output_fp32, 1, d_ff);

    // SiLU(0) = 0 × sigmoid(0) = 0 × 0.5 = 0
    EXPECT_NEAR(output_fp32[0], 0.0f, 0.05f) << "SiLU(0) should be ≈ 0";

    // SiLU(1) = 1 × sigmoid(1) ≈ 1 × 0.731 ≈ 0.731
    EXPECT_NEAR(output_fp32[1], 0.731f, 0.1f) << "SiLU(1) should be ≈ 0.731";

    // SiLU is odd-ish (not strictly odd, but negative for negative inputs)
    EXPECT_LT(output_fp32[2], 0.0f) << "SiLU(-1) should be negative";
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/**
 * @brief Test single token (edge case)
 */
TEST_F(Test__INT8SwiGLUKernel, SingleToken)
{
    const int batch = 1;
    const int seq_len = 1;
    const int d_ff = 16;

    std::vector<int8_t> gate_int8(batch * seq_len * d_ff);
    std::vector<int8_t> up_int8(batch * seq_len * d_ff);

    fill_random_int8(gate_int8, -50, 50);
    fill_random_int8(up_int8, -50, 50);

    std::vector<float> gate_row_scales(batch * seq_len, 0.03f);
    std::vector<float> up_row_scales(batch * seq_len, 0.03f);

    std::vector<int8_t> output_int8(batch * seq_len * d_ff);
    std::vector<float> output_row_scales(batch * seq_len);

    llaminar2::INT8SwiGLUKernel swiglu;
    bool success = swiglu.forward(
        gate_int8.data(), gate_row_scales.data(),
        up_int8.data(), up_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len, d_ff);

    ASSERT_TRUE(success);
}

/**
 * @brief Test large batch
 */
TEST_F(Test__INT8SwiGLUKernel, LargeBatch)
{
    const int batch = 8;
    const int seq_len = 16;
    const int d_ff = 128;

    std::vector<int8_t> gate_int8(batch * seq_len * d_ff);
    std::vector<int8_t> up_int8(batch * seq_len * d_ff);

    fill_random_int8(gate_int8, -50, 50);
    fill_random_int8(up_int8, -50, 50);

    std::vector<float> gate_row_scales(batch * seq_len);
    std::vector<float> up_row_scales(batch * seq_len);

    for (auto &scale : gate_row_scales)
        scale = 0.02f;
    for (auto &scale : up_row_scales)
        scale = 0.02f;

    std::vector<int8_t> output_int8(batch * seq_len * d_ff);
    std::vector<float> output_row_scales(batch * seq_len);

    llaminar2::INT8SwiGLUKernel swiglu;
    bool success = swiglu.forward(
        gate_int8.data(), gate_row_scales.data(),
        up_int8.data(), up_row_scales.data(),
        output_int8.data(), output_row_scales.data(),
        batch, seq_len, d_ff);

    ASSERT_TRUE(success);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/**
 * @brief Test null pointer handling
 */
TEST_F(Test__INT8SwiGLUKernel, NullPointerHandling)
{
    const int batch = 1;
    const int seq_len = 2;
    const int d_ff = 8;

    std::vector<int8_t> gate_int8(batch * seq_len * d_ff);
    std::vector<int8_t> up_int8(batch * seq_len * d_ff);
    std::vector<float> gate_row_scales(batch * seq_len);
    std::vector<float> up_row_scales(batch * seq_len);
    std::vector<int8_t> output_int8(batch * seq_len * d_ff);
    std::vector<float> output_row_scales(batch * seq_len);

    llaminar2::INT8SwiGLUKernel swiglu;

    // Test null gate_int8
    EXPECT_FALSE(swiglu.forward(nullptr, gate_row_scales.data(),
                                up_int8.data(), up_row_scales.data(),
                                output_int8.data(), output_row_scales.data(),
                                batch, seq_len, d_ff));

    // Test null up_int8
    EXPECT_FALSE(swiglu.forward(gate_int8.data(), gate_row_scales.data(),
                                nullptr, up_row_scales.data(),
                                output_int8.data(), output_row_scales.data(),
                                batch, seq_len, d_ff));

    // Test null output_int8
    EXPECT_FALSE(swiglu.forward(gate_int8.data(), gate_row_scales.data(),
                                up_int8.data(), up_row_scales.data(),
                                nullptr, output_row_scales.data(),
                                batch, seq_len, d_ff));
}

/**
 * @brief Test invalid dimensions
 */
TEST_F(Test__INT8SwiGLUKernel, InvalidDimensions)
{
    std::vector<int8_t> gate_int8(16);
    std::vector<int8_t> up_int8(16);
    std::vector<float> gate_row_scales(2);
    std::vector<float> up_row_scales(2);
    std::vector<int8_t> output_int8(16);
    std::vector<float> output_row_scales(2);

    llaminar2::INT8SwiGLUKernel swiglu;

    // Test batch = 0
    EXPECT_FALSE(swiglu.forward(gate_int8.data(), gate_row_scales.data(),
                                up_int8.data(), up_row_scales.data(),
                                output_int8.data(), output_row_scales.data(),
                                0, 2, 8));

    // Test seq_len = 0
    EXPECT_FALSE(swiglu.forward(gate_int8.data(), gate_row_scales.data(),
                                up_int8.data(), up_row_scales.data(),
                                output_int8.data(), output_row_scales.data(),
                                1, 0, 8));

    // Test d_ff = 0
    EXPECT_FALSE(swiglu.forward(gate_int8.data(), gate_row_scales.data(),
                                up_int8.data(), up_row_scales.data(),
                                output_int8.data(), output_row_scales.data(),
                                1, 2, 0));

    // Test negative batch
    EXPECT_FALSE(swiglu.forward(gate_int8.data(), gate_row_scales.data(),
                                up_int8.data(), up_row_scales.data(),
                                output_int8.data(), output_row_scales.data(),
                                -1, 2, 8));
}

/**
 * @brief Test invalid device
 */
TEST_F(Test__INT8SwiGLUKernel, InvalidDevice)
{
    std::vector<int8_t> gate_int8(16);
    std::vector<int8_t> up_int8(16);
    std::vector<float> gate_row_scales(2);
    std::vector<float> up_row_scales(2);
    std::vector<int8_t> output_int8(16);
    std::vector<float> output_row_scales(2);

    llaminar2::INT8SwiGLUKernel swiglu_gpu(0); // GPU device

    EXPECT_FALSE(swiglu_gpu.forward(gate_int8.data(), gate_row_scales.data(),
                                    up_int8.data(), up_row_scales.data(),
                                    output_int8.data(), output_row_scales.data(),
                                    1, 2, 8));
}
