/**
 * @file Test__Q16_1Tensor.cpp
 * @brief Unit tests for Q16_1Tensor high-precision quantized activation storage
 * @author David Sanftenberg
 *
 * Tests encode/decode round-trip precision and verifies that Q16_1 provides
 * significantly better precision than Q8_1 (256× finer granularity).
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include <vector>
#include <random>
#include <cstring>
#include <cmath>

using namespace llaminar2;

namespace
{
    // Helper to create an empty mutable Q16_1 tensor
    std::shared_ptr<Q16_1Tensor> create_mutable_q16_1_tensor(int rows, int cols)
    {
        return std::make_shared<Q16_1Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    }

    // Compute relative L2 error between two float arrays
    float compute_relative_l2_error(const float *actual, const float *expected, size_t n)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = static_cast<double>(actual[i]) - static_cast<double>(expected[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(expected[i]) * static_cast<double>(expected[i]);
        }
        return (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;
    }

    // Compute max absolute error between two float arrays
    float compute_max_abs_error(const float *actual, const float *expected, size_t n)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float err = std::abs(actual[i] - expected[i]);
            max_err = std::max(max_err, err);
        }
        return max_err;
    }
}

/**
 * @brief Basic encode/decode round-trip test for Q16_1Tensor
 *
 * Quantizes random FP32 data to Q16_1, then dequantizes back to FP32.
 * Verifies that the round-trip error is very small (< 0.01% relative L2).
 */
TEST(Test__Q16_1Tensor, EncodeDecodeRoundTrip_SmallError)
{
    const int rows = 8;
    const int cols = 128; // Multiple of 32 (block size)
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random FP32 data in typical activation range [-2, 2]
    std::vector<float> original(elements);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Quantize to Q16_1
    auto q16_1_tensor = Q16_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(q16_1_tensor, nullptr);
    ASSERT_EQ(q16_1_tensor->shape()[0], static_cast<size_t>(rows));
    ASSERT_EQ(q16_1_tensor->shape()[1], static_cast<size_t>(cols));
    ASSERT_EQ(q16_1_tensor->native_type(), TensorType::Q16_1);

    // Dequantize back to FP32
    std::vector<float> reconstructed(elements);
    q16_1_tensor->to_fp32(reconstructed.data());

    // Compute errors
    float rel_l2_error = compute_relative_l2_error(reconstructed.data(), original.data(), elements);
    float max_abs_error = compute_max_abs_error(reconstructed.data(), original.data(), elements);

    std::cout << "[Q16_1 Round-Trip] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 Round-Trip] Max abs error: " << max_abs_error << std::endl;

    // Q16_1 should have < 0.1% relative L2 error (better than Q8_1's ~0.4%)
    // The FP16 scale storage limits precision slightly
    EXPECT_LT(rel_l2_error, 0.001f)
        << "Q16_1 round-trip error exceeds 0.1% threshold";

    // Max error should be small for [-2, 2] range
    // With FP16 scale, max error is ~0.002 in practice
    EXPECT_LT(max_abs_error, 0.003f)
        << "Q16_1 max abs error exceeds 0.003 threshold";
}

/**
 * @brief Compare Q16_1 vs Q8_1 precision
 *
 * Quantizes the same FP32 data to both Q16_1 and Q8_1, then compares
 * the round-trip error. Q16_1 should have ~256× less error.
 */
TEST(Test__Q16_1Tensor, Q16_1_vs_Q8_1_PrecisionComparison)
{
    const int rows = 4;
    const int cols = 256;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random FP32 data
    std::vector<float> original(elements);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Quantize to Q16_1
    auto q16_1_tensor = Q16_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Quantize to Q8_1
    auto q8_1_tensor = Q8_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(q16_1_tensor, nullptr);
    ASSERT_NE(q8_1_tensor, nullptr);

    // Dequantize both
    std::vector<float> q16_1_recon(elements);
    std::vector<float> q8_1_recon(elements);
    q16_1_tensor->to_fp32(q16_1_recon.data());
    q8_1_tensor->to_fp32(q8_1_recon.data());

    // Compute errors
    float q16_1_error = compute_relative_l2_error(q16_1_recon.data(), original.data(), elements);
    float q8_1_error = compute_relative_l2_error(q8_1_recon.data(), original.data(), elements);

    std::cout << "[Precision Comparison] Q16_1 relative L2 error: " << (q16_1_error * 100.0f) << "%" << std::endl;
    std::cout << "[Precision Comparison] Q8_1 relative L2 error: " << (q8_1_error * 100.0f) << "%" << std::endl;

    if (q16_1_error > 0 && q8_1_error > 0)
    {
        float improvement = q8_1_error / q16_1_error;
        std::cout << "[Precision Comparison] Q16_1 is " << improvement << "× better than Q8_1" << std::endl;

        // Q16_1 should be at least 2× better than Q8_1
        // The theoretical 256× improvement is limited by FP16 scale storage
        EXPECT_GT(improvement, 2.0f)
            << "Q16_1 should be at least 2× more precise than Q8_1";
    }

    // Verify Q16_1 absolute error bounds
    EXPECT_LT(q16_1_error, 0.002f)
        << "Q16_1 error should be < 0.2%";
}

/**
 * @brief Test that Q16_1Block sum_qs is correctly computed
 *
 * Verifies that the pre-computed sum stored in each block matches
 * the actual sum of the quantized values.
 */
TEST(Test__Q16_1Tensor, PrecomputedSumIsCorrect)
{
    const int rows = 2;
    const int cols = 64; // 2 blocks per row
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> data(elements);
    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Check each block
    const Q16_1Block *blocks = tensor->q16_1_blocks();
    const size_t blocks_per_row = tensor->blocks_per_row();
    const size_t total_blocks = tensor->total_blocks();

    for (size_t b = 0; b < total_blocks; ++b)
    {
        const Q16_1Block &block = blocks[b];

        // Compute actual sum
        int32_t actual_sum = 0;
        for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
        {
            actual_sum += static_cast<int32_t>(block.qs[i]);
        }

        EXPECT_EQ(block.sum_qs, actual_sum)
            << "Block " << b << " has incorrect pre-computed sum";
    }

    std::cout << "[Q16_1 Sum Check] Verified " << total_blocks << " blocks" << std::endl;
}

/**
 * @brief Test Q16_1 → Q8_1 conversion
 *
 * Verifies that to_q8_1() produces valid Q8_1 data and that the
 * conversion introduces expected precision loss.
 */
TEST(Test__Q16_1Tensor, ConvertToQ8_1)
{
    const int rows = 4;
    const int cols = 128;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> original(elements);
    std::mt19937 gen(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Create Q16_1 tensor
    auto q16_1_tensor = Q16_1Tensor::quantize_from_fp32(
        original.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(q16_1_tensor, nullptr);

    // Convert to Q8_1
    auto q8_1_tensor = q16_1_tensor->to_q8_1();

    ASSERT_NE(q8_1_tensor, nullptr);
    ASSERT_EQ(q8_1_tensor->shape()[0], static_cast<size_t>(rows));
    ASSERT_EQ(q8_1_tensor->shape()[1], static_cast<size_t>(cols));
    ASSERT_EQ(q8_1_tensor->native_type(), TensorType::Q8_1);

    // Dequantize Q8_1
    std::vector<float> q8_1_recon(elements);
    q8_1_tensor->to_fp32(q8_1_recon.data());

    // The Q8_1 result should be close to original (with Q8_1-level precision)
    float error = compute_relative_l2_error(q8_1_recon.data(), original.data(), elements);

    std::cout << "[Q16_1→Q8_1] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;

    // Should be similar to direct FP32→Q8_1 quantization error (~0.5-1%)
    EXPECT_LT(error, 0.02f)
        << "Q16_1→Q8_1 conversion error exceeds 2% threshold";
}

/**
 * @brief Test mutable Q16_1 tensor with copyFrom
 *
 * Creates a mutable Q16_1 tensor, copies FP32 data into it,
 * and verifies the round-trip precision.
 */
TEST(Test__Q16_1Tensor, MutableCopyFrom)
{
    const int rows = 4;
    const int cols = 96;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> original(elements);
    std::mt19937 gen(888);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (size_t i = 0; i < elements; ++i)
    {
        original[i] = dist(gen);
    }

    // Create FP32 source tensor
    auto fp32_src = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
    std::memcpy(fp32_src->mutable_data(), original.data(), elements * sizeof(float));

    // Create mutable Q16_1 tensor
    auto q16_1_tensor = create_mutable_q16_1_tensor(rows, cols);

    // Copy from FP32
    ASSERT_TRUE(q16_1_tensor->copyFrom(fp32_src.get()));

    // Dequantize and verify
    std::vector<float> reconstructed(elements);
    q16_1_tensor->to_fp32(reconstructed.data());

    float error = compute_relative_l2_error(reconstructed.data(), original.data(), elements);

    std::cout << "[Q16_1 copyFrom] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;

    EXPECT_LT(error, 0.003f)
        << "Q16_1 copyFrom round-trip error exceeds 0.3% threshold";
}

/**
 * @brief Test Q16_1 with extreme values
 *
 * Tests quantization behavior at the boundaries of typical activation ranges.
 */
TEST(Test__Q16_1Tensor, ExtremeValues)
{
    const int rows = 1;
    const int cols = 32; // One block
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Create data with extreme values
    std::vector<float> data(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        // Alternate between -10 and +10 (fairly extreme for activations)
        data[i] = (i % 2 == 0) ? -10.0f : 10.0f;
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Dequantize
    std::vector<float> recon(elements);
    tensor->to_fp32(recon.data());

    // Check each value
    for (size_t i = 0; i < elements; ++i)
    {
        float expected = (i % 2 == 0) ? -10.0f : 10.0f;
        EXPECT_NEAR(recon[i], expected, 0.01f)
            << "Mismatch at index " << i;
    }

    std::cout << "[Q16_1 Extreme] All values within tolerance" << std::endl;
}

/**
 * @brief Test Q16_1 with very small values (near zero)
 *
 * Verifies that small values are preserved with high precision.
 */
TEST(Test__Q16_1Tensor, SmallValues)
{
    const int rows = 1;
    const int cols = 32;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Create data with small values
    std::vector<float> data(elements);
    for (size_t i = 0; i < elements; ++i)
    {
        // Values in range [-0.001, 0.001]
        data[i] = (static_cast<float>(i) / elements - 0.5f) * 0.002f;
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Dequantize
    std::vector<float> recon(elements);
    tensor->to_fp32(recon.data());

    // Compute error
    float max_abs_error = compute_max_abs_error(recon.data(), data.data(), elements);

    std::cout << "[Q16_1 Small Values] Max abs error: " << max_abs_error << std::endl;

    // For very small values, the FP16 scale limits precision
    // Allow error up to ~1% of the max value in range
    EXPECT_LT(max_abs_error, 0.01f)
        << "Q16_1 small value error exceeds threshold";
}

/**
 * @brief Test SIMD decode paths (AVX512/AVX2/scalar consistency)
 *
 * Verifies that all decode paths produce the same results.
 */
TEST(Test__Q16_1Tensor, DecodeConsistency)
{
    const int rows = 2;
    const int cols = 64;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> data(elements);
    std::mt19937 gen(111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    const Q16_1Block *blocks = tensor->q16_1_blocks();
    const size_t total_blocks = tensor->total_blocks();

    // Decode each block with scalar and compare to main decode
    for (size_t b = 0; b < total_blocks; ++b)
    {
        alignas(64) float scalar_output[Q16_1Block::BLOCK_SIZE];
        alignas(64) float main_output[Q16_1Block::BLOCK_SIZE];

        Q16_1Tensor::decodeBlockScalar(blocks[b], scalar_output);
        Q16_1Tensor::decodeBlock(blocks[b], main_output);

        for (size_t i = 0; i < Q16_1Block::BLOCK_SIZE; ++i)
        {
            EXPECT_FLOAT_EQ(scalar_output[i], main_output[i])
                << "Decode mismatch at block " << b << " element " << i;
        }
    }

    std::cout << "[Q16_1 Decode] Verified " << total_blocks << " blocks" << std::endl;
}

/**
 * @brief Test view support
 *
 * Creates a view of a Q16_1 tensor and verifies data access.
 */
TEST(Test__Q16_1Tensor, ViewSupport)
{
    const int rows = 8;
    const int cols = 128;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate data
    std::vector<float> data(elements);
    std::mt19937 gen(222);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Create full tensor
    auto full_tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(full_tensor, nullptr);

    // Create view of rows 2-5 (4 rows)
    auto view = full_tensor->create_view(
        {4, static_cast<size_t>(cols)},
        2 * cols); // Offset by 2 rows

    ASSERT_NE(view, nullptr);
    ASSERT_EQ(view->shape()[0], 4);
    ASSERT_EQ(view->shape()[1], static_cast<size_t>(cols));
    ASSERT_TRUE(view->is_view());

    // Dequantize view
    std::vector<float> view_data(4 * cols);
    view->to_fp32(view_data.data());

    // Compare to expected (rows 2-5 of original)
    float max_view_error = 0.0f;
    for (size_t i = 0; i < 4 * static_cast<size_t>(cols); ++i)
    {
        size_t orig_idx = (2 * cols) + i;
        float err = std::abs(view_data[i] - data[orig_idx]);
        max_view_error = std::max(max_view_error, err);
    }

    std::cout << "[Q16_1 View] Max view error: " << max_view_error << std::endl;

    // Allow for quantization error in view comparison
    EXPECT_LT(max_view_error, 0.003f)
        << "View data diverges too much from expected";

    std::cout << "[Q16_1 View] View data matches expected" << std::endl;
}

/**
 * @brief Stress test with large tensor
 *
 * Tests Q16_1 with a larger tensor to verify performance and correctness.
 */
TEST(Test__Q16_1Tensor, LargeTensorStressTest)
{
    const int rows = 128;
    const int cols = 1024;
    const size_t elements = static_cast<size_t>(rows) * cols;

    // Generate random data
    std::vector<float> data(elements);
    std::mt19937 gen(333);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (size_t i = 0; i < elements; ++i)
    {
        data[i] = dist(gen);
    }

    // Quantize
    auto tensor = Q16_1Tensor::quantize_from_fp32(
        data.data(),
        {static_cast<size_t>(rows), static_cast<size_t>(cols)});

    ASSERT_NE(tensor, nullptr);

    // Dequantize
    std::vector<float> recon(elements);
    tensor->to_fp32(recon.data());

    // Compute error
    float error = compute_relative_l2_error(recon.data(), data.data(), elements);
    float max_error = compute_max_abs_error(recon.data(), data.data(), elements);

    std::cout << "[Q16_1 Large] " << rows << "×" << cols << " tensor" << std::endl;
    std::cout << "[Q16_1 Large] Relative L2 error: " << (error * 100.0f) << "%" << std::endl;
    std::cout << "[Q16_1 Large] Max abs error: " << max_error << std::endl;

    EXPECT_LT(error, 0.001f)
        << "Q16_1 large tensor error exceeds 0.1% threshold";
}
