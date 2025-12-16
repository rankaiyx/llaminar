/**
 * @file Test__ResidualAddStage_Precision.cpp
 * @brief Unit tests for precision-aware ResidualAddStage
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests verify that ResidualAddStage correctly handles all precision modes:
 * - FP32: Standard float addition
 * - BF16: BF16 addition with conversion
 * - FP16: FP16 addition with conversion
 * - Q8_1: Native Q8_1 block addition using SIMD
 */

#include <gtest/gtest.h>
#include "execution/ComputeStage.h"
#include "execution/DeviceContext.h"
#include "tensors/SIMDHelpers.h"
#include "pipelines/PipelineConfig.h"
#include <vector>
#include <cmath>
#include <random>
#include <cstring>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class ResidualAddStagePrecisionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<CPUDeviceContext>(0, 4);
        ASSERT_NE(ctx_, nullptr);

        // Initialize RNG for test data
        rng_.seed(42);
    }

    // Fill FP32 vector with random values in [-1, 1]
    void fillRandomFP32(std::vector<float> &data)
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : data)
        {
            v = dist(rng_);
        }
    }

    // Fill Q8_1 blocks with random quantized data
    void fillRandomQ8_1(Q8_1Block *blocks, size_t num_blocks)
    {
        std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);
        std::uniform_int_distribution<int> qs_dist(-127, 127);

        for (size_t i = 0; i < num_blocks; ++i)
        {
            float scale = scale_dist(rng_);
            blocks[i].d = simd::fp32_to_fp16(scale);

            int32_t sum = 0;
            for (int j = 0; j < 32; ++j)
            {
                int8_t q = static_cast<int8_t>(qs_dist(rng_));
                blocks[i].qs[j] = q;
                sum += q;
            }
            blocks[i].sum_qs = simd::fp32_to_fp16(static_cast<float>(sum));
        }
    }

    // Dequantize Q8_1 block to FP32 for verification
    void dequantQ8_1(const Q8_1Block &block, float *output)
    {
        float scale = simd::fp16_to_fp32(block.d);
        for (int i = 0; i < 32; ++i)
        {
            output[i] = static_cast<float>(block.qs[i]) * scale;
        }
    }

    std::unique_ptr<CPUDeviceContext> ctx_;
    std::mt19937 rng_;
};

// =============================================================================
// FP32 Tests
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, FP32_BasicAddition)
{
    const size_t n = 128;
    std::vector<float> input(n), residual(n), output(n, 0.0f);

    // Fill with simple test values
    for (size_t i = 0; i < n; ++i)
    {
        input[i] = static_cast<float>(i);
        residual[i] = static_cast<float>(i * 2);
    }

    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = output.data();
    params.num_elements = n;
    params.precision = ActivationPrecision::FP32;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Verify: output[i] = input[i] + residual[i] = i + 2i = 3i
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], static_cast<float>(3 * i))
            << "Mismatch at index " << i;
    }
}

TEST_F(ResidualAddStagePrecisionTest, FP32_InPlace)
{
    const size_t n = 64;
    std::vector<float> input(n), residual(n);

    fillRandomFP32(input);
    fillRandomFP32(residual);

    // Keep copies for verification
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i)
    {
        expected[i] = input[i] + residual[i];
    }

    // In-place: output = residual (same buffer)
    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = residual.data(); // In-place into residual
    params.num_elements = n;
    params.precision = ActivationPrecision::FP32;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(residual[i], expected[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

TEST_F(ResidualAddStagePrecisionTest, FP32_QwenDimensions)
{
    // Test with Qwen2 0.5B dimensions: seq_len=32, d_model=896
    const int rows = 32;
    const int cols = 896;
    const size_t n = rows * cols;

    std::vector<float> input(n), residual(n), output(n);
    fillRandomFP32(input);
    fillRandomFP32(residual);

    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = output.data();
    params.num_elements = n;
    params.rows = rows;
    params.cols = cols;
    params.precision = ActivationPrecision::FP32;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Spot check
    for (size_t i = 0; i < n; i += 100)
    {
        EXPECT_NEAR(output[i], input[i] + residual[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// Q8_1 Tests
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, Q8_1_BasicAddition)
{
    // 4 blocks = 128 elements
    const size_t num_blocks = 4;
    const size_t n = num_blocks * 32;

    std::vector<Q8_1Block> input(num_blocks), residual(num_blocks), output(num_blocks);
    fillRandomQ8_1(input.data(), num_blocks);
    fillRandomQ8_1(residual.data(), num_blocks);

    // Compute expected result in FP32
    std::vector<float> input_fp32(n), residual_fp32(n), expected_fp32(n);
    for (size_t blk = 0; blk < num_blocks; ++blk)
    {
        dequantQ8_1(input[blk], &input_fp32[blk * 32]);
        dequantQ8_1(residual[blk], &residual_fp32[blk * 32]);
    }
    for (size_t i = 0; i < n; ++i)
    {
        expected_fp32[i] = input_fp32[i] + residual_fp32[i];
    }

    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = output.data();
    params.num_elements = n;
    params.precision = ActivationPrecision::Q8_1;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Dequant output and compare
    std::vector<float> output_fp32(n);
    for (size_t blk = 0; blk < num_blocks; ++blk)
    {
        dequantQ8_1(output[blk], &output_fp32[blk * 32]);
    }

    // Q8_1 has quantization error, use relative tolerance
    // Note: Q8_1 addition has inherent quantization noise from the
    // dequant-add-requant cycle. What matters is that the stage
    // produces identical output to direct simd::q8_1_add_q8_1 call
    // (tested in Q8_1_MatchesDirectSIMD test)
    for (size_t i = 0; i < n; ++i)
    {
        float expected = expected_fp32[i];
        float actual = output_fp32[i];
        // Q8_1 can have significant quantization error
        float tolerance = std::max(0.2f, std::abs(expected) * 0.15f); // 15% or 0.2 absolute
        EXPECT_NEAR(actual, expected, tolerance)
            << "Q8_1 mismatch at element " << i
            << ": expected=" << expected << ", actual=" << actual;
    }
}

TEST_F(ResidualAddStagePrecisionTest, Q8_1_QwenDimensions)
{
    // Qwen2 0.5B: seq_len=32, d_model=896
    // 896 / 32 = 28 blocks per row
    const int rows = 32;
    const int cols = 896;
    const size_t blocks_per_row = (cols + 31) / 32;
    const size_t total_blocks = rows * blocks_per_row;
    const size_t n = rows * cols;

    std::vector<Q8_1Block> input(total_blocks), residual(total_blocks), output(total_blocks);
    fillRandomQ8_1(input.data(), total_blocks);
    fillRandomQ8_1(residual.data(), total_blocks);

    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = output.data();
    params.num_elements = n;
    params.rows = rows;
    params.cols = cols;
    params.precision = ActivationPrecision::Q8_1;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Just verify it completes without error for large dimensions
    // Full numerical verification done in smaller test
    SUCCEED() << "Q8_1 ResidualAdd completed for Qwen dimensions";
}

TEST_F(ResidualAddStagePrecisionTest, Q8_1_MatchesDirectSIMD)
{
    // Verify that stage produces identical results to direct simd::q8_1_add_q8_1
    const size_t num_blocks = 8;
    const size_t n = num_blocks * 32;

    std::vector<Q8_1Block> input(num_blocks), residual(num_blocks);
    std::vector<Q8_1Block> stage_output(num_blocks), direct_output(num_blocks);

    fillRandomQ8_1(input.data(), num_blocks);
    fillRandomQ8_1(residual.data(), num_blocks);

    // Execute via stage
    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = stage_output.data();
    params.num_elements = n;
    params.precision = ActivationPrecision::Q8_1;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Execute directly
    simd::q8_1_add_q8_1(input.data(), residual.data(), direct_output.data(), n);

    // Compare block-by-block (should be bit-identical)
    for (size_t blk = 0; blk < num_blocks; ++blk)
    {
        EXPECT_EQ(std::memcmp(&stage_output[blk], &direct_output[blk], sizeof(Q8_1Block)), 0)
            << "Block " << blk << " differs between stage and direct SIMD";
    }
}

// =============================================================================
// BF16 Tests
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, BF16_BasicAddition)
{
    const size_t n = 64;
    std::vector<uint16_t> input(n), residual(n), output(n);

    // Fill with BF16 values
    for (size_t i = 0; i < n; ++i)
    {
        input[i] = simd::fp32_to_bf16(static_cast<float>(i) * 0.1f);
        residual[i] = simd::fp32_to_bf16(static_cast<float>(i) * 0.2f);
    }

    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = output.data();
    params.num_elements = n;
    params.precision = ActivationPrecision::BF16;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Verify (with BF16 precision tolerance)
    for (size_t i = 0; i < n; ++i)
    {
        float expected = static_cast<float>(i) * 0.3f; // 0.1i + 0.2i
        float actual = simd::bf16_to_fp32(output[i]);
        // BF16 has limited precision
        EXPECT_NEAR(actual, expected, 0.01f * std::abs(expected) + 0.01f)
            << "BF16 mismatch at index " << i;
    }
}

// =============================================================================
// FP16 Tests
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, FP16_BasicAddition)
{
    const size_t n = 64;
    std::vector<uint16_t> input(n), residual(n), output(n);

    // Fill with FP16 values
    for (size_t i = 0; i < n; ++i)
    {
        input[i] = simd::fp32_to_fp16(static_cast<float>(i) * 0.1f);
        residual[i] = simd::fp32_to_fp16(static_cast<float>(i) * 0.2f);
    }

    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = output.data();
    params.num_elements = n;
    params.precision = ActivationPrecision::FP16;

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Verify (with FP16 precision tolerance)
    for (size_t i = 0; i < n; ++i)
    {
        float expected = static_cast<float>(i) * 0.3f;
        float actual = simd::fp16_to_fp32(output[i]);
        EXPECT_NEAR(actual, expected, 0.01f * std::abs(expected) + 0.01f)
            << "FP16 mismatch at index " << i;
    }
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, NullContext_Fails)
{
    std::vector<float> data(32, 1.0f);

    ResidualAddStage::Params params;
    params.input = data.data();
    params.residual = data.data();
    params.output = data.data();
    params.num_elements = 32;
    params.precision = ActivationPrecision::FP32;

    ResidualAddStage stage(params);
    EXPECT_FALSE(stage.execute(nullptr));
}

TEST_F(ResidualAddStagePrecisionTest, EstimatedFlops_PrecisionAware)
{
    ResidualAddStage::Params params;
    params.num_elements = 1000;

    // FP32: 1 flop per element
    params.precision = ActivationPrecision::FP32;
    ResidualAddStage fp32_stage(params);
    EXPECT_EQ(fp32_stage.estimatedFlops(), 1000);

    // Q8_1: ~3 flops per element (dequant + add + requant)
    params.precision = ActivationPrecision::Q8_1;
    ResidualAddStage q8_1_stage(params);
    EXPECT_EQ(q8_1_stage.estimatedFlops(), 3000);
}

TEST_F(ResidualAddStagePrecisionTest, EstimatedMemory_PrecisionAware)
{
    ResidualAddStage::Params params;
    params.num_elements = 1000;

    // FP32: 4 bytes per element, 3 buffers
    params.precision = ActivationPrecision::FP32;
    ResidualAddStage fp32_stage(params);
    EXPECT_EQ(fp32_stage.estimatedMemoryBytes(), 1000 * 4 * 3);

    // BF16: 2 bytes per element, 3 buffers
    params.precision = ActivationPrecision::BF16;
    ResidualAddStage bf16_stage(params);
    EXPECT_EQ(bf16_stage.estimatedMemoryBytes(), 1000 * 2 * 3);

    // Q8_1: ~1 byte per element (approximate)
    params.precision = ActivationPrecision::Q8_1;
    ResidualAddStage q8_1_stage(params);
    EXPECT_EQ(q8_1_stage.estimatedMemoryBytes(), 1000 * 1 * 3);
}

// =============================================================================
// Stage Introspection Tests
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, StageType)
{
    ResidualAddStage::Params params;
    params.num_elements = 10;
    params.precision = ActivationPrecision::FP32;

    ResidualAddStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);
    EXPECT_STREQ(stage.name().c_str(), "ADD_RESIDUAL");
}

TEST_F(ResidualAddStagePrecisionTest, SnapshotInfo_FP32)
{
    std::vector<float> output(100);

    ResidualAddStage::Params params;
    params.output = output.data();
    params.num_elements = 100;
    params.rows = 10;
    params.cols = 10;
    params.precision = ActivationPrecision::FP32;

    ResidualAddStage stage(params);
    auto dump_info = stage.getDumpInfo();

    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output.data());
    EXPECT_EQ(dump_info.outputs[0].rows, 10);
    EXPECT_EQ(dump_info.outputs[0].cols, 10);
}

TEST_F(ResidualAddStagePrecisionTest, DumpInfo_Q8_1_ReturnsWithDtype)
{
    // Q8_1 dump info returns the raw pointer with dtype="Q8_1"
    // Consumers can check dtype to know if dequantization is needed
    std::vector<Q8_1Block> output(4);

    ResidualAddStage::Params params;
    params.output = output.data();
    params.num_elements = 128; // 4 blocks * 32
    params.precision = ActivationPrecision::Q8_1;

    ResidualAddStage stage(params);
    auto dump_info = stage.getDumpInfo();

    // Q8_1 outputs are present with correct dtype annotation
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output.data());
    EXPECT_STREQ(dump_info.outputs[0].dtype, "Q8_1");
    // element_size reflects the Q8_1Block size
    EXPECT_EQ(dump_info.outputs[0].element_size, sizeof(Q8_1Block));
}
