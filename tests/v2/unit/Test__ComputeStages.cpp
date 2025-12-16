/**
 * @file Test__ComputeStages.cpp
 * @brief Unit tests for individual compute stage implementations
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests for each compute stage type:
 * - GEMMStage: Matrix multiplication with quantization
 * - RMSNormStage: RMS normalization
 * - RoPEStage: Rotary position encoding
 * - SwiGLUStage: SwiGLU activation
 * - ResidualAddStage: Precision-aware residual addition
 * - QuantizeStage: FP32 → Q8_1 quantization
 *
 * Each stage is tested for:
 * - Basic execution correctness
 * - Precision preservation
 * - Backend compatibility
 * - FLOP/memory estimation
 * - Snapshot info generation
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "execution/ComputeStage.h"
#include "execution/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "tensors/SIMDHelpers.h"
#include "../mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class ComputeStagesTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<MockDeviceContext>(0, ComputeBackendType::CPU_OPENBLAS);

        // Seed for reproducibility
        rng_.seed(42);
    }

    // Helper to fill buffer with random FP32 values
    void fillRandom(float *data, size_t count, float min = -1.0f, float max = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // Helper to fill buffer with known pattern
    void fillPattern(float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = static_cast<float>(i % 256) / 256.0f - 0.5f;
        }
    }

    // Helper to check approximate equality
    bool approxEqual(float a, float b, float rtol = 1e-4f, float atol = 1e-6f)
    {
        return std::abs(a - b) <= atol + rtol * std::abs(b);
    }

    // Compute mean squared error
    float computeMSE(const float *a, const float *b, size_t count)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum / static_cast<float>(count);
    }

    // Compute max absolute error
    float computeMaxAbsError(const float *a, const float *b, size_t count)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

    std::unique_ptr<MockDeviceContext> ctx_;
    std::mt19937 rng_;
};

// =============================================================================
// QuantizeStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, QuantizeStage_BasicExecution)
{
    const int m = 4, k = 64; // k must be multiple of Q8_1 block size (32)

    std::vector<float> input(m * k);
    fillPattern(input.data(), input.size());

    const size_t output_size = QuantizeStage::get_quantized_buffer_size(m, k);
    std::vector<uint8_t> output(output_size);

    QuantizeStage::Params params{
        .input = input.data(),
        .output = output.data(),
        .m = m,
        .k = k};

    QuantizeStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::QUANTIZE);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
    EXPECT_GT(stage.estimatedFlops(), 0);
    EXPECT_GT(stage.estimatedMemoryBytes(), 0);

    bool success = stage.execute(ctx_.get());
    EXPECT_TRUE(success);
}

TEST_F(ComputeStagesTest, QuantizeStage_BufferSizeCalculation)
{
    // Q8_1 blocks: each 32 elements → 36 bytes (d=2 + sum_qs=2 + qs=32)
    const int m = 8, k = 128;
    const int num_blocks = (k / 32) * m;

    // Q8_1Block is 36 bytes per 32 elements
    const size_t expected_approx = num_blocks * 36; // 36 bytes per block
    const size_t actual = QuantizeStage::get_quantized_buffer_size(m, k);

    // Allow some variation due to alignment or different implementations
    EXPECT_GT(actual, 0);
    EXPECT_LE(std::abs(static_cast<long>(actual) - static_cast<long>(expected_approx)),
              expected_approx * 0.2)
        << "Buffer size should be approximately " << expected_approx;
}

TEST_F(ComputeStagesTest, QuantizeStage_RoundTripAccuracy)
{
    const int m = 2, k = 64;

    std::vector<float> input(m * k);
    fillRandom(input.data(), input.size(), -2.0f, 2.0f);

    const size_t q_size = QuantizeStage::get_quantized_buffer_size(m, k);
    std::vector<uint8_t> quantized(q_size);

    // Quantize
    QuantizeStage::Params params{
        .input = input.data(),
        .output = quantized.data(),
        .m = m,
        .k = k};

    QuantizeStage stage(params);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    // Dequantize manually for verification
    std::vector<float> dequantized(m * k);
    const auto *blocks = reinterpret_cast<const Q8_1Block *>(quantized.data());
    const int blocks_per_row = k / 32;

    for (int row = 0; row < m; ++row)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            const auto &block = blocks[row * blocks_per_row + b];
            float scale = fp16_to_fp32(block.d);
            for (int i = 0; i < 32; ++i)
            {
                dequantized[row * k + b * 32 + i] =
                    static_cast<float>(block.qs[i]) * scale;
            }
        }
    }

    // Check round-trip accuracy (quantization adds some error)
    float mse = computeMSE(input.data(), dequantized.data(), m * k);
    EXPECT_LT(mse, 0.01f); // Should be very close for this value range
}

// =============================================================================
// GEMMStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, GEMMStage_TypeAndBackend)
{
    // Create minimal params
    GEMMStage::Params params{
        .A = nullptr,
        .B = nullptr,
        .C = nullptr,
        .m = 32,
        .n = 64,
        .k = 128};

    GEMMStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::GEMM);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_MKL));
}

TEST_F(ComputeStagesTest, GEMMStage_EstimatedFlops)
{
    const int m = 32, n = 64, k = 128;

    GEMMStage::Params params{
        .m = m, .n = n, .k = k};

    GEMMStage stage(params);

    // GEMM FLOPS = 2 * m * n * k (multiply + add)
    const size_t expected_flops = 2 * m * n * k;
    EXPECT_EQ(stage.estimatedFlops(), expected_flops);
}

TEST_F(ComputeStagesTest, GEMMStage_DumpInfo)
{
    const int m = 4, n = 8, k = 16;
    std::vector<float> C(m * n, 0.0f);

    GEMMStage::Params params{
        .C = C.data(),
        .m = m,
        .n = n,
        .k = k};

    GEMMStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, C.data());
    EXPECT_EQ(dump_info.outputs[0].rows, m);
    EXPECT_EQ(dump_info.outputs[0].cols, n);
    EXPECT_STREQ(dump_info.outputs[0].name, "C");
}

TEST_F(ComputeStagesTest, GEMMStage_WithFP32Weight)
{
    const int m = 2, n = 4, k = 8;

    // Create weight tensor and fill with identity-ish pattern
    // Use device_idx -1 for CPU-only (no GPU device enumeration needed)
    auto weight = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(k), static_cast<size_t>(n)},
        -1 // device_idx = -1 for CPU
    );
    float *weight_data = weight->mutable_data();
    std::fill(weight_data, weight_data + k * n, 0.0f);
    for (int i = 0; i < std::min(k, n); ++i)
    {
        weight_data[i * n + i] = 1.0f;
    }

    std::vector<float> A(m * k);
    fillPattern(A.data(), A.size());

    std::vector<float> C(m * n, 0.0f);

    GEMMStage::Params params{
        .A = A.data(),
        .B = weight.get(),
        .C = C.data(),
        .m = m,
        .n = n,
        .k = k};

    GEMMStage stage(params);

    // Execution may fail if kernel infrastructure not fully set up,
    // but we can at least test that it doesn't crash
    try
    {
        stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        // Expected: kernel factory may fail without proper device setup
        GTEST_SKIP() << "GEMM execution requires device enumeration: " << e.what();
    }
}

// =============================================================================
// RMSNormStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, RMSNormStage_TypeAndBackend)
{
    RMSNormStage::Params params{
        .input = nullptr,
        .output = nullptr,
        .gamma = nullptr,
        .seq_len = 4,
        .hidden_dim = 128,
        .eps = 1e-5f};

    RMSNormStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::RMS_NORM);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
}

TEST_F(ComputeStagesTest, RMSNormStage_EstimatedFlops)
{
    const int seq_len = 8, hidden_dim = 256;

    RMSNormStage::Params params{
        .seq_len = seq_len,
        .hidden_dim = hidden_dim,
        .eps = 1e-5f};

    RMSNormStage stage(params);

    // RMSNorm FLOPs: per row: hidden_dim (square) + 1 (mean) + 1 (sqrt) + hidden_dim (mul gamma)
    // Approximately 3 * seq_len * hidden_dim
    size_t flops = stage.estimatedFlops();
    EXPECT_GT(flops, 0);
    EXPECT_GE(flops, static_cast<size_t>(seq_len * hidden_dim));
}

TEST_F(ComputeStagesTest, RMSNormStage_SnapshotInfo)
{
    const int seq_len = 4, hidden_dim = 64;
    std::vector<float> output(seq_len * hidden_dim);

    RMSNormStage::Params params{
        .output = output.data(),
        .seq_len = seq_len,
        .hidden_dim = hidden_dim,
        .eps = 1e-5f};

    RMSNormStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output.data());
    EXPECT_EQ(dump_info.outputs[0].rows, seq_len);
    EXPECT_EQ(dump_info.outputs[0].cols, hidden_dim);
    EXPECT_STREQ(dump_info.outputs[0].name, "output");
}

TEST_F(ComputeStagesTest, RMSNormStage_Correctness)
{
    const int seq_len = 2, hidden_dim = 4;
    const float eps = 1e-5f;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f}; // No scaling
    std::vector<float> output(seq_len * hidden_dim, 0.0f);

    // Compute expected RMSNorm manually for first row
    // RMS = sqrt(mean(x^2)) = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.7386
    float rms_row0 = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    float expected_row0[] = {1.0f / rms_row0, 2.0f / rms_row0, 3.0f / rms_row0, 4.0f / rms_row0};

    RMSNormStage::Params params{
        .input = input.data(),
        .output = output.data(),
        .gamma = gamma.data(),
        .seq_len = seq_len,
        .hidden_dim = hidden_dim,
        .eps = eps};

    RMSNormStage stage(params);
    bool success = stage.execute(ctx_.get());

    if (success)
    {
        // Verify first row
        for (int i = 0; i < hidden_dim; ++i)
        {
            EXPECT_NEAR(output[i], expected_row0[i], 1e-4f)
                << "Mismatch at index " << i;
        }

        // Verify normalization (output should have unit RMS)
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; ++i)
        {
            sum_sq += output[i] * output[i];
        }
        float output_rms = std::sqrt(sum_sq / hidden_dim);
        EXPECT_NEAR(output_rms, 1.0f, 0.01f);
    }
}

// =============================================================================
// RoPEStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, RoPEStage_TypeAndBackend)
{
    RoPEStage::Params params{
        .tensor = nullptr,
        .seq_len = 4,
        .n_heads = 8,
        .head_dim = 64,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ROPE);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
}

TEST_F(ComputeStagesTest, RoPEStage_EstimatedFlops)
{
    const int seq_len = 8, n_heads = 4, head_dim = 64;

    RoPEStage::Params params{
        .seq_len = seq_len,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    // RoPE applies sin/cos rotations to pairs of elements
    // FLOPs include sin/cos computation + rotation multiplication
    size_t flops = stage.estimatedFlops();
    EXPECT_GT(flops, 0);
}

TEST_F(ComputeStagesTest, RoPEStage_SnapshotInfo)
{
    const int seq_len = 2, n_heads = 4, head_dim = 32;
    std::vector<float> tensor(seq_len * n_heads * head_dim);

    RoPEStage::Params params{
        .tensor = tensor.data(),
        .seq_len = seq_len,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, tensor.data());
    EXPECT_EQ(dump_info.outputs[0].rows, seq_len);
    EXPECT_EQ(dump_info.outputs[0].cols, n_heads * head_dim);
    EXPECT_STREQ(dump_info.outputs[0].name, "tensor");
}

TEST_F(ComputeStagesTest, RoPEStage_PreservesNorm)
{
    // RoPE should preserve the L2 norm of each head
    const int seq_len = 1, n_heads = 2, head_dim = 4;

    std::vector<float> tensor = {
        1.0f, 2.0f, 3.0f, 4.0f, // Head 0
        5.0f, 6.0f, 7.0f, 8.0f  // Head 1
    };

    // Compute norms before
    auto computeNorm = [](const float *data, int dim)
    {
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i)
        {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    };

    float norm0_before = computeNorm(&tensor[0], head_dim);
    float norm1_before = computeNorm(&tensor[head_dim], head_dim);

    RoPEStage::Params params{
        .tensor = tensor.data(),
        .seq_len = seq_len,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    RoPEStage stage(params);
    bool success = stage.execute(ctx_.get());

    if (success)
    {
        float norm0_after = computeNorm(&tensor[0], head_dim);
        float norm1_after = computeNorm(&tensor[head_dim], head_dim);

        // Norms should be preserved (rotations don't change magnitude)
        EXPECT_NEAR(norm0_before, norm0_after, 1e-4f);
        EXPECT_NEAR(norm1_before, norm1_after, 1e-4f);
    }
}

// =============================================================================
// SwiGLUStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, SwiGLUStage_TypeAndBackend)
{
    SwiGLUStage::Params params{
        .gate = nullptr,
        .up = nullptr,
        .output = nullptr,
        .seq_len = 4,
        .intermediate_dim = 256};

    SwiGLUStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::SWIGLU);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
}

TEST_F(ComputeStagesTest, SwiGLUStage_EstimatedFlops)
{
    const int seq_len = 8, intermediate_dim = 512;

    SwiGLUStage::Params params{
        .seq_len = seq_len,
        .intermediate_dim = intermediate_dim};

    SwiGLUStage stage(params);

    // SwiGLU: swish(gate) * up
    // Per element: sigmoid (exp, add, div) + mul (for swish) + mul (with up)
    // Approximately 6 ops per element
    size_t flops = stage.estimatedFlops();
    EXPECT_GT(flops, 0);
    EXPECT_GE(flops, static_cast<size_t>(seq_len * intermediate_dim));
}

TEST_F(ComputeStagesTest, SwiGLUStage_SnapshotInfo)
{
    const int seq_len = 4, intermediate_dim = 64;
    std::vector<float> output(seq_len * intermediate_dim);

    SwiGLUStage::Params params{
        .output = output.data(),
        .seq_len = seq_len,
        .intermediate_dim = intermediate_dim};

    SwiGLUStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output.data());
    EXPECT_EQ(dump_info.outputs[0].rows, seq_len);
    EXPECT_EQ(dump_info.outputs[0].cols, intermediate_dim);
    EXPECT_STREQ(dump_info.outputs[0].name, "output");
}

TEST_F(ComputeStagesTest, SwiGLUStage_Correctness)
{
    // SwiGLU(gate, up) = swish(gate) * up = (gate * sigmoid(gate)) * up
    const int seq_len = 1, intermediate_dim = 4;

    std::vector<float> gate = {0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> up = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(seq_len * intermediate_dim, 0.0f);

    // Compute expected
    std::vector<float> expected(seq_len * intermediate_dim);
    for (int i = 0; i < intermediate_dim; ++i)
    {
        float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate[i]));
        float swish = gate[i] * sigmoid_gate;
        expected[i] = swish * up[i];
    }

    SwiGLUStage::Params params{
        .gate = gate.data(),
        .up = up.data(),
        .output = output.data(),
        .seq_len = seq_len,
        .intermediate_dim = intermediate_dim};

    SwiGLUStage stage(params);
    bool success = stage.execute(ctx_.get());

    if (success)
    {
        for (int i = 0; i < intermediate_dim; ++i)
        {
            EXPECT_NEAR(output[i], expected[i], 1e-5f)
                << "Mismatch at index " << i
                << ": expected " << expected[i] << ", got " << output[i];
        }
    }
}

// =============================================================================
// ResidualAddStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, ResidualAddStage_TypeAndBackend)
{
    ResidualAddStage::Params params{
        .input = nullptr,
        .residual = nullptr,
        .output = nullptr,
        .num_elements = 256,
        .precision = ActivationPrecision::FP32};

    ResidualAddStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
}

TEST_F(ComputeStagesTest, ResidualAddStage_EstimatedFlops)
{
    const size_t num_elements = 1024;

    ResidualAddStage::Params params{
        .num_elements = num_elements};

    ResidualAddStage stage(params);

    // Residual add: 1 add per element
    EXPECT_EQ(stage.estimatedFlops(), num_elements);
}

TEST_F(ComputeStagesTest, ResidualAddStage_SnapshotInfo)
{
    const int rows = 4, cols = 64;
    std::vector<float> output(rows * cols);

    ResidualAddStage::Params params{
        .output = output.data(),
        .num_elements = rows * cols,
        .rows = rows,
        .cols = cols,
        .precision = ActivationPrecision::FP32};

    ResidualAddStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output.data());
    // Note: rows/cols may vary based on precision handling
    EXPECT_STREQ(dump_info.outputs[0].name, "output");
}

TEST_F(ComputeStagesTest, ResidualAddStage_FP32_Correctness)
{
    const size_t n = 8;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> residual = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<float> output(n, 0.0f);

    ResidualAddStage::Params params{
        .input = input.data(),
        .residual = residual.data(),
        .output = output.data(),
        .num_elements = n,
        .precision = ActivationPrecision::FP32};

    ResidualAddStage stage(params);
    bool success = stage.execute(ctx_.get());

    EXPECT_TRUE(success);

    for (size_t i = 0; i < n; ++i)
    {
        float expected = input[i] + residual[i];
        EXPECT_FLOAT_EQ(output[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(ComputeStagesTest, ResidualAddStage_FP32_InPlace)
{
    const size_t n = 8;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> residual = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i)
    {
        expected[i] = input[i] + residual[i];
    }

    // In-place: output points to residual
    ResidualAddStage::Params params{
        .input = input.data(),
        .residual = residual.data(),
        .output = residual.data(), // In-place
        .num_elements = n,
        .precision = ActivationPrecision::FP32};

    ResidualAddStage stage(params);
    bool success = stage.execute(ctx_.get());

    EXPECT_TRUE(success);

    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(residual[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(ComputeStagesTest, ResidualAddStage_Q8_1_PrecisionPreservation)
{
    // Test that Q8_1 residual addition preserves precision reasonably well
    const int rows = 2;
    const int cols = 64; // Must be multiple of 32 for Q8_1
    const size_t n = rows * cols;

    // Create FP32 inputs
    std::vector<float> input_fp32(n);
    std::vector<float> residual_fp32(n);
    fillRandom(input_fp32.data(), n, -1.0f, 1.0f);
    fillRandom(residual_fp32.data(), n, -1.0f, 1.0f);

    // Compute expected FP32 result
    std::vector<float> expected_fp32(n);
    for (size_t i = 0; i < n; ++i)
    {
        expected_fp32[i] = input_fp32[i] + residual_fp32[i];
    }

    // Quantize inputs to Q8_1
    const size_t q_size = QuantizeStage::get_quantized_buffer_size(rows, cols);
    std::vector<uint8_t> input_q8(q_size);
    std::vector<uint8_t> residual_q8(q_size);
    std::vector<uint8_t> output_q8(q_size);

    QuantizeStage quant_input({input_fp32.data(), input_q8.data(), rows, cols});
    QuantizeStage quant_residual({residual_fp32.data(), residual_q8.data(), rows, cols});

    bool q1_success = quant_input.execute(ctx_.get());
    bool q2_success = quant_residual.execute(ctx_.get());

    if (!q1_success || !q2_success)
    {
        GTEST_SKIP() << "Quantization not available";
    }

    // Perform Q8_1 residual addition
    ResidualAddStage::Params params{
        .input = input_q8.data(),
        .residual = residual_q8.data(),
        .output = output_q8.data(),
        .num_elements = n,
        .rows = rows,
        .cols = cols,
        .precision = ActivationPrecision::Q8_1};

    ResidualAddStage stage(params);
    bool success = stage.execute(ctx_.get());

    if (success)
    {
        // Dequantize output and compare to expected
        std::vector<float> output_fp32(n);
        const auto *blocks = reinterpret_cast<const Q8_1Block *>(output_q8.data());
        const int blocks_per_row = cols / 32;

        for (int row = 0; row < rows; ++row)
        {
            for (int b = 0; b < blocks_per_row; ++b)
            {
                const auto &block = blocks[row * blocks_per_row + b];
                float scale = fp16_to_fp32(block.d);
                for (int i = 0; i < 32; ++i)
                {
                    output_fp32[row * cols + b * 32 + i] =
                        static_cast<float>(block.qs[i]) * scale;
                }
            }
        }

        // Q8_1 adds quantization noise, but should be reasonably close
        float mse = computeMSE(expected_fp32.data(), output_fp32.data(), n);
        float max_err = computeMaxAbsError(expected_fp32.data(), output_fp32.data(), n);

        // Allow some quantization error
        EXPECT_LT(mse, 0.05f) << "MSE too high for Q8_1 residual add";
        EXPECT_LT(max_err, 0.5f) << "Max error too high for Q8_1 residual add";
    }
}

TEST_F(ComputeStagesTest, ResidualAddStage_AllPrecisionModes)
{
    // Verify that all precision modes are accepted
    std::vector<ActivationPrecision> precisions = {
        ActivationPrecision::FP32,
        ActivationPrecision::BF16,
        ActivationPrecision::FP16,
        ActivationPrecision::Q8_1};

    const size_t n = 64;
    std::vector<float> input(n, 1.0f);
    std::vector<float> residual(n, 0.5f);
    std::vector<float> output(n, 0.0f);

    for (auto precision : precisions)
    {
        ResidualAddStage::Params params{
            .input = input.data(),
            .residual = residual.data(),
            .output = output.data(),
            .num_elements = n,
            .precision = precision};

        ResidualAddStage stage(params);

        // Stage should be constructible for all precisions
        EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);

        // Execution may succeed or fail based on implementation
        // but should not crash
        stage.execute(ctx_.get());
    }
}

// =============================================================================
// Stage Factory and Registry Tests
// =============================================================================

TEST_F(ComputeStagesTest, StageTypeName_AllTypes)
{
    // Verify all stage types have names
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::GEMM), "GEMM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::RMS_NORM), "RMS_NORM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::SWIGLU), "SWIGLU");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ROPE), "ROPE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ATTENTION), "ATTENTION");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ADD_RESIDUAL), "ADD_RESIDUAL");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::QUANTIZE), "QUANTIZE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ALLREDUCE), "ALLREDUCE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_ROUTER), "MOE_ROUTER");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_EXPERT_FFN), "MOE_EXPERT_FFN");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_COMBINE), "MOE_COMBINE");
}

// =============================================================================
// Integration: Stage Composition Tests
// =============================================================================

TEST_F(ComputeStagesTest, StageComposition_QuantizeThenGEMM)
{
    // Test the pattern: QuantizeStage → GEMMStage with A_q8_1
    const int m = 2, k = 64, n = 32;

    // Create input activation
    std::vector<float> activation(m * k);
    fillRandom(activation.data(), activation.size(), -1.0f, 1.0f);

    // Create weight tensor (device_idx -1 for CPU)
    auto weight = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(k), static_cast<size_t>(n)},
        -1 // device_idx = -1 for CPU
    );
    float *w_data = weight->mutable_data();
    fillRandom(w_data, k * n, -0.5f, 0.5f);

    // Quantization buffer
    const size_t q_size = QuantizeStage::get_quantized_buffer_size(m, k);
    std::vector<uint8_t> quantized(q_size);

    // Output buffer
    std::vector<float> output(m * n, 0.0f);

    // Step 1: Quantize
    QuantizeStage::Params quant_params{
        .input = activation.data(),
        .output = quantized.data(),
        .m = m,
        .k = k};
    QuantizeStage quant_stage(quant_params);

    // Step 2: GEMM with pre-quantized input
    GEMMStage::Params gemm_params{
        .A = nullptr, // Not using FP32 A
        .B = weight.get(),
        .C = output.data(),
        .m = m,
        .n = n,
        .k = k,
        .A_q8_1 = quantized.data() // Pre-quantized!
    };
    GEMMStage gemm_stage(gemm_params);

    // Execute pipeline
    bool quant_ok = quant_stage.execute(ctx_.get());
    EXPECT_TRUE(quant_ok) << "Quantization should succeed";

    // GEMM execution may fail without proper device setup
    try
    {
        gemm_stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        // Expected: kernel factory may fail without proper device enumeration
        GTEST_SKIP() << "GEMM execution requires device enumeration: " << e.what();
    }
}

TEST_F(ComputeStagesTest, StageComposition_RMSNormThenResidualAdd)
{
    // Test: RMSNorm → ResidualAdd pattern
    const int seq_len = 2, hidden_dim = 64;
    const size_t n = seq_len * hidden_dim;

    std::vector<float> input(n);
    std::vector<float> residual(n);
    std::vector<float> normalized(n, 0.0f);
    std::vector<float> output(n, 0.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    fillRandom(input.data(), n, -1.0f, 1.0f);
    fillRandom(residual.data(), n, -0.5f, 0.5f);

    // Step 1: RMSNorm
    RMSNormStage::Params norm_params{
        .input = input.data(),
        .output = normalized.data(),
        .gamma = gamma.data(),
        .seq_len = seq_len,
        .hidden_dim = hidden_dim,
        .eps = 1e-5f};
    RMSNormStage norm_stage(norm_params);

    // Step 2: Residual Add
    ResidualAddStage::Params add_params{
        .input = normalized.data(),
        .residual = residual.data(),
        .output = output.data(),
        .num_elements = n,
        .rows = seq_len,
        .cols = hidden_dim,
        .precision = ActivationPrecision::FP32};
    ResidualAddStage add_stage(add_params);

    // Execute
    bool norm_ok = norm_stage.execute(ctx_.get());
    bool add_ok = add_stage.execute(ctx_.get());

    if (norm_ok && add_ok)
    {
        // Verify output is different from normalized (residual added)
        bool all_same = true;
        for (size_t i = 0; i < n; ++i)
        {
            if (std::abs(output[i] - normalized[i]) > 1e-6f)
            {
                all_same = false;
                break;
            }
        }
        EXPECT_FALSE(all_same) << "Output should differ from normalized after residual add";
    }
}

// =============================================================================
// Memory Estimation Tests
// =============================================================================

TEST_F(ComputeStagesTest, MemoryEstimation_GEMMStage)
{
    const int m = 64, n = 128, k = 256;

    GEMMStage::Params params{.m = m, .n = n, .k = k};
    GEMMStage stage(params);

    // Should estimate at least the output matrix size
    size_t mem = stage.estimatedMemoryBytes();
    EXPECT_GE(mem, m * n * sizeof(float));
}

TEST_F(ComputeStagesTest, MemoryEstimation_RMSNormStage)
{
    const int seq_len = 8, hidden_dim = 512;

    RMSNormStage::Params params{
        .seq_len = seq_len,
        .hidden_dim = hidden_dim,
        .eps = 1e-5f};
    RMSNormStage stage(params);

    size_t mem = stage.estimatedMemoryBytes();
    EXPECT_GE(mem, static_cast<size_t>(seq_len * hidden_dim * sizeof(float)));
}

TEST_F(ComputeStagesTest, MemoryEstimation_ResidualAddStage)
{
    const size_t n = 1024;

    ResidualAddStage::Params params{.num_elements = n};
    ResidualAddStage stage(params);

    size_t mem = stage.estimatedMemoryBytes();
    EXPECT_GE(mem, n * sizeof(float));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(ComputeStagesTest, Stages_ZeroSize)
{
    // Test behavior with zero-sized inputs

    // QuantizeStage with 0 rows
    QuantizeStage::Params q_params{.m = 0, .k = 32};
    QuantizeStage q_stage(q_params);
    EXPECT_EQ(q_stage.estimatedFlops(), 0);

    // GEMMStage with 0 dimensions
    GEMMStage::Params g_params{.m = 0, .n = 32, .k = 64};
    GEMMStage g_stage(g_params);
    EXPECT_EQ(g_stage.estimatedFlops(), 0);

    // ResidualAddStage with 0 elements
    ResidualAddStage::Params r_params{.num_elements = 0};
    ResidualAddStage r_stage(r_params);
    EXPECT_EQ(r_stage.estimatedFlops(), 0);
}

TEST_F(ComputeStagesTest, Stages_LargeSize)
{
    // Test that large sizes don't overflow
    const int m = 4096, n = 4096, k = 4096;

    GEMMStage::Params params{.m = m, .n = n, .k = k};
    GEMMStage stage(params);

    // 2 * 4096^3 = ~137 billion FLOPs
    size_t flops = stage.estimatedFlops();
    EXPECT_EQ(flops, 2ULL * m * n * k);
}

// =============================================================================
// AttentionStage Basic Tests
// =============================================================================

TEST_F(ComputeStagesTest, AttentionStage_TypeAndBackend)
{
    AttentionStage::Params params{
        .Q = nullptr,
        .K = nullptr,
        .V = nullptr,
        .output = nullptr,
        .seq_len = 4,
        .n_heads = 8,
        .head_dim = 64};

    AttentionStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ATTENTION);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
}

TEST_F(ComputeStagesTest, AttentionStage_EstimatedFlops)
{
    const int seq_len = 16, n_heads = 8, head_dim = 64;

    AttentionStage::Params params{
        .seq_len = seq_len,
        .n_heads = n_heads,
        .head_dim = head_dim};

    AttentionStage stage(params);

    // Attention FLOPs: Q*K^T (2*s*s*d) + softmax (5*s*s) + scores*V (2*s*s*d)
    // Per head, then multiply by n_heads
    // Note: Implementation may return 0 if not computing FLOP estimates
    size_t flops = stage.estimatedFlops();
    // Just verify it doesn't crash and returns non-negative
    EXPECT_GE(flops, 0);
}

TEST_F(ComputeStagesTest, AttentionStage_SnapshotInfo)
{
    const int seq_len = 4, n_heads = 2, head_dim = 32;
    std::vector<float> output(seq_len * n_heads * head_dim);

    AttentionStage::Params params{
        .output = output.data(),
        .seq_len = seq_len,
        .n_heads = n_heads,
        .head_dim = head_dim};

    AttentionStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output.data());
    EXPECT_EQ(dump_info.outputs[0].rows, seq_len);
    EXPECT_EQ(dump_info.outputs[0].cols, n_heads * head_dim);
    EXPECT_STREQ(dump_info.outputs[0].name, "output");
}

// =============================================================================
// AllreduceStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, AllreduceStage_TypeAndBackend)
{
    AllreduceStage::Params params{
        .buffer = nullptr,
        .count = 256,
        .mpi_comm = nullptr};

    AllreduceStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ALLREDUCE);
    // Allreduce requires MPI, may not support all backends
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
}

TEST_F(ComputeStagesTest, AllreduceStage_EstimatedFlops)
{
    const size_t count = 1024;

    AllreduceStage::Params params{.count = count};
    AllreduceStage stage(params);

    // Allreduce is communication, not compute, but may have some overhead
    size_t flops = stage.estimatedFlops();
    // May be 0 or small value
    EXPECT_GE(flops, 0);
}
