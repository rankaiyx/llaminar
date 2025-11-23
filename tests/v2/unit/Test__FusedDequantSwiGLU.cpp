/**
 * @file Test__FusedDequantSwiGLU.cpp
 * @brief Unit tests for FusedDequantSwiGLU kernel (dequantization + SwiGLU activation)
 * @author David Sanftenberg
 * @date 2025-11-23
 *
 * Tests the fused dequant+SwiGLU kernel against separate dequant + SwiGLU operations
 * to validate correctness and measure performance improvement.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

#include "kernels/cpu/fused/FusedDequantSwiGLU.h"
#include "utils/Logger.h"

namespace llaminar2
{
    namespace
    {
        // Helper: Generate random INT32 data
        void fill_random_int32(int32_t *data, size_t count, int range = 10000, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_int_distribution<int32_t> dist(-range, range);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Helper: Generate random FP32 scales
        void fill_random_scales(float *data, size_t count, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(0.001f, 0.1f);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Reference SiLU implementation
        float silu_ref(float x)
        {
            return x / (1.0f + std::exp(-x));
        }

        // Reference implementation: separate dequant + SwiGLU
        void reference_dequant_swiglu(
            const int32_t *gate_int32,
            const int32_t *up_int32,
            float *output,
            const float *activation_scales,
            const float *gate_col_scales,
            const float *up_col_scales,
            int m, int n)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    int idx = i * n + j;

                    // Dequantize gate
                    float gate_fp32 = static_cast<float>(gate_int32[idx]) *
                                      activation_scales[i] * gate_col_scales[j];

                    // Dequantize up
                    float up_fp32 = static_cast<float>(up_int32[idx]) *
                                    activation_scales[i] * up_col_scales[j];

                    // Apply SwiGLU
                    output[idx] = gate_fp32 * silu_ref(up_fp32);
                }
            }
        }

        // Compute relative error
        float compute_relative_error(const float *a, const float *b, size_t count)
        {
            double sum_sq_diff = 0.0;
            double sum_sq_ref = 0.0;

            for (size_t i = 0; i < count; ++i)
            {
                double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                sum_sq_diff += diff * diff;
                sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }

            return std::sqrt(sum_sq_diff / (sum_sq_ref + 1e-12));
        }
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__FusedDequantSwiGLU : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Test dimensions
            m_ = 32;   // sequence length
            n_ = 4864; // intermediate_size

            // Allocate buffers
            gate_int32_.resize(m_ * n_);
            up_int32_.resize(m_ * n_);
            output_fp32_.resize(m_ * n_);
            output_ref_.resize(m_ * n_);
            activation_scales_.resize(m_);
            gate_col_scales_.resize(n_);
            up_col_scales_.resize(n_);

            // Fill with random data
            fill_random_int32(gate_int32_.data(), m_ * n_);
            fill_random_int32(up_int32_.data(), m_ * n_);
            fill_random_scales(activation_scales_.data(), m_);
            fill_random_scales(gate_col_scales_.data(), n_);
            fill_random_scales(up_col_scales_.data(), n_);
        }

        int m_, n_;
        std::vector<int32_t> gate_int32_;
        std::vector<int32_t> up_int32_;
        std::vector<float> output_fp32_;
        std::vector<float> output_ref_;
        std::vector<float> activation_scales_;
        std::vector<float> gate_col_scales_;
        std::vector<float> up_col_scales_;
    };

    // =============================================================================
    // Correctness Tests
    // =============================================================================

    TEST_F(Test__FusedDequantSwiGLU, BasicCorrectness)
    {
        // Execute fused kernel
        FusedDequantSwiGLU kernel;

        bool success = kernel.execute(
            gate_int32_.data(),
            up_int32_.data(),
            output_fp32_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_);

        ASSERT_TRUE(success) << "FusedDequantSwiGLU execution failed";

        // Compute reference
        reference_dequant_swiglu(
            gate_int32_.data(),
            up_int32_.data(),
            output_ref_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_);

        // Compare outputs
        float rel_error = compute_relative_error(
            output_fp32_.data(), output_ref_.data(), m_ * n_);

        EXPECT_LT(rel_error, 1e-5f) << "Relative error too large: " << rel_error;
    }

    TEST_F(Test__FusedDequantSwiGLU, SingleToken)
    {
        m_ = 1;
        gate_int32_.resize(m_ * n_);
        up_int32_.resize(m_ * n_);
        output_fp32_.resize(m_ * n_);
        output_ref_.resize(m_ * n_);
        activation_scales_.resize(m_);

        fill_random_int32(gate_int32_.data(), m_ * n_);
        fill_random_int32(up_int32_.data(), m_ * n_);
        fill_random_scales(activation_scales_.data(), m_);

        FusedDequantSwiGLU kernel;

        bool success = kernel.execute(
            gate_int32_.data(),
            up_int32_.data(),
            output_fp32_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_);

        ASSERT_TRUE(success);

        // Compute reference and compare
        reference_dequant_swiglu(
            gate_int32_.data(),
            up_int32_.data(),
            output_ref_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_);

        float rel_error = compute_relative_error(
            output_fp32_.data(), output_ref_.data(), m_ * n_);

        EXPECT_LT(rel_error, 1e-5f);
    }

    TEST_F(Test__FusedDequantSwiGLU, LargeBatch)
    {
        m_ = 128;
        gate_int32_.resize(m_ * n_);
        up_int32_.resize(m_ * n_);
        output_fp32_.resize(m_ * n_);
        output_ref_.resize(m_ * n_);
        activation_scales_.resize(m_);

        fill_random_int32(gate_int32_.data(), m_ * n_);
        fill_random_int32(up_int32_.data(), m_ * n_);
        fill_random_scales(activation_scales_.data(), m_);

        FusedDequantSwiGLU kernel;

        bool success = kernel.execute(
            gate_int32_.data(),
            up_int32_.data(),
            output_fp32_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_);

        ASSERT_TRUE(success);

        reference_dequant_swiglu(
            gate_int32_.data(),
            up_int32_.data(),
            output_ref_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_);

        float rel_error = compute_relative_error(
            output_fp32_.data(), output_ref_.data(), m_ * n_);

        EXPECT_LT(rel_error, 1e-5f);
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    TEST_F(Test__FusedDequantSwiGLU, NullPointerHandling)
    {
        FusedDequantSwiGLU kernel;

        // Null gate input
        EXPECT_FALSE(kernel.execute(
            nullptr,
            up_int32_.data(),
            output_fp32_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_));

        // Null output
        EXPECT_FALSE(kernel.execute(
            gate_int32_.data(),
            up_int32_.data(),
            nullptr,
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            m_, n_));
    }

    TEST_F(Test__FusedDequantSwiGLU, InvalidDimensions)
    {
        FusedDequantSwiGLU kernel;

        // Zero dimensions
        EXPECT_FALSE(kernel.execute(
            gate_int32_.data(),
            up_int32_.data(),
            output_fp32_.data(),
            activation_scales_.data(),
            gate_col_scales_.data(),
            up_col_scales_.data(),
            0, n_));
    }

    // =============================================================================
    // Kernel Contract Tests
    // =============================================================================

    TEST_F(Test__FusedDequantSwiGLU, KernelContract)
    {
        FusedDequantSwiGLU kernel;
        auto contract = kernel.get_contract();

        EXPECT_EQ(contract.output_format, TensorFormat::FP32);
        EXPECT_TRUE(contract.is_fusable);
        EXPECT_FALSE(contract.supports_inplace);
        EXPECT_TRUE(kernel.supports_fusion());
        EXPECT_EQ(kernel.preferred_fusion_format(), TensorFormat::FP32);
        EXPECT_TRUE(kernel.supports_device(-1)); // CPU
    }

} // namespace llaminar2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
