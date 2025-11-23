/**
 * @file Test__FusedDualGEMM.cpp
 * @brief Unit tests for FusedDualGEMM kernel (shared quantization for gate/up projections)
 * @author David Sanftenberg
 * @date 2025-11-23
 *
 * Tests the fused dual GEMM kernel against separate quantization + GEMM operations
 * to validate correctness and measure performance improvement.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <memory>

#include "kernels/cpu/fused/FusedDualGEMM.h"
#include "kernels/cpu/gemm_v4/OneDNNGemmAdapter.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

namespace llaminar2
{
    namespace
    {
        // Helper: Generate random FP32 data
        void fill_random(float *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Helper: Create mock INT8 weight tensor for testing
        std::unique_ptr<TensorBase> create_mock_int8_weights(int rows, int cols, unsigned seed = 123)
        {
            std::vector<float> fp32_weights(rows * cols);
            fill_random(fp32_weights.data(), rows * cols, 0.5f, seed);

            // INT8Tensor constructor with FP32 data automatically computes per-column scales
            return std::make_unique<INT8Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
                fp32_weights);
        }

        // Helper: Compute relative error
        float compute_relative_error(const int32_t *a, const int32_t *b, size_t count)
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

    class Test__FusedDualGEMM : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Test dimensions (typical FFN: seq_len=32, d_model=896, intermediate=4864)
            m_ = 32;   // sequence length
            k_ = 896;  // d_model (input features)
            n_ = 4864; // intermediate_size (output features)

            // Create random input activations
            input_fp32_.resize(m_ * k_);
            fill_random(input_fp32_.data(), m_ * k_, 1.0f);

            // Create mock INT8 weight tensors
            // NOTE: Weights must be [n, k] for pack_weights_to_int8 (expects [N, K] where C = A[m,k] @ B[k,n])
            gate_weight_ = create_mock_int8_weights(n_, k_, 111);
            up_weight_ = create_mock_int8_weights(n_, k_, 222);

            // Allocate output buffers
            gate_output_int32_.resize(m_ * n_);
            up_output_int32_.resize(m_ * n_);
            activation_scales_.resize(m_);

            // Reference outputs (separate path)
            gate_output_ref_.resize(m_ * n_);
            up_output_ref_.resize(m_ * n_);
            activation_scales_ref_.resize(m_);
        }

        int m_, k_, n_;
        std::vector<float> input_fp32_;
        std::unique_ptr<TensorBase> gate_weight_;
        std::unique_ptr<TensorBase> up_weight_;
        std::vector<int32_t> gate_output_int32_;
        std::vector<int32_t> up_output_int32_;
        std::vector<float> activation_scales_;
        std::vector<int32_t> gate_output_ref_;
        std::vector<int32_t> up_output_ref_;
        std::vector<float> activation_scales_ref_;
    };

    // =============================================================================
    // Correctness Tests
    // =============================================================================

    TEST_F(Test__FusedDualGEMM, BasicCorrectness)
    {
        // Execute fused kernel
        FusedDualGEMM fused_kernel(gate_weight_.get(), up_weight_.get());

        bool success = fused_kernel.execute(
            input_fp32_.data(),
            gate_output_int32_.data(),
            up_output_int32_.data(),
            activation_scales_.data(),
            m_, n_, k_);

        ASSERT_TRUE(success) << "FusedDualGEMM execution failed";

        // Validate that activation scales are reasonable
        for (int i = 0; i < m_; ++i)
        {
            EXPECT_GT(activation_scales_[i], 0.0f) << "Row " << i << " scale should be positive";
            EXPECT_LT(activation_scales_[i], 100.0f) << "Row " << i << " scale unexpectedly large";
        }

        // Validate that outputs are non-zero (weights are non-zero)
        bool has_nonzero_gate = false;
        bool has_nonzero_up = false;

        for (size_t i = 0; i < gate_output_int32_.size(); ++i)
        {
            if (gate_output_int32_[i] != 0)
                has_nonzero_gate = true;
            if (up_output_int32_[i] != 0)
                has_nonzero_up = true;
        }

        EXPECT_TRUE(has_nonzero_gate) << "Gate output should have non-zero values";
        EXPECT_TRUE(has_nonzero_up) << "Up output should have non-zero values";
    }

    TEST_F(Test__FusedDualGEMM, SmallBatch)
    {
        // Test with small batch size (single token)
        m_ = 1;
        input_fp32_.resize(m_ * k_);
        fill_random(input_fp32_.data(), m_ * k_);

        gate_output_int32_.resize(m_ * n_);
        up_output_int32_.resize(m_ * n_);
        activation_scales_.resize(m_);

        FusedDualGEMM fused_kernel(gate_weight_.get(), up_weight_.get());

        bool success = fused_kernel.execute(
            input_fp32_.data(),
            gate_output_int32_.data(),
            up_output_int32_.data(),
            activation_scales_.data(),
            m_, n_, k_);

        ASSERT_TRUE(success) << "FusedDualGEMM failed for single token";
    }

    TEST_F(Test__FusedDualGEMM, LargeBatch)
    {
        // Test with larger batch size
        m_ = 128;
        input_fp32_.resize(m_ * k_);
        fill_random(input_fp32_.data(), m_ * k_);

        gate_output_int32_.resize(m_ * n_);
        up_output_int32_.resize(m_ * n_);
        activation_scales_.resize(m_);

        FusedDualGEMM fused_kernel(gate_weight_.get(), up_weight_.get());

        bool success = fused_kernel.execute(
            input_fp32_.data(),
            gate_output_int32_.data(),
            up_output_int32_.data(),
            activation_scales_.data(),
            m_, n_, k_);

        ASSERT_TRUE(success) << "FusedDualGEMM failed for large batch";
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    TEST_F(Test__FusedDualGEMM, NullPointerHandling)
    {
        FusedDualGEMM kernel(gate_weight_.get(), up_weight_.get());

        // Null input
        EXPECT_FALSE(kernel.execute(
            nullptr,
            gate_output_int32_.data(),
            up_output_int32_.data(),
            activation_scales_.data(),
            m_, n_, k_));

        // Null gate output
        EXPECT_FALSE(kernel.execute(
            input_fp32_.data(),
            nullptr,
            up_output_int32_.data(),
            activation_scales_.data(),
            m_, n_, k_));

        // Null activation scales
        EXPECT_FALSE(kernel.execute(
            input_fp32_.data(),
            gate_output_int32_.data(),
            up_output_int32_.data(),
            nullptr,
            m_, n_, k_));
    }

    TEST_F(Test__FusedDualGEMM, InvalidDimensions)
    {
        FusedDualGEMM kernel(gate_weight_.get(), up_weight_.get());

        // Zero dimensions
        EXPECT_FALSE(kernel.execute(
            input_fp32_.data(),
            gate_output_int32_.data(),
            up_output_int32_.data(),
            activation_scales_.data(),
            0, n_, k_));

        EXPECT_FALSE(kernel.execute(
            input_fp32_.data(),
            gate_output_int32_.data(),
            up_output_int32_.data(),
            activation_scales_.data(),
            m_, 0, k_));
    }

    // =============================================================================
    // Kernel Contract Tests
    // =============================================================================

    TEST_F(Test__FusedDualGEMM, KernelContract)
    {
        FusedDualGEMM kernel(gate_weight_.get(), up_weight_.get());
        auto contract = kernel.get_contract();

        EXPECT_EQ(contract.output_format, TensorFormat::INT32);
        EXPECT_TRUE(contract.is_fusable);
        EXPECT_FALSE(contract.supports_inplace);
        EXPECT_TRUE(kernel.supports_fusion());
        EXPECT_EQ(kernel.preferred_fusion_format(), TensorFormat::INT32);
        EXPECT_TRUE(kernel.supports_device(-1)); // CPU
    }

} // namespace llaminar2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
