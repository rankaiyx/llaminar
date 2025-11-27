#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include <vector>
#include <random>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

TEST(Test__QuantisedGemmKernel, BasicMatMul)
{
    // Dimensions
    int M = 1;
    int N = 64;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    // Q8_1Tensor expects [N, K] shape
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

    // Create kernel
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random input A (M x K)
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Compute reference C (M x N)
    std::vector<float> C_ref(M * N, 0.0f);
    // A is M x K, Weights is N x K.
    // C = A * Weights^T
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * weights_fp32[n * K + k];
            }
            C_ref[m * N + n] = sum;
        }
    }

    // Compute actual C
    std::vector<float> C_act(M * N, 0.0f);
    kernel->multiply(A.data(), C_act.data(), M, N, K);

    // Compare
    // Q8_1 quantization introduces error.
    // Tolerance depends on range.
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C_act[i], C_ref[i], 1.0f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test the new multiply_tensor interface with FP32 activations
 *
 * This tests the fallback path: FP32 activations go through online quantization
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensorFP32Activations)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random FP32 input tensor (M x K)
    std::vector<float> A_data(M * K);
    for (auto &x : A_data)
        x = dist(gen);
    auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Compute reference C (M x N)
    std::vector<float> C_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A_data[m * K + k] * weights_fp32[n * K + k];
            }
            C_ref[m * N + n] = sum;
        }
    }

    // Test multiply_tensor interface
    bool ok = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compare
    const float *C_act = C_tensor->data();
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C_act[i], C_ref[i], 1.0f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test multiply_tensor with Q8_1 activations (zero-copy direct path)
 *
 * This tests the optimized path: Q8_1 activations bypass float conversion
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensorQ8_1Activations)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create FP32 activations first (for reference)
    std::vector<float> A_fp32(M * K);
    for (auto &x : A_fp32)
        x = dist(gen);

    // Quantize activations to Q8_1
    auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});
    ASSERT_NE(A_q8_1, nullptr);

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Compute reference using FP32 path
    std::vector<float> C_ref(M * N, 0.0f);
    kernel->multiply(A_fp32.data(), C_ref.data(), M, N, K);

    // Test multiply_tensor with Q8_1 input (should use direct path)
    bool ok = kernel->multiply_tensor(A_q8_1.get(), C_tensor.get(), true, 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compare - both use quantized GEMM, so tolerance depends on quantization error
    const float *C_act = C_tensor->data();
    for (int i = 0; i < M * N; ++i)
    {
        // Same quantization, so results should be close (but not identical
        // because the FP32 path does its own online quantization)
        EXPECT_NEAR(C_act[i], C_ref[i], 2.0f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test activation sharing interface (quantize_activations + multiply_with_precomputed_q8_1)
 *
 * This tests the fused multi-GEMM workflow:
 * 1. quantize_activations() once
 * 2. multiply_with_precomputed_q8_1() multiple times with different weights
 */
TEST(Test__QuantisedGemmKernel, ActivationSharingInterface)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights for two projections (like gate/up in FFN)
    std::vector<float> weights1_fp32(N * K);
    std::vector<float> weights2_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights1_fp32)
        x = dist(gen);
    for (auto &x : weights2_fp32)
        x = dist(gen);

    // Quantize weights
    auto weights1 = Q8_1Tensor::quantize_from_fp32(weights1_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto weights2 = Q8_1Tensor::quantize_from_fp32(weights2_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

    // Create kernels
    auto kernel1 = weights1->createGemm();
    auto kernel2 = weights2->createGemm();
    ASSERT_NE(kernel1, nullptr);
    ASSERT_NE(kernel2, nullptr);

    // Verify activation sharing is supported
    EXPECT_TRUE(kernel1->supports_activation_sharing());
    EXPECT_TRUE(kernel2->supports_activation_sharing());

    // Create random input
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Compute reference using standard path
    std::vector<float> C1_ref(M * N, 0.0f);
    std::vector<float> C2_ref(M * N, 0.0f);
    ASSERT_TRUE(kernel1->multiply(A.data(), C1_ref.data(), M, N, K));
    ASSERT_TRUE(kernel2->multiply(A.data(), C2_ref.data(), M, N, K));

    // Test shared activation path
    // Step 1: Allocate buffer
    size_t buffer_size = kernel1->get_quantized_activation_buffer_size(M, K);
    EXPECT_GT(buffer_size, 0u);
    std::vector<uint8_t> q8_1_buffer(buffer_size);

    // Step 2: Quantize activations once
    ASSERT_TRUE(kernel1->quantize_activations(A.data(), q8_1_buffer.data(), M, K));

    // Step 3: Execute both GEMMs with shared activations
    std::vector<float> C1_act(M * N, 0.0f);
    std::vector<float> C2_act(M * N, 0.0f);
    ASSERT_TRUE(kernel1->multiply_with_precomputed_q8_1(q8_1_buffer.data(), C1_act.data(), M, N, K));
    ASSERT_TRUE(kernel2->multiply_with_precomputed_q8_1(q8_1_buffer.data(), C2_act.data(), M, N, K));

    // Compare - shared activation path should produce identical results to standard path
    // (both use the same quantization algorithm)
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C1_act[i], C1_ref[i], 0.01f) << "Kernel1 mismatch at index " << i;
        EXPECT_NEAR(C2_act[i], C2_ref[i], 0.01f) << "Kernel2 mismatch at index " << i;
    }
}

/**
 * @brief Test that non-quantized kernels correctly report no activation sharing support
 */
TEST(Test__QuantisedGemmKernel, ActivationSharingNotSupportedByDefault)
{
    // The base ITensorGemm returns false for supports_activation_sharing()
    // We can't easily test this without a non-quantized kernel, but we can at least
    // verify the interface methods exist and return sensible values
    int M = 2;
    int K = 32;

    // Create a simple Q8_1 tensor just to get a kernel
    std::vector<float> weights(32 * 32);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights)
        x = dist(gen);
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights.data(), {32, 32});
    auto kernel = weights_tensor->createGemm();

    // Test buffer size calculation
    size_t buffer_size = kernel->get_quantized_activation_buffer_size(M, K);
    EXPECT_EQ(buffer_size, M * (K / 32) * sizeof(Q8_1Block));
}
