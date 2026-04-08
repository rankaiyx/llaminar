/**
 * @file Test__CPUQuantizedGemmParity.cpp
 * @brief Comprehensive GEMV (M=1) and GEMM (M>1) parity tests for all quantized tensor types.
 *
 * For each quantized format, this test:
 *   1. Creates a random quantized weight tensor
 *   2. Dequantizes to FP32 for reference
 *   3. Runs quantized GEMM/GEMV via createGemm()->multiply_tensor()
 *   4. Runs FP32 reference GEMM via FloatingPointGemmKernel
 *   5. Compares relative L2 error against format-appropriate threshold
 *
 * This covers both the deferred VNNI repacking path (non-superblock formats)
 * and the permanent interleaved path (superblock formats).
 */

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test fixture with shared GEMM helpers
// ============================================================================

class CPUQuantizedGemmParity : public ::testing::Test
{
protected:
    static float computeRelativeL2(const float *a, const float *b, size_t n)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum_sq_diff += diff * diff;
            sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return (sum_sq_ref > 0) ? static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref)) : 0.0f;
    }

    // Runs quantized GEMM and FP32 reference GEMM, returns relative L2 error.
    // weight_tensor must already be created (quantized) and support to_fp32().
    static float runParityTest(TensorBase *weight_tensor, int m, int n, int k)
    {
        // Dequantize to FP32 for reference
        auto fp32_weights = TestTensorFactory::createFP32({static_cast<size_t>(n),
                                                           static_cast<size_t>(k)});
        weight_tensor->to_fp32(fp32_weights->mutable_data());

        // Create random input activations
        auto input = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(m), static_cast<size_t>(k)}, -1.0f, 1.0f, 42);

        // Allocate outputs
        auto output_quant = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(m), static_cast<size_t>(n)});
        auto output_fp32 = TestTensorFactory::createFP32Zeros(
            {static_cast<size_t>(m), static_cast<size_t>(n)});

        // Run quantized path
        auto quantized_gemm = weight_tensor->createGemm();
        EXPECT_NE(quantized_gemm, nullptr);
        if (!quantized_gemm)
            return 1.0f;

        bool q_ok = quantized_gemm->multiply_tensor(
            input.get(), output_quant.get(), m, n, k);
        EXPECT_TRUE(q_ok);
        if (!q_ok)
            return 1.0f;

        // Run FP32 reference
        gemm::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());
        bool f_ok = fp32_gemm.multiply_tensor(
            input.get(), output_fp32.get(), m, n, k);
        EXPECT_TRUE(f_ok);
        if (!f_ok)
            return 1.0f;

        return computeRelativeL2(output_quant->data(), output_fp32->data(),
                                 static_cast<size_t>(m) * n);
    }
};

// ============================================================================
// Helper macro: generates both GEMV (M=1) and GEMM (M=64) tests for a type
// ============================================================================

#define QUANT_PARITY_TESTS(TypeName, FactoryMethod, N, K, Threshold)                \
                                                                                     \
    TEST_F(CPUQuantizedGemmParity, TypeName##_GEMV)                                  \
    {                                                                                \
        auto weights = TestTensorFactory::FactoryMethod(                              \
            {static_cast<size_t>(N), static_cast<size_t>(K)});                       \
        ASSERT_NE(weights, nullptr);                                                 \
                                                                                     \
        float err = runParityTest(weights.get(), 1, N, K);                           \
        std::cout << "[" #TypeName " GEMV] Relative L2 error: "                      \
                  << (err * 100.0f) << "%" << std::endl;                             \
        EXPECT_LT(err, Threshold)                                                    \
            << #TypeName " GEMV error exceeds " << (Threshold * 100.0f) << "% threshold"; \
    }                                                                                \
                                                                                     \
    TEST_F(CPUQuantizedGemmParity, TypeName##_GEMM)                                  \
    {                                                                                \
        auto weights = TestTensorFactory::FactoryMethod(                              \
            {static_cast<size_t>(N), static_cast<size_t>(K)});                       \
        ASSERT_NE(weights, nullptr);                                                 \
                                                                                     \
        float err = runParityTest(weights.get(), 64, N, K);                          \
        std::cout << "[" #TypeName " GEMM] Relative L2 error: "                      \
                  << (err * 100.0f) << "%" << std::endl;                             \
        EXPECT_LT(err, Threshold)                                                    \
            << #TypeName " GEMM error exceeds " << (Threshold * 100.0f) << "% threshold"; \
    }

// ============================================================================
// Per-block formats (32 elements, deferred packing path)
// ============================================================================

// N=512, K=512 — exercises multiple VNNI chunks and blocks_per_row
// 1% threshold for high-bit formats, higher for low-bit

QUANT_PARITY_TESTS(Q8_0,   createQ8_0Random,   512, 512, 0.01f)
QUANT_PARITY_TESTS(Q8_1,   createQ8_1Random,   512, 512, 0.01f)
QUANT_PARITY_TESTS(Q4_0,   createQ4_0Random,   512, 512, 0.01f)
QUANT_PARITY_TESTS(IQ4_NL, createIQ4_NLRandom, 512, 512, 0.01f)
QUANT_PARITY_TESTS(Q4_1,   createQ4_1Random,   512, 512, 0.01f)
QUANT_PARITY_TESTS(Q5_0,   createQ5_0Random,   512, 512, 0.01f)
QUANT_PARITY_TESTS(Q5_1,   createQ5_1Random,   512, 512, 0.01f)

// ============================================================================
// Superblock formats (256 elements, permanent interleaved path)
// ============================================================================

// N=512, K=512 — 2 superblocks per row
// Higher threshold for very low bit formats

QUANT_PARITY_TESTS(Q6_K,  createQ6_KRandom,  512, 512, 0.01f)
QUANT_PARITY_TESTS(Q5_K,  createQ5_KRandom,  512, 512, 0.01f)
QUANT_PARITY_TESTS(Q4_K,  createQ4_KRandom,  512, 512, 0.01f)
QUANT_PARITY_TESTS(Q3_K,  createQ3_KRandom,  512, 512, 0.02f)
QUANT_PARITY_TESTS(Q2_K,  createQ2_KRandom,  512, 512, 0.05f)

// ============================================================================
// IQ formats (256 elements, superblock path)
// Wider thresholds for very-low-bit importance-quantized formats
// ============================================================================

QUANT_PARITY_TESTS(IQ4_XS,  createIQ4_XSRandom,  512, 512, 0.01f)
QUANT_PARITY_TESTS(IQ3_S,   createIQ3_SRandom,   512, 512, 0.05f)
QUANT_PARITY_TESTS(IQ3_XXS, createIQ3_XXSRandom, 512, 512, 0.05f)
QUANT_PARITY_TESTS(IQ2_S,   createIQ2_SRandom,   512, 512, 0.10f)
QUANT_PARITY_TESTS(IQ2_XS,  createIQ2_XSRandom,  512, 512, 0.10f)
QUANT_PARITY_TESTS(IQ2_XXS, createIQ2_XXSRandom, 512, 512, 0.10f)
QUANT_PARITY_TESTS(IQ1_S,   createIQ1_SRandom,   512, 512, 0.15f)
QUANT_PARITY_TESTS(IQ1_M,   createIQ1_MRandom,   512, 512, 0.15f)
