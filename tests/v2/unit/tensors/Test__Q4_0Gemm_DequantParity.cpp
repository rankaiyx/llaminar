/**
 * @file Test__Q4_0Gemm_DequantParity.cpp
 * @brief Unit tests validating Q4_0 fused dequant+GEMM parity with explicit dequant+CBLAS
 * @author David Sanftenberg
 * @date November 7, 2025
 *
 * Tests verify that Q4_0Tensor's fused dequant+GEMM produces numerically equivalent
 * results to the reference path: dequant to FP32 → cblas_sgemm().
 *
 * Key validation scenarios:
 * - Single token (m=1): Inference decode step
 * - Small batch (m=8-32): Typical batch sizes
 * - Medium batch (m=128): Larger batches
 * - Large batch (m=512): Stress testing
 * - Various K dimensions: 896 (Qwen 0.5B), 2048 (Qwen 1.5B), 4096 (Llama 7B)
 * - Various N dimensions: Small (64), Medium (896), Large (4096)
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/utils/DebugEnv.h"
#include "loaders/ModelLoader.h"
#include <cblas.h>
#include <vector>
#include <cmath>
#include <random>
#include <memory>

namespace llaminar2
{
    namespace test
    {
        /**
         * @brief Test fixture for Q4_0 GEMM parity testing
         */
        class Test__Q4_0Gemm_DequantParity : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                // Initialize random number generator with fixed seed for reproducibility
                rng_.seed(42);
            }

            /**
             * @brief Create a random Q4_0 tensor with given shape
             *
             * @param rows Number of rows (N dimension for weight matrix)
             * @param cols Number of columns (K dimension for weight matrix)
             * @return Shared pointer to Q4_0Tensor
             */
            std::shared_ptr<Q4_0Tensor> createRandomQ4_0Tensor(size_t rows, size_t cols)
            {
                // Calculate number of blocks needed
                const size_t blocks_per_row = (cols + Q4_0Block::BLOCK_SIZE - 1) / Q4_0Block::BLOCK_SIZE;
                const size_t total_blocks = rows * blocks_per_row;
                const size_t raw_data_size = total_blocks * sizeof(Q4_0Block);

                std::vector<uint8_t> raw_data(raw_data_size);
                Q4_0Block *blocks = reinterpret_cast<Q4_0Block *>(raw_data.data());

                std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);
                std::uniform_int_distribution<int> nibble_dist(0, 15);

                for (size_t i = 0; i < total_blocks; ++i)
                {
                    // Random scale (convert to FP16)
                    float scale_fp32 = scale_dist(rng_);
                    blocks[i].d = fp32_to_fp16(scale_fp32);

                    // Random nibbles
                    for (size_t j = 0; j < 16; ++j)
                    {
                        uint8_t low = nibble_dist(rng_);
                        uint8_t high = nibble_dist(rng_);
                        blocks[i].qs[j] = (high << 4) | low;
                    }
                }

                return std::make_shared<Q4_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
            }

            /**
             * @brief Create a random FP32 tensor for A matrix (input activations)
             *
             * @param rows Number of rows (M dimension)
             * @param cols Number of columns (K dimension)
             * @return Vector of FP32 values in row-major order
             */
            std::vector<float> createRandomFP32Matrix(size_t rows, size_t cols)
            {
                std::vector<float> data(rows * cols);
                std::normal_distribution<float> dist(0.0f, 1.0f);

                for (size_t i = 0; i < rows * cols; ++i)
                {
                    data[i] = dist(rng_);
                }

                return data;
            }

            /**
             * @brief Dequantize Q4_0 tensor to FP32 (reference implementation)
             *
             * @param q4_tensor Q4_0 tensor to dequantize
             * @return Vector of FP32 values
             */
            std::vector<float> dequantizeToFP32(const Q4_0Tensor *q4_tensor)
            {
                const auto &shape = q4_tensor->shape();
                const size_t rows = shape[0];
                const size_t cols = shape[1];

                std::vector<float> fp32_data(rows * cols);
                q4_tensor->to_fp32(fp32_data.data());

                return fp32_data;
            }

            /**
             * @brief Perform GEMM using CBLAS (reference implementation)
             *
             * C = A @ B^T  (where B is transposed weight matrix)
             *
             * @param A Input activations [m, k]
             * @param B Weight matrix [n, k]
             * @param m Number of rows in A
             * @param n Number of rows in B (columns in output)
             * @param k Number of columns in A and B
             * @return Output matrix C [m, n]
             */
            std::vector<float> gemm_cblas(const float *A, const float *B,
                                          int m, int n, int k)
            {
                std::vector<float> C(m * n, 0.0f);

                // C = A @ B^T
                // A: [m, k] row-major
                // B: [n, k] row-major → B^T: [k, n] column-major
                // C: [m, n] row-major
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans, // A is not transposed
                    CblasTrans,   // B is transposed
                    m,            // rows of A and C
                    n,            // columns of B^T and C
                    k,            // columns of A, rows of B^T
                    1.0f,         // alpha
                    A, k,         // A and lda
                    B, k,         // B and ldb (note: B is [n, k] row-major)
                    0.0f,         // beta
                    C.data(), n   // C and ldc
                );

                return C;
            }

            /**
             * @brief Perform GEMM using Q4_0 fused dequant+GEMM kernel
             *
             * @param A Input activations [m, k]
             * @param B_q4 Q4_0 weight matrix [n, k]
             * @param m Number of rows in A
             * @param n Number of rows in B
             * @param k Number of columns
             * @return Output matrix C [m, n]
             */
            std::vector<float> gemm_fused_q4(const float *A, Q4_0Tensor *B_q4,
                                             int m, int n, int k)
            {
                std::vector<float> C(m * n, 0.0f);

                // Create GEMM kernel for B
                auto gemm_kernel = B_q4->createGemm();

                // Perform fused dequant+GEMM via legacy weight-owned GEMM path.
                // This path has been retired; parity is now validated via
                // activation-tensor-centric tests elsewhere in the suite.

                return C;
            }

            /**
             * @brief Compare two matrices and compute error metrics
             *
             * @param reference Reference matrix (from CBLAS)
             * @param test Test matrix (from fused kernel)
             * @param m Number of rows
             * @param n Number of columns
             * @param tolerance Maximum acceptable relative L2 error
             * @return True if parity check passed
             */
            struct ComparisonResult
            {
                bool passed;
                float max_abs_diff;
                float mean_abs_diff;
                float rel_l2_norm;
                size_t num_mismatches;
                size_t total_elements;
            };

            ComparisonResult compareMatrices(const float *reference, const float *test,
                                             int m, int n, float tolerance = 1e-3f)
            {
                ComparisonResult result{};
                result.total_elements = m * n;

                double sum_sq_diff = 0.0;
                double sum_sq_ref = 0.0;
                double sum_abs_diff = 0.0;
                result.max_abs_diff = 0.0f;
                result.num_mismatches = 0;

                for (int i = 0; i < m * n; ++i)
                {
                    float diff = std::fabs(reference[i] - test[i]);
                    sum_abs_diff += diff;
                    sum_sq_diff += diff * diff;
                    sum_sq_ref += reference[i] * reference[i];

                    result.max_abs_diff = std::max(result.max_abs_diff, diff);

                    // Count significant mismatches (> 1% relative error)
                    float rel_err = (std::fabs(reference[i]) > 1e-6f)
                                        ? diff / std::fabs(reference[i])
                                        : diff;
                    if (rel_err > 0.01f)
                    {
                        ++result.num_mismatches;
                    }
                }

                result.mean_abs_diff = sum_abs_diff / result.total_elements;
                result.rel_l2_norm = std::sqrt(sum_sq_diff / (sum_sq_ref + 1e-10));
                result.passed = (result.rel_l2_norm < tolerance);

                return result;
            }

            /**
             * @brief Log comparison result details
             */
            void logComparisonResult(const ComparisonResult &result, const char *test_name)
            {
                std::cout << "=== " << test_name << " ===" << std::endl;
                std::cout << "  Elements:       " << result.total_elements << std::endl;
                std::cout << "  Max abs diff:   " << result.max_abs_diff << std::endl;
                std::cout << "  Mean abs diff:  " << result.mean_abs_diff << std::endl;
                std::cout << "  Rel L2 norm:    " << result.rel_l2_norm << std::endl;
                std::cout << "  Mismatches:     " << result.num_mismatches
                          << " / " << result.total_elements
                          << " (" << (100.0 * result.num_mismatches / result.total_elements)
                          << "%)" << std::endl;
                std::cout << "  Status:         " << (result.passed ? "✓ PASSED" : "✗ FAILED")
                          << std::endl
                          << std::endl;
            }

            std::mt19937 rng_; ///< Random number generator
        };

        //=============================================================================
        // TEST CASES: Single Token (m=1)
        //=============================================================================

        /**
         * @brief Test single token inference with Qwen 0.5B dimensions
         *
         * Configuration:
         * - m=1 (single token)
         * - k=896 (Qwen 0.5B d_model)
         * - n=896 (attention projection)
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, SingleToken_Qwen05B_Attention)
        {
            const int m = 1;
            const int k = 896;
            const int n = 896;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            // Reference path: dequant → CBLAS
            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);

            // Fused path: Q4_0 GEMM
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            // Compare
            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "SingleToken_Qwen05B_Attention");

            EXPECT_TRUE(result.passed)
                << "Rel L2: " << result.rel_l2_norm << ", Max abs: " << result.max_abs_diff;
        }

        /**
         * @brief Test single token with FFN gate projection (large N)
         *
         * Configuration:
         * - m=1
         * - k=896
         * - n=4864 (Qwen 0.5B FFN intermediate dimension)
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, SingleToken_Qwen05B_FFN)
        {
            const int m = 1;
            const int k = 896;
            const int n = 4864;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "SingleToken_Qwen05B_FFN");

            EXPECT_TRUE(result.passed);
        }

        /**
         * @brief Test single token with larger model (Llama-style 4096)
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, SingleToken_Llama7B_Dimensions)
        {
            const int m = 1;
            const int k = 4096;
            const int n = 4096;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "SingleToken_Llama7B_Dimensions");

            EXPECT_TRUE(result.passed);
        }

        //=============================================================================
        // TEST CASES: Small Batch (m=8-32)
        //=============================================================================

        /**
         * @brief Test small batch (m=8) with Qwen 0.5B dimensions
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, SmallBatch8_Qwen05B)
        {
            const int m = 8;
            const int k = 896;
            const int n = 896;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "SmallBatch8_Qwen05B");

            EXPECT_TRUE(result.passed);
        }

        /**
         * @brief Test batch size 16
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, SmallBatch16_Qwen05B)
        {
            const int m = 16;
            const int k = 896;
            const int n = 896;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "SmallBatch16_Qwen05B");

            EXPECT_TRUE(result.passed);
        }

        /**
         * @brief Test batch size 32
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, SmallBatch32_Qwen05B)
        {
            const int m = 32;
            const int k = 896;
            const int n = 896;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "SmallBatch32_Qwen05B");

            EXPECT_TRUE(result.passed);
        }

        //=============================================================================
        // TEST CASES: Medium Batch (m=128)
        //=============================================================================

        /**
         * @brief Test medium batch (prefill-like workload)
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, MediumBatch128_Qwen05B)
        {
            const int m = 128;
            const int k = 896;
            const int n = 896;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "MediumBatch128_Qwen05B");

            EXPECT_TRUE(result.passed);
        }

        /**
         * @brief Test medium batch with FFN projection
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, MediumBatch128_Qwen05B_FFN)
        {
            const int m = 128;
            const int k = 896;
            const int n = 4864;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "MediumBatch128_Qwen05B_FFN");

            EXPECT_TRUE(result.passed);
        }

        //=============================================================================
        // TEST CASES: Large Batch (m=512) - Stress Testing
        //=============================================================================

        /**
         * @brief Test large batch (stress testing)
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, LargeBatch512_Qwen05B)
        {
            const int m = 512;
            const int k = 896;
            const int n = 896;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "LargeBatch512_Qwen05B");

            EXPECT_TRUE(result.passed);
        }

        //=============================================================================
        // TEST CASES: Edge Cases
        //=============================================================================

        /**
         * @brief Test with tiny dimensions (corner case)
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, EdgeCase_TinyDimensions)
        {
            const int m = 1;
            const int k = 64;
            const int n = 64;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "EdgeCase_TinyDimensions");

            EXPECT_TRUE(result.passed);
        }

        /**
         * @brief Test with non-power-of-2 dimensions
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, EdgeCase_NonPowerOf2)
        {
            const int m = 13;
            const int k = 777;
            const int n = 333;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "EdgeCase_NonPowerOf2");

            EXPECT_TRUE(result.passed);
        }

        /**
         * @brief Test with K not multiple of block size (32)
         */
        TEST_F(Test__Q4_0Gemm_DequantParity, EdgeCase_K_NotMultipleOfBlockSize)
        {
            const int m = 8;
            const int k = 100; // Not a multiple of 32
            const int n = 128;

            auto A = createRandomFP32Matrix(m, k);
            auto B_q4 = createRandomQ4_0Tensor(n, k);

            auto B_fp32 = dequantizeToFP32(B_q4.get());
            auto C_ref = gemm_cblas(A.data(), B_fp32.data(), m, n, k);
            auto C_fused = gemm_fused_q4(A.data(), B_q4.get(), m, n, k);

            auto result = compareMatrices(C_ref.data(), C_fused.data(), m, n, 1e-3f);
            logComparisonResult(result, "EdgeCase_K_NotMultipleOfBlockSize");

            EXPECT_TRUE(result.passed);
        }

    } // namespace test
} // namespace llaminar2
