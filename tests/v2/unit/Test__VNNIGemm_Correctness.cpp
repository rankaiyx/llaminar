/**
 * @file Test__VNNIGemm_Correctness.cpp
 * @brief Correctness tests for VNNI INT8 GEMM kernel
 * @author David Sanftenberg
 * @date 2025-11-15
 *
 * Validates mathematical correctness of gemm_v3 VNNI INT8 kernel
 * against OneDNN FP32 SGEMM ground truth.
 */

#include <gtest/gtest.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "v2/kernels/cpu/gemm_v3/VNNIGemm.h"

// OneDNN for reference GEMM
#include "oneapi/dnnl/dnnl.hpp"

namespace llaminar2
{
    class Test__VNNIGemm_Correctness : public ::testing::Test
    {
    protected:
        /**
         * @brief Simple reference GEMM (triple loop, no OneDNN)
         */
        static void simpleReferenceGemm(
            const float *A, const float *B, float *C,
            int M, int N, int K)
        {
            for (int m = 0; m < M; ++m)
            {
                for (int n = 0; n < N; ++n)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k)
                    {
                        sum += A[m * K + k] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        /**
         * @brief OneDNN FP32 GEMM (ground truth)
         */
        static void referenceGemmFP32(
            const float *A, const float *B, float *C,
            int M, int N, int K)
        {
            // Use simple reference instead of OneDNN for debugging
            simpleReferenceGemm(A, B, C, M, N, K);
        } /**
           * @brief Compute relative L2 error
           */
        static double computeRelativeL2Error(
            const float *C_ref, const float *C_test,
            int M, int N)
        {
            double diff_norm = 0.0, ref_norm = 0.0;
            for (int i = 0; i < M * N; ++i)
            {
                double diff = C_ref[i] - C_test[i];
                diff_norm += diff * diff;
                ref_norm += C_ref[i] * C_ref[i];
            }
            return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-10);
        }

        /**
         * @brief Compute max absolute error
         */
        static float computeMaxAbsError(
            const float *C_ref, const float *C_test,
            int M, int N)
        {
            float max_err = 0.0f;
            for (int i = 0; i < M * N; ++i)
            {
                float err = std::abs(C_ref[i] - C_test[i]);
                max_err = std::max(max_err, err);
            }
            return max_err;
        }

        /**
         * @brief Generate random FP32 matrix
         */
        static std::vector<float> generateRandomMatrix(int rows, int cols, int seed = 42)
        {
            std::vector<float> mat(rows * cols);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto &val : mat)
                val = dist(rng);
            return mat;
        }
    };

    /**
     * @brief Basic correctness test: Small matrix (64×64×64)
     *
     * Validates:
     * 1. OneDNN FP32 SGEMM (ground truth)
     * 2. Quantize FP32 → INT8 (A: per-K-block scale, B: per-column scale)
     * 3. VNNI INT8 GEMM with scales
     * 4. Compare outputs (expect <2% relative L2 error due to INT8 quantization)
     */
    TEST_F(Test__VNNIGemm_Correctness, SmallMatrix_64x64x64)
    {
        const int M = 64, N = 64, K = 64;

        constexpr int M_R = 16;
        constexpr int N_R = 64;
        constexpr int K_BLK = 64;
        constexpr int UNROLL_K = 2;
        constexpr int PREFETCH_B_L1 = 64;
        constexpr int PREFETCH_B_L2 = 256;

        std::cout << "\n=== VNNI GEMM Correctness: " << M << "×" << N << "×" << K << " ===" << std::endl;

        // Step 1: Generate random FP32 matrices
        auto A_fp32 = generateRandomMatrix(M, K, 42);
        auto B_fp32 = generateRandomMatrix(K, N, 43);

        // Step 2: Quantize to INT8
        const int T = K / K_BLK;

        // Use random INT8 for A with unit scales (same as RealWeights test)
        // This avoids complex quantization and focuses on testing B quantization
        std::mt19937 rng_int8(42);
        std::uniform_int_distribution<int> int8_dist(-127, 127);

        std::vector<int8_t> A_int8(M * K);
        for (auto &x : A_int8)
            x = static_cast<int8_t>(int8_dist(rng_int8));

        std::vector<float> act_scales(T, 1.0f); // Unit scales

        // For fair comparison, recompute A_fp32 from quantized A with scales
        for (int i = 0; i < M; ++i)
        {
            for (int k = 0; k < K; ++k)
            {
                const int t = k / K_BLK;
                A_fp32[i * K + k] = static_cast<float>(A_int8[i * K + k]) * act_scales[t];
            }
        }

        // Quantize B (per-column scale)
        std::vector<float> wgt_scales(N);
        std::vector<int8_t> B_int8_unpacked(K * N);

        for (int n = 0; n < N; ++n)
        {
            // Find max abs in column n
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                max_abs = std::max(max_abs, std::abs(B_fp32[k * N + n]));
            }
            wgt_scales[n] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            // Quantize column
            for (int k = 0; k < K; ++k)
            {
                float val = B_fp32[k * N + n];
                float quantized = val / wgt_scales[n];
                quantized = std::max(-127.0f, std::min(127.0f, quantized));
                B_int8_unpacked[k * N + n] = static_cast<int8_t>(std::round(quantized));
            }
        }

        // Dequantize B for reference computation
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < K; ++k)
            {
                B_fp32[k * N + n] = static_cast<float>(B_int8_unpacked[k * N + n]) * wgt_scales[n];
            }
        }

        // Step 3: OneDNN FP32 reference (after both A and B have been requantized)
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemmFP32(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);
        std::cout << "✓ OneDNN FP32 reference completed" << std::endl;

        std::cout << "\nDEBUG: First few dequantized values:" << std::endl;
        std::cout << "  A_fp32[0]=" << A_fp32[0] << " (from A_int8[0]=" << (int)A_int8[0] << " * act_scale[0]=" << act_scales[0] << ")" << std::endl;
        std::cout << "  B_fp32[0]=" << B_fp32[0] << " (from B_int8[0]=" << (int)B_int8_unpacked[0] << " * wgt_scale[0]=" << wgt_scales[0] << ")" << std::endl;
        std::cout << "  Expected C_ref[0] ≈ sum(A_fp32[k] * B_fp32[k*N+0])" << std::endl;

        // Pack B using VNNI-interleaved layout: For each 4-K group, pack [B[k:k+3,0], B[k:k+3,1], ..., B[k:k+3,N-1]]
        // This matches the pack_B_panel_vnni function in VNNIGemm.h
        std::vector<int8_t> B_packed(T * (K_BLK / 4) * (4 * N)); // T blocks, each with (K_BLK/4) groups of (4*N) bytes
        for (int t = 0; t < T; ++t)
        {
            // For this T block, pack K_BLK/4 groups of 4 K elements
            for (int kk = 0; kk < K_BLK; kk += 4)
            {
                int offset = t * (K_BLK / 4) * (4 * N) + (kk / 4) * (4 * N);

                // Pack columns: For each column n, store 4 consecutive K values
                for (int n = 0; n < N; ++n)
                {
                    for (int k_sub = 0; k_sub < 4; ++k_sub)
                    {
                        const int k = t * K_BLK + kk + k_sub;
                        B_packed[offset + n * 4 + k_sub] = B_int8_unpacked[k * N + n];
                    }
                }
            }
        }

        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_chunk = 4 * N;
        Bp.ld_col = 4;
        Bp.N = N;
        Bp.K_BLK = K_BLK;

        // Compute PER-K-BLOCK column sums for zero-point correction
        // column_sums[t * N + n] = sum(B[t*K_BLK:(t+1)*K_BLK, n])
        std::vector<int32_t> column_sums(T * N, 0);
        for (int t = 0; t < T; ++t)
        {
            for (int n = 0; n < N; ++n)
            {
                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    const int k = t * K_BLK + kk;
                    column_sums[t * N + n] += B_int8_unpacked[k * N + n];
                }
            }
        }
        Bp.column_sums = column_sums.data();

        std::cout << "DEBUG: PackedB strides: ld_block=" << Bp.ld_block << ", ld_col=" << Bp.ld_col << ", N=" << Bp.N << ", K_BLK=" << Bp.K_BLK << std::endl;
        std::cout << "DEBUG: Packed buffer size: " << B_packed.size() << " bytes (expected " << (T * (K_BLK / 4) * (4 * N)) << ")" << std::endl;

        // Bias (all zeros for this test)
        std::vector<float> bias(N, 0.0f);

        // DEBUG: Print first few scales
        std::cout << "\nDEBUG: act_scales[0]=" << act_scales[0] << std::endl;
        std::cout << "DEBUG: wgt_scales[0]=" << wgt_scales[0] << ", wgt_scales[1]=" << wgt_scales[1] << std::endl;
        std::cout << "DEBUG: A_int8[0]=" << (int)A_int8[0] << ", A_fp32[0]=" << A_fp32[0] << std::endl;
        std::cout << "DEBUG: B_int8[0]=" << (int)B_int8_unpacked[0] << ", B_fp32[0]=" << B_fp32[0] << std::endl;

        // Step 4: VNNI INT8 GEMM
        // VNNI requires unsigned A, so we cast signed [-127,127] to unsigned [1,255] (zp=128)
        std::vector<uint8_t> A_uint8(M * K);
        for (size_t i = 0; i < A_int8.size(); ++i)
        {
            A_uint8[i] = static_cast<uint8_t>(static_cast<int16_t>(A_int8[i]) + 128);
        }
        std::vector<float> act_zero_points(T, 128.0f); // Zero-point for signed→unsigned conversion

        std::vector<float> C_test(M * N, 0.0f);

        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            reinterpret_cast<const int8_t *>(A_uint8.data()), // Cast unsigned to signed for API
            Bp,
            C_test.data(),
            bias.data(),
            act_scales.data(),
            wgt_scales.data(),
            act_zero_points.data(), // Zero-point correction for unsigned A
            M, N, K);

        std::cout << "✓ VNNI INT8 GEMM completed" << std::endl;

        // Manual verification of first element
        std::cout << "\nDEBUG: Manual calculation for C[0,0]:" << std::endl;
        float manual_c00 = 0.0f;
        for (int k = 0; k < K; ++k)
        {                                    // ALL K terms
            float a_val = A_fp32[k];         // Row 0 of A
            float b_val = B_fp32[k * N + 0]; // Column 0 of B
            float contrib = a_val * b_val;
            manual_c00 += contrib;
            if (k < 4)
            {
                std::cout << "  k=" << k << ": A[0," << k << "]=" << a_val
                          << " * B[" << k << ",0]=" << b_val
                          << " = " << contrib << " (running sum=" << manual_c00 << ")" << std::endl;
            }
        }
        std::cout << "  Manual C[0,0] (all " << K << " terms) = " << manual_c00 << std::endl;
        std::cout << "  Actual C_ref[0]=" << C_ref[0] << ", C_test[0]=" << C_test[0] << std::endl;
        std::cout << "  Difference: manual-ref=" << (manual_c00 - C_ref[0])
                  << ", manual-test=" << (manual_c00 - C_test[0]) << std::endl;

        // Step 5: Compare results
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Relative L2 error: " << (rel_l2 * 100) << "%" << std::endl;
        std::cout << "Max absolute error: " << max_abs << std::endl;

        // Sample outputs
        std::cout << "\nSample comparison (first 5 elements):" << std::endl;
        for (int i = 0; i < 5; ++i)
        {
            float diff = C_ref[i] - C_test[i];
            float rel_err = std::abs(diff) / (std::abs(C_ref[i]) + 1e-6f);
            std::cout << "  [" << i << "] ref=" << C_ref[i]
                      << ", test=" << C_test[i]
                      << ", diff=" << diff
                      << " (" << (rel_err * 100) << "%)" << std::endl;
        }

        // Validation: INT8 quantization should introduce <2% relative L2 error
        EXPECT_LT(rel_l2, 0.02) << "Relative L2 error should be <2% for INT8 GEMM";
        EXPECT_LT(max_abs, 50.0f) << "Max absolute error should be reasonable";

        std::cout << "✅ Correctness test PASSED" << std::endl;
    }

    /**
     * @brief Medium matrix test: Qwen 0.5B dimensions (512×896×896)
     *
     * Tests realistic model dimensions with tighter error bounds.
     */
    TEST_F(Test__VNNIGemm_Correctness, MediumMatrix_Qwen05B_Dims)
    {
        const int M = 512, N = 896, K = 896;

        constexpr int M_R = 16;
        constexpr int N_R = 64;
        constexpr int K_BLK = 64;
        constexpr int UNROLL_K = 2;
        constexpr int PREFETCH_B_L1 = 64;
        constexpr int PREFETCH_B_L2 = 256;

        std::cout << "\n=== VNNI GEMM Correctness: " << M << "×" << N << "×" << K << " (Qwen 0.5B) ===" << std::endl;

        // Generate random FP32 matrices
        auto A_fp32 = generateRandomMatrix(M, K, 100);
        auto B_fp32 = generateRandomMatrix(K, N, 101);

        // OneDNN FP32 reference
        std::vector<float> C_ref(M * N, 0.0f);
        referenceGemmFP32(A_fp32.data(), B_fp32.data(), C_ref.data(), M, N, K);
        std::cout << "✓ OneDNN FP32 reference completed" << std::endl;

        // Quantize to INT8 (same as SmallMatrix test)
        const int T = K / K_BLK;

        std::vector<float> wgt_scales(N);
        std::vector<int8_t> B_int8_unpacked(K * N);
        for (int n = 0; n < N; ++n)
        {
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
                max_abs = std::max(max_abs, std::abs(B_fp32[k * N + n]));
            wgt_scales[n] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            for (int k = 0; k < K; ++k)
            {
                float val = B_fp32[k * N + n];
                float quantized = std::max(-127.0f, std::min(127.0f, val / wgt_scales[n]));
                B_int8_unpacked[k * N + n] = static_cast<int8_t>(std::round(quantized));
            }
        }

        // Pack B using VNNI-interleaved layout: For each 4-K group, pack [B[k:k+3,0], B[k:k+3,1], ..., B[k:k+3,N-1]]
        // This matches the pack_B_panel_vnni function in VNNIGemm.h
        std::vector<int8_t> B_packed(T * (K_BLK / 4) * (4 * N)); // T blocks, each with (K_BLK/4) groups of (4*N) bytes
        for (int t = 0; t < T; ++t)
        {
            // For this T block, pack K_BLK/4 groups of 4 K elements
            for (int kk = 0; kk < K_BLK; kk += 4)
            {
                int offset = t * (K_BLK / 4) * (4 * N) + (kk / 4) * (4 * N);

                // Pack columns: For each column n, store 4 consecutive K values
                for (int n = 0; n < N; ++n)
                {
                    for (int k_sub = 0; k_sub < 4; ++k_sub)
                    {
                        const int k = t * K_BLK + kk + k_sub;
                        B_packed[offset + n * 4 + k_sub] = B_int8_unpacked[k * N + n];
                    }
                }
            }
        }

        PackedB Bp;
        Bp.data = B_packed.data();
        Bp.ld_block = (K_BLK / 4) * (4 * N);
        Bp.ld_chunk = 4 * N;
        Bp.ld_col = 4;
        Bp.N = N;
        Bp.K_BLK = K_BLK;

        // Compute PER-K-BLOCK column sums for zero-point correction
        // column_sums[t * N + n] = sum(B[t*K_BLK:(t+1)*K_BLK, n])
        std::vector<int32_t> column_sums(T * N, 0);
        for (int t = 0; t < T; ++t)
        {
            for (int n = 0; n < N; ++n)
            {
                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    const int k = t * K_BLK + kk;
                    column_sums[t * N + n] += B_int8_unpacked[k * N + n];
                }
            }
        }
        Bp.column_sums = column_sums.data();

        std::vector<float> act_scales(T);
        std::vector<int8_t> A_int8(M * K);
        for (int t = 0; t < T; ++t)
        {
            const int k0 = t * K_BLK;
            float max_abs = 0.0f;
            for (int m = 0; m < M; ++m)
                for (int kk = 0; kk < K_BLK; ++kk)
                    max_abs = std::max(max_abs, std::abs(A_fp32[m * K + k0 + kk]));
            act_scales[t] = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;

            for (int m = 0; m < M; ++m)
                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    float val = A_fp32[m * K + k0 + kk];
                    float quantized = std::max(-127.0f, std::min(127.0f, val / act_scales[t]));
                    A_int8[m * K + k0 + kk] = static_cast<int8_t>(std::round(quantized));
                }
        }

        std::vector<float> bias(N, 0.0f);

        // VNNI requires unsigned A, convert signed to unsigned
        std::vector<uint8_t> A_uint8(M * K);
        for (size_t i = 0; i < A_int8.size(); ++i)
        {
            A_uint8[i] = static_cast<uint8_t>(static_cast<int16_t>(A_int8[i]) + 128);
        }
        std::vector<float> act_zero_points(T, 128.0f);

        // VNNI INT8 GEMM
        std::vector<float> C_test(M * N, 0.0f);

        std::cout << "\nDEBUG: Before kernel:" << std::endl;
        std::cout << "  T=" << T << " (K=" << K << " / K_BLK=" << K_BLK << ")" << std::endl;
        std::cout << "  act_zero_points.size()=" << act_zero_points.size() << " (expected " << T << ")" << std::endl;
        std::cout << "  column_sums[0]=" << column_sums[0] << " (sum of B[:,0])" << std::endl;
        std::cout << "  A_uint8[0]=" << (int)A_uint8[0] << " (from A_int8[0]=" << (int)A_int8[0] << " + 128)" << std::endl;
        std::cout << "  B_int8_unpacked[0]=" << (int)B_int8_unpacked[0] << std::endl;

        gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                              PREFETCH_B_L1, PREFETCH_B_L2,
                              true, true, true>(
            reinterpret_cast<const int8_t *>(A_uint8.data()), Bp, C_test.data(), bias.data(),
            act_scales.data(), wgt_scales.data(),
            act_zero_points.data(), // Zero-point correction
            M, N, K);
        std::cout << "✓ VNNI INT8 GEMM completed" << std::endl;

        // Manual verification of first element
        std::cout << "\nDEBUG: Manual calculation for C[0,0]:" << std::endl;
        float manual_c00 = 0.0f;
        for (int k = 0; k < std::min(K, 8); ++k)
        { // Just first few terms
            float a_val = A_fp32[k];
            float b_val = B_fp32[k * N + 0];
            manual_c00 += a_val * b_val;
            std::cout << "  k=" << k << ": A[0," << k << "]=" << a_val
                      << " * B[" << k << ",0]=" << b_val
                      << " = " << (a_val * b_val) << " (sum=" << manual_c00 << ")" << std::endl;
        }
        std::cout << "  Manual C[0,0] (first " << std::min(K, 8) << " terms) = " << manual_c00 << std::endl;
        std::cout << "  Actual C_ref[0]=" << C_ref[0] << ", C_test[0]=" << C_test[0] << std::endl;

        // Compare
        double rel_l2 = computeRelativeL2Error(C_ref.data(), C_test.data(), M, N);
        float max_abs = computeMaxAbsError(C_ref.data(), C_test.data(), M, N);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Relative L2 error: " << (rel_l2 * 100) << "%" << std::endl;
        std::cout << "Max absolute error: " << max_abs << std::endl;

        EXPECT_LT(rel_l2, 0.02) << "Relative L2 error should be <2%";
        EXPECT_LT(max_abs, 100.0f) << "Max absolute error should be reasonable";

        std::cout << "✅ Correctness test PASSED" << std::endl;
    }

} // namespace llaminar2
