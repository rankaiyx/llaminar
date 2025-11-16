/**
 * @file Test__VNNIGemm_Minimal.cpp
 * @brief Minimal VNNI GEMM test to debug correctness issue
 *
 * ISSUE: Full correctness test shows 171% error (ref=-430.94, test=632.04)
 * This test uses SIMPLE data to isolate the bug.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "kernels/cpu/gemm_v3/VNNIGemm.h"

using namespace llaminar2;

class Test__VNNIGemm_Minimal : public ::testing::Test
{
protected:
    /**
     * @brief Simple reference GEMM (triple loop)
     */
    static void referenceGemm(
        const int8_t *A, const int8_t *B, float *C,
        const float *act_scales, const float *wgt_scales,
        int M, int N, int K, int K_BLK)
    {
        const int T = K / K_BLK;

        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    const int t = k / K_BLK;
                    const int8_t a_val = A[m * K + k];
                    const int8_t b_val = B[k * N + n];
                    sum += (float)a_val * (float)b_val * act_scales[t] * wgt_scales[n];
                }
                C[m * N + n] = sum;
            }
        }
    }
};

/**
 * @brief Simplest possible test: 4×16×4 with simple values
 */
TEST_F(Test__VNNIGemm_Minimal, TinyMatrix_4x16x4)
{
    const int M = 4, N = 16, K = 4;
    constexpr int M_R = 4;
    constexpr int N_R = 16;
    constexpr int K_BLK = 4;
    constexpr int UNROLL_K = 1;
    constexpr int PREFETCH_B_L1 = 0;
    constexpr int PREFETCH_B_L2 = 0;
    const int T = K / K_BLK; // T = 1

    std::cout << "\n=== Minimal VNNI Test: " << M << "×" << N << "×" << K << " ===" << std::endl;
    std::cout << "T = " << T << ", K_BLK = " << K_BLK << std::endl;

    // Create simple data: A = identity-like, B = identity-like
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B_unpacked(K * N);

    // A: diagonal pattern
    for (int i = 0; i < M * K; ++i)
        A[i] = 0;
    for (int i = 0; i < std::min(M, K); ++i)
        A[i * K + i] = 1; // Diagonal

    // B: diagonal pattern
    for (int i = 0; i < K * N; ++i)
        B_unpacked[i] = 0;
    for (int i = 0; i < std::min(K, N); ++i)
        B_unpacked[i * N + i] = 1; // Diagonal

    std::cout << "\nA (M×K):" << std::endl;
    for (int m = 0; m < M; ++m)
    {
        std::cout << "  ";
        for (int k = 0; k < K; ++k)
        {
            std::cout << (int)A[m * K + k] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nB (K×N):" << std::endl;
    for (int k = 0; k < K; ++k)
    {
        std::cout << "  ";
        for (int n = 0; n < N; ++n)
        {
            std::cout << (int)B_unpacked[k * N + n] << " ";
        }
        std::cout << std::endl;
    }

    // Unit scales
    std::vector<float> act_scales(T, 1.0f);
    std::vector<float> wgt_scales(N, 1.0f);

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
                    B_packed[offset + n * 4 + k_sub] = B_unpacked[k * N + n];
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

    std::vector<float> bias(N, 0.0f);

    // Reference computation
    std::vector<float> C_ref(M * N, 0.0f);
    referenceGemm(A.data(), B_unpacked.data(), C_ref.data(),
                  act_scales.data(), wgt_scales.data(),
                  M, N, K, K_BLK);

    std::cout << "\nC_ref (expected):" << std::endl;
    for (int m = 0; m < M; ++m)
    {
        std::cout << "  ";
        for (int n = 0; n < N; ++n)
        {
            std::cout << C_ref[m * N + n] << " ";
        }
        std::cout << std::endl;
    }

    // VNNI kernel
    std::vector<float> C_test(M * N, 0.0f);
    gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                          PREFETCH_B_L1, PREFETCH_B_L2,
                          true, true, true>(
        A.data(),
        Bp,
        C_test.data(),
        bias.data(),
        act_scales.data(),
        wgt_scales.data(),
        nullptr, // No zero-point correction
        M, N, K);

    std::cout << "\nC_test (actual):" << std::endl;
    for (int m = 0; m < M; ++m)
    {
        std::cout << "  ";
        for (int n = 0; n < N; ++n)
        {
            std::cout << C_test[m * N + n] << " ";
        }
        std::cout << std::endl;
    }

    // Compare
    std::cout << "\nComparison:" << std::endl;
    bool all_match = true;
    for (int i = 0; i < M * N; ++i)
    {
        if (std::abs(C_ref[i] - C_test[i]) > 0.01f)
        {
            std::cout << "  [" << i << "] ref=" << C_ref[i] << ", test=" << C_test[i]
                      << ", diff=" << (C_test[i] - C_ref[i]) << " ✗" << std::endl;
            all_match = false;
        }
    }

    if (all_match)
    {
        std::cout << "  All elements match! ✅" << std::endl;
    }
    else
    {
        std::cout << "  MISMATCH DETECTED! ❌" << std::endl;
    }

    EXPECT_TRUE(all_match) << "VNNI kernel output does not match reference";
}

/**
 * @brief Small test with non-zero values: 8×16×8
 */
TEST_F(Test__VNNIGemm_Minimal, SmallMatrix_8x16x8)
{
    const int M = 8, N = 16, K = 8;
    constexpr int M_R = 8;
    constexpr int N_R = 16;
    constexpr int K_BLK = 8;
    constexpr int UNROLL_K = 1; // Try without unrolling
    constexpr int PREFETCH_B_L1 = 0;
    constexpr int PREFETCH_B_L2 = 0;
    const int T = K / K_BLK;

    std::cout << "\n=== Small VNNI Test: " << M << "×" << N << "×" << K << " ===" << std::endl;

    // Simple values: A[i,j] = i+1, B[i,j] = j+1
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B_unpacked(K * N);

    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            A[m * K + k] = (m + 1);
        }
    }

    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            B_unpacked[k * N + n] = (n + 1);
        }
    }

    std::vector<float> act_scales(T, 1.0f);
    std::vector<float> wgt_scales(N, 1.0f);

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
                    B_packed[offset + n * 4 + k_sub] = B_unpacked[k * N + n];
                }
            }
        }
    }

    // Debug: Verify packing for column 1
    std::cout << "\nDEBUG: B column 1 (expecting all 2s):" << std::endl;
    std::cout << "  Unpacked B[:,1] = ";
    for (int k = 0; k < K; ++k)
        std::cout << (int)B_unpacked[k * N + 1] << " ";
    std::cout << std::endl;
    std::cout << "  Packed B[:,1] (should be same order) = ";
    for (int k = 0; k < K; ++k)
    {
        const int t = k / K_BLK;
        const int kk = k % K_BLK;
        // VNNI-interleaved layout: offset = t * (K_BLK/4) * (4*N) + (kk/4) * (4*N) + n*4 + (kk%4)
        const int offset = t * (K_BLK / 4) * (4 * N) + (kk / 4) * (4 * N);
        const int packed_idx = offset + 1 * 4 + (kk % 4);
        std::cout << (int)B_packed[packed_idx] << " ";
    }
    std::cout << std::endl;

    PackedB Bp;
    Bp.data = B_packed.data();
    Bp.ld_block = (K_BLK / 4) * (4 * N);
    Bp.ld_chunk = 4 * N;
    Bp.ld_col = 4;
    Bp.N = N;
    Bp.K_BLK = K_BLK;

    std::vector<float> bias(N, 0.0f);

    // Reference
    std::vector<float> C_ref(M * N, 0.0f);
    referenceGemm(A.data(), B_unpacked.data(), C_ref.data(),
                  act_scales.data(), wgt_scales.data(),
                  M, N, K, K_BLK);

    // VNNI
    std::vector<float> C_test(M * N, 0.0f);
    gemm_int8_vnni_kernel<M_R, N_R, K_BLK, UNROLL_K,
                          PREFETCH_B_L1, PREFETCH_B_L2,
                          true, true, true>(
        A.data(),
        Bp,
        C_test.data(),
        bias.data(),
        act_scales.data(),
        wgt_scales.data(),
        nullptr, // No zero-point correction for signed A
        M, N, K);

    // Expected: C[m,n] = (m+1) * (n+1) * K = (m+1) * (n+1) * 8
    std::cout << "Expected C[0,0] = " << (1 * 1 * K) << std::endl;
    std::cout << "Expected C[0,1] = " << (1 * 2 * K) << std::endl;
    std::cout << "Expected C[1,0] = " << (2 * 1 * K) << std::endl;
    std::cout << "Expected C[1,1] = " << (2 * 2 * K) << std::endl;

    std::cout << "Actual C_ref[0,0] = " << C_ref[0] << std::endl;
    std::cout << "Actual C_ref[0,1] = " << C_ref[1] << std::endl;
    std::cout << "Actual C_test[0,0] = " << C_test[0] << std::endl;
    std::cout << "Actual C_test[0,1] = " << C_test[1] << std::endl;

    // Compare
    bool all_match = true;
    for (int i = 0; i < M * N; ++i)
    {
        if (std::abs(C_ref[i] - C_test[i]) > 0.01f)
        {
            const int m = i / N;
            const int n = i % N;
            std::cout << "MISMATCH at [" << m << "," << n << "]: ref=" << C_ref[i]
                      << ", test=" << C_test[i] << ", diff=" << (C_test[i] - C_ref[i]) << std::endl;
            all_match = false;
        }
    }

    if (all_match)
    {
        std::cout << "All elements match! ✅" << std::endl;
    }

    EXPECT_TRUE(all_match) << "VNNI kernel output does not match reference";
}
