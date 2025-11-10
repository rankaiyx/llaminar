/**
 * @file FP16GemmImpl.h
 * @brief Multi-ISA FP16×FP16→FP32 GEMM implementations (AVX512F, AVX2, Scalar)
 *
 * This file provides separate inline implementations for FP16 GEMM across different
 * instruction sets, enabling:
 * 1. Runtime ISA detection and fallback (AVX512F → AVX2 → Scalar)
 * 2. Unit testing: Verify correctness by comparing AVX512 vs AVX2 vs Scalar results
 * 3. Performance analysis: Benchmark each ISA independently
 *
 * All implementations compute: C[m×n] = A[m×k] * B^T[n×k] in FP32
 * - FP16 input, FP32 accumulation (for numerical stability)
 * - Supports alpha/beta scaling (fp32)
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "../../../../tensors/FP16Utils.h"
#include <cstdint>
#include <algorithm>
#include <cstring>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __F16C__
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {
            namespace fp16_impl
            {

                // ============================================================================
                // AVX512F Implementation (Fastest)
                // ============================================================================

#if defined(__AVX512F__)

                /**
                 * @brief AVX512F FP16 GEMM micro-kernel with FP32 accumulation
                 *
                 * Strategy:
                 *   - Load FP16 values (16-bit)
                 *   - Convert to FP32 using _mm512_cvtph_ps (native FP16→FP32 conversion)
                 *   - Accumulate in FP32 for numerical stability
                 *   - Processes 16 FP16 elements per iteration
                 *
                 * Hardware requirements:
                 *   - AVX512F: Core AVX-512 instructions
                 *   - F16C is implied by AVX512F on all modern CPUs
                 *
                 * @param A_panel Input matrix A (m × k, FP16, row-major)
                 * @param B_panel Input matrix B^T (n × k, FP16, row-major)
                 * @param C Output matrix C (m × n, FP32, row-major with stride ldc)
                 * @param m Number of rows in A
                 * @param n Number of rows in B^T (columns in B)
                 * @param k Number of columns in A and B^T
                 * @param ldc Leading dimension of C
                 * @param alpha Scaling factor for A*B
                 * @param beta Scaling factor for existing C
                 */
                inline void gemm_avx512f_impl(
                    const uint16_t *A_panel,
                    const uint16_t *B_panel,
                    float *C,
                    int m, int n, int k,
                    int ldc,
                    float alpha = 1.0f,
                    float beta = 0.0f)
                {
                    constexpr int VEC_WIDTH = 16; // 16 FP16s per __m256i (256 bits)

                    const __m512 alpha_vec = _mm512_set1_ps(alpha);
                    const __m512 beta_vec = _mm512_set1_ps(beta);

                    for (int i = 0; i < m; ++i)
                    {
                        const uint16_t *A_row = A_panel + i * k;

                        for (int j = 0; j < n; ++j)
                        {
                            const uint16_t *B_row = B_panel + j * k;

                            // Accumulator register (16 FP32s)
                            __m512 accum = _mm512_setzero_ps();

                            // Vectorized K-loop: Process 16 FP16s at a time
                            int p = 0;
                            for (; p + VEC_WIDTH <= k; p += VEC_WIDTH)
                            {
                                // Load 16 FP16 values (256 bits = __m256i)
                                __m256i a_vec_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(A_row + p));
                                __m256i b_vec_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(B_row + p));

                                // Convert FP16 → FP32 (native instruction)
                                __m512 a_vec_fp32 = _mm512_cvtph_ps(a_vec_fp16);
                                __m512 b_vec_fp32 = _mm512_cvtph_ps(b_vec_fp16);

                                // FMA: accum += a * b
                                accum = _mm512_fmadd_ps(a_vec_fp32, b_vec_fp32, accum);
                            }

                            // Horizontal reduction: Sum 16 FP32s to single FP32
                            float sum = _mm512_reduce_add_ps(accum);

                            // Scalar tail: Handle remaining k % 16 elements
                            for (; p < k; ++p)
                            {
                                float a_val = fp16_to_fp32(A_row[p]);
                                float b_val = fp16_to_fp32(B_row[p]);
                                sum += a_val * b_val;
                            }

                            // Write result with alpha/beta scaling
                            if (beta == 0.0f)
                            {
                                C[i * ldc + j] = alpha * sum;
                            }
                            else
                            {
                                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                            }
                        }
                    }
                }

#endif // AVX512F

                // ============================================================================
                // AVX2 + F16C Implementation (Fallback #1)
                // ============================================================================

#if defined(__AVX2__) && defined(__F16C__)

                /**
                 * @brief AVX2 + F16C FP16 GEMM micro-kernel with FP32 accumulation
                 *
                 * Strategy:
                 *   - Load FP16 values (16-bit)
                 *   - Convert to FP32 using _mm256_cvtph_ps (F16C extension)
                 *   - Accumulate in FP32 (8 elements per __m256)
                 *   - Processes 8 FP16 elements per iteration
                 *
                 * Hardware requirements:
                 *   - AVX2: Haswell and later
                 *   - F16C: FP16 conversion instructions (also Haswell+)
                 *
                 * @param A_panel Input matrix A (m × k, FP16, row-major)
                 * @param B_panel Input matrix B^T (n × k, FP16, row-major)
                 * @param C Output matrix C (m × n, FP32, row-major with stride ldc)
                 * @param m Number of rows in A
                 * @param n Number of rows in B^T
                 * @param k Number of columns in A and B^T
                 * @param ldc Leading dimension of C
                 * @param alpha Scaling factor for A*B
                 * @param beta Scaling factor for existing C
                 */
                inline void gemm_avx2_f16c_impl(
                    const uint16_t *A_panel,
                    const uint16_t *B_panel,
                    float *C,
                    int m, int n, int k,
                    int ldc,
                    float alpha = 1.0f,
                    float beta = 0.0f)
                {
                    constexpr int VEC_WIDTH = 8; // 8 FP16s per __m128i (128 bits)

                    const __m256 alpha_vec = _mm256_set1_ps(alpha);
                    const __m256 beta_vec = _mm256_set1_ps(beta);

                    for (int i = 0; i < m; ++i)
                    {
                        const uint16_t *A_row = A_panel + i * k;

                        for (int j = 0; j < n; ++j)
                        {
                            const uint16_t *B_row = B_panel + j * k;

                            // Accumulator register (8 FP32s)
                            __m256 accum = _mm256_setzero_ps();

                            // Vectorized K-loop: Process 8 FP16s at a time
                            int p = 0;
                            for (; p + VEC_WIDTH <= k; p += VEC_WIDTH)
                            {
                                // Load 8 FP16 values (128 bits = __m128i)
                                __m128i a_vec_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(A_row + p));
                                __m128i b_vec_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(B_row + p));

                                // Convert FP16 → FP32 (F16C instruction)
                                __m256 a_vec_fp32 = _mm256_cvtph_ps(a_vec_fp16);
                                __m256 b_vec_fp32 = _mm256_cvtph_ps(b_vec_fp16);

                                // FMA: accum += a * b
                                accum = _mm256_fmadd_ps(a_vec_fp32, b_vec_fp32, accum);
                            }

                            // Horizontal reduction: Sum 8 FP32s to single FP32
                            float sum = 0.0f;
                            alignas(32) float accum_arr[8];
                            _mm256_store_ps(accum_arr, accum);
                            for (int kk = 0; kk < 8; ++kk)
                            {
                                sum += accum_arr[kk];
                            }

                            // Scalar tail: Handle remaining k % 8 elements
                            for (; p < k; ++p)
                            {
                                float a_val = fp16_to_fp32(A_row[p]);
                                float b_val = fp16_to_fp32(B_row[p]);
                                sum += a_val * b_val;
                            }

                            // Write result with alpha/beta scaling
                            if (beta == 0.0f)
                            {
                                C[i * ldc + j] = alpha * sum;
                            }
                            else
                            {
                                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                            }
                        }
                    }
                }

#endif // AVX2 && F16C

                // ============================================================================
                // Scalar Implementation (Universal Fallback)
                // ============================================================================

                /**
                 * @brief Scalar FP16 GEMM micro-kernel (portable, no SIMD)
                 *
                 * Pure C++ implementation using fp16_to_fp32 conversion:
                 *   - Works on any platform (no ISA requirements)
                 *   - Serves as ground truth for testing AVX512/AVX2 correctness
                 *   - Slowest but most portable
                 *
                 * @param A_panel Input matrix A (m × k, FP16, row-major)
                 * @param B_panel Input matrix B^T (n × k, FP16, row-major)
                 * @param C Output matrix C (m × n, FP32, row-major with stride ldc)
                 * @param m Number of rows in A
                 * @param n Number of rows in B^T
                 * @param k Number of columns in A and B^T
                 * @param ldc Leading dimension of C
                 * @param alpha Scaling factor for A*B
                 * @param beta Scaling factor for existing C
                 */
                inline void gemm_scalar_impl(
                    const uint16_t *A_panel,
                    const uint16_t *B_panel,
                    float *C,
                    int m, int n, int k,
                    int ldc,
                    float alpha = 1.0f,
                    float beta = 0.0f)
                {
                    for (int i = 0; i < m; ++i)
                    {
                        const uint16_t *A_row = A_panel + i * k;

                        for (int j = 0; j < n; ++j)
                        {
                            const uint16_t *B_row = B_panel + j * k;

                            // Scalar dot product: sum(A[i,:] * B[j,:])
                            float sum = 0.0f;
                            for (int p = 0; p < k; ++p)
                            {
                                float a_val = fp16_to_fp32(A_row[p]);
                                float b_val = fp16_to_fp32(B_row[p]);
                                sum += a_val * b_val;
                            }

                            // Write result with alpha/beta scaling
                            if (beta == 0.0f)
                            {
                                C[i * ldc + j] = alpha * sum;
                            }
                            else
                            {
                                C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                            }
                        }
                    }
                }

                // ============================================================================
                // Runtime ISA Detection and Dispatch
                // ============================================================================

                /**
                 * @brief Detect available ISA and dispatch to optimal implementation
                 *
                 * Priority:
                 * 1. AVX512F (if __AVX512F__ defined AND runtime support)
                 * 2. AVX2 + F16C (if both defined AND runtime support)
                 * 3. Scalar (always available)
                 *
                 * @return String indicating which implementation was used
                 */
                inline const char *gemm_fp16_auto(
                    const uint16_t *A_panel,
                    const uint16_t *B_panel,
                    float *C,
                    int m, int n, int k,
                    int ldc,
                    float alpha = 1.0f,
                    float beta = 0.0f)
                {
#if defined(__AVX512F__)
                    gemm_avx512f_impl(A_panel, B_panel, C, m, n, k, ldc, alpha, beta);
                    return "AVX512F";
#elif defined(__AVX2__) && defined(__F16C__)
                    gemm_avx2_f16c_impl(A_panel, B_panel, C, m, n, k, ldc, alpha, beta);
                    return "AVX2+F16C";
#else
                    gemm_scalar_impl(A_panel, B_panel, C, m, n, k, ldc, alpha, beta);
                    return "Scalar";
#endif
                }

                /**
                 * @brief Force specific ISA for testing (compile-time dispatch)
                 *
                 * Usage:
                 *   gemm_fp16_force<ISA_AVX512F>(...);    // Test AVX512F
                 *   gemm_fp16_force<ISA_AVX2_F16C>(...);  // Test AVX2+F16C
                 *   gemm_fp16_force<ISA_SCALAR>(...);     // Test Scalar
                 */
                enum ISAVariant
                {
                    ISA_AUTO = 0,
                    ISA_AVX512F = 1,
                    ISA_AVX2_F16C = 2,
                    ISA_SCALAR = 3
                };

                template <ISAVariant ISA>
                inline const char *gemm_fp16_force(
                    const uint16_t *A_panel,
                    const uint16_t *B_panel,
                    float *C,
                    int m, int n, int k,
                    int ldc,
                    float alpha = 1.0f,
                    float beta = 0.0f)
                {
                    if constexpr (ISA == ISA_AVX512F)
                    {
#if defined(__AVX512F__)
                        gemm_avx512f_impl(A_panel, B_panel, C, m, n, k, ldc, alpha, beta);
                        return "AVX512F";
#else
                        static_assert(ISA != ISA_AVX512F, "AVX512F not available on this platform");
                        return nullptr;
#endif
                    }
                    else if constexpr (ISA == ISA_AVX2_F16C)
                    {
#if defined(__AVX2__) && defined(__F16C__)
                        gemm_avx2_f16c_impl(A_panel, B_panel, C, m, n, k, ldc, alpha, beta);
                        return "AVX2+F16C";
#else
                        static_assert(ISA != ISA_AVX2_F16C, "AVX2+F16C not available on this platform");
                        return nullptr;
#endif
                    }
                    else if constexpr (ISA == ISA_SCALAR)
                    {
                        gemm_scalar_impl(A_panel, B_panel, C, m, n, k, ldc, alpha, beta);
                        return "Scalar";
                    }
                    else
                    {
                        return gemm_fp16_auto(A_panel, B_panel, C, m, n, k, ldc, alpha, beta);
                    }
                }

            } // namespace fp16_impl
        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
