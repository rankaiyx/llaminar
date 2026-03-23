/**
 * @file Q8_0NativeGemv.h
 * @brief Row-parallel GEMV reading directly from native Q8_0 blocks.
 *
 * Two entry points:
 *
 * 1. **q8_0_native_gemv**: Single-projection GEMV with own OMP region.
 * 2. **q8_0_native_gemv_fused**: Multi-projection GEMV in a single OMP
 *    region. Eliminates fork/join overhead between projections (e.g.
 *    QKV uses 1 region instead of 3, GateUp uses 1 instead of 2).
 *
 * Both use FP32 dequantize-and-multiply: weights are dequantized to FP32
 * and multiplied by FP32 activations, using AVX-512 FMA for accumulation.
 */

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/OpenMPUtils.h"

namespace llaminar2::cpu::native_vnni
{

#if defined(__AVX512F__)

    /**
     * @brief Hardware FP16→FP32 conversion using F16C (vcvtph2ps).
     * Single instruction vs the software path which has branches and bit ops.
     */
    static inline float hw_fp16_to_fp32(uint16_t h)
    {
        return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
    }

    /**
     * @brief Inline Q8_0 GEMV inner loop for one row.
     *
     * 4-block unrolled with 8 independent FMA chains and hardware F16C.
     * Returns the dot product: Σ scale[kb] * dot(qs[kb], A[kb*32..])
     */
    static inline float gemv_dot_row_q8_0(
        const Q8_0Block *__restrict row,
        const float *__restrict A,
        int bpr)
    {
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();

        int kb = 0;

        // Main loop: 4-block unroll for maximum ILP
        for (; kb + 3 < bpr; kb += 4)
        {
            // Prefetch 4 blocks ahead (~2 cache lines)
            if (kb + 7 < bpr)
            {
                _mm_prefetch(reinterpret_cast<const char *>(&row[kb + 4]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char *>(&row[kb + 6]), _MM_HINT_T0);
            }

            // Block 0
            {
                const __m512 ws = _mm512_set1_ps(hw_fp16_to_fp32(row[kb].d));
                acc0 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb].qs))))),
                    _mm512_loadu_ps(A + kb * 32), acc0);
                acc1 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb].qs + 16))))),
                    _mm512_loadu_ps(A + kb * 32 + 16), acc1);
            }
            // Block 1
            {
                const __m512 ws = _mm512_set1_ps(hw_fp16_to_fp32(row[kb + 1].d));
                acc2 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb + 1].qs))))),
                    _mm512_loadu_ps(A + (kb + 1) * 32), acc2);
                acc3 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb + 1].qs + 16))))),
                    _mm512_loadu_ps(A + (kb + 1) * 32 + 16), acc3);
            }
            // Block 2
            {
                const __m512 ws = _mm512_set1_ps(hw_fp16_to_fp32(row[kb + 2].d));
                acc4 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb + 2].qs))))),
                    _mm512_loadu_ps(A + (kb + 2) * 32), acc4);
                acc5 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb + 2].qs + 16))))),
                    _mm512_loadu_ps(A + (kb + 2) * 32 + 16), acc5);
            }
            // Block 3
            {
                const __m512 ws = _mm512_set1_ps(hw_fp16_to_fp32(row[kb + 3].d));
                acc6 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb + 3].qs))))),
                    _mm512_loadu_ps(A + (kb + 3) * 32), acc6);
                acc7 = _mm512_fmadd_ps(
                    _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb + 3].qs + 16))))),
                    _mm512_loadu_ps(A + (kb + 3) * 32 + 16), acc7);
            }
        }

        // Tail: remaining 1-3 blocks
        for (; kb < bpr; ++kb)
        {
            const __m512 ws = _mm512_set1_ps(hw_fp16_to_fp32(row[kb].d));
            acc0 = _mm512_fmadd_ps(
                _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb].qs))))),
                _mm512_loadu_ps(A + kb * 32), acc0);
            acc1 = _mm512_fmadd_ps(
                _mm512_mul_ps(ws, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(row[kb].qs + 16))))),
                _mm512_loadu_ps(A + kb * 32 + 16), acc1);
        }

        // Horizontal reduce: 8 accumulators → 1 scalar
        return _mm512_reduce_add_ps(
            _mm512_add_ps(
                _mm512_add_ps(_mm512_add_ps(acc0, acc2), _mm512_add_ps(acc4, acc6)),
                _mm512_add_ps(_mm512_add_ps(acc1, acc3), _mm512_add_ps(acc5, acc7))));
    }

    /**
     * @brief Row-parallel Q8_0 GEMV from native blocks.
     *
     * C[n] = Σ_kb  scale[n][kb] * dot(qs[n][kb], A[kb*32..(kb+1)*32-1])
     *
     * @param blocks  Native Q8_0 blocks [N × bpr], contiguous in memory
     * @param A       FP32 activation vector [K]
     * @param C       FP32 output vector [N]
     * @param N       Number of output rows
     * @param K       Input dimension
     * @param bpr     Blocks per row (= K / 32)
     */
    inline void q8_0_native_gemv(
        const Q8_0Block *__restrict blocks,
        const float *__restrict A,
        float *__restrict C,
        int N,
        int K,
        int bpr)
    {
        (void)K;

        auto do_gemv = [&]()
        {
#pragma omp for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                const Q8_0Block *__restrict row = blocks + static_cast<size_t>(n) * bpr;
                C[n] = gemv_dot_row_q8_0(row, A, bpr);
            }
        };

        OMP_WORKSHARE_REGION(do_gemv);
    }

    // =========================================================================
    // Fused multi-projection FP32-dequant GEMV (single OMP region)
    // =========================================================================

    /**
     * @brief Descriptor for one projection in a fused GEMV call.
     */
    struct FusedProjectionDesc
    {
        const Q8_0Block *weights; // weight blocks [N × bpr]
        float *output;            // output vector [N]
        const float *bias;        // optional bias [N], nullptr if none
        int N;                    // number of output rows
        int bpr;                  // blocks per row
    };

    /**
     * @brief Fused multi-projection FP32-dequant GEMV in a single OMP region.
     *
     * Processes all projections without re-entering the OMP parallel region.
     * Uses `nowait` between projections so threads finishing one projection
     * can immediately start the next.
     *
     * For QKV (3 projections): eliminates 2 OMP fork/join barriers per layer.
     * For GateUp (2 projections): eliminates 1 OMP fork/join barrier per layer.
     */
    inline void q8_0_native_gemv_fused(
        const float *__restrict A,
        const FusedProjectionDesc *projections,
        int num_projections)
    {
        auto do_gemv = [&]()
        {
            for (int p = 0; p < num_projections; ++p)
            {
                const auto &proj = projections[p];
                const int bpr = proj.bpr;
                const bool last = (p == num_projections - 1);

                if (!last)
                {
#pragma omp for schedule(static) nowait
                    for (int n = 0; n < proj.N; ++n)
                    {
                        const Q8_0Block *__restrict row = proj.weights + static_cast<size_t>(n) * bpr;
                        float val = gemv_dot_row_q8_0(row, A, bpr);
                        if (proj.bias) val += proj.bias[n];
                        proj.output[n] = val;
                    }
                }
                else
                {
#pragma omp for schedule(static)
                    for (int n = 0; n < proj.N; ++n)
                    {
                        const Q8_0Block *__restrict row = proj.weights + static_cast<size_t>(n) * bpr;
                        float val = gemv_dot_row_q8_0(row, A, bpr);
                        if (proj.bias) val += proj.bias[n];
                        proj.output[n] = val;
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_gemv);
    }

#endif // __AVX512F__

} // namespace llaminar2::cpu::native_vnni
