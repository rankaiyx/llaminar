/**
 * @file SoftmaxPrimitivesImpl.h
 * @brief Inline implementations of softmax primitives (all precisions, all SIMD levels)
 * @author David Sanftenberg
 *
 * Separated implementations allow:
 * - Individual testing of scalar vs AVX2 vs AVX512
 * - Performance benchmarking per SIMD level
 * - Parity validation (scalar as ground truth)
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__AVX512F__)
#include <immintrin.h>
#define LLAMINAR_HAS_AVX512
#elif defined(__AVX2__)
#include <immintrin.h>
#define LLAMINAR_HAS_AVX2
#endif

#if defined(__F16C__)
#define LLAMINAR_HAS_F16C
#endif

#if defined(__AVX512BF16__)
#define LLAMINAR_HAS_AVX512BF16
#endif

namespace llaminar2::primitives
{

    // ============================================================================
    // Helper Functions
    // ============================================================================

    namespace detail
    {
#if defined(LLAMINAR_HAVE_LIBMVEC)
        extern "C"
        {
#if defined(__AVX512F__)
            __m512 _ZGVeN16v_expf(__m512) __attribute__((weak));
#endif
#if defined(__AVX2__)
            __m256 _ZGVcN8v_expf(__m256) __attribute__((weak));
#endif
#if defined(__SSE2__)
            __m128 _ZGVbN4v_expf(__m128) __attribute__((weak));
#endif
        }
#endif
        // BF16 ↔ FP32 conversion (manual)
        inline float bf16_to_fp32_scalar(uint16_t bf16)
        {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16) << 16;
            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        inline uint16_t fp32_to_bf16_scalar(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(float));
            // Round to nearest even (RNE)
            uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
            uint32_t rounded = fp32_bits + rounding_bias;
            return static_cast<uint16_t>(rounded >> 16);
        }

#if defined(__SSE2__)
        inline float hsum128(__m128 v)
        {
            __m128 shuf = _mm_movehdup_ps(v);
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        }

        inline __m128 floor_ps_compat(__m128 x)
        {
            __m128i truncated = _mm_cvttps_epi32(x);
            __m128 floored = _mm_cvtepi32_ps(truncated);
            __m128 mask = _mm_cmpgt_ps(floored, x);
            return _mm_sub_ps(floored, _mm_and_ps(mask, _mm_set1_ps(1.0f)));
        }
#endif

#if defined(__AVX2__)
        inline float hsum256(__m256 v)
        {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            return hsum128(sum);
        }

        inline __m256 floor_ps_compat(__m256 x)
        {
            __m256i truncated = _mm256_cvttps_epi32(x);
            __m256 floored = _mm256_cvtepi32_ps(truncated);
            __m256 mask = _mm256_cmp_ps(floored, x, _CMP_GT_OQ);
            return _mm256_sub_ps(floored, _mm256_and_ps(mask, _mm256_set1_ps(1.0f)));
        }
#endif

#if defined(LLAMINAR_HAS_AVX512)
        inline float hsum512(__m512 v)
        {
            return _mm512_reduce_add_ps(v);
        }

        inline __m512 floor_ps_compat(__m512 x)
        {
            __m512i truncated = _mm512_cvttps_epi32(x);
            __m512 floored = _mm512_cvtepi32_ps(truncated);
            __mmask16 mask = _mm512_cmp_ps_mask(floored, x, _CMP_GT_OQ);
            return _mm512_mask_sub_ps(floored, mask, floored, _mm512_set1_ps(1.0f));
        }
#endif

#if defined(LLAMINAR_HAS_F16C)
        // FP16 ↔ FP32 conversion (using F16C)
        inline float fp16_to_fp32_f16c(uint16_t fp16)
        {
            __m128i vec = _mm_cvtsi32_si128(fp16);
            __m128 fp32_vec = _mm_cvtph_ps(vec);
            return _mm_cvtss_f32(fp32_vec);
        }

        inline uint16_t fp32_to_fp16_f16c(float fp32)
        {
            __m128 fp32_vec = _mm_set_ss(fp32);
            __m128i fp16_vec = _mm_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
            return static_cast<uint16_t>(_mm_cvtsi128_si32(fp16_vec));
        }
#else
        // FP16 ↔ FP32 conversion (manual, fallback)
        inline float fp16_to_fp32_scalar(uint16_t fp16)
        {
            uint32_t sign = (fp16 & 0x8000) << 16;
            uint32_t exp = (fp16 & 0x7C00) >> 10;
            uint32_t mant = (fp16 & 0x03FF);

            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    fp32_bits = sign; // Zero
                }
                else
                {
                    // Denormal
                    exp = 1;
                    while ((mant & 0x0400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x03FF;
                    fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                }
            }
            else if (exp == 0x1F)
            {
                // Inf/NaN
                fp32_bits = sign | 0x7F800000 | (mant << 13);
            }
            else
            {
                // Normal
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }

            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        inline uint16_t fp32_to_fp16_scalar(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(float));

            uint32_t sign = (fp32_bits & 0x80000000) >> 16;
            int32_t exp = ((fp32_bits & 0x7F800000) >> 23) - 127 + 15;
            uint32_t mant = (fp32_bits & 0x007FFFFF);

            uint16_t fp16;
            if (exp <= 0)
            {
                if (exp < -10)
                {
                    fp16 = static_cast<uint16_t>(sign); // Underflow to zero
                }
                else
                {
                    // Denormal
                    mant |= 0x00800000;
                    mant >>= (1 - exp);
                    fp16 = static_cast<uint16_t>(sign | (mant >> 13));
                }
            }
            else if (exp >= 0x1F)
            {
                fp16 = static_cast<uint16_t>(sign | 0x7C00); // Overflow to inf
            }
            else
            {
                fp16 = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
            }

            return fp16;
        }
#endif

#if defined(__SSE2__)
        inline __m128 exp128_ps_poly(__m128 x)
        {
            const __m128 max_val = _mm_set1_ps(88.3762626647949f);
            const __m128 min_val = _mm_set1_ps(-88.3762626647949f);
            x = _mm_min_ps(x, max_val);
            x = _mm_max_ps(x, min_val);

            const __m128 log2e = _mm_set1_ps(1.44269504088896341f);
            const __m128 half = _mm_set1_ps(0.5f);
            __m128 fx = _mm_add_ps(_mm_mul_ps(x, log2e), half);
            __m128 floored = floor_ps_compat(fx);

            const __m128 c1 = _mm_set1_ps(0.693359375f);
            const __m128 c2 = _mm_set1_ps(-2.12194440e-4f);
            __m128 temp = _mm_mul_ps(floored, c1);
            __m128 z = _mm_sub_ps(x, temp);
            temp = _mm_mul_ps(floored, c2);
            x = _mm_sub_ps(z, temp);

            const __m128 p0 = _mm_set1_ps(1.9875691500e-4f);
            const __m128 p1 = _mm_set1_ps(1.3981999507e-3f);
            const __m128 p2 = _mm_set1_ps(8.3334519073e-3f);
            const __m128 p3 = _mm_set1_ps(4.1665795894e-2f);
            const __m128 p4 = _mm_set1_ps(1.6666665459e-1f);
            const __m128 p5 = _mm_set1_ps(5.0000001201e-1f);

            __m128 y = p0;
            y = _mm_add_ps(_mm_mul_ps(y, x), p1);
            y = _mm_add_ps(_mm_mul_ps(y, x), p2);
            y = _mm_add_ps(_mm_mul_ps(y, x), p3);
            y = _mm_add_ps(_mm_mul_ps(y, x), p4);
            y = _mm_add_ps(_mm_mul_ps(y, x), p5);
            y = _mm_add_ps(_mm_mul_ps(y, x), _mm_set1_ps(1.0f));

            __m128i emm0 = _mm_cvttps_epi32(floored);
            emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(0x7f));
            emm0 = _mm_slli_epi32(emm0, 23);
            __m128 pow2n = _mm_castsi128_ps(emm0);
            return _mm_mul_ps(y, pow2n);
        }

        inline __m128 exp128_ps(__m128 x)
        {
#if defined(LLAMINAR_HAVE_LIBMVEC)
            if (_ZGVbN4v_expf)
            {
                return _ZGVbN4v_expf(x);
            }
#endif
            return exp128_ps_poly(x);
        }
#endif

#if defined(__AVX2__)
        inline __m256 exp256_ps_poly(__m256 x)
        {
            const __m256 max_val = _mm256_set1_ps(88.3762626647949f);
            const __m256 min_val = _mm256_set1_ps(-88.3762626647949f);
            x = _mm256_min_ps(x, max_val);
            x = _mm256_max_ps(x, min_val);

            const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
            const __m256 half = _mm256_set1_ps(0.5f);
            __m256 fx = _mm256_add_ps(_mm256_mul_ps(x, log2e), half);
            __m256 floored = floor_ps_compat(fx);

            const __m256 c1 = _mm256_set1_ps(0.693359375f);
            const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
            __m256 temp = _mm256_mul_ps(floored, c1);
            __m256 z = _mm256_sub_ps(x, temp);
            temp = _mm256_mul_ps(floored, c2);
            x = _mm256_sub_ps(z, temp);

            const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
            const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
            const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
            const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
            const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
            const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);

            __m256 y = p0;
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p1);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p2);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p3);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p4);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), p5);
            y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.0f));

            __m256i emm0 = _mm256_cvttps_epi32(floored);
            emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
            emm0 = _mm256_slli_epi32(emm0, 23);
            __m256 pow2n = _mm256_castsi256_ps(emm0);
            return _mm256_mul_ps(y, pow2n);
        }

        inline __m256 exp256_ps(__m256 x)
        {
#if defined(LLAMINAR_HAVE_LIBMVEC)
            if (_ZGVcN8v_expf)
            {
                return _ZGVcN8v_expf(x);
            }
#endif
            return exp256_ps_poly(x);
        }
#endif

#if defined(LLAMINAR_HAS_AVX512)
        inline __m512 exp512_ps_poly(__m512 x)
        {
            const __m512 max_val = _mm512_set1_ps(88.3762626647949f);
            const __m512 min_val = _mm512_set1_ps(-88.3762626647949f);
            x = _mm512_min_ps(x, max_val);
            x = _mm512_max_ps(x, min_val);

            const __m512 log2e = _mm512_set1_ps(1.44269504088896341f);
            const __m512 half = _mm512_set1_ps(0.5f);
            __m512 fx = _mm512_add_ps(_mm512_mul_ps(x, log2e), half);
            __m512 floored = floor_ps_compat(fx);

            const __m512 c1 = _mm512_set1_ps(0.693359375f);
            const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);
            __m512 temp = _mm512_mul_ps(floored, c1);
            __m512 z = _mm512_sub_ps(x, temp);
            temp = _mm512_mul_ps(floored, c2);
            x = _mm512_sub_ps(z, temp);

            const __m512 p0 = _mm512_set1_ps(1.9875691500e-4f);
            const __m512 p1 = _mm512_set1_ps(1.3981999507e-3f);
            const __m512 p2 = _mm512_set1_ps(8.3334519073e-3f);
            const __m512 p3 = _mm512_set1_ps(4.1665795894e-2f);
            const __m512 p4 = _mm512_set1_ps(1.6666665459e-1f);
            const __m512 p5 = _mm512_set1_ps(5.0000001201e-1f);

            __m512 y = p0;
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p1);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p2);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p3);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p4);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), p5);
            y = _mm512_add_ps(_mm512_mul_ps(y, x), _mm512_set1_ps(1.0f));

            __m512i emm0 = _mm512_cvttps_epi32(floored);
            emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7f));
            emm0 = _mm512_slli_epi32(emm0, 23);
            __m512 pow2n = _mm512_castsi512_ps(emm0);
            return _mm512_mul_ps(y, pow2n);
        }

        inline __m512 exp512_ps(__m512 x)
        {
#if defined(LLAMINAR_HAVE_LIBMVEC)
            if (_ZGVeN16v_expf)
            {
                return _ZGVeN16v_expf(x);
            }
#endif
            return exp512_ps_poly(x);
        }
#endif

#if defined(LLAMINAR_HAS_AVX2)
        // Horizontal max reduction for AVX2
        inline float hmax256(__m256 v)
        {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 m = _mm_max_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(m);
            m = _mm_max_ps(m, shuf);
            shuf = _mm_movehl_ps(shuf, m);
            m = _mm_max_ps(m, shuf);
            return _mm_cvtss_f32(m);
        }
#endif

    } // namespace detail

    // ============================================================================
    // FP32 Softmax - Scalar Implementation
    // ============================================================================

    inline void softmax_row_fp32_scalar(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Pass 1: Find max (with causal masking)
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Compute sum of exp(x - max)
        double sum = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0.0f;
            }
            else
            {
                float v = row[c] * scale;
                row[c] = std::exp(v - row_max) * inv;
            }
        }
    }

    // ============================================================================
    // FP32 Softmax - AVX2 Implementation
    // ============================================================================

    inline void softmax_row_fp32_avx2(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX2)
        const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        const __m256 scale_ps = _mm256_set1_ps(scale);

        // Pass 1: Find max
        __m256 vmax = neg_inf;
        int c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m256 v = _mm256_loadu_ps(row + c);
            v = _mm256_mul_ps(v, scale_ps);

            if (causal)
            {
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, v);
                for (int lane = 0; lane < 8; ++lane)
                {
                    if (c + lane > row_idx)
                        tmp[lane] = -std::numeric_limits<float>::infinity();
                }
                v = _mm256_load_ps(tmp);
            }

            vmax = _mm256_max_ps(vmax, v);
        }

        float row_max = detail::hmax256(vmax);

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize
        const __m256 inv_ps = _mm256_set1_ps(inv);
        const __m256 max_ps = _mm256_set1_ps(row_max);
        const __m256 zero_ps = _mm256_setzero_ps();

        c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m256 v = _mm256_loadu_ps(row + c);
            v = _mm256_mul_ps(v, scale_ps);

            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, v);

            float out[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                if (causal && c + lane > row_idx)
                {
                    out[lane] = 0.0f;
                }
                else
                {
                    out[lane] = std::exp(tmp[lane] - row_max) * inv;
                }
            }

            _mm256_storeu_ps(row + c, _mm256_loadu_ps(out));
        }

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0.0f;
            }
            else
            {
                float v = row[c] * scale;
                row[c] = std::exp(v - row_max) * inv;
            }
        }
#else
        // Fallback to scalar if AVX2 not available
        softmax_row_fp32_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP32 Softmax - AVX512 Implementation
    // ============================================================================

    inline void softmax_row_fp32_avx512(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        const __m512 scale_ps = _mm512_set1_ps(scale);

        // Pass 1: Find max
        __m512 vmax = neg_inf;
        int c = 0;
        for (; c + 16 <= cols; c += 16)
        {
            __m512 v = _mm512_loadu_ps(row + c);
            v = _mm512_mul_ps(v, scale_ps);

            if (causal)
            {
                // Mask where c+lane > row_idx
                __mmask16 mask = 0;
                for (int lane = 0; lane < 16; ++lane)
                {
                    if (c + lane <= row_idx)
                        mask |= (1 << lane);
                }
                v = _mm512_mask_blend_ps(mask, neg_inf, v);
            }

            vmax = _mm512_max_ps(vmax, v);
        }

        float row_max = _mm512_reduce_max_ps(vmax);

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = row[c] * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize
        const __m512 inv_ps = _mm512_set1_ps(inv);
        const __m512 max_ps = _mm512_set1_ps(row_max);
        const __m512 zero_ps = _mm512_setzero_ps();

        c = 0;
        for (; c + 16 <= cols; c += 16)
        {
            __m512 v = _mm512_loadu_ps(row + c);
            v = _mm512_mul_ps(v, scale_ps);

            alignas(64) float tmp[16];
            _mm512_store_ps(tmp, v);

            float out[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                if (causal && c + lane > row_idx)
                {
                    out[lane] = 0.0f;
                }
                else
                {
                    out[lane] = std::exp(tmp[lane] - row_max) * inv;
                }
            }

            _mm512_storeu_ps(row + c, _mm512_loadu_ps(out));
        }

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0.0f;
            }
            else
            {
                float v = row[c] * scale;
                row[c] = std::exp(v - row_max) * inv;
            }
        }
#else
        // Fallback to AVX2 if AVX512 not available
        softmax_row_fp32_avx2(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP32 Softmax - Compile-Time Dispatch
    // ============================================================================

    inline void softmax_row_fp32(
        float *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        softmax_row_fp32_avx512(row, cols, causal, scale, row_idx);
#elif defined(LLAMINAR_HAS_AVX2)
        softmax_row_fp32_avx2(row, cols, causal, scale, row_idx);
#else
        softmax_row_fp32_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // BF16 Softmax - Scalar Implementation
    // ============================================================================

    inline void softmax_row_bf16_scalar(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Pass 1: Find max (convert BF16→FP32 on the fly)
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::bf16_to_fp32_scalar(row[c]) * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::bf16_to_fp32_scalar(row[c]) * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize and convert FP32→BF16
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0;
            }
            else
            {
                float v = detail::bf16_to_fp32_scalar(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_bf16_scalar(result);
            }
        }
    }

    // ============================================================================
    // BF16 Softmax - AVX2 Implementation
    // ============================================================================

    inline void softmax_row_bf16_avx2(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Fallback to scalar for now (TODO: Implement BF16 AVX2 conversion)
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
    }

    // ============================================================================
    // BF16 Softmax - AVX512 Implementation
    // ============================================================================

    inline void softmax_row_bf16_avx512(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512) && defined(LLAMINAR_HAS_AVX512BF16)
        // TODO: Implement using AVX512_BF16 instructions
        // For now, fallback to scalar
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
#else
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // BF16 Softmax - Compile-Time Dispatch
    // ============================================================================

    inline void softmax_row_bf16(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512) && defined(LLAMINAR_HAS_AVX512BF16)
        softmax_row_bf16_avx512(row, cols, causal, scale, row_idx);
#elif defined(LLAMINAR_HAS_AVX2)
        softmax_row_bf16_avx2(row, cols, causal, scale, row_idx);
#else
        softmax_row_bf16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP16 Softmax - Scalar Implementation
    // ============================================================================

    inline void softmax_row_fp16_scalar(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // Pass 1: Find max (convert FP16→FP32 on the fly)
        float row_max = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
#if defined(LLAMINAR_HAS_F16C)
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
#else
            float v = detail::fp16_to_fp32_scalar(row[c]) * scale;
#endif
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
#if defined(LLAMINAR_HAS_F16C)
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
#else
            float v = detail::fp16_to_fp32_scalar(row[c]) * scale;
#endif
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize and convert FP32→FP16
        for (int c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0;
            }
            else
            {
#if defined(LLAMINAR_HAS_F16C)
                float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_fp16_f16c(result);
#else
                float v = detail::fp16_to_fp32_scalar(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_fp16_scalar(result);
#endif
            }
        }
    }

    // ============================================================================
    // FP16 Softmax - AVX2 Implementation (uses F16C)
    // ============================================================================

    inline void softmax_row_fp16_avx2(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX2) && defined(LLAMINAR_HAS_F16C)
        const __m256 scale_ps = _mm256_set1_ps(scale);
        const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

        // Pass 1: Find max (convert FP16→FP32)
        __m256 vmax = neg_inf;
        int c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + c));
            __m256 v = _mm256_cvtph_ps(fp16_vec);
            v = _mm256_mul_ps(v, scale_ps);

            if (causal)
            {
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, v);
                for (int lane = 0; lane < 8; ++lane)
                {
                    if (c + lane > row_idx)
                        tmp[lane] = -std::numeric_limits<float>::infinity();
                }
                v = _mm256_load_ps(tmp);
            }

            vmax = _mm256_max_ps(vmax, v);
        }

        float row_max = detail::hmax256(vmax);

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
            if (v > row_max)
                row_max = v;
        }

        if (!std::isfinite(row_max))
            row_max = 0.0f;

        // Pass 2: Sum of exp
        double sum = 0.0;
        for (c = 0; c < cols; ++c)
        {
            if (causal && c > row_idx)
                continue;
            float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
            sum += std::exp(v - row_max);
        }

        if (sum <= 0.0)
            sum = 1.0;

        float inv = static_cast<float>(1.0 / sum);

        // Pass 3: Normalize and convert FP32→FP16
        const __m256 inv_ps = _mm256_set1_ps(inv);
        const __m256 max_ps = _mm256_set1_ps(row_max);

        c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + c));
            __m256 v = _mm256_cvtph_ps(fp16_vec);
            v = _mm256_mul_ps(v, scale_ps);

            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, v);

            float out[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                if (causal && c + lane > row_idx)
                {
                    out[lane] = 0.0f;
                }
                else
                {
                    out[lane] = std::exp(tmp[lane] - row_max) * inv;
                }
            }

            __m256 out_ps = _mm256_load_ps(out);
            __m128i fp16_out = _mm256_cvtps_ph(out_ps, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(row + c), fp16_out);
        }

        // Tail
        for (; c < cols; ++c)
        {
            if (causal && c > row_idx)
            {
                row[c] = 0;
            }
            else
            {
                float v = detail::fp16_to_fp32_f16c(row[c]) * scale;
                float result = std::exp(v - row_max) * inv;
                row[c] = detail::fp32_to_fp16_f16c(result);
            }
        }
#else
        // Fallback to scalar if F16C not available
        softmax_row_fp16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

    // ============================================================================
    // FP16 Softmax - AVX512 Implementation
    // ============================================================================

    inline void softmax_row_fp16_avx512(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
        // TODO: Implement using AVX512FP16 if available
        // For now, fallback to AVX2 (F16C)
        softmax_row_fp16_avx2(row, cols, causal, scale, row_idx);
    }

    // ============================================================================
    // FP16 Softmax - Compile-Time Dispatch
    // ============================================================================

    inline void softmax_row_fp16(
        uint16_t *row,
        int cols,
        bool causal,
        float scale,
        int row_idx)
    {
#if defined(LLAMINAR_HAS_AVX512)
        softmax_row_fp16_avx512(row, cols, causal, scale, row_idx);
#elif defined(LLAMINAR_HAS_AVX2) && defined(LLAMINAR_HAS_F16C)
        softmax_row_fp16_avx2(row, cols, causal, scale, row_idx);
#else
        softmax_row_fp16_scalar(row, cols, causal, scale, row_idx);
#endif
    }

} // namespace llaminar2::primitives
