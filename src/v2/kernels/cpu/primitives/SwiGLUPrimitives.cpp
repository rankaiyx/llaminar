/**
 * @file SwiGLUPrimitives.cpp
 * @brief SwiGLU primitives implementation
 * @author GitHub Copilot
 */

#include "SwiGLUPrimitives.h"
#include <cmath>
#include <omp.h>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

// Libmvec declarations for vectorized exp
#if defined(__GLIBC__)
extern "C"
{
#if defined(__AVX512F__)
    __m512 _ZGVeN16v_expf(__m512) __attribute__((weak));
#endif
#if defined(__AVX2__)
    __m256 _ZGVcN8v_expf(__m256) __attribute__((weak));
#endif
}
#endif

namespace llaminar2::primitives
{

    // Scalar implementation
    inline float silu_scalar(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

#if defined(__AVX2__)
    // Fast exp approximation (AVX2)
    // Based on Schraudolph/Polynomial approximation
    static inline __m256 fast_exp256(__m256 x)
    {
        __m256 log2e = _mm256_set1_ps(1.4426950408f);
        __m256 ln2 = _mm256_set1_ps(0.69314718056f);
        __m256 max_input = _mm256_set1_ps(88.0f);
        __m256 min_input = _mm256_set1_ps(-88.0f);

        // Clamp input to avoid overflow/underflow
        x = _mm256_min_ps(x, max_input);
        x = _mm256_max_ps(x, min_input);

        // k = round(x * log2(e))
        __m256 k_f = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i k_i = _mm256_cvtps_epi32(k_f);

        // r = x - k * ln2
        __m256 r = _mm256_fnmadd_ps(k_f, ln2, x); // x - k*ln2

        // Polynomial approximation for e^r on [-0.5, 0.5]
        // p(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
        __m256 c2 = _mm256_set1_ps(0.5f);
        __m256 c3 = _mm256_set1_ps(0.166666667f);
        __m256 c4 = _mm256_set1_ps(0.041666667f);
        __m256 c5 = _mm256_set1_ps(0.008333333f);
        __m256 one = _mm256_set1_ps(1.0f);

        __m256 p = _mm256_fmadd_ps(c5, r, c4);
        p = _mm256_fmadd_ps(p, r, c3);
        p = _mm256_fmadd_ps(p, r, c2);
        p = _mm256_fmadd_ps(p, r, one);
        p = _mm256_fmadd_ps(p, r, one);

        // 2^k
        __m256i bias = _mm256_set1_epi32(127);
        __m256i k_biased = _mm256_add_epi32(k_i, bias);
        __m256i k_shifted = _mm256_slli_epi32(k_biased, 23);
        __m256 two_k = _mm256_castsi256_ps(k_shifted);

        return _mm256_mul_ps(p, two_k);
    }

    static void compute_swiglu_avx2(const float *gate, const float *up, float *output, int size)
    {
        int i = 0;
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 zero = _mm256_setzero_ps();
        __m256 two = _mm256_set1_ps(2.0f);

        for (; i + 8 <= size; i += 8)
        {
            __m256 g = _mm256_loadu_ps(gate + i);
            __m256 u = _mm256_loadu_ps(up + i);

            // silu(u) = u / (1 + exp(-u))
            __m256 neg_u = _mm256_sub_ps(zero, u);
            __m256 exp_neg_u = fast_exp256(neg_u);
            __m256 denom = _mm256_add_ps(one, exp_neg_u);

            // Fast reciprocal with Newton-Raphson
            __m256 rcp = _mm256_rcp_ps(denom);
            __m256 term = _mm256_fnmadd_ps(denom, rcp, two); // 2 - d*rcp
            __m256 sigmoid_u = _mm256_mul_ps(rcp, term);

            __m256 silu_u = _mm256_mul_ps(u, sigmoid_u);

            // output = g * silu(u)
            __m256 out = _mm256_mul_ps(g, silu_u);
            _mm256_storeu_ps(output + i, out);
        }

        // Tail
        for (; i < size; ++i)
        {
            float g = gate[i];
            float u = up[i];
            output[i] = g * silu_scalar(u);
        }
    }
#endif

#if defined(__AVX512F__)
    static inline __m512 fast_exp512(__m512 x)
    {
        __m512 log2e = _mm512_set1_ps(1.4426950408f);
        __m512 ln2 = _mm512_set1_ps(0.69314718056f);
        __m512 max_input = _mm512_set1_ps(88.0f);
        __m512 min_input = _mm512_set1_ps(-88.0f);

        x = _mm512_min_ps(x, max_input);
        x = _mm512_max_ps(x, min_input);

        __m512 k_f = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512i k_i = _mm512_cvtps_epi32(k_f);

        __m512 r = _mm512_fnmadd_ps(k_f, ln2, x);

        __m512 c2 = _mm512_set1_ps(0.5f);
        __m512 c3 = _mm512_set1_ps(0.166666667f);
        __m512 c4 = _mm512_set1_ps(0.041666667f);
        __m512 c5 = _mm512_set1_ps(0.008333333f);
        __m512 one = _mm512_set1_ps(1.0f);

        __m512 p = _mm512_fmadd_ps(c5, r, c4);
        p = _mm512_fmadd_ps(p, r, c3);
        p = _mm512_fmadd_ps(p, r, c2);
        p = _mm512_fmadd_ps(p, r, one);
        p = _mm512_fmadd_ps(p, r, one);

        __m512i bias = _mm512_set1_epi32(127);
        __m512i k_biased = _mm512_add_epi32(k_i, bias);
        __m512i k_shifted = _mm512_slli_epi32(k_biased, 23);
        __m512 two_k = _mm512_castsi512_ps(k_shifted);

        return _mm512_mul_ps(p, two_k);
    }

    static void compute_swiglu_avx512(const float *gate, const float *up, float *output, int size)
    {
        int i = 0;
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_setzero_ps();
        __m512 two = _mm512_set1_ps(2.0f);

        for (; i + 16 <= size; i += 16)
        {
            __m512 g = _mm512_loadu_ps(gate + i);
            __m512 u = _mm512_loadu_ps(up + i);

            __m512 neg_u = _mm512_sub_ps(zero, u);
            __m512 exp_neg_u = fast_exp512(neg_u);
            __m512 denom = _mm512_add_ps(one, exp_neg_u);

            __m512 rcp = _mm512_rcp14_ps(denom); // rcp14 is more accurate than rcp
            // Newton-Raphson for better precision (optional, rcp14 might be enough)
            // x1 = x0 * (2 - d * x0)
            __m512 term = _mm512_fnmadd_ps(denom, rcp, two);
            __m512 sigmoid_u = _mm512_mul_ps(rcp, term);

            __m512 silu_u = _mm512_mul_ps(u, sigmoid_u);

            __m512 out = _mm512_mul_ps(g, silu_u);
            _mm512_storeu_ps(output + i, out);
        }

        for (; i < size; ++i)
        {
            float g = gate[i];
            float u = up[i];
            output[i] = g * silu_scalar(u);
        }
    }
#endif

    void compute_swiglu(const float *gate, const float *up, float *output, int size)
    {
        // OpenMP parallelization
        // Use a chunk size that fits in L1/L2 cache and provides enough work
        const int chunk_size = 1024;

#pragma omp parallel for schedule(static)
        for (int i = 0; i < size; i += chunk_size)
        {
            int current_chunk = std::min(chunk_size, size - i);
            const float *g_ptr = gate + i;
            const float *u_ptr = up + i;
            float *o_ptr = output + i;

#if defined(__AVX512F__)
            compute_swiglu_avx512(g_ptr, u_ptr, o_ptr, current_chunk);
#elif defined(__AVX2__)
            compute_swiglu_avx2(g_ptr, u_ptr, o_ptr, current_chunk);
#else
            for (int j = 0; j < current_chunk; ++j)
            {
                o_ptr[j] = g_ptr[j] * silu_scalar(u_ptr[j]);
            }
#endif
        }
    }

} // namespace llaminar2::primitives
