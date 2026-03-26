/**
 * @file TurboQuantQJL.h
 * @brief 1-bit QJL residual estimator for TurboQuant inner-product quantization
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{

    struct TurboQuantProjection
    {
        int dim = 0;
        std::vector<float> matrix;
        uint64_t seed = 0;

        float at(int row, int col) const { return matrix[row * dim + col]; }
        const float *row_ptr(int r) const { return matrix.data() + r * dim; }
    };

    inline TurboQuantProjection generate_qjl_projection_matrix(
        int head_dim, uint64_t seed = 0)
    {
        if (seed == 0)
            seed = 131ULL;

        TurboQuantProjection proj;
        proj.dim = head_dim;
        proj.seed = seed;
        proj.matrix.resize(static_cast<size_t>(head_dim) * static_cast<size_t>(head_dim));

        std::mt19937_64 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float &value : proj.matrix)
            value = dist(rng);

        return proj;
    }

    inline void apply_projection(const TurboQuantProjection &proj,
                                 const float *x, float *y)
    {
        const int d = proj.dim;

#if defined(__AVX512F__)
        for (int i = 0; i < d; ++i) {
            const float *row = proj.row_ptr(i);
            __m512 acc = _mm512_setzero_ps();

            int j = 0;
            for (; j + 16 <= d; j += 16) {
                __m512 r = _mm512_loadu_ps(row + j);
                __m512 v = _mm512_loadu_ps(x + j);
                acc = _mm512_fmadd_ps(r, v, acc);
            }

            float sum = _mm512_reduce_add_ps(acc);
            for (; j < d; ++j)
                sum += row[j] * x[j];
            y[i] = sum;
        }
#elif defined(__AVX2__)
        for (int i = 0; i < d; ++i) {
            const float *row = proj.row_ptr(i);
            __m256 acc = _mm256_setzero_ps();

            int j = 0;
            for (; j + 8 <= d; j += 8) {
                __m256 r = _mm256_loadu_ps(row + j);
                __m256 v = _mm256_loadu_ps(x + j);
                acc = _mm256_fmadd_ps(r, v, acc);
            }

            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            for (; j < d; ++j)
                sum += row[j] * x[j];
            y[i] = sum;
        }
#else
        for (int i = 0; i < d; ++i) {
            const float *row = proj.row_ptr(i);
            float sum = 0.0f;
            for (int j = 0; j < d; ++j)
                sum += row[j] * x[j];
            y[i] = sum;
        }
#endif
    }

    inline void tq_pack_signs_8(const float *values, uint8_t *out)
    {
        uint8_t packed = 0;
        for (int i = 0; i < 8; ++i) {
            if (values[i] >= 0.0f)
                packed |= static_cast<uint8_t>(1u << i);
        }
        *out = packed;
    }

    inline void tq_unpack_signs_8(const uint8_t *packed, float *out)
    {
        const uint8_t bits = *packed;
        for (int i = 0; i < 8; ++i)
            out[i] = (bits & static_cast<uint8_t>(1u << i)) ? 1.0f : -1.0f;
    }

    template <int D>
    inline void qjl_quantize_signs(const TurboQuantProjection &proj,
                                   const float *input,
                                   uint8_t *out_packed_signs,
                                   float *scratch)
    {
        static_assert(D > 0 && D % 8 == 0, "QJL sign packing requires dimensions divisible by 8");

        apply_projection(proj, input, scratch);
        for (int i = 0; i < D; i += 8)
            tq_pack_signs_8(scratch + i, out_packed_signs + (i / 8));
    }

    template <int D>
    inline void qjl_dequantize_unit(const TurboQuantProjection &proj,
                                    const uint8_t *packed_signs,
                                    float *output)
    {
        static_assert(D > 0 && D % 8 == 0, "QJL sign packing requires dimensions divisible by 8");

        for (int j = 0; j < D; ++j)
            output[j] = 0.0f;

        alignas(64) float signs[D];
        for (int i = 0; i < D; i += 8)
            tq_unpack_signs_8(packed_signs + (i / 8), signs + i);

#if defined(__AVX512F__)
        for (int i = 0; i < D; ++i) {
            const float *row = proj.row_ptr(i);
            const __m512 si = _mm512_set1_ps(signs[i]);

            int j = 0;
            for (; j + 16 <= D; j += 16) {
                const __m512 r = _mm512_loadu_ps(row + j);
                __m512 out = _mm512_loadu_ps(output + j);
                out = _mm512_fmadd_ps(si, r, out);
                _mm512_storeu_ps(output + j, out);
            }
            for (; j < D; ++j)
                output[j] += signs[i] * row[j];
        }
#elif defined(__AVX2__)
        for (int i = 0; i < D; ++i) {
            const float *row = proj.row_ptr(i);
            const __m256 si = _mm256_set1_ps(signs[i]);

            int j = 0;
            for (; j + 8 <= D; j += 8) {
                const __m256 r = _mm256_loadu_ps(row + j);
                __m256 out = _mm256_loadu_ps(output + j);
                out = _mm256_fmadd_ps(si, r, out);
                _mm256_storeu_ps(output + j, out);
            }
            for (; j < D; ++j)
                output[j] += signs[i] * row[j];
        }
#else
        for (int i = 0; i < D; ++i) {
            const float *row = proj.row_ptr(i);
            for (int j = 0; j < D; ++j)
                output[j] += signs[i] * row[j];
        }
#endif

        constexpr float kSqrtPiOverTwo = 1.2533141373155001f;
        const float scale = kSqrtPiOverTwo / static_cast<float>(D);
        for (int j = 0; j < D; ++j)
            output[j] *= scale;
    }

} // namespace llaminar2