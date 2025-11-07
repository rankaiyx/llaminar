/**
 * @file RMSNormPrimitives.cpp
 * @brief Vectorized RMSNorm implementation (ported from V1)
 * @author David Sanftenberg
 */

#include "RMSNormPrimitives.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives
{

    namespace
    {
        inline bool want_parallel(std::size_t rows, std::size_t cols, const RMSNormExecOptions &opts)
        {
            if (opts.force_scalar || !opts.allow_parallel)
                return false;

            std::size_t elems = rows * cols;

            // Single-row decode for large models (d_model >= 2048)
            if (rows <= 1 && cols < 2048)
                return false;

#ifdef _OPENMP
            if (omp_in_parallel())
                return false;
#endif

            return elems >= opts.parallel_threshold_elems;
        }
    } // anonymous namespace

    void rmsnorm_compute_row_sumsq_vectorized(
        const float *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts)
    {
        if (!src || !row_sumsq || rows == 0 || cols == 0)
            return;

        // T5 compatibility: float32 accumulation
        if (opts.t5_compat_mode)
        {
            bool parallel = want_parallel(rows, cols, opts);
#pragma omp parallel for if (parallel)
            for (long long r = 0; r < (long long)rows; ++r)
            {
                const float *row = src + (std::size_t)r * cols;
                float sum_sq = 0.0f;
                for (std::size_t c = 0; c < cols; ++c)
                {
                    float val = row[c];
                    sum_sq += val * val;
                }
                row_sumsq[r] = (double)sum_sq;
            }
            return;
        }

        // Double precision vectorized path
        bool parallel = want_parallel(rows, cols, opts);

#pragma omp parallel for if (parallel)
        for (long long r = 0; r < (long long)rows; ++r)
        {
            const float *row = src + (std::size_t)r * cols;
            double acc = 0.0;

#if defined(__AVX512F__)
            // Multi-accumulator AVX512: process 64 elements per iteration
            long long c = 0;
            __m512d dacc0 = _mm512_setzero_pd();
            __m512d dacc1 = _mm512_setzero_pd();
            __m512d dacc2 = _mm512_setzero_pd();
            __m512d dacc3 = _mm512_setzero_pd();

            for (; c + 64 <= (long long)cols; c += 64)
            {
                const float *base = row + c;

                // Load 4x 16-float blocks
                __m512 v0 = _mm512_loadu_ps(base + 0);
                __m512 v1 = _mm512_loadu_ps(base + 16);
                __m512 v2 = _mm512_loadu_ps(base + 32);
                __m512 v3 = _mm512_loadu_ps(base + 48);

                // Convert to double (split each 16-float into 2x 8-double)
                __m256 lo0 = _mm512_castps512_ps256(v0);
                __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v0), 1));
                __m256 lo1 = _mm512_castps512_ps256(v1);
                __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v1), 1));
                __m256 lo2 = _mm512_castps512_ps256(v2);
                __m256 hi2 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v2), 1));
                __m256 lo3 = _mm512_castps512_ps256(v3);
                __m256 hi3 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v3), 1));

                __m512d d0a = _mm512_cvtps_pd(lo0);
                __m512d d0b = _mm512_cvtps_pd(hi0);
                __m512d d1a = _mm512_cvtps_pd(lo1);
                __m512d d1b = _mm512_cvtps_pd(hi1);
                __m512d d2a = _mm512_cvtps_pd(lo2);
                __m512d d2b = _mm512_cvtps_pd(hi2);
                __m512d d3a = _mm512_cvtps_pd(lo3);
                __m512d d3b = _mm512_cvtps_pd(hi3);

                // FMA: acc += d * d
                dacc0 = _mm512_fmadd_pd(d0a, d0a, dacc0);
                dacc0 = _mm512_fmadd_pd(d0b, d0b, dacc0);
                dacc1 = _mm512_fmadd_pd(d1a, d1a, dacc1);
                dacc1 = _mm512_fmadd_pd(d1b, d1b, dacc1);
                dacc2 = _mm512_fmadd_pd(d2a, d2a, dacc2);
                dacc2 = _mm512_fmadd_pd(d2b, d2b, dacc2);
                dacc3 = _mm512_fmadd_pd(d3a, d3a, dacc3);
                dacc3 = _mm512_fmadd_pd(d3b, d3b, dacc3);
            }

            // Horizontal reduction
            acc += _mm512_reduce_add_pd(dacc0);
            acc += _mm512_reduce_add_pd(dacc1);
            acc += _mm512_reduce_add_pd(dacc2);
            acc += _mm512_reduce_add_pd(dacc3);

            // Tail: 16-float blocks
            for (; c + 16 <= (long long)cols; c += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c);
                __m256 vlo = _mm512_castps512_ps256(v);
                __m256 vhi = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v), 1));

                __m512d d0 = _mm512_cvtps_pd(vlo);
                __m512d d1 = _mm512_cvtps_pd(vhi);

                __m512d sq0 = _mm512_mul_pd(d0, d0);
                __m512d sq1 = _mm512_mul_pd(d1, d1);

                acc += _mm512_reduce_add_pd(sq0);
                acc += _mm512_reduce_add_pd(sq1);
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }

#elif defined(__AVX2__)
            // AVX2 path: 4-accumulator with 32 elements per iteration
            long long c = 0;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (; c + 32 <= (long long)cols; c += 32)
            {
                const float *base = row + c;
                __m256 v0 = _mm256_loadu_ps(base + 0);
                __m256 v1 = _mm256_loadu_ps(base + 8);
                __m256 v2 = _mm256_loadu_ps(base + 16);
                __m256 v3 = _mm256_loadu_ps(base + 24);

                acc0 = _mm256_fmadd_ps(v0, v0, acc0);
                acc1 = _mm256_fmadd_ps(v1, v1, acc1);
                acc2 = _mm256_fmadd_ps(v2, v2, acc2);
                acc3 = _mm256_fmadd_ps(v3, v3, acc3);
            }

            // Combine accumulators
            __m256 acc01 = _mm256_add_ps(acc0, acc1);
            __m256 acc23 = _mm256_add_ps(acc2, acc3);
            __m256 acc_all = _mm256_add_ps(acc01, acc23);

            // Horizontal reduction
            __m128 lo = _mm256_castps256_ps128(acc_all);
            __m128 hi = _mm256_extractf128_ps(acc_all, 1);
            __m128 sum2 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum2);
            __m128 sums = _mm_add_ps(sum2, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);

            float partial;
            _mm_store_ss(&partial, sums);
            acc += (double)partial;

            // 8-element blocks
            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c);
                __m256 sq = _mm256_mul_ps(v, v);

                __m128 lo2 = _mm256_castps256_ps128(sq);
                __m128 hi2 = _mm256_extractf128_ps(sq, 1);
                __m128 sumv = _mm_add_ps(lo2, hi2);
                __m128 sh2 = _mm_movehdup_ps(sumv);
                __m128 sumv2 = _mm_add_ps(sumv, sh2);
                sh2 = _mm_movehl_ps(sh2, sumv2);
                sumv2 = _mm_add_ss(sumv2, sh2);

                float p;
                _mm_store_ss(&p, sumv2);
                acc += (double)p;
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }

#else
            // Scalar fallback
            double s_scalar = 0.0;
#pragma omp simd reduction(+ : s_scalar)
            for (long long c = 0; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                s_scalar += v * v;
            }
            acc = s_scalar;
#endif

            row_sumsq[r] = acc;
        }
    }

    void rmsnorm_compute_inv(
        const double *row_sumsq,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        float *inv_out)
    {
        if (!row_sumsq || !inv_out || rows == 0)
            return;

        double denom = (double)cols;
        if (denom <= 0.0)
            denom = 1.0;

        for (std::size_t r = 0; r < rows; ++r)
        {
            double val = row_sumsq[r] / denom + (double)epsilon;
            double inv = (val > 0.0) ? 1.0 / std::sqrt(val) : 0.0;
            inv_out[r] = (float)inv;
        }
    }

    void rmsnorm_apply_vectorized(
        const float *src,
        const float *gamma,
        const float *inv,
        std::size_t rows,
        std::size_t cols,
        float *dst,
        const RMSNormExecOptions &opts)
    {
        if (!src || !inv || !dst || rows == 0 || cols == 0)
            return;

        bool parallel = want_parallel(rows, cols, opts);
        bool has_gamma = (gamma != nullptr);

#pragma omp parallel for if (parallel)
        for (long long r = 0; r < (long long)rows; ++r)
        {
            const float *row_in = src + (std::size_t)r * cols;
            float *row_out = dst + (std::size_t)r * cols;
            float scale = inv[(std::size_t)r];

            if (scale == 0.0f)
            {
                std::fill(row_out, row_out + cols, 0.0f);
                continue;
            }

            if (!has_gamma)
            {
#pragma omp simd
                for (long long c = 0; c < (long long)cols; ++c)
                {
                    row_out[c] = row_in[c] * scale;
                }
            }
            else
            {
#pragma omp simd
                for (long long c = 0; c < (long long)cols; ++c)
                {
                    row_out[c] = row_in[c] * scale * gamma[c];
                }
            }
        }
    }

    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        thread_local RMSNormScratch tls_scratch;
        tls_scratch.ensure(rows);

        rmsnorm_compute_row_sumsq_vectorized(src, rows, cols, tls_scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(tls_scratch.row_sumsq.data(), rows, cols, epsilon, tls_scratch.inv.data());
        rmsnorm_apply_vectorized(src, gamma, tls_scratch.inv.data(), rows, cols, dst, opts);
    }

    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        RMSNormScratch &scratch,
        const RMSNormExecOptions &opts)
    {
        scratch.ensure(rows);
        rmsnorm_compute_row_sumsq_vectorized(src, rows, cols, scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(scratch.row_sumsq.data(), rows, cols, epsilon, scratch.inv.data());
        rmsnorm_apply_vectorized(src, gamma, scratch.inv.data(), rows, cols, dst, opts);
    }

    // ========================================================================
    // INT32 RMSNorm Implementation (for full INT8 pipelines)
    // ========================================================================

    void rmsnorm_compute_row_sumsq_int32_vectorized(
        const int32_t *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts)
    {
        if (!src || !row_sumsq || rows == 0 || cols == 0)
            return;

        bool parallel = want_parallel(rows, cols, opts);

#pragma omp parallel for if (parallel)
        for (long long r = 0; r < (long long)rows; ++r)
        {
            const int32_t *row = src + (std::size_t)r * cols;
            double acc = 0.0;

#if defined(__AVX512F__)
            // AVX512: Process 16 INT32 values per iteration (double accumulation)
            long long c = 0;
            __m512d dacc0 = _mm512_setzero_pd();
            __m512d dacc1 = _mm512_setzero_pd();

            for (; c + 16 <= (long long)cols; c += 16)
            {
                // Load 16 INT32 values
                __m512i i32_lo = _mm512_loadu_si512((__m512i *)(row + c)); // 0-15

                // Convert INT32 to double (8 at a time)
                __m256i i32_lo_256 = _mm512_castsi512_si256(i32_lo);
                __m256i i32_hi_256 = _mm512_extracti64x4_epi64(i32_lo, 1);

                __m512d d0 = _mm512_cvtepi32_pd(i32_lo_256); // 0-7
                __m512d d1 = _mm512_cvtepi32_pd(i32_hi_256); // 8-15

                // FMA: acc += d * d
                dacc0 = _mm512_fmadd_pd(d0, d0, dacc0);
                dacc1 = _mm512_fmadd_pd(d1, d1, dacc1);
            }

            // Horizontal reduction
            acc += _mm512_reduce_add_pd(dacc0);
            acc += _mm512_reduce_add_pd(dacc1);

            // Tail: 8-int32 blocks
            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256i i32 = _mm256_loadu_si256((__m256i *)(row + c));
                __m512d d = _mm512_cvtepi32_pd(i32);
                __m512d sq = _mm512_mul_pd(d, d);
                acc += _mm512_reduce_add_pd(sq);
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }

#elif defined(__AVX2__)
            // AVX2: Process 16 INT32 values per iteration (FP32 accumulation, then convert to double)
            long long c = 0;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (; c + 16 <= (long long)cols; c += 16)
            {
                // Load 16 INT32 values
                __m256i i32_0 = _mm256_loadu_si256((__m256i *)(row + c));
                __m256i i32_1 = _mm256_loadu_si256((__m256i *)(row + c + 8));

                // Convert INT32 to FP32
                __m256 f0 = _mm256_cvtepi32_ps(i32_0);
                __m256 f1 = _mm256_cvtepi32_ps(i32_1);

                // FMA: acc += f * f
                acc0 = _mm256_fmadd_ps(f0, f0, acc0);
                acc1 = _mm256_fmadd_ps(f1, f1, acc1);
            }

            // Combine accumulators and reduce to scalar
            __m256 acc_all = _mm256_add_ps(acc0, acc1);

            // Horizontal reduction
            __m128 lo = _mm256_castps256_ps128(acc_all);
            __m128 hi = _mm256_extractf128_ps(acc_all, 1);
            __m128 sum2 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum2);
            __m128 sums = _mm_add_ps(sum2, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);

            float partial;
            _mm_store_ss(&partial, sums);
            acc += (double)partial;

            // Tail: 8-element blocks
            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256i i32 = _mm256_loadu_si256((__m256i *)(row + c));
                __m256 f = _mm256_cvtepi32_ps(i32);
                __m256 sq = _mm256_mul_ps(f, f);

                __m128 lo2 = _mm256_castps256_ps128(sq);
                __m128 hi2 = _mm256_extractf128_ps(sq, 1);
                __m128 sumv = _mm_add_ps(lo2, hi2);
                __m128 sh2 = _mm_movehdup_ps(sumv);
                __m128 sumv2 = _mm_add_ps(sumv, sh2);
                sh2 = _mm_movehl_ps(sh2, sumv2);
                sumv2 = _mm_add_ss(sumv2, sh2);

                float p;
                _mm_store_ss(&p, sumv2);
                acc += (double)p;
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }

#else
            // Scalar fallback
            double s_scalar = 0.0;
#pragma omp simd reduction(+ : s_scalar)
            for (long long c = 0; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                s_scalar += v * v;
            }
            acc = s_scalar;
#endif

            row_sumsq[r] = acc;
        }
    }

    void rmsnorm_apply_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        const float *inv,
        std::size_t rows,
        std::size_t cols,
        int8_t *dst_int8,
        float *dst_row_scales,
        const RMSNormExecOptions &opts)
    {
        if (!src || !inv || !dst_int8 || !dst_row_scales || rows == 0 || cols == 0)
            return;

        bool parallel = want_parallel(rows, cols, opts);
        bool has_gamma = (gamma != nullptr);

#pragma omp parallel for if (parallel)
        for (long long r = 0; r < (long long)rows; ++r)
        {
            const int32_t *row_in = src + (std::size_t)r * cols;
            int8_t *row_out = dst_int8 + (std::size_t)r * cols;
            float rms_inv = inv[(std::size_t)r];

            if (rms_inv == 0.0f)
            {
                std::fill(row_out, row_out + cols, (int8_t)0);
                dst_row_scales[r] = 1.0f;
                continue;
            }

            // Step 1: Normalize INT32 to FP32 and apply gamma
            // We'll use thread-local buffer to avoid allocation
            thread_local std::vector<float> fp32_buffer;
            if (fp32_buffer.size() < cols)
                fp32_buffer.resize(cols);

            if (!has_gamma)
            {
#pragma omp simd
                for (long long c = 0; c < (long long)cols; ++c)
                {
                    fp32_buffer[c] = (float)row_in[c] * rms_inv;
                }
            }
            else
            {
#pragma omp simd
                for (long long c = 0; c < (long long)cols; ++c)
                {
                    fp32_buffer[c] = (float)row_in[c] * rms_inv * gamma[c];
                }
            }

            // Step 2: Requantize FP32 to INT8 with per-row dynamic scaling
            // Find max absolute value
            float max_abs = 0.0f;
            for (std::size_t c = 0; c < cols; ++c)
            {
                max_abs = std::max(max_abs, std::fabs(fp32_buffer[c]));
            }

            // Compute scale to fit into INT8 range [-127, 127]
            float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            dst_row_scales[r] = scale;

            // Quantize to INT8
            float inv_scale = 1.0f / scale;
#pragma omp simd
            for (long long c = 0; c < (long long)cols; ++c)
            {
                float scaled = fp32_buffer[c] * inv_scale;
                // Clamp to [-127, 127] and round
                scaled = std::max(-127.0f, std::min(127.0f, scaled));
                row_out[c] = (int8_t)std::round(scaled);
            }
        }
    }

    void rmsnorm_fused_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        int8_t *dst_int8,
        float *dst_row_scales,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        thread_local RMSNormScratch tls_scratch;
        tls_scratch.ensure(rows);

        rmsnorm_compute_row_sumsq_int32_vectorized(src, rows, cols, tls_scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(tls_scratch.row_sumsq.data(), rows, cols, epsilon, tls_scratch.inv.data());
        rmsnorm_apply_int32_to_int8_vectorized(src, gamma, tls_scratch.inv.data(), rows, cols, dst_int8, dst_row_scales, opts);
    }

    void rmsnorm_fused_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        int8_t *dst_int8,
        float *dst_row_scales,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        RMSNormScratch &scratch,
        const RMSNormExecOptions &opts)
    {
        scratch.ensure(rows);
        rmsnorm_compute_row_sumsq_int32_vectorized(src, rows, cols, scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(scratch.row_sumsq.data(), rows, cols, epsilon, scratch.inv.data());
        rmsnorm_apply_int32_to_int8_vectorized(src, gamma, scratch.inv.data(), rows, cols, dst_int8, dst_row_scales, opts);
    }

} // namespace llaminar2::primitives
