/**
 * @file CPUShortConvolution.cpp
 * @brief CPU implementation of causal depthwise conv1d + SiLU
 *
 * Two execution paths:
 *   Prefill (seq_len > 1): Full causal conv1d with zero-padding, stores tail in conv_state
 *   Decode  (seq_len == 1): Conv1d update using conv_state history
 *
 * Prefill is AVX-512 vectorized (16-wide) with pre-transposed weights for
 * contiguous SIMD loads across channels. Decode is AVX-512 vectorized across
 * the channel dimension.
 *
 * Reference: torch_causal_conv1d_update() and F.conv1d() in HuggingFace transformers
 */

#include "CPUShortConvolution.h"
#include "../../../utils/OpenMPUtils.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace llaminar2
{

    bool CPUShortConvolution::forward(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        if (seq_len == 1)
        {
            return executeDecode(input, weight, bias, output, conv_state,
                                 channels, kernel_size, apply_silu);
        }
        else
        {
            return executePrefill(input, weight, bias, output, conv_state,
                                  seq_len, channels, kernel_size, apply_silu);
        }
    }

#ifdef __AVX512F__
    // Fast SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) using AVX-512.
    // Uses range-reduced exp polynomial: exp(x) = 2^n * P(f) where n = round(x*log2e).
    static inline __m512 avx512_silu(__m512 vx)
    {
        const __m512 vone = _mm512_set1_ps(1.0f);
        const __m512 vlog2e = _mm512_set1_ps(1.4426950408889634f);
        const __m512 vln2 = _mm512_set1_ps(0.6931471805599453f);

        // Clamp -x to [-88, 88] to avoid overflow in exp
        __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), vx);
        neg_x = _mm512_max_ps(_mm512_set1_ps(-88.0f), _mm512_min_ps(_mm512_set1_ps(88.0f), neg_x));

        // Range reduction: exp(-x) = 2^n * 2^f
        __m512 neg_x_scaled = _mm512_mul_ps(neg_x, vlog2e);
        __m512 vn = _mm512_roundscale_ps(neg_x_scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 vf = _mm512_sub_ps(neg_x_scaled, vn);

        // Polynomial approximation of 2^f for f in [-0.5, 0.5]
        __m512 vp = _mm512_fmadd_ps(_mm512_set1_ps(0.0013333558146428f), vf, _mm512_set1_ps(0.0096181291076285f));
        vp = _mm512_fmadd_ps(vp, vf, _mm512_set1_ps(0.0555041086648216f));
        vp = _mm512_fmadd_ps(vp, vf, _mm512_set1_ps(0.2402265069591007f));
        vp = _mm512_fmadd_ps(vp, vf, vln2);
        vp = _mm512_fmadd_ps(vp, vf, vone);

        // Reconstruct 2^n via integer exponent
        __m512i vi_n = _mm512_add_epi32(_mm512_cvtps_epi32(vn), _mm512_set1_epi32(127));
        __m512 v2n = _mm512_castsi512_ps(_mm512_slli_epi32(vi_n, 23));
        __m512 vexp = _mm512_mul_ps(vp, v2n);

        // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        __m512 vsig = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
        return _mm512_mul_ps(vx, vsig);
    }
#endif

    bool CPUShortConvolution::executePrefill(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int seq_len, int channels, int kernel_size,
        bool apply_silu)
    {
        // Causal depthwise conv1d + optional SiLU for full sequence
        //
        // Input layout:  [seq_len, channels] (row-major, channels is inner dim)
        // Weight layout: [channels, kernel_size]
        // Conv state:    [channels, kernel_size - 1]
        //
        // For in-place operation (input == output), process time in reverse order
        // so that each output position only reads from not-yet-overwritten inputs.
        // Causal conv reads positions [t - state_len, ..., t], so writing from
        // t=seq_len-1 down to t=0 ensures all reads see original input values.

        const int state_len = kernel_size - 1;

#ifdef __AVX512F__
        // =====================================================================
        // AVX-512 vectorized path: process 16 channels at a time
        //
        // Pre-transpose weights from [channels, kernel_size] to [kernel_size, channels]
        // so that loading 16 consecutive channels' weights for one tap is a single
        // contiguous SIMD load (no gather needed).
        // =====================================================================

        // Weight transpose: [channels, kernel_size] → [kernel_size, channels]
        // Cost: channels * kernel_size copies (e.g., 8192 * 4 = 32K), negligible.
        std::vector<float> wt(static_cast<size_t>(kernel_size) * channels);
        for (int k = 0; k < kernel_size; ++k)
            for (int c = 0; c < channels; ++c)
                wt[k * channels + c] = weight[c * kernel_size + k];

        auto do_work = [&]()
        {
            // Save conv_state (tail of input) BEFORE any in-place overwrites.
            // Each thread saves its own channel block.
            if (conv_state)
            {
#pragma omp for schedule(static) nowait
                for (int c = 0; c < channels; ++c)
                {
                    for (int s = 0; s < state_len; ++s)
                    {
                        const int src_t = seq_len - state_len + s;
                        conv_state[c * state_len + s] =
                            (src_t >= 0) ? input[src_t * channels + c] : 0.0f;
                    }
                }
            }

            // Main convolution loop: parallelize over blocks of 16 channels.
            // Each thread processes its channel block for ALL time steps.
            // Time runs in reverse for in-place safety.
            const int n_blocks = (channels + 15) / 16;

#pragma omp for schedule(static)
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int c_start = blk * 16;
                const int c_width = (c_start + 16 <= channels) ? 16 : (channels - c_start);

                if (c_width == 16)
                {
                    // Full 16-wide SIMD path

                    // Pre-load bias vector (constant across all time steps)
                    __m512 vbias = bias
                                       ? _mm512_loadu_ps(bias + c_start)
                                       : _mm512_setzero_ps();

                    // Pre-load weight vectors for each kernel tap (constant across time)
                    // Stack array for up to 8 taps (typical: 4)
                    __m512 vw[8];
                    for (int k = 0; k < kernel_size && k < 8; ++k)
                        vw[k] = _mm512_loadu_ps(&wt[k * channels + c_start]);

                    for (int t = seq_len - 1; t >= 0; --t)
                    {
                        __m512 vsum = vbias;

                        for (int k = 0; k < kernel_size; ++k)
                        {
                            const int input_t = t - state_len + k;
                            if (input_t >= 0)
                            {
                                __m512 vin = _mm512_loadu_ps(&input[input_t * channels + c_start]);
                                vsum = _mm512_fmadd_ps(vw[k], vin, vsum);
                            }
                        }

                        if (apply_silu)
                            vsum = avx512_silu(vsum);

                        _mm512_storeu_ps(&output[t * channels + c_start], vsum);
                    }
                }
                else
                {
                    // Scalar tail for remaining channels (< 16)
                    for (int ci = 0; ci < c_width; ++ci)
                    {
                        const int c = c_start + ci;
                        const float b = bias ? bias[c] : 0.0f;

                        for (int t = seq_len - 1; t >= 0; --t)
                        {
                            float sum = b;
                            for (int k = 0; k < kernel_size; ++k)
                            {
                                const int input_t = t - state_len + k;
                                if (input_t >= 0)
                                    sum += wt[k * channels + c] * input[input_t * channels + c];
                            }
                            if (apply_silu)
                            {
                                const float sig = 1.0f / (1.0f + std::exp(-sum));
                                output[t * channels + c] = sum * sig;
                            }
                            else
                            {
                                output[t * channels + c] = sum;
                            }
                        }
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

#else
        // Scalar fallback
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int c = 0; c < channels; ++c)
            {
                const float *w = weight + c * kernel_size;
                const float b = bias ? bias[c] : 0.0f;

                if (conv_state)
                {
                    for (int s = 0; s < state_len; ++s)
                    {
                        const int src_t = seq_len - state_len + s;
                        conv_state[c * state_len + s] =
                            (src_t >= 0) ? input[src_t * channels + c] : 0.0f;
                    }
                }

                for (int t = seq_len - 1; t >= 0; --t)
                {
                    float sum = b;
                    for (int k = 0; k < kernel_size; ++k)
                    {
                        const int input_t = t - state_len + k;
                        if (input_t >= 0)
                            sum += w[k] * input[input_t * channels + c];
                    }
                    if (apply_silu)
                    {
                        const float sig = 1.0f / (1.0f + std::exp(-sum));
                        output[t * channels + c] = sum * sig;
                    }
                    else
                    {
                        output[t * channels + c] = sum;
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);
#endif

        return true;
    }

    bool CPUShortConvolution::executeDecode(
        const float *input, const float *weight, const float *bias,
        float *output, float *conv_state,
        int channels, int kernel_size,
        bool apply_silu)
    {
        if (!conv_state)
            return false;

        const int state_len = kernel_size - 1;

        // Decode: single time step per channel.
        // Scalar dot product + state shift (weight/state are per-channel contiguous,
        // no transpose needed). Batched AVX-512 SiLU replaces per-channel std::exp().
        auto do_work = [&]()
        {
#ifdef __AVX512F__
            // Process blocks of 16 channels: scalar dot + state shift, then
            // vectorized SiLU for the batch. This avoids weight transpose while
            // eliminating 16 scalar exp() calls per block.
            const int n_blocks = (channels + 15) / 16;

#pragma omp for schedule(static)
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int c_start = blk * 16;
                const int c_end = (c_start + 16 <= channels) ? c_start + 16 : channels;
                const int c_width = c_end - c_start;

                // Scalar dot product + state shift for each channel in the block
                alignas(64) float sums[16];
                for (int ci = 0; ci < c_width; ++ci)
                {
                    const int c = c_start + ci;
                    const float *w = weight + c * kernel_size;
                    float *state = conv_state + c * state_len;
                    const float b = bias ? bias[c] : 0.0f;

                    float sum = b;
                    for (int k = 0; k < state_len; ++k)
                        sum += w[k] * state[k];
                    sum += w[state_len] * input[c];

                    // Shift state left, append new input
                    for (int k = 0; k < state_len - 1; ++k)
                        state[k] = state[k + 1];
                    if (state_len > 0)
                        state[state_len - 1] = input[c];

                    sums[ci] = sum;
                }

                // Apply SiLU to the batch
                if (apply_silu && c_width == 16)
                {
                    __m512 vsum = _mm512_load_ps(sums);
                    vsum = avx512_silu(vsum);
                    _mm512_storeu_ps(&output[c_start], vsum);
                }
                else if (apply_silu)
                {
                    // Scalar tail (< 16 channels remaining)
                    for (int ci = 0; ci < c_width; ++ci)
                    {
                        const float s = sums[ci];
                        const float sig = 1.0f / (1.0f + std::exp(-s));
                        output[c_start + ci] = s * sig;
                    }
                }
                else
                {
                    for (int ci = 0; ci < c_width; ++ci)
                        output[c_start + ci] = sums[ci];
                }
            }
#else
        // Scalar fallback
#pragma omp for schedule(static)
            for (int c = 0; c < channels; ++c)
            {
                const float *w = weight + c * kernel_size;
                float *state = conv_state + c * state_len;
                const float b = bias ? bias[c] : 0.0f;

                float sum = b;
                for (int k = 0; k < state_len; ++k)
                    sum += w[k] * state[k];
                sum += w[state_len] * input[c];

                for (int k = 0; k < state_len - 1; ++k)
                    state[k] = state[k + 1];
                if (state_len > 0)
                    state[state_len - 1] = input[c];

                if (apply_silu)
                {
                    const float sig = 1.0f / (1.0f + std::exp(-sum));
                    output[c] = sum * sig;
                }
                else
                {
                    output[c] = sum;
                }
            }
#endif
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

} // namespace llaminar2
