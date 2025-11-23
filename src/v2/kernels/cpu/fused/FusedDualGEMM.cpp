/**
 * @file FusedDualGEMM.cpp
 * @brief Implementation of fused dual GEMM for FFN gate/up projections
 * @author David Sanftenberg
 * @date 2025-11-23
 */

#include "FusedDualGEMM.h"
#include "../gemm_v4/OneDNNGemmKernel.h"
#include "../gemm_v4/OneDNNGemmAdapter.h"
#include "../../../utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace llaminar2
{
    FusedDualGEMM::FusedDualGEMM(TensorBase *gate_weight, TensorBase *up_weight)
        : gate_weight_(gate_weight), up_weight_(up_weight)
    {
        if (!gate_weight_ || !up_weight_)
        {
            throw std::invalid_argument("FusedDualGEMM: Weight tensors cannot be null");
        }

        // Validate that weights have the same dimensions
        const auto &gate_shape = gate_weight_->shape();
        const auto &up_shape = up_weight_->shape();
        if (gate_shape.size() != 2 || up_shape.size() != 2 ||
            gate_shape[0] != up_shape[0] || gate_shape[1] != up_shape[1])
        {
            throw std::invalid_argument("[FusedDualGEMM] Gate and Up weights must have same 2D shape");
        }
    }

    bool FusedDualGEMM::execute(
        const float *input,
        int32_t *gate_output,
        int32_t *up_output,
        float *activation_scales,
        int m, int n, int k)
    {
        if (!input || !gate_output || !up_output || !activation_scales)
        {
            LOG_ERROR("[FusedDualGEMM] Null pointer in execute()");
            return false;
        }

        if (m <= 0 || n <= 0 || k <= 0)
        {
            LOG_ERROR("[FusedDualGEMM] Invalid dimensions: m=" << m << " n=" << n << " k=" << k);
            return false;
        }

        // =====================================================================
        // Step 1: Quantize input activations ONCE (FP32 → INT8)
        // =====================================================================

        // Ensure int8_activations_ buffer is allocated
        const size_t activation_size = static_cast<size_t>(m) * static_cast<size_t>(k);
        if (int8_activations_.size() < activation_size)
        {
            int8_activations_.resize(activation_size);
        }

        quantize_per_row(input, int8_activations_.data(), activation_scales, m, k);

        // =====================================================================
        // Step 2: Execute gate projection GEMM (INT8×INT8 → INT32)
        // =====================================================================

        if (!execute_int8_gemm(int8_activations_.data(), gate_weight_, gate_output, m, n, k))
        {
            LOG_ERROR("[FusedDualGEMM] Gate GEMM failed");
            return false;
        }

        // =====================================================================
        // Step 3: Execute up projection GEMM (INT8×INT8 → INT32)
        // =====================================================================

        if (!execute_int8_gemm(int8_activations_.data(), up_weight_, up_output, m, n, k))
        {
            LOG_ERROR("[FusedDualGEMM] Up GEMM failed");
            return false;
        }

        return true;
    }

    void FusedDualGEMM::quantize_per_row(
        const float *input,
        int8_t *output,
        float *row_scales,
        int m, int k)
    {
        // Per-row quantization: find max absolute value per row, then quantize
        // Formula: scale[i] = max(|input[i,:]|) / 127.0
        //          output[i,j] = round(input[i,j] / scale[i])

        constexpr float MAX_INT8 = 127.0f;

        for (int i = 0; i < m; ++i)
        {
            const float *input_row = input + i * k;
            int8_t *output_row = output + i * k;

            // Find max absolute value in row
            float max_abs = 0.0f;
            int j = 0; // Declare once for all code paths

#ifdef __AVX512F__
            // AVX512: 16-way vectorized max reduction
            __m512 vmax = _mm512_setzero_ps();
            for (; j + 15 < k; j += 16)
            {
                __m512 v = _mm512_loadu_ps(input_row + j);
                __m512 vabs = _mm512_abs_ps(v);
                vmax = _mm512_max_ps(vmax, vabs);
            }

            // Reduce 16 lanes to scalar
            float vmax_arr[16];
            _mm512_storeu_ps(vmax_arr, vmax);
            for (int l = 0; l < 16; ++l)
            {
                max_abs = std::max(max_abs, vmax_arr[l]);
            }

            // Process tail
            for (; j < k; ++j)
            {
                max_abs = std::max(max_abs, std::abs(input_row[j]));
            }
#elif defined(__AVX2__)
            // AVX2: 8-way vectorized max reduction
            __m256 vmax = _mm256_setzero_ps();
            for (; j + 7 < k; j += 8)
            {
                __m256 v = _mm256_loadu_ps(input_row + j);
                __m256 vabs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v); // abs
                vmax = _mm256_max_ps(vmax, vabs);
            }

            // Reduce 8 lanes to scalar
            float vmax_arr[8];
            _mm256_storeu_ps(vmax_arr, vmax);
            for (int l = 0; l < 8; ++l)
            {
                max_abs = std::max(max_abs, vmax_arr[l]);
            }

            // Process tail
            for (; j < k; ++j)
            {
                max_abs = std::max(max_abs, std::abs(input_row[j]));
            }
#else
            // Scalar fallback
            for (j = 0; j < k; ++j)
            {
                max_abs = std::max(max_abs, std::abs(input_row[j]));
            }
#endif

            // Compute scale (avoid division by zero)
            const float scale = (max_abs > 1e-8f) ? (max_abs / MAX_INT8) : 1.0f;
            row_scales[i] = scale;
            const float inv_scale = 1.0f / scale;

            // Quantize row: output[j] = round(input[j] / scale)
            j = 0; // Reset for quantization phase
            j = 0; // Reset for quantization phase

#ifdef __AVX512F__
            // AVX512: 16-way vectorized quantization
            __m512 vinv_scale = _mm512_set1_ps(inv_scale);
            for (; j + 15 < k; j += 16)
            {
                __m512 v = _mm512_loadu_ps(input_row + j);
                __m512 vscaled = _mm512_mul_ps(v, vinv_scale);
                __m512i vint32 = _mm512_cvtps_epi32(vscaled); // round + convert to int32

                // Convert int32 → int16 → int8 (with saturation)
                __m256i vint16 = _mm512_cvtsepi32_epi16(vint32);
                __m128i vint8 = _mm256_cvtsepi16_epi8(vint16);
                _mm_storeu_si128((__m128i *)(output_row + j), vint8);
            }

            // Process tail
            for (; j < k; ++j)
            {
                float scaled = input_row[j] * inv_scale;
                int32_t quantized = static_cast<int32_t>(std::roundf(scaled));
                output_row[j] = static_cast<int8_t>(std::max(-127, std::min(127, quantized)));
            }
#elif defined(__AVX2__)
            // AVX2: 8-way vectorized quantization
            __m256 vinv_scale = _mm256_set1_ps(inv_scale);
            for (; j + 7 < k; j += 8)
            {
                __m256 v = _mm256_loadu_ps(input_row + j);
                __m256 vscaled = _mm256_mul_ps(v, vinv_scale);
                __m256i vint32 = _mm256_cvtps_epi32(vscaled); // round + convert to int32

                // Convert int32 → int16 (with saturation)
                // AVX2 doesn't have direct int32→int8, so go through int16
                __m128i vint16_lo = _mm256_castsi256_si128(vint32);
                __m128i vint16_hi = _mm256_extracti128_si256(vint32, 1);
                __m128i vint16 = _mm_packs_epi32(vint16_lo, vint16_hi);

                // Convert int16 → int8 (with saturation)
                __m128i vint8 = _mm_packs_epi16(vint16, vint16);
                _mm_storel_epi64((__m128i *)(output_row + j), vint8);
            }

            // Process tail
            for (; j < k; ++j)
            {
                float scaled = input_row[j] * inv_scale;
                int32_t quantized = static_cast<int32_t>(std::roundf(scaled));
                output_row[j] = static_cast<int8_t>(std::max(-127, std::min(127, quantized)));
            }
#else
            // Scalar fallback
            for (j = 0; j < k; ++j)
            {
                float scaled = input_row[j] * inv_scale;
                int32_t quantized = static_cast<int32_t>(std::roundf(scaled));
                output_row[j] = static_cast<int8_t>(std::max(-127, std::min(127, quantized)));
            }
#endif
        }
    }

    bool FusedDualGEMM::execute_int8_gemm(
        const int8_t *input_int8,
        TensorBase *weight_tensor,
        int32_t *output_int32,
        int m, int n, int k)
    {
        // Pack weights to INT8 if not already cached
        // This happens once per weight tensor (cached in weight_tensor->cache_)
        if (!weight_tensor->cache_.has_value())
        {
            try
            {
                weight_tensor->cache_ = gemm_v4::pack_weights_to_int8(*weight_tensor, k, n);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("[FusedDualGEMM] Failed to pack weights: " << e.what());
                return false;
            }
        }

        // Execute OneDNN INT8×INT8 → INT32 GEMM
        // Note: We're directly calling run_onednn_int8_matmul instead of the full
        // onednn_gemm_from_packed pipeline because we want INT32 accumulators,
        // not dequantized FP32 output

        // Extract weight data from packed cache
        const auto &weight_pack = std::any_cast<const gemm_v4::WeightPack &>(weight_tensor->cache_);

        // Call OneDNN INT8 matmul: C = A @ B
        // A: [m, k] INT8 (input activations)
        // B: [k, n] INT8 (weights, column-major in weight_pack)
        // C: [m, n] INT32 (output accumulators)
        return gemm_v4::run_onednn_int8_matmul(
            input_int8,
            weight_pack.data.data(),
            output_int32,
            m, n, k);
    }

} // namespace llaminar2
