/**
 * @file FusedDequantSwiGLU.h
 * @brief Fused dequantization + SwiGLU activation for FFN blocks
 *
 * Fuses dequantization of INT32 accumulators with SwiGLU activation:
 * 1. Dequantize gate INT32 accumulators → FP32
 * 2. Dequantize up INT32 accumulators → FP32
 * 3. Apply SwiGLU: output = gate * silu(up)
 *    where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Performance Benefits:
 * - Eliminates 2 separate dequantization passes (fused into activation)
 * - Better cache locality (gate/up dequantized and consumed immediately)
 * - SIMD-optimized SwiGLU (AVX512/AVX2 fused multiply-add)
 *
 * Expected Speedup: 3-5% in FFN block (eliminates 2 dequant round trips)
 *
 * Algorithm:
 * 1. Dequantize gate: gate_fp32[i,j] = gate_int32[i,j] * act_scales[i] * gate_col_scales[j]
 * 2. Dequantize up: up_fp32[i,j] = up_int32[i,j] * act_scales[i] * up_col_scales[j]
 * 3. Apply SwiGLU: output[i,j] = gate_fp32[i,j] * silu(up_fp32[i,j])
 *
 * Fusion Chain:
 *   FusedDualGEMM → FusedDequantSwiGLU
 *   (FP32→INT32) → (INT32→FP32+activation)
 *
 * @author David Sanftenberg
 * @date 2025-11-23
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../tensors/TensorKernels.h"
#include <cstdint>
#include <cmath> // For std::exp in silu()
#include <vector>

namespace llaminar2
{
    /**
     * @brief Fused dequantization + SwiGLU activation kernel
     *
     * Replaces:
     *   Dequant(gate_int32) → gate_fp32
     *   Dequant(up_int32) → up_fp32
     *   SwiGLU(gate_fp32, up_fp32) → output
     *
     * With:
     *   FusedDequantSwiGLU(gate_int32, up_int32, scales) → output
     *
     * Usage:
     *   FusedDequantSwiGLU kernel;
     *   kernel.execute(gate_int32, up_int32, output_fp32,
     *                  activation_scales, gate_col_scales, up_col_scales,
     *                  m, n);
     */
    class FusedDequantSwiGLU : public CPUKernelBase
    {
    public:
        FusedDequantSwiGLU() = default;
        ~FusedDequantSwiGLU() override = default;

        // =============================================================================
        // CPUKernelBase Interface (Fusion Framework)
        // =============================================================================

        /**
         * @brief Get kernel I/O contract for fusion pattern detection
         */
        KernelContract get_contract() const override
        {
            return KernelContract{
                .accepted_input_formats = {TensorFormat::INT32}, // Accept INT32 accumulators
                .output_format = TensorFormat::FP32,             // Produce FP32 output
                .supports_inplace = false,                       // Need separate output buffer
                .is_fusable = true                               // Can fuse with downstream ops
            };
        }

        bool supports_fusion() const override
        {
            return true; // High-priority fusion candidate
        }

        TensorFormat preferred_fusion_format() const override
        {
            return TensorFormat::FP32; // Output format for downstream fusion
        }

        // =============================================================================
        // Execution Interface
        // =============================================================================

        /**
         * @brief Execute fused dequantization + SwiGLU activation
         *
         * Performs:
         * 1. Dequantize gate: gate_fp32 = gate_int32 * act_scales * gate_col_scales
         * 2. Dequantize up: up_fp32 = up_int32 * act_scales * up_col_scales
         * 3. Apply SwiGLU: output = gate_fp32 * silu(up_fp32)
         *
         * @param gate_int32 Gate INT32 accumulators [m, n]
         * @param up_int32 Up INT32 accumulators [m, n]
         * @param output Output FP32 activations [m, n]
         * @param activation_scales Per-row activation quantization scales [m]
         * @param gate_col_scales Per-column gate weight quantization scales [n]
         * @param up_col_scales Per-column up weight quantization scales [n]
         * @param m Batch size (sequence length)
         * @param n Hidden dimension (intermediate_size)
         * @return true on success, false on error
         *
         * Note: Scales come from:
         *   - activation_scales: FusedDualGEMM quantization output
         *   - gate_col_scales / up_col_scales: Weight tensor quantization metadata
         */
        bool execute(
            const int32_t *gate_int32,
            const int32_t *up_int32,
            float *output,
            const float *activation_scales,
            const float *gate_col_scales,
            const float *up_col_scales,
            int m, int n);

        /**
         * @brief Check if kernel supports given device
         * @param device_idx Device index (-1 = CPU)
         * @return true if device is supported
         */
        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only for now
        }
    };

} // namespace llaminar2
