/**
 * @file FusedDualGEMM.h
 * @brief Fused dual GEMM for FFN gate/up projections with shared input quantization
 *
 * Fuses gate and up projections in FFN blocks by:
 * 1. Quantizing input activations ONCE (FP32 → INT8 with per-row scales)
 * 2. Performing 2 INT8×INT8 GEMMs with same input (gate_proj, up_proj)
 * 3. Outputting INT32 accumulators (defer dequantization to fused SwiGLU)
 *
 * Performance Benefits:
 * - Eliminates 1 redundant quantization pass (gate and up share input)
 * - Defers dequantization until SwiGLU fusion (saves 2 dequant passes)
 * - Better cache locality (input stays hot between GEMMs)
 *
 * Expected Speedup: 5-8% in FFN block (1 quantization instead of 2)
 *
 * Algorithm:
 * 1. Quantize input once: FP32[m,k] → INT8[m,k] + row_scales[m]
 * 2. Gate GEMM: INT8[m,k] × INT8_weight_gate[k,n] → INT32_gate[m,n]
 * 3. Up GEMM: INT8[m,k] × INT8_weight_up[k,n] → INT32_up[m,n]
 * 4. Return INT32 accumulators for downstream fused SwiGLU
 *
 * Fusion Chain:
 *   FusedRMSNormQuantize → FusedDualGEMM → FusedDequantSwiGLU
 *   (FP32→INT8)         → (INT8→INT32)  → (INT32→FP32+activation)
 *
 * @author David Sanftenberg
 * @date 2025-11-23
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include <cstdint>
#include <vector>
#include <memory>

namespace llaminar2
{
    /**
     * @brief Fused dual GEMM kernel for FFN gate/up projections
     *
     * Replaces:
     *   Quant(input) → Gate GEMM → Dequant(gate_out)
     *   Quant(input) → Up GEMM → Dequant(up_out)
     *
     * With:
     *   Quant(input) → FusedDualGEMM(gate_weight, up_weight) → [gate_int32, up_int32]
     *
     * Usage:
     *   FusedDualGEMM kernel(gate_weight, up_weight);
     *   kernel.execute(input_fp32, gate_output_int32, up_output_int32,
     *                  activation_scales, m, n, k);
     */
    class FusedDualGEMM : public CPUKernelBase
    {
    public:
        /**
         * @brief Construct fused dual GEMM kernel with weight tensors
         *
         * @param gate_weight INT8 quantized gate projection weights [k, n]
         * @param up_weight INT8 quantized up projection weights [k, n]
         */
        FusedDualGEMM(TensorBase *gate_weight, TensorBase *up_weight);
        ~FusedDualGEMM() override = default;

        // =============================================================================
        // CPUKernelBase Interface (Fusion Framework)
        // =============================================================================

        /**
         * @brief Get kernel I/O contract for fusion pattern detection
         */
        KernelContract get_contract() const override
        {
            return KernelContract{
                .accepted_input_formats = {TensorFormat::FP32}, // Accept FP32 activations
                .output_format = TensorFormat::INT32,           // Produce INT32 accumulators
                .supports_inplace = false,                      // Need separate output buffers
                .is_fusable = true                              // Can fuse with FusedDequantSwiGLU
            };
        }

        bool supports_fusion() const override
        {
            return true; // High-priority fusion candidate
        }

        TensorFormat preferred_fusion_format() const override
        {
            return TensorFormat::INT32; // Output format for downstream fusion
        }

        // =============================================================================
        // Execution Interface
        // =============================================================================

        /**
         * @brief Execute fused dual GEMM with shared input quantization
         *
         * Performs:
         * 1. Quantize input: FP32[m,k] → INT8[m,k] + row_scales[m]
         * 2. Gate GEMM: INT8[m,k] × gate_weight[k,n] → gate_output[m,n] (INT32)
         * 3. Up GEMM: INT8[m,k] × up_weight[k,n] → up_output[m,n] (INT32)
         *
         * @param input Input activations [m, k] FP32
         * @param gate_output Output gate INT32 accumulators [m, n]
         * @param up_output Output up INT32 accumulators [m, n]
         * @param activation_scales Per-row quantization scales [m] FP32 (output)
         * @param m Batch size (sequence length)
         * @param n Hidden dimension (output features)
         * @param k Input features
         * @return true on success, false on error
         *
         * Note: activation_scales are computed during quantization and needed
         *       for downstream dequantization in FusedDequantSwiGLU.
         */
        bool execute(
            const float *input,
            int32_t *gate_output,
            int32_t *up_output,
            float *activation_scales,
            int m, int n, int k);

        /**
         * @brief Check if kernel supports given device
         * @param device_idx Device index (-1 = CPU)
         * @return true if device is supported
         */
        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only for now
        }

    private:
        /**
         * @brief Quantize input activations to INT8 with per-row scales
         *
         * Quantization formula:
         *   row_scale[i] = max(|input[i,:]|) / 127.0
         *   quantized[i,j] = round(input[i,j] / row_scale[i])
         *
         * @param input Input FP32 activations [m, k]
         * @param output Output INT8 activations [m, k]
         * @param row_scales Output per-row scales [m]
         * @param m Number of rows
         * @param k Number of columns
         */
        void quantize_per_row(
            const float *input,
            int8_t *output,
            float *row_scales,
            int m, int k);

        /**
         * @brief Execute single INT8×INT8 → INT32 GEMM via OneDNN
         *
         * @param input_int8 Quantized input [m, k] INT8
         * @param weight_tensor Weight tensor (contains packed INT8 weights)
         * @param output_int32 Output accumulator [m, n] INT32
         * @param m Batch size
         * @param n Output features
         * @param k Input features
         * @return true on success
         */
        bool execute_int8_gemm(
            const int8_t *input_int8,
            TensorBase *weight_tensor,
            int32_t *output_int32,
            int m, int n, int k);

        // Weight tensors (owned externally, not by kernel)
        TensorBase *gate_weight_;
        TensorBase *up_weight_;

        // Temporary INT8 activation buffer (reused across calls)
        std::vector<int8_t> int8_activations_;
    };

} // namespace llaminar2
