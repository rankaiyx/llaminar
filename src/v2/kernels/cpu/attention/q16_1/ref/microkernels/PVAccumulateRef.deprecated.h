/**
 * @file PVAccumulateRef.h
 * @brief Reference microkernel for P×V accumulation (attention-weighted V sum)
 *
 * This microkernel computes the weighted sum of value vectors using
 * softmax attention weights.
 *
 * AVX512-VNNI COMPATIBILITY:
 * Uses INT16 weights and INT32 accumulators to match VPDPWSSD behavior:
 * - VPDPWSSD: INT16 × INT16 → INT32 (signed multiply-accumulate)
 * - Weights from Index16Softmax are INT16 in range [0, 32767]
 * - V values are INT16 (Q16_1 block quantized)
 * - Accumulators are INT32 (matching VNNI output)
 *
 * Two execution paths:
 *
 * FLASH DECODE (GEMV, single weight row):
 * - Processes all KV positions at once
 * - Single-pass accumulation without rescaling
 * - `pv_accumulate_gemv_*()` variants
 *
 * FA2 PREFILL (GEMM, tiled):
 * - Processes KV in tiles with OnlineSoftmaxState
 * - Uses INT16 weights and INT32 accumulators
 *
 * Scale factor handling:
 * - V values are dequantized using per-block scale factors
 * - INT16 weights from softmax are applied
 * - Output is FP32 (dequantized)
 *
 * @see PROJECT_Q16_INTEGER_ATTENTION.md
 */
#pragma once

#include <cstdint>
#include "tensors/BlockStructures.h"

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Configuration
    // ============================================================================

    /**
     * @brief Parameters for P×V accumulation microkernel
     */
    struct PVAccumulateParams
    {
        const Q16_1Block *V = nullptr; ///< Value blocks [kv_len × kv_blocks_per_row]

        int kv_len = 0;            ///< Total KV cache length
        int head_dim = 0;          ///< Dimension per head
        int kv_head = 0;           ///< KV head index
        int kv_blocks_per_row = 0; ///< Blocks per V row
    };

    // ============================================================================
    // GEMV Variant (Flash Decode: Single Weight Row, All KV at Once)
    // ============================================================================

    /**
     * @brief Accumulate weighted V vectors using FP32 weights (Flash Decode)
     *
     * Computes: output[d] = Σ_kv weights[kv] × V[kv, kv_head, d]
     *
     * FLASH DECODE: Accumulates over all KV positions in a single pass.
     * No rescaling needed since weights are already normalized.
     * V values are properly dequantized using per-block scale factors.
     *
     * @param params Microkernel parameters
     * @param weights FP32 attention weights [kv_count] (normalized softmax)
     * @param kv_start Starting KV position
     * @param kv_count Number of KV positions to accumulate
     * @param output Output FP32 vector [head_dim]
     */
    void pv_accumulate_gemv_fp32(
        const PVAccumulateParams &params,
        const float *weights,
        int kv_start,
        int kv_count,
        float *output);

    /**
     * @brief Accumulate weighted V vectors using INT16 weights (decode path)
     *
     * Integer-domain variant that uses INT16 weights from Index16Softmax.
     * Uses INT32 accumulators matching AVX512-VNNI VPDPWSSD behavior.
     * Normalizes by weight_sum at the end.
     *
     * Computes: output[d] = (Σ_kv weights[kv] × V_int16[kv, d]) × d_v / weight_sum
     *
     * VNNI MODEL: INT16 × INT16 → INT32 accumulator
     *
     * @param params Microkernel parameters
     * @param weights INT16 attention weights [kv_count] (from Index16Softmax, range [0, 32767])
     * @param weight_sum Sum of INT16 weights (for normalization), INT32 to match accumulator
     * @param kv_start Starting KV position
     * @param kv_count Number of KV positions
     * @param output Output FP32 vector [head_dim]
     */
    void pv_accumulate_gemv_int16(
        const PVAccumulateParams &params,
        const int16_t *weights,
        int32_t weight_sum,
        int kv_start,
        int kv_count,
        float *output);

    // ============================================================================
    // GEMM Variant (FA2 Prefill: Batched Weight Rows per Tile)
    // ============================================================================

    /**
     * @brief Accumulate weighted V using INT16 weights (prefill path)
     *
     * Integer-domain variant for prefill. Each query row is normalized
     * by its corresponding weight_sum. Uses INT32 accumulators to match
     * AVX512-VNNI VPDPWSSD behavior.
     *
     * VNNI MODEL: INT16 × INT16 → INT32 accumulator
     *
     * @param params Microkernel parameters
     * @param weights INT16 attention weights [num_queries × kv_count]
     * @param weight_sums Per-row sums [num_queries] (INT32 to match accumulator)
     * @param num_queries Number of query rows
     * @param kv_start Starting KV position
     * @param kv_count Number of KV positions
     * @param output Output FP32 matrix [num_queries × head_dim]
     */
    void pv_accumulate_gemm_int16(
        const PVAccumulateParams &params,
        const int16_t *weights,
        const int32_t *weight_sums,
        int num_queries,
        int kv_start,
        int kv_count,
        float *output);

} // namespace llaminar2::kernels::q16_1::microkernels
