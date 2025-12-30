/**
 * @file Q16DotProductRef.h
 * @brief Reference microkernel for Q16_1 dot products (Q×K^T)
 *
 * This microkernel computes scaled dot products between Q16_1 quantized
 * query and key vectors. It handles block-wise scale factors correctly.
 *
 * Two execution paths:
 *
 * FLASH DECODE (decode path, seq_len_q=1):
 * - GEMV: Single query × all KV positions in parallel
 * - Full softmax row computed at once (no tiling)
 * - Optimized for latency with single token generation
 *
 * FA2 PREFILL (prefill path, seq_len_q>1):
 * - GEMM: Query tile (Br) × KV tile (Bc)
 * - Tiled processing with online softmax state
 * - Optimized for throughput with batched queries
 *
 * @see PROJECT_Q16_INTEGER_ATTENTION.md
 */
#pragma once

#include <cstdint>
#include <vector>
#include "tensors/BlockStructures.h"

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Configuration
    // ============================================================================

    /**
     * @brief Parameters for Q16 dot product microkernel
     */
    struct Q16DotProductParams
    {
        const Q16_1Block *Q = nullptr; ///< Query blocks [q_rows × q_blocks_per_row]
        const Q16_1Block *K = nullptr; ///< Key blocks [k_rows × kv_blocks_per_row]

        int q_rows = 0;               ///< Number of query rows
        int k_rows = 0;               ///< Number of key rows
        int head_dim = 0;             ///< Dimension per head
        int q_head = 0;               ///< Query head index
        int kv_head = 0;              ///< KV head index (for GQA)
        int q_blocks_per_row = 0;     ///< Blocks per Q row
        int kv_blocks_per_row = 0;    ///< Blocks per K row
        float attention_scale = 0.0f; ///< 1/sqrt(head_dim)
    };

    // ============================================================================
    // GEMV Variant (Flash Decode: Single Query, Parallel over KV)
    // ============================================================================

    /**
     * @brief Compute dot products for single query vs all keys (Flash Decode)
     *
     * Computes: scores[kv] = Q[0, head, :] · K[kv, kv_head, :] * scale
     *
     * FLASH DECODE: Processes all KV positions at once (no tiling).
     * This enables single-pass softmax without online state tracking.
     * The output scores are properly scaled FP32 values (block scales applied).
     *
     * @param params Microkernel parameters
     * @param q_row Query row index (typically 0 for decode)
     * @param k_start Starting key row
     * @param k_count Number of keys to process
     * @param scores Output: [k_count] scaled FP32 scores
     */
    void q16_dot_product_gemv(
        const Q16DotProductParams &params,
        int q_row,
        int k_start,
        int k_count,
        float *scores);

    /**
     * @brief Compute single dot product with proper scale handling
     *
     * @param params Microkernel parameters
     * @param q_row Query row index
     * @param k_row Key row index
     * @return Scaled FP32 dot product (with attention_scale applied)
     */
    float q16_dot_product_single(
        const Q16DotProductParams &params,
        int q_row,
        int k_row);

    // ============================================================================
    // GEMM Variant (FA2 Prefill: Tiled Processing)
    // ============================================================================

    /**
     * @brief Compute dot products for query tile vs key tile (FA2 Prefill)
     *
     * Computes: S[q_local, kv] = Q[q_start+q_local, head, :] · K[kv, kv_head, :] * scale
     *
     * FA2 PREFILL: Processes tiles for cache efficiency with long sequences.
     * Used with OnlineSoftmaxState to maintain running softmax across KV tiles.
     * Output layout: scores[q_local * k_count + kv]
     *
     * @param params Microkernel parameters
     * @param q_start Starting query row
     * @param q_count Number of queries (Br tile size)
     * @param k_start Starting key row
     * @param k_count Number of keys (Bc tile size)
     * @param scores Output: [q_count × k_count] scaled FP32 scores
     */
    void q16_dot_product_gemm(
        const Q16DotProductParams &params,
        int q_start,
        int q_count,
        int k_start,
        int k_count,
        float *scores);

    // ============================================================================
    // Integer-Domain Variant (for future JIT parity testing)
    // ============================================================================

    /**
     * @brief Compute raw INT32 dot product without scale factors
     *
     * This variant returns the raw integer dot product, useful for
     * testing JIT kernels that operate in pure integer domain.
     *
     * @param Q Query blocks
     * @param K Key blocks
     * @param q_row Query row
     * @param k_row Key row
     * @param head Query head index
     * @param kv_head KV head index
     * @param head_dim Dimension per head
     * @param q_blocks_per_row Blocks per Q row
     * @param kv_blocks_per_row Blocks per K row
     * @return Raw INT32 dot product (no scaling applied)
     */
    int32_t q16_dot_product_int32(
        const Q16_1Block *Q,
        const Q16_1Block *K,
        int q_row,
        int k_row,
        int head,
        int kv_head,
        int head_dim,
        int q_blocks_per_row,
        int kv_blocks_per_row);

} // namespace llaminar2::kernels::q16_1::microkernels
