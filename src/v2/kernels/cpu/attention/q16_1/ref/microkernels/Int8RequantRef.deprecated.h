/**
 * @file Int8RequantRef.h
 * @brief INT8 requantization utilities for Q16_1 attention
 *
 * When using exp2 fixed-point softmax with Q16_1 tensors, we need a stable
 * alpha = sQ * sK * attention_scale. Since Q16_1 blocks have per-block scales,
 * we temporarily requantize Q/K heads to INT8 with per-head maxabs scaling.
 *
 * This produces INT8 vectors for Q and K (one head at a time), with
 * corresponding scales sQ and sK such that alpha is well-defined.
 *
 * @see Exp2FixedSoftmaxRef.h for the softmax that consumes these INT8 logits
 */
#pragma once

#include "tensors/BlockStructures.h"
#include <cstdint>

namespace llaminar2::kernels::q16_1::microkernels
{

    /**
     * @brief Compute maxabs of a head slice from Q16_1 row blocks
     *
     * @param blocks Row of Q16_1 blocks
     * @param blocks_per_row Number of blocks in the row
     * @param start_col Starting column index (head * head_dim)
     * @param head_dim Number of elements per head
     * @return Maximum absolute value (FP32)
     */
    float q16_maxabs_head_row(
        const Q16_1Block *blocks,
        int blocks_per_row,
        int start_col,
        int head_dim);

    /**
     * @brief Compute maxabs across multiple K rows for a head slice
     *
     * @param blocks K tensor as rows of Q16_1 blocks
     * @param num_rows Number of K rows
     * @param blocks_per_row Number of blocks per row
     * @param start_col Starting column index (kv_head * head_dim)
     * @param head_dim Number of elements per head
     * @return Maximum absolute value across all rows (FP32)
     */
    float q16_maxabs_head_across_rows(
        const Q16_1Block *blocks,
        int num_rows,
        int blocks_per_row,
        int start_col,
        int head_dim);

    /**
     * @brief Quantize a head slice from Q16_1 blocks to INT8
     *
     * @param blocks Row of Q16_1 blocks
     * @param blocks_per_row Number of blocks in the row
     * @param start_col Starting column index
     * @param head_dim Number of elements
     * @param scale Scale factor (maxabs / 127)
     * @param out Output INT8 buffer [head_dim]
     */
    void q16_quantize_head_to_int8(
        const Q16_1Block *blocks,
        int blocks_per_row,
        int start_col,
        int head_dim,
        float scale,
        int8_t *out);

    /**
     * @brief Compute INT8 dot product
     *
     * @param a First INT8 vector
     * @param b Second INT8 vector
     * @param n Length
     * @return INT32 dot product
     */
    int32_t dot_int8(const int8_t *a, const int8_t *b, int n);

    /**
     * @brief Parameters for computing INT8 requantized logits for a single head
     */
    struct Int8RequantParams
    {
        const Q16_1Block *Q;   ///< Q tensor blocks
        const Q16_1Block *K;   ///< K tensor blocks
        int q_row;             ///< Query row index
        int head;              ///< Query head index
        int kv_head;           ///< KV head index
        int head_dim;          ///< Dimension per head
        int kv_end;            ///< Number of K positions to process
        int q_blocks_per_row;  ///< Q blocks per row
        int kv_blocks_per_row; ///< K blocks per row
        float attention_scale; ///< 1/sqrt(head_dim)
    };

    /**
     * @brief Compute INT8 requantized Q·K^T scores for a single head
     *
     * Requantizes Q[q_row, head] and K[0:kv_end, kv_head] to INT8 with
     * per-head maxabs scaling, then computes INT32 dot products.
     *
     * @param params Configuration parameters
     * @param scores Output INT32 scores [kv_end]
     * @param alpha_out Output combined scale factor (sQ * sK * attention_scale)
     */
    void compute_int8_requant_logits(
        const Int8RequantParams &params,
        int32_t *scores,
        float *alpha_out);

} // namespace llaminar2::kernels::q16_1::microkernels
