/**
 * @file INT8AttentionKernel.h
 * @brief INT8 multi-head attention kernel with INT32 accumulators
 * @author David Sanftenberg
 * @date 2025-11-06
 *
 * Implements multi-head attention for full INT8 transformers:
 *
 * Pipeline:
 *   1. INT8 Q × INT8 K^T → INT32 scores [batch, n_heads, seq_len, seq_len]
 *   2. INT32 → FP32 dequantization (for softmax numerical stability)
 *   3. FP32 Softmax → INT8 attention weights (requantize)
 *   4. INT8 attn_weights × INT8 V → INT32 context [batch, n_heads, seq_len, d_head]
 *   5. INT32 → INT8 requantization (output for next layer)
 *
 * Key Features:
 * - Per-head quantization for better precision
 * - Causal masking support (for autoregressive decoding)
 * - INT32 accumulator prevents overflow in score computation
 * - Softmax operates in FP32 (required for numerical stability)
 * - Output requantized to INT8 for next layer
 *
 * Memory Layout:
 * - Q, K, V: INT8 [batch, seq_len, n_heads * d_head] (interleaved heads)
 * - Scores: INT32 [batch, n_heads, seq_len, seq_len] (per-head)
 * - Attn weights: INT8 [batch, n_heads, seq_len, seq_len] after softmax
 * - Context: INT32 [batch, n_heads, seq_len, d_head]
 * - Output: INT8 [batch, seq_len, n_heads * d_head] (interleaved heads)
 *
 * Usage Example:
 * @code
 *   INT8AttentionKernel attn(n_heads, d_head);
 *
 *   std::vector<int8_t> output_int8(batch * seq_len * d_model);
 *   std::vector<float> output_row_scales(batch * seq_len);
 *
 *   bool success = attn.forward(
 *       q_int8, q_row_scales,
 *       k_int8, k_row_scales,
 *       v_int8, v_row_scales,
 *       output_int8.data(), output_row_scales.data(),
 *       batch, seq_len, use_causal_mask);
 * @endcode
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

namespace llaminar2
{

    /**
     * @brief INT8 multi-head attention kernel
     *
     * Performs scaled dot-product attention in INT8/INT32 arithmetic:
     *   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_head)) @ V
     *
     * All inputs and outputs are INT8, with INT32 intermediate accumulators.
     * Softmax operates in FP32 for numerical stability.
     */
    class INT8AttentionKernel
    {
    public:
        /**
         * @brief Construct attention kernel
         *
         * @param n_heads Number of attention heads
         * @param d_head Dimension per head
         * @param device_idx Device index (-1 for CPU)
         */
        INT8AttentionKernel(int n_heads, int d_head, int device_idx = -1);

        /**
         * @brief Forward pass: INT8 Q/K/V → INT32 scores → FP32 softmax → INT8 output
         *
         * @param q_int8 Query tensor INT8 [batch, seq_len, n_heads * d_head]
         * @param q_row_scales Query per-row scales [batch * seq_len]
         * @param k_int8 Key tensor INT8 [batch, seq_len, n_heads * d_head]
         * @param k_row_scales Key per-row scales [batch * seq_len]
         * @param v_int8 Value tensor INT8 [batch, seq_len, n_heads * d_head]
         * @param v_row_scales Value per-row scales [batch * seq_len]
         * @param output_int8 Output INT8 [batch, seq_len, n_heads * d_head] (OUT)
         * @param output_row_scales Output per-row scales [batch * seq_len] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         * @param use_causal_mask Whether to apply causal mask (for autoregressive)
         * @param eps Epsilon for numerical stability (default 1e-8)
         *
         * @return true on success, false on error
         *
         * @note All pointers must be non-null. Dimensions must match:
         *       q/k/v shape: [batch, seq_len, n_heads * d_head]
         *       output shape: [batch, seq_len, n_heads * d_head]
         */
        bool forward(
            const int8_t *q_int8,
            const float *q_row_scales,
            const int8_t *k_int8,
            const float *k_row_scales,
            const int8_t *v_int8,
            const float *v_row_scales,
            int8_t *output_int8,
            float *output_row_scales,
            int batch,
            int seq_len,
            bool use_causal_mask = false,
            float eps = 1e-8f);

        /**
         * @brief Get number of attention heads
         */
        int num_heads() const { return n_heads_; }

        /**
         * @brief Get dimension per head
         */
        int head_dim() const { return d_head_; }

    private:
        /**
         * @brief Compute attention scores: Q @ K^T
         *
         * @param q_int8 Query INT8 [batch, n_heads, seq_len, d_head]
         * @param q_row_scales Query scales [batch * seq_len]
         * @param k_int8 Key INT8 [batch, n_heads, seq_len, d_head]
         * @param k_row_scales Key scales [batch * seq_len]
         * @param scores_int32 Output INT32 [batch, n_heads, seq_len, seq_len] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         */
        void compute_scores(
            const int8_t *q_int8,
            const float *q_row_scales,
            const int8_t *k_int8,
            const float *k_row_scales,
            int batch,
            int seq_len);

        /**
         * @brief Apply softmax to attention scores (FP32 → INT8)
         *
         * Reads from scores_fp32_buffer_, applies softmax, outputs INT8 weights.
         * @param attn_weights_int8 Output INT8 [batch, n_heads, seq_len, seq_len] (OUT)
         * @param attn_weights_row_scales Per-row scales [batch * n_heads * seq_len] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         * @param use_causal_mask Apply causal mask before softmax
         * @param eps Epsilon for numerical stability
         */
        void apply_softmax(
            int8_t *attn_weights_int8,
            float *attn_weights_row_scales,
            int batch,
            int seq_len,
            bool use_causal_mask,
            float eps);

        /**
         * @brief Compute attention output: attn_weights @ V
         *
         * @param attn_weights_int8 Attention weights INT8 [batch, n_heads, seq_len, seq_len]
         * @param attn_weights_row_scales Attention weights scales [batch * n_heads * seq_len]
         * @param v_int8 Value tensor INT8 [batch, n_heads, seq_len, d_head]
         * @param v_row_scales Value scales [batch * seq_len]
         * @param context_fp32 Output context FP32 [batch, n_heads, seq_len, d_head] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         */
        void compute_context(
            const int8_t *attn_weights_int8,
            const float *attn_weights_row_scales,
            const int8_t *v_int8,
            const float *v_row_scales,
            float *context_fp32,
            int batch,
            int seq_len);

        /**
         * @brief Requantize context to INT8 (FP32 → INT8)
         *
         * @param context_fp32 Input context FP32 [batch, n_heads, seq_len, d_head]
         * @param output_int8 Output INT8 [batch, seq_len, n_heads * d_head] (OUT)
         * @param output_row_scales Output scales [batch * seq_len] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         */
        void requantize_output(
            const float *context_fp32,
            int8_t *output_int8,
            float *output_row_scales,
            int batch,
            int seq_len);

        int n_heads_;    ///< Number of attention heads
        int d_head_;     ///< Dimension per head
        int device_idx_; ///< Device index (-1 for CPU)

        // Temporary buffers (reused across forward calls)
        std::vector<int32_t> scores_buffer_;            ///< [batch, n_heads, seq_len, seq_len]
        std::vector<float> scores_fp32_buffer_;         ///< For dequantized scores
        std::vector<int8_t> attn_weights_buffer_;       ///< [batch, n_heads, seq_len, seq_len]
        std::vector<float> attn_weights_scales_buffer_; ///< [batch * n_heads * seq_len]
        std::vector<float> context_buffer_;             ///< [batch, n_heads, seq_len, d_head] (FP32)
    };

} // namespace llaminar2
