/**
 * @file FP32AttentionKernel.h
 * @brief FP32 multi-head attention kernel for V2 architecture
 * @author David Sanftenberg
 * @date 2025-11-06
 *
 * Pure FP32 attention implementation for validation against PyTorch ground truth.
 * This serves as a validated reference before testing INT8 quantized attention.
 *
 * Pipeline: FP32 Q/K/V → FP32 scores → FP32 softmax → FP32 output
 *
 * Key features:
 * - Multi-head attention with Grouped Query Attention (GQA) support
 * - Optional causal masking for autoregressive generation
 * - RoPE (Rotary Position Embeddings) not included (applied before attention)
 * - CPU-only implementation (device_idx=-1)
 *
 * Memory layout:
 * - Q: [batch, seq_len, n_heads * d_head] (interleaved heads)
 * - K: [batch, seq_len, n_kv_heads * d_head] (interleaved heads)
 * - V: [batch, seq_len, n_kv_heads * d_head] (interleaved heads)
 * - Output: [batch, seq_len, n_heads * d_head] (interleaved heads)
 *
 * Usage:
 *   FP32AttentionKernel attn(n_heads=14, d_head=64);
 *   attn.forward(q, k, v, output, batch=1, seq_len=4, use_causal_mask=false);
 */

#pragma once

#include <vector>
#include <cstdint>

namespace llaminar2
{

    class FP32AttentionKernel
    {
    public:
        /**
         * @brief Construct FP32 attention kernel
         * @param n_heads Number of query heads
         * @param d_head Dimension per head
         * @param n_kv_heads Number of key/value heads (for GQA, defaults to n_heads for MHA)
         * @param device_idx Device index (-1 for CPU, >=0 for GPU)
         */
        FP32AttentionKernel(int n_heads, int d_head, int n_kv_heads = 0, int device_idx = -1);

        /**
         * @brief Forward pass: Multi-head attention
         *
         * Computes: output = Softmax(Q @ K^T / sqrt(d_head)) @ V
         *
         * @param q Query tensor [batch, seq_len, n_heads * d_head]
         * @param k Key tensor [batch, seq_len, n_kv_heads * d_head]
         * @param v Value tensor [batch, seq_len, n_kv_heads * d_head]
         * @param output Output tensor [batch, seq_len, n_heads * d_head] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         * @param use_causal_mask Apply causal mask (future positions masked)
         * @param eps Epsilon for numerical stability (softmax)
         * @return true if successful, false on error
         */
        bool forward(
            const float *q,
            const float *k,
            const float *v,
            float *output,
            int batch,
            int seq_len,
            bool use_causal_mask = false,
            float eps = 1e-8f);

    private:
        /**
         * @brief Compute attention scores: Q @ K^T
         *
         * Populates scores_buffer_ with raw dot products (before scaling).
         *
         * @param q Query [batch, seq_len, n_heads * d_head]
         * @param k Key [batch, seq_len, n_kv_heads * d_head]
         * @param batch Batch size
         * @param seq_len Sequence length
         */
        void compute_scores(
            const float *q,
            const float *k,
            int batch,
            int seq_len);

        /**
         * @brief Apply softmax to attention scores
         *
         * Reads from scores_buffer_, applies scaling, causal masking (optional),
         * softmax, then writes to attn_weights_buffer_.
         *
         * @param batch Batch size
         * @param seq_len Sequence length
         * @param use_causal_mask Apply causal mask before softmax
         * @param eps Epsilon for numerical stability
         */
        void apply_softmax(
            int batch,
            int seq_len,
            bool use_causal_mask,
            float eps);

        /**
         * @brief Compute attention output: attn_weights @ V
         *
         * @param v Value tensor [batch, seq_len, n_kv_heads * d_head]
         * @param output Output [batch, seq_len, n_heads * d_head] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         */
        void compute_output(
            const float *v,
            float *output,
            int batch,
            int seq_len);

        /**
         * @brief Expand K/V heads for GQA (Grouped Query Attention)
         *
         * If n_heads > n_kv_heads, we need to repeat each KV head.
         * Example: n_heads=14, n_kv_heads=2 → each KV head repeated 7 times
         *
         * @param kv_heads Input K or V [batch, seq_len, n_kv_heads, d_head]
         * @param expanded_heads Output [batch, seq_len, n_heads, d_head] (OUT)
         * @param batch Batch size
         * @param seq_len Sequence length
         */
        void expand_kv_heads(
            const float *kv_heads,
            float *expanded_heads,
            int batch,
            int seq_len);

        int n_heads_;    ///< Number of query heads
        int d_head_;     ///< Dimension per head
        int device_idx_; ///< Device index (-1=CPU)
        int n_kv_heads_; ///< Number of key/value heads (computed from K shape)

        // Temporary buffers (reused across forward calls)
        std::vector<float> scores_buffer_;       ///< [batch, n_heads, seq_len, seq_len]
        std::vector<float> attn_weights_buffer_; ///< [batch, n_heads, seq_len, seq_len]
        std::vector<float> k_expanded_buffer_;   ///< [batch, n_heads, seq_len, d_head]
        std::vector<float> v_expanded_buffer_;   ///< [batch, n_heads, seq_len, d_head]
    };

} // namespace llaminar2
