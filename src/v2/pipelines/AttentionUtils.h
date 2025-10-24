/**
 * @file AttentionUtils.h
 * @brief Utilities for attention computation across different architectures
 *
 * Provides helper functions for GQA (Grouped Query Attention), MHA, and MQA:
 * - K/V head broadcasting for GQA
 * - Causal and sliding window mask generation
 * - Head reshaping utilities
 *
 * Used by PipelineBase::attention_gqa() default orchestration.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <cstddef>
#include <cmath>
#include <cstring>
#include <limits>

namespace llaminar2
{
    namespace attention_utils
    {

        /**
         * @brief Broadcast K/V heads to match Q heads for GQA
         *
         * In GQA (Grouped Query Attention), each KV head serves multiple Q heads:
         *   - n_heads > n_kv_heads (e.g., 14 Q heads, 2 KV heads)
         *   - Each KV head is replicated to (n_heads / n_kv_heads) Q heads
         *
         * Special cases:
         *   - MHA: n_heads == n_kv_heads (no broadcasting, just copy)
         *   - MQA: n_kv_heads == 1 (broadcast single KV head to all Q heads)
         *
         * @param kv_in Input K or V tensor [seq_len, n_kv_heads, head_dim] (flattened)
         * @param kv_out Output K or V tensor [seq_len, n_heads, head_dim] (flattened)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         */
        inline void broadcast_kv_heads(
            const float *kv_in, float *kv_out,
            int seq_len, int n_heads, int n_kv_heads, int head_dim)
        {
            const int heads_per_kv = n_heads / n_kv_heads;

            for (int s = 0; s < seq_len; ++s)
            {
                for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
                {
                    // Source: kv_in[s, kv_h, :] (one KV head)
                    const float *src = kv_in + s * n_kv_heads * head_dim + kv_h * head_dim;

                    // Broadcast to multiple Q heads: kv_out[s, kv_h*heads_per_kv + i, :]
                    for (int i = 0; i < heads_per_kv; ++i)
                    {
                        int q_h = kv_h * heads_per_kv + i;
                        float *dst = kv_out + s * n_heads * head_dim + q_h * head_dim;
                        std::memcpy(dst, src, head_dim * sizeof(float));
                    }
                }
            }
        }

        /**
         * @brief Create causal attention mask (lower triangular)
         *
         * For autoregressive generation: token i can only attend to tokens [0..i]
         *
         * mask[i][j] = 0.0    if i >= j (can attend)
         *            = -inf   if i < j  (cannot attend to future)
         *
         * Sliding window variant (window_size > 0):
         * mask[i][j] = 0.0    if i >= j AND i - j < window_size
         *            = -inf   otherwise
         *
         * @param mask Output mask [seq_len, seq_len]
         * @param seq_len Sequence length
         * @param window_size Sliding window size (-1 = full attention, 0+ = local window)
         */
        inline void create_causal_mask(
            float *mask, int seq_len, int window_size = -1)
        {
            const float neg_inf = -std::numeric_limits<float>::infinity();

            for (int i = 0; i < seq_len; ++i)
            {
                for (int j = 0; j < seq_len; ++j)
                {
                    bool can_attend = (i >= j); // Causal: can't attend to future

                    // Sliding window: also check distance
                    if (window_size >= 0 && can_attend)
                    {
                        can_attend = (i - j < window_size);
                    }

                    mask[i * seq_len + j] = can_attend ? 0.0f : neg_inf;
                }
            }
        }

        /**
         * @brief Apply attention mask to scores (adds mask to scores)
         *
         * scores[i][j] += mask[i][j]
         *
         * This adds -inf to masked positions, which become 0 after softmax.
         *
         * @param scores Attention scores [rows, cols] (modified in-place)
         * @param mask Attention mask [rows, cols]
         * @param rows Number of rows (typically seq_len)
         * @param cols Number of columns (typically seq_len)
         */
        inline void apply_attention_mask(
            float *scores, const float *mask,
            int rows, int cols)
        {
            for (int i = 0; i < rows * cols; ++i)
            {
                scores[i] += mask[i];
            }
        }

        /**
         * @brief Reshape flat tensor to head dimensions
         *
         * Reshapes [seq_len, n_heads * head_dim] to [seq_len, n_heads, head_dim]
         * This is a no-op in terms of memory layout (just dimensional interpretation),
         * but helps clarify the head structure for attention computation.
         *
         * Note: This is conceptual - actual memory layout stays the same.
         * Use for documentation/validation purposes.
         *
         * @param total_size Expected total size (seq_len * n_heads * head_dim)
         * @param seq_len Sequence length
         * @param n_heads Number of heads
         * @param head_dim Dimension per head
         * @return true if dimensions are consistent
         */
        inline bool validate_head_reshape(
            size_t total_size, int seq_len, int n_heads, int head_dim)
        {
            return (total_size == static_cast<size_t>(seq_len * n_heads * head_dim));
        }

        /**
         * @brief Scale attention scores by 1/sqrt(head_dim)
         *
         * Applies the standard attention scaling factor to prevent softmax saturation.
         *
         * scores[i] *= 1.0 / sqrt(head_dim)
         *
         * @param scores Attention scores (modified in-place)
         * @param count Number of elements
         * @param head_dim Dimension per head
         */
        inline void scale_attention_scores(
            float *scores, int count, int head_dim)
        {
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            for (int i = 0; i < count; ++i)
            {
                scores[i] *= scale;
            }
        }

    } // namespace attention_utils
} // namespace llaminar2
