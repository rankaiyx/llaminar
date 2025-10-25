/**
 * @file RoPEPrimitives.h
 * @brief Vectorized RoPE primitives for V2 (ported from V1)
 * @author David Sanftenberg
 *
 * High-performance RoPE implementation with:
 * - AVX512/AVX2 vectorization (8-16× speedup)
 * - Persistent thread-local state for decode
 * - Complex recurrence for angle advancement
 * - Inverse frequency caching
 * - Angle recurrence across tokens in prefill
 */
#pragma once

#include <vector>
#include <cstddef>

namespace llaminar2::primitives
{

    /**
     * @brief Persistent state for single-token decode optimization
     */
    struct RoPEPersistentState
    {
        int cached_head_dim = -1;
        float cached_freq_base = -1.0f;
        int last_pos = -1;
        std::vector<float> cos_curr;
        std::vector<float> sin_curr;
        std::vector<float> cos_delta;
        std::vector<float> sin_delta;

        void reset()
        {
            last_pos = -1;
            cos_curr.clear();
            sin_curr.clear();
            cos_delta.clear();
            sin_delta.clear();
        }
    };

    /**
     * @brief Apply RoPE to Q and K tensors (vectorized implementation)
     *
     * Layout: [seq_len, num_heads, head_dim] in row-major order
     *
     * @param q Query tensor (modified in-place)
     * @param k Key tensor (modified in-place)
     * @param seq_len Sequence length
     * @param head_dim Dimension per head (must be even)
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads (may differ for GQA)
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE (model-specific)
     * @param persistent_state Optional persistent state for decode optimization
     */
    void apply_rope_vectorized(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Get cached inverse frequencies for (head_dim, freq_base)
     * @return Vector of inverse frequencies, length = head_dim/2
     */
    const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base);

} // namespace llaminar2::primitives
