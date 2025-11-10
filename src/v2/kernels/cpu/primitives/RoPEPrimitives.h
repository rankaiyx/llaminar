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
#include <cstdint>

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

    /**
     * @brief Apply RoPE to Q and K tensors (native BF16 implementation)
     *
     * Operates directly on BF16 buffers without intermediate FP32 conversion.
     * Uses vectorized BF16 operations where available (AVX512 BF16, etc.).
     *
     * @param q_bf16 Query tensor in BF16 format (modified in-place)
     * @param k_bf16 Key tensor in BF16 format (modified in-place)
     * @param seq_len Sequence length
     * @param head_dim Dimension per head (must be even)
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads (may differ for GQA)
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE (model-specific)
     * @param persistent_state Optional persistent state for decode optimization
     */
    void apply_rope_bf16(
        uint16_t *q_bf16, uint16_t *k_bf16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Apply RoPE to Q and K tensors (native FP16 implementation)
     *
     * Operates directly on FP16 buffers without intermediate FP32 conversion.
     * Uses vectorized FP16 operations where available (F16C, AVX512 FP16, etc.).
     *
     * @param q_fp16 Query tensor in FP16 format (modified in-place)
     * @param k_fp16 Key tensor in FP16 format (modified in-place)
     * @param seq_len Sequence length
     * @param head_dim Dimension per head (must be even)
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads (may differ for GQA)
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE (model-specific)
     * @param persistent_state Optional persistent state for decode optimization
     */
    void apply_rope_fp16(
        uint16_t *q_fp16, uint16_t *k_fp16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Apply RoPE to Q and K tensors (INT32 not supported)
     *
     * RoPE is an activation operation and cannot be applied to quantized INT32 accumulators.
     * This function exists for API completeness but will always fail.
     *
     * @return Always false (operation not supported)
     */
    bool apply_rope_int32(
        int32_t *q_int32, int32_t *k_int32,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base);

    // ============================================================================
    // Individual Implementation Functions (for testing)
    // ============================================================================

    /**
     * @brief Apply RoPE rotation using scalar implementation (reference)
     *
     * This is the baseline implementation that all vectorized versions must match.
     * Used for testing and validation.
     *
     * @param head_ptr Pointer to head data [head_dim] (modified in-place)
     * @param position Token position
     * @param inv_freq Inverse frequencies [head_dim/2]
     * @param head_dim Dimension per head (must be even)
     * @param start_idx Starting pair index (for partial processing)
     */
    void apply_rope_to_head_scalar(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx = 0);

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation using AVX2 implementation
     *
     * Processes 8 float pairs at a time. Tail must be handled by scalar version.
     *
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_avx2(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation using AVX512 implementation
     *
     * Processes 16 float pairs at a time. Tail must be handled by scalar version.
     *
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_avx512(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

    // ============================================================================
    // BF16 Native Precision Implementations (for testing)
    // ============================================================================

    /**
     * @brief Apply RoPE rotation to a single head (BF16 scalar)
     */
    void apply_rope_to_head_bf16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx = 0);

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation to a single head (BF16 AVX2)
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_bf16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation to a single head (BF16 AVX512)
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_bf16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

    // ============================================================================
    // FP16 Native Precision Implementations (for testing)
    // ============================================================================

    /**
     * @brief Apply RoPE rotation to a single head (FP16 scalar)
     */
    void apply_rope_to_head_fp16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx = 0);

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation to a single head (FP16 AVX2)
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_fp16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation to a single head (FP16 AVX512)
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_fp16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

} // namespace llaminar2::primitives
