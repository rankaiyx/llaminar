/**
 * @file RoPEPrimitives.cpp
 * @brief Vectorized RoPE implementation (ported from V1)
 * @author David Sanftenberg
 */

#include "RoPEPrimitives.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/FP16Utils.h"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives
{

    // ============================================================================
    // Inverse Frequency Cache
    // ============================================================================

    namespace
    {
        static std::unordered_map<uint64_t, std::vector<float>> g_inv_freq_cache;
        static std::mutex g_inv_freq_mutex;

        uint64_t make_cache_key(int head_dim, float freq_base)
        {
            uint32_t freq_bits;
            std::memcpy(&freq_bits, &freq_base, sizeof(float));
            return ((uint64_t)head_dim << 32) | freq_bits;
        }
    } // anonymous namespace

    const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base)
    {
        uint64_t key = make_cache_key(head_dim, freq_base);

        {
            std::lock_guard<std::mutex> lock(g_inv_freq_mutex);
            auto it = g_inv_freq_cache.find(key);
            if (it != g_inv_freq_cache.end())
            {
                return it->second;
            }
        }

        // Compute inverse frequencies
        const int half_dim = head_dim / 2;
        std::vector<float> inv_freq(half_dim);
        const float log_base = std::log(freq_base);

        for (int i = 0; i < half_dim; ++i)
        {
            float exponent = (2.0f * i) / head_dim;
            inv_freq[i] = std::exp(-log_base * exponent);
        }

        std::lock_guard<std::mutex> lock(g_inv_freq_mutex);
        auto res = g_inv_freq_cache.emplace(key, std::move(inv_freq));
        return res.first->second;
    }

    // ============================================================================
    // Vectorized RoPE Application - Separated Implementations
    // ============================================================================

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation using cached sin/cos (AVX2)
     */
    void apply_rope_to_head_cached_avx2(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 8 <= half_dim; i += 8)
        {
            __m256 x_first = _mm256_loadu_ps(head_ptr + i);
            __m256 x_second = _mm256_loadu_ps(head_ptr + i + half_dim);
            __m256 cos_vec = _mm256_loadu_ps(cos_cache + i);
            __m256 sin_vec = _mm256_loadu_ps(sin_cache + i);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            _mm256_storeu_ps(head_ptr + i, new_first);
            _mm256_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];
            float cos_val = cos_cache[i];
            float sin_val = sin_cache[i];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation using cached sin/cos (AVX512)
     */
    void apply_rope_to_head_cached_avx512(
        float *head_ptr,
        const float *cos_cache,
        const float *sin_cache,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        for (; i + 16 <= half_dim; i += 16)
        {
            __m512 x_first = _mm512_loadu_ps(head_ptr + i);
            __m512 x_second = _mm512_loadu_ps(head_ptr + i + half_dim);
            __m512 cos_vec = _mm512_loadu_ps(cos_cache + i);
            __m512 sin_vec = _mm512_loadu_ps(sin_cache + i);

            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            _mm512_storeu_ps(head_ptr + i, new_first);
            _mm512_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        // Tail
        for (; i < half_dim; ++i)
        {
            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];
            float cos_val = cos_cache[i];
            float sin_val = sin_cache[i];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }
#endif

    /**
     * @brief Apply RoPE rotation to a single head (scalar implementation)
     *
     * This is the reference implementation used for:
     * - Scalar-only builds
     * - Tail processing after vectorized loops
     * - Testing/validation
     */
    void apply_rope_to_head_scalar(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
#if defined(__GNUC__)
            float sin_val, cos_val;
            sincosf(angle, &sin_val, &cos_val);
#else
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
#endif

            float x_first = head_ptr[i];
            float x_second = head_ptr[i + half_dim];

            head_ptr[i] = x_first * cos_val - x_second * sin_val;
            head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
        }
    }

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation to a single head (AVX2 implementation)
     *
     * Processes 8 float pairs at a time using AVX2 intrinsics
     *
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_avx2(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
#if defined(__GNUC__)
                sincosf(angles[lane], &sin_vals[lane], &cos_vals[lane]);
#else
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
#endif
            }

            // Load and rotate
            __m256 x_first = _mm256_loadu_ps(head_ptr + i);
            __m256 x_second = _mm256_loadu_ps(head_ptr + i + half_dim);
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first, cos_vec),
                _mm256_mul_ps(x_second, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first, sin_vec),
                _mm256_mul_ps(x_second, cos_vec));

            _mm256_storeu_ps(head_ptr + i, new_first);
            _mm256_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation to a single head (AVX512 implementation)
     *
     * Processes 16 float pairs at a time using AVX512 intrinsics
     *
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_avx512(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
#if defined(__GNUC__)
                sincosf(angles[lane], &sin_vals[lane], &cos_vals[lane]);
#else
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
#endif
            }

            // Load values
            __m512 x_first = _mm512_loadu_ps(head_ptr + i);
            __m512 x_second = _mm512_loadu_ps(head_ptr + i + half_dim);
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate: new_first = x_first * cos - x_second * sin
            //         new_second = x_first * sin + x_second * cos
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first, cos_vec),
                _mm512_mul_ps(x_second, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first, sin_vec),
                _mm512_mul_ps(x_second, cos_vec));

            _mm512_storeu_ps(head_ptr + i, new_first);
            _mm512_storeu_ps(head_ptr + i + half_dim, new_second);
        }

        return i;
    }
#endif

    /**
     * @brief Apply RoPE rotation to a single head (dispatches to best available implementation)
     */
    static void apply_rope_to_head_vectorized(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if defined(__AVX512F__)
        processed = apply_rope_to_head_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    /**
     * @brief Apply RoPE to tensor with angle recurrence (prefill optimization)
     *
     * Uses pre-computed sin/cos tables generated via recurrence to avoid
     * expensive trigonometric calls in the hot loop.
     */
    static void apply_rope_to_tensor_recurrence(
        float *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
        const int half_dim = head_dim / 2;

        // 1. Pre-compute sin/cos tables for the whole sequence using recurrence
        // This avoids computing sin/cos for every head and every token repeatedly
        std::vector<float> cos_table(seq_len * half_dim);
        std::vector<float> sin_table(seq_len * half_dim);

        // Compute deltas (rotation per position step)
        std::vector<float> cos_delta(half_dim);
        std::vector<float> sin_delta(half_dim);
        for (int i = 0; i < half_dim; ++i)
        {
            cos_delta[i] = std::cos(inv_freq[i]);
            sin_delta[i] = std::sin(inv_freq[i]);
        }

        // Initialize first position (n_past)
        for (int i = 0; i < half_dim; ++i)
        {
            float ang = n_past * inv_freq[i];
#if defined(__GNUC__)
            sincosf(ang, &sin_table[i], &cos_table[i]);
#else
            cos_table[i] = std::cos(ang);
            sin_table[i] = std::sin(ang);
#endif
        }

        // Recurrence for t > 0
        // This is serial but extremely fast (vectorizable by compiler)
        for (int t = 1; t < seq_len; ++t)
        {
            int prev_offset = (t - 1) * half_dim;
            int curr_offset = t * half_dim;

            for (int i = 0; i < half_dim; ++i)
            {
                float c = cos_table[prev_offset + i];
                float s = sin_table[prev_offset + i];
                float cd = cos_delta[i];
                float sd = sin_delta[i];

                cos_table[curr_offset + i] = c * cd - s * sd;
                sin_table[curr_offset + i] = s * cd + c * sd;
            }
        }

        // 2. Apply rotation in parallel using cached tables
#pragma omp parallel for collapse(2) schedule(static)
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                float *head_ptr = tensor + (t * num_heads + h) * head_dim;
                const float *cos_ptr = cos_table.data() + t * half_dim;
                const float *sin_ptr = sin_table.data() + t * half_dim;

#if defined(__AVX512F__)
                apply_rope_to_head_cached_avx512(head_ptr, cos_ptr, sin_ptr, head_dim);
#elif defined(__AVX2__)
                apply_rope_to_head_cached_avx2(head_ptr, cos_ptr, sin_ptr, head_dim);
#else
                // Fallback scalar implementation
                for (int i = 0; i < half_dim; ++i)
                {
                    float x_first = head_ptr[i];
                    float x_second = head_ptr[i + half_dim];
                    float cos_val = cos_ptr[i];
                    float sin_val = sin_ptr[i];

                    head_ptr[i] = x_first * cos_val - x_second * sin_val;
                    head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
                }
#endif
            }
        }
    }

    /**
     * @brief Apply RoPE with persistent state (single-token decode optimization)
     */
    static void apply_rope_with_persistent_state(
        float *q, float *k,
        int q_heads, int k_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq,
        RoPEPersistentState &state)
    {
        const int half_dim = head_dim / 2;
        int target_pos = n_past;

        // Initialize or reset if position jumped
        if (state.last_pos == -1 || target_pos < state.last_pos)
        {
            state.cos_curr.resize(half_dim);
            state.sin_curr.resize(half_dim);
            state.cos_delta.resize(half_dim);
            state.sin_delta.resize(half_dim);

            for (int i = 0; i < half_dim; ++i)
            {
                const float delta = inv_freq[i];
                state.cos_delta[i] = std::cos(delta);
                state.sin_delta[i] = std::sin(delta);

                const float ang = target_pos * inv_freq[i];
#if defined(__GNUC__)
                sincosf(ang, &state.sin_curr[i], &state.cos_curr[i]);
#else
                state.cos_curr[i] = std::cos(ang);
                state.sin_curr[i] = std::sin(ang);
#endif
            }
            state.last_pos = target_pos;
        }
        // Advance via complex recurrence
        else if (target_pos > state.last_pos)
        {
            int steps = target_pos - state.last_pos;
            for (int step = 0; step < steps; ++step)
            {
                for (int i = 0; i < half_dim; ++i)
                {
                    float c = state.cos_curr[i];
                    float s = state.sin_curr[i];
                    float cd = state.cos_delta[i];
                    float sd = state.sin_delta[i];

                    state.cos_curr[i] = c * cd - s * sd;
                    state.sin_curr[i] = s * cd + c * sd;
                }
            }
            state.last_pos = target_pos;
        }

        // Apply rotation to all heads using cached sin/cos
        auto apply_to_heads = [&](float *tensor, int num_heads)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                float *head_ptr = tensor + h * head_dim;
#if defined(__AVX512F__)
                apply_rope_to_head_cached_avx512(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#elif defined(__AVX2__)
                apply_rope_to_head_cached_avx2(head_ptr, state.cos_curr.data(), state.sin_curr.data(), head_dim);
#else
                for (int i = 0; i < half_dim; ++i)
                {
                    float x_first = head_ptr[i];
                    float x_second = head_ptr[i + half_dim];
                    float cos_val = state.cos_curr[i];
                    float sin_val = state.sin_curr[i];

                    head_ptr[i] = x_first * cos_val - x_second * sin_val;
                    head_ptr[i + half_dim] = x_first * sin_val + x_second * cos_val;
                }
#endif
            }
        };

        apply_to_heads(q, q_heads);
        apply_to_heads(k, k_heads);
    }

    // ============================================================================
    // Public API
    // ============================================================================

    void apply_rope_vectorized(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        // Single-token decode with persistent state
        if (seq_len == 1 && persistent_state)
        {
            // Reset state if parameters changed
            if (persistent_state->cached_head_dim != head_dim ||
                persistent_state->cached_freq_base != freq_base)
            {
                persistent_state->cached_head_dim = head_dim;
                persistent_state->cached_freq_base = freq_base;
                persistent_state->reset();
            }

            apply_rope_with_persistent_state(
                q, k, q_heads, k_heads, head_dim,
                n_past, inv_freq, *persistent_state);
        }
        else
        {
            // Prefill or no persistent state: use angle recurrence
            apply_rope_to_tensor_recurrence(q, seq_len, q_heads, head_dim, n_past, inv_freq);
            apply_rope_to_tensor_recurrence(k, seq_len, k_heads, head_dim, n_past, inv_freq);
        }
    }

    // ============================================================================
    // BF16 Native Precision Implementations
    // ============================================================================

    void apply_rope_to_head_bf16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
#if defined(__GNUC__)
            float sin_val, cos_val;
            sincosf(angle, &sin_val, &cos_val);
#else
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
#endif

            // Convert BF16 → FP32
            float x_first = simd::bf16_to_fp32(head_ptr[i]);
            float x_second = simd::bf16_to_fp32(head_ptr[i + half_dim]);

            // Rotate
            float new_first = x_first * cos_val - x_second * sin_val;
            float new_second = x_first * sin_val + x_second * cos_val;

            // Convert FP32 → BF16
            head_ptr[i] = simd::fp32_to_bf16(new_first);
            head_ptr[i + half_dim] = simd::fp32_to_bf16(new_second);
        }
    }

#if defined(__AVX2__)
    int apply_rope_to_head_bf16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
#if defined(__GNUC__)
                sincosf(angles[lane], &sin_vals[lane], &cos_vals[lane]);
#else
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
#endif
            }

            // Load BF16 values
            alignas(32) uint16_t first_bf16[8];
            alignas(32) uint16_t second_bf16[8];
            std::memcpy(first_bf16, head_ptr + i, 8 * sizeof(uint16_t));
            std::memcpy(second_bf16, head_ptr + i + half_dim, 8 * sizeof(uint16_t));

            // Convert BF16 → FP32
            alignas(32) float x_first[8];
            alignas(32) float x_second[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                x_first[lane] = simd::bf16_to_fp32(first_bf16[lane]);
                x_second[lane] = simd::bf16_to_fp32(second_bf16[lane]);
            }

            // Load and rotate
            __m256 x_first_vec = _mm256_loadu_ps(x_first);
            __m256 x_second_vec = _mm256_loadu_ps(x_second);
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first_vec, cos_vec),
                _mm256_mul_ps(x_second_vec, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first_vec, sin_vec),
                _mm256_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm256_storeu_ps(x_first, new_first);
            _mm256_storeu_ps(x_second, new_second);

            // Convert FP32 → BF16
            for (int lane = 0; lane < 8; ++lane)
            {
                first_bf16[lane] = simd::fp32_to_bf16(x_first[lane]);
                second_bf16[lane] = simd::fp32_to_bf16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_bf16, 8 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_bf16, 8 * sizeof(uint16_t));
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    int apply_rope_to_head_bf16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
#if defined(__GNUC__)
                sincosf(angles[lane], &sin_vals[lane], &cos_vals[lane]);
#else
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
#endif
            }

            // Load BF16 values
            alignas(64) uint16_t first_bf16[16];
            alignas(64) uint16_t second_bf16[16];
            std::memcpy(first_bf16, head_ptr + i, 16 * sizeof(uint16_t));
            std::memcpy(second_bf16, head_ptr + i + half_dim, 16 * sizeof(uint16_t));

            // Convert BF16 → FP32
            alignas(64) float x_first[16];
            alignas(64) float x_second[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                x_first[lane] = simd::bf16_to_fp32(first_bf16[lane]);
                x_second[lane] = simd::bf16_to_fp32(second_bf16[lane]);
            }

            // Load values
            __m512 x_first_vec = _mm512_loadu_ps(x_first);
            __m512 x_second_vec = _mm512_loadu_ps(x_second);
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first_vec, cos_vec),
                _mm512_mul_ps(x_second_vec, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first_vec, sin_vec),
                _mm512_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm512_storeu_ps(x_first, new_first);
            _mm512_storeu_ps(x_second, new_second);

            // Convert FP32 → BF16
            for (int lane = 0; lane < 16; ++lane)
            {
                first_bf16[lane] = simd::fp32_to_bf16(x_first[lane]);
                second_bf16[lane] = simd::fp32_to_bf16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_bf16, 16 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_bf16, 16 * sizeof(uint16_t));
        }

        return i;
    }
#endif

    static void apply_rope_to_head_bf16_vectorized(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if defined(__AVX512F__)
        processed = apply_rope_to_head_bf16_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_bf16_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_bf16_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    static void apply_rope_to_tensor_bf16(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                int position = n_past + t;
                apply_rope_to_head_bf16_vectorized(head_ptr, position, inv_freq, head_dim);
            }
        }
    }

    void apply_rope_bf16(
        uint16_t *q_bf16, uint16_t *k_bf16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        // For BF16, we don't use persistent state optimization
        // (Would require BF16 sin/cos storage which adds complexity)
        apply_rope_to_tensor_bf16(q_bf16, seq_len, q_heads, head_dim, n_past, inv_freq);
        apply_rope_to_tensor_bf16(k_bf16, seq_len, k_heads, head_dim, n_past, inv_freq);
    }

    // ============================================================================
    // FP16 Native Precision Implementations
    // ============================================================================

    void apply_rope_to_head_fp16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx)
    {
        const int half_dim = head_dim / 2;

        for (int i = start_idx; i < half_dim; ++i)
        {
            float angle = position * inv_freq[i];
#if defined(__GNUC__)
            float sin_val, cos_val;
            sincosf(angle, &sin_val, &cos_val);
#else
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
#endif

            // Convert FP16 → FP32
            float x_first = fp16_to_fp32(head_ptr[i]);
            float x_second = fp16_to_fp32(head_ptr[i + half_dim]);

            // Rotate
            float new_first = x_first * cos_val - x_second * sin_val;
            float new_second = x_first * sin_val + x_second * cos_val;

            // Convert FP32 → FP16
            head_ptr[i] = fp32_to_fp16(new_first);
            head_ptr[i + half_dim] = fp32_to_fp16(new_second);
        }
    }

#if defined(__AVX2__)
    int apply_rope_to_head_fp16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 8 pairs at a time
        for (; i + 8 <= half_dim; i += 8)
        {
            // Compute angles
            alignas(32) float angles[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(32) float cos_vals[8];
            alignas(32) float sin_vals[8];
            for (int lane = 0; lane < 8; ++lane)
            {
#if defined(__GNUC__)
                sincosf(angles[lane], &sin_vals[lane], &cos_vals[lane]);
#else
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
#endif
            }

            // Load FP16 values
            alignas(32) uint16_t first_fp16[8];
            alignas(32) uint16_t second_fp16[8];
            std::memcpy(first_fp16, head_ptr + i, 8 * sizeof(uint16_t));
            std::memcpy(second_fp16, head_ptr + i + half_dim, 8 * sizeof(uint16_t));

            // Convert FP16 → FP32
            alignas(32) float x_first[8];
            alignas(32) float x_second[8];
            for (int lane = 0; lane < 8; ++lane)
            {
                x_first[lane] = fp16_to_fp32(first_fp16[lane]);
                x_second[lane] = fp16_to_fp32(second_fp16[lane]);
            }

            // Load and rotate
            __m256 x_first_vec = _mm256_loadu_ps(x_first);
            __m256 x_second_vec = _mm256_loadu_ps(x_second);
            __m256 cos_vec = _mm256_loadu_ps(cos_vals);
            __m256 sin_vec = _mm256_loadu_ps(sin_vals);

            __m256 new_first = _mm256_sub_ps(
                _mm256_mul_ps(x_first_vec, cos_vec),
                _mm256_mul_ps(x_second_vec, sin_vec));
            __m256 new_second = _mm256_add_ps(
                _mm256_mul_ps(x_first_vec, sin_vec),
                _mm256_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm256_storeu_ps(x_first, new_first);
            _mm256_storeu_ps(x_second, new_second);

            // Convert FP32 → FP16
            for (int lane = 0; lane < 8; ++lane)
            {
                first_fp16[lane] = fp32_to_fp16(x_first[lane]);
                second_fp16[lane] = fp32_to_fp16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_fp16, 8 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_fp16, 8 * sizeof(uint16_t));
        }

        return i;
    }
#endif

#if defined(__AVX512F__)
    int apply_rope_to_head_fp16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        const int half_dim = head_dim / 2;
        int i = 0;

        // Process 16 pairs at a time
        for (; i + 16 <= half_dim; i += 16)
        {
            // Compute angles for 16 pairs
            alignas(64) float angles[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                angles[lane] = position * inv_freq[i + lane];
            }

            // Compute sin/cos
            alignas(64) float cos_vals[16];
            alignas(64) float sin_vals[16];
            for (int lane = 0; lane < 16; ++lane)
            {
#if defined(__GNUC__)
                sincosf(angles[lane], &sin_vals[lane], &cos_vals[lane]);
#else
                cos_vals[lane] = std::cos(angles[lane]);
                sin_vals[lane] = std::sin(angles[lane]);
#endif
            }

            // Load FP16 values
            alignas(64) uint16_t first_fp16[16];
            alignas(64) uint16_t second_fp16[16];
            std::memcpy(first_fp16, head_ptr + i, 16 * sizeof(uint16_t));
            std::memcpy(second_fp16, head_ptr + i + half_dim, 16 * sizeof(uint16_t));

            // Convert FP16 → FP32
            alignas(64) float x_first[16];
            alignas(64) float x_second[16];
            for (int lane = 0; lane < 16; ++lane)
            {
                x_first[lane] = fp16_to_fp32(first_fp16[lane]);
                x_second[lane] = fp16_to_fp32(second_fp16[lane]);
            }

            // Load values
            __m512 x_first_vec = _mm512_loadu_ps(x_first);
            __m512 x_second_vec = _mm512_loadu_ps(x_second);
            __m512 cos_vec = _mm512_loadu_ps(cos_vals);
            __m512 sin_vec = _mm512_loadu_ps(sin_vals);

            // Rotate
            __m512 new_first = _mm512_sub_ps(
                _mm512_mul_ps(x_first_vec, cos_vec),
                _mm512_mul_ps(x_second_vec, sin_vec));
            __m512 new_second = _mm512_add_ps(
                _mm512_mul_ps(x_first_vec, sin_vec),
                _mm512_mul_ps(x_second_vec, cos_vec));

            // Store back to FP32 buffers
            _mm512_storeu_ps(x_first, new_first);
            _mm512_storeu_ps(x_second, new_second);

            // Convert FP32 → FP16
            for (int lane = 0; lane < 16; ++lane)
            {
                first_fp16[lane] = fp32_to_fp16(x_first[lane]);
                second_fp16[lane] = fp32_to_fp16(x_second[lane]);
            }

            // Write back
            std::memcpy(head_ptr + i, first_fp16, 16 * sizeof(uint16_t));
            std::memcpy(head_ptr + i + half_dim, second_fp16, 16 * sizeof(uint16_t));
        }

        return i;
    }
#endif

    static void apply_rope_to_head_fp16_vectorized(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim)
    {
        int processed = 0;

#if defined(__AVX512F__)
        processed = apply_rope_to_head_fp16_avx512(head_ptr, position, inv_freq, head_dim);
#elif defined(__AVX2__)
        processed = apply_rope_to_head_fp16_avx2(head_ptr, position, inv_freq, head_dim);
#endif

        // Scalar tail
        apply_rope_to_head_fp16_scalar(head_ptr, position, inv_freq, head_dim, processed);
    }

    static void apply_rope_to_tensor_fp16(
        uint16_t *tensor,
        int seq_len, int num_heads, int head_dim,
        int n_past,
        const std::vector<float> &inv_freq)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                uint16_t *head_ptr = tensor + (t * num_heads + h) * head_dim;
                int position = n_past + t;
                apply_rope_to_head_fp16_vectorized(head_ptr, position, inv_freq, head_dim);
            }
        }
    }

    void apply_rope_fp16(
        uint16_t *q_fp16, uint16_t *k_fp16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state)
    {
        if (head_dim % 2 != 0)
        {
            return; // head_dim must be even
        }

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        // For FP16, we don't use persistent state optimization
        apply_rope_to_tensor_fp16(q_fp16, seq_len, q_heads, head_dim, n_past, inv_freq);
        apply_rope_to_tensor_fp16(k_fp16, seq_len, k_heads, head_dim, n_past, inv_freq);
    }

    // ============================================================================
    // INT32 Stub (Not Supported)
    // ============================================================================

    bool apply_rope_int32(
        int32_t *q_int32, int32_t *k_int32,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base)
    {
        // RoPE is an activation operation and cannot be applied to quantized INT32 accumulators
        return false;
    }

} // namespace llaminar2::primitives
