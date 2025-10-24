/**
 * @file CPURoPEKernel.cpp
 * @brief CPU RoPE kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPURoPEKernel.h"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace llaminar2
{

    // Thread-local state for decode optimization
    thread_local CPURoPEKernel::PersistentState CPURoPEKernel::tls_state_;

    // Inverse frequency cache
    static std::unordered_map<uint64_t, std::vector<float>> g_inv_freq_cache;

    const std::vector<float> &CPURoPEKernel::get_inv_freq_cached(int head_dim, float freq_base)
    {
        // Use (head_dim << 32 | freq_base_bits) as cache key
        uint64_t freq_bits;
        std::memcpy(&freq_bits, &freq_base, sizeof(float));
        uint64_t key = ((uint64_t)head_dim << 32) | (freq_bits & 0xFFFFFFFFULL);

        auto it = g_inv_freq_cache.find(key);
        if (it != g_inv_freq_cache.end())
        {
            return it->second;
        }

        // Build inverse frequency vector
        std::vector<float> inv;
        inv.reserve(head_dim / 2);
        const float log_base = std::log(freq_base);

        for (int i = 0; i < head_dim / 2; ++i)
        {
            float exponent = (2.f * i) / head_dim;
            inv.push_back(std::exp(-log_base * exponent));
        }

        auto res = g_inv_freq_cache.emplace(key, std::move(inv));
        return res.first->second;
    }

    bool CPURoPEKernel::apply(
        float *Q, float *K,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU kernel only supports device_idx = -1
        }

        if (head_dim % 2 != 0)
        {
            return false; // head_dim must be even
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        // For now, assume contiguous position_ids [n_past, n_past+1, ..., n_past+seq_len-1]
        // This matches V1 behavior where position_ids isn't actually used (n_past determines base)
        int n_past = position_ids ? position_ids[0] : 0;

        // Standard freq_base for Qwen/LLaMA models
        const float freq_base = 10000.0f;

        apply_rotation(Q, K, seq_len, head_dim, n_heads, n_kv_heads, n_past, freq_base);

        return true;
    }

    void CPURoPEKernel::apply_rotation(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base)
    {
        const int half = head_dim / 2;
        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

        // Single-token decode fast path with persistent state
        if (seq_len == 1)
        {
            auto &state = tls_state_;

            // Reset state if parameters changed
            if (state.cached_head_dim != head_dim ||
                state.cached_freq_base != freq_base ||
                (int)state.cos_curr.size() != half)
            {
                state.cached_head_dim = head_dim;
                state.cached_freq_base = freq_base;
                state.last_pos = -1;
                state.cos_curr.assign(half, 0.f);
                state.sin_curr.assign(half, 0.f);
                state.cos_delta.assign(half, 0.f);
                state.sin_delta.assign(half, 0.f);
            }

            int target_pos = n_past;

            // Initialize or reset if position jumped
            if (state.last_pos == -1 || target_pos < state.last_pos)
            {
                for (int i = 0; i < half; ++i)
                {
                    const float delta = inv_freq[i];
                    state.cos_delta[i] = std::cos(delta);
                    state.sin_delta[i] = std::sin(delta);

                    const float ang = target_pos * inv_freq[i];
                    state.cos_curr[i] = std::cos(ang);
                    state.sin_curr[i] = std::sin(ang);
                }
                state.last_pos = target_pos;
            }
            // Advance via complex recurrence
            else if (target_pos > state.last_pos)
            {
                int steps = target_pos - state.last_pos;
                for (int step = 0; step < steps; ++step)
                {
                    for (int i = 0; i < half; ++i)
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

            // Apply rotation to all heads (single token)
            for (int h = 0; h < q_heads; ++h)
            {
                float *head_q = q + h * head_dim;
                for (int i = 0; i < half; ++i)
                {
                    float q_i = head_q[i];
                    float q_j = head_q[i + half];
                    float c = state.cos_curr[i];
                    float s = state.sin_curr[i];

                    head_q[i] = q_i * c - q_j * s;
                    head_q[i + half] = q_i * s + q_j * c;
                }
            }

            for (int h = 0; h < k_heads; ++h)
            {
                float *head_k = k + h * head_dim;
                for (int i = 0; i < half; ++i)
                {
                    float k_i = head_k[i];
                    float k_j = head_k[i + half];
                    float c = state.cos_curr[i];
                    float s = state.sin_curr[i];

                    head_k[i] = k_i * c - k_j * s;
                    head_k[i + half] = k_i * s + k_j * c;
                }
            }
        }
        // Multi-token prefill path with angle recurrence
        else
        {
#pragma omp parallel for collapse(2)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < q_heads; ++h)
                {
                    int position = n_past + t;
                    float *head_q = q + (t * q_heads + h) * head_dim;

                    for (int i = 0; i < half; ++i)
                    {
                        float ang = position * inv_freq[i];
                        float c = std::cos(ang);
                        float s = std::sin(ang);

                        float q_i = head_q[i];
                        float q_j = head_q[i + half];

                        head_q[i] = q_i * c - q_j * s;
                        head_q[i + half] = q_i * s + q_j * c;
                    }
                }
            }

#pragma omp parallel for collapse(2)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < k_heads; ++h)
                {
                    int position = n_past + t;
                    float *head_k = k + (t * k_heads + h) * head_dim;

                    for (int i = 0; i < half; ++i)
                    {
                        float ang = position * inv_freq[i];
                        float c = std::cos(ang);
                        float s = std::sin(ang);

                        float k_i = head_k[i];
                        float k_j = head_k[i + half];

                        head_k[i] = k_i * c - k_j * s;
                        head_k[i + half] = k_i * s + k_j * c;
                    }
                }
            }
        }
    }

} // namespace llaminar2
