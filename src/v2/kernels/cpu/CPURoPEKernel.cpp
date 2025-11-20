/**
 * @file CPURoPEKernel.cpp
 * @brief CPU RoPE kernel implementation (uses vectorized primitives)
 *
 * @author David Sanftenberg
 */

#include "CPURoPEKernel.h"
#include "primitives/RoPEPrimitives.h"
#include "../../tensors/SIMDHelpers.h"
#include "../../tensors/FP16Utils.h"
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <omp.h>

namespace llaminar2
{

    // Thread-local state for decode optimization
    thread_local primitives::RoPEPersistentState CPURoPEKernel::tls_state_;

    // Use primitives library for inverse frequency cache
    const std::vector<float> &CPURoPEKernel::get_inv_freq_cached(int head_dim, float freq_base)
    {
        return primitives::get_inv_freq_cached(head_dim, freq_base);
    }

    bool CPURoPEKernel::apply(
        float *Q, float *K,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)device_idx; // Device index ignored - always operates on CPU buffers

        if (head_dim % 2 != 0)
        {
            return false; // head_dim must be even
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        // Apply RoPE per-token with individual positions from position_ids array
        // This supports batched inference where sequences have independent positions
        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        // Check for contiguous positions to enable optimized block processing
        bool contiguous = true;
        int start_pos = 0;

        if (position_ids)
        {
            start_pos = position_ids[0];
            // Only check if we have more than 1 token
            if (seq_len > 1)
            {
                // Check contiguity
                for (int i = 1; i < seq_len; ++i)
                {
                    if (position_ids[i] != start_pos + i)
                    {
                        contiguous = false;
                        break;
                    }
                }
            }
        }
        else
        {
            // If position_ids is null, we assume 0, 1, 2... (standard prefill)
            contiguous = true;
            start_pos = 0;
        }

        // If positions are contiguous and valid (no padding), use optimized block processing
        // This enables OpenMP parallelism in the underlying primitives
        if (contiguous && start_pos >= 0)
        {
            apply_rotation(Q, K, seq_len, head_dim, n_heads, n_kv_heads, start_pos, rope_theta);
            return true;
        }

        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue; // Padding token - leave Q/K unrotated
            }

            float *q_token = Q + tok * q_stride;
            float *k_token = K ? (K + tok * k_stride) : nullptr;

            // Apply RoPE to single token (seq_len=1, n_past=position)
            apply_rotation(q_token, k_token, 1, head_dim, n_heads, n_kv_heads, position, rope_theta);
        }

        return true;
    }

    void CPURoPEKernel::apply_rotation(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base)
    {
        // Use vectorized primitives implementation
        primitives::apply_rope_vectorized(
            q, k,
            seq_len, head_dim,
            q_heads, k_heads,
            n_past, freq_base,
            (seq_len == 1) ? &tls_state_ : nullptr);
    }

    bool CPURoPEKernel::apply_bf16(
        uint16_t *Q_bf16, uint16_t *K_bf16,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU kernel only
        }

        if (head_dim % 2 != 0)
        {
            return false; // head_dim must be even
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        // Apply RoPE per-token with individual positions from position_ids array
        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue; // Padding token - leave Q/K unrotated
            }

            uint16_t *q_token = Q_bf16 + tok * q_stride;
            uint16_t *k_token = K_bf16 ? (K_bf16 + tok * k_stride) : nullptr;

            // Apply RoPE to single token (seq_len=1, n_past=position)
            primitives::apply_rope_bf16(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_); // Always use persistent state for single-token
        }

        return true;
    }

    bool CPURoPEKernel::apply_fp16(
        uint16_t *Q_fp16, uint16_t *K_fp16,
        const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        float rope_theta,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU kernel only
        }

        if (head_dim % 2 != 0)
        {
            return false; // head_dim must be even
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        // Apply RoPE per-token with individual positions from position_ids array
        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue; // Padding token - leave Q/K unrotated
            }

            uint16_t *q_token = Q_fp16 + tok * q_stride;
            uint16_t *k_token = K_fp16 ? (K_fp16 + tok * k_stride) : nullptr;

            // Apply RoPE to single token (seq_len=1, n_past=position)
            primitives::apply_rope_fp16(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_); // Always use persistent state for single-token
        }

        return true;
    }

} // namespace llaminar2
