/**
 * @file CPURoPEKernelTyped.cpp
 * @brief Implementation of typed RoPE kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This file implements the CPURoPEKernelTyped template specializations for
 * FP32, BF16, FP16, and Q8_1 precision types.
 *
 * Implementation mirrors CPURoPEKernelT - uses existing primitives with n_past
 * and handles position_ids at the kernel level.
 */

#include "CPURoPEKernelTyped.h"
#include "../primitives/RoPEPrimitives.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../utils/Logger.h"
#include <cstring>

namespace llaminar2
{

    // Thread-local state for decode optimization (mirrors CPURoPEKernelT)
    static thread_local primitives::RoPEPersistentState tls_state_;

    // ============================================================================
    // FP32 Specialization Implementation
    // ============================================================================

    bool CPURoPEKernelTyped<ActivationPrecision::FP32>::apply_typed(
        float *Q,
        float *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelTyped<FP32>: Null Q pointer");
            return false;
        }

        if (head_dim % 2 != 0)
        {
            LOG_ERROR("CPURoPEKernelTyped<FP32>: head_dim must be even");
            return false;
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        // Check for contiguous positions to enable optimized block processing
        bool contiguous = true;
        int start_pos = 0;

        if (position_ids)
        {
            start_pos = position_ids[0];
            if (seq_len > 1)
            {
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

        // If positions are contiguous, use optimized block processing
        if (contiguous && start_pos >= 0)
        {
            primitives::apply_rope_vectorized(
                Q, K,
                seq_len, head_dim,
                n_heads, n_kv_heads,
                start_pos, rope_theta,
                (seq_len == 1) ? &tls_state_ : nullptr);
            return true;
        }

        // Fall back to per-token processing for non-contiguous positions
        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue;
            }

            float *q_token = Q + tok * q_stride;
            float *k_token = K ? (K + tok * k_stride) : nullptr;

            // Apply RoPE to single token
            primitives::apply_rope_vectorized(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_);
        }

        return true;
    }

    // ============================================================================
    // BF16 Specialization Implementation
    // ============================================================================

    bool CPURoPEKernelTyped<ActivationPrecision::BF16>::apply_typed(
        uint16_t *Q,
        uint16_t *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelTyped<BF16>: Null Q pointer");
            return false;
        }

        if (head_dim % 2 != 0)
        {
            LOG_ERROR("CPURoPEKernelTyped<BF16>: head_dim must be even");
            return false;
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        // Check for contiguous positions to enable optimized block processing
        bool contiguous = true;
        int start_pos = 0;

        if (position_ids)
        {
            start_pos = position_ids[0];
            if (seq_len > 1)
            {
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

        // If positions are contiguous, use optimized block processing
        if (contiguous && start_pos >= 0)
        {
            primitives::apply_rope_bf16(
                Q, K,
                seq_len, head_dim,
                n_heads, n_kv_heads,
                start_pos, rope_theta,
                (seq_len == 1) ? &tls_state_ : nullptr);
            return true;
        }

        // Fall back to per-token processing for non-contiguous positions
        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue;
            }

            uint16_t *q_token = Q + tok * q_stride;
            uint16_t *k_token = K ? (K + tok * k_stride) : nullptr;

            // Apply RoPE to single token
            primitives::apply_rope_bf16(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_);
        }

        return true;
    }

    // ============================================================================
    // FP16 Specialization Implementation
    // ============================================================================

    bool CPURoPEKernelTyped<ActivationPrecision::FP16>::apply_typed(
        uint16_t *Q,
        uint16_t *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelTyped<FP16>: Null Q pointer");
            return false;
        }

        if (head_dim % 2 != 0)
        {
            LOG_ERROR("CPURoPEKernelTyped<FP16>: head_dim must be even");
            return false;
        }

        if (seq_len <= 0)
        {
            return true; // Nothing to do
        }

        const int q_stride = n_heads * head_dim;
        const int k_stride = n_kv_heads * head_dim;

        // Check for contiguous positions to enable optimized block processing
        bool contiguous = true;
        int start_pos = 0;

        if (position_ids)
        {
            start_pos = position_ids[0];
            if (seq_len > 1)
            {
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

        // If positions are contiguous, use optimized block processing
        if (contiguous && start_pos >= 0)
        {
            primitives::apply_rope_fp16(
                Q, K,
                seq_len, head_dim,
                n_heads, n_kv_heads,
                start_pos, rope_theta,
                (seq_len == 1) ? &tls_state_ : nullptr);
            return true;
        }

        // Fall back to per-token processing for non-contiguous positions
        for (int tok = 0; tok < seq_len; ++tok)
        {
            int position = position_ids ? position_ids[tok] : tok;

            // Skip RoPE for padding tokens (position_id = -1)
            if (position < 0)
            {
                continue;
            }

            uint16_t *q_token = Q + tok * q_stride;
            uint16_t *k_token = K ? (K + tok * k_stride) : nullptr;

            // Apply RoPE to single token
            primitives::apply_rope_fp16(
                q_token, k_token,
                1, head_dim,
                n_heads, n_kv_heads,
                position, rope_theta,
                &tls_state_);
        }

        return true;
    }

    // ============================================================================
    // Q8_1 Specialization Implementation (Pure-Integer)
    // ============================================================================

    bool CPURoPEKernelTyped<ActivationPrecision::Q8_1>::apply_typed(
        Q8_1Block *Q,
        Q8_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!Q)
        {
            LOG_ERROR("CPURoPEKernelTyped<Q8_1>: Null Q pointer");
            return false;
        }

        // Validate head_dim divisibility by Q8_1 block size
        if (head_dim % 32 != 0)
        {
            LOG_ERROR("CPURoPEKernelTyped<Q8_1>: head_dim ("
                      << head_dim << ") must be divisible by 32 (Q8_1 block size)");
            return false;
        }

        // Call pure-integer Q8_1 RoPE primitive
        primitives::apply_rope_q8_1_integer(
            Q, K,
            position_ids,
            seq_len,
            n_heads, n_kv_heads,
            head_dim,
            rope_theta,
            (seq_len == 1) ? &tls_state_ : nullptr);

        return true;
    }

    // ============================================================================
    // Destructor Definitions (required for vtable emission)
    // ============================================================================

    CPURoPEKernelTyped<ActivationPrecision::FP32>::~CPURoPEKernelTyped() = default;
    CPURoPEKernelTyped<ActivationPrecision::BF16>::~CPURoPEKernelTyped() = default;
    CPURoPEKernelTyped<ActivationPrecision::FP16>::~CPURoPEKernelTyped() = default;
    CPURoPEKernelTyped<ActivationPrecision::Q8_1>::~CPURoPEKernelTyped() = default;

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template class CPURoPEKernelTyped<ActivationPrecision::FP32>;
    template class CPURoPEKernelTyped<ActivationPrecision::BF16>;
    template class CPURoPEKernelTyped<ActivationPrecision::FP16>;
    template class CPURoPEKernelTyped<ActivationPrecision::Q8_1>;

} // namespace llaminar2
