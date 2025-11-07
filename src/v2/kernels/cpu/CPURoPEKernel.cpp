/**
 * @file CPURoPEKernel.cpp
 * @brief CPU RoPE kernel implementation (uses vectorized primitives)
 *
 * @author David Sanftenberg
 */

#include "CPURoPEKernel.h"
#include "primitives/RoPEPrimitives.h"
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

        // Use rope_theta from model config (Qwen2.5: 1000000.0, LLaMA: 10000.0)
        apply_rotation(Q, K, seq_len, head_dim, n_heads, n_kv_heads, n_past, rope_theta);

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

} // namespace llaminar2
