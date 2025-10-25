/**
 * @file CPURoPEKernel.h
 * @brief CPU implementation of rotary position embeddings (uses vectorized primitives)
 *
 * Optimized RoPE implementation with:
 * - AVX2/AVX512 vectorization (8-16× speedup)
 * - Cached inverse frequencies
 * - Persistent decode state (complex recurrence)
 * - OpenMP parallelization for prefill
 * - Angle recurrence across tokens
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "primitives/RoPEPrimitives.h"
#include <vector>
#include <cmath>

namespace llaminar2
{

    /**
     * @brief CPU implementation of RoPE kernel
     *
     * Uses vectorized primitives from V1.
     */
    class CPURoPEKernel : public ITensorRoPE
    {
    public:
        CPURoPEKernel() = default;
        ~CPURoPEKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool apply(
            float *Q, float *K,
            const int *position_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

    private:
        // Thread-local state for decode (uses primitives type)
        thread_local static primitives::RoPEPersistentState tls_state_;

        // Inverse frequency cache (delegates to primitives)
        static const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base);

        // Core rotation implementation (delegates to primitives)
        static void apply_rotation(
            float *q, float *k,
            int seq_len, int head_dim,
            int q_heads, int k_heads,
            int n_past, float freq_base);
    };

} // namespace llaminar2
