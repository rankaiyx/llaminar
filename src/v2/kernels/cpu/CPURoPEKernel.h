/**
 * @file CPURoPEKernel.h
 * @brief CPU implementation of rotary position embeddings
 *
 * Optimized RoPE implementation with:
 * - Cached inverse frequencies
 * - Persistent decode state (single-token recurrence)
 * - OpenMP parallelization for prefill
 * - AVX2/AVX512 vectorization
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include <vector>
#include <cmath>

namespace llaminar2
{

    /**
     * @brief CPU implementation of RoPE kernel
     *
     * Ported from V1's optimized AttentionPrimitives implementation.
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
        /**
         * @brief Persistent state for single-token decode optimization
         */
        struct PersistentState
        {
            int last_pos = -1;
            int cached_head_dim = 0;
            float cached_freq_base = 0.f;
            std::vector<float> cos_curr;
            std::vector<float> sin_curr;
            std::vector<float> cos_delta;
            std::vector<float> sin_delta;
        };

        // Thread-local state for decode
        thread_local static PersistentState tls_state_;

        // Inverse frequency cache
        static const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base);

        // Core rotation implementation
        static void apply_rotation(
            float *q, float *k,
            int seq_len, int head_dim,
            int q_heads, int k_heads,
            int n_past, float freq_base);
    };

} // namespace llaminar2
