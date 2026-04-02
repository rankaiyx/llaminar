/**
 * @file IMoEExpertExecutor.h
 * @brief Interface for MoE expert FFN execution
 *
 * Decouples expert computation from stage infrastructure and device placement.
 * Implementations handle the actual FFN: gate_proj + up_proj → SwiGLU → down_proj.
 */

#pragma once

#include "MoETypes.h"
#include <cstddef>

namespace llaminar2
{

    /**
     * @brief Interface for executing a single expert's FFN on a batch of tokens
     *
     * Implementations may use CPU kernels, CUDA kernels, or even delegate
     * to a remote device.
     */
    class IMoEExpertExecutor
    {
    public:
        virtual ~IMoEExpertExecutor() = default;

        /**
         * @brief Execute a single expert's FFN on its assigned tokens
         *
         * Implements: SwiGLU(x @ gate_w, x @ up_w) @ down_w
         *
         * @param input     Full hidden states [seq_len, d_model], row-major
         * @param output    Output buffer [seq_len, d_model], row-major.
         *                  Only rows in batch.token_indices are written.
         * @param gate_w    Gate projection weights [intermediate, d_model]
         * @param up_w      Up projection weights [intermediate, d_model]
         * @param down_w    Down projection weights [d_model, intermediate]
         * @param batch     Which tokens to process and their routing weights
         * @param d_model   Hidden dimension
         * @param intermediate_dim Expert FFN intermediate dimension
         * @return true on success
         */
        virtual bool executeExpert(
            const float *input,
            float *output,
            const float *gate_w,
            const float *up_w,
            const float *down_w,
            const ExpertBatch &batch,
            int d_model,
            int intermediate_dim) = 0;
    };

    /**
     * @brief CPU reference implementation of expert FFN
     *
     * Simple nested-loop implementation for correctness testing.
     * Not optimized — production code should use GEMM kernels.
     */
    class CPUMoEExpertExecutor : public IMoEExpertExecutor
    {
    public:
        bool executeExpert(
            const float *input,
            float *output,
            const float *gate_w,
            const float *up_w,
            const float *down_w,
            const ExpertBatch &batch,
            int d_model,
            int intermediate_dim) override;
    };

} // namespace llaminar2
