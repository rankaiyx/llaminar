/**
 * @file CPUSwiGLUKernel.h
 * @brief CPU implementation of SwiGLU activation
 *
 * SwiGLU: output = swish(gate) * up
 * where swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"

namespace llaminar2
{

    /**
     * @brief CPU implementation of SwiGLU kernel
     */
    class CPUSwiGLUKernel : public ITensorSwiGLU
    {
    public:
        CPUSwiGLUKernel() = default;
        ~CPUSwiGLUKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool apply(
            const float *gate, const float *up, float *output,
            int seq_len, int d_ff,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;
    };

} // namespace llaminar2
