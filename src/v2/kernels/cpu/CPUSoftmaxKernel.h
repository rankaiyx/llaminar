/**
 * @file CPUSoftmaxKernel.h
 * @brief CPU implementation of softmax normalization
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"

namespace llaminar2
{

    /**
     * @brief CPU implementation of Softmax kernel
     */
    class CPUSoftmaxKernel : public ITensorSoftmax
    {
    public:
        CPUSoftmaxKernel() = default;
        ~CPUSoftmaxKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool apply(
            const float *input, float *output,
            int rows, int cols,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;
    };

} // namespace llaminar2
