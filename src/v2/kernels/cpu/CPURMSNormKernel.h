/**
 * @file CPURMSNormKernel.h
 * @brief CPU implementation of RMS normalization
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"

namespace llaminar2
{

    /**
     * @brief CPU implementation of RMSNorm kernel
     */
    class CPURMSNormKernel : public ITensorRMSNorm
    {
    public:
        CPURMSNormKernel() = default;
        ~CPURMSNormKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool apply(
            const float *input, const float *gamma, float *output,
            int seq_len, int d_model,
            float eps = 1e-6f,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;
    };

} // namespace llaminar2
