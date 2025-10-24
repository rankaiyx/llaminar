/**
 * @file CPURMSNormKernel.cpp
 * @brief CPU RMSNorm kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPURMSNormKernel.h"
#include <cmath>
#include <omp.h>

namespace llaminar2
{

    bool CPURMSNormKernel::apply(
        const float *input, const float *gamma, float *output,
        int seq_len, int d_model,
        float eps,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

#pragma omp parallel for
        for (int t = 0; t < seq_len; ++t)
        {
            const float *in_row = input + t * d_model;
            float *out_row = output + t * d_model;

            // Compute RMS
            float sum_sq = 0.0f;
            for (int i = 0; i < d_model; ++i)
            {
                sum_sq += in_row[i] * in_row[i];
            }
            float rms = std::sqrt(sum_sq / d_model + eps);
            float scale = 1.0f / rms;

            // Normalize and apply gamma
            for (int i = 0; i < d_model; ++i)
            {
                out_row[i] = in_row[i] * scale * gamma[i];
            }
        }

        return true;
    }

} // namespace llaminar2
