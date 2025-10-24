/**
 * @file CPUSoftmaxKernel.cpp
 * @brief CPU Softmax kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPUSoftmaxKernel.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace llaminar2
{

    bool CPUSoftmaxKernel::apply(
        const float *input, float *output,
        int rows, int cols,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

#pragma omp parallel for
        for (int r = 0; r < rows; ++r)
        {
            const float *in_row = input + r * cols;
            float *out_row = output + r * cols;

            // Find max for numerical stability
            float max_val = in_row[0];
            for (int c = 1; c < cols; ++c)
            {
                max_val = std::max(max_val, in_row[c]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c)
            {
                out_row[c] = std::exp(in_row[c] - max_val);
                sum += out_row[c];
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (int c = 0; c < cols; ++c)
            {
                out_row[c] *= inv_sum;
            }
        }

        return true;
    }

} // namespace llaminar2
