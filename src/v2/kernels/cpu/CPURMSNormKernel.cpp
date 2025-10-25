/**
 * @file CPURMSNormKernel.cpp
 * @brief CPU RMSNorm kernel implementation (uses vectorized primitives)
 *
 * @author David Sanftenberg
 */

#include "CPURMSNormKernel.h"
#include "primitives/RMSNormPrimitives.h"
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

        // Use vectorized primitives implementation
        primitives::RMSNormExecOptions opts;
        opts.allow_parallel = true;
        opts.force_scalar = false;
        opts.parallel_threshold_elems = 2048;
        opts.t5_compat_mode = false;

        primitives::rmsnorm_fused_vectorized(
            input, gamma, output,
            seq_len, d_model,
            eps, opts);

        return true;
    }

} // namespace llaminar2
