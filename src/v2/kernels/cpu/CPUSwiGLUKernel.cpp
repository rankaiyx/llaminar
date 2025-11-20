/**
 * @file CPUSwiGLUKernel.cpp
 * @brief CPU SwiGLU kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPUSwiGLUKernel.h"
#include "primitives/SwiGLUPrimitives.h"
#include <cmath>
#include <omp.h>

namespace llaminar2
{

    bool CPUSwiGLUKernel::apply(
        const float *gate, const float *up, float *output,
        int seq_len, int d_ff,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)device_idx; // Device index ignored - always operates on CPU buffers

        const int total_elements = seq_len * d_ff;

        // Use vectorized primitives
        primitives::compute_swiglu(gate, up, output, total_elements);

        return true;
    }

} // namespace llaminar2
