/**
 * @file CPUSoftmaxKernel.cpp
 * @brief CPU Softmax kernel implementation (uses vectorized primitives)
 *
 * @author David Sanftenberg
 */

#include "CPUSoftmaxKernel.h"
#include "primitives/SoftmaxPrimitives.h"
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
        (void)device_idx; // Device index ignored - always operates on CPU buffers

        // Copy input to output if different (softmax is typically in-place)
        if (input != output)
        {
            std::copy(input, input + rows * cols, output);
        }

        // Use vectorized primitives implementation
        primitives::SoftmaxRowArgs args;
        args.scores = output;
        args.rows = rows;
        args.cols = cols;
        args.causal = false;
        args.scale = 1.0f;

        primitives::SoftmaxExecOptions opts;
        opts.force_scalar = false;
        opts.fast_exp = false; // Can enable for 2-3× speedup
        opts.parallel_elems_threshold = 8192;
        opts.parallel_row_threshold = 4;

        primitives::softmax_row_major_vectorized(args, opts);

        return true;
    }

} // namespace llaminar2
