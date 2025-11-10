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
        opts.parallel_threshold_elems = 2048;
        opts.t5_compat_mode = false;

        primitives::rmsnorm_fused_vectorized(
            input, gamma, output,
            seq_len, d_model,
            eps, opts);

        return true;
    }

    bool CPURMSNormKernel::apply_bf16(
        const uint16_t *input_bf16,
        const float *gamma,
        uint16_t *output_bf16,
        int seq_len,
        int d_model,
        float eps,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        // Use vectorized BF16 primitives implementation
        primitives::RMSNormExecOptions opts;
        opts.allow_parallel = true;
        opts.parallel_threshold_elems = 2048;
        opts.t5_compat_mode = false;

        primitives::rmsnorm_fused_bf16_vectorized(
            input_bf16, gamma, output_bf16,
            seq_len, d_model,
            eps, opts);

        return true;
    }

    bool CPURMSNormKernel::apply_fp16(
        const uint16_t *input_fp16,
        const float *gamma,
        uint16_t *output_fp16,
        int seq_len,
        int d_model,
        float eps,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        // Use vectorized FP16 primitives implementation
        primitives::RMSNormExecOptions opts;
        opts.allow_parallel = true;
        opts.parallel_threshold_elems = 2048;
        opts.t5_compat_mode = false;

        primitives::rmsnorm_fused_fp16_vectorized(
            input_fp16, gamma, output_fp16,
            seq_len, d_model,
            eps, opts);

        return true;
    }

    bool CPURMSNormKernel::apply_int32_to_int8(
        const int32_t *input_int32,
        const float *gamma,
        int8_t *output_int8,
        float *output_row_scales,
        int seq_len,
        int d_model,
        float eps,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        if (!input_int32 || !output_int8 || !output_row_scales)
        {
            return false;
        }

        if (seq_len <= 0 || d_model <= 0)
        {
            return false;
        }

        // Use vectorized INT32→INT8 primitives
        primitives::RMSNormExecOptions opts;
        opts.allow_parallel = true;
        opts.parallel_threshold_elems = 2048;
        opts.t5_compat_mode = false;

        primitives::rmsnorm_fused_int32_to_int8_vectorized(
            input_int32, gamma,
            output_int8, output_row_scales,
            seq_len, d_model,
            eps, opts);

        return true;
    }

} // namespace llaminar2
