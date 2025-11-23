/**
 * @file FusedRMSNormQuantize.cpp
 * @brief Implementation of fused RMSNorm + INT8 quantization kernel
 * @author David Sanftenberg
 * @date 2025-11-22
 */

#include "FusedRMSNormQuantize.h"
#include "../primitives/RMSNormPrimitives.h"
#include "../../../utils/Logger.h"
#include "../../../utils/CPUFeatures.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

namespace llaminar2
{

    bool FusedRMSNormQuantize::execute(
        const float *input,
        const float *weight,
        int8_t *output,
        float *scales,
        int rows,
        int cols,
        float epsilon,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;    // Unused for now
        (void)device_idx; // Must be -1 (CPU)

        // Validate inputs
        if (!input || !weight || !output || !scales)
        {
            LOG_ERROR("[FusedRMSNormQuantize] Null pointer in inputs");
            return false;
        }

        if (rows <= 0 || cols <= 0)
        {
            LOG_ERROR("[FusedRMSNormQuantize] Invalid dimensions: rows="
                      << rows << ", cols=" << cols);
            return false;
        }

        // Process rows in parallel
#pragma omp parallel for
        for (int i = 0; i < rows; ++i)
        {
            const float *input_row = input + i * cols;
            int8_t *output_row = output + i * cols;
            float &scale = scales[i];

            process_row_fused(input_row, weight, output_row, scale, cols, epsilon);
        }

        return true;
    }

    void FusedRMSNormQuantize::process_row_fused(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        // Dispatch to SIMD implementation at compile time (march=native)
#if defined(__AVX512F__)
        primitives::rmsnorm_quantize_row_avx512(input_row, weight, output_row, out_scale, cols, epsilon);
#elif defined(__AVX2__)
        primitives::rmsnorm_quantize_row_avx2(input_row, weight, output_row, out_scale, cols, epsilon);
#else
        primitives::rmsnorm_quantize_row_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
#endif
    }

} // namespace llaminar2
