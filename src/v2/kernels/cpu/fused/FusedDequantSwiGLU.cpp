/**
 * @file FusedDequantSwiGLU.cpp
 * @brief Implementation of fused dequantization + SwiGLU activation
 * @author David Sanftenberg
 * @date 2025-11-23
 */

#include "FusedDequantSwiGLU.h"
#include "../../../utils/Logger.h"
#include "../primitives/SwiGLUPrimitives.h"
#include <algorithm>
#include <vector>

namespace llaminar2
{
    bool FusedDequantSwiGLU::execute(
        const int32_t *gate_int32,
        const int32_t *up_int32,
        float *output,
        const float *activation_scales,
        const float *gate_col_scales,
        const float *up_col_scales,
        int m, int n)
    {
        if (!gate_int32 || !up_int32 || !output || !activation_scales ||
            !gate_col_scales || !up_col_scales)
        {
            LOG_ERROR("[FusedDequantSwiGLU] Null pointer in execute()");
            return false;
        }

        if (m <= 0 || n <= 0)
        {
            LOG_ERROR("[FusedDequantSwiGLU] Invalid dimensions: m=" << m << " n=" << n);
            return false;
        }

        // Allocate temporary buffers for dequantized gate and up projections
        std::vector<float> gate_fp32(static_cast<size_t>(m) * static_cast<size_t>(n));
        std::vector<float> up_fp32(static_cast<size_t>(m) * static_cast<size_t>(n));

        // Step 1: Dequantize gate and up projections (INT32 → FP32)
        for (int i = 0; i < m; ++i)
        {
            const int32_t *gate_row = gate_int32 + i * n;
            const int32_t *up_row = up_int32 + i * n;
            float *gate_fp32_row = gate_fp32.data() + i * n;
            float *up_fp32_row = up_fp32.data() + i * n;
            const float row_scale = activation_scales[i];

            for (int j = 0; j < n; ++j)
            {
                // Dequantize: fp32 = int32 * row_scale * col_scale
                gate_fp32_row[j] = static_cast<float>(gate_row[j]) * row_scale * gate_col_scales[j];
                up_fp32_row[j] = static_cast<float>(up_row[j]) * row_scale * up_col_scales[j];
            }
        }

        // Step 2: Apply SwiGLU using well-tested primitives
        // SwiGLU: output = gate * silu(up)
        primitives::compute_swiglu(gate_fp32.data(), up_fp32.data(), output, m * n);

        return true;
    }

} // namespace llaminar2
