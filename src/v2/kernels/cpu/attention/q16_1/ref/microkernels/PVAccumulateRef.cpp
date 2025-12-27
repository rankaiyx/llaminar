/**
 * @file PVAccumulateRef.cpp
 * @brief Reference implementation of P×V accumulation microkernel
 */

#include "PVAccumulateRef.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Helper: Dequantize V element
    // ============================================================================

    static inline float dequantize_v_element(
        const Q16_1Block *V,
        int kv_row,
        int v_col,
        int kv_blocks_per_row)
    {
        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

        int v_block_idx = v_col / BLOCK_SIZE;
        int v_elem_idx = v_col % BLOCK_SIZE;

        const Q16_1Block &v_block = V[kv_row * kv_blocks_per_row + v_block_idx];
        int16_t v_val = v_block.qs[v_elem_idx];

        return static_cast<float>(v_val) * v_block.d;
    }

    // ============================================================================
    // GEMV Variant (Decode)
    // ============================================================================

    void pv_accumulate_gemv_fp32(
        const PVAccumulateParams &params,
        const float *weights,
        int kv_start,
        int kv_count,
        float *output)
    {
        const int head_dim = params.head_dim;
        const int v_start_col = params.kv_head * head_dim;

        // Zero output
        std::memset(output, 0, head_dim * sizeof(float));

        // Accumulate weighted V
        for (int kv_local = 0; kv_local < kv_count; ++kv_local)
        {
            float w = weights[kv_local];
            if (w < 1e-10f)
                continue; // Skip near-zero weights

            int kv_row = kv_start + kv_local;

            for (int d = 0; d < head_dim; ++d)
            {
                int v_col = v_start_col + d;
                float actual_v = dequantize_v_element(
                    params.V, kv_row, v_col, params.kv_blocks_per_row);

                output[d] += w * actual_v;
            }
        }
    }

    void pv_accumulate_gemv_int16(
        const PVAccumulateParams &params,
        const int16_t *weights,
        int32_t weight_sum,
        int kv_start,
        int kv_count,
        float *output)
    {
        const int head_dim = params.head_dim;
        const int v_start_col = params.kv_head * head_dim;
        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

        // Zero output first
        std::memset(output, 0, head_dim * sizeof(float));

        if (weight_sum <= 0)
            return;

        // ════════════════════════════════════════════════════════════════════════
        // MIXED-PRECISION ACCUMULATION
        // ════════════════════════════════════════════════════════════════════════
        //
        // The key insight is that each V row can have DIFFERENT scale factors (d_v).
        // We cannot simply accumulate INT32 and apply a single d_v at the end.
        //
        // Correct formula:
        //   output[d] = Σ_kv (weight[kv] × v_qs[kv,d] × d_v[kv]) / weight_sum
        //
        // We use FP32 accumulators to handle the per-row d_v scaling correctly.
        // This is mathematically equivalent to the true weighted average.
        //
        // NOTE: For VNNI JIT, we would need to either:
        //   1. Ensure all V rows share the same d_v (require uniform quantization)
        //   2. Group by d_v and accumulate separately
        //   3. Use FP32 accumulators (as done here for reference correctness)
        // ════════════════════════════════════════════════════════════════════════

        // FP32 accumulators for correctness with variable d_v
        std::vector<float> acc(head_dim, 0.0f);

        // Accumulate with per-row d_v scaling
        for (int kv_local = 0; kv_local < kv_count; ++kv_local)
        {
            int16_t w = weights[kv_local];
            if (w == 0)
                continue;

            int kv_row = kv_start + kv_local;
            float w_fp = static_cast<float>(w);

            for (int d = 0; d < head_dim; ++d)
            {
                int v_col = v_start_col + d;
                int v_block_idx = v_col / BLOCK_SIZE;
                int v_elem_idx = v_col % BLOCK_SIZE;

                const Q16_1Block &v_block = params.V[kv_row * params.kv_blocks_per_row + v_block_idx];
                int16_t v_val = v_block.qs[v_elem_idx];
                float d_v = v_block.d;

                // Accumulate: weight × (v_qs × d_v) = weight × actual_v
                acc[d] += w_fp * static_cast<float>(v_val) * d_v;
            }
        }

        // Final normalization by weight_sum
        float inv_sum = 1.0f / static_cast<float>(weight_sum);
        for (int d = 0; d < head_dim; ++d)
        {
            output[d] = acc[d] * inv_sum;
        }
    }

    // ============================================================================
    // GEMM Variant (Prefill)
    // ============================================================================

    void pv_accumulate_gemm_int16(
        const PVAccumulateParams &params,
        const int16_t *weights,
        const int32_t *weight_sums,
        int num_queries,
        int kv_start,
        int kv_count,
        float *output)
    {
        const int head_dim = params.head_dim;
        const int v_start_col = params.kv_head * head_dim;
        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

        // Process each query row
        for (int q = 0; q < num_queries; ++q)
        {
            float *output_row = output + q * head_dim;
            const int16_t *weight_row = weights + q * kv_count;
            int32_t weight_sum = weight_sums[q];

            // Zero output row
            std::memset(output_row, 0, head_dim * sizeof(float));

            if (weight_sum <= 0)
                continue;

            // ════════════════════════════════════════════════════════════════════
            // MIXED-PRECISION ACCUMULATION
            // ════════════════════════════════════════════════════════════════════
            // Each V row can have different scale factors (d_v).
            // We use FP32 accumulators with per-row d_v scaling.
            // ════════════════════════════════════════════════════════════════════

            std::vector<float> acc(head_dim, 0.0f);

            for (int kv_local = 0; kv_local < kv_count; ++kv_local)
            {
                int16_t w = weight_row[kv_local];
                if (w == 0)
                    continue;

                int kv_row = kv_start + kv_local;
                float w_fp = static_cast<float>(w);

                for (int d = 0; d < head_dim; ++d)
                {
                    int v_col = v_start_col + d;
                    int v_block_idx = v_col / BLOCK_SIZE;
                    int v_elem_idx = v_col % BLOCK_SIZE;

                    const Q16_1Block &v_block = params.V[kv_row * params.kv_blocks_per_row + v_block_idx];
                    int16_t v_val = v_block.qs[v_elem_idx];
                    float d_v = v_block.d;

                    // Accumulate: weight × (v_qs × d_v) = weight × actual_v
                    acc[d] += w_fp * static_cast<float>(v_val) * d_v;
                }
            }

            // Final normalization
            float inv_sum = 1.0f / static_cast<float>(weight_sum);
            for (int d = 0; d < head_dim; ++d)
            {
                output_row[d] = acc[d] * inv_sum;
            }
        }
    }

} // namespace llaminar2::kernels::q16_1::microkernels
