/**
 * @file CPURMSNormTypedKernel.cpp
 * @brief Typed RMSNorm kernel implementation with fused precision conversion
 *
 * Key optimization: Process data in row-sized chunks that fit in L1/L2 cache.
 * Never allocate full-sized FP32 scratch buffers - stream through the data.
 *
 * @author David Sanftenberg
 * @date 2025-12-04
 */

#include "CPURMSNormTypedKernel.h"
#include "../primitives/RMSNormPrimitives.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../utils/Logger.h"

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar2
{

    // =========================================================================
    // Constants for cache-friendly streaming
    // =========================================================================

    // Maximum row size we support with stack-allocated scratch (16KB per row)
    // This covers d_model up to 4096 (16KB) which fits comfortably in L1 (32KB)
    // For larger models, we fall back to heap allocation per-thread
    static constexpr size_t MAX_STACK_ROW_SIZE = 4096;

    // =========================================================================
    // FP32 Specialization Implementation
    // =========================================================================

    bool CPURMSNormTypedKernel<ActivationPrecision::FP32>::apply_typed(
        const float *input,
        const float *gamma,
        float *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Direct FP32 path - no conversion needed
        primitives::rmsnorm_fused_vectorized(
            input, gamma, output,
            static_cast<size_t>(rows),
            static_cast<size_t>(cols),
            epsilon);

        return true;
    }

    bool CPURMSNormTypedKernel<ActivationPrecision::FP32>::apply_with_residual_add(
        const float *residual,
        const float *fp32_input,
        const float *gamma,
        float *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);

        // Process row-by-row to keep scratch in L1/L2
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            // Thread-local scratch buffer (stack for small cols, heap for large)
            std::array<float, MAX_STACK_ROW_SIZE> stack_scratch;
            std::vector<float> heap_scratch;
            float *scratch = nullptr;

            if (ucols <= MAX_STACK_ROW_SIZE)
            {
                scratch = stack_scratch.data();
            }
            else
            {
                heap_scratch.resize(ucols);
                scratch = heap_scratch.data();
            }

#ifdef _OPENMP
#pragma omp for
#endif
            for (int row = 0; row < rows; ++row)
            {
                const float *res_row = residual + row * ucols;
                const float *in_row = fp32_input + row * ucols;
                float *out_row = output + row * ucols;

                // Fused add into scratch (stays in L1)
                for (size_t i = 0; i < ucols; ++i)
                {
                    scratch[i] = res_row[i] + in_row[i];
                }

                // RMSNorm on single row (scratch → out_row)
                primitives::rmsnorm_fused_row_avx512(scratch, gamma, out_row, ucols, epsilon);
            }
        }

        return true;
    }

    // =========================================================================
    // BF16 Specialization Implementation (Cache-Streaming)
    // =========================================================================

    bool CPURMSNormTypedKernel<ActivationPrecision::BF16>::apply_typed(
        const uint16_t *input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);

        // Process row-by-row: BF16 → FP32 (L1) → RMSNorm → FP32 (L1) → BF16
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            // Thread-local scratch buffers for one row
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fp32_in, heap_fp32_out;
            float *fp32_in = nullptr;
            float *fp32_out = nullptr;

            if (ucols <= MAX_STACK_ROW_SIZE)
            {
                fp32_in = stack_fp32_in.data();
                fp32_out = stack_fp32_out.data();
            }
            else
            {
                heap_fp32_in.resize(ucols);
                heap_fp32_out.resize(ucols);
                fp32_in = heap_fp32_in.data();
                fp32_out = heap_fp32_out.data();
            }

#ifdef _OPENMP
#pragma omp for
#endif
            for (int row = 0; row < rows; ++row)
            {
                const uint16_t *in_row = input + row * ucols;
                uint16_t *out_row = output + row * ucols;

                // 1. Dequantize BF16 → FP32 (one row, stays in L1)
                simd::convert_bf16_to_fp32(in_row, fp32_in, ucols);

                // 2. RMSNorm in FP32 (one row)
                primitives::rmsnorm_fused_row_avx512(fp32_in, gamma, fp32_out, ucols, epsilon);

                // 3. Quantize FP32 → BF16 (one row)
                simd::convert_fp32_to_bf16(fp32_out, out_row, ucols);
            }
        }

        return true;
    }

    bool CPURMSNormTypedKernel<ActivationPrecision::BF16>::apply_with_residual_add(
        const uint16_t *residual,
        const float *fp32_input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            // Thread-local scratch for fused input and output (two rows worth)
            std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fused, heap_fp32_out;
            float *fused = nullptr;
            float *fp32_out = nullptr;

            if (ucols <= MAX_STACK_ROW_SIZE)
            {
                fused = stack_fused.data();
                fp32_out = stack_fp32_out.data();
            }
            else
            {
                heap_fused.resize(ucols);
                heap_fp32_out.resize(ucols);
                fused = heap_fused.data();
                fp32_out = heap_fp32_out.data();
            }

#ifdef _OPENMP
#pragma omp for
#endif
            for (int row = 0; row < rows; ++row)
            {
                const uint16_t *res_row = residual + row * ucols;
                const float *in_row = fp32_input + row * ucols;
                uint16_t *out_row = output + row * ucols;

                // 1. Fused dequant + add (one row): fused = dequant(res_bf16) + fp32_in
                simd::fused_bf16_residual_add(res_row, in_row, fused, ucols);

                // 2. RMSNorm in FP32 (one row)
                primitives::rmsnorm_fused_row_avx512(fused, gamma, fp32_out, ucols, epsilon);

                // 3. Quantize FP32 → BF16 (one row)
                simd::convert_fp32_to_bf16(fp32_out, out_row, ucols);
            }
        }

        return true;
    }

    // =========================================================================
    // FP16 Specialization Implementation (Cache-Streaming)
    // =========================================================================

    bool CPURMSNormTypedKernel<ActivationPrecision::FP16>::apply_typed(
        const uint16_t *input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fp32_in, heap_fp32_out;
            float *fp32_in = nullptr;
            float *fp32_out = nullptr;

            if (ucols <= MAX_STACK_ROW_SIZE)
            {
                fp32_in = stack_fp32_in.data();
                fp32_out = stack_fp32_out.data();
            }
            else
            {
                heap_fp32_in.resize(ucols);
                heap_fp32_out.resize(ucols);
                fp32_in = heap_fp32_in.data();
                fp32_out = heap_fp32_out.data();
            }

#ifdef _OPENMP
#pragma omp for
#endif
            for (int row = 0; row < rows; ++row)
            {
                const uint16_t *in_row = input + row * ucols;
                uint16_t *out_row = output + row * ucols;

                // 1. Dequantize FP16 → FP32 (one row)
                simd::convert_fp16_to_fp32(in_row, fp32_in, ucols);

                // 2. RMSNorm in FP32 (one row)
                primitives::rmsnorm_fused_row_avx512(fp32_in, gamma, fp32_out, ucols, epsilon);

                // 3. Quantize FP32 → FP16 (one row)
                simd::convert_fp32_to_fp16(fp32_out, out_row, ucols);
            }
        }

        return true;
    }

    bool CPURMSNormTypedKernel<ActivationPrecision::FP16>::apply_with_residual_add(
        const uint16_t *residual,
        const float *fp32_input,
        const float *gamma,
        uint16_t *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fused, heap_fp32_out;
            float *fused = nullptr;
            float *fp32_out = nullptr;

            if (ucols <= MAX_STACK_ROW_SIZE)
            {
                fused = stack_fused.data();
                fp32_out = stack_fp32_out.data();
            }
            else
            {
                heap_fused.resize(ucols);
                heap_fp32_out.resize(ucols);
                fused = heap_fused.data();
                fp32_out = heap_fp32_out.data();
            }

#ifdef _OPENMP
#pragma omp for
#endif
            for (int row = 0; row < rows; ++row)
            {
                const uint16_t *res_row = residual + row * ucols;
                const float *in_row = fp32_input + row * ucols;
                uint16_t *out_row = output + row * ucols;

                // 1. Fused dequant + add (one row)
                simd::fused_fp16_residual_add(res_row, in_row, fused, ucols);

                // 2. RMSNorm in FP32 (one row)
                primitives::rmsnorm_fused_row_avx512(fused, gamma, fp32_out, ucols, epsilon);

                // 3. Quantize FP32 → FP16 (one row)
                simd::convert_fp32_to_fp16(fp32_out, out_row, ucols);
            }
        }

        return true;
    }

    // =========================================================================
    // Q8_1 Specialization Implementation (Cache-Streaming)
    // =========================================================================

    bool CPURMSNormTypedKernel<ActivationPrecision::Q8_1>::apply_typed(
        const Q8_1Block *input,
        const float *gamma,
        Q8_1Block *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Q8_1 requires cols to be multiple of 32 (block size)
        if (cols % 32 != 0)
        {
            LOG_ERROR("Q8_1 RMSNorm requires cols to be multiple of 32, got " << cols);
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const size_t blocks_per_row = ucols / 32;

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fp32_in, heap_fp32_out;
            float *fp32_in = nullptr;
            float *fp32_out = nullptr;

            if (ucols <= MAX_STACK_ROW_SIZE)
            {
                fp32_in = stack_fp32_in.data();
                fp32_out = stack_fp32_out.data();
            }
            else
            {
                heap_fp32_in.resize(ucols);
                heap_fp32_out.resize(ucols);
                fp32_in = heap_fp32_in.data();
                fp32_out = heap_fp32_out.data();
            }

#ifdef _OPENMP
#pragma omp for
#endif
            for (int row = 0; row < rows; ++row)
            {
                const Q8_1Block *in_row = input + row * blocks_per_row;
                Q8_1Block *out_row = output + row * blocks_per_row;

                // 1. Dequantize Q8_1 → FP32 (one row)
                simd::dequantize_q8_1_to_fp32(in_row, fp32_in, ucols);

                // 2. RMSNorm in FP32 (one row)
                primitives::rmsnorm_fused_row_avx512(fp32_in, gamma, fp32_out, ucols, epsilon);

                // 3. Quantize FP32 → Q8_1 (one row)
                simd::quantize_fp32_to_q8_1_blocks(fp32_out, out_row, ucols);
            }
        }

        return true;
    }

    bool CPURMSNormTypedKernel<ActivationPrecision::Q8_1>::apply_with_residual_add(
        const Q8_1Block *residual,
        const float *fp32_input,
        const float *gamma,
        Q8_1Block *output,
        int rows,
        int cols,
        float epsilon,
        int device_idx)
    {
        (void)device_idx;

        if (!residual || !fp32_input || !gamma || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Q8_1 requires cols to be multiple of 32 (block size)
        if (cols % 32 != 0)
        {
            LOG_ERROR("Q8_1 RMSNorm requires cols to be multiple of 32, got " << cols);
            return false;
        }

        const size_t ucols = static_cast<size_t>(cols);
        const size_t blocks_per_row = ucols / 32;

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            std::array<float, MAX_STACK_ROW_SIZE> stack_fused;
            std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
            std::vector<float> heap_fused, heap_fp32_out;
            float *fused = nullptr;
            float *fp32_out = nullptr;

            if (ucols <= MAX_STACK_ROW_SIZE)
            {
                fused = stack_fused.data();
                fp32_out = stack_fp32_out.data();
            }
            else
            {
                heap_fused.resize(ucols);
                heap_fp32_out.resize(ucols);
                fused = heap_fused.data();
                fp32_out = heap_fp32_out.data();
            }

#ifdef _OPENMP
#pragma omp for
#endif
            for (int row = 0; row < rows; ++row)
            {
                const Q8_1Block *res_row = residual + row * blocks_per_row;
                const float *in_row = fp32_input + row * ucols;
                Q8_1Block *out_row = output + row * blocks_per_row;

                // 1. Fused dequant + add (one row)
                simd::fused_q8_1_residual_add(res_row, in_row, fused, ucols);

                // 2. RMSNorm in FP32 (one row)
                primitives::rmsnorm_fused_row_avx512(fused, gamma, fp32_out, ucols, epsilon);

                // 3. Quantize FP32 → Q8_1 (one row)
                simd::quantize_fp32_to_q8_1_blocks(fp32_out, out_row, ucols);
            }
        }

        return true;
    }

} // namespace llaminar2
