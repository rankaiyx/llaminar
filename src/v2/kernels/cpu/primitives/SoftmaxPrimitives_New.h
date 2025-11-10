/**
 * @file SoftmaxPrimitives.h
 * @brief Multi-precision Softmax primitives with separated SIMD variants
 * @author David Sanftenberg
 *
 * V2 Softmax implementation with:
 * - Native precision support (FP32, BF16, FP16)
 * - Separated scalar/AVX2/AVX512 variants for testing
 * - Causal masking support
 * - Numerical stability (max subtraction)
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace llaminar2::primitives
{

    // ============================================================================
    // FP32 Softmax Primitives
    // ============================================================================

    /**
     * @brief Softmax for single FP32 row (scalar implementation)
     *
     * @param row Input/output [cols] (in-place)
     * @param cols Row length
     * @param causal Apply causal masking (j > i → 0 probability)
     * @param scale Scale factor applied before exp
     * @param row_idx Row index (for causal masking: only needed if causal=true)
     *
     * Algorithm:
     *   1. max_val = max(row) with causal masking
     *   2. sum = Σ exp(row[j] - max_val) for valid j
     *   3. row[j] = exp(row[j] - max_val) / sum
     */
    inline void softmax_row_fp32_scalar(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP32 row (AVX2 implementation)
     */
    inline void softmax_row_fp32_avx2(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP32 row (AVX512 implementation)
     */
    inline void softmax_row_fp32_avx512(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP32 row (compile-time dispatch)
     *
     * Dispatches to best available SIMD implementation.
     */
    inline void softmax_row_fp32(
        float *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    // ============================================================================
    // BF16 Softmax Primitives
    // ============================================================================

    /**
     * @brief Softmax for single BF16 row (scalar implementation)
     *
     * Native BF16 computation:
     *   - Convert BF16 → FP32 for exp/div (unavoidable)
     *   - Accumulation in FP32 for numerical stability
     *   - Convert FP32 → BF16 for output
     *
     * @param row Input/output [cols] (in-place, BF16 as uint16_t)
     * @param cols Row length
     * @param causal Apply causal masking
     * @param scale Scale factor
     * @param row_idx Row index (for causal masking)
     */
    inline void softmax_row_bf16_scalar(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single BF16 row (AVX2 implementation)
     *
     * Uses _mm256_cvtne2ps_pbh (AVX512_BF16) if available, otherwise manual conversion.
     */
    inline void softmax_row_bf16_avx2(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single BF16 row (AVX512 implementation)
     *
     * Uses native AVX512_BF16 instructions for conversion.
     */
    inline void softmax_row_bf16_avx512(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single BF16 row (compile-time dispatch)
     */
    inline void softmax_row_bf16(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    // ============================================================================
    // FP16 Softmax Primitives
    // ============================================================================

    /**
     * @brief Softmax for single FP16 row (scalar implementation)
     *
     * Native FP16 computation:
     *   - Convert FP16 → FP32 for exp/div
     *   - Accumulation in FP32 for numerical stability
     *   - Convert FP32 → FP16 for output
     *
     * @param row Input/output [cols] (in-place, FP16 as uint16_t)
     * @param cols Row length
     * @param causal Apply causal masking
     * @param scale Scale factor
     * @param row_idx Row index (for causal masking)
     */
    inline void softmax_row_fp16_scalar(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP16 row (AVX2 implementation)
     *
     * Uses F16C instructions (_mm256_cvtph_ps, _mm256_cvtps_ph).
     */
    inline void softmax_row_fp16_avx2(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP16 row (AVX512 implementation)
     *
     * Uses AVX512FP16 if available, otherwise falls back to F16C.
     */
    inline void softmax_row_fp16_avx512(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    /**
     * @brief Softmax for single FP16 row (compile-time dispatch)
     */
    inline void softmax_row_fp16(
        uint16_t *row,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        int row_idx = 0);

    // ============================================================================
    // Row-Major Batch Softmax (Multi-row)
    // ============================================================================

    /**
     * @brief Apply softmax to multiple FP32 rows (row-major layout)
     *
     * @param scores Input/output [rows, cols] (in-place)
     * @param rows Number of rows
     * @param cols Number of columns per row
     * @param causal Apply causal masking (row i: only j <= i valid)
     * @param scale Scale factor applied before exp
     * @param parallel Enable OpenMP parallelization
     */
    inline void softmax_row_major_fp32(
        float *scores,
        int rows,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        bool parallel = true)
    {
#pragma omp parallel for if (parallel)
        for (int r = 0; r < rows; ++r)
        {
            softmax_row_fp32(scores + r * cols, cols, causal, scale, r);
        }
    }

    /**
     * @brief Apply softmax to multiple BF16 rows (row-major layout)
     */
    inline void softmax_row_major_bf16(
        uint16_t *scores,
        int rows,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        bool parallel = true)
    {
#pragma omp parallel for if (parallel)
        for (int r = 0; r < rows; ++r)
        {
            softmax_row_bf16(scores + r * cols, cols, causal, scale, r);
        }
    }

    /**
     * @brief Apply softmax to multiple FP16 rows (row-major layout)
     */
    inline void softmax_row_major_fp16(
        uint16_t *scores,
        int rows,
        int cols,
        bool causal = false,
        float scale = 1.0f,
        bool parallel = true)
    {
#pragma omp parallel for if (parallel)
        for (int r = 0; r < rows; ++r)
        {
            softmax_row_fp16(scores + r * cols, cols, causal, scale, r);
        }
    }

} // namespace llaminar2::primitives

// ============================================================================
// Inline Implementations
// ============================================================================

#include "SoftmaxPrimitivesImpl.h"
