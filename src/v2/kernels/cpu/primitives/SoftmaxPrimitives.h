/**
 * @file SoftmaxPrimitives.h
 * @brief Vectorized Softmax primitives for V2 (ported from V1)
 * @author David Sanftenberg
 *
 * High-performance Softmax implementation with:
 * - AVX512/AVX2 vectorization (3-5× speedup)
 * - Fused 3-pass algorithm (max → exp+sum → normalize)
 * - Fast exp approximation (optional)
 * - Causal masking support
 *
 * @deprecated Legacy FP32-only entry point kept for older kernels. New code
 *             should include `src/v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h`
 *             for the split scalar/AVX2/AVX512 implementations with FP32/BF16/FP16
 *             support.
 */
#pragma once

#include <cstddef>

namespace llaminar2::primitives
{

    /**
     * @brief Execution options for Softmax
     */
    struct SoftmaxExecOptions
    {
        bool force_scalar = false;           // Force scalar path (no SIMD)
        bool fast_exp = false;               // Use fast exp approximation
        bool validate = false;               // Validate fast exp against std::exp
        int parallel_elems_threshold = 8192; // Total elements threshold for parallelization
        int parallel_row_threshold = 4;      // Minimum rows for parallelization
    };

    /**
     * @brief Row-major softmax arguments
     */
    struct SoftmaxRowArgs
    {
        float *scores{nullptr}; // In-place scores [rows, cols]
        int rows{0};
        int cols{0};
        bool causal{false}; // Causal masking (j > i → 0 probability)
        float scale{1.0f};  // Optional multiplicative scale before exp
    };

    /**
     * @brief Apply row-major stable softmax (vectorized)
     *
     * Performs in-place softmax with:
     * - Numerical stability (max subtraction)
     * - Optional causal masking
     * - Optional scaling
     * - AVX512/AVX2 vectorization
     *
     * @param args Softmax arguments
     * @param opts Execution options
     */
    void softmax_row_major_vectorized(
        const SoftmaxRowArgs &args,
        const SoftmaxExecOptions &opts = {});

} // namespace llaminar2::primitives
