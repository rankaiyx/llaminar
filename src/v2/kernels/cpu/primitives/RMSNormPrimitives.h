/**
 * @file RMSNormPrimitives.h
 * @brief Vectorized RMSNorm primitives for V2 (ported from V1)
 * @author David Sanftenberg
 *
 * High-performance RMSNorm implementation with:
 * - AVX512/AVX2 vectorization (4-8× speedup)
 * - Double precision accumulation for accuracy
 * - Thread-local scratch buffers
 * - T5 compatibility mode
 */
#pragma once

#include <cstddef>
#include <vector>

namespace llaminar2::primitives
{

    /**
     * @brief Execution options for RMSNorm
     */
    struct RMSNormExecOptions
    {
        bool allow_parallel = true;                  // Permit OpenMP parallelization
        bool force_scalar = false;                   // Force scalar path (no SIMD)
        std::size_t parallel_threshold_elems = 2048; // Threshold for parallelization
        bool t5_compat_mode = false;                 // Use float32 accumulation for T5 parity
    };

    /**
     * @brief Reusable scratch buffers to avoid per-call allocations
     */
    struct RMSNormScratch
    {
        std::vector<double> row_sumsq; // Per-row sum of squares
        std::vector<float> inv;        // Per-row inverse RMS

        void ensure(std::size_t rows)
        {
            if (row_sumsq.size() < rows)
                row_sumsq.resize(rows);
            if (inv.size() < rows)
                inv.resize(rows);
        }

        void clear()
        {
            row_sumsq.clear();
            inv.clear();
        }
    };

    /**
     * @brief Compute per-row sum of squares (vectorized)
     *
     * @param src Input tensor [rows, cols]
     * @param rows Number of rows
     * @param cols Number of columns
     * @param row_sumsq Output per-row sum of squares (length = rows)
     * @param opts Execution options
     */
    void rmsnorm_compute_row_sumsq_vectorized(
        const float *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Convert row sum of squares to inverse RMS scaling factors
     *
     * @param row_sumsq Per-row sum of squares
     * @param rows Number of rows
     * @param cols Number of columns (for mean calculation)
     * @param epsilon Epsilon for numerical stability
     * @param inv_out Output inverse RMS values (length = rows)
     */
    void rmsnorm_compute_inv(
        const double *row_sumsq,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        float *inv_out);

    /**
     * @brief Apply RMSNorm scaling and gamma weights (vectorized)
     *
     * @param src Input tensor [rows, cols]
     * @param gamma Gamma weights [cols]
     * @param inv Per-row inverse RMS values
     * @param rows Number of rows
     * @param cols Number of columns
     * @param dst Output tensor [rows, cols]
     * @param opts Execution options
     */
    void rmsnorm_apply_vectorized(
        const float *src,
        const float *gamma,
        const float *inv,
        std::size_t rows,
        std::size_t cols,
        float *dst,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Fused RMSNorm computation (all stages)
     *
     * @param src Input tensor [rows, cols]
     * @param gamma Gamma weights [cols]
     * @param dst Output tensor [rows, cols]
     * @param rows Number of rows
     * @param cols Number of columns
     * @param epsilon Epsilon for numerical stability
     * @param opts Execution options
     */
    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts = {});

    /**
     * @brief Fused RMSNorm with provided scratch buffer
     */
    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        RMSNormScratch &scratch,
        const RMSNormExecOptions &opts = {});

} // namespace llaminar2::primitives
