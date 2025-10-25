/**
 * @file FP32StandaloneGemm.h
 * @brief Standalone FP32 GEMM kernel (no weight binding)
 *
 * Unlike FP32GemmKernel, this kernel accepts B matrix as a parameter,
 * making it suitable for attention operations where B changes per call.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../utils/MPIContext.h"

namespace llaminar2
{

    /**
     * @brief Standalone FP32 GEMM kernel for attention operations
     *
     * Performs C = alpha * A @ B^T + beta * C (if transpose_B = true)
     *         or C = alpha * A @ B + beta * C (if transpose_B = false)
     *
     * Unlike FP32GemmKernel, this accepts B as a parameter to multiply_with_b(),
     * allowing different B matrices per call (needed for attention Q·K^T and scores·V).
     */
    class FP32StandaloneGemm
    {
    public:
        /**
         * @brief Perform standalone GEMM operation
         *
         * @param A Input matrix A [m, k]
         * @param B Input matrix B [n, k] if transpose_B, else [k, n]
         * @param C Output matrix C [m, n]
         * @param m Number of rows in A and C
         * @param n Number of columns in B and C
         * @param k Number of columns in A and rows in B
         * @param transpose_B If true, B is [n, k] and we compute A @ B^T
         * @param alpha Scaling factor for product
         * @param beta Scaling factor for C (0.0f for overwrite)
         * @return true if successful, false otherwise
         */
        static bool multiply_with_b(
            const float *A,
            const float *B,
            float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f,
            float beta = 0.0f);
    };

} // namespace llaminar2
