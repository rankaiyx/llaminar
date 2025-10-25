/**
 * @file FP32StandaloneGemm.cpp
 * @brief Implementation of standalone FP32 GEMM kernel
 *
 * @author David Sanftenberg
 */

#include "FP32StandaloneGemm.h"
#include "../../utils/Logger.h"
#include <cmath>

#ifdef HAVE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace llaminar2
{

    bool FP32StandaloneGemm::multiply_with_b(
        const float *A,
        const float *B,
        float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha,
        float beta)
    {
        // Validate inputs
        if (!A || !B || !C)
        {
            LOG_ERROR("[FP32StandaloneGemm] Null pointer: A=" << A << " B=" << B << " C=" << C);
            return false;
        }

        if (m <= 0 || n <= 0 || k <= 0)
        {
            LOG_ERROR("[FP32StandaloneGemm] Invalid dimensions: m=" << m << " n=" << n << " k=" << k);
            return false;
        }

        LOG_DEBUG("[FP32StandaloneGemm] multiply_with_b: m=" << m << " n=" << n << " k=" << k
                                                             << " transpose_B=" << transpose_B << " alpha=" << alpha << " beta=" << beta);

        // OpenBLAS/MKL GEMM
        if (transpose_B)
        {
            // B is [n, k], compute A @ B^T
            // A: [m, k], B: [n, k], C: [m, n]
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                m, n, k,
                alpha,
                A, k, // A is [m, k], lda = k
                B, k, // B is [n, k], ldb = k
                beta,
                C, n // C is [m, n], ldc = n
            );
        }
        else
        {
            // B is [k, n], compute A @ B
            // A: [m, k], B: [k, n], C: [m, n]
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                A, k, // A is [m, k], lda = k
                B, n, // B is [k, n], ldb = n
                beta,
                C, n // C is [m, n], ldc = n
            );
        }

        // Validate output (check for NaN/Inf)
        for (int i = 0; i < m * n; ++i)
        {
            if (!std::isfinite(C[i]))
            {
                LOG_ERROR("[FP32StandaloneGemm] Non-finite value in output at index " << i << ": " << C[i]);
                return false;
            }
        }

        return true;
    }

} // namespace llaminar2
