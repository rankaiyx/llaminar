/**
 * @file FP32GemmKernel.cpp
 * @brief CPU FP32 GEMM kernel implementation
 *
 * Supports OpenBLAS and Intel MKL backends.
 * Both use the same cblas_sgemm API.
 *
 * @author David Sanftenberg
 */

#include "FP32GemmKernel.h"

#ifdef HAVE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace llaminar2
{

    bool FP32GemmKernel::multiply(
        const float *A, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        if (!weight_tensor_)
        {
            return false;
        }

        const float *B = weight_tensor_->data();
        auto shape = weight_tensor_->shape();

        // Validate dimensions
        if (shape.size() != 2)
        {
            return false;
        }

        int B_rows = shape[0];
        int B_cols = shape[1];

        // Check dimension compatibility
        if (transpose_B)
        {
            // B is [n, k], transposed to [k, n] for multiplication
            if (B_rows != n || B_cols != k)
            {
                return false;
            }
        }
        else
        {
            // B is [k, n], no transpose
            if (B_rows != k || B_cols != n)
            {
                return false;
            }
        }

        // OpenBLAS GEMM: C = alpha * A @ B^T + beta * C
        // A: [m, k]
        // B: [n, k] if transpose_B, else [k, n]
        // C: [m, n]

        if (transpose_B)
        {
            // B stored as [n, k], we want to compute A @ B^T
            // cblas_sgemm with CblasRowMajor, CblasNoTrans for A, CblasTrans for B
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
            // B stored as [k, n], we want to compute A @ B
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

        return true;
    }

    bool FP32GemmKernel::multiply_activations(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        // Activation-activation GEMM: C = alpha * A @ B^T + beta * C
        // A: [m, k]
        // B: [n, k] if transpose_B, else [k, n]
        // C: [m, n]

        if (transpose_B)
        {
            // B stored as [n, k], we want to compute A @ B^T
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
            // B stored as [k, n], we want to compute A @ B
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

        return true;
    }

    bool FP32GemmKernel::multiply_activations_strided(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        int lda, int ldb, int ldc,
        bool transpose_B,
        float alpha, float beta,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (device_idx != -1)
        {
            return false; // CPU only
        }

        // Strided activation-activation GEMM with custom leading dimensions
        // Enables zero-copy multi-head attention by using strides
        // A: [m, k] with stride lda
        // B: [n, k] (transpose_B=true) or [k, n] (transpose_B=false) with stride ldb
        // C: [m, n] with stride ldc

        if (transpose_B)
        {
            // B stored as [n, k], compute A @ B^T
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                m, n, k,
                alpha,
                A, lda, // Custom stride for A
                B, ldb, // Custom stride for B
                beta,
                C, ldc // Custom stride for C
            );
        }
        else
        {
            // B stored as [k, n], compute A @ B
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha,
                A, lda, // Custom stride for A
                B, ldb, // Custom stride for B
                beta,
                C, ldc // Custom stride for C
            );
        }

        return true;
    }

} // namespace llaminar2
