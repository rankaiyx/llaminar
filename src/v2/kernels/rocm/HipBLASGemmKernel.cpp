/**
 * @file HipBLASGemmKernel.cpp
 * @brief hipBLAS GEMM kernel implementation for AMD GPUs
 *
 * This file is compiled with hipcc to use HIP runtime APIs.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "HipBLASGemmKernel.h"
#include "utils/Logger.h"

#include <iostream>
#include <stdexcept>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // For __half and __float2half
#include <hipblas/hipblas.h>
#endif

namespace llaminar2
{
    namespace rocm
    {

#ifdef HAVE_ROCM

        // =====================================================================
        // Helper macros for error checking
        // =====================================================================

#define HIPBLAS_CHECK(call)                                                      \
    do                                                                           \
    {                                                                            \
        hipblasStatus_t status = call;                                           \
        if (status != HIPBLAS_STATUS_SUCCESS)                                    \
        {                                                                        \
            LOG_ERROR("[HipBLASGemmKernel] hipBLAS error: " << status            \
                                                           << " at " << __FILE__ \
                                                           << ":" << __LINE__);  \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define HIP_CHECK(call)                                                           \
    do                                                                            \
    {                                                                             \
        hipError_t err = call;                                                    \
        if (err != hipSuccess)                                                    \
        {                                                                         \
            LOG_ERROR("[HipBLASGemmKernel] HIP error: " << hipGetErrorString(err) \
                                                        << " at " << __FILE__     \
                                                        << ":" << __LINE__);      \
            return false;                                                         \
        }                                                                         \
    } while (0)

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        HipBLASGemmKernel::HipBLASGemmKernel(int device_id, Precision precision)
            : device_id_(device_id), precision_(precision)
        {
            // Set device
            hipError_t hip_err = hipSetDevice(device_id_);
            if (hip_err != hipSuccess)
            {
                throw std::runtime_error(
                    std::string("[HipBLASGemmKernel] Failed to set HIP device ") +
                    std::to_string(device_id_) + ": " + hipGetErrorString(hip_err));
            }

            // Create hipBLAS handle
            hipblasStatus_t hipblas_err = hipblasCreate(&handle_);
            if (hipblas_err != HIPBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    std::string("[HipBLASGemmKernel] Failed to create hipBLAS handle: ") +
                    std::to_string(static_cast<int>(hipblas_err)));
            }

            // Note: hipBLAS doesn't have explicit Tensor Core mode like cuBLAS
            // It will automatically use the best available math mode

            LOG_DEBUG("[HipBLASGemmKernel] Created on device " << device_id_
                                                               << " with precision "
                                                               << static_cast<int>(precision_));
        }

        HipBLASGemmKernel::~HipBLASGemmKernel()
        {
            if (handle_)
            {
                hipblasDestroy(handle_);
                handle_ = nullptr;
            }
        }

        // Move constructor
        HipBLASGemmKernel::HipBLASGemmKernel(HipBLASGemmKernel &&other) noexcept
            : handle_(other.handle_),
              device_id_(other.device_id_),
              precision_(other.precision_)
        {
            other.handle_ = nullptr;
        }

        // Move assignment
        HipBLASGemmKernel &HipBLASGemmKernel::operator=(HipBLASGemmKernel &&other) noexcept
        {
            if (this != &other)
            {
                if (handle_)
                {
                    hipblasDestroy(handle_);
                }
                handle_ = other.handle_;
                device_id_ = other.device_id_;
                precision_ = other.precision_;
                other.handle_ = nullptr;
            }
            return *this;
        }

        // =====================================================================
        // FP32 GEMM Implementation
        // =====================================================================

        bool HipBLASGemmKernel::execute(
            const float *d_A, const float *d_B, float *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            if (!handle_)
            {
                LOG_ERROR("[HipBLASGemmKernel::execute] hipBLAS handle is null");
                return false;
            }

            // Ensure we're on the correct device
            hipError_t hip_err = hipSetDevice(device_id_);
            if (hip_err != hipSuccess)
            {
                LOG_ERROR("[HipBLASGemmKernel::execute] Failed to set device: " << hipGetErrorString(hip_err));
                return false;
            }

            // =========================================================================
            // Row-major to column-major conversion for hipBLAS
            // =========================================================================
            //
            // Our data is stored in row-major format:
            //   A[M×K] row-major: element (i,j) at offset i*K + j, stride = K
            //   B[K×N] row-major: element (i,j) at offset i*N + j, stride = N
            //   (Or if transB=true, B is stored as N×K row-major with stride K)
            //
            // hipBLAS interprets memory as column-major:
            //   A memory with stride K seen as K×M column-major matrix = A^T
            //   B memory with stride N seen as N×K column-major matrix = B^T
            //
            // We want: C = A @ B (row-major)
            // Which is: C^T = (A @ B)^T = B^T @ A^T (column-major)
            //
            // hipBLAS already sees A^T and B^T in our memory, so:
            //   - For no user transpose (transA=false, transB=false):
            //     Call hipBLAS with HIPBLAS_OP_N to use A^T and B^T as-is
            //   - For user transpose (transA=true):
            //     We want A^T in the product, but hipBLAS sees A^T already,
            //     so we need HIPBLAS_OP_T to transpose it back to A
            //
            // Summary:
            //   transA=false → opA=HIPBLAS_OP_N (use hipBLAS's A^T view)
            //   transA=true  → opA=HIPBLAS_OP_T (transpose hipBLAS's A^T back to A)

            hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

            // Leading dimensions are the memory strides:
            //   lda = K (stride of A memory)
            //   ldb = N if transB=false (B is K×N row-major, stride N)
            //       = K if transB=true  (B is N×K row-major, stride K)
            //   ldc = N (stride of C memory, C is M×N row-major)
            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

            // hipBLAS call: gemm(opB, opA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
            // Computes: op(B_cm) @ op(A_cm) where B_cm, A_cm are hipBLAS's col-major views

            HIPBLAS_CHECK(hipblasSgemm(
                handle_,
                opB, opA,
                N, M, K,
                &alpha,
                d_B, ldb,
                d_A, lda,
                &beta,
                d_C, ldc));

            return true;
        }

        // =====================================================================
        // FP16 GEMM Implementation
        // =====================================================================

        bool HipBLASGemmKernel::execute_fp16(
            const void *d_A, const void *d_B, void *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            if (!handle_)
            {
                LOG_ERROR("[HipBLASGemmKernel::execute_fp16] hipBLAS handle is null");
                return false;
            }

            // Ensure we're on the correct device
            hipError_t hip_err = hipSetDevice(device_id_);
            if (hip_err != hipSuccess)
            {
                LOG_ERROR("[HipBLASGemmKernel::execute_fp16] Failed to set device: " << hipGetErrorString(hip_err));
                return false;
            }

            hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
            hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

            // Convert alpha/beta to half precision
            hipblasHalf alpha_h = __float2half(alpha);
            hipblasHalf beta_h = __float2half(beta);

            // hipblasHgemm for native FP16 computation
            HIPBLAS_CHECK(hipblasHgemm(
                handle_,
                opB, opA,
                N, M, K,
                &alpha_h,
                static_cast<const hipblasHalf *>(d_B), ldb,
                static_cast<const hipblasHalf *>(d_A), lda,
                &beta_h,
                static_cast<hipblasHalf *>(d_C), ldc));

            return true;
        }

        // =====================================================================
        // Factory function
        // =====================================================================

        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(
            int device_id,
            HipBLASGemmKernel::Precision precision)
        {
            return std::make_unique<HipBLASGemmKernel>(device_id, precision);
        }

#else // !HAVE_ROCM

        // Stub implementations when ROCm is not available

        HipBLASGemmKernel::HipBLASGemmKernel(int device_id, Precision precision)
            : device_id_(device_id), precision_(precision)
        {
            throw std::runtime_error("[HipBLASGemmKernel] ROCm support not compiled");
        }

        HipBLASGemmKernel::~HipBLASGemmKernel() {}

        HipBLASGemmKernel::HipBLASGemmKernel(HipBLASGemmKernel &&) noexcept = default;
        HipBLASGemmKernel &HipBLASGemmKernel::operator=(HipBLASGemmKernel &&) noexcept = default;

        bool HipBLASGemmKernel::execute(
            const float *, const float *, float *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool HipBLASGemmKernel::execute_fp16(
            const void *, const void *, void *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        std::unique_ptr<HipBLASGemmKernel> createHipBLASGemm(int, HipBLASGemmKernel::Precision)
        {
            return nullptr;
        }

#endif // HAVE_ROCM

    } // namespace rocm
} // namespace llaminar2
