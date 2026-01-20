/**
 * @file CuBLASGemmKernel.cu
 * @brief cuBLAS GEMM kernel implementation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CuBLASGemmKernel.h"
#include "utils/Logger.h"

#include <iostream>
#include <stdexcept>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#endif

namespace llaminar2
{
    namespace cuda
    {

#ifdef HAVE_CUDA

        // =====================================================================
        // Helper macros for error checking
        // =====================================================================

#define CUBLAS_CHECK(call)                                                      \
    do                                                                          \
    {                                                                           \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                       \
            LOG_ERROR("[CuBLASGemmKernel] cuBLAS error: " << status             \
                                                          << " at " << __FILE__ \
                                                          << ":" << __LINE__);  \
            return false;                                                       \
        }                                                                       \
    } while (0)

#define CUBLASLT_CHECK(call)                                                      \
    do                                                                            \
    {                                                                             \
        cublasStatus_t status = call;                                             \
        if (status != CUBLAS_STATUS_SUCCESS)                                      \
        {                                                                         \
            LOG_ERROR("[CuBLASGemmKernel] cuBLASLt error: " << status             \
                                                            << " at " << __FILE__ \
                                                            << ":" << __LINE__);  \
            return false;                                                         \
        }                                                                         \
    } while (0)

#define CUDA_CHECK(call)                                                           \
    do                                                                             \
    {                                                                              \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess)                                                    \
        {                                                                          \
            LOG_ERROR("[CuBLASGemmKernel] CUDA error: " << cudaGetErrorString(err) \
                                                        << " at " << __FILE__      \
                                                        << ":" << __LINE__);       \
            return false;                                                          \
        }                                                                          \
    } while (0)

        // =====================================================================
        // Constructor / Destructor
        // =====================================================================

        CuBLASGemmKernel::CuBLASGemmKernel(int device_id, Precision precision)
            : device_id_(device_id), precision_(precision)
        {
            // Set device
            cudaError_t cuda_err = cudaSetDevice(device_id_);
            if (cuda_err != cudaSuccess)
            {
                throw std::runtime_error(
                    std::string("[CuBLASGemmKernel] Failed to set CUDA device ") +
                    std::to_string(device_id_) + ": " + cudaGetErrorString(cuda_err));
            }

            // Create cuBLAS handle
            cublasStatus_t cublas_err = cublasCreate(&handle_);
            if (cublas_err != CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(
                    std::string("[CuBLASGemmKernel] Failed to create cuBLAS handle: ") +
                    std::to_string(static_cast<int>(cublas_err)));
            }

            // Enable Tensor Core math for better performance on compatible GPUs
            // CUBLAS_TENSOR_OP_MATH enables use of Tensor Cores when available
            cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);

            // Create cuBLASLt handle for fused operations (e.g., GEMM + bias)
            cublasStatus_t lt_err = cublasLtCreate(&lt_handle_);
            if (lt_err != CUBLAS_STATUS_SUCCESS)
            {
                cublasDestroy(handle_);
                throw std::runtime_error(
                    std::string("[CuBLASGemmKernel] Failed to create cuBLASLt handle: ") +
                    std::to_string(static_cast<int>(lt_err)));
            }

            LOG_DEBUG("[CuBLASGemmKernel] Created on device " << device_id_
                                                              << " with precision "
                                                              << static_cast<int>(precision_));
        }

        CuBLASGemmKernel::~CuBLASGemmKernel()
        {
            if (lt_handle_)
            {
                cublasLtDestroy(lt_handle_);
                lt_handle_ = nullptr;
            }
            if (handle_)
            {
                cublasDestroy(handle_);
                handle_ = nullptr;
            }
        }

        // Move constructor
        CuBLASGemmKernel::CuBLASGemmKernel(CuBLASGemmKernel &&other) noexcept
            : handle_(other.handle_),
              lt_handle_(other.lt_handle_),
              device_id_(other.device_id_),
              precision_(other.precision_)
        {
            other.handle_ = nullptr;
            other.lt_handle_ = nullptr;
        }

        // Move assignment
        CuBLASGemmKernel &CuBLASGemmKernel::operator=(CuBLASGemmKernel &&other) noexcept
        {
            if (this != &other)
            {
                if (lt_handle_)
                {
                    cublasLtDestroy(lt_handle_);
                }
                if (handle_)
                {
                    cublasDestroy(handle_);
                }
                handle_ = other.handle_;
                lt_handle_ = other.lt_handle_;
                device_id_ = other.device_id_;
                precision_ = other.precision_;
                other.handle_ = nullptr;
                other.lt_handle_ = nullptr;
            }
            return *this;
        }

        // =====================================================================
        // GEMM Implementation
        // =====================================================================

        bool CuBLASGemmKernel::execute(
            const float *d_A, const float *d_B, float *d_C,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            if (!handle_)
            {
                LOG_ERROR("[CuBLASGemmKernel::execute] cuBLAS handle is null");
                return false;
            }

            // Ensure we're on the correct device
            CUDA_CHECK(cudaSetDevice(device_id_));

            // cuBLAS expects column-major. We have row-major data.
            // To compute C = A @ B in row-major:
            //   - Treat row-major A[M×K] as column-major A^T[K×M]
            //   - Treat row-major B[K×N] as column-major B^T[N×K]
            //   - Compute B^T @ A^T = (A @ B)^T in column-major
            //   - Which is A @ B in row-major when stored in C[M×N]
            //
            // So we swap A and B, and adjust transposes:
            //   C_cm = B_cm @ A_cm with appropriate transposes
            //
            // For row-major C[M×N] = A[M×K] @ B[K×N]:
            //   cublasSgemm(handle,
            //               transB_cublas, transA_cublas,
            //               N, M, K,  // Swapped M/N
            //               &alpha,
            //               B, ldb,   // Swapped A/B
            //               A, lda,
            //               &beta,
            //               C, N);    // ldc = N for row-major

            // =========================================================================
            // Row-major to column-major conversion for cuBLAS
            // =========================================================================
            //
            // Our data is stored in row-major format:
            //   A[M×K] row-major: element (i,j) at offset i*K + j, stride = K
            //   B[K×N] row-major: element (i,j) at offset i*N + j, stride = N
            //   (Or if transB=true, B is stored as N×K row-major with stride K)
            //
            // cuBLAS interprets memory as column-major:
            //   A memory with stride K seen as K×M column-major matrix = A^T
            //   B memory with stride N seen as N×K column-major matrix = B^T
            //
            // We want: C = A @ B (row-major)
            // Which is: C^T = (A @ B)^T = B^T @ A^T (column-major)
            //
            // cuBLAS already sees A^T and B^T in our memory, so:
            //   - For no user transpose (transA=false, transB=false):
            //     Call cuBLAS with CUBLAS_OP_N to use A^T and B^T as-is
            //   - For user transpose (transA=true):
            //     We want A^T in the product, but cuBLAS sees A^T already,
            //     so we need CUBLAS_OP_T to transpose it back to A
            //
            // Summary:
            //   transA=false → opA=CUBLAS_OP_N (use cuBLAS's A^T view)
            //   transA=true  → opA=CUBLAS_OP_T (transpose cuBLAS's A^T back to A)

            cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

            // Leading dimensions are the memory strides:
            //   lda = K (stride of A memory, which is K×M in cuBLAS's view)
            //   ldb = N if transB=false (B is K×N row-major, stride N)
            //       = K if transB=true  (B is N×K row-major, stride K)
            //   ldc = N (stride of C memory, C is M×N row-major)
            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

            // cuBLAS call: gemm(opB, opA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
            // Computes: op(B_cm) @ op(A_cm) where B_cm, A_cm are cuBLAS's col-major views

            CUBLAS_CHECK(cublasSgemm(
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
        // GEMM with Fused Bias (using cuBLASLt)
        // =====================================================================

        bool CuBLASGemmKernel::execute_with_bias(
            const float *d_A, const float *d_B, float *d_C,
            const float *d_bias,
            int M, int N, int K,
            bool transA, bool transB,
            float alpha, float beta)
        {
            if (!lt_handle_)
            {
                LOG_ERROR("[CuBLASGemmKernel::execute_with_bias] cuBLASLt handle is null");
                return false;
            }

            // Ensure we're on the correct device
            CUDA_CHECK(cudaSetDevice(device_id_));

            // Create operation descriptors
            cublasLtMatmulDesc_t operationDesc = nullptr;
            cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
            cublasLtMatmulPreference_t preference = nullptr;

            // =================================================================
            // Create operation descriptor with epilogue
            // =================================================================

            CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUDA_R_32F));

            // Set transpose operations
            // For row-major data with cuBLASLt, we swap A and B:
            //   C_rm = A_rm @ B_rm  becomes  C_cm^T = B_cm^T @ A_cm^T
            // cuBLASLt sees our row-major A[M×K] as column-major A^T[K×M]
            cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

            CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_TRANSA,
                                                          &opB, sizeof(opB)));
            CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_TRANSB,
                                                          &opA, sizeof(opA)));

            // Set epilogue with bias
            cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                          &epilogue, sizeof(epilogue)));

            // Set bias pointer
            CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc,
                                                          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                          &d_bias, sizeof(d_bias)));

            // =================================================================
            // Create matrix layouts
            // =================================================================
            // For row-major data interpreted as column-major:
            // Our A[M×K] row-major is seen as A^T[K×M] column-major (ld=K)
            // Our B[K×N] or B[N×K] needs appropriate layout
            // Our C[M×N] row-major is seen as C^T[N×M] column-major (ld=N)

            int lda = K;
            int ldb = transB ? K : N;
            int ldc = N;

            // We're computing in column-major, so we swap A and B in the call:
            // cuBLASLt gemm: D = alpha * op(A_lt) @ op(B_lt) + beta * C
            // where A_lt (in cuBLASLt call) = B (our matrix), B_lt = A (our matrix)

            // Create layout for "A" in cuBLASLt call (which is our B)
            // If transB=true: our B is [N×K] row-major → [K×N] col-major
            // If transB=false: our B is [K×N] row-major → [N×K] col-major
            if (transB)
            {
                // B[N×K] row-major → cuBLASLt sees [K×N] col-major
                CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, N, ldb));
            }
            else
            {
                // B[K×N] row-major → cuBLASLt sees [N×K] col-major
                CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, N, K, ldb));
            }

            // Create layout for "B" in cuBLASLt call (which is our A[M×K])
            // A[M×K] row-major → cuBLASLt sees [K×M] col-major
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, M, lda));

            // Create layout for C (and D): C[M×N] row-major → [N×M] col-major
            CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, ldc));

            // =================================================================
            // Set up algorithm heuristics
            // =================================================================

            CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));

            // Request workspace (optional, can improve performance)
            size_t workspaceSize = 4 * 1024 * 1024; // 4MB workspace
            void *workspace = nullptr;
            CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

            CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference,
                                                                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                                &workspaceSize, sizeof(workspaceSize)));

            // Get best algorithm
            int returnedResults = 0;
            cublasLtMatmulHeuristicResult_t heuristicResult = {};
            CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_handle_,
                                                          operationDesc,
                                                          Adesc, Bdesc, Cdesc, Cdesc,
                                                          preference,
                                                          1, &heuristicResult, &returnedResults));

            if (returnedResults == 0)
            {
                LOG_ERROR("[CuBLASGemmKernel::execute_with_bias] No suitable algorithm found");
                cudaFree(workspace);
                cublasLtMatmulPreferenceDestroy(preference);
                cublasLtMatrixLayoutDestroy(Cdesc);
                cublasLtMatrixLayoutDestroy(Bdesc);
                cublasLtMatrixLayoutDestroy(Adesc);
                cublasLtMatmulDescDestroy(operationDesc);
                return false;
            }

            // =================================================================
            // Execute the matmul with fused bias
            // =================================================================

            cublasStatus_t status = cublasLtMatmul(lt_handle_,
                                                   operationDesc,
                                                   &alpha,
                                                   d_B, Adesc, // A in cuBLASLt = our B
                                                   d_A, Bdesc, // B in cuBLASLt = our A
                                                   &beta,
                                                   d_C, Cdesc, // C
                                                   d_C, Cdesc, // D (output, same as C)
                                                   &heuristicResult.algo,
                                                   workspace, workspaceSize,
                                                   0); // stream = 0 (default)

            // Cleanup
            cudaFree(workspace);
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(Cdesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatmulDescDestroy(operationDesc);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[CuBLASGemmKernel::execute_with_bias] cublasLtMatmul failed: " << status);
                return false;
            }

            return true;
        }

        // =====================================================================
        // Factory function
        // =====================================================================

        std::unique_ptr<CuBLASGemmKernel> createCuBLASGemm(
            int device_id,
            CuBLASGemmKernel::Precision precision)
        {
            return std::make_unique<CuBLASGemmKernel>(device_id, precision);
        }

#else // !HAVE_CUDA

        // Stub implementations when CUDA is not available

        CuBLASGemmKernel::CuBLASGemmKernel(int device_id, Precision precision)
            : device_id_(device_id), precision_(precision)
        {
            throw std::runtime_error("[CuBLASGemmKernel] CUDA support not compiled");
        }

        CuBLASGemmKernel::~CuBLASGemmKernel() {}

        CuBLASGemmKernel::CuBLASGemmKernel(CuBLASGemmKernel &&) noexcept = default;
        CuBLASGemmKernel &CuBLASGemmKernel::operator=(CuBLASGemmKernel &&) noexcept = default;

        bool CuBLASGemmKernel::execute(
            const float *, const float *, float *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        bool CuBLASGemmKernel::execute_with_bias(
            const float *, const float *, float *,
            const float *,
            int, int, int,
            bool, bool, float, float)
        {
            return false;
        }

        std::unique_ptr<CuBLASGemmKernel> createCuBLASGemm(int, CuBLASGemmKernel::Precision)
        {
            return nullptr;
        }

#endif // HAVE_CUDA

    } // namespace cuda
} // namespace llaminar2
