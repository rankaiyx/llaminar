/**
 * @file FP32GemmKernel.h
 * @brief CPU FP32 GEMM kernel using OpenBLAS
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../tensors/Tensors.h"

namespace llaminar2
{

    /**
     * @brief CPU FP32 GEMM kernel (OpenBLAS cblas_sgemm wrapper)
     */
    class FP32GemmKernel : public ITensorGemm
    {
    public:
        /**
         * @brief Construct GEMM kernel for FP32 weight tensor
         *
         * @param weight_tensor FP32 weight tensor (B matrix)
         */
        explicit FP32GemmKernel(const FP32Tensor *weight_tensor)
            : weight_tensor_(weight_tensor)
        {
        }

        ~FP32GemmKernel() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        bool multiply(
            const float *A, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool multiply_activations(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        bool multiply_activations_strided(
            const float *A, const float *B, float *C,
            int m, int n, int k,
            int lda, int ldb, int ldc,
            bool transpose_B = true,
            float alpha = 1.0f, float beta = 0.0f,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

    private:
        const FP32Tensor *weight_tensor_;
    };

} // namespace llaminar2
