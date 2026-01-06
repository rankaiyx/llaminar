/**
 * @file CUDAResidualAddKernels.cu
 * @brief CUDA Residual Add kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32, BF16, and FP16 element-wise addition kernels
 * with extern "C" wrapper functions.
 */

#include "CUDAHelpers.cuh"
#include <cstdio>

// =========================================================================
// Residual Add CUDA Kernels
// =========================================================================

/**
 * @brief FP32 Residual Add kernel - element-wise addition
 */
__global__ void residual_add_fp32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ residual,
    float *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] + residual[idx];
    }
}

/**
 * @brief BF16 Residual Add kernel - FP32 computation with BF16 I/O
 */
__global__ void residual_add_bf16_kernel(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ residual,
    uint16_t *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float in_f = bf16_to_float(input[idx]);
        float res_f = bf16_to_float(residual[idx]);
        output[idx] = float_to_bf16(in_f + res_f);
    }
}

/**
 * @brief FP16 Residual Add kernel - FP32 computation with FP16 I/O
 */
__global__ void residual_add_fp16_kernel(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ residual,
    uint16_t *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float in_f = fp16_to_float(input[idx]);
        float res_f = fp16_to_float(residual[idx]);
        output[idx] = float_to_fp16(in_f + res_f);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    bool cudaOps_residual_add_fp32(
        const float *input,
        const float *residual,
        float *output,
        int size,
        int device_idx)
    {
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        residual_add_fp32_kernel<<<num_blocks, threads_per_block>>>(input, residual, output, size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Residual Add FP32 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_residual_add_bf16(
        const uint16_t *input,
        const uint16_t *residual,
        uint16_t *output,
        int size,
        int device_idx)
    {
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        residual_add_bf16_kernel<<<num_blocks, threads_per_block>>>(input, residual, output, size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Residual Add BF16 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_residual_add_fp16(
        const uint16_t *input,
        const uint16_t *residual,
        uint16_t *output,
        int size,
        int device_idx)
    {
        cudaSetDevice(device_idx);

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        residual_add_fp16_kernel<<<num_blocks, threads_per_block>>>(input, residual, output, size);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA Residual Add FP16 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
