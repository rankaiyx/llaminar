/**
 * @file CUDABiasAddKernels.cu
 * @brief CUDA Bias Add kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32 bias add kernel with broadcasting across rows.
 * Given output [M, N] and bias [N], computes output[i,j] += bias[j]
 */

#include "CUDAHelpers.cuh"
#include <cstdio>

// =========================================================================
// Bias Add CUDA Kernels
// =========================================================================

/**
 * @brief FP32 Bias Add kernel - broadcasts bias across rows
 *
 * @param output In/out tensor of shape [M, N]
 * @param bias Bias vector of length N
 * @param M Number of rows
 * @param N Row width (bias length)
 */
__global__ void bias_add_fp32_kernel(
    float *__restrict__ output,
    const float *__restrict__ bias,
    int M,
    int N)
{
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total)
    {
        int col = idx % N; // Which bias element to use
        output[idx] += bias[col];
    }
}

/**
 * @brief FP32 Bias Add kernel with fused scale - broadcasts bias across rows
 *
 * Useful for quantized GEMM where output needs scaling before bias add:
 * output[i,j] = output[i,j] * scale + bias[j]
 *
 * @param output In/out tensor of shape [M, N]
 * @param bias Bias vector of length N
 * @param scale Scale factor to apply before bias
 * @param M Number of rows
 * @param N Row width (bias length)
 */
__global__ void bias_add_fp32_scaled_kernel(
    float *__restrict__ output,
    const float *__restrict__ bias,
    float scale,
    int M,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total)
    {
        int col = idx % N;
        output[idx] = output[idx] * scale + bias[col];
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    bool cudaOps_bias_add_fp32(
        float *output,
        const float *bias,
        int M,
        int N,
        int device_idx)
    {
        cudaError_t err = cudaSetDevice(device_idx);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDABiasAdd] cudaSetDevice(%d) failed: %s\n",
                    device_idx, cudaGetErrorString(err));
            return false;
        }

        int total = M * N;
        if (total == 0)
            return true;

        const int block_size = 256;
        int num_blocks = (total + block_size - 1) / block_size;

        bias_add_fp32_kernel<<<num_blocks, block_size>>>(output, bias, M, N);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDABiasAdd] kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    bool cudaOps_bias_add_fp32_scaled(
        float *output,
        const float *bias,
        float scale,
        int M,
        int N,
        int device_idx)
    {
        cudaError_t err = cudaSetDevice(device_idx);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDABiasAdd] cudaSetDevice(%d) failed: %s\n",
                    device_idx, cudaGetErrorString(err));
            return false;
        }

        int total = M * N;
        if (total == 0)
            return true;

        const int block_size = 256;
        int num_blocks = (total + block_size - 1) / block_size;

        bias_add_fp32_scaled_kernel<<<num_blocks, block_size>>>(output, bias, scale, M, N);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDABiasAdd] scaled kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        return true;
    }

} // extern "C"
