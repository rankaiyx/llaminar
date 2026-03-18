/**
 * @file ROCmFusedOpsKernels.hip
 * @brief Fused HIP element-wise kernels for LLM inference optimization (ROCm port)
 *
 * ROCm/HIP port of CUDAFusedOpsKernels.cu. Contains fused operations that
 * eliminate intermediate memory round-trips:
 * 1. Fused SwiGLU + Blockwise INT8 Quantization
 * 2. Fused Residual Add + RMSNorm (FP32, BF16, FP16)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"

namespace {

// =========================================================================
// BF16/FP16 Conversion Helpers
// =========================================================================

__device__ __forceinline__ float bf16_to_float(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

__device__ __forceinline__ uint16_t float_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float fp16_to_float(uint16_t fp16)
{
    return __half2float(*reinterpret_cast<const __half*>(&fp16));
}

__device__ __forceinline__ uint16_t float_to_fp16(float f)
{
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

} // anonymous namespace

// =========================================================================
// 1. Fused SwiGLU + Blockwise INT8 Quantization
// =========================================================================

__global__ void rocm_fused_swiglu_quantize_blockwise_kernel(
    const float *__restrict__ gate,
    const float *__restrict__ up,
    int8_t *__restrict__ A_int8,
    float *__restrict__ scales_A_blockwise,
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M)
        return;

    constexpr int BLOCK_SIZE = 32;
    const int num_blocks = K / BLOCK_SIZE;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int NUM_WARPS = 8;

    const float *row_gate = gate + row * K;
    const float *row_up = up + row * K;
    int8_t *row_int8 = A_int8 + row * K;
    float *row_scales = scales_A_blockwise + row * num_blocks;

    for (int b = warp_id; b < num_blocks; b += NUM_WARPS)
    {
        const int k_start = b * BLOCK_SIZE;

        float g = row_gate[k_start + lane];
        float u = row_up[k_start + lane];

        // SwiGLU: silu(gate) * up
        float val = (g / (1.0f + expf(-g))) * u;

        // Warp-level max_abs reduction via shuffle
        float abs_val = fabsf(val);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            abs_val = fmaxf(abs_val, __shfl_xor(abs_val, mask));
        }

        float scale = (abs_val > 0.0f) ? (abs_val / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;

        if (lane == 0)
            row_scales[b] = scale;

        float qval = val * inv_scale;
        row_int8[k_start + lane] = static_cast<int8_t>(
            rintf(fminf(127.0f, fmaxf(-127.0f, qval))));
    }
}

// =========================================================================
// 2. Fused Residual Add + RMSNorm
// =========================================================================

__global__ void rocm_fused_residual_rmsnorm_fp32_kernel(
    const float *__restrict__ input,
    const float *__restrict__ residual,
    const float *__restrict__ gamma,
    float *__restrict__ residual_output,
    float *__restrict__ norm_output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float *row_input = input + row * cols;
    const float *row_residual = residual + row * cols;
    float *row_res_out = residual_output + row * cols;
    float *row_norm_out = norm_output + row * cols;

    // Pass 1: Add residual, compute sum of squares, write residual output
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = row_input[i] + row_residual[i];
        row_res_out[i] = sum;
        sum_sq += sum * sum;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    // Pass 2: Normalize and write output
    for (int i = tid; i < cols; i += stride)
    {
        row_norm_out[i] = row_res_out[i] * inv_rms * gamma[i];
    }
}

__global__ void rocm_fused_residual_rmsnorm_bf16_kernel(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ residual,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ residual_output,
    uint16_t *__restrict__ norm_output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    const uint16_t *row_residual = residual + row * cols;
    uint16_t *row_res_out = residual_output + row * cols;
    uint16_t *row_norm_out = norm_output + row * cols;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = bf16_to_float(row_input[i]) + bf16_to_float(row_residual[i]);
        row_res_out[i] = float_to_bf16(sum);
        sum_sq += sum * sum;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    for (int i = tid; i < cols; i += stride)
    {
        float sum = bf16_to_float(row_res_out[i]);
        row_norm_out[i] = float_to_bf16(sum * inv_rms * gamma[i]);
    }
}

__global__ void rocm_fused_residual_rmsnorm_fp16_kernel(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ residual,
    const float *__restrict__ gamma,
    uint16_t *__restrict__ residual_output,
    uint16_t *__restrict__ norm_output,
    int cols,
    float epsilon)
{
    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const uint16_t *row_input = input + row * cols;
    const uint16_t *row_residual = residual + row * cols;
    uint16_t *row_res_out = residual_output + row * cols;
    uint16_t *row_norm_out = norm_output + row * cols;

    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += stride)
    {
        float sum = fp16_to_float(row_input[i]) + fp16_to_float(row_residual[i]);
        row_res_out[i] = float_to_fp16(sum);
        sum_sq += sum * sum;
    }

    sdata[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / cols + epsilon);

    for (int i = tid; i < cols; i += stride)
    {
        float sum = fp16_to_float(row_res_out[i]);
        row_norm_out[i] = float_to_fp16(sum * inv_rms * gamma[i]);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    // =====================================================================
    // Fused SwiGLU + Blockwise Quantization
    // =====================================================================

    bool hipOps_fused_swiglu_quantize_blockwise(
        const float *gate,
        const float *up,
        int8_t *A_int8,
        float *scales_A_blockwise,
        int M, int K,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError(); // Clear stale errors

        dim3 grid(M);
        dim3 block(256);

        hipLaunchKernelGGL(rocm_fused_swiglu_quantize_blockwise_kernel,
                           grid, block, 0, static_cast<hipStream_t>(stream),
                           gate, up, A_int8, scales_A_blockwise, M, K);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_swiglu_quantize_blockwise launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =====================================================================
    // Fused Residual Add + RMSNorm
    // =====================================================================

    bool hipOps_fused_residual_rmsnorm_fp32(
        const float *input,
        const float *residual,
        const float *gamma,
        float *residual_output,
        float *norm_output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError();

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_fp32_kernel,
                           dim3(num_blocks), dim3(threads_per_block), shared_mem_size,
                           static_cast<hipStream_t>(stream),
                           input, residual, gamma, residual_output, norm_output, cols, epsilon);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_residual_rmsnorm_fp32 launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_fused_residual_rmsnorm_bf16(
        const uint16_t *input,
        const uint16_t *residual,
        const float *gamma,
        uint16_t *residual_output,
        uint16_t *norm_output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError();

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_bf16_kernel,
                           dim3(num_blocks), dim3(threads_per_block), shared_mem_size,
                           static_cast<hipStream_t>(stream),
                           input, residual, gamma, residual_output, norm_output, cols, epsilon);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_residual_rmsnorm_bf16 launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_fused_residual_rmsnorm_fp16(
        const uint16_t *input,
        const uint16_t *residual,
        const float *gamma,
        uint16_t *residual_output,
        uint16_t *norm_output,
        int rows, int cols,
        float epsilon,
        int device_idx,
        void *stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);
        (void)hipGetLastError();

        int threads_per_block = 256;
        int num_blocks = rows;
        size_t shared_mem_size = threads_per_block * sizeof(float);

        hipLaunchKernelGGL(rocm_fused_residual_rmsnorm_fp16_kernel,
                           dim3(num_blocks), dim3(threads_per_block), shared_mem_size,
                           static_cast<hipStream_t>(stream),
                           input, residual, gamma, residual_output, norm_output, cols, epsilon);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCmFusedOps] fused_residual_rmsnorm_fp16 launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
