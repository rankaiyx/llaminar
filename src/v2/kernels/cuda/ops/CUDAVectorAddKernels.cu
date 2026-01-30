/**
 * @file CUDAVectorAddKernels.cu
 * @brief CUDA kernels for element-wise vector addition (used in collective reductions)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CUDAVectorAddKernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace llaminar2
{
    namespace cuda
    {

        // =============================================================================
        // Kernel Implementations
        // =============================================================================

        /**
         * @brief Element-wise vector addition: output = input1 + input2
         *
         * Optimized for large buffers with coalesced memory access.
         * Uses vectorized float4 loads/stores when count is divisible by 4.
         */
        __global__ void vectorAddKernel_f32(
            float *__restrict__ output,
            const float *__restrict__ input1,
            const float *__restrict__ input2,
            size_t count)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            // Main loop with grid-stride pattern
            for (size_t i = idx; i < count; i += stride)
            {
                output[i] = input1[i] + input2[i];
            }
        }

        /**
         * @brief Vectorized element-wise addition using float4
         *
         * Processes 4 floats per thread for better memory bandwidth utilization.
         * Caller must ensure count is divisible by 4.
         */
        __global__ void vectorAddKernel_f32_vec4(
            float4 *__restrict__ output,
            const float4 *__restrict__ input1,
            const float4 *__restrict__ input2,
            size_t count4) // count / 4
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count4; i += stride)
            {
                float4 a = input1[i];
                float4 b = input2[i];
                output[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
        }

        /**
         * @brief In-place addition: output += input (output = output + input)
         *
         * Common pattern for AllReduce where we accumulate into a single buffer.
         */
        __global__ void vectorAddInplaceKernel_f32(
            float *__restrict__ output,
            const float *__restrict__ input,
            size_t count)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count; i += stride)
            {
                output[i] += input[i];
            }
        }

        /**
         * @brief Vectorized in-place addition using float4
         */
        __global__ void vectorAddInplaceKernel_f32_vec4(
            float4 *__restrict__ output,
            const float4 *__restrict__ input,
            size_t count4)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count4; i += stride)
            {
                float4 a = output[i];
                float4 b = input[i];
                output[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
        }

        // =============================================================================
        // FP16 (half) Kernels
        // =============================================================================

        /**
         * @brief In-place FP16 addition: output += input
         */
        __global__ void vectorAddInplaceKernel_f16(
            __half *__restrict__ output,
            const __half *__restrict__ input,
            size_t count)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count; i += stride)
            {
                output[i] = __hadd(output[i], input[i]);
            }
        }

        /**
         * @brief Vectorized in-place FP16 addition using half2 (2 elements per op)
         */
        __global__ void vectorAddInplaceKernel_f16_vec2(
            __half2 *__restrict__ output,
            const __half2 *__restrict__ input,
            size_t count2)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count2; i += stride)
            {
                output[i] = __hadd2(output[i], input[i]);
            }
        }

        // =============================================================================
        // BF16 (bfloat16) Kernels
        // =============================================================================

        /**
         * @brief In-place BF16 addition: output += input
         */
        __global__ void vectorAddInplaceKernel_bf16(
            __nv_bfloat16 *__restrict__ output,
            const __nv_bfloat16 *__restrict__ input,
            size_t count)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count; i += stride)
            {
                output[i] = __hadd(output[i], input[i]);
            }
        }

        /**
         * @brief Vectorized in-place BF16 addition using bfloat162 (2 elements per op)
         */
        __global__ void vectorAddInplaceKernel_bf16_vec2(
            __nv_bfloat162 *__restrict__ output,
            const __nv_bfloat162 *__restrict__ input,
            size_t count2)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count2; i += stride)
            {
                output[i] = __hadd2(output[i], input[i]);
            }
        }

        // =============================================================================
        // INT8 Kernels
        // =============================================================================

        /**
         * @brief In-place INT8 addition: output += input
         *
         * Note: Uses saturating arithmetic to prevent overflow
         */
        __global__ void vectorAddInplaceKernel_i8(
            int8_t *__restrict__ output,
            const int8_t *__restrict__ input,
            size_t count)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count; i += stride)
            {
                // Use int32 for intermediate to avoid overflow, then clamp
                int32_t sum = static_cast<int32_t>(output[i]) + static_cast<int32_t>(input[i]);
                output[i] = static_cast<int8_t>(max(-128, min(127, sum)));
            }
        }

        /**
         * @brief Vectorized in-place INT8 addition using char4 (4 elements per op)
         */
        __global__ void vectorAddInplaceKernel_i8_vec4(
            char4 *__restrict__ output,
            const char4 *__restrict__ input,
            size_t count4)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count4; i += stride)
            {
                char4 a = output[i];
                char4 b = input[i];
                // Saturating addition for each component
                int32_t x = max(-128, min(127, static_cast<int32_t>(a.x) + static_cast<int32_t>(b.x)));
                int32_t y = max(-128, min(127, static_cast<int32_t>(a.y) + static_cast<int32_t>(b.y)));
                int32_t z = max(-128, min(127, static_cast<int32_t>(a.z) + static_cast<int32_t>(b.z)));
                int32_t w = max(-128, min(127, static_cast<int32_t>(a.w) + static_cast<int32_t>(b.w)));
                output[i] = make_char4(x, y, z, w);
            }
        }

        // =============================================================================
        // INT32 Kernels
        // =============================================================================

        /**
         * @brief In-place INT32 addition: output += input
         */
        __global__ void vectorAddInplaceKernel_i32(
            int32_t *__restrict__ output,
            const int32_t *__restrict__ input,
            size_t count)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count; i += stride)
            {
                output[i] += input[i];
            }
        }

        /**
         * @brief Vectorized in-place INT32 addition using int4 (4 elements per op)
         */
        __global__ void vectorAddInplaceKernel_i32_vec4(
            int4 *__restrict__ output,
            const int4 *__restrict__ input,
            size_t count4)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < count4; i += stride)
            {
                int4 a = output[i];
                int4 b = input[i];
                output[i] = make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
            }
        }

        // =============================================================================
        // Host Interface
        // =============================================================================

        bool launchVectorAdd_f32(
            float *output,
            const float *input1,
            const float *input2,
            size_t count,
            cudaStream_t stream)
        {
            if (count == 0)
                return true;

            // Configuration: 256 threads/block, enough blocks to saturate SMs
            const int threadsPerBlock = 256;
            const int maxBlocks = 65535;

            // Use vectorized kernel if count is divisible by 4 and large enough
            if (count >= 1024 && (count % 4) == 0)
            {
                size_t count4 = count / 4;
                int blocks = std::min(static_cast<int>((count4 + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);

                vectorAddKernel_f32_vec4<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<float4 *>(output),
                    reinterpret_cast<const float4 *>(input1),
                    reinterpret_cast<const float4 *>(input2),
                    count4);
            }
            else
            {
                int blocks = std::min(static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);
                vectorAddKernel_f32<<<blocks, threadsPerBlock, 0, stream>>>(output, input1, input2, count);
            }

            return cudaGetLastError() == cudaSuccess;
        }

        bool launchVectorAddInplace_f32(
            float *output,
            const float *input,
            size_t count,
            cudaStream_t stream)
        {
            if (count == 0)
                return true;

            const int threadsPerBlock = 256;
            const int maxBlocks = 65535;

            // Use vectorized kernel if count is divisible by 4 and large enough
            if (count >= 1024 && (count % 4) == 0)
            {
                size_t count4 = count / 4;
                int blocks = std::min(static_cast<int>((count4 + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);

                vectorAddInplaceKernel_f32_vec4<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<float4 *>(output),
                    reinterpret_cast<const float4 *>(input),
                    count4);
            }
            else
            {
                int blocks = std::min(static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);
                vectorAddInplaceKernel_f32<<<blocks, threadsPerBlock, 0, stream>>>(output, input, count);
            }

            (void)cudaGetLastError();  // Clear stale errors
        cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAVectorAdd] Kernel launch error: %s (code %d)\n",
                        cudaGetErrorString(err), static_cast<int>(err));
                return false;
            }
            return true;
        }

        bool launchVectorAddInplace_f16(
            void *output,
            const void *input,
            size_t count,
            cudaStream_t stream)
        {
            if (count == 0)
                return true;

            const int threadsPerBlock = 256;
            const int maxBlocks = 65535;

            // Use vectorized kernel if count is divisible by 2 and large enough
            if (count >= 512 && (count % 2) == 0)
            {
                size_t count2 = count / 2;
                int blocks = std::min(static_cast<int>((count2 + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);

                vectorAddInplaceKernel_f16_vec2<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<__half2 *>(output),
                    reinterpret_cast<const __half2 *>(input),
                    count2);
            }
            else
            {
                int blocks = std::min(static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);
                vectorAddInplaceKernel_f16<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<__half *>(output),
                    reinterpret_cast<const __half *>(input),
                    count);
            }

            return cudaGetLastError() == cudaSuccess;
        }

        bool launchVectorAddInplace_bf16(
            void *output,
            const void *input,
            size_t count,
            cudaStream_t stream)
        {
            if (count == 0)
                return true;

            const int threadsPerBlock = 256;
            const int maxBlocks = 65535;

            // Use vectorized kernel if count is divisible by 2 and large enough
            if (count >= 512 && (count % 2) == 0)
            {
                size_t count2 = count / 2;
                int blocks = std::min(static_cast<int>((count2 + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);

                vectorAddInplaceKernel_bf16_vec2<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<__nv_bfloat162 *>(output),
                    reinterpret_cast<const __nv_bfloat162 *>(input),
                    count2);
            }
            else
            {
                int blocks = std::min(static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);
                vectorAddInplaceKernel_bf16<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<__nv_bfloat16 *>(output),
                    reinterpret_cast<const __nv_bfloat16 *>(input),
                    count);
            }

            return cudaGetLastError() == cudaSuccess;
        }

        bool launchVectorAddInplace_i8(
            int8_t *output,
            const int8_t *input,
            size_t count,
            cudaStream_t stream)
        {
            if (count == 0)
                return true;

            const int threadsPerBlock = 256;
            const int maxBlocks = 65535;

            // Use vectorized kernel if count is divisible by 4 and large enough
            if (count >= 1024 && (count % 4) == 0)
            {
                size_t count4 = count / 4;
                int blocks = std::min(static_cast<int>((count4 + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);

                vectorAddInplaceKernel_i8_vec4<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<char4 *>(output),
                    reinterpret_cast<const char4 *>(input),
                    count4);
            }
            else
            {
                int blocks = std::min(static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);
                vectorAddInplaceKernel_i8<<<blocks, threadsPerBlock, 0, stream>>>(output, input, count);
            }

            return cudaGetLastError() == cudaSuccess;
        }

        bool launchVectorAddInplace_i32(
            int32_t *output,
            const int32_t *input,
            size_t count,
            cudaStream_t stream)
        {
            if (count == 0)
                return true;

            const int threadsPerBlock = 256;
            const int maxBlocks = 65535;

            // Use vectorized kernel if count is divisible by 4 and large enough
            if (count >= 1024 && (count % 4) == 0)
            {
                size_t count4 = count / 4;
                int blocks = std::min(static_cast<int>((count4 + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);

                vectorAddInplaceKernel_i32_vec4<<<blocks, threadsPerBlock, 0, stream>>>(
                    reinterpret_cast<int4 *>(output),
                    reinterpret_cast<const int4 *>(input),
                    count4);
            }
            else
            {
                int blocks = std::min(static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock), maxBlocks);
                vectorAddInplaceKernel_i32<<<blocks, threadsPerBlock, 0, stream>>>(output, input, count);
            }

            return cudaGetLastError() == cudaSuccess;
        }

    } // namespace cuda
} // namespace llaminar2
