/**
 * @file CUDAHelpers.cuh
 * @brief Shared CUDA helper functions for ops kernels
 * @author David Sanftenberg
 *
 * Contains BF16/FP16 conversion helpers and activation functions
 * used across multiple kernel files.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// =========================================================================
// BF16 Conversion Helpers
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
    // Round to nearest even
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

// =========================================================================
// FP16 Conversion Helpers (using CUDA intrinsics)
// =========================================================================

__device__ __forceinline__ float fp16_to_float(uint16_t fp16)
{
    __half h;
    memcpy(&h, &fp16, sizeof(__half));
    return __half2float(h);
}

__device__ __forceinline__ uint16_t float_to_fp16(float f)
{
    __half h = __float2half(f);
    uint16_t result;
    memcpy(&result, &h, sizeof(uint16_t));
    return result;
}

// =========================================================================
// Activation Functions
// =========================================================================

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}
