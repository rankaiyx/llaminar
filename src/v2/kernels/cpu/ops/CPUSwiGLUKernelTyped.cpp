/**
 * @file CPUSwiGLUKernelTyped.cpp
 * @brief Implementation of typed SwiGLU kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This file implements the CPUSwiGLUKernelTyped template specializations for
 * FP32, BF16, FP16, and Q8_1 precision types.
 *
 * Each precision uses native primitives without unnecessary type conversions:
 * - FP32: Direct FP32 SwiGLU with AVX512/AVX2 vectorization
 * - BF16: Native BF16 SwiGLU (converts to FP32 for silu, back to BF16)
 * - FP16: Native FP16 SwiGLU (converts to FP32 for silu, back to FP16)
 * - Q8_1: Integer-aware SwiGLU (dequant, compute, requant per block)
 */

#include "CPUSwiGLUKernelTyped.h"
#include "../primitives/SwiGLUPrimitives.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    // ============================================================================
    // FP32 Specialization Implementation
    // ============================================================================

    bool CPUSwiGLUKernelTyped<ActivationPrecision::FP32>::apply_typed(
        const float *gate,
        const float *up,
        float *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelTyped<FP32>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelTyped<FP32>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use vectorized SwiGLU primitive (AVX512/AVX2 with OpenMP)
        primitives::compute_swiglu(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // BF16 Specialization Implementation
    // ============================================================================

    bool CPUSwiGLUKernelTyped<ActivationPrecision::BF16>::apply_typed(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelTyped<BF16>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelTyped<BF16>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use native BF16 SwiGLU primitive
        // Internally converts to FP32 for silu computation, converts back to BF16
        primitives::compute_swiglu_bf16(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // FP16 Specialization Implementation
    // ============================================================================

    bool CPUSwiGLUKernelTyped<ActivationPrecision::FP16>::apply_typed(
        const uint16_t *gate,
        const uint16_t *up,
        uint16_t *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelTyped<FP16>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelTyped<FP16>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use native FP16 SwiGLU primitive
        // Internally converts to FP32 for silu computation, converts back to FP16
        primitives::compute_swiglu_fp16(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // Q8_1 Specialization Implementation (Integer-Aware)
    // ============================================================================

    bool CPUSwiGLUKernelTyped<ActivationPrecision::Q8_1>::apply_typed(
        const Q8_1Block *gate,
        const Q8_1Block *up,
        Q8_1Block *output,
        int size,
        int device_idx)
    {
        (void)device_idx; // Unused for CPU kernel

        if (!gate || !up || !output)
        {
            LOG_ERROR("CPUSwiGLUKernelTyped<Q8_1>: Null data pointer");
            return false;
        }

        if (size <= 0)
        {
            LOG_DEBUG("CPUSwiGLUKernelTyped<Q8_1>: Empty input (size=" << size << ")");
            return true; // Nothing to do
        }

        // Use Q8_1 SwiGLU primitive
        // This handles:
        //   - Block-wise dequantization
        //   - SwiGLU computation: silu(gate) * up
        //   - Requantization with new scale factor
        primitives::compute_swiglu_q8_1(gate, up, output, size);

        return true;
    }

    // ============================================================================
    // Destructor Definitions (required for vtable emission)
    // ============================================================================

    CPUSwiGLUKernelTyped<ActivationPrecision::FP32>::~CPUSwiGLUKernelTyped() = default;
    CPUSwiGLUKernelTyped<ActivationPrecision::BF16>::~CPUSwiGLUKernelTyped() = default;
    CPUSwiGLUKernelTyped<ActivationPrecision::FP16>::~CPUSwiGLUKernelTyped() = default;
    CPUSwiGLUKernelTyped<ActivationPrecision::Q8_1>::~CPUSwiGLUKernelTyped() = default;

} // namespace llaminar2
