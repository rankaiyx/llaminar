/**
 * @file ComputeStageUtils.h
 * @brief Shared utilities for compute stage implementations
 */
#pragma once

#include "../../tensors/ITensor.h"
#include "../../tensors/Tensors.h"
#include "../../utils/Logger.h"
#include "../../utils/Assertions.h"

namespace llaminar2
{

    // =========================================================================
    // ITensor to TensorBase casting utilities
    // =========================================================================
    // These helpers provide safe downcasting from device-agnostic ITensor*
    // to CPU-specific TensorBase*. Used during the transition period where
    // stage interfaces use ITensor* but implementations still need TensorBase*.

    /**
     * @brief Cast ITensor* to TensorBase* (mutable)
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages (optional)
     * @return TensorBase pointer, or nullptr if cast fails
     */
    inline TensorBase *asTensorBase(ITensor *tensor, const char *name = nullptr)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<TensorBase *>(tensor);
        if (!base && name)
        {
            LOG_WARN("asTensorBase: " << name << " is not a TensorBase (likely GPU tensor)");
        }
        return base;
    }

    /**
     * @brief Cast const ITensor* to const TensorBase*
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages (optional)
     * @return const TensorBase pointer, or nullptr if cast fails
     */
    inline const TensorBase *asTensorBase(const ITensor *tensor, const char *name = nullptr)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<const TensorBase *>(tensor);
        if (!base && name)
        {
            LOG_WARN("asTensorBase: " << name << " is not a TensorBase (likely GPU tensor)");
        }
        return base;
    }

    /**
     * @brief Cast ITensor* to TensorBase* with assertion
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages
     * @return TensorBase pointer (never nullptr if input is non-null)
     * @throws Assertion failure if cast fails
     */
    inline TensorBase *requireTensorBase(ITensor *tensor, const char *name)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<TensorBase *>(tensor);
        LLAMINAR_ASSERTF(base, name << " must be a CPU TensorBase (GPU not yet supported)");
        return base;
    }

    /**
     * @brief Cast const ITensor* to const TensorBase* with assertion
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages
     * @return const TensorBase pointer (never nullptr if input is non-null)
     * @throws Assertion failure if cast fails
     */
    inline const TensorBase *requireTensorBase(const ITensor *tensor, const char *name)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<const TensorBase *>(tensor);
        LLAMINAR_ASSERTF(base, name << " must be a CPU TensorBase (GPU not yet supported)");
        return base;
    }

    // =========================================================================
    // FP32 Data Access
    // =========================================================================

    /**
     * @brief Safe FP32 data access for getDumpInfo()
     *
     * For Q8_1 tensors, uses fp32_data() (explicit dequant for debugging only)
     * For other tensors, uses data()
     *
     * @note This function only works with CPU tensors (TensorBase).
     *       For GPU tensors, returns nullptr.
     */
    inline const float *getSafeFp32Data(const ITensor *tensor)
    {
        if (!tensor)
            return nullptr;

        // For GPU tensors, we can't access data directly - return nullptr
        if (!tensor->is_on_cpu())
            return nullptr;

        // Try to get as TensorBase for CPU tensor operations
        auto *cpu_tensor = dynamic_cast<const TensorBase *>(tensor);
        if (!cpu_tensor)
            return nullptr;

        if (cpu_tensor->native_type() == TensorType::Q8_1)
        {
            // Q8_1 tensors throw on data() - use explicit fp32_data() for debugging
            auto *q8_1 = dynamic_cast<const Q8_1Tensor *>(cpu_tensor);
            return q8_1 ? q8_1->fp32_data() : nullptr;
        }

        if (cpu_tensor->native_type() == TensorType::Q16_1)
        {
            // Q16_1 tensors also need special handling
            auto *q16_1 = dynamic_cast<const Q16_1Tensor *>(cpu_tensor);
            return q16_1 ? q16_1->fp32_data() : nullptr;
        }

        // Standard FP32/BF16/FP16 tensors
        return cpu_tensor->data();
    }

} // namespace llaminar2
