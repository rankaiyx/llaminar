#pragma once
/**
 * @file TensorBase.h
 * @brief Minimal shared tensor base class - CUDA-safe
 *
 * This provides the common foundation for both CPU (CPUTensorBase) and
 * GPU (CUDATensorBase) tensor implementations. Contains no platform-specific
 * code (no SIMD, no CUDA kernels).
 *
 * Design note: The existing TensorBase in Tensors.h is the CPU tensor base class.
 * In a future refactoring step, it will be renamed to CPUTensorBase and will
 * inherit from this SharedTensorBase class (or this class will be renamed).
 * For now, this serves as the base for CUDATensorBase.
 */

#include "ITensor.h"
#include "TensorType.h"
#include "../backends/DeviceId.h"
#include <vector>
#include <cstddef>
#include <numeric>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Minimal shared base for CPU and CUDA tensors
     *
     * Provides common implementations of ITensor methods that are
     * identical across CPU and GPU tensors.
     *
     * This class is intentionally minimal - it only provides:
     * - Shape storage and accessors
     * - Element count computation (numel)
     * - Type information storage
     * - Common helper methods (rows, cols, dtype_name)
     *
     * Device-specific behavior (memory allocation, data pointers, device ID)
     * must be implemented by derived classes.
     */
    class SharedTensorBase : public ITensor
    {
    public:
        virtual ~SharedTensorBase() = default;

        // Non-copyable by default (tensors own memory)
        SharedTensorBase(const SharedTensorBase &) = delete;
        SharedTensorBase &operator=(const SharedTensorBase &) = delete;

        // =========================================================================
        // ITensor Interface - Common Implementations
        // =========================================================================

        const std::vector<size_t> &shape() const override { return shape_; }

        size_t numel() const override
        {
            if (shape_.empty())
                return 0;
            return std::accumulate(shape_.begin(), shape_.end(),
                                   size_t(1), std::multiplies<size_t>());
        }

        int native_type_id() const override { return static_cast<int>(dtype_); }

        // =========================================================================
        // Common Helper Methods
        // =========================================================================

        /**
         * @brief Get human-readable dtype name
         */
        const char *dtype_name() const { return tensorTypeName(dtype_); }

        /**
         * @brief Get tensor data type
         */
        TensorType dtype() const { return dtype_; }

        /**
         * @brief Get number of rows (first dimension)
         */
        size_t rows() const
        {
            return shape_.empty() ? 1 : shape_[0];
        }

        /**
         * @brief Get number of columns (second dimension)
         */
        size_t cols() const
        {
            return shape_.size() < 2 ? 1 : shape_[1];
        }

    protected:
        // Protected constructor - only derived classes can instantiate
        SharedTensorBase() = default;
        SharedTensorBase(std::vector<size_t> shape, TensorType dtype)
            : shape_(std::move(shape)), dtype_(dtype) {}

        // Move is allowed for derived classes
        SharedTensorBase(SharedTensorBase &&) = default;
        SharedTensorBase &operator=(SharedTensorBase &&) = default;

        std::vector<size_t> shape_;
        TensorType dtype_ = TensorType::FP32;
    };

} // namespace llaminar2
