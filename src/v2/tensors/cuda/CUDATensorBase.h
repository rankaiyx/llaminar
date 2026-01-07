#pragma once
/**
 * @file CUDATensorBase.h
 * @brief Base class for GPU-resident tensors
 *
 * CUDATensorBase provides the foundation for tensors whose PRIMARY data lives on
 * the GPU (CUDA device). Unlike TensorBase which is CPU-primary with optional GPU
 * shadow copies, CUDATensorBase tensors are GPU-native.
 *
 * Key differences from TensorBase:
 * - data_ptr() returns a DEVICE pointer (not host pointer)
 * - device_id() returns DeviceId::cuda(n)
 * - No automatic lazy transfer (explicit copy methods instead)
 * - to_fp32() performs D2H copy
 *
 * @author David Sanftenberg
 */

#include "../ITensor.h"
#include "../TensorType.h" // For TensorType enum (CUDA-safe, no SIMD)
#include "../../backends/DeviceId.h"
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

namespace llaminar2
{

    /**
     * @brief Base class for GPU-resident tensors
     *
     * Unlike TensorBase (CPUTensorBase), CUDATensorBase tensors have their
     * PRIMARY data on the GPU. The data_ptr() returns a device pointer.
     *
     * Memory ownership:
     * - owns_memory_ = true: tensor allocated the GPU memory and will free it
     * - owns_memory_ = false: tensor wraps external GPU memory (view/slice)
     */
    class CUDATensorBase : public ITensor
    {
    public:
        CUDATensorBase(std::vector<size_t> shape, TensorType dtype, int device_idx, cudaStream_t stream = nullptr);
        virtual ~CUDATensorBase();

        // Non-copyable (owns GPU memory)
        CUDATensorBase(const CUDATensorBase &) = delete;
        CUDATensorBase &operator=(const CUDATensorBase &) = delete;

        // Move semantics
        CUDATensorBase(CUDATensorBase &&other) noexcept;
        CUDATensorBase &operator=(CUDATensorBase &&other) noexcept;

        // =========================================================================
        // ITensor Implementation
        // =========================================================================

        const std::vector<size_t> &shape() const override { return shape_; }
        size_t numel() const override;
        size_t size_bytes() const override { return size_bytes_; }
        int native_type_id() const override { return static_cast<int>(dtype_); }

        // Device awareness - GPU tensors are GPU-primary
        DeviceId device_id() const override { return DeviceId::cuda(device_idx_); }
        bool is_on_cpu() const override { return false; }
        bool is_on_gpu() const override { return true; }

        // FP32 host data access - GPU tensors don't support direct host access
        // Use to_fp32() to copy data to a host buffer instead
        const float *data() const override { return nullptr; }
        float *mutable_data() override { return nullptr; }
        const float *fp32_data() const override { return nullptr; }

        // data_ptr() returns DEVICE pointer for GPU tensors
        const void *data_ptr() const override { return device_ptr_; }
        void *mutable_data_ptr() override { return device_ptr_; }

        // raw_data/raw_mutable_data also return device pointers for consistency
        const void *raw_data() const override { return device_ptr_; }
        void *raw_mutable_data() override { return device_ptr_; }

        // Type-safe device identification
        DeviceId home_device() const override { return DeviceId::cuda(device_idx_); }

        // Convert to FP32 (performs D2H copy + dequant if needed)
        void to_fp32(float *dst) const override;

        // =========================================================================
        // GPU-Specific API
        // =========================================================================

        /**
         * @brief Get the raw device pointer
         * @return CUDA device pointer to tensor data
         */
        void *device_ptr() { return device_ptr_; }
        const void *device_ptr() const { return device_ptr_; }

        /**
         * @brief Get the CUDA device index (0-based)
         * @return Device ordinal this tensor is allocated on
         */
        int device_index() const { return device_idx_; }

        /**
         * @brief Get the CUDA stream associated with this tensor
         * @return CUDA stream for async operations, or nullptr for default stream
         */
        cudaStream_t stream() const { return stream_; }

        /**
         * @brief Get the tensor's data type
         * @return TensorType enum value
         */
        TensorType dtype() const { return dtype_; }

        // =========================================================================
        // Host Copy Utilities
        // =========================================================================

        /**
         * @brief Copy tensor data from device to host
         * @param host_dst Host destination buffer (must be pre-allocated)
         * @param bytes Number of bytes to copy
         */
        void copyToHost(void *host_dst, size_t bytes) const;

        /**
         * @brief Copy data from host to device tensor
         * @param host_src Host source buffer
         * @param bytes Number of bytes to copy
         */
        void copyFromHost(const void *host_src, size_t bytes);

        // =========================================================================
        // Memory Management (for derived classes)
        // =========================================================================

        /**
         * @brief Allocate device memory
         * @param bytes Number of bytes to allocate
         * @return true on success, false on allocation failure
         */
        bool allocateDevice(size_t bytes);

        /**
         * @brief Free device memory if owned
         */
        void freeDevice();

    protected:
        void *device_ptr_ = nullptr;    ///< CUDA device pointer
        int device_idx_ = 0;            ///< CUDA device ordinal (0-based)
        cudaStream_t stream_ = nullptr; ///< CUDA stream for async ops
        std::vector<size_t> shape_;     ///< Tensor dimensions
        TensorType dtype_;              ///< Data type enum
        size_t size_bytes_ = 0;         ///< Total size in bytes
        bool owns_memory_ = true;       ///< False for view/slice tensors
    };

} // namespace llaminar2
