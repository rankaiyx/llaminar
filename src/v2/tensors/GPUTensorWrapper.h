/**
 * @file GPUTensorWrapper.h
 * @brief Lightweight GPU tensor wrapper for KV cache access
 * @author David Sanftenberg
 * @date January 2026
 *
 * This header provides a minimal tensor wrapper that wraps existing GPU memory
 * without owning it. It's designed to be safely included in CUDA files (no x86
 * SIMD intrinsics) and provides the TensorBase interface that stages expect.
 *
 * Primary use case: CUDARingKVCache::get_k_base() and get_v_base() methods need
 * to return TensorBase* pointers that wrap the ring buffer's GPU memory.
 */

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    // Forward declaration - full TensorBase from Tensors.h
    class TensorBase;

    /**
     * @brief Tensor type enum (copied here for CUDA compatibility)
     *
     * This mirrors TensorType from Tensors.h but is safe for CUDA compilation.
     */
    enum class GPUTensorType
    {
        FP32,
        FP16,
        BF16
    };

    /**
     * @brief Lightweight wrapper around existing GPU memory
     *
     * This class wraps a raw GPU pointer without owning the memory.
     * It provides the minimum interface needed for graph stages to
     * access KV cache data through TensorBase*.
     *
     * Key features:
     * - Does NOT own GPU memory (no allocation/deallocation)
     * - Shape can be updated as ring buffer grows
     * - GPU pointer can be updated when buffer is linearized
     * - Safe to compile with nvcc (no x86 intrinsics)
     *
     * Limitations:
     * - No host data (host_invalid_ = true)
     * - Read-only conceptually (stages should not modify KV cache directly)
     * - No format conversion methods
     */
    class GPUTensorWrapper
    {
    public:
        /**
         * @brief Construct a GPU tensor wrapper
         *
         * @param dtype The tensor data type (FP32, FP16, BF16)
         * @param device_id CUDA device ID where memory resides
         */
        GPUTensorWrapper(GPUTensorType dtype, int device_id)
            : dtype_(dtype), device_id_(device_id), gpu_ptr_(nullptr), shape_({0, 0}), numel_(0), byte_size_(0)
        {
        }

        ~GPUTensorWrapper() = default;

        // Non-copyable, non-movable (wrapper state should not be shared)
        GPUTensorWrapper(const GPUTensorWrapper &) = delete;
        GPUTensorWrapper &operator=(const GPUTensorWrapper &) = delete;
        GPUTensorWrapper(GPUTensorWrapper &&) = delete;
        GPUTensorWrapper &operator=(GPUTensorWrapper &&) = delete;

        // =====================================================================
        // Mutable State Updates
        // =====================================================================

        /**
         * @brief Update the GPU pointer and shape
         *
         * Call this when the ring buffer is linearized or when the cached
         * token count changes.
         *
         * @param gpu_ptr Raw GPU memory pointer
         * @param rows Number of tokens (rows)
         * @param cols KV dimension (cols)
         */
        void update(void *gpu_ptr, size_t rows, size_t cols)
        {
            gpu_ptr_ = gpu_ptr;
            shape_[0] = rows;
            shape_[1] = cols;
            numel_ = rows * cols;
            byte_size_ = numel_ * element_size();
        }

        /**
         * @brief Update only the GPU pointer (shape unchanged)
         *
         * Use when the ring buffer switches between direct and scratch pointers.
         */
        void update_ptr(void *gpu_ptr)
        {
            gpu_ptr_ = gpu_ptr;
        }

        /**
         * @brief Update only the row count (cols unchanged)
         *
         * Use when tokens are appended and only the first dimension changes.
         */
        void update_rows(size_t rows)
        {
            shape_[0] = rows;
            numel_ = rows * shape_[1];
            byte_size_ = numel_ * element_size();
        }

        // =====================================================================
        // TensorBase-like Interface
        // =====================================================================

        const std::vector<size_t> &shape() const { return shape_; }
        size_t rows() const { return shape_[0]; }
        size_t cols() const { return shape_[1]; }
        size_t numel() const { return numel_; }
        size_t size_bytes() const { return byte_size_; }

        GPUTensorType dtype() const { return dtype_; }
        int device_id() const { return device_id_; }

        // GPU data access
        void *gpu_data_ptr() { return gpu_ptr_; }
        const void *gpu_data_ptr() const { return gpu_ptr_; }
        bool isOnGPU() const { return gpu_ptr_ != nullptr; }

        // Host data is not available
        bool isOnCPU() const { return false; }

        /**
         * @brief Get element size in bytes based on dtype
         */
        size_t element_size() const
        {
            switch (dtype_)
            {
            case GPUTensorType::FP32:
                return sizeof(float);
            case GPUTensorType::FP16:
            case GPUTensorType::BF16:
                return sizeof(uint16_t);
            default:
                return sizeof(float);
            }
        }

    private:
        GPUTensorType dtype_;
        int device_id_;
        void *gpu_ptr_;
        std::vector<size_t> shape_;
        size_t numel_;
        size_t byte_size_;
    };

    /**
     * @brief Adapter to expose GPUTensorWrapper as TensorBase*
     *
     * Since GPUTensorWrapper cannot directly inherit from TensorBase (which has
     * x86 SIMD dependencies), we use a separate adapter class that inherits from
     * TensorBase and holds a GPUTensorWrapper internally.
     *
     * This class is defined in GPUTensorWrapperAdapter.cpp (compiled without CUDA)
     * and provides the TensorBase interface by delegating to GPUTensorWrapper.
     */
    class GPUTensorWrapperAdapter;

    /**
     * @brief Create a TensorBase adapter for a GPUTensorWrapper
     *
     * This factory function creates a GPUTensorWrapperAdapter that wraps the
     * given GPUTensorWrapper and exposes it as TensorBase*.
     *
     * @param wrapper The GPUTensorWrapper to adapt
     * @return Pointer to adapter (ownership retained by caller)
     *
     * @note The wrapper must outlive the adapter
     * @note Implemented in GPUTensorWrapperAdapter.cpp
     */
    GPUTensorWrapperAdapter *createGPUTensorAdapter(GPUTensorWrapper *wrapper);

} // namespace llaminar2
