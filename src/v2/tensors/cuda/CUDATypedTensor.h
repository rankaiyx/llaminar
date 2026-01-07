#pragma once
/**
 * @file CUDATypedTensor.h
 * @brief Template class for typed CUDA tensors with type-safe accessors
 *
 * CUDATypedTensor provides strongly-typed GPU tensors that inherit all ITensor
 * functionality from CUDATensorBase while adding compile-time type safety.
 *
 * Key features:
 * - Type-safe device pointer accessors (typed_device_ptr(), typed_data())
 * - Static type ID for runtime type checking (is<T>(), try_as<T>())
 * - Automatic GPU memory allocation based on shape and element size
 * - Support for non-owning views over existing device memory
 *
 * Usage:
 * @code
 * // Allocating tensor
 * CUDAFp32Tensor tensor({batch, seq_len, hidden}, gpu_idx);
 * float* d_ptr = tensor.typed_data();  // Type-safe device pointer
 *
 * // View over existing memory
 * float* existing_d_ptr = ...;
 * CUDAFp32Tensor view({batch, seq_len}, existing_d_ptr, gpu_idx);
 * // view doesn't own memory, won't free on destruction
 * @endcode
 *
 * @author David Sanftenberg
 */

#include "CUDATensorBase.h"
#include <memory>

namespace llaminar2 {

/**
 * @brief Template for strongly-typed CUDA tensors
 *
 * Provides type-safe device pointer accessors while inheriting
 * all ITensor functionality from CUDATensorBase.
 *
 * @tparam T Element type (float, __half, __nv_bfloat16, int8_t, etc.)
 * @tparam DType TensorType enum value for runtime type identification
 */
template<typename T, TensorType DType>
class CUDATypedTensor : public CUDATensorBase {
public:
    using ElementType = T;
    static constexpr TensorType StaticDType = DType;
    
    /**
     * @brief Construct tensor with automatic GPU memory allocation
     * @param shape Tensor dimensions
     * @param device_idx CUDA device ordinal (0-based)
     * @param stream CUDA stream for async operations (nullptr for default)
     */
    CUDATypedTensor(std::vector<size_t> shape, int device_idx, cudaStream_t stream = nullptr)
        : CUDATensorBase(std::move(shape), DType, device_idx, stream) {
        // Allocate device memory
        size_t bytes = numel() * sizeof(T);
        if (bytes > 0) {
            allocateDevice(bytes);
        }
    }
    
    /**
     * @brief Wrap existing device pointer (non-owning view)
     * @param shape Tensor dimensions
     * @param device_ptr Existing CUDA device pointer (NOT host pointer!)
     * @param device_idx CUDA device ordinal where device_ptr was allocated
     * @param stream CUDA stream for async operations (nullptr for default)
     *
     * @warning The caller must ensure device_ptr remains valid for the tensor's lifetime.
     *          This tensor will NOT free the memory on destruction.
     */
    CUDATypedTensor(std::vector<size_t> shape, T* device_ptr, int device_idx, cudaStream_t stream = nullptr)
        : CUDATensorBase(std::move(shape), DType, device_idx, stream) {
        device_ptr_ = device_ptr;
        size_bytes_ = numel() * sizeof(T);
        owns_memory_ = false;  // Don't free on destruction
    }
    
    ~CUDATypedTensor() override = default;
    
    // =========================================================================
    // Type-Safe Device Pointer Accessors
    // =========================================================================
    
    /**
     * @brief Get typed device pointer
     * @return T* pointing to GPU memory
     */
    T* typed_device_ptr() { return static_cast<T*>(device_ptr_); }
    const T* typed_device_ptr() const { return static_cast<const T*>(device_ptr_); }
    
    /**
     * @brief Convenience alias for typed_device_ptr()
     * @return T* pointing to GPU memory
     */
    T* typed_data() { return typed_device_ptr(); }
    const T* typed_data() const { return typed_device_ptr(); }
    
    /**
     * @brief Get typed device pointer (const)
     * @return const T* pointing to GPU memory
     * @note For compatibility with tensor interfaces expecting data()
     */
    const T* data() const { return typed_device_ptr(); }
    
    /**
     * @brief Get typed mutable device pointer
     * @return T* pointing to GPU memory
     */
    T* mutable_data() { return typed_device_ptr(); }
    
    // =========================================================================
    // Static Type Information (for ITensor::is<T>() support)
    // =========================================================================
    
    /**
     * @brief Get static type ID for this tensor type
     * @return Integer type ID matching TensorTypeId constants
     *
     * This enables runtime type checking:
     * @code
     * if (tensor.is<CUDAFp32Tensor>()) { ... }
     * @endcode
     */
    static constexpr int static_type_id() { return static_cast<int>(DType); }
};

// =========================================================================
// Concrete CUDA Tensor Type Aliases
// =========================================================================

// FP32 is always available
using CUDAFp32Tensor = CUDATypedTensor<float, TensorType::FP32>;

// Half-precision types require CUDA compilation
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * @brief CUDA FP16 tensor (IEEE half-precision)
 */
using CUDAFP16Tensor = CUDATypedTensor<__half, TensorType::FP16>;

/**
 * @brief CUDA BF16 tensor (Brain Float 16)
 */
using CUDABF16Tensor = CUDATypedTensor<__nv_bfloat16, TensorType::BF16>;

#endif // __CUDACC__

// INT8 tensor for quantized activations/weights
using CUDAINT8Tensor = CUDATypedTensor<int8_t, TensorType::INT8>;

// INT32 tensor for accumulator buffers
using CUDAINT32Tensor = CUDATypedTensor<int32_t, TensorType::INT32>;

} // namespace llaminar2
