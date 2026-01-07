/**
 * @file CUDATensorBase.cu
 * @brief Implementation of CUDATensorBase GPU tensor base class
 */

#include "CUDATensorBase.h"
#include "../../utils/Logger.h"
#include <cuda_runtime.h>
#include <cstring>
#include <numeric>

namespace llaminar2
{

    CUDATensorBase::CUDATensorBase(std::vector<size_t> shape, TensorType dtype, int device_idx, cudaStream_t stream)
        : shape_(std::move(shape)), dtype_(dtype), device_idx_(device_idx), stream_(stream)
    {
    }

    CUDATensorBase::~CUDATensorBase()
    {
        freeDevice();
    }

    CUDATensorBase::CUDATensorBase(CUDATensorBase &&other) noexcept
        : device_ptr_(other.device_ptr_), device_idx_(other.device_idx_), stream_(other.stream_), shape_(std::move(other.shape_)), dtype_(other.dtype_), size_bytes_(other.size_bytes_), owns_memory_(other.owns_memory_)
    {
        other.device_ptr_ = nullptr;
        other.owns_memory_ = false;
    }

    CUDATensorBase &CUDATensorBase::operator=(CUDATensorBase &&other) noexcept
    {
        if (this != &other)
        {
            freeDevice();
            device_ptr_ = other.device_ptr_;
            device_idx_ = other.device_idx_;
            stream_ = other.stream_;
            shape_ = std::move(other.shape_);
            dtype_ = other.dtype_;
            size_bytes_ = other.size_bytes_;
            owns_memory_ = other.owns_memory_;
            other.device_ptr_ = nullptr;
            other.owns_memory_ = false;
        }
        return *this;
    }

    size_t CUDATensorBase::numel() const
    {
        if (shape_.empty())
            return 0;
        return std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<size_t>());
    }

    bool CUDATensorBase::allocateDevice(size_t bytes)
    {
        if (device_ptr_)
        {
            freeDevice();
        }

        cudaError_t err = cudaSetDevice(device_idx_);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDATensorBase] Failed to set device " << device_idx_ << ": " << cudaGetErrorString(err));
            return false;
        }

        err = cudaMalloc(&device_ptr_, bytes);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDATensorBase] Failed to allocate " << bytes << " bytes on device "
                                                             << device_idx_ << ": " << cudaGetErrorString(err));
            device_ptr_ = nullptr;
            return false;
        }

        size_bytes_ = bytes;
        owns_memory_ = true;
        LOG_TRACE("[CUDATensorBase] Allocated " << bytes << " bytes on device " << device_idx_);
        return true;
    }

    void CUDATensorBase::freeDevice()
    {
        if (device_ptr_ && owns_memory_)
        {
            cudaError_t err = cudaFree(device_ptr_);
            if (err != cudaSuccess)
            {
                LOG_WARN("[CUDATensorBase] cudaFree failed: " << cudaGetErrorString(err));
            }
            LOG_TRACE("[CUDATensorBase] Freed device memory on device " << device_idx_);
        }
        device_ptr_ = nullptr;
        size_bytes_ = 0;
    }

    void CUDATensorBase::copyToHost(void *host_dst, size_t bytes) const
    {
        if (!device_ptr_ || !host_dst || bytes == 0)
            return;

        cudaError_t err;
        if (stream_)
        {
            err = cudaMemcpyAsync(host_dst, device_ptr_, bytes, cudaMemcpyDeviceToHost, stream_);
            if (err == cudaSuccess)
            {
                err = cudaStreamSynchronize(stream_);
            }
        }
        else
        {
            err = cudaMemcpy(host_dst, device_ptr_, bytes, cudaMemcpyDeviceToHost);
        }

        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDATensorBase] D2H copy failed: " << cudaGetErrorString(err));
        }
    }

    void CUDATensorBase::copyFromHost(const void *host_src, size_t bytes)
    {
        if (!device_ptr_ || !host_src || bytes == 0)
            return;

        cudaError_t err;
        if (stream_)
        {
            err = cudaMemcpyAsync(device_ptr_, host_src, bytes, cudaMemcpyHostToDevice, stream_);
            // Note: Don't synchronize here - caller may want async behavior
        }
        else
        {
            err = cudaMemcpy(device_ptr_, host_src, bytes, cudaMemcpyHostToDevice);
        }

        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDATensorBase] H2D copy failed: " << cudaGetErrorString(err));
        }
    }

    void CUDATensorBase::to_fp32(float *dst) const
    {
        if (!dst)
            return;

        // For FP32 tensors, just D2H copy
        // For other types, would need kernel-based dequantization
        if (dtype_ == TensorType::FP32)
        {
            copyToHost(dst, numel() * sizeof(float));
        }
        else
        {
            // TODO: Implement kernel-based dequantization for other types
            // For now, log error - derived classes should override for quantized types
            LOG_ERROR("[CUDATensorBase] to_fp32 not implemented for "
                      << tensorTypeName(dtype_) << " CUDA tensors - override in derived class");
        }
    }

} // namespace llaminar2
