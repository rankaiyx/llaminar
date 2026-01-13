/**
 * @file HostBackendROCm.cpp
 * @brief ROCm-specific helper functions for HostBackend
 *
 * Isolated HIP runtime calls in separate compilation unit to avoid
 * conflicts with CUDA headers.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <hip/hip_runtime.h>

namespace llaminar2 {
namespace host_backend_detail {

bool hipCopyToHost(void* host_dst, const void* device_src, int device_ordinal, size_t bytes)
{
    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        return false;
    }
    
    err = hipMemcpy(host_dst, device_src, bytes, hipMemcpyDeviceToHost);
    return (err == hipSuccess);
}

bool hipCopyFromHost(void* device_dst, const void* host_src, int device_ordinal, size_t bytes)
{
    hipError_t err = hipSetDevice(device_ordinal);
    if (err != hipSuccess)
    {
        return false;
    }
    
    err = hipMemcpy(device_dst, host_src, bytes, hipMemcpyHostToDevice);
    return (err == hipSuccess);
}

bool hipHostRegisterBuffer(void* ptr, size_t size)
{
    hipError_t err = hipHostRegister(ptr, size, hipHostRegisterPortable);
    return (err == hipSuccess);
}

void hipHostUnregisterBuffer(void* ptr)
{
    (void)hipHostUnregister(ptr);  // Ignore return value in cleanup
}

} // namespace host_backend_detail
} // namespace llaminar2
