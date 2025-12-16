/**
 * @file BackendManager.cpp
 * @brief Global GPU backend accessor implementation (Phase 6: Heterogeneous Multi-GPU)
 *
 * Supports BOTH CUDA and ROCm backends simultaneously for heterogeneous multi-GPU.
 *
 * @author David Sanftenberg
 */

#include "BackendManager.h"
#include "../utils/Logger.h"

#ifdef HAVE_CUDA
#include "cuda/CUDABackend.h"
#endif

#ifdef HAVE_ROCM
#include "rocm/ROCmBackend.h"
#endif

#include <mutex>

namespace llaminar2
{

    namespace
    {
        // Phase 6: Support both CUDA and ROCm backends simultaneously
        IBackend *g_cuda_backend = nullptr;
        IBackend *g_rocm_backend = nullptr;
        std::once_flag g_cuda_init_flag;
        std::once_flag g_rocm_init_flag;

        void initCUDABackend()
        {
#ifdef HAVE_CUDA
            g_cuda_backend = new CUDABackend();
            LOG_INFO("[BackendManager] Initialized CUDA backend (" << g_cuda_backend->deviceCount() << " devices)");
#else
            g_cuda_backend = nullptr;
            LOG_DEBUG("[BackendManager] CUDA backend not available (HAVE_CUDA not defined)");
#endif
        }

        void initROCmBackend()
        {
#ifdef HAVE_ROCM
            g_rocm_backend = new ROCmBackend();
            LOG_INFO("[BackendManager] Initialized ROCm backend (" << g_rocm_backend->deviceCount() << " devices)");
#else
            g_rocm_backend = nullptr;
            LOG_DEBUG("[BackendManager] ROCm backend not available (HAVE_ROCM not defined)");
#endif
        }
    } // anonymous namespace

    IBackend *getCUDABackend()
    {
        std::call_once(g_cuda_init_flag, initCUDABackend);
        return g_cuda_backend;
    }

    IBackend *getROCmBackend()
    {
        std::call_once(g_rocm_init_flag, initROCmBackend);
        return g_rocm_backend;
    }

    IBackend *getGPUBackend()
    {
        // Legacy function - prefer CUDA, fall back to ROCm
        IBackend *cuda = getCUDABackend();
        if (cuda)
            return cuda;
        return getROCmBackend();
    }

    IBackend *getBackendForDeviceType(ComputeBackendType type)
    {
        switch (type)
        {
        case ComputeBackendType::GPU_CUDA:
            return getCUDABackend();
        case ComputeBackendType::GPU_ROCM:
            return getROCmBackend();
        default:
            return nullptr; // CPU or unknown types don't use IBackend
        }
    }

    bool hasGPUBackend()
    {
        return getGPUBackend() != nullptr;
    }

    bool hasCUDABackend()
    {
        return getCUDABackend() != nullptr;
    }

    bool hasROCmBackend()
    {
        return getROCmBackend() != nullptr;
    }

} // namespace llaminar2
