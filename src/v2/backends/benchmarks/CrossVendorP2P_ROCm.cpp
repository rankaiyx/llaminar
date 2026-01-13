/**
 * @file CrossVendorP2P_ROCm.cpp
 * @brief ROCm/HIP implementation of cross-vendor P2P helpers
 *
 * This file is compiled as HIP source (via CMake set_source_files_properties).
 */

#include "CrossVendorP2P_ROCm.h"
#include <hip/hip_runtime.h>

namespace llaminar2
{
    namespace rocm_p2p
    {
        bool registerHostMemory(void *ptr, size_t size)
        {
            hipError_t err = hipHostRegister(ptr, size, hipHostRegisterPortable);
            return err == hipSuccess;
        }

        void unregisterHostMemory(void *ptr)
        {
            hipHostUnregister(ptr);
        }

        void *createStream(int device_ordinal)
        {
            hipSetDevice(device_ordinal);
            hipStream_t stream;
            hipError_t err = hipStreamCreate(&stream);
            if (err != hipSuccess)
                return nullptr;
            return reinterpret_cast<void *>(stream);
        }

        void destroyStream(void *stream)
        {
            if (stream)
            {
                hipStreamDestroy(reinterpret_cast<hipStream_t>(stream));
            }
        }

        bool asyncD2H(void *stream, int device_ordinal,
                      void *h_dst, const void *d_src, size_t num_bytes)
        {
            hipSetDevice(device_ordinal);
            hipError_t err = hipMemcpyAsync(h_dst, d_src, num_bytes,
                                             hipMemcpyDeviceToHost,
                                             reinterpret_cast<hipStream_t>(stream));
            return err == hipSuccess;
        }

        bool asyncH2D(void *stream, int device_ordinal,
                      void *d_dst, const void *h_src, size_t num_bytes)
        {
            hipSetDevice(device_ordinal);
            hipError_t err = hipMemcpyAsync(d_dst, h_src, num_bytes,
                                             hipMemcpyHostToDevice,
                                             reinterpret_cast<hipStream_t>(stream));
            return err == hipSuccess;
        }

        void *recordEvent(void *stream, int device_ordinal)
        {
            hipSetDevice(device_ordinal);
            hipEvent_t event;
            hipError_t err = hipEventCreate(&event);
            if (err != hipSuccess)
                return nullptr;
            err = hipEventRecord(event, reinterpret_cast<hipStream_t>(stream));
            if (err != hipSuccess)
            {
                hipEventDestroy(event);
                return nullptr;
            }
            return reinterpret_cast<void *>(event);
        }

        void destroyEvent(void *event)
        {
            if (event)
            {
                hipEventDestroy(reinterpret_cast<hipEvent_t>(event));
            }
        }

        void syncStream(void *stream)
        {
            hipStreamSynchronize(reinterpret_cast<hipStream_t>(stream));
        }

        float getElapsedMs(void *start_event, void *end_event)
        {
            float ms = 0;
            hipEventElapsedTime(&ms,
                                 reinterpret_cast<hipEvent_t>(start_event),
                                 reinterpret_cast<hipEvent_t>(end_event));
            return ms;
        }

        void *allocDevice(int device_ordinal, size_t size)
        {
            hipSetDevice(device_ordinal);
            void *ptr = nullptr;
            hipError_t err = hipMalloc(&ptr, size);
            if (err != hipSuccess)
                return nullptr;
            return ptr;
        }

        void freeDevice(int device_ordinal, void *ptr)
        {
            if (ptr)
            {
                hipSetDevice(device_ordinal);
                hipFree(ptr);
            }
        }

        void memsetDevice(int device_ordinal, void *ptr, int value, size_t size)
        {
            hipSetDevice(device_ordinal);
            hipMemset(ptr, value, size);
            hipDeviceSynchronize();
        }

    } // namespace rocm_p2p
} // namespace llaminar2
