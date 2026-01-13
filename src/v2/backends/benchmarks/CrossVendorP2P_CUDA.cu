/**
 * @file CrossVendorP2P_CUDA.cu
 * @brief CUDA implementation of cross-vendor P2P helpers
 */

#include "CrossVendorP2P_CUDA.h"
#include <cuda_runtime.h>

namespace llaminar2
{
    namespace cuda_p2p
    {
        bool registerHostMemory(void *ptr, size_t size)
        {
            cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
            return err == cudaSuccess;
        }

        void unregisterHostMemory(void *ptr)
        {
            cudaHostUnregister(ptr);
        }

        void *createStream(int device_ordinal)
        {
            cudaSetDevice(device_ordinal);
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err != cudaSuccess)
                return nullptr;
            return reinterpret_cast<void *>(stream);
        }

        void destroyStream(void *stream)
        {
            if (stream)
            {
                cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
            }
        }

        bool asyncD2H(void *stream, int device_ordinal,
                      void *h_dst, const void *d_src, size_t num_bytes)
        {
            cudaSetDevice(device_ordinal);
            cudaError_t err = cudaMemcpyAsync(h_dst, d_src, num_bytes,
                                               cudaMemcpyDeviceToHost,
                                               reinterpret_cast<cudaStream_t>(stream));
            return err == cudaSuccess;
        }

        bool asyncH2D(void *stream, int device_ordinal,
                      void *d_dst, const void *h_src, size_t num_bytes)
        {
            cudaSetDevice(device_ordinal);
            cudaError_t err = cudaMemcpyAsync(d_dst, h_src, num_bytes,
                                               cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream));
            return err == cudaSuccess;
        }

        void *recordEvent(void *stream, int device_ordinal)
        {
            cudaSetDevice(device_ordinal);
            cudaEvent_t event;
            cudaError_t err = cudaEventCreate(&event);
            if (err != cudaSuccess)
                return nullptr;
            err = cudaEventRecord(event, reinterpret_cast<cudaStream_t>(stream));
            if (err != cudaSuccess)
            {
                cudaEventDestroy(event);
                return nullptr;
            }
            return reinterpret_cast<void *>(event);
        }

        void destroyEvent(void *event)
        {
            if (event)
            {
                cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
            }
        }

        void syncStream(void *stream)
        {
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        }

        float getElapsedMs(void *start_event, void *end_event)
        {
            float ms = 0;
            cudaEventElapsedTime(&ms,
                                  reinterpret_cast<cudaEvent_t>(start_event),
                                  reinterpret_cast<cudaEvent_t>(end_event));
            return ms;
        }

        void *allocDevice(int device_ordinal, size_t size)
        {
            cudaSetDevice(device_ordinal);
            void *ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess)
                return nullptr;
            return ptr;
        }

        void freeDevice(int device_ordinal, void *ptr)
        {
            if (ptr)
            {
                cudaSetDevice(device_ordinal);
                cudaFree(ptr);
            }
        }

        void memsetDevice(int device_ordinal, void *ptr, int value, size_t size)
        {
            cudaSetDevice(device_ordinal);
            cudaMemset(ptr, value, size);
            cudaDeviceSynchronize();
        }

    } // namespace cuda_p2p
} // namespace llaminar2
