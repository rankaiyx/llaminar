/**
 * @file CrossVendorP2P_CUDA.h
 * @brief CUDA-side helpers for cross-vendor P2P transfers
 *
 * Provides CUDA operations that can be called from the common
 * CrossVendorP2P.cpp without including cuda_runtime.h directly.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    namespace cuda_p2p
    {
        /**
         * @brief Register host memory with CUDA for fast DMA access
         * @return true if registration succeeded
         */
        bool registerHostMemory(void *ptr, size_t size);

        /**
         * @brief Unregister host memory from CUDA
         */
        void unregisterHostMemory(void *ptr);

        /**
         * @brief Create a CUDA stream for async operations
         * @return Opaque stream handle (cudaStream_t cast to void*)
         */
        void *createStream(int device_ordinal);

        /**
         * @brief Destroy a CUDA stream
         */
        void destroyStream(void *stream);

        /**
         * @brief Async copy from device to host
         * @return true if copy was queued successfully
         */
        bool asyncD2H(void *stream, int device_ordinal,
                      void *h_dst, const void *d_src, size_t num_bytes);

        /**
         * @brief Async copy from host to device
         * @return true if copy was queued successfully
         */
        bool asyncH2D(void *stream, int device_ordinal,
                      void *d_dst, const void *h_src, size_t num_bytes);

        /**
         * @brief Record an event on a stream
         * @return Opaque event handle (cudaEvent_t cast to void*)
         */
        void *recordEvent(void *stream, int device_ordinal);

        /**
         * @brief Destroy an event
         */
        void destroyEvent(void *event);

        /**
         * @brief Synchronize stream (wait for all operations to complete)
         */
        void syncStream(void *stream);

        /**
         * @brief Get elapsed time between two events in milliseconds
         */
        float getElapsedMs(void *start_event, void *end_event);

        /**
         * @brief Allocate device memory
         * @return Device pointer or nullptr on failure
         */
        void *allocDevice(int device_ordinal, size_t size);

        /**
         * @brief Free device memory
         */
        void freeDevice(int device_ordinal, void *ptr);

        /**
         * @brief Set device memory to a pattern
         */
        void memsetDevice(int device_ordinal, void *ptr, int value, size_t size);

    } // namespace cuda_p2p
} // namespace llaminar2
