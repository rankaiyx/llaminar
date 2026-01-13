/**
 * @file CrossVendorP2P_ROCm.h
 * @brief ROCm-side helpers for cross-vendor P2P transfers
 *
 * Provides HIP operations that can be called from the common
 * CrossVendorP2P.cpp without including hip_runtime.h directly.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    namespace rocm_p2p
    {
        /**
         * @brief Register host memory with HIP for fast DMA access
         * @return true if registration succeeded
         */
        bool registerHostMemory(void *ptr, size_t size);

        /**
         * @brief Unregister host memory from HIP
         */
        void unregisterHostMemory(void *ptr);

        /**
         * @brief Create a HIP stream for async operations
         * @return Opaque stream handle (hipStream_t cast to void*)
         */
        void *createStream(int device_ordinal);

        /**
         * @brief Destroy a HIP stream
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
         * @return Opaque event handle (hipEvent_t cast to void*)
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

    } // namespace rocm_p2p
} // namespace llaminar2
