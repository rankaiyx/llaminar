/**
 * @file DeviceAddressAdapter.h
 * @brief Utilities for converting between legacy device identification and GlobalDeviceAddress
 *
 * Provides bridge functions to convert between:
 * - (DeviceType, ordinal) pairs and GlobalDeviceAddress
 * - Raw device_id integers and GlobalDeviceAddress
 * - DeviceId and GlobalDeviceAddress
 *
 * **Migration Strategy**:
 * Use these adapters at API boundaries to gradually migrate from legacy
 * device identification to GlobalDeviceAddress throughout the codebase.
 *
 * **Example**:
 * ```cpp
 * // Legacy code
 * void legacyKernel(int device_id, DeviceType type);
 *
 * // New code
 * void newKernel(const GlobalDeviceAddress& addr) {
 *     int device_id = DeviceAddressAdapter::toOrdinal(addr);
 *     DeviceType type = DeviceAddressAdapter::extractType(addr);
 *     legacyKernel(device_id, type);
 * }
 *
 * // Or migrate callers
 * GlobalDeviceAddress addr = DeviceAddressAdapter::fromTypeAndOrdinal(type, ordinal);
 * newKernel(addr);
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "GlobalDeviceAddress.h"
#include "DeviceId.h"
#include <string>
#include <optional>

namespace llaminar2
{

    /**
     * @brief Utilities for converting between legacy device identification and GlobalDeviceAddress
     */
    class DeviceAddressAdapter
    {
    public:
        // =========================================================================
        // From Legacy to GlobalDeviceAddress
        // =========================================================================

        /**
         * @brief Convert (DeviceType, ordinal) to GlobalDeviceAddress
         *
         * Creates a local device address with the given type and ordinal.
         * Uses NUMA node 0 by default.
         *
         * @param type Device type (CPU, CUDA, ROCm)
         * @param ordinal Device ordinal (0-indexed)
         * @param numa_node NUMA node (default 0)
         * @return GlobalDeviceAddress for the device
         */
        static GlobalDeviceAddress fromTypeAndOrdinal(
            DeviceType type,
            int ordinal,
            int numa_node = 0);

        /**
         * @brief Convert raw device_id to GlobalDeviceAddress
         *
         * For legacy code that uses raw integers as device IDs.
         *
         * @param device_id Raw device ID (0-indexed)
         * @param assumed_type Device type to assume
         * @param numa_node NUMA node (default 0)
         * @return GlobalDeviceAddress for the device
         */
        static GlobalDeviceAddress fromDeviceId(
            int device_id,
            DeviceType assumed_type,
            int numa_node = 0);

        /**
         * @brief Convert from DeviceId to GlobalDeviceAddress
         *
         * @param device_id DeviceId struct
         * @param numa_node NUMA node (default 0)
         * @param hostname Hostname (default "localhost")
         * @return GlobalDeviceAddress for the device
         */
        static GlobalDeviceAddress fromDeviceId(
            const DeviceId &device_id,
            int numa_node = 0,
            const std::string &hostname = "localhost");

        /**
         * @brief Convert from CUDA device ordinal
         *
         * Convenience method for CUDA-specific code.
         *
         * @param cuda_device CUDA device ordinal (0-indexed)
         * @param numa_node NUMA node (default 0)
         * @return GlobalDeviceAddress for the CUDA device
         */
        static GlobalDeviceAddress fromCudaDevice(int cuda_device, int numa_node = 0);

        /**
         * @brief Convert from ROCm/HIP device ordinal
         *
         * Convenience method for ROCm-specific code.
         *
         * @param rocm_device ROCm device ordinal (0-indexed)
         * @param numa_node NUMA node (default 0)
         * @return GlobalDeviceAddress for the ROCm device
         */
        static GlobalDeviceAddress fromRocmDevice(int rocm_device, int numa_node = 0);

        /**
         * @brief Convert from CPU socket/NUMA ID
         *
         * @param socket_id Socket/NUMA node ID
         * @return GlobalDeviceAddress for the CPU
         */
        static GlobalDeviceAddress fromCpuSocket(int socket_id);

        // =========================================================================
        // From GlobalDeviceAddress to Legacy
        // =========================================================================

        /**
         * @brief Get device ordinal (for use with cuda/hip APIs)
         *
         * @param addr GlobalDeviceAddress
         * @return Device ordinal (0-indexed)
         */
        static int toOrdinal(const GlobalDeviceAddress &addr);

        /**
         * @brief Convert to DeviceId
         *
         * @param addr GlobalDeviceAddress
         * @return DeviceId struct
         */
        static DeviceId toDeviceId(const GlobalDeviceAddress &addr);

        /**
         * @brief Get CUDA device ID
         *
         * @param addr GlobalDeviceAddress (must be CUDA type)
         * @return CUDA device ordinal
         * @throws std::invalid_argument if not a CUDA device
         */
        static int toCudaDevice(const GlobalDeviceAddress &addr);

        /**
         * @brief Get ROCm device ID
         *
         * @param addr GlobalDeviceAddress (must be ROCm type)
         * @return ROCm device ordinal
         * @throws std::invalid_argument if not a ROCm device
         */
        static int toRocmDevice(const GlobalDeviceAddress &addr);

        /**
         * @brief Get NUMA node
         *
         * @param addr GlobalDeviceAddress
         * @return NUMA node
         */
        static int toNumaNode(const GlobalDeviceAddress &addr);

        // =========================================================================
        // DeviceType Utilities
        // =========================================================================

        /**
         * @brief Get DeviceType from GlobalDeviceAddress
         *
         * @param addr GlobalDeviceAddress
         * @return DeviceType enum value
         */
        static DeviceType extractType(const GlobalDeviceAddress &addr);

        /**
         * @brief Check if GPU (CUDA or ROCm)
         *
         * @param addr GlobalDeviceAddress
         * @return true if device is a GPU
         */
        static bool isGpu(const GlobalDeviceAddress &addr);

        /**
         * @brief Check if CPU
         *
         * @param addr GlobalDeviceAddress
         * @return true if device is a CPU
         */
        static bool isCpu(const GlobalDeviceAddress &addr);

        /**
         * @brief Check if CUDA
         *
         * @param addr GlobalDeviceAddress
         * @return true if device is a CUDA GPU
         */
        static bool isCuda(const GlobalDeviceAddress &addr);

        /**
         * @brief Check if ROCm
         *
         * @param addr GlobalDeviceAddress
         * @return true if device is a ROCm GPU
         */
        static bool isRocm(const GlobalDeviceAddress &addr);

        // =========================================================================
        // Validation
        // =========================================================================

        /**
         * @brief Validate that an address is suitable for kernel dispatch
         *
         * Checks that:
         * - Ordinal is non-negative
         * - Type is a recognized compute type (CPU, CUDA, ROCm)
         *
         * @param addr GlobalDeviceAddress
         * @return true if address is valid for compute
         */
        static bool isValidForCompute(const GlobalDeviceAddress &addr);
    };

} // namespace llaminar2
