/**
 * @file IDeviceRegistry.h
 * @brief Abstract interface for device registry
 *
 * Provides a mockable interface for device discovery and management.
 * The DeviceRegistry singleton implements this interface for production use.
 *
 * **Design Philosophy**:
 * - Unified device discovery across CPU, CUDA, ROCm
 * - GlobalDeviceAddress as the canonical identifier
 * - Thread-safe, lazy initialization
 * - Topology-aware (NUMA, PCIe bus, P2P capability)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "GlobalDeviceAddress.h"
#include <memory>
#include <vector>
#include <string>
#include <utility>

namespace llaminar2
{

    // Forward declarations
    class IBackend;

    /**
     * @brief Abstract interface for device registry
     *
     * This interface provides a mockable abstraction for device discovery
     * and management. The DeviceRegistry singleton implements this interface.
     *
     * **Thread Safety**: All methods are thread-safe in the concrete implementation.
     */
    class IDeviceRegistry
    {
    public:
        virtual ~IDeviceRegistry() = default;

        // =========================================================================
        // Discovery
        // =========================================================================

        /**
         * @brief Discover all available devices on this machine
         *
         * Scans for CPU NUMA nodes, CUDA GPUs, and ROCm GPUs.
         * Safe to call multiple times (re-scans if topology changed).
         */
        virtual void discover() = 0;

        /**
         * @brief Check if discovery has been run
         * @return true if discover() has been called
         */
        virtual bool isDiscovered() const = 0;

        /**
         * @brief Get all discovered devices
         * @return Vector of all device addresses
         */
        virtual std::vector<GlobalDeviceAddress> allDevices() const = 0;

        /**
         * @brief Get devices filtered by type
         * @param type Device type to filter by (CPU, CUDA, ROCm)
         * @return Vector of matching device addresses
         */
        virtual std::vector<GlobalDeviceAddress> devicesByType(DeviceType type) const = 0;

        /**
         * @brief Get devices filtered by NUMA node
         * @param numa_node NUMA node to filter by
         * @return Vector of matching device addresses
         */
        virtual std::vector<GlobalDeviceAddress> devicesByNuma(int numa_node) const = 0;

        // =========================================================================
        // Validation
        // =========================================================================

        /**
         * @brief Check if a device address is valid (exists on this machine)
         * @param addr Device address to validate
         * @return true if the device exists
         */
        virtual bool isValid(const GlobalDeviceAddress &addr) const = 0;

        /**
         * @brief Check if a device is available for use
         *
         * A device may exist but be unavailable if:
         * - It's already in exclusive use
         * - It has insufficient memory
         * - It's in an error state
         *
         * @param addr Device address to check
         * @return true if the device is available
         */
        virtual bool isAvailable(const GlobalDeviceAddress &addr) const = 0;

        // =========================================================================
        // Device Properties
        // =========================================================================

        /**
         * @brief Get total memory capacity for a device
         * @param addr Device address
         * @return Memory capacity in bytes (0 if device not found)
         */
        virtual size_t memoryCapacity(const GlobalDeviceAddress &addr) const = 0;

        /**
         * @brief Get available memory for a device
         * @param addr Device address
         * @return Available memory in bytes (0 if device not found)
         */
        virtual size_t memoryAvailable(const GlobalDeviceAddress &addr) const = 0;

        /**
         * @brief Get compute capability for a device
         *
         * For CUDA devices: SM version (e.g., {8, 6} for sm_86 Ampere)
         * For ROCm devices: GCN architecture version
         * For CPU: {0, 0}
         *
         * @param addr Device address
         * @return Pair of (major, minor) compute capability
         */
        virtual std::pair<int, int> computeCapability(const GlobalDeviceAddress &addr) const = 0;

        /**
         * @brief Get device name/description
         * @param addr Device address
         * @return Human-readable device name (empty if not found)
         */
        virtual std::string deviceName(const GlobalDeviceAddress &addr) const = 0;

        // =========================================================================
        // Backend Access
        // =========================================================================

        /**
         * @brief Get the IBackend for a device
         *
         * Returns the appropriate backend (CPUBackend, CUDABackend, ROCmBackend)
         * for memory operations on the device.
         *
         * @param addr Device address
         * @return Backend pointer (nullptr if device not found or backend unavailable)
         */
        virtual IBackend *backendFor(const GlobalDeviceAddress &addr) = 0;

        // =========================================================================
        // Topology
        // =========================================================================

        /**
         * @brief Get NUMA affinity for a device
         *
         * For GPUs: Returns the NUMA node closest to the GPU (for optimal data transfers)
         * For CPU: Returns the NUMA node specified in the address
         *
         * @param addr Device address
         * @return NUMA node (-1 if unknown or device not found)
         */
        virtual int numaAffinity(const GlobalDeviceAddress &addr) const = 0;

        /**
         * @brief Get PCIe bus ID for a device
         *
         * Format: "domain:bus:device.function" (e.g., "0000:3b:00.0")
         * Used for P2P capability detection and topology analysis.
         *
         * @param addr Device address
         * @return PCIe bus ID string (empty for CPU devices or if not found)
         */
        virtual std::string pcieBusId(const GlobalDeviceAddress &addr) const = 0;

        /**
         * @brief Check P2P (peer-to-peer) capability between two devices
         *
         * Returns true if direct memory access is possible between devices
         * without going through host memory.
         *
         * @param a First device address
         * @param b Second device address
         * @return true if P2P is supported between the devices
         */
        virtual bool canP2P(const GlobalDeviceAddress &a, const GlobalDeviceAddress &b) const = 0;

        // =========================================================================
        // Count Helpers
        // =========================================================================

        /**
         * @brief Get count of devices by type
         * @param type Device type
         * @return Number of devices of that type
         */
        virtual size_t deviceCount(DeviceType type) const = 0;

        /**
         * @brief Get total device count
         * @return Total number of discovered devices
         */
        virtual size_t totalDeviceCount() const = 0;
    };

} // namespace llaminar2
