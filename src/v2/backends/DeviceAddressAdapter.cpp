/**
 * @file DeviceAddressAdapter.cpp
 * @brief Implementation of DeviceAddressAdapter utilities
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "DeviceAddressAdapter.h"
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // From Legacy to GlobalDeviceAddress
    // =========================================================================

    GlobalDeviceAddress DeviceAddressAdapter::fromTypeAndOrdinal(
        DeviceType type,
        int ordinal,
        int numa_node)
    {
        GlobalDeviceAddress addr;
        addr.hostname = "localhost";
        addr.numa_node = numa_node;
        addr.device_type = type;
        addr.device_ordinal = ordinal;
        return addr;
    }

    GlobalDeviceAddress DeviceAddressAdapter::fromDeviceId(
        int device_id,
        DeviceType assumed_type,
        int numa_node)
    {
        return fromTypeAndOrdinal(assumed_type, device_id, numa_node);
    }

    GlobalDeviceAddress DeviceAddressAdapter::fromDeviceId(
        const DeviceId &device_id,
        int numa_node,
        const std::string &hostname)
    {
        GlobalDeviceAddress addr;
        addr.hostname = hostname;
        addr.numa_node = numa_node;
        addr.device_type = device_id.type;
        addr.device_ordinal = device_id.ordinal;
        return addr;
    }

    GlobalDeviceAddress DeviceAddressAdapter::fromCudaDevice(int cuda_device, int numa_node)
    {
        return GlobalDeviceAddress::cuda(cuda_device, numa_node);
    }

    GlobalDeviceAddress DeviceAddressAdapter::fromRocmDevice(int rocm_device, int numa_node)
    {
        return GlobalDeviceAddress::rocm(rocm_device, numa_node);
    }

    GlobalDeviceAddress DeviceAddressAdapter::fromCpuSocket(int socket_id)
    {
        return GlobalDeviceAddress::cpu(socket_id);
    }

    // =========================================================================
    // From GlobalDeviceAddress to Legacy
    // =========================================================================

    int DeviceAddressAdapter::toOrdinal(const GlobalDeviceAddress &addr)
    {
        return addr.device_ordinal;
    }

    DeviceId DeviceAddressAdapter::toDeviceId(const GlobalDeviceAddress &addr)
    {
        return addr.toLocalDeviceId();
    }

    int DeviceAddressAdapter::toCudaDevice(const GlobalDeviceAddress &addr)
    {
        if (addr.device_type != DeviceType::CUDA)
        {
            throw std::invalid_argument(
                "toCudaDevice called on non-CUDA device: " + addr.toString());
        }
        return addr.device_ordinal;
    }

    int DeviceAddressAdapter::toRocmDevice(const GlobalDeviceAddress &addr)
    {
        if (addr.device_type != DeviceType::ROCm)
        {
            throw std::invalid_argument(
                "toRocmDevice called on non-ROCm device: " + addr.toString());
        }
        return addr.device_ordinal;
    }

    int DeviceAddressAdapter::toNumaNode(const GlobalDeviceAddress &addr)
    {
        return addr.numa_node;
    }

    // =========================================================================
    // DeviceType Utilities
    // =========================================================================

    DeviceType DeviceAddressAdapter::extractType(const GlobalDeviceAddress &addr)
    {
        return addr.device_type;
    }

    bool DeviceAddressAdapter::isGpu(const GlobalDeviceAddress &addr)
    {
        return addr.isGPU();
    }

    bool DeviceAddressAdapter::isCpu(const GlobalDeviceAddress &addr)
    {
        return addr.isCPU();
    }

    bool DeviceAddressAdapter::isCuda(const GlobalDeviceAddress &addr)
    {
        return addr.isCUDA();
    }

    bool DeviceAddressAdapter::isRocm(const GlobalDeviceAddress &addr)
    {
        return addr.isROCm();
    }

    // =========================================================================
    // Validation
    // =========================================================================

    bool DeviceAddressAdapter::isValidForCompute(const GlobalDeviceAddress &addr)
    {
        // Check ordinal is valid
        if (addr.device_ordinal < 0)
        {
            return false;
        }

        // Check type is a recognized compute type
        switch (addr.device_type)
        {
        case DeviceType::CPU:
        case DeviceType::CUDA:
        case DeviceType::ROCm:
            return true;
        case DeviceType::Vulkan:
        case DeviceType::Metal:
            // Not yet supported for compute
            return false;
        default:
            return false;
        }
    }

} // namespace llaminar2
