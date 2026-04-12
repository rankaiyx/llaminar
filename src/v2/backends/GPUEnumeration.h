/**
 * @file GPUEnumeration.h
 * @brief Declarations for GPU device enumeration (separate compilation units)
 *
 * This header declares functions implemented in separate CUDA/ROCm compilation units
 * to avoid header conflicts when both backends are enabled.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <vector>
#include <string>
#include "ComputeBackend.h"

namespace llaminar2
{

    // Shared PCIe link info reader (works for any GPU via sysfs)
    namespace pcie_enumeration
    {
        /**
         * @brief Read PCIe link speed/width from sysfs for a PCI device
         * @param pci_domain PCI domain (usually 0)
         * @param pci_bus PCI bus number
         * @param pci_device PCI device number
         * @return Populated PCIeLinkInfo (zeros if sysfs read fails)
         */
        PCIeLinkInfo read_pcie_link_info(int pci_domain, int pci_bus, int pci_device);
    }

// CUDA enumeration (implemented in CUDAEnumeration.cu)
#ifdef HAVE_CUDA
    namespace cuda_enumeration
    {
        std::vector<ComputeDevice> enumerate_cuda_devices();
        int get_cuda_device_numa_node(int device_id);
        P2PMatrix query_p2p_matrix(const std::vector<ComputeDevice> &devices);
    }
#endif

// ROCm enumeration (implemented in ROCmEnumeration.cpp)
#ifdef HAVE_ROCM
    namespace rocm_enumeration
    {
        std::vector<ComputeDevice> enumerate_rocm_devices();
        int get_rocm_device_numa_node(int device_id);
        P2PMatrix query_p2p_matrix(const std::vector<ComputeDevice> &devices);
    }
#endif

} // namespace llaminar2
