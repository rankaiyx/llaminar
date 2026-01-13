/**
 * @file DeviceGroup.h
 * @brief Device group concept for collective operations
 *
 * A DeviceGroup represents a set of devices participating in collective
 * communication operations. Groups can be:
 * - Homogeneous: All same device type (enables optimal backend selection)
 * - Heterogeneous: Mixed device types (requires Host backend fallback)
 *
 * Hierarchical parallelism:
 * - LocalGroup: Devices within single MPI rank (intra-node)
 * - GlobalGroup: One device per rank across MPI (inter-node)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include "../backends/DeviceType.h"
#include <algorithm>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    struct RankInventory;
    struct ClusterInventory;

    /**
     * @brief Scope of collective operation
     */
    enum class CollectiveScope
    {
        LOCAL,  ///< Within single MPI rank (intra-node)
        GLOBAL, ///< Across MPI ranks (inter-node)
        HYBRID  ///< Both local and global (hierarchical)
    };

    /**
     * @brief A group of devices participating in collective operations
     *
     * DeviceGroups are used to:
     * 1. Define which devices participate in a collective
     * 2. Determine optimal backend (NCCL vs RCCL vs Host)
     * 3. Map local rank to device within the group
     *
     * Example groups:
     * - "cuda_gpus_rank0": [GPU:0, GPU:1, GPU:2, GPU:3] (4 CUDA GPUs on rank 0)
     * - "all_devices_rank0": [GPU:0, GPU:1, CPU:0] (2 GPUs + CPU on rank 0)
     * - "global_rank_devices": [rank0:GPU:0, rank1:GPU:0] (one GPU per rank)
     */
    struct DeviceGroup
    {
        /// Human-readable name for debugging
        std::string name;

        /// Ordered list of participating devices
        std::vector<DeviceId> devices;

        /// This process's index in the group (0 to size()-1)
        /// For local groups: index of local device
        /// For global groups: MPI rank
        int local_rank = 0;

        /// Scope of this group
        CollectiveScope scope = CollectiveScope::LOCAL;

        // =====================================================================
        // Topology metadata (used for backend selection)
        // =====================================================================

        /// True if all devices are same type
        bool is_homogeneous = true;

        /// Primary device type (most common, or only type if homogeneous)
        DeviceType primary_type = DeviceType::CPU;

        /// Count of each device type
        int cuda_count = 0;
        int rocm_count = 0;
        int cpu_count = 0;

        // =====================================================================
        // Convenience methods
        // =====================================================================

        /// Number of devices in group
        size_t size() const { return devices.size(); }

        /// Get this process's device in the group
        DeviceId localDevice() const
        {
            if (local_rank >= 0 && local_rank < static_cast<int>(devices.size()))
            {
                return devices[local_rank];
            }
            return DeviceId::cpu();
        }

        /// Check if all devices are CUDA
        bool allCUDA() const { return is_homogeneous && primary_type == DeviceType::CUDA; }

        /// Check if all devices are ROCm
        bool allROCm() const { return is_homogeneous && primary_type == DeviceType::ROCm; }

        /// Check if all devices are CPU
        bool allCPU() const { return is_homogeneous && primary_type == DeviceType::CPU; }

        /// Check if group has any GPU
        bool hasGPU() const { return cuda_count > 0 || rocm_count > 0; }

        /// Check if group is heterogeneous (mixed types)
        bool isHeterogeneous() const { return !is_homogeneous; }

        /// Check if this is a local (intra-node) group
        bool isLocal() const { return scope == CollectiveScope::LOCAL; }

        /// Check if this is a global (inter-node) group
        bool isGlobal() const { return scope == CollectiveScope::GLOBAL; }

        // =====================================================================
        // String conversion
        // =====================================================================

        std::string toString() const
        {
            std::string result = name + " [";
            for (size_t i = 0; i < devices.size(); ++i)
            {
                if (i > 0)
                    result += ", ";
                result += devices[i].toString();
                if (static_cast<int>(i) == local_rank)
                    result += "*"; // Mark local device
            }
            result += "]";
            return result;
        }

        // =====================================================================
        // Validation
        // =====================================================================

        bool isValid() const
        {
            return !devices.empty() &&
                   local_rank >= 0 &&
                   local_rank < static_cast<int>(devices.size());
        }
    };

    /**
     * @brief Builder for creating DeviceGroups
     *
     * Provides fluent API for constructing groups with automatic
     * metadata computation (homogeneity, device counts, etc.)
     */
    class DeviceGroupBuilder
    {
    public:
        DeviceGroupBuilder &setName(const std::string &name)
        {
            group_.name = name;
            return *this;
        }

        DeviceGroupBuilder &addDevice(DeviceId device)
        {
            group_.devices.push_back(device);
            updateMetadata(device);
            return *this;
        }

        DeviceGroupBuilder &addDevices(const std::vector<DeviceId> &devices)
        {
            for (const auto &d : devices)
            {
                addDevice(d);
            }
            return *this;
        }

        DeviceGroupBuilder &setLocalRank(int rank)
        {
            group_.local_rank = rank;
            return *this;
        }

        DeviceGroupBuilder &setScope(CollectiveScope scope)
        {
            group_.scope = scope;
            return *this;
        }

        DeviceGroup build()
        {
            finalizeMetadata();
            return group_;
        }

    private:
        DeviceGroup group_;

        void updateMetadata(DeviceId device)
        {
            if (device.is_cuda())
            {
                group_.cuda_count++;
            }
            else if (device.is_rocm())
            {
                group_.rocm_count++;
            }
            else if (device.is_cpu())
            {
                group_.cpu_count++;
            }
        }

        void finalizeMetadata()
        {
            // Determine homogeneity and primary type
            int type_count = 0;
            if (group_.cuda_count > 0)
            {
                type_count++;
                group_.primary_type = DeviceType::CUDA;
            }
            if (group_.rocm_count > 0)
            {
                type_count++;
                group_.primary_type = DeviceType::ROCm;
            }
            if (group_.cpu_count > 0)
            {
                type_count++;
                if (type_count == 1)
                    group_.primary_type = DeviceType::CPU;
            }

            group_.is_homogeneous = (type_count <= 1);

            // Set primary type to most common
            int max_count = std::max({group_.cuda_count, group_.rocm_count, group_.cpu_count});
            if (group_.cuda_count == max_count)
            {
                group_.primary_type = DeviceType::CUDA;
            }
            else if (group_.rocm_count == max_count)
            {
                group_.primary_type = DeviceType::ROCm;
            }
            else
            {
                group_.primary_type = DeviceType::CPU;
            }
        }
    };

    /**
     * @brief Factory for creating common device group configurations
     */
    class DeviceGroupFactory
    {
    public:
        /**
         * @brief Create group of all CUDA GPUs on this rank
         * @param inventory Rank's device inventory
         * @param local_device_idx Which local GPU is "this" device (for local_rank)
         */
        static DeviceGroup createLocalCUDAGroup(
            const RankInventory &inventory,
            int local_device_idx);

        /**
         * @brief Create group of all ROCm GPUs on this rank
         */
        static DeviceGroup createLocalROCmGroup(
            const RankInventory &inventory,
            int local_device_idx);

        /**
         * @brief Create group of all devices on this rank (heterogeneous)
         */
        static DeviceGroup createLocalAllDevicesGroup(
            const RankInventory &inventory,
            int local_device_idx);

        /**
         * @brief Create group of CPUs only (one per rank for MPI)
         */
        static DeviceGroup createLocalCPUGroup(int rank);

        /**
         * @brief Create global group spanning all MPI ranks
         *
         * Used for inter-node MPI collectives. Each rank contributes
         * one device (typically its primary compute device).
         *
         * @param inventory Full cluster inventory
         * @param rank This rank's index
         * @param local_device Device this rank contributes to the group
         */
        static DeviceGroup createGlobalGroup(
            const ClusterInventory &inventory,
            int rank,
            DeviceId local_device);

        /**
         * @brief Create a subgroup from an existing group
         *
         * Useful for creating same-type subgroups from heterogeneous groups.
         *
         * @param parent Parent group to filter
         * @param type Device type to include
         * @param local_device_idx Index of local device in new subgroup
         */
        static DeviceGroup createSubgroup(
            const DeviceGroup &parent,
            DeviceType type,
            int local_device_idx);

        /**
         * @brief Partition a heterogeneous group into homogeneous subgroups
         *
         * Returns groups for each device type present.
         * Used for multi-phase heterogeneous collectives.
         */
        static std::vector<DeviceGroup> partitionByType(const DeviceGroup &group);
    };

} // namespace llaminar2
