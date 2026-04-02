/**
 * @file ExpertPlacementMap.h
 * @brief Maps experts to devices and tracks usage for dynamic rebalancing
 *
 * The ExpertPlacementMap is the single source of truth for where each
 * expert's weights live. It supports:
 * - Static placement (configured at startup)
 * - Dynamic rebalancing based on usage histograms
 * - Querying device for any expert at O(1) cost
 */

#pragma once

#include "../../backends/DeviceId.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Placement strategy for initial expert distribution
     */
    enum class ExpertPlacementStrategy
    {
        ALL_LOCAL,      ///< All experts on the execution device
        ROUND_ROBIN,    ///< Distribute across devices in round-robin
        CAPACITY_AWARE, ///< Distribute based on device memory capacity
    };

    /**
     * @brief Maps every expert to a device and tracks access frequency
     *
     * Thread-safe for reads after initialization. Usage recording is
     * internally synchronized.
     */
    class ExpertPlacementMap
    {
    public:
        /**
         * @brief Construct with uniform placement
         * @param num_experts   Total expert count
         * @param default_device Device for all experts initially
         */
        ExpertPlacementMap(int num_experts, DeviceId default_device);

        /**
         * @brief Construct with multi-device placement
         * @param num_experts Total expert count
         * @param devices    Available devices
         * @param strategy   How to distribute across devices
         */
        ExpertPlacementMap(int num_experts,
                           const std::vector<DeviceId> &devices,
                           ExpertPlacementStrategy strategy);

        // ── Queries ──────────────────────────────────────────────────────

        /// Device where expert_id's weights currently reside
        DeviceId deviceForExpert(int expert_id) const;

        /// All experts currently on a given device
        std::vector<int> expertsOnDevice(DeviceId device) const;

        /// Total expert count
        int numExperts() const { return num_experts_; }

        /// Devices in use
        const std::vector<DeviceId> &devices() const { return devices_; }

        // ── Mutation (thread-safe) ───────────────────────────────────────

        /// Move an expert to a new device
        void moveExpert(int expert_id, DeviceId new_device);

        /// Bulk move: apply a complete new placement
        void applyPlacement(const std::vector<DeviceId> &expert_to_device);

        // ── Usage tracking (thread-safe) ─────────────────────────────────

        /// Record that expert was activated (called from router)
        void recordActivation(int expert_id);

        /// Get activation count for an expert since last reset
        uint64_t activationCount(int expert_id) const;

        /// Get full histogram (indexed by expert_id)
        std::vector<uint64_t> activationHistogram() const;

        /// Reset all activation counts to zero
        void resetActivationCounts();

        // ── Diagnostics ──────────────────────────────────────────────────

        /// Human-readable summary
        std::string summary() const;

    private:
        int num_experts_;
        std::vector<DeviceId> devices_;
        std::vector<DeviceId> expert_to_device_; ///< expert_id → device
        mutable std::mutex usage_mutex_;
        std::vector<uint64_t> activation_counts_;
    };

} // namespace llaminar2
