/**
 * @file ExpertRebalancer.h
 * @brief Histogram-based dynamic expert rebalancing
 *
 * Analyzes expert activation patterns and proposes placement changes
 * to improve locality. Hot experts get moved to fast devices; cold
 * experts get offloaded to CPU.
 */

#pragma once

#include "ExpertPlacementMap.h"
#include <vector>

namespace llaminar2
{

    /**
     * @brief Proposed expert movement
     */
    struct ExpertMovement
    {
        int expert_id;
        DeviceId from_device;
        DeviceId to_device;
        uint64_t activation_count; ///< Why this expert was chosen for movement
    };

    /**
     * @brief Rebalance proposal: a set of movements to apply
     */
    struct RebalanceProposal
    {
        std::vector<ExpertMovement> movements;
        double expected_locality_improvement; ///< Estimated % improvement

        int numMovements() const { return static_cast<int>(movements.size()); }
        bool empty() const { return movements.empty(); }
    };

    /**
     * @brief Parameters for the rebalancing algorithm
     */
    struct RebalanceParams
    {
        /// Minimum activation count ratio (hot/cold) to trigger a move
        float activation_ratio_threshold = 2.0f;

        /// Maximum number of expert moves per rebalance cycle
        int max_moves_per_cycle = 8;

        /// Minimum total activations before rebalancing is considered
        uint64_t min_total_activations = 100;

        /// Target: fraction of total activations that are "local"
        /// (on the device that's doing the computation)
        float target_locality = 0.8f;
    };

    /**
     * @brief Analyzes activation histograms and proposes expert movements
     *
     * Algorithm:
     * 1. Rank experts by activation frequency
     * 2. For each device, compute "local hit rate"
     * 3. If a hot expert is on a slow device and a cold expert is on a
     *    fast device, propose a swap
     * 4. Limit movements per cycle to avoid thrashing
     */
    class ExpertRebalancer
    {
    public:
        explicit ExpertRebalancer(RebalanceParams params = {})
            : params_(params) {}

        /**
         * @brief Analyze placement and propose movements
         *
         * @param placement   Current expert placement (provides histogram)
         * @param hot_device  Preferred device for hot experts
         * @param cold_device Device for cold experts (e.g. CPU)
         * @return Proposal with movements (may be empty if no benefit)
         */
        RebalanceProposal propose(
            const ExpertPlacementMap &placement,
            DeviceId hot_device,
            DeviceId cold_device) const;

        /**
         * @brief Apply a proposal to the placement map
         *
         * @param placement Map to modify
         * @param proposal  Movements to apply
         * @return Number of movements actually applied
         */
        static int apply(ExpertPlacementMap &placement, const RebalanceProposal &proposal);

        /// Access params for testing
        const RebalanceParams &params() const { return params_; }

    private:
        RebalanceParams params_;
    };

} // namespace llaminar2
