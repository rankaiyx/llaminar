/**
 * @file ExpertRebalancer.cpp
 * @brief Histogram-based expert rebalancing implementation
 */

#include "ExpertRebalancer.h"
#include <algorithm>
#include <numeric>

namespace llaminar2
{

    RebalanceProposal ExpertRebalancer::propose(
        const ExpertPlacementMap &placement,
        DeviceId hot_device,
        DeviceId cold_device) const
    {
        RebalanceProposal proposal;
        proposal.expected_locality_improvement = 0.0;

        auto histogram = placement.activationHistogram();
        const int num_experts = placement.numExperts();

        // Check minimum total activations
        uint64_t total_activations = std::accumulate(histogram.begin(), histogram.end(), uint64_t{0});
        if (total_activations < params_.min_total_activations)
            return proposal;

        // Build sorted list of (expert_id, activation_count)
        struct ExpertFreq
        {
            int expert_id;
            uint64_t count;
        };
        std::vector<ExpertFreq> sorted_experts(num_experts);
        for (int e = 0; e < num_experts; ++e)
            sorted_experts[e] = {e, histogram[e]};

        std::sort(sorted_experts.begin(), sorted_experts.end(),
                  [](const ExpertFreq &a, const ExpertFreq &b)
                  { return a.count > b.count; });

        // Find hot experts on cold_device → move to hot_device
        // Find cold experts on hot_device → move to cold_device
        int moves = 0;

        for (const auto &ef : sorted_experts)
        {
            if (moves >= params_.max_moves_per_cycle)
                break;

            DeviceId current = placement.deviceForExpert(ef.expert_id);

            if (ef.count > 0 && current == cold_device)
            {
                // Hot expert on cold device → move to hot device
                float avg = static_cast<float>(total_activations) / num_experts;
                if (static_cast<float>(ef.count) > avg * params_.activation_ratio_threshold)
                {
                    proposal.movements.push_back(ExpertMovement{
                        .expert_id = ef.expert_id,
                        .from_device = cold_device,
                        .to_device = hot_device,
                        .activation_count = ef.count,
                    });
                    moves++;
                }
            }
        }

        // Also find cold experts on hot_device that could be offloaded
        for (auto it = sorted_experts.rbegin(); it != sorted_experts.rend(); ++it)
        {
            if (moves >= params_.max_moves_per_cycle)
                break;

            DeviceId current = placement.deviceForExpert(it->expert_id);

            if (current == hot_device)
            {
                float avg = static_cast<float>(total_activations) / num_experts;
                if (static_cast<float>(it->count) < avg / params_.activation_ratio_threshold)
                {
                    proposal.movements.push_back(ExpertMovement{
                        .expert_id = it->expert_id,
                        .from_device = hot_device,
                        .to_device = cold_device,
                        .activation_count = it->count,
                    });
                    moves++;
                }
            }
        }

        // Estimate locality improvement
        if (!proposal.movements.empty() && total_activations > 0)
        {
            uint64_t moved_activations = 0;
            for (const auto &m : proposal.movements)
            {
                if (m.to_device == hot_device)
                    moved_activations += m.activation_count;
            }
            proposal.expected_locality_improvement =
                static_cast<double>(moved_activations) / total_activations * 100.0;
        }

        return proposal;
    }

    int ExpertRebalancer::apply(ExpertPlacementMap &placement, const RebalanceProposal &proposal)
    {
        int applied = 0;
        for (const auto &movement : proposal.movements)
        {
            placement.moveExpert(movement.expert_id, movement.to_device);
            applied++;
        }
        return applied;
    }

} // namespace llaminar2
