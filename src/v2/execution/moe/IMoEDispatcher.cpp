/**
 * @file IMoEDispatcher.cpp
 * @brief Implementation of StandardMoEDispatcher
 */

#include "IMoEDispatcher.h"
#include <unordered_map>

namespace llaminar2
{

    DispatchPlan StandardMoEDispatcher::dispatch(
        const RoutingTable &table,
        int d_model)
    {
        DispatchPlan plan;
        plan.seq_len = table.seq_len;
        plan.d_model = d_model;

        // Group entries by expert_id
        std::unordered_map<int, size_t> expert_to_batch; // expert_id -> index in plan.batches

        for (const auto &entry : table.entries)
        {
            auto it = expert_to_batch.find(entry.expert_id);
            if (it == expert_to_batch.end())
            {
                // First token for this expert — create a new batch
                size_t idx = plan.batches.size();
                expert_to_batch[entry.expert_id] = idx;
                plan.batches.push_back(ExpertBatch{
                    .expert_id = entry.expert_id,
                    .token_indices = {entry.token_idx},
                    .weights = {entry.weight},
                });
            }
            else
            {
                // Append to existing batch
                auto &batch = plan.batches[it->second];
                batch.token_indices.push_back(entry.token_idx);
                batch.weights.push_back(entry.weight);
            }
        }

        return plan;
    }

} // namespace llaminar2
