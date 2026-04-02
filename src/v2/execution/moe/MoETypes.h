/**
 * @file MoETypes.h
 * @brief Core MoE value types for routing, dispatch, and expert execution
 *
 * These types carry the results of routing decisions through the MoE
 * pipeline: Router → Dispatch → Expert → Combine.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Result of routing one token to its top-k experts
     */
    struct TokenRouteEntry
    {
        int token_idx; ///< Index in the sequence
        int expert_id; ///< Which expert was selected
        float weight;  ///< Router-assigned weight for this expert
    };

    /**
     * @brief Routing table: the complete output of the router stage
     *
     * For a batch of T tokens with top-k routing, contains T*top_k entries.
     * Sorted by token_idx (all experts for token 0 first, then token 1, etc.)
     */
    struct RoutingTable
    {
        int seq_len = 0;
        int top_k = 0;
        std::vector<TokenRouteEntry> entries;

        /// Number of entries
        size_t size() const { return entries.size(); }

        /// Get the entries for a single token (contiguous block of top_k)
        const TokenRouteEntry *entriesForToken(int token_idx) const
        {
            size_t offset = static_cast<size_t>(token_idx) * top_k;
            if (offset + top_k > entries.size())
                return nullptr;
            return &entries[offset];
        }
    };

    /**
     * @brief Per-expert work assignment after dispatch
     *
     * Groups tokens by expert for efficient batched execution.
     */
    struct ExpertBatch
    {
        int expert_id = -1;
        std::vector<int> token_indices; ///< Indices into the sequence
        std::vector<float> weights;     ///< Corresponding routing weights

        int numTokens() const { return static_cast<int>(token_indices.size()); }
        bool empty() const { return token_indices.empty(); }
    };

    /**
     * @brief Complete dispatch plan: one ExpertBatch per active expert
     *
     * This is the output of the dispatch stage and drives expert execution.
     */
    struct DispatchPlan
    {
        int seq_len = 0;
        int d_model = 0;
        std::vector<ExpertBatch> batches; ///< One per active expert

        /// Number of experts that have at least one token
        int numActiveExperts() const
        {
            int count = 0;
            for (const auto &b : batches)
                if (!b.empty())
                    ++count;
            return count;
        }

        /// Find the batch for a specific expert (nullptr if expert has no tokens)
        const ExpertBatch *batchForExpert(int expert_id) const
        {
            for (const auto &b : batches)
                if (b.expert_id == expert_id)
                    return &b;
            return nullptr;
        }
    };

} // namespace llaminar2
