/**
 * @file MoEPipeline.h
 * @brief Complete MoE FFN pipeline: route → dispatch → execute → combine
 *
 * Wires together Router, Dispatcher, ExpertExecutor, and Combiner through
 * injectable interfaces. Each component can be mocked independently for
 * unit testing.
 *
 * Also integrates:
 * - Shared expert execution (always-active dense FFN)
 * - Expert activation tracking (feeds ExpertRebalancer)
 * - Expert weight caching (for offloaded experts)
 */

#pragma once

#include "IMoERouter.h"
#include "IMoEDispatcher.h"
#include "IMoEExpertExecutor.h"
#include "IMoECombiner.h"
#include "MoEConfig.h"
#include "MoETypes.h"
#include "ExpertPlacementMap.h"
#include "ExpertWeightCache.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Weights for a single expert (pointers, not owned)
     */
    struct ExpertWeights
    {
        const float *gate_w = nullptr; ///< [intermediate, d_model]
        const float *up_w = nullptr;   ///< [intermediate, d_model]
        const float *down_w = nullptr; ///< [d_model, intermediate]
    };

    /**
     * @brief Provider of expert weights by (layer, expert_id)
     *
     * Mockable interface. In production, delegates to WeightManager or
     * ExpertWeightCache. In tests, can return synthetic weights.
     */
    class IExpertWeightProvider
    {
    public:
        virtual ~IExpertWeightProvider() = default;

        /// Get weights for a specific expert. Returns {nullptr,nullptr,nullptr} on failure.
        virtual ExpertWeights getWeights(int layer_idx, int expert_id) = 0;
    };

    /**
     * @brief Result of running the MoE pipeline for one layer
     */
    struct MoEPipelineResult
    {
        bool success = false;
        int num_active_experts = 0;
        int total_token_expert_pairs = 0; ///< seq_len * top_k
        RoutingTable routing;             ///< For diagnostics/logging
        DispatchPlan dispatch_plan;       ///< For diagnostics/logging
    };

    /**
     * @brief Complete MoE FFN pipeline with all components injectable
     *
     * Lifecycle:
     * 1. Construct with all interfaces
     * 2. Call forward() for each MoE layer
     *
     * Thread safety: forward() is NOT reentrant (shares scratch buffers).
     * Different layers must be called sequentially.
     */
    class MoEPipeline
    {
    public:
        struct Config
        {
            int num_experts = 0;
            int top_k = 0;
            int d_model = 0;
            int intermediate_size = 0;
            bool has_shared_expert = false;
            int shared_intermediate_size = 0;
        };

        MoEPipeline(
            Config config,
            std::shared_ptr<IMoERouter> router,
            std::shared_ptr<IMoEDispatcher> dispatcher,
            std::shared_ptr<IMoEExpertExecutor> expert_executor,
            std::shared_ptr<IMoECombiner> combiner,
            std::shared_ptr<IExpertWeightProvider> weight_provider,
            ExpertPlacementMap *placement = nullptr);

        /**
         * @brief Execute complete MoE FFN for one layer
         *
         * @param hidden       Input hidden states [seq_len, d_model], row-major
         * @param gate_logits  Pre-computed gate logits [seq_len, num_experts] (from router GEMM)
         * @param output       Output buffer [seq_len, d_model]
         * @param layer_idx    Which model layer (for weight lookup)
         * @param seq_len      Number of tokens
         * @return MoEPipelineResult with diagnostics
         */
        MoEPipelineResult forward(
            const float *hidden,
            const float *gate_logits,
            float *output,
            int layer_idx,
            int seq_len);

        /**
         * @brief Execute shared expert FFN (always-active dense expert)
         *
         * @param hidden       Input hidden states [seq_len, d_model]
         * @param shared_gate_logit  Sigmoid gating scalar (or null for no gating)
         * @param output       Output to ADD into [seq_len, d_model]
         * @param layer_idx    Which model layer
         * @param seq_len      Number of tokens
         * @return true on success
         */
        bool forwardSharedExpert(
            const float *hidden,
            const float *shared_gate_logit,
            float *output,
            int layer_idx,
            int seq_len);

        /// Access config for testing
        const Config &config() const { return config_; }

    private:
        Config config_;
        std::shared_ptr<IMoERouter> router_;
        std::shared_ptr<IMoEDispatcher> dispatcher_;
        std::shared_ptr<IMoEExpertExecutor> expert_executor_;
        std::shared_ptr<IMoECombiner> combiner_;
        std::shared_ptr<IExpertWeightProvider> weight_provider_;
        ExpertPlacementMap *placement_; // nullable — for activation tracking

        // Scratch buffer for expert outputs [seq_len, d_model]
        std::vector<float> expert_output_scratch_;
    };

} // namespace llaminar2
