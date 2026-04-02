/**
 * @file MoEPipeline.cpp
 * @brief MoE pipeline: route → dispatch → execute → combine
 */

#include "MoEPipeline.h"
#include <cmath>
#include <cstring>

namespace llaminar2
{

    MoEPipeline::MoEPipeline(
        Config config,
        std::shared_ptr<IMoERouter> router,
        std::shared_ptr<IMoEDispatcher> dispatcher,
        std::shared_ptr<IMoEExpertExecutor> expert_executor,
        std::shared_ptr<IMoECombiner> combiner,
        std::shared_ptr<IExpertWeightProvider> weight_provider,
        ExpertPlacementMap* placement)
        : config_(std::move(config))
        , router_(std::move(router))
        , dispatcher_(std::move(dispatcher))
        , expert_executor_(std::move(expert_executor))
        , combiner_(std::move(combiner))
        , weight_provider_(std::move(weight_provider))
        , placement_(placement)
    {
    }

    MoEPipelineResult MoEPipeline::forward(
        const float* hidden,
        const float* gate_logits,
        float* output,
        int layer_idx,
        int seq_len)
    {
        MoEPipelineResult result;

        // Step 1: Route tokens to experts
        result.routing = router_->route(
            gate_logits, seq_len, config_.num_experts, config_.top_k);

        // Step 2: Dispatch — group tokens by expert
        result.dispatch_plan = dispatcher_->dispatch(result.routing, config_.d_model);
        result.num_active_experts = result.dispatch_plan.numActiveExperts();
        result.total_token_expert_pairs = seq_len * config_.top_k;

        // Step 3: Allocate/resize scratch for per-expert outputs
        size_t scratch_size = static_cast<size_t>(seq_len) * config_.d_model;
        if (expert_output_scratch_.size() < scratch_size)
            expert_output_scratch_.resize(scratch_size, 0.0f);

        // Zero the final output — we accumulate weighted expert contributions
        std::memset(output, 0, scratch_size * sizeof(float));

        // Step 4: Execute each expert on its batch, accumulate weighted output
        // NOTE: Each expert gets a fresh scratch buffer. After execution, we
        // accumulate weight * expert_output into the final output. This is
        // necessary because multiple experts can process the same token, and
        // a single shared buffer would cause later experts to overwrite earlier
        // experts' results.
        for (const auto& batch : result.dispatch_plan.batches)
        {
            if (batch.empty())
                continue;

            // Track activation
            if (placement_)
                placement_->recordActivation(batch.expert_id);

            // Get weights
            ExpertWeights weights = weight_provider_->getWeights(layer_idx, batch.expert_id);
            if (!weights.gate_w || !weights.up_w || !weights.down_w)
            {
                result.success = false;
                return result;
            }

            // Zero scratch for this expert
            std::memset(expert_output_scratch_.data(), 0, scratch_size * sizeof(float));

            // Execute expert into scratch
            bool ok = expert_executor_->executeExpert(
                hidden,
                expert_output_scratch_.data(),
                weights.gate_w,
                weights.up_w,
                weights.down_w,
                batch,
                config_.d_model,
                config_.intermediate_size);

            if (!ok)
            {
                result.success = false;
                return result;
            }

            // Accumulate weighted expert output into final output
            for (int i = 0; i < batch.numTokens(); ++i)
            {
                int t = batch.token_indices[i];
                float w = batch.weights[i];
                float* dst = output + static_cast<size_t>(t) * config_.d_model;
                const float* src = expert_output_scratch_.data() +
                                   static_cast<size_t>(t) * config_.d_model;
                for (int d = 0; d < config_.d_model; ++d)
                    dst[d] += w * src[d];
            }
        }

        bool ok = true;

        result.success = ok;
        return result;
    }

    bool MoEPipeline::forwardSharedExpert(
        const float* hidden,
        const float* shared_gate_logit,
        float* output,
        int layer_idx,
        int seq_len)
    {
        if (!config_.has_shared_expert)
            return true; // No shared expert, nothing to do

        // Get shared expert weights (expert_id = -1 convention)
        ExpertWeights weights = weight_provider_->getWeights(layer_idx, -1);
        if (!weights.gate_w || !weights.up_w || !weights.down_w)
            return false;

        // Build a batch containing all tokens
        ExpertBatch all_tokens;
        all_tokens.expert_id = -1;
        for (int t = 0; t < seq_len; ++t)
        {
            all_tokens.token_indices.push_back(t);
            all_tokens.weights.push_back(1.0f);
        }

        // Use same scratch buffer
        size_t scratch_size = static_cast<size_t>(seq_len) * config_.d_model;
        if (expert_output_scratch_.size() < scratch_size)
            expert_output_scratch_.resize(scratch_size, 0.0f);

        std::vector<float> shared_output(scratch_size, 0.0f);

        bool ok = expert_executor_->executeExpert(
            hidden,
            shared_output.data(),
            weights.gate_w,
            weights.up_w,
            weights.down_w,
            all_tokens,
            config_.d_model,
            config_.shared_intermediate_size);

        if (!ok)
            return false;

        // Apply sigmoid gating if present, then add to output
        for (int t = 0; t < seq_len; ++t)
        {
            float gate = 1.0f;
            if (shared_gate_logit)
            {
                // Sigmoid gate
                float logit = shared_gate_logit[t];
                gate = 1.0f / (1.0f + std::exp(-logit));
            }

            float* out = output + static_cast<size_t>(t) * config_.d_model;
            const float* shared = shared_output.data() + static_cast<size_t>(t) * config_.d_model;

            for (int d = 0; d < config_.d_model; ++d)
                out[d] += gate * shared[d];
        }

        return true;
    }

} // namespace llaminar2
