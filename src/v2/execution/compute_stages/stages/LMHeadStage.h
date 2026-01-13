/**
 * @file LMHeadStage.h
 * @brief Language model head projection stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

namespace llaminar2
{

    /**
     * @brief Language model head projection stage
     *
     * Projects hidden states to vocabulary logits for token prediction.
     * Typically the final stage in a forward pass before sampling.
     *
     * This stage wraps a GEMM operation but provides semantic clarity
     * in compute graphs and enables LM head-specific optimizations.
     */
    class LMHeadStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input/output tensors
            const ITensor *hidden_states = nullptr;
            const ITensor *lm_head_weight = nullptr;
            ITensor *logits = nullptr;

            // Dimensions
            int seq_len = 0;
            int d_model = 0;
            int vocab_size = 0;

            // Optional bias
            const float *bias = nullptr;
        };

        explicit LMHeadStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::LM_HEAD; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
