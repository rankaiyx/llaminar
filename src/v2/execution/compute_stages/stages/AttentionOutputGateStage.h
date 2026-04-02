/**
 * @file AttentionOutputGateStage.h
 * @brief Sigmoid-gated attention output stage for GDN layers
 *
 * Computes: output = sigmoid(gate) * input
 *
 * Used by models with gated attention output (e.g., Qwen 3.5 GDN layers)
 * where the attention output is scaled by a learned sigmoid gate.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    /**
     * @brief Sigmoid-gated attention output
     *
     * Computes element-wise: output[i] = sigmoid(gate[i]) * input[i]
     * where gate is a learned projection (GEMM output) and input is
     * the attention output before residual addition.
     */
    class AttentionOutputGateStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *input = nullptr;  ///< Attention output [seq_len, d_model]
            ITensor *gate = nullptr;   ///< Gate projection output [seq_len, d_model]
            ITensor *output = nullptr; ///< Gated output [seq_len, d_model] (can alias input)

            int seq_len = 0; ///< Explicit sequence length

            // Optional BufferIds for contract-based coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> gate_buffer_id;
            std::optional<BufferId> output_buffer_id;
        };

        static_assert(StageParamsRequired<Params>);

        explicit AttentionOutputGateStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION_OUTPUT_GATE; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;

        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
