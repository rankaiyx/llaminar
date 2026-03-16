/**
 * @file FusedResidualNormStage.h
 * @brief Fused Residual Add + RMSNorm stage (GPU optimization)
 *
 * Replaces separate ResidualAddStage + RMSNormStage with a single GPU kernel
 * that computes: residual_out = input + residual; norm_out = RMSNorm(residual_out, gamma)
 * Saves one global memory write + read (2 * rows * cols * sizeof(float)).
 *
 * Falls back to sequential ResidualAdd + RMSNorm on CPU.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"

#include <optional>

namespace llaminar2
{

    class FusedResidualNormStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            /// Projection output to add (e.g. Wo output, down_proj output)
            const ITensor *input = nullptr;

            /// Residual stream (previous hidden state)
            ITensor *residual = nullptr;

            /// Gamma weights for RMSNorm
            const ITensor *gamma = nullptr;

            /// Normalized output (written by RMSNorm)
            ITensor *norm_output = nullptr;

            /// RMSNorm epsilon
            float eps = 1e-6f;

            /// Explicit seq_len (0 = derive from tensor)
            int seq_len = 0;

            /// Hidden dimension
            int hidden_dim = 0;

            // Buffer IDs for coherence
            std::optional<BufferId> input_buffer_id;
            std::optional<BufferId> residual_buffer_id;
            std::optional<BufferId> norm_output_buffer_id;
        };

        explicit FusedResidualNormStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::FUSED_RESIDUAL_NORM; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::CPU ||
                   backend == ComputeBackendType::GPU_CUDA ||
                   backend == ComputeBackendType::GPU_ROCM;
        }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferContract bufferContract() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
