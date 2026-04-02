/**
 * @file GatedRMSNormStage.cpp
 * @brief Implementation of gated RMS normalization stage
 */

#include "GatedRMSNormStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <cmath>

namespace llaminar2
{

    GatedRMSNormStage::GatedRMSNormStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool GatedRMSNormStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GatedRMSNormStage"))
            return false;

        if (!ensureRequiredPointers("GatedRMSNormStage",
                                    {{"input", params_.input},
                                     {"gate", params_.gate},
                                     {"output", params_.output},
                                     {"gamma", const_cast<ITensor *>(params_.gamma)}}))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *gate_base = requireTensorBasePtr(params_.gate, "gate");
        auto *output_base = requireTensorBasePtr(params_.output, "output");
        auto *gamma_base = dynamic_cast<const TensorBase *>(params_.gamma);

        if (!input_base || !gate_base || !output_base || !gamma_base)
        {
            LOG_ERROR("[GatedRMSNormStage] Tensor cast failed");
            return false;
        }

        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(input_base->rows());

        const size_t d_model = input_base->shape().size() > 1 ? input_base->shape()[1] : input_base->numel();

        const float *input_data = input_base->data();
        const float *gate_data = gate_base->data();
        const float *gamma_data = gamma_base->data();
        float *output_data = output_base->mutable_data();

        if (!input_data || !gate_data || !gamma_data || !output_data)
        {
            LOG_ERROR("[GatedRMSNormStage] Null data pointer");
            return false;
        }

        // For each token: RMSNorm(input) * gate
        for (int t = 0; t < seq_len; ++t)
        {
            const size_t offset = static_cast<size_t>(t) * d_model;

            // Compute RMS
            float sum_sq = 0.0f;
            for (size_t j = 0; j < d_model; ++j)
            {
                const float v = input_data[offset + j];
                sum_sq += v * v;
            }
            const float rms = std::sqrt(sum_sq / static_cast<float>(d_model) + params_.eps);
            const float inv_rms = 1.0f / rms;

            // Normalize, apply gamma (with optional subtract_one), multiply by gate
            for (size_t j = 0; j < d_model; ++j)
            {
                const float normalized = input_data[offset + j] * inv_rms;
                const float gamma_eff = params_.subtract_one
                                            ? (1.0f + gamma_data[j])
                                            : gamma_data[j];
                output_data[offset + j] = normalized * gamma_eff * gate_data[offset + j];
            }
        }

        LOG_DEBUG("[GatedRMSNormStage] Executed: seq_len=" << seq_len
                                                           << " d_model=" << d_model
                                                           << " subtract_one=" << params_.subtract_one);
        return true;
    }

    size_t GatedRMSNormStage::estimatedFlops() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // RMSNorm (~3 ops/element) + gate multiply (1 op/element)
        return input_base->numel() * 4;
    }

    size_t GatedRMSNormStage::estimatedMemoryBytes() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // Read input + gate + gamma, write output
        return input_base->numel() * 4 * sizeof(float);
    }

    bool GatedRMSNormStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo GatedRMSNormStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        auto *gate_base = dynamic_cast<TensorBase *>(params_.gate);
        auto *output_base = dynamic_cast<TensorBase *>(params_.output);
        auto *gamma_base = dynamic_cast<const TensorBase *>(params_.gamma);

        if (input_base)
            info.inputs.push_back({"input", input_base});
        if (gate_base)
            info.inputs.push_back({"gate", gate_base});
        if (gamma_base)
            info.weights.push_back({"gamma", const_cast<TensorBase *>(gamma_base)});
        if (output_base)
            info.outputs.push_back({"output", output_base});

        return info;
    }

    StageBufferRequirements GatedRMSNormStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GatedRMSNormStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.gate_buffer_id)
            contract.addInput(*params_.gate_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
