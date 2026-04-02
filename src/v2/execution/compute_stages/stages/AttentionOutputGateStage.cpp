/**
 * @file AttentionOutputGateStage.cpp
 * @brief Implementation of sigmoid-gated attention output stage
 */

#include "AttentionOutputGateStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <cmath>

namespace llaminar2
{

    AttentionOutputGateStage::AttentionOutputGateStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool AttentionOutputGateStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "AttentionOutputGateStage"))
            return false;

        if (!ensureRequiredPointers("AttentionOutputGateStage",
                                    {{"input", params_.input},
                                     {"gate", params_.gate},
                                     {"output", params_.output}}))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *gate_base = requireTensorBasePtr(params_.gate, "gate");
        auto *output_base = requireTensorBasePtr(params_.output, "output");

        if (!input_base || !gate_base || !output_base)
        {
            LOG_ERROR("[AttentionOutputGateStage] Tensor cast failed");
            return false;
        }

        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(input_base->rows());

        const size_t cols = input_base->shape().size() > 1 ? input_base->shape()[1] : input_base->numel();
        const size_t total = static_cast<size_t>(seq_len) * cols;

        const float *input_data = input_base->data();
        const float *gate_data = gate_base->data();
        float *output_data = output_base->mutable_data();

        if (!input_data || !gate_data || !output_data)
        {
            LOG_ERROR("[AttentionOutputGateStage] Null data pointer");
            return false;
        }

        // output[i] = sigmoid(gate[i]) * input[i]
        for (size_t i = 0; i < total; ++i)
        {
            const float sig = 1.0f / (1.0f + std::exp(-gate_data[i]));
            output_data[i] = sig * input_data[i];
        }

        LOG_DEBUG("[AttentionOutputGateStage] Executed: seq_len=" << seq_len
                                                                  << " cols=" << cols
                                                                  << " total=" << total);
        return true;
    }

    size_t AttentionOutputGateStage::estimatedFlops() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // sigmoid (4 ops) + multiply (1 op) per element
        return input_base->numel() * 5;
    }

    size_t AttentionOutputGateStage::estimatedMemoryBytes() const
    {
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        if (!input_base)
            return 0;
        // Read input + gate, write output (each float)
        return input_base->numel() * 3 * sizeof(float);
    }

    bool AttentionOutputGateStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo AttentionOutputGateStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        auto *input_base = dynamic_cast<TensorBase *>(params_.input);
        auto *gate_base = dynamic_cast<TensorBase *>(params_.gate);
        auto *output_base = dynamic_cast<TensorBase *>(params_.output);

        if (input_base)
            info.inputs.push_back({"input", input_base});
        if (gate_base)
            info.inputs.push_back({"gate", gate_base});
        if (output_base)
            info.outputs.push_back({"output", output_base});

        return info;
    }

    StageBufferRequirements AttentionOutputGateStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract AttentionOutputGateStage::bufferContract() const
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
