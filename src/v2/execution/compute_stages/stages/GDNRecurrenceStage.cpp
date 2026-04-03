/**
 * @file GDNRecurrenceStage.cpp
 * @brief Implementation of delta rule recurrence for GDN linear attention
 *
 * Pure glue: extracts raw pointers from tensors, validates them, and
 * delegates all computation to the ITensorGatedDeltaNet kernel.
 *
 * All preprocessing (L2 normalization, query scaling, gate computation)
 * is handled by the kernel, keeping this stage device-agnostic.
 */

#include "GDNRecurrenceStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    GDNRecurrenceStage::GDNRecurrenceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    // =========================================================================
    // Main execute
    // =========================================================================

    bool GDNRecurrenceStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GDNRecurrenceStage"))
            return false;

        if (!ensureRequiredPointers("GDNRecurrenceStage",
                                    {{"Q", params_.Q},
                                     {"K", params_.K},
                                     {"V", params_.V},
                                     {"alpha", params_.alpha},
                                     {"beta", params_.beta},
                                     {"A_log", params_.A_log},
                                     {"dt_bias", params_.dt_bias},
                                     {"output", params_.output}}))
            return false;

        if (!params_.kernel)
        {
            LOG_ERROR("[GDNRecurrenceStage] kernel (ITensorGatedDeltaNet) not set");
            return false;
        }

        auto *q_base = requireTensorBasePtr(params_.Q, "Q");
        auto *k_base = requireTensorBasePtr(params_.K, "K");
        auto *v_base = requireTensorBasePtr(params_.V, "V");
        auto *alpha_base = requireTensorBasePtr(params_.alpha, "alpha");
        auto *beta_base = requireTensorBasePtr(params_.beta, "beta");
        auto *alog_base = requireTensorBasePtr(params_.A_log, "A_log");
        auto *dtbias_base = requireTensorBasePtr(params_.dt_bias, "dt_bias");
        auto *out_base = requireTensorBasePtr(params_.output, "output");

        if (!q_base || !k_base || !v_base || !alpha_base || !beta_base ||
            !alog_base || !dtbias_base || !out_base)
            return false;

        if (!params_.recurrence_state)
        {
            LOG_ERROR("[GDNRecurrenceStage] recurrence_state is null");
            return false;
        }

        // Extract raw pointers — all computation delegated to kernel
        const float *q_data = q_base->data();
        const float *k_data = k_base->data();
        const float *v_data = v_base->data();
        const float *alpha_data = alpha_base->data();
        const float *beta_data = beta_base->data();
        const float *alog_data = alog_base->data();
        const float *dtbias_data = dtbias_base->data();
        float *output_data = out_base->mutable_data();

        bool ok;
        if (params_.seq_len == 1)
        {
            ok = params_.kernel->recurrent_step(
                q_data, k_data, v_data,
                alpha_data, beta_data,
                alog_data, dtbias_data,
                output_data, params_.recurrence_state,
                params_.n_heads, params_.d_k, params_.d_v,
                params_.use_qk_l2norm);
        }
        else
        {
            ok = params_.kernel->chunk_forward(
                q_data, k_data, v_data,
                alpha_data, beta_data,
                alog_data, dtbias_data,
                output_data, params_.recurrence_state,
                params_.seq_len, params_.n_heads, params_.d_k, params_.d_v,
                params_.chunk_size, params_.use_qk_l2norm);
        }

        if (!ok)
        {
            LOG_ERROR("[GDNRecurrenceStage] Kernel failed");
            return false;
        }

        LOG_DEBUG("[GDNRecurrenceStage] layer=" << params_.layer_idx
                                                << " seq_len=" << params_.seq_len
                                                << " n_heads=" << params_.n_heads
                                                << " d_k=" << params_.d_k
                                                << " d_v=" << params_.d_v
                                                << (params_.seq_len == 1 ? " (decode)" : " (prefill)"));
        return ok;
    }

    // =========================================================================
    // Estimation and metadata
    // =========================================================================

    size_t GDNRecurrenceStage::estimatedFlops() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t H = static_cast<size_t>(params_.n_heads);
        const size_t dk = static_cast<size_t>(params_.d_k);
        const size_t dv = static_cast<size_t>(params_.d_v);
        // Per timestep per head: decay(dk*dv) + kv_mem(dk*dv) + delta(dv) + rank1(dk*dv) + output(dk*dv)
        // ≈ 4*dk*dv + dv per step per head
        return S * H * (4 * dk * dv + dv);
    }

    size_t GDNRecurrenceStage::estimatedMemoryBytes() const
    {
        const size_t S = static_cast<size_t>(params_.seq_len);
        const size_t H = static_cast<size_t>(params_.n_heads);
        const size_t dk = static_cast<size_t>(params_.d_k);
        const size_t dv = static_cast<size_t>(params_.d_v);
        // State: H*dk*dv, Q: S*H*dk, K: S*H*dk, V: S*H*dv, output: S*H*dv
        return (H * dk * dv + S * H * (2 * dk + 2 * dv)) * sizeof(float);
    }

    bool GDNRecurrenceStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo GDNRecurrenceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        auto add = [](auto &vec, const char *name, const ITensor *t)
        {
            auto *base = dynamic_cast<const TensorBase *>(t);
            if (base)
                vec.push_back({name, const_cast<TensorBase *>(base)});
        };

        add(info.inputs, "Q", params_.Q);
        add(info.inputs, "K", params_.K);
        add(info.inputs, "V", params_.V);
        add(info.inputs, "alpha", params_.alpha);
        add(info.inputs, "beta", params_.beta);
        add(info.inputs, "A_log", params_.A_log);
        add(info.inputs, "dt_bias", params_.dt_bias);
        add(info.outputs, "output", params_.output);

        return info;
    }

    StageBufferRequirements GDNRecurrenceStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNRecurrenceStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
