/**
 * @file GDNProjectionStage.cpp
 * @brief Implementation of GDN 4-projection stage
 */

#include "GDNProjectionStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    GDNProjectionStage::GDNProjectionStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool GDNProjectionStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "GDNProjectionStage"))
            return false;

        if (!ensureRequiredPointers("GDNProjectionStage",
                                    {{"input", params_.input},
                                     {"w_qkv", params_.w_qkv},
                                     {"output_qkv", params_.output_qkv},
                                     {"w_z", params_.w_z},
                                     {"output_z", params_.output_z},
                                     {"w_a", params_.w_a},
                                     {"output_a", params_.output_a},
                                     {"w_b", params_.w_b},
                                     {"output_b", params_.output_b}}))
            return false;

        const int M = params_.m;
        const int K = params_.k;
        auto *A_base = requireTensorBasePtr(params_.input, "input");
        if (!A_base)
            return false;

        // Execute each projection via its GEMM kernel
        // QKV projection
        if (params_.gemm_qkv)
        {
            auto *C_base = asTensorBase(params_.output_qkv, "output_qkv");
            if (!params_.gemm_qkv->multiply_tensor(
                    A_base, C_base, M, params_.n_qkv, K))
            {
                LOG_ERROR("[GDNProjectionStage] QKV GEMM failed");
                return false;
            }
        }
        else
        {
            LOG_ERROR("[GDNProjectionStage] gemm_qkv kernel not set");
            return false;
        }

        // Z projection
        if (params_.gemm_z)
        {
            auto *C_base = asTensorBase(params_.output_z, "output_z");
            if (!params_.gemm_z->multiply_tensor(
                    A_base, C_base, M, params_.n_z, K))
            {
                LOG_ERROR("[GDNProjectionStage] Z GEMM failed");
                return false;
            }
        }
        else
        {
            LOG_ERROR("[GDNProjectionStage] gemm_z kernel not set");
            return false;
        }

        // A projection
        if (params_.gemm_a)
        {
            auto *C_base = asTensorBase(params_.output_a, "output_a");
            if (!params_.gemm_a->multiply_tensor(
                    A_base, C_base, M, params_.n_a, K))
            {
                LOG_ERROR("[GDNProjectionStage] A GEMM failed");
                return false;
            }
        }
        else
        {
            LOG_ERROR("[GDNProjectionStage] gemm_a kernel not set");
            return false;
        }

        // B projection
        if (params_.gemm_b)
        {
            auto *C_base = asTensorBase(params_.output_b, "output_b");
            if (!params_.gemm_b->multiply_tensor(
                    A_base, C_base, M, params_.n_b, K))
            {
                LOG_ERROR("[GDNProjectionStage] B GEMM failed");
                return false;
            }
        }
        else
        {
            LOG_ERROR("[GDNProjectionStage] gemm_b kernel not set");
            return false;
        }

        LOG_DEBUG("[GDNProjectionStage] Executed: M=" << M << " K=" << K
                                                      << " n_qkv=" << params_.n_qkv
                                                      << " n_z=" << params_.n_z
                                                      << " n_a=" << params_.n_a
                                                      << " n_b=" << params_.n_b);
        return true;
    }

    size_t GDNProjectionStage::estimatedFlops() const
    {
        // Each GEMM: 2*M*N*K flops
        const size_t M = static_cast<size_t>(params_.m);
        const size_t K = static_cast<size_t>(params_.k);
        return 2 * M * K * (params_.n_qkv + params_.n_z + params_.n_a + params_.n_b);
    }

    size_t GDNProjectionStage::estimatedMemoryBytes() const
    {
        const size_t M = static_cast<size_t>(params_.m);
        // Read input once, read 4 weight matrices, write 4 outputs
        const size_t input_bytes = M * params_.k * sizeof(float);
        const size_t output_bytes = M * (params_.n_qkv + params_.n_z + params_.n_a + params_.n_b) * sizeof(float);
        return input_bytes + output_bytes;
    }

    bool GDNProjectionStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageDumpInfo GDNProjectionStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        auto add_tensor = [](auto &vec, const char *name, const ITensor *t)
        {
            auto *base = dynamic_cast<const TensorBase *>(t);
            if (base)
                vec.push_back({name, const_cast<TensorBase *>(base)});
        };

        add_tensor(info.inputs, "input", params_.input);
        add_tensor(info.inputs, "w_qkv", params_.w_qkv);
        add_tensor(info.inputs, "w_z", params_.w_z);
        add_tensor(info.inputs, "w_a", params_.w_a);
        add_tensor(info.inputs, "w_b", params_.w_b);
        add_tensor(info.outputs, "output_qkv", params_.output_qkv);
        add_tensor(info.outputs, "output_z", params_.output_z);
        add_tensor(info.outputs, "output_a", params_.output_a);
        add_tensor(info.outputs, "output_b", params_.output_b);

        return info;
    }

    StageBufferRequirements GDNProjectionStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract GDNProjectionStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.output_qkv_buffer_id)
            contract.addOutput(*params_.output_qkv_buffer_id);
        if (params_.output_z_buffer_id)
            contract.addOutput(*params_.output_z_buffer_id);
        if (params_.output_a_buffer_id)
            contract.addOutput(*params_.output_a_buffer_id);
        if (params_.output_b_buffer_id)
            contract.addOutput(*params_.output_b_buffer_id);
        return contract;
    }

} // namespace llaminar2
