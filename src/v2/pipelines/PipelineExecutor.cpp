/**
 * @file PipelineExecutor.cpp
 * @brief Implementation of PipelineExecutor adapter
 * @author David Sanftenberg
 * @date December 2025
 */

#include "PipelineExecutor.h"
#include "../execution/DeviceContext.h"
#include "../execution/ComputeStage.h"
#include "../utils/Logger.h"
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // Construction
    // =============================================================================

    PipelineExecutor::PipelineExecutor(const PipelineExecutorConfig &config,
                                       std::shared_ptr<MPIContext> mpi_ctx)
        : config_(config), mpi_ctx_(mpi_ctx), mpi_ctx_raw_(mpi_ctx.get()) // Store raw pointer for operations
    {
        initializeLayerExecutor();
    }

    PipelineExecutor::PipelineExecutor(const PipelineExecutorConfig &config,
                                       MPIContext *mpi_ctx)
        : config_(config), mpi_ctx_(nullptr) // No shared ownership
          ,
          mpi_ctx_raw_(mpi_ctx) // Store raw pointer for operations
    {
        initializeLayerExecutor();
    }

    void PipelineExecutor::initializeLayerExecutor()
    {
        // Create LayerExecutor with matching config
        LayerExecutorConfig exec_config;
        exec_config.mode = config_.execution_mode;
        exec_config.enable_profiling = config_.enable_profiling;
        exec_config.default_device = 0; // Updated when setDeviceContext called

        layer_executor_ = std::make_unique<LayerExecutor>(exec_config);

        LOG_DEBUG("PipelineExecutor created with execution_mode="
                  << static_cast<int>(config_.execution_mode)
                  << ", profiling=" << config_.enable_profiling);
    }

    PipelineExecutor::~PipelineExecutor() = default;

    // =============================================================================
    // Configuration
    // =============================================================================

    void PipelineExecutor::setDeviceContext(int device_idx, int num_threads)
    {
        if (device_contexts_.find(device_idx) == device_contexts_.end())
        {
            // Create CPU context (GPU contexts added later)
            int threads = num_threads > 0 ? num_threads : 4;
            device_contexts_[device_idx] = std::make_unique<CPUDeviceContext>(device_idx, threads);

            LOG_DEBUG("Created CPU device context for device " << device_idx
                                                               << " with " << threads << " threads");
        }
    }

    IDeviceContext *PipelineExecutor::getDeviceContext(int device_idx) const
    {
        auto it = device_contexts_.find(device_idx);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }
        return nullptr;
    }

    IDeviceContext *PipelineExecutor::ensureContext(int device_idx)
    {
        if (device_contexts_.find(device_idx) == device_contexts_.end())
        {
            setDeviceContext(device_idx, 4); // Default threads
        }
        return device_contexts_[device_idx].get();
    }

    // =============================================================================
    // Individual Operation Execution
    // =============================================================================

    bool PipelineExecutor::executeRMSNorm(TensorBase *input,
                                          const TensorBase *gamma,
                                          TensorBase *output,
                                          int seq_len,
                                          int hidden_dim,
                                          float eps,
                                          int device_idx)
    {
        if (!config_.use_layer_executor)
        {
            LOG_WARN("executeRMSNorm called but use_layer_executor is false");
            return false;
        }

        auto *ctx = ensureContext(device_idx);
        if (!ctx)
        {
            LOG_ERROR("Failed to get device context for device " << device_idx);
            return false;
        }

        // Create RMSNorm stage params - pass TensorBase* directly
        RMSNormStage::Params params;
        params.input = input;  // TensorBase* - modifies in-place
        params.output = input; // TensorBase* - same as input for in-place
        params.gamma = gamma;  // TensorBase*
        params.eps = eps;

        // Create and execute stage
        auto stage = ComputeStageFactory::createRMSNorm(params);
        bool success = stage->execute(ctx);

        if (!success)
        {
            LOG_ERROR("RMSNorm execution failed on device " << device_idx);
        }

        return success;
    }

    bool PipelineExecutor::executeSwiGLU(TensorBase *gate,
                                         TensorBase *up,
                                         TensorBase *output,
                                         int seq_len,
                                         int intermediate_dim,
                                         int device_idx)
    {
        if (!config_.use_layer_executor)
        {
            LOG_WARN("executeSwiGLU called but use_layer_executor is false");
            return false;
        }

        auto *ctx = ensureContext(device_idx);
        if (!ctx)
        {
            LOG_ERROR("Failed to get device context for device " << device_idx);
            return false;
        }

        // Create SwiGLU stage params - pass TensorBase* directly
        SwiGLUStage::Params params;
        params.gate = gate;     // TensorBase*
        params.up = up;         // TensorBase*
        params.output = output; // TensorBase*

        // Create and execute stage
        auto stage = ComputeStageFactory::createSwiGLU(params);
        bool success = stage->execute(ctx);

        if (!success)
        {
            LOG_ERROR("SwiGLU execution failed on device " << device_idx);
        }

        return success;
    }

    bool PipelineExecutor::executeResidualAdd(const TensorBase *input,
                                              const TensorBase *residual,
                                              TensorBase *output,
                                              size_t num_elements,
                                              int device_idx)
    {
        if (!config_.use_layer_executor)
        {
            LOG_WARN("executeResidualAdd called but use_layer_executor is false");
            return false;
        }

        auto *ctx = ensureContext(device_idx);
        if (!ctx)
        {
            LOG_ERROR("Failed to get device context for device " << device_idx);
            return false;
        }

        // Create residual add stage params - pass TensorBase* directly
        ResidualAddStage::Params params;
        params.input = input;       // const TensorBase*
        params.residual = residual; // const TensorBase*
        params.output = output;     // TensorBase*

        // Create and execute stage
        auto stage = ComputeStageFactory::createResidualAdd(params);
        bool success = stage->execute(ctx);

        if (!success)
        {
            LOG_ERROR("ResidualAdd execution failed on device " << device_idx);
        }

        return success;
    }

    bool PipelineExecutor::executeRoPE(TensorBase *Q,
                                       TensorBase *K,
                                       const int *position_ids,
                                       int seq_len,
                                       int n_heads,
                                       int n_kv_heads,
                                       int head_dim,
                                       float theta,
                                       int device_idx)
    {
        if (!config_.use_layer_executor)
        {
            LOG_WARN("executeRoPE called but use_layer_executor is false");
            return false;
        }

        auto *ctx = ensureContext(device_idx);
        if (!ctx)
        {
            LOG_ERROR("Failed to get device context for device " << device_idx);
            return false;
        }

        // RoPE is applied to Q and K separately (or together if using TensorBase* mode)
        // Use TensorBase* mode - pass Q and optionally K
        RoPEStage::Params q_params;
        q_params.Q = Q; // TensorBase*
        q_params.K = K; // TensorBase* (optional, can process both at once)
        q_params.n_heads = n_heads;
        q_params.n_kv_heads = n_kv_heads;
        q_params.head_dim = head_dim;
        q_params.pos_offset = 0; // TODO: pass position_ids properly
        q_params.theta_base = theta;

        auto rope_stage = ComputeStageFactory::createRoPE(q_params);
        if (!rope_stage->execute(ctx))
        {
            LOG_ERROR("RoPE execution failed on device " << device_idx);
            return false;
        }

        return true;
    }

    // =============================================================================
    // Statistics
    // =============================================================================

    const LayerExecutorStats &PipelineExecutor::stats() const
    {
        return layer_executor_->stats();
    }

    void PipelineExecutor::resetStats()
    {
        layer_executor_->resetStats();
    }

} // namespace llaminar2
