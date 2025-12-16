/**
 * @file Qwen2LayerExecutor.cpp
 * @brief Implementation of Qwen2-specific layer executor
 * @author David Sanftenberg
 * @date December 2025
 */

#include "Qwen2LayerExecutor.h"
#include "Qwen2Pipeline.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../backends/ComputeBackend.h"
#include "../../tensors/TensorSlice.h"
#include "../../utils/MPIContext.h"
#include "../MPIStrategy.h"

namespace llaminar2
{

    // =============================================================================
    // Helper Functions
    // =============================================================================

    /**
     * @brief Check if a weight tensor is sharded row-parallel
     *
     * Row-parallel weights need allreduce after GEMM to combine partial results
     * from all MPI ranks.
     *
     * @param weight Weight tensor (may be TensorSlice or raw tensor)
     * @return true if weight is row-parallel sharded
     */
    static bool isRowParallelSharded(const TensorBase *weight)
    {
        if (!weight)
            return false;

        const auto *slice = dynamic_cast<const TensorSlice *>(weight);
        bool result = slice && slice->is_row_parallel();
        LOG_DEBUG("[isRowParallelSharded] weight=" << weight << " is_slice=" << (slice != nullptr)
                                                   << " is_row_parallel=" << (slice ? slice->is_row_parallel() : false)
                                                   << " result=" << result);
        return result;
    }

    /**
     * @brief Get MPI communicator as void* for AllreduceStage
     */
    static void *getMPIComm(const MPIContext *mpi_ctx)
    {
        if (!mpi_ctx)
        {
            LOG_WARN("[getMPIComm] mpi_ctx is null!");
            return nullptr;
        }
        MPI_Comm comm = mpi_ctx->comm();
        void *result = static_cast<void *>(comm);
        LOG_DEBUG("[getMPIComm] mpi_ctx=" << mpi_ctx << " comm=" << comm << " result=" << result);
        return result;
    }

    // =============================================================================
    // Constructor
    // =============================================================================

    Qwen2LayerExecutor::Qwen2LayerExecutor(const Qwen2ExecutorConfig &config,
                                           std::shared_ptr<MPIContext> mpi_ctx)
        : config_(config), mpi_ctx_(std::move(mpi_ctx))
    {
        LOG_INFO("[Qwen2LayerExecutor] config: d_model=" << config_.d_model
                                                         << " d_ff=" << config_.d_ff
                                                         << " ffn_column_parallel=" << config_.ffn_column_parallel
                                                         << " n_heads=" << config_.n_heads
                                                         << " n_kv_heads=" << config_.n_kv_heads);

        // Configure underlying LayerExecutor
        LayerExecutorConfig exec_config;
        exec_config.default_device = config.default_device;
        exec_config.enable_profiling = config.enable_profiling;
        exec_config.enable_validation = config.enable_validation;

        // Parse execution mode from environment
        const auto &env = debugEnv();
        if (env.execution.execution_mode == "parallel")
        {
            exec_config.mode = ExecutionMode::PARALLEL;
        }
        else if (env.execution.execution_mode == "pipelined")
        {
            exec_config.mode = ExecutionMode::PIPELINED;
        }
        else
        {
            exec_config.mode = ExecutionMode::SEQUENTIAL;
        }

        executor_ = LayerExecutor(exec_config);

        LOG_DEBUG("[Qwen2LayerExecutor] Initialized with d_model=" << config_.d_model
                                                                   << ", n_heads=" << config_.n_heads
                                                                   << ", mode=" << executionModeName(exec_config.mode));
    }

    // =============================================================================
    // Device Context Management
    // =============================================================================

    IDeviceContext *Qwen2LayerExecutor::getDeviceContext(int device_idx)
    {
        auto it = device_contexts_.find(device_idx);
        if (it != device_contexts_.end())
        {
            return it->second.get();
        }

        // Create new context
        auto ctx = IDeviceContext::create(device_idx);
        if (!ctx)
        {
            LOG_ERROR("[Qwen2LayerExecutor] Failed to create device context for device " << device_idx);
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device_idx] = std::move(ctx);
        return raw_ptr;
    }

    // =============================================================================
    // Attention Block Execution
    // =============================================================================

    bool Qwen2LayerExecutor::executeAttention(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        const auto &env = debugEnv();

        // Debug: dump input to attention (for layer 0 only to reduce noise)
        if (layer_idx == 0)
        {
            const float *input = buffers.current_hidden->data();
            LOG_INFO("[EXEC_ATTN_INPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                << " input[0:4]=" << input[0] << "," << input[1] << "," << input[2] << "," << input[3]);
        }

        // Build compute graph for this attention block
        ComputeGraph graph = buildAttentionGraph(layer, buffers, layer_idx, seq_len,
                                                 kv_cache, position_ids, device_idx);

        // Get or create device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        // Execute graph
        bool success = executor_.execute(graph, ctx);

        // Debug: dump intermediate buffers (for layer 0 only)
        if (layer_idx == 0)
        {
            const float *normalized = buffers.normalized->data();
            const float *Q = buffers.Q->data();
            const float *attn_output = buffers.attn_output->data();
            const float *attn_proj = buffers.attn_proj->data();
            const float *output = buffers.current_hidden->data();
            LOG_INFO("[EXEC_ATTN] normalized ptr=" << static_cast<const void *>(normalized)
                                                   << " Q ptr=" << static_cast<const void *>(Q));
            LOG_INFO("[EXEC_ATTN] normalized[0:4]=" << std::setprecision(10) << normalized[0] << "," << normalized[1] << "," << normalized[2] << "," << normalized[3]);
            LOG_INFO("[EXEC_ATTN] Q[0:4]=" << std::setprecision(10) << Q[0] << "," << Q[1] << "," << Q[2] << "," << Q[3]);
            LOG_INFO("[EXEC_ATTN] attn_output[0:4]=" << attn_output[0] << "," << attn_output[1] << "," << attn_output[2] << "," << attn_output[3]);
            LOG_INFO("[EXEC_ATTN] attn_proj[0:4]=" << attn_proj[0] << "," << attn_proj[1] << "," << attn_proj[2] << "," << attn_proj[3]);
            LOG_INFO("[EXEC_ATTN_OUTPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                 << " output[0:4]=" << output[0] << "," << output[1] << "," << output[2] << "," << output[3]);
        }

        if (!success)
        {
            LOG_ERROR("[Qwen2LayerExecutor] Attention block failed at layer " << layer_idx);
        }

        return success;
    }

    ComputeGraph Qwen2LayerExecutor::buildAttentionGraph(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        LOG_DEBUG("[buildAttentionGraph] layer_idx=" << layer_idx << " seq_len=" << seq_len
                                                     << " layer.wo=" << layer.wo << " world_size="
                                                     << (mpi_ctx_ ? mpi_ctx_->world_size() : 1));

        // Determine backend type for stage creation
        auto &dm = DeviceManager::instance();
        ComputeBackendType backend = ComputeBackendType::CPU_OPENBLAS;
        if (static_cast<size_t>(device_idx) < dm.devices().size())
        {
            backend = dm.devices()[device_idx].type;
        }

        // Stage 1: Pre-attention RMSNorm
        if (env.execution.exec_rmsnorm)
        {
            RMSNormStage::Params attn_norm_params;
            attn_norm_params.input = buffers.current_hidden->data();      // Read from current_hidden (no copy needed)
            attn_norm_params.output = buffers.normalized->mutable_data(); // Write to normalized
            attn_norm_params.gamma = nullptr;
            if (layer.attn_norm != nullptr)
            {
                attn_norm_params.gamma = layer.attn_norm->data();
            }
            attn_norm_params.seq_len = seq_len;
            attn_norm_params.hidden_dim = config_.d_model;
            attn_norm_params.eps = config_.rms_norm_eps;

            graph.addNode(prefix + "attn_norm",
                          ComputeStageFactory::createRMSNorm(attn_norm_params, backend),
                          device_idx);
        }

        // Stage 2: Q/K/V projections using pre-quantized activations (like FusedGEMM)
        if (env.execution.exec_gemm)
        {
            // Calculate Q8_1 buffer size needed
            int k = config_.d_model;
            size_t q8_1_size = QuantizeStage::get_quantized_buffer_size(seq_len, k);

            LOG_INFO("[Qwen2LayerExecutor] Checking Q8_1 buffer: ptr=" << buffers.q8_1_attn_buffer
                                                                       << " size=" << buffers.q8_1_attn_size << " required=" << q8_1_size);

            // Ensure buffer is allocated (pipeline should pre-allocate, but handle dynamic case)
            if (!buffers.q8_1_attn_buffer || buffers.q8_1_attn_size < q8_1_size)
            {
                LOG_INFO("[Qwen2LayerExecutor] Q8_1 buffer not pre-allocated or too small, using FP32 GEMM path");
                // Fall back to FP32 path if buffer not available
                // (In production, buffer should be pre-allocated by pipeline)

                // Q projection
                if (layer.wq)
                {
                    int q_n = static_cast<int>(layer.wq->shape()[0]);
                    int q_k = static_cast<int>(layer.wq->shape()[1]);
                    graph.addNode(prefix + "q_proj",
                                  ComputeStageFactory::createGEMM(
                                      GEMMStage::Params{
                                          buffers.normalized->data(), layer.wq,
                                          buffers.Q->mutable_data(),
                                          seq_len, q_n, q_k,
                                          1.0f, 0.0f, false},
                                      backend),
                                  device_idx);
                }

                // K projection
                if (layer.wk)
                {
                    int k_n = static_cast<int>(layer.wk->shape()[0]);
                    int k_k = static_cast<int>(layer.wk->shape()[1]);
                    graph.addNode(prefix + "k_proj",
                                  ComputeStageFactory::createGEMM(
                                      GEMMStage::Params{
                                          buffers.normalized->data(), layer.wk,
                                          buffers.K->mutable_data(),
                                          seq_len, k_n, k_k,
                                          1.0f, 0.0f, false},
                                      backend),
                                  device_idx);
                }

                // V projection
                if (layer.wv)
                {
                    int v_n = static_cast<int>(layer.wv->shape()[0]);
                    int v_k = static_cast<int>(layer.wv->shape()[1]);
                    graph.addNode(prefix + "v_proj",
                                  ComputeStageFactory::createGEMM(
                                      GEMMStage::Params{
                                          buffers.normalized->data(), layer.wv,
                                          buffers.V->mutable_data(),
                                          seq_len, v_n, v_k,
                                          1.0f, 0.0f, false},
                                      backend),
                                  device_idx);
                }
            }
            else
            {
                // Use Q8_1 path: quantize once, reuse for Q/K/V (matches FusedGEMM)
                LOG_DEBUG("[Qwen2LayerExecutor] Using Q8_1 pre-quantized GEMM path");

                // Stage 2a: Quantize normalized activations to Q8_1
                graph.addNode(prefix + "attn_quantize",
                              ComputeStageFactory::createQuantize(
                                  QuantizeStage::Params{
                                      buffers.normalized->data(),
                                      buffers.q8_1_attn_buffer,
                                      seq_len, k},
                                  backend),
                              device_idx);

                // Stage 2b: Q projection using pre-quantized input
                if (layer.wq)
                {
                    int q_n = static_cast<int>(layer.wq->shape()[0]);
                    int q_k = static_cast<int>(layer.wq->shape()[1]);

                    GEMMStage::Params q_params;
                    q_params.A_q8_1 = buffers.q8_1_attn_buffer; // Pre-quantized
                    q_params.B = layer.wq;
                    q_params.C = buffers.Q->mutable_data();
                    q_params.m = seq_len;
                    q_params.n = q_n;
                    q_params.k = q_k;

                    graph.addNode(prefix + "q_proj",
                                  ComputeStageFactory::createGEMM(q_params, backend),
                                  device_idx);
                }

                // Stage 2c: K projection using pre-quantized input
                if (layer.wk)
                {
                    int k_n = static_cast<int>(layer.wk->shape()[0]);
                    int k_k = static_cast<int>(layer.wk->shape()[1]);

                    GEMMStage::Params k_params;
                    k_params.A_q8_1 = buffers.q8_1_attn_buffer; // Pre-quantized
                    k_params.B = layer.wk;
                    k_params.C = buffers.K->mutable_data();
                    k_params.m = seq_len;
                    k_params.n = k_n;
                    k_params.k = k_k;

                    graph.addNode(prefix + "k_proj",
                                  ComputeStageFactory::createGEMM(k_params, backend),
                                  device_idx);
                }

                // Stage 2d: V projection using pre-quantized input
                if (layer.wv)
                {
                    int v_n = static_cast<int>(layer.wv->shape()[0]);
                    int v_k = static_cast<int>(layer.wv->shape()[1]);

                    GEMMStage::Params v_params;
                    v_params.A_q8_1 = buffers.q8_1_attn_buffer; // Pre-quantized
                    v_params.B = layer.wv;
                    v_params.C = buffers.V->mutable_data();
                    v_params.m = seq_len;
                    v_params.n = v_n;
                    v_params.k = v_k;

                    graph.addNode(prefix + "v_proj",
                                  ComputeStageFactory::createGEMM(v_params, backend),
                                  device_idx);
                }

                // Dependencies: projections depend on quantization
                if (layer.wq)
                    graph.addDependency(prefix + "q_proj", prefix + "attn_quantize");
                if (layer.wk)
                    graph.addDependency(prefix + "k_proj", prefix + "attn_quantize");
                if (layer.wv)
                    graph.addDependency(prefix + "v_proj", prefix + "attn_quantize");
            }

            // Dependencies: projections/quantize depend on norm
            if (env.execution.exec_rmsnorm)
            {
                if (buffers.q8_1_attn_buffer && buffers.q8_1_attn_size >= q8_1_size)
                {
                    graph.addDependency(prefix + "attn_quantize", prefix + "attn_norm");
                }
                else
                {
                    if (layer.wq)
                        graph.addDependency(prefix + "q_proj", prefix + "attn_norm");
                    if (layer.wk)
                        graph.addDependency(prefix + "k_proj", prefix + "attn_norm");
                    if (layer.wv)
                        graph.addDependency(prefix + "v_proj", prefix + "attn_norm");
                }
            }
        }

        // Stage 3: RoPE on Q and K
        if (env.execution.exec_rope)
        {
            int pos_offset = 0; // TODO: Get from position_ids

            graph.addNode(prefix + "q_rope",
                          ComputeStageFactory::createRoPE(
                              RoPEStage::Params{
                                  buffers.Q->mutable_data(),
                                  seq_len,
                                  config_.n_heads,
                                  config_.head_dim,
                                  pos_offset,
                                  config_.rope_theta},
                              backend),
                          device_idx);

            graph.addNode(prefix + "k_rope",
                          ComputeStageFactory::createRoPE(
                              RoPEStage::Params{
                                  buffers.K->mutable_data(),
                                  seq_len,
                                  config_.n_kv_heads,
                                  config_.head_dim,
                                  pos_offset,
                                  config_.rope_theta},
                              backend),
                          device_idx);

            // Dependencies: RoPE depends on projections
            if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "q_rope", prefix + "q_proj");
                graph.addDependency(prefix + "k_rope", prefix + "k_proj");
            }
        }

        // Stage 4: Attention computation with KV cache integration
        if (env.execution.exec_attention)
        {
            // Use the production attention stage with KV cache support
            AttentionWithKVCacheStage::Params attn_params;
            attn_params.Q = buffers.Q;
            attn_params.K = buffers.K;
            attn_params.V = buffers.V;
            attn_params.output = buffers.attn_output;
            attn_params.kv_cache = kv_cache; // Integrate with KV cache
            attn_params.layer_idx = layer_idx;
            attn_params.mode = AttentionWithKVCacheStage::Mode::AUTO; // Auto-detect prefill vs decode
            attn_params.batch_size = 1;
            attn_params.seq_len = seq_len;
            attn_params.n_heads = config_.n_heads;
            attn_params.n_kv_heads = config_.n_kv_heads;
            attn_params.head_dim = config_.head_dim;
            attn_params.causal = true;
            attn_params.window_size = -1;   // Full attention
            attn_params.mpi_ctx = mpi_ctx_; // MPI support
            // Determine MPI strategy: TensorParallel if multi-rank, else None
            MPIStrategy mpi_strategy = MPIStrategy::None;
            if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
            {
                mpi_strategy = MPIStrategy::TensorParallel;
            }
            attn_params.mpi_strategy = static_cast<int>(mpi_strategy);
            // Workspace buffers from PipelineBase (required for MPI TP with causal masking)
            attn_params.workspace_scores = buffers.workspace_scores;
            attn_params.workspace_context = buffers.workspace_context;
            attn_params.workspace_mask = buffers.workspace_mask;
            attn_params.sequence_lengths = nullptr; // Single sequence
            attn_params.position_offset = position_ids ? position_ids[0] : 0;

            graph.addNode(prefix + "attention",
                          ComputeStageFactory::createAttentionWithKVCache(attn_params, backend),
                          device_idx);

            // Dependencies: attention depends on RoPE (or projections if no RoPE)
            if (env.execution.exec_rope)
            {
                graph.addDependency(prefix + "attention", prefix + "q_rope");
                graph.addDependency(prefix + "attention", prefix + "k_rope");
            }
            else if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "attention", prefix + "q_proj");
                graph.addDependency(prefix + "attention", prefix + "k_proj");
                graph.addDependency(prefix + "attention", prefix + "v_proj");
            }
        }

        // Stage 5: Output projection (Wo)
        if (env.execution.exec_gemm && layer.wo)
        {
            // Use weight tensor's actual dimensions (may be sharded)
            // shape()[0] = n (output dim), shape()[1] = k (input dim)
            int wo_n = static_cast<int>(layer.wo->shape()[0]);
            int wo_k = static_cast<int>(layer.wo->shape()[1]);

            graph.addNode(prefix + "wo_proj",
                          ComputeStageFactory::createGEMM(
                              GEMMStage::Params{
                                  buffers.attn_output->data(),
                                  layer.wo,
                                  buffers.attn_proj->mutable_data(),
                                  seq_len, wo_n, wo_k,
                                  1.0f, 0.0f, false},
                              backend),
                          device_idx);

            // Add Allreduce stage when weights are sharded row-parallel
            // Row-parallel GEMM produces partial results that must be summed across ranks
            bool wo_is_sharded = isRowParallelSharded(layer.wo);
            bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

            if (wo_is_sharded && has_multi_rank)
            {
                graph.addNode(prefix + "wo_allreduce",
                              ComputeStageFactory::createAllreduce(
                                  AllreduceStage::Params{
                                      buffers.attn_proj->mutable_data(),
                                      static_cast<size_t>(seq_len) * config_.d_model,
                                      getMPIComm(mpi_ctx_.get())},
                                  backend),
                              device_idx);

                // Allreduce depends on Wo projection
                graph.addDependency(prefix + "wo_allreduce", prefix + "wo_proj");

                LOG_TRACE("[Qwen2LayerExecutor] Layer " << layer_idx
                                                        << " Wo: row-parallel sharded, adding allreduce");
            }

            // Dependencies: Wo depends on attention
            if (env.execution.exec_attention)
            {
                graph.addDependency(prefix + "wo_proj", prefix + "attention");
            }
        }

        // Stage 6: Residual connection (in-place: current_hidden += attn_proj)
        if (env.execution.exec_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = buffers.attn_proj->data();
            res_params.residual = buffers.current_hidden->data();       // Residual = current_hidden (before modification)
            res_params.output = buffers.current_hidden->mutable_data(); // Output = current_hidden (in-place add)
            res_params.num_elements = static_cast<size_t>(seq_len) * config_.d_model;
            res_params.rows = seq_len;
            res_params.cols = config_.d_model;
            res_params.precision = config_.activation_precision; // Pass precision from config

            graph.addNode(prefix + "attn_residual",
                          ComputeStageFactory::createResidualAdd(res_params, backend),
                          device_idx);

            // Dependencies: residual depends on Wo (or Wo allreduce if sharded)
            if (env.execution.exec_gemm && layer.wo)
            {
                bool wo_is_sharded = isRowParallelSharded(layer.wo);
                bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

                if (wo_is_sharded && has_multi_rank)
                {
                    graph.addDependency(prefix + "attn_residual", prefix + "wo_allreduce");
                }
                else
                {
                    graph.addDependency(prefix + "attn_residual", prefix + "wo_proj");
                }
            }
        }

        return graph;
    }

    // =============================================================================
    // FFN Block Execution
    // =============================================================================

    bool Qwen2LayerExecutor::executeFFN(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int device_idx)
    {
        const auto &env = debugEnv();

        // Build compute graph for this FFN block
        ComputeGraph graph = buildFFNGraph(layer, buffers, layer_idx, seq_len, device_idx);

        // Get or create device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        // Execute graph
        bool success = executor_.execute(graph, ctx);

        if (!success)
        {
            LOG_ERROR("[Qwen2LayerExecutor] FFN block failed at layer " << layer_idx);
        }

        return success;
    }

    ComputeGraph Qwen2LayerExecutor::buildFFNGraph(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int device_idx)
    {
        ComputeGraph graph;
        const auto &env = debugEnv();
        std::string prefix = "layer" + std::to_string(layer_idx) + "_";

        // Determine backend type
        auto &dm = DeviceManager::instance();
        ComputeBackendType backend = ComputeBackendType::CPU_OPENBLAS;
        if (static_cast<size_t>(device_idx) < dm.devices().size())
        {
            backend = dm.devices()[device_idx].type;
        }

        // Stage 1: Pre-FFN RMSNorm
        if (env.execution.exec_rmsnorm)
        {
            RMSNormStage::Params ffn_norm_params;
            ffn_norm_params.input = buffers.current_hidden->data();      // Read from post-attention hidden
            ffn_norm_params.output = buffers.normalized->mutable_data(); // Write to normalized
            ffn_norm_params.gamma = nullptr;
            if (layer.ffn_norm != nullptr)
            {
                ffn_norm_params.gamma = layer.ffn_norm->data();
            }
            ffn_norm_params.seq_len = seq_len;
            ffn_norm_params.hidden_dim = config_.d_model;
            ffn_norm_params.eps = config_.rms_norm_eps;

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createRMSNorm(ffn_norm_params, backend),
                          device_idx);
        }

        // Stage 2: Gate and Up projections using pre-quantized activations (like FusedGEMM)
        if (env.execution.exec_gemm)
        {
            // Calculate Q8_1 buffer size needed
            int k = config_.d_model;
            size_t q8_1_size = QuantizeStage::get_quantized_buffer_size(seq_len, k);

            // Check if Q8_1 buffer is available
            if (!buffers.q8_1_ffn_buffer || buffers.q8_1_ffn_size < q8_1_size)
            {
                LOG_DEBUG("[Qwen2LayerExecutor] FFN Q8_1 buffer not pre-allocated, using FP32 GEMM path");
                // Fall back to FP32 path

                // Gate projection
                if (layer.gate_proj)
                {
                    int gate_n = static_cast<int>(layer.gate_proj->shape()[0]);
                    int gate_k = static_cast<int>(layer.gate_proj->shape()[1]);
                    graph.addNode(prefix + "gate_proj",
                                  ComputeStageFactory::createGEMM(
                                      GEMMStage::Params{
                                          buffers.normalized->data(), layer.gate_proj,
                                          buffers.gate->mutable_data(),
                                          seq_len, gate_n, gate_k,
                                          1.0f, 0.0f, false},
                                      backend),
                                  device_idx);
                }

                // Up projection
                if (layer.up_proj)
                {
                    int up_n = static_cast<int>(layer.up_proj->shape()[0]);
                    int up_k = static_cast<int>(layer.up_proj->shape()[1]);
                    graph.addNode(prefix + "up_proj",
                                  ComputeStageFactory::createGEMM(
                                      GEMMStage::Params{
                                          buffers.normalized->data(), layer.up_proj,
                                          buffers.up->mutable_data(),
                                          seq_len, up_n, up_k,
                                          1.0f, 0.0f, false},
                                      backend),
                                  device_idx);
                }

                // Dependencies: projections depend on norm
                if (env.execution.exec_rmsnorm)
                {
                    if (layer.gate_proj)
                        graph.addDependency(prefix + "gate_proj", prefix + "ffn_norm");
                    if (layer.up_proj)
                        graph.addDependency(prefix + "up_proj", prefix + "ffn_norm");
                }
            }
            else
            {
                // Use Q8_1 path: quantize once, reuse for gate/up (matches FusedGEMM)
                LOG_DEBUG("[Qwen2LayerExecutor] FFN using Q8_1 pre-quantized GEMM path");

                // Stage 2a: Quantize normalized activations to Q8_1
                graph.addNode(prefix + "ffn_quantize",
                              ComputeStageFactory::createQuantize(
                                  QuantizeStage::Params{
                                      buffers.normalized->data(),
                                      buffers.q8_1_ffn_buffer,
                                      seq_len, k},
                                  backend),
                              device_idx);

                // Stage 2b: Gate projection using pre-quantized input
                if (layer.gate_proj)
                {
                    int gate_n = static_cast<int>(layer.gate_proj->shape()[0]);
                    int gate_k = static_cast<int>(layer.gate_proj->shape()[1]);

                    GEMMStage::Params gate_params;
                    gate_params.A_q8_1 = buffers.q8_1_ffn_buffer; // Pre-quantized
                    gate_params.B = layer.gate_proj;
                    gate_params.C = buffers.gate->mutable_data();
                    gate_params.m = seq_len;
                    gate_params.n = gate_n;
                    gate_params.k = gate_k;

                    graph.addNode(prefix + "gate_proj",
                                  ComputeStageFactory::createGEMM(gate_params, backend),
                                  device_idx);
                }

                // Stage 2c: Up projection using pre-quantized input
                if (layer.up_proj)
                {
                    int up_n = static_cast<int>(layer.up_proj->shape()[0]);
                    int up_k = static_cast<int>(layer.up_proj->shape()[1]);

                    GEMMStage::Params up_params;
                    up_params.A_q8_1 = buffers.q8_1_ffn_buffer; // Pre-quantized
                    up_params.B = layer.up_proj;
                    up_params.C = buffers.up->mutable_data();
                    up_params.m = seq_len;
                    up_params.n = up_n;
                    up_params.k = up_k;

                    graph.addNode(prefix + "up_proj",
                                  ComputeStageFactory::createGEMM(up_params, backend),
                                  device_idx);
                }

                // Dependencies: projections depend on quantization, quantize depends on norm
                if (layer.gate_proj)
                    graph.addDependency(prefix + "gate_proj", prefix + "ffn_quantize");
                if (layer.up_proj)
                    graph.addDependency(prefix + "up_proj", prefix + "ffn_quantize");
                if (env.execution.exec_rmsnorm)
                {
                    graph.addDependency(prefix + "ffn_quantize", prefix + "ffn_norm");
                }
            }
        }

        // Stage 3: SwiGLU activation
        if (env.execution.exec_swiglu)
        {
            graph.addNode(prefix + "swiglu",
                          ComputeStageFactory::createSwiGLU(
                              SwiGLUStage::Params{
                                  buffers.gate->data(),
                                  buffers.up->data(),
                                  buffers.ffn_output->mutable_data(),
                                  seq_len,
                                  config_.d_ff},
                              backend),
                          device_idx);

            // Dependencies: SwiGLU depends on gate and up projections
            if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "swiglu", prefix + "gate_proj");
                graph.addDependency(prefix + "swiglu", prefix + "up_proj");
            }
        }

        // Stage 4: Down projection (needs allreduce in both row-parallel and column-parallel modes)
        if (env.execution.exec_gemm && layer.down_proj)
        {
            // Use weight tensor's actual dimensions
            int down_n = static_cast<int>(layer.down_proj->shape()[0]);
            int down_k = static_cast<int>(layer.down_proj->shape()[1]);

            graph.addNode(prefix + "down_proj",
                          ComputeStageFactory::createGEMM(
                              GEMMStage::Params{
                                  buffers.ffn_output->data(),
                                  layer.down_proj,
                                  buffers.attn_proj->mutable_data(), // Reuse attn_proj as temp buffer
                                  seq_len, down_n, down_k,
                                  1.0f, 0.0f, false},
                              backend),
                          device_idx);

            // Dependencies: down depends on SwiGLU
            if (env.execution.exec_swiglu)
            {
                graph.addDependency(prefix + "down_proj", prefix + "swiglu");
            }

            // Allreduce is needed in:
            // 1. Row-parallel sharding: down_proj K dimension is split, outputs need summing
            // 2. Column-parallel sharding: gate/up split N dimension, down inputs are local,
            //    but each rank computes partial result that needs summing
            bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
            bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);
            bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

            if (needs_allreduce && has_multi_rank)
            {
                // Add AllreduceStage to combine partial GEMM results across ranks
                size_t allreduce_count = static_cast<size_t>(seq_len) * down_n;
                MPI_Comm comm = static_cast<MPI_Comm>(getMPIComm(mpi_ctx_.get()));

                LOG_DEBUG("[buildFFNGraph] Adding down_allreduce: ffn_column_parallel="
                          << config_.ffn_column_parallel << " down_is_row_sharded=" << down_is_row_sharded
                          << " count=" << allreduce_count);

                graph.addNode(prefix + "down_allreduce",
                              ComputeStageFactory::createAllreduce(
                                  AllreduceStage::Params{
                                      buffers.attn_proj->mutable_data(),
                                      allreduce_count,
                                      comm},
                                  backend),
                              device_idx);

                graph.addDependency(prefix + "down_allreduce", prefix + "down_proj");
            }
        }

        // Stage 5: Residual connection
        if (env.execution.exec_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = buffers.attn_proj->data();               // Down proj output
            res_params.residual = buffers.current_hidden->data();       // Previous residual
            res_params.output = buffers.current_hidden->mutable_data(); // In-place update
            res_params.num_elements = static_cast<size_t>(seq_len) * config_.d_model;
            res_params.rows = seq_len;
            res_params.cols = config_.d_model;
            res_params.precision = config_.activation_precision; // Pass precision from config

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params, backend),
                          device_idx);

            // Dependencies: residual depends on down projection (or allreduce if sharded)
            if (env.execution.exec_gemm && layer.down_proj)
            {
                bool down_is_row_sharded = isRowParallelSharded(layer.down_proj);
                bool needs_allreduce = (down_is_row_sharded || config_.ffn_column_parallel);
                bool has_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;

                if (needs_allreduce && has_multi_rank)
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_allreduce");
                }
                else
                {
                    graph.addDependency(prefix + "ffn_residual", prefix + "down_proj");
                }
            }
        }

        return graph;
    }

    // =============================================================================
    // Full Layer Execution
    // =============================================================================

    bool Qwen2LayerExecutor::executeLayer(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_INFO("[Qwen2LayerExecutor::executeLayer] LAYER_EXEC_ENTERED layer_idx=" << layer_idx << " seq_len=" << seq_len);

        // Execute attention block
        if (!executeAttention(layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device_idx))
        {
            return false;
        }

        // Execute FFN block
        if (!executeFFN(layer, buffers, layer_idx, seq_len, device_idx))
        {
            return false;
        }

        return true;
    }

} // namespace llaminar2
