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
#include "../../tensors/Tensors.h" // For FP32Tensor
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
            const float *input = buffers.current_hidden->fp32_data();
            LOG_INFO("[EXEC_ATTN_INPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                << " input[0:4]=" << input[0] << "," << input[1] << "," << input[2] << "," << input[3]);
            LOG_INFO("[EXEC_ATTN_INPUT] wq=" << static_cast<const void *>(layer.wq)
                                             << " wq_shape=[" << layer.wq->shape()[0] << "," << layer.wq->shape()[1] << "]");
        }

        // Build compute graph for this attention block
        ComputeGraph graph = buildAttentionGraph(layer, buffers, layer_idx, seq_len,
                                                 kv_cache, position_ids, device_idx);

        // Debug: log graph structure
        if (layer_idx == 0)
        {
            auto order = graph.getExecutionOrder();
            LOG_INFO("[EXEC_ATTN] Graph has " << graph.size() << " nodes, execution order:");
            for (const auto &name : order)
            {
                LOG_INFO("[EXEC_ATTN]   - " << name);
            }
        }

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
            const float *normalized = buffers.normalized->fp32_data();
            const float *Q = buffers.Q->fp32_data();
            const float *attn_output = buffers.attn_output->fp32_data();
            const float *attn_proj = buffers.attn_proj->fp32_data();
            const float *output = buffers.current_hidden->fp32_data();
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
                                                     << " layer.wq=" << static_cast<const void *>(layer.wq)
                                                     << " layer.wo=" << layer.wo << " world_size="
                                                     << (mpi_ctx_ ? mpi_ctx_->world_size() : 1));

        // Determine backend type for stage creation
        auto &dm = DeviceManager::instance();
        ComputeBackendType backend = ComputeBackendType::CPU;
        if (static_cast<size_t>(device_idx) < dm.devices().size())
        {
            backend = dm.devices()[device_idx].type;
        }

        // Stage 1: Pre-attention RMSNorm
        if (env.execution.exec_rmsnorm)
        {
            RMSNormStage::Params attn_norm_params;
            // Use TensorBase* API (preferred) - tensor carries type/device/dimension info
            attn_norm_params.input = buffers.current_hidden; // TensorBase* - read from current_hidden
            attn_norm_params.output = buffers.normalized;    // TensorBase* - write to normalized
            attn_norm_params.gamma = layer.attn_norm;        // TensorBase* - gamma weights (can be nullptr)
            attn_norm_params.eps = config_.rms_norm_eps;
            attn_norm_params.seq_len = seq_len; // CRITICAL: Explicit seq_len for pre-allocated buffers

            graph.addNode(prefix + "attn_norm",
                          ComputeStageFactory::createRMSNorm(attn_norm_params),
                          device_idx);
        }

        // Stage 2: Q/K/V projections using FusedQKVGEMMStage
        // The fused stage handles quantization internally via ITensorGemm::multiply_fused()
        // - For quantized weights: Optimized shared Q8_1 quantization
        // - For FP32/FP16/BF16 weights: Falls back to sequential GEMMs
        if (env.execution.exec_gemm && layer.wq && layer.wk && layer.wv)
        {
            LOG_DEBUG("[Qwen2LayerExecutor] Using FusedQKVGEMMStage");

            int k = config_.d_model;
            int q_n = static_cast<int>(layer.wq->shape()[0]);
            int k_n = static_cast<int>(layer.wk->shape()[0]);
            int v_n = static_cast<int>(layer.wv->shape()[0]);

            // Extract bias pointers (nullptr if model doesn't have biases)
            const float *q_bias_ptr = nullptr;
            const float *k_bias_ptr = nullptr;
            const float *v_bias_ptr = nullptr;

            if (layer.q_bias)
            {
                auto *q_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.q_bias);
                if (q_bias_fp32)
                    q_bias_ptr = q_bias_fp32->data();
            }
            if (layer.k_bias)
            {
                auto *k_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.k_bias);
                if (k_bias_fp32)
                    k_bias_ptr = k_bias_fp32->data();
            }
            if (layer.v_bias)
            {
                auto *v_bias_fp32 = dynamic_cast<FP32Tensor *>(layer.v_bias);
                if (v_bias_fp32)
                    v_bias_ptr = v_bias_fp32->data();
            }

            FusedQKVGEMMStage::Params qkv_params;
            qkv_params.input = buffers.normalized; // TensorBase* - stage extracts FP32 internally
            qkv_params.m = seq_len;
            qkv_params.k = k;
            qkv_params.wq = layer.wq;
            qkv_params.output_q = buffers.Q; // TensorBase* - stage extracts mutable FP32 internally
            qkv_params.n_q = q_n;
            qkv_params.bias_q = q_bias_ptr;
            qkv_params.wk = layer.wk;
            qkv_params.output_k = buffers.K; // TensorBase* - stage extracts mutable FP32 internally
            qkv_params.n_k = k_n;
            qkv_params.bias_k = k_bias_ptr;
            qkv_params.wv = layer.wv;
            qkv_params.output_v = buffers.V; // TensorBase* - stage extracts mutable FP32 internally
            qkv_params.n_v = v_n;
            qkv_params.bias_v = v_bias_ptr;

            graph.addNode(prefix + "qkv_proj",
                          ComputeStageFactory::createFusedQKVGEMM(qkv_params),
                          device_idx);

            // Dependency: fused projection depends on norm
            if (env.execution.exec_rmsnorm)
            {
                graph.addDependency(prefix + "qkv_proj", prefix + "attn_norm");
            }
        }

        // Stage 3: RoPE on Q and K
        if (env.execution.exec_rope)
        {
            // For decode mode, position_ids[0] contains the position of the new token
            // (e.g., 9 for the 10th token after a 9-token prefill)
            int pos_offset = position_ids ? position_ids[0] : 0;

            // Use TensorBase* API - pass both Q and K to RoPE stage
            RoPEStage::Params rope_params;
            rope_params.Q = buffers.Q;
            rope_params.K = buffers.K;
            rope_params.n_heads = config_.n_heads;
            rope_params.n_kv_heads = config_.n_kv_heads;
            rope_params.head_dim = config_.head_dim;
            rope_params.pos_offset = pos_offset;
            rope_params.theta_base = config_.rope_theta;

            graph.addNode(prefix + "rope",
                          ComputeStageFactory::createRoPE(rope_params),
                          device_idx);

            // Dependency: RoPE depends on fused QKV projection
            if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "rope", prefix + "qkv_proj");
            }
        }

        // Stage 4: Attention computation with KV cache integration
        if (env.execution.exec_attention)
        {
            if (config_.use_decomposed_attention)
            {
                // =================================================================
                // Phase 9 Decomposed Path: KVCacheAppendStage + AttentionComputeStage
                // =================================================================
                // This path uses KernelFactory for device-aware kernel dispatch

                // Stage 4a: Append K/V to cache (if cache provided)
                if (kv_cache)
                {
                    KVCacheAppendStage::Params kv_append_params;
                    kv_append_params.K = buffers.K;
                    kv_append_params.V = buffers.V;
                    kv_append_params.kv_cache = kv_cache;
                    kv_append_params.layer_idx = layer_idx;
                    kv_append_params.seq_idx = 0;
                    kv_append_params.num_tokens = seq_len;

                    graph.addNode(prefix + "kv_append",
                                  ComputeStageFactory::createKVCacheAppend(kv_append_params),
                                  device_idx);

                    // KV append depends on RoPE (or projections)
                    if (env.execution.exec_rope)
                    {
                        graph.addDependency(prefix + "kv_append", prefix + "rope");
                    }
                    else if (env.execution.exec_gemm)
                    {
                        graph.addDependency(prefix + "kv_append", prefix + "qkv_proj");
                    }
                }

                // Stage 4b: Pure attention compute
                // Get K/V from cache if available, otherwise use buffers directly
                TensorBase *K_for_attn = buffers.K;
                TensorBase *V_for_attn = buffers.V;
                int kv_len = seq_len;

                if (kv_cache)
                {
                    // Note: We'll use cached K/V which includes just-appended tokens
                    // This must be set up at graph execution time via a lambda or
                    // deferred parameter resolution. For now, we still use buffer pointers
                    // and let the kernel handle cache lookup internally.
                    K_for_attn = kv_cache->get_k_base(layer_idx, 0);
                    V_for_attn = kv_cache->get_v_base(layer_idx, 0);
                    kv_len = kv_cache->get_cached_tokens(layer_idx, 0);
                    // During first forward, cache may be empty - use seq_len
                    if (kv_len == 0)
                        kv_len = seq_len;
                }

                // Detect attention mode for optimized kernel dispatch
                AttentionMode mode = detect_attention_mode(1, seq_len, kv_len);
                LOG_TRACE("[Qwen2LayerExecutor] Layer " << layer_idx
                                                        << " attention mode: " << attention_mode_name(mode)
                                                        << " (seq_len=" << seq_len << ", kv_len=" << kv_len << ")");

                AttentionComputeStage::Params attn_params;
                attn_params.Q = buffers.Q;
                attn_params.K = K_for_attn;
                attn_params.V = V_for_attn;
                attn_params.output = buffers.attn_output;
                attn_params.batch_size = 1;
                attn_params.seq_len = seq_len;
                attn_params.kv_len = kv_len;
                attn_params.n_heads = config_.n_heads;
                attn_params.n_kv_heads = config_.n_kv_heads;
                attn_params.head_dim = config_.head_dim;
                attn_params.causal = true;
                attn_params.window_size = -1;
                attn_params.attention_mode = mode;
                attn_params.auto_detect_mode = false; // Already detected above
                attn_params.workspace_scores = buffers.workspace_scores;
                attn_params.workspace_context = buffers.workspace_context;
                attn_params.workspace_mask = buffers.workspace_mask;
                attn_params.mpi_ctx = mpi_ctx_.get();
                attn_params.device_idx = device_idx;

                graph.addNode(prefix + "attention",
                              ComputeStageFactory::createAttentionCompute(attn_params),
                              device_idx);

                // Dependencies: attention depends on KV append (or RoPE if no cache)
                if (kv_cache)
                {
                    graph.addDependency(prefix + "attention", prefix + "kv_append");
                }
                else if (env.execution.exec_rope)
                {
                    graph.addDependency(prefix + "attention", prefix + "rope");
                }
                else if (env.execution.exec_gemm)
                {
                    graph.addDependency(prefix + "attention", prefix + "qkv_proj");
                }

                LOG_DEBUG("[Qwen2LayerExecutor] Using decomposed attention path (Phase 9)");
            }
            else
            {
                // =================================================================
                // Legacy Path: AttentionWithKVCacheStage (uses MpiAttentionOrchestrator)
                // =================================================================
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
                              ComputeStageFactory::createAttentionWithKVCache(attn_params),
                              device_idx);

                // Dependencies: attention depends on RoPE (or projections if no RoPE)
                if (env.execution.exec_rope)
                {
                    graph.addDependency(prefix + "attention", prefix + "rope");
                }
                else if (env.execution.exec_gemm)
                {
                    graph.addDependency(prefix + "attention", prefix + "q_proj");
                    graph.addDependency(prefix + "attention", prefix + "k_proj");
                    graph.addDependency(prefix + "attention", prefix + "v_proj");
                }
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
                                  .A = buffers.attn_output, // TensorBase* - stage extracts FP32 internally
                                  .B = layer.wo,
                                  .C = buffers.attn_proj, // TensorBase* - stage extracts mutable FP32 internally
                                  .m = seq_len,
                                  .n = wo_n,
                                  .k = wo_k,
                                  .alpha = 1.0f,
                                  .beta = 0.0f,
                                  .transpose_B = false}),
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
                                      getMPIComm(mpi_ctx_.get())}),
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
            res_params.input = buffers.attn_proj;         // TensorBase* directly
            res_params.residual = buffers.current_hidden; // TensorBase* directly
            res_params.output = buffers.current_hidden;   // In-place add
            // CRITICAL: Set num_elements explicitly for decode mode where buffers are
            // pre-allocated to max_seq_len but we only process seq_len tokens
            res_params.num_elements = static_cast<size_t>(seq_len) * static_cast<size_t>(config_.d_model);

            graph.addNode(prefix + "attn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
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
        ComputeBackendType backend = ComputeBackendType::CPU;
        if (static_cast<size_t>(device_idx) < dm.devices().size())
        {
            backend = dm.devices()[device_idx].type;
        }

        // Stage 1: Pre-FFN RMSNorm
        if (env.execution.exec_rmsnorm)
        {
            RMSNormStage::Params ffn_norm_params;
            // Use TensorBase* API (preferred) - tensor carries type/device/dimension info
            ffn_norm_params.input = buffers.current_hidden; // TensorBase* - read from post-attention hidden
            ffn_norm_params.output = buffers.normalized;    // TensorBase* - write to normalized
            ffn_norm_params.gamma = layer.ffn_norm;         // TensorBase* - gamma weights (can be nullptr)
            ffn_norm_params.eps = config_.rms_norm_eps;
            ffn_norm_params.seq_len = seq_len; // CRITICAL: Explicit seq_len for pre-allocated buffers

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createRMSNorm(ffn_norm_params),
                          device_idx);
        }

        // Stage 2: Gate and Up projections using FusedGateUpGEMMStage
        // The fused stage handles quantization internally via ITensorGemm::multiply_fused()
        // - For quantized weights: Optimized shared Q8_1 quantization
        // - For FP32/FP16/BF16 weights: Falls back to sequential GEMMs
        if (env.execution.exec_gemm && layer.gate_proj && layer.up_proj)
        {
            LOG_DEBUG("[Qwen2LayerExecutor] FFN using FusedGateUpGEMMStage");

            int k = config_.d_model;
            int gate_n = static_cast<int>(layer.gate_proj->shape()[0]);
            int up_n = static_cast<int>(layer.up_proj->shape()[0]);

            FusedGateUpGEMMStage::Params gate_up_params;
            gate_up_params.input = buffers.normalized; // TensorBase* - stage extracts FP32 internally
            gate_up_params.m = seq_len;
            gate_up_params.k = k;
            gate_up_params.w_gate = layer.gate_proj;
            gate_up_params.output_gate = buffers.gate; // TensorBase* - supports Q8_1 via multiply_tensor
            gate_up_params.n_gate = gate_n;
            gate_up_params.w_up = layer.up_proj;
            gate_up_params.output_up = buffers.up; // TensorBase* - supports Q8_1 via multiply_tensor
            gate_up_params.n_up = up_n;
            gate_up_params.mpi_ctx = mpi_ctx_.get(); // For Q8_1 multiply_tensor path
            gate_up_params.device_idx = device_idx;

            graph.addNode(prefix + "gate_up_proj",
                          ComputeStageFactory::createFusedGateUpGEMM(gate_up_params),
                          device_idx);

            // Dependency: fused projection depends on norm
            if (env.execution.exec_rmsnorm)
            {
                graph.addDependency(prefix + "gate_up_proj", prefix + "ffn_norm");
            }
        }

        // Stage 3: SwiGLU activation
        // NOTE: SwiGLU writes output in-place to buffers.up (same as legacy pipeline)
        // buffers.up has shape (seq_len, d_ff), which is correct for SwiGLU output.
        // buffers.ffn_output has shape (seq_len, d_model) for the down projection result.
        if (env.execution.exec_swiglu)
        {
            // Use TensorBase* API - tensor carries type/device/dimension info
            SwiGLUStage::Params swiglu_params;
            swiglu_params.gate = buffers.gate;
            swiglu_params.up = buffers.up;
            swiglu_params.output = buffers.up; // In-place: SwiGLU output to 'up' buffer

            graph.addNode(prefix + "swiglu",
                          ComputeStageFactory::createSwiGLU(swiglu_params),
                          device_idx);

            // Dependency: SwiGLU depends on fused gate/up projection
            if (env.execution.exec_gemm)
            {
                graph.addDependency(prefix + "swiglu", prefix + "gate_up_proj");
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
                                  .A = buffers.up, // TensorBase* - SwiGLU output is in buffers.up (in-place)
                                  .B = layer.down_proj,
                                  .C = buffers.attn_proj, // TensorBase* - Reuse attn_proj as temp buffer
                                  .m = seq_len,
                                  .n = down_n,
                                  .k = down_k,
                                  .alpha = 1.0f,
                                  .beta = 0.0f,
                                  .transpose_B = false}),
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
                                      comm}),
                              device_idx);

                graph.addDependency(prefix + "down_allreduce", prefix + "down_proj");
            }
        }

        // Stage 5: Residual connection
        if (env.execution.exec_residual)
        {
            ResidualAddStage::Params res_params;
            res_params.input = buffers.attn_proj;         // Down proj output (TensorBase*)
            res_params.residual = buffers.current_hidden; // Previous residual (TensorBase*)
            res_params.output = buffers.current_hidden;   // In-place update
            // CRITICAL: Set num_elements explicitly for decode mode where buffers are
            // pre-allocated to max_seq_len but we only process seq_len tokens
            res_params.num_elements = static_cast<size_t>(seq_len) * static_cast<size_t>(config_.d_model);

            graph.addNode(prefix + "ffn_residual",
                          ComputeStageFactory::createResidualAdd(res_params),
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
