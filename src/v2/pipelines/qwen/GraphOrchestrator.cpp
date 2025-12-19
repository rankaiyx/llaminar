/**
 * @file GraphOrchestrator.cpp
 * @brief Implementation of Qwen2 compute graph orchestrator
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file implements the execution layer for Qwen2 models, managing
 * graph execution, device contexts, and caching.
 */

#include "GraphOrchestrator.h"
#include "../../utils/Logger.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/MPIContext.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/UnifiedKVCache.h"
#include <chrono>

namespace llaminar2
{

    // =========================================================================
    // Constructors
    // =========================================================================

    GraphOrchestrator::GraphOrchestrator(
        std::shared_ptr<Qwen2Graph> graph_builder,
        std::shared_ptr<MPIContext> mpi_ctx,
        const GraphCacheConfig &cache_config)
        : graph_builder_(std::move(graph_builder)),
          mpi_ctx_(std::move(mpi_ctx)),
          cache_config_(cache_config)
    {
        if (!graph_builder_)
        {
            throw std::invalid_argument("GraphOrchestrator requires a valid graph builder");
        }

        // Configure executor from graph builder's config
        GraphExecutorConfig exec_config;
        exec_config.default_device = graph_builder_->config().default_device;
        exec_config.enable_profiling = graph_builder_->config().enable_profiling;
        exec_config.enable_validation = graph_builder_->config().enable_validation;

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

        executor_ = GraphExecutor(exec_config);

        // Propagate MPI rank to executor for stage dumping
        if (mpi_ctx_)
        {
            executor_.setMPIRank(mpi_ctx_->rank());
        }

        LOG_INFO("[GraphOrchestrator] Initialized with graph builder, caching="
                 << (cache_config_.enabled ? "enabled" : "disabled"));
    }

    GraphOrchestrator::GraphOrchestrator(
        const Qwen2GraphConfig &graph_config,
        std::shared_ptr<MPIContext> mpi_ctx,
        const GraphCacheConfig &cache_config)
        : GraphOrchestrator(
              std::make_shared<Qwen2Graph>(graph_config, mpi_ctx),
              std::move(mpi_ctx),
              cache_config)
    {
    }

    // =========================================================================
    // Device Context Management
    // =========================================================================

    IDeviceContext *GraphOrchestrator::getDeviceContext(int device_idx)
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
            LOG_ERROR("[GraphOrchestrator] Failed to create device context for device " << device_idx);
            return nullptr;
        }

        IDeviceContext *raw_ptr = ctx.get();
        device_contexts_[device_idx] = std::move(ctx);

        LOG_DEBUG("[GraphOrchestrator] Created device context for device " << device_idx);
        return raw_ptr;
    }

    // =========================================================================
    // Weight and Buffer Configuration
    // =========================================================================

    void GraphOrchestrator::setWeights(const Qwen2ModelWeights &weights)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[GraphOrchestrator] Cannot set weights: graph builder not initialized");
            return;
        }
        graph_builder_->setWeights(weights);
        LOG_DEBUG("[GraphOrchestrator] Model weights configured for full forward pass");
    }

    void GraphOrchestrator::setBuffers(const Qwen2ModelBuffers &buffers)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[GraphOrchestrator] Cannot set buffers: graph builder not initialized");
            return;
        }
        graph_builder_->setBuffers(buffers);
        LOG_DEBUG("[GraphOrchestrator] Model buffers configured for full forward pass");
    }

    bool GraphOrchestrator::hasGlobalWeights() const
    {
        if (!graph_builder_)
        {
            return false;
        }
        // Check if the builder's isInitialized returns true AND we have global weights
        // isInitialized() checks get_layer_weights, but we also need embedding_table etc.
        // For now, rely on the graph builder's internal state
        return graph_builder_->isInitialized();
    }

    // =========================================================================
    // Execution Methods
    // =========================================================================

    bool GraphOrchestrator::executeForward(
        const Qwen2ForwardInput &input,
        Qwen2ForwardOutput &output)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!input.token_ids && !input.batches)
        {
            LOG_ERROR("[GraphOrchestrator] No token input provided");
            return false;
        }

        if (input.seq_len <= 0)
        {
            LOG_ERROR("[GraphOrchestrator] Invalid sequence length: " << input.seq_len);
            return false;
        }

        LOG_TRACE("[GraphOrchestrator] executeForward: batch_size=" << input.batch_size
                                                                    << ", seq_len=" << input.seq_len
                                                                    << ", device=" << input.device_idx);

        // Build position IDs if not provided externally
        std::vector<int> position_ids_storage;
        Qwen2ForwardInput effective_input = input;

        if (!input.position_ids)
        {
            position_ids_storage = Qwen2Graph::buildPositionIds(
                input.seq_len, input.batch_size, input.position_offset);
            effective_input.position_ids = position_ids_storage.data();
        }

        // Build forward graph via declarative builder
        ComputeGraph graph = graph_builder_->buildFullForwardGraph(effective_input, output);

        if (graph.size() == 0)
        {
            LOG_ERROR("[GraphOrchestrator] Empty forward graph");
            return false;
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(input.device_idx);
        if (!ctx)
        {
            LOG_ERROR("[GraphOrchestrator] Failed to get device context");
            return false;
        }

        // Execute
        bool success = executor_.execute(graph, ctx);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        LOG_DEBUG("[GraphOrchestrator] Forward completed in " << ms << "ms, success=" << success);

        return success;
    }

    bool GraphOrchestrator::execute(ComputeGraph &graph, IDeviceContext *ctx)
    {
        return executor_.execute(graph, ctx);
    }

    bool GraphOrchestrator::executeAttention(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        // Debug: dump input to attention (for layer 0 only)
        if (layer_idx == 0)
        {
            const float *input = buffers.current_hidden->fp32_data();
            LOG_INFO("[ORCH_ATTN_INPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                << " input[0:4]=" << input[0] << "," << input[1]
                                                << "," << input[2] << "," << input[3]);
        }

        // Get device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        int pos_offset = position_ids ? position_ids[0] : 0;

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_attention &&
            seq_len == cache_config_.decode_seq_len &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached graph
            if (cache.attention_decode && cache.cached_seq_len == seq_len && cache.valid)
            {
                LOG_DEBUG("[GraphOrchestrator] Reusing cached attention graph for layer "
                          << layer_idx << " (pos_offset=" << pos_offset << ")");

                // Update dynamic parameters (position offset)
                updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);

                // Execute cached graph
                bool success = executor_.execute(*cache.attention_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[GraphOrchestrator] Cached attention graph failed at layer " << layer_idx);
                }

                cache.attention_decode->reset();
                cache_stats_.attention_cache_hits++;
                return success;
            }

            // Build and cache the graph
            LOG_DEBUG("[GraphOrchestrator] Building and caching attention graph for layer "
                      << layer_idx << " (decode mode)");

            cache.attention_decode = std::make_unique<ComputeGraph>(
                graph_builder_->buildAttentionGraph(layer, buffers, layer_idx, seq_len,
                                                    kv_cache, position_ids, device_idx));
            cache.cached_seq_len = seq_len;
            cache.valid = true;
            cache_stats_.attention_cache_misses++;

            // Execute the newly built graph
            bool success = executor_.execute(*cache.attention_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[GraphOrchestrator] Attention block failed at layer " << layer_idx);
            }

            cache.attention_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled)
        // =============================================================================
        cache_stats_.attention_cache_misses++;

        ComputeGraph graph = graph_builder_->buildAttentionGraph(
            layer, buffers, layer_idx, seq_len, kv_cache, position_ids, device_idx);

        // Debug: log graph structure
        if (layer_idx == 0)
        {
            auto order = graph.getExecutionOrder();
            LOG_INFO("[ORCH_ATTN] Graph has " << graph.size() << " nodes, execution order:");
            for (const auto &name : order)
            {
                LOG_INFO("[ORCH_ATTN]   - " << name);
            }
        }

        bool success = executor_.execute(graph, ctx);

        // Debug: dump intermediate buffers (for layer 0 only)
        if (layer_idx == 0 && buffers.normalized && buffers.Q &&
            buffers.attn_output && buffers.attn_proj)
        {
            const float *output = buffers.current_hidden->fp32_data();
            LOG_INFO("[ORCH_ATTN_OUTPUT] layer=" << layer_idx << " seq_len=" << seq_len
                                                 << " output[0:4]=" << output[0] << "," << output[1]
                                                 << "," << output[2] << "," << output[3]);
        }

        if (!success)
        {
            LOG_ERROR("[GraphOrchestrator] Attention block failed at layer " << layer_idx);
        }

        return success;
    }

    bool GraphOrchestrator::executeFFN(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int device_idx)
    {
        // Get device context
        IDeviceContext *ctx = getDeviceContext(device_idx);
        if (!ctx)
        {
            return false;
        }

        // =============================================================================
        // Graph Caching for Decode Mode (Phase 10)
        // =============================================================================
        if (cache_config_.enabled && cache_config_.cache_ffn &&
            seq_len == cache_config_.decode_seq_len &&
            layer_idx >= 0 && static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            auto &cache = layer_graph_cache_[layer_idx];

            // Check if we have a valid cached FFN graph
            if (cache.ffn_decode && cache.valid)
            {
                LOG_DEBUG("[GraphOrchestrator] Reusing cached FFN graph for layer " << layer_idx);

                // Execute cached graph (no params to update for FFN)
                bool success = executor_.execute(*cache.ffn_decode, ctx);
                if (!success)
                {
                    LOG_ERROR("[GraphOrchestrator] Cached FFN graph failed at layer " << layer_idx);
                }

                cache.ffn_decode->reset();
                cache_stats_.ffn_cache_hits++;
                return success;
            }

            // Build and cache the graph
            LOG_DEBUG("[GraphOrchestrator] Building and caching FFN graph for layer "
                      << layer_idx << " (decode mode)");

            cache.ffn_decode = std::make_unique<ComputeGraph>(
                graph_builder_->buildFFNGraph(layer, buffers, layer_idx, seq_len, device_idx));
            cache_stats_.ffn_cache_misses++;

            // Execute the newly built graph
            bool success = executor_.execute(*cache.ffn_decode, ctx);
            if (!success)
            {
                LOG_ERROR("[GraphOrchestrator] FFN block failed at layer " << layer_idx);
            }

            cache.ffn_decode->reset();
            return success;
        }

        // =============================================================================
        // Non-cached path (prefill or caching disabled)
        // =============================================================================
        cache_stats_.ffn_cache_misses++;

        ComputeGraph graph = graph_builder_->buildFFNGraph(layer, buffers, layer_idx, seq_len, device_idx);

        bool success = executor_.execute(graph, ctx);

        if (!success)
        {
            LOG_ERROR("[GraphOrchestrator] FFN block failed at layer " << layer_idx);
        }

        return success;
    }

    bool GraphOrchestrator::executeLayer(
        const Qwen2LayerWeights &layer,
        Qwen2ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        IUnifiedKVCache *kv_cache,
        const int *position_ids,
        int device_idx)
    {
        LOG_INFO("[GraphOrchestrator::executeLayer] LAYER_EXEC_ENTERED layer_idx="
                 << layer_idx << " seq_len=" << seq_len);

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

    // =========================================================================
    // Cache Management
    // =========================================================================

    void GraphOrchestrator::clearCache()
    {
        // Clear graph caches
        for (auto &cache : layer_graph_cache_)
        {
            cache.invalidate();
        }

        // Clear device contexts
        device_contexts_.clear();

        // Reset state
        last_pos_offset_ = -1;

        // Reset stats
        cache_stats_ = CacheStats{};

        LOG_DEBUG("[GraphOrchestrator] All caches cleared");
    }

    void GraphOrchestrator::invalidateGraphCache(int layer_idx)
    {
        if (layer_idx < 0)
        {
            // Invalidate all layers
            for (auto &cache : layer_graph_cache_)
            {
                cache.invalidate();
            }
            cache_stats_.cached_layers = 0;
            LOG_DEBUG("[GraphOrchestrator] All layer graph caches invalidated");
        }
        else if (static_cast<size_t>(layer_idx) < layer_graph_cache_.size())
        {
            layer_graph_cache_[layer_idx].invalidate();
            LOG_DEBUG("[GraphOrchestrator] Layer " << layer_idx << " graph cache invalidated");
        }
    }

    bool GraphOrchestrator::hasValidCachedGraph(int layer_idx, bool is_attention) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        if (!cache.valid)
            return false;

        return is_attention ? (cache.attention_decode != nullptr) : (cache.ffn_decode != nullptr);
    }

    void GraphOrchestrator::setGraphCachingEnabled(bool enabled)
    {
        if (cache_config_.enabled != enabled)
        {
            cache_config_.enabled = enabled;
            if (!enabled)
            {
                invalidateGraphCache(-1);
            }
            LOG_INFO("[GraphOrchestrator] Graph caching "
                     << (enabled ? "enabled" : "disabled"));
        }
    }

    void GraphOrchestrator::initializeGraphCache(int n_layers)
    {
        layer_graph_cache_.resize(n_layers);
        cache_stats_.cached_layers = n_layers;
        LOG_INFO("[GraphOrchestrator] Graph cache initialized for " << n_layers << " layers");
    }

    // =========================================================================
    // Inference State Management (Phase 5)
    // =========================================================================

    bool GraphOrchestrator::initializeInferenceState(
        int batch_size,
        int max_seq_len,
        int device_idx)
    {
        if (!graph_builder_)
        {
            LOG_ERROR("[GraphOrchestrator] Cannot initialize state: no graph builder");
            return false;
        }

        const auto &config = graph_builder_->config();
        int d_model = config.d_model;
        int vocab_size = config.vocab_size;
        int n_layers = config.n_layers;
        int n_heads = config.n_heads;
        int n_kv_heads = config.n_kv_heads;
        int head_dim = config.head_dim;
        int d_ff = config.d_ff;

        LOG_INFO("[GraphOrchestrator] Initializing inference state: batch_size=" << batch_size
                                                                                 << " max_seq_len=" << max_seq_len
                                                                                 << " d_model=" << d_model
                                                                                 << " vocab_size=" << vocab_size);

        // Create a default MPI context if none was provided
        std::shared_ptr<MPIContext> local_mpi_ctx = mpi_ctx_;
        if (!local_mpi_ctx)
        {
            // Create a single-rank MPI context for non-MPI usage
            local_mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_NULL);
            LOG_DEBUG("[GraphOrchestrator] Created default single-rank MPI context");
        }

        // Create tensor factory
        TensorFactory factory(*local_mpi_ctx);

        // Allocate core buffers
        state_.hidden = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);
        state_.logits = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(vocab_size)},
            device_idx);

        // Allocate activation buffers
        state_.normalized = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);
        state_.residual = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);

        // QKV buffers
        state_.Q = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(n_heads * head_dim)},
            device_idx);
        state_.K = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(n_kv_heads * head_dim)},
            device_idx);
        state_.V = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(n_kv_heads * head_dim)},
            device_idx);
        state_.attn_output = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(n_heads * head_dim)},
            device_idx);
        state_.attn_proj = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);

        // FFN buffers
        state_.gate = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_ff)},
            device_idx);
        state_.up = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_ff)},
            device_idx);
        state_.ffn_output = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(d_model)},
            device_idx);

        // Attention workspace
        // scores: [batch_size * n_heads, max_seq_len, max_seq_len]
        state_.workspace_scores = factory.createFP32(
            {static_cast<size_t>(batch_size * n_heads * max_seq_len), static_cast<size_t>(max_seq_len)},
            device_idx);
        // context: [batch_size * n_heads, max_seq_len, head_dim]
        state_.workspace_context = factory.createFP32(
            {static_cast<size_t>(batch_size * n_heads * max_seq_len), static_cast<size_t>(head_dim)},
            device_idx);
        // mask: [batch_size, max_seq_len, max_seq_len]
        state_.workspace_mask = factory.createFP32(
            {static_cast<size_t>(batch_size * max_seq_len), static_cast<size_t>(max_seq_len)},
            device_idx);

        // Create KV cache
        state_.kv_cache = createUnifiedKVCache(
            ActivationPrecision::FP32,
            *local_mpi_ctx,
            n_layers,
            batch_size,
            max_seq_len,
            n_kv_heads,
            head_dim,
            device_idx);

        // Initialize position tracking
        state_.positions.assign(batch_size, 0);
        state_.sequence_lengths.assign(batch_size, 0);

        // Store config
        state_.batch_size = batch_size;
        state_.max_seq_len = max_seq_len;
        state_.d_model = d_model;
        state_.vocab_size = vocab_size;
        state_.device_idx = device_idx;

        LOG_INFO("[GraphOrchestrator] Inference state initialized successfully");
        return true;
    }

    const float *GraphOrchestrator::forward(
        const int *tokens,
        int seq_len,
        int batch_size)
    {
        if (!state_.isInitialized())
        {
            LOG_ERROR("[GraphOrchestrator] forward() called without initialized state");
            return nullptr;
        }

        if (!hasGlobalWeights())
        {
            LOG_ERROR("[GraphOrchestrator] forward() called without global weights set");
            return nullptr;
        }

        if (batch_size > state_.batch_size)
        {
            LOG_ERROR("[GraphOrchestrator] Batch size " << batch_size
                                                        << " exceeds initialized batch size " << state_.batch_size);
            return nullptr;
        }

        int total_tokens = batch_size * seq_len;
        if (total_tokens > state_.batch_size * state_.max_seq_len)
        {
            LOG_ERROR("[GraphOrchestrator] Total tokens " << total_tokens
                                                          << " exceeds buffer capacity "
                                                          << state_.batch_size * state_.max_seq_len);
            return nullptr;
        }

        // Build position IDs
        std::vector<int> position_ids;
        position_ids.reserve(total_tokens);
        for (int b = 0; b < batch_size; ++b)
        {
            int pos_offset = state_.positions[b];
            for (int s = 0; s < seq_len; ++s)
            {
                position_ids.push_back(pos_offset + s);
            }
        }

        // Prepare model buffers from state
        Qwen2ModelBuffers model_buffers;
        model_buffers.current_hidden = state_.hidden.get();
        model_buffers.logits = state_.logits.get();

        // Populate layer buffers
        model_buffers.layer_buffers.current_hidden = state_.hidden.get();
        model_buffers.layer_buffers.normalized = state_.normalized.get();
        model_buffers.layer_buffers.residual = state_.residual.get();
        model_buffers.layer_buffers.Q = state_.Q.get();
        model_buffers.layer_buffers.K = state_.K.get();
        model_buffers.layer_buffers.V = state_.V.get();
        model_buffers.layer_buffers.attn_output = state_.attn_output.get();
        model_buffers.layer_buffers.attn_proj = state_.attn_proj.get();
        model_buffers.layer_buffers.gate = state_.gate.get();
        model_buffers.layer_buffers.up = state_.up.get();
        model_buffers.layer_buffers.ffn_output = state_.ffn_output.get();
        model_buffers.layer_buffers.workspace_scores = state_.workspace_scores.get();
        model_buffers.layer_buffers.workspace_context = state_.workspace_context.get();
        model_buffers.layer_buffers.workspace_mask = state_.workspace_mask.get();

        setBuffers(model_buffers);

        // Build forward input
        Qwen2ForwardInput input;
        input.token_ids = tokens;
        input.position_ids = position_ids.data();
        input.batch_size = batch_size;
        input.seq_len = seq_len;
        input.position_offset = state_.positions[0]; // Legacy compat
        input.device_idx = state_.device_idx;
        input.kv_cache = state_.kv_cache.get();

        // Build forward output
        Qwen2ForwardOutput output;
        output.logits = state_.logits.get();
        output.hidden = state_.hidden.get();

        // Execute forward pass
        bool success = executeForward(input, output);

        if (!success)
        {
            LOG_ERROR("[GraphOrchestrator] forward() execution failed");
            return nullptr;
        }

        // Update positions
        for (int b = 0; b < batch_size; ++b)
        {
            state_.positions[b] += seq_len;
            state_.sequence_lengths[b] += seq_len;
        }

        // Return logits pointer
        return state_.logits->fp32_data();
    }

    const float *GraphOrchestrator::logits() const
    {
        if (!state_.logits)
        {
            return nullptr;
        }
        return state_.logits->fp32_data();
    }

    int GraphOrchestrator::getPosition(int seq_idx) const
    {
        if (seq_idx < 0 || static_cast<size_t>(seq_idx) >= state_.positions.size())
        {
            return 0;
        }
        return state_.positions[seq_idx];
    }

    void GraphOrchestrator::clearInferenceState()
    {
        state_.clear();
        LOG_DEBUG("[GraphOrchestrator] Inference state cleared");
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    void GraphOrchestrator::updateCachedGraphParams(ComputeGraph &graph, int pos_offset, int seq_len)
    {
        // Update all stages in the graph that have dynamic parameters
        auto order = graph.getExecutionOrder();

        for (const auto &node_name : order)
        {
            ComputeNode *node = graph.getNode(node_name);
            if (!node || !node->stage)
                continue;

            // Update dynamic params (pos_offset, seq_len)
            // Only stages that override updateDynamicParams will actually do anything
            node->stage->updateDynamicParams(pos_offset, seq_len);
        }

        LOG_TRACE("[GraphOrchestrator] Updated cached graph params: pos_offset="
                  << pos_offset << " seq_len=" << seq_len);
    }

    bool GraphOrchestrator::canUseCachedGraph(int layer_idx, int seq_len) const
    {
        if (!cache_config_.enabled)
            return false;
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= layer_graph_cache_.size())
            return false;

        const auto &cache = layer_graph_cache_[layer_idx];
        return cache.valid && cache.cached_seq_len == seq_len;
    }

} // namespace llaminar2
