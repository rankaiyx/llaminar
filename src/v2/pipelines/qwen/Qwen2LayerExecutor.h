/**
 * @file Qwen2LayerExecutor.h
 * @brief Qwen2-specific layer executor using ComputeStage graphs
 * @author David Sanftenberg
 * @date December 2025
 *
 * Qwen2LayerExecutor bridges Qwen2Pipeline's layer weights and buffers
 * to the execution framework's ComputeStage graph abstraction.
 *
 * This enables:
 * 1. Declarative compute graph construction for attention/FFN blocks
 * 2. Automatic device-aware stage scheduling
 * 3. Gradual migration from imperative pipeline code
 * 4. Future: parallel/pipelined execution modes
 */

#pragma once

#include "../../execution/LayerExecutor.h"
#include "../../execution/ComputeStage.h"
#include "../../execution/DeviceContext.h"
#include "../../pipelines/PipelineConfig.h" // For ActivationPrecision
#include "../../tensors/Tensors.h"
#include "../../tensors/UnifiedKVCache.h"
#include <memory>
#include <string>

namespace llaminar2
{

    // Forward declarations
    class MPIContext;
    class Qwen2Pipeline; // Forward declare the pipeline class

    /**
     * @brief Layer weights structure for Qwen2LayerExecutor
     *
     * This mirrors the weight pointers from Qwen2Pipeline::LayerWeights
     * but only includes what the executor actually needs. Using raw pointers
     * since executor does NOT own these weights.
     */
    struct Qwen2LayerWeights
    {
        // Attention weights
        TensorBase *wq = nullptr;        ///< Query projection
        TensorBase *wk = nullptr;        ///< Key projection
        TensorBase *wv = nullptr;        ///< Value projection
        TensorBase *wo = nullptr;        ///< Output projection
        TensorBase *attn_norm = nullptr; ///< Pre-attention norm gamma

        // FFN weights
        TensorBase *gate_proj = nullptr; ///< FFN gate projection
        TensorBase *up_proj = nullptr;   ///< FFN up projection
        TensorBase *down_proj = nullptr; ///< FFN down projection
        TensorBase *ffn_norm = nullptr;  ///< Pre-FFN norm gamma
    };

    /**
     * @brief Configuration for Qwen2LayerExecutor
     */
    struct Qwen2ExecutorConfig
    {
        int d_model = 0;                  ///< Model hidden dimension
        int n_heads = 0;                  ///< Number of attention heads
        int n_kv_heads = 0;               ///< Number of KV heads (GQA)
        int head_dim = 0;                 ///< Dimension per head
        int d_ff = 0;                     ///< FFN intermediate dimension (local if column-parallel)
        bool ffn_column_parallel = false; ///< Whether FFN uses column-parallel sharding
        float rms_norm_eps = 1e-6f;       ///< RMSNorm epsilon
        float rope_theta = 10000.0f;      ///< RoPE theta base

        int default_device = 0; ///< Default device index
        bool enable_profiling = false;
        bool enable_validation = false;

        /// Activation precision for compute stages (affects residual add, norms, etc.)
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
    };

    /**
     * @brief Activation buffers passed to layer executor
     *
     * These point to pre-allocated tensors in Qwen2Pipeline.
     * The executor does NOT own these - it just uses them for graph construction.
     * Member names match ActivationBuffers in PipelineBase.h
     */
    struct Qwen2ActivationBuffers
    {
        TensorBase *residual = nullptr;       ///< Current hidden state
        TensorBase *normalized = nullptr;     ///< Post-norm buffer
        TensorBase *Q = nullptr;              ///< Query projection output
        TensorBase *K = nullptr;              ///< Key projection output
        TensorBase *V = nullptr;              ///< Value projection output
        TensorBase *attn_output = nullptr;    ///< Attention output (pre-Wo)
        TensorBase *attn_proj = nullptr;      ///< Post-Wo projection
        TensorBase *gate = nullptr;           ///< FFN gate projection output
        TensorBase *up = nullptr;             ///< FFN up projection output
        TensorBase *ffn_output = nullptr;     ///< FFN intermediate (SwiGLU output)
        TensorBase *current_hidden = nullptr; ///< Output hidden state

        // Q8_1 quantized activation buffers for fused GEMM patterns
        // These are pre-quantized once and reused across Q/K/V and gate/up projections
        void *q8_1_attn_buffer = nullptr; ///< Q8_1 buffer for attention projections
        size_t q8_1_attn_size = 0;        ///< Size in bytes
        void *q8_1_ffn_buffer = nullptr;  ///< Q8_1 buffer for FFN projections
        size_t q8_1_ffn_size = 0;         ///< Size in bytes

        // Attention workspace buffers (pre-allocated by pipeline, reused across layers)
        // Required for MPI tensor-parallel attention with causal masking
        TensorBase *workspace_scores = nullptr;  ///< [n_heads * max_seq, max_seq]
        TensorBase *workspace_context = nullptr; ///< [max_threads * max_seq * head_dim]
        TensorBase *workspace_mask = nullptr;    ///< [max_seq * max_seq] causal/padding mask
    };

    /**
     * @brief Qwen2-specific layer executor
     *
     * Builds ComputeGraph instances for attention and FFN blocks,
     * delegating to LayerExecutor for execution orchestration.
     */
    class Qwen2LayerExecutor
    {
    public:
        /**
         * @brief Construct Qwen2LayerExecutor
         *
         * @param config Architecture configuration
         * @param mpi_ctx MPI context (nullable for single-rank)
         */
        Qwen2LayerExecutor(const Qwen2ExecutorConfig &config,
                           std::shared_ptr<MPIContext> mpi_ctx = nullptr);

        ~Qwen2LayerExecutor() = default;

        // Non-copyable
        Qwen2LayerExecutor(const Qwen2LayerExecutor &) = delete;
        Qwen2LayerExecutor &operator=(const Qwen2LayerExecutor &) = delete;

        /**
         * @brief Set snapshot callback for debugging
         *
         * When set, this callback is invoked after each compute stage executes,
         * allowing capture of intermediate tensors for comparison with legacy path.
         *
         * @param callback Function called with (node_name, snapshot_info) after each stage
         */
        void setSnapshotCallback(StageSnapshotCallback callback)
        {
            executor_.setSnapshotCallback(std::move(callback));
        }

        /**
         * @brief Execute attention block using compute graph
         *
         * @param layer Layer weights (wq, wk, wv, wo, attn_norm)
         * @param buffers Activation buffers
         * @param layer_idx Layer index (for snapshot keys)
         * @param seq_len Sequence length
         * @param kv_cache KV cache (nullable)
         * @param position_ids Position IDs for RoPE
         * @param device_idx Target device
         * @return true on success
         */
        bool executeAttention(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        /**
         * @brief Execute FFN block using compute graph
         *
         * @param layer Layer weights (gate_proj, up_proj, down_proj, ffn_norm)
         * @param buffers Activation buffers
         * @param layer_idx Layer index (for snapshot keys)
         * @param seq_len Sequence length
         * @param device_idx Target device
         * @return true on success
         */
        bool executeFFN(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int device_idx);

        /**
         * @brief Execute full transformer layer (attention + FFN)
         *
         * Convenience method that combines executeAttention and executeFFN.
         */
        bool executeLayer(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        /**
         * @brief Get execution statistics
         */
        const LayerExecutorStats &stats() const { return executor_.stats(); }

        /**
         * @brief Reset statistics
         */
        void resetStats() { executor_.resetStats(); }

    private:
        /**
         * @brief Build compute graph for attention block
         */
        ComputeGraph buildAttentionGraph(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            IUnifiedKVCache *kv_cache,
            const int *position_ids,
            int device_idx);

        /**
         * @brief Build compute graph for FFN block
         */
        ComputeGraph buildFFNGraph(
            const Qwen2LayerWeights &layer,
            Qwen2ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int device_idx);

        /**
         * @brief Get device context for given device index
         */
        IDeviceContext *getDeviceContext(int device_idx);

        Qwen2ExecutorConfig config_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        LayerExecutor executor_;

        // Device context cache
        std::unordered_map<int, std::unique_ptr<IDeviceContext>> device_contexts_;
    };

} // namespace llaminar2
