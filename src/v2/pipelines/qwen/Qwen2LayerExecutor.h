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
#include "../../execution/ExecutionPolicy.h" // For ExecutionPolicy
#include "../../pipelines/PipelineConfig.h"  // For ActivationPrecision
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

        // Attention biases (Qwen2 uses Q/K/V biases)
        TensorBase *q_bias = nullptr; ///< Query bias [d_model]
        TensorBase *k_bias = nullptr; ///< Key bias [n_kv_heads * head_dim]
        TensorBase *v_bias = nullptr; ///< Value bias [n_kv_heads * head_dim]

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

        /// Use decomposed attention path (Phase 9): KVCacheAppendStage + AttentionComputeStage
        /// When false (default), uses legacy AttentionWithKVCacheStage + MpiAttentionOrchestrator
        /// When true, uses decomposed stages that route through KernelFactory
        bool use_decomposed_attention = false;

        /**
         * @brief Execution policy controlling which operations run
         *
         * Default: All operations enabled (ExecutionPolicy::allEnabled())
         *
         * Use ExecutionPolicy::fromEnvironment() to read from LLAMINAR_EXEC_* vars
         * for backward compatibility with environment-based configuration.
         *
         * Example usage for testing:
         * @code
         * Qwen2ExecutorConfig config;
         * config.execution_policy = ExecutionPolicy::ffnOnly();  // Test FFN in isolation
         * @endcode
         */
        ExecutionPolicy execution_policy = ExecutionPolicy::allEnabled();
    };

    /**
     * @brief Activation buffers passed to layer executor
     *
     * ## Buffer Lifecycle Semantics
     *
     * Buffers are categorized by their data flow role to clarify ownership
     * and modification semantics:
     *
     * ### Buffer Categories:
     *
     * **INOUT (Input modified in-place)**:
     *   The buffer provides input and is modified during execution.
     *   The original input value is consumed/overwritten.
     *   - `residual`: Accumulates attention/FFN outputs via +=
     *   - `normalized`: Receives norm output, may be reused
     *
     * **SCRATCH (Temporary workspace)**:
     *   Used for intermediate results. Content undefined after execution.
     *   May be reused across layers without preservation.
     *   - `Q`, `K`, `V`: Projection outputs, consumed by attention
     *   - `attn_output`: Pre-Wo output, consumed by projection
     *   - `gate`, `up`: FFN projections, consumed by SwiGLU
     *   - `ffn_output`: SwiGLU output, consumed by down projection
     *   - `workspace_scores`, `workspace_context`, `workspace_mask`: Attention scratch
     *
     * **OUTPUT (Write-only)**:
     *   Written by execution, should be consumed before next execution.
     *   - `current_hidden`: Final hidden state output
     *   - `attn_proj`: After Wo projection, before residual add
     *
     * ## In-Place Modification Pattern
     *
     * Some operations modify inputs in-place for efficiency:
     * - `residual += attn_proj` (residual is INOUT)
     * - RoPE modifies Q/K in-place (Q/K are INOUT during attention)
     *
     * For these cases, the buffer serves as both input AND output.
     * The semantic category should be understood as "INPUT that becomes OUTPUT".
     *
     * ## Thread Safety
     *
     * The executor does NOT own these buffers. Caller must ensure:
     * - Buffers outlive the executor call
     * - No concurrent writes from other threads
     * - SCRATCH buffers may be aliased if execution is sequential
     */
    struct Qwen2ActivationBuffers
    {
        // === INOUT Buffers (modified in-place) ===
        TensorBase *residual = nullptr;   ///< [INOUT] Hidden state, accumulates via +=
        TensorBase *normalized = nullptr; ///< [INOUT] Post-norm, receives norm output

        // === SCRATCH Buffers (intermediate, content undefined after use) ===
        TensorBase *Q = nullptr;                 ///< [SCRATCH] Query projection output
        TensorBase *K = nullptr;                 ///< [SCRATCH] Key projection output
        TensorBase *V = nullptr;                 ///< [SCRATCH] Value projection output
        TensorBase *attn_output = nullptr;       ///< [SCRATCH] Attention output (pre-Wo)
        TensorBase *gate = nullptr;              ///< [SCRATCH] FFN gate projection
        TensorBase *up = nullptr;                ///< [SCRATCH] FFN up projection
        TensorBase *ffn_output = nullptr;        ///< [SCRATCH] SwiGLU output
        TensorBase *workspace_scores = nullptr;  ///< [SCRATCH] Attention scores workspace
        TensorBase *workspace_context = nullptr; ///< [SCRATCH] Attention context workspace
        TensorBase *workspace_mask = nullptr;    ///< [SCRATCH] Causal/padding mask workspace

        // === OUTPUT Buffers (write-only, consumed before next call) ===
        TensorBase *attn_proj = nullptr;      ///< [OUTPUT] Post-Wo projection
        TensorBase *current_hidden = nullptr; ///< [OUTPUT] Final hidden state
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
