/**
 * @file PrefillProvider.h
 * @brief Abstract prefill execution provider with built-in snapshot capture
 * @author David Sanftenberg
 *
 * This file defines the PrefillProvider abstraction that enables:
 * - Multiple prefill backend implementations (OpenBLAS, COSMA, GPU)
 * - Built-in snapshot capture for parity testing against PyTorch
 * - Stage-by-stage instrumentation and validation
 * - Runtime backend selection based on workload characteristics
 *
 * Architecture:
 *   AbstractPipeline::prefill()
 *     └─> QwenPipeline::prefill()
 *          └─> PrefillProvider::execute()  [with snapshot hooks]
 *               ├─> OpenBLASPrefillProvider
 *               ├─> COSMAPrefillProvider
 *               └─> (future) GPUPrefillProvider
 *
 * Design Principles:
 * - Strategy pattern: Swap providers at runtime
 * - Base class provides snapshot utilities (all providers inherit)
 * - Zero overhead when snapshots disabled (compiled out in release)
 * - MPI-aware: Providers handle distributed execution
 * - Testable in isolation: Can test providers without full pipeline
 */

#pragma once

#include "AbstractPipeline.h"
#include "PipelineStages.h"
#include "TransformerConfig.h"
#include "tensors/tensor_base.h"
#include "MpiContext.h"
#include "KvCacheProvider.h"
#include <memory>
#include <vector>
#include <string>

namespace llaminar
{
    // Forward declarations
    class PipelineSnapshotManager;

    /**
     * @brief Prefill execution metrics for instrumentation
     */
    struct PrefillMetrics
    {
        int64_t total_flops = 0;    ///< Total floating point operations
        double embedding_ms = 0.0;  ///< Time spent in embedding lookup
        double attention_ms = 0.0;  ///< Total attention time across layers
        double ffn_ms = 0.0;        ///< Total FFN time across layers
        double norm_ms = 0.0;       ///< Total normalization time
        double lm_head_ms = 0.0;    ///< Language model head time
        int layers_executed = 0;    ///< Number of transformer layers executed
        int snapshots_captured = 0; ///< Number of snapshots captured (debug only)
        std::string backend_name;   ///< Backend identifier (e.g., "OpenBLAS", "COSMA")

        void reset()
        {
            total_flops = 0;
            embedding_ms = 0.0;
            attention_ms = 0.0;
            ffn_ms = 0.0;
            norm_ms = 0.0;
            lm_head_ms = 0.0;
            layers_executed = 0;
            snapshots_captured = 0;
            backend_name.clear();
        }

        double total_ms() const
        {
            return embedding_ms + attention_ms + ffn_ms + norm_ms + lm_head_ms;
        }

        double gflops() const
        {
            double total = total_ms();
            if (total <= 0.0)
                return 0.0;
            return (static_cast<double>(total_flops) / 1e9) / (total / 1000.0);
        }
    };

    /**
     * @brief Abstract prefill execution provider (strategy pattern)
     *
     * Base class for all prefill implementations. Provides:
     * - Common interface for prefill execution
     * - Built-in snapshot capture utilities for parity testing
     * - Stage-by-stage instrumentation hooks
     * - Metrics tracking and reporting
     *
     * Derived classes implement specific backends:
     * - OpenBLASPrefillProvider: CPU-based with OpenBLAS matmuls
     * - COSMAPrefillProvider: Distributed execution with COSMA
     * - GPUPrefillProvider: GPU acceleration (future)
     *
     * Snapshot Capture:
     * - Automatically integrates with PipelineSnapshotManager
     * - Enabled via LLAMINAR_PARITY_CAPTURE=1 in debug builds
     * - Zero overhead in release builds (compiled out)
     * - Captures at standardized stages for cross-backend comparison
     *
     * Usage Example:
     * @code
     *   auto provider = PrefillProviderFactory::create(config, plan);
     *   PrefillMetrics metrics;
     *   bool success = provider->execute(tokens, weights, output, ctx, metrics);
     *   if (success) {
     *       LOG_INFO("Prefill: " << metrics.total_ms() << "ms, "
     *                << metrics.gflops() << " GFLOPS");
     *   }
     * @endcode
     */
    class PrefillProvider
    {
    public:
        virtual ~PrefillProvider() = default;

        /**
         * @brief Execute prefill for a batch of input tokens
         *
         * @param tokens Input token IDs to process
         * @param weights Model weights (architecture-specific, cast to concrete type)
         * @param output Output tensor for final hidden states or logits
         * @param ctx Stage context (sequence length, KV cache state, etc.)
         * @param metrics Output metrics for timing and FLOP counting
         * @param cache_provider Optional KV cache provider for populating cache during execution
         *
         * @return true if execution succeeded, false on error
         *
         * This method:
         * 1. Performs token embedding lookup
         * 2. Executes N transformer layers (attention + FFN)
         * 3. Applies final normalization
         * 4. Computes LM head output (if needed)
         * 5. Captures snapshots at each stage (if enabled)
         * 6. Populates metrics with timing and FLOP counts
         * 7. Populates KV cache via cache_provider (if provided)
         *
         * Cache Population:
         * - If cache_provider is non-null, implementations MUST populate it during layer execution
         * - Call cache_provider->setKCache(layer_idx, k_cache) after each attention layer
         * - Call cache_provider->setVCache(layer_idx, v_cache) after each attention layer
         * - Pipeline will retrieve cache after execute() completes for use in decode path
         *
         * Implementations should call captureSnapshot() at standardized stages
         * to enable parity testing against reference implementations.
         */
        virtual bool execute(
            const std::vector<int> &tokens,
            const IModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            StageContext &ctx,
            PrefillMetrics &metrics,
            KVCacheProvider *cache_provider = nullptr) = 0;

        /**
         * @brief Get provider name for logging and metrics
         * @return Provider identifier (e.g., "OpenBLAS", "COSMA", "GPU")
         */
        virtual std::string name() const = 0;

        /**
         * @brief Get model configuration
         * @return Reference to model configuration
         */
        const ModelConfig &config() const { return config_; }

        /**
         * @brief Get MPI context
         * @return Reference to MPI context (rank, size, communicator)
         */
        const MPIContext &mpiContext() const { return mpi_ctx_; }

    protected:
        /**
         * @brief Protected constructor (only derived classes can instantiate)
         *
         * @param config Model configuration including architecture and layer settings
         * @param mpi_ctx MPI context for distributed execution
         */
        PrefillProvider(const ModelConfig &config, const MPIContext &mpi_ctx)
            : config_(config), mpi_ctx_(mpi_ctx)
        {
        }

        /**
         * @brief Capture a snapshot of intermediate state for parity testing
         *
         * This method delegates to PipelineSnapshotManager, which:
         * - Only captures if LLAMINAR_PARITY_CAPTURE=1 in debug builds
         * - Is completely compiled out in release builds (zero overhead)
         * - Is thread-safe and MPI-aware (typically only rank 0 captures)
         *
         * Derived classes should call this at standardized stages:
         * - EMBEDDING: After token embedding lookup
         * - ATTENTION_NORM: After attention normalization (per layer)
         * - QKV_PROJECTION, Q_PROJECTION, K_PROJECTION, V_PROJECTION: After projections
         * - ROPE_APPLICATION: After RoPE applied to Q and K
         * - ATTENTION_SCORES: After Q @ K^T
         * - ATTENTION_SOFTMAX: After softmax
         * - ATTENTION_CONTEXT: After attention weights @ V
         * - ATTENTION_OUTPUT: After output projection W_o
         * - ATTENTION_RESIDUAL: After attention residual connection
         * - FFN_NORM: After FFN normalization
         * - FFN_GATE, FFN_UP: After gate/up projections
         * - FFN_SWIGLU: After SwiGLU activation
         * - FFN_DOWN: After down projection
         * - FFN_RESIDUAL: After FFN residual connection
         * - FINAL_NORM: After final normalization (post-layer-loop)
         * - LM_HEAD: After language model head output
         *
         * @param stage Pipeline stage identifier
         * @param layer_index Layer index (-1 for non-layer stages)
         * @param data Pointer to tensor data (float array)
         * @param seq_len Sequence length dimension
         * @param feature_dim Feature dimension (hidden size, vocab size, etc.)
         *
         * @note In release builds, this function is a no-op
         * @note Override only if you need custom snapshot behavior
         */
        void captureSnapshot(
            PipelineStage stage,
            int layer_index,
            const float *data,
            int seq_len,
            int feature_dim);

        /**
         * @brief Check if snapshot capture is enabled
         *
         * @return true if LLAMINAR_PARITY_CAPTURE=1 in debug builds, false otherwise
         *
         * Derived classes can use this to conditionally skip snapshot preparation
         * work when capture is disabled. However, calling captureSnapshot() is
         * already a no-op when disabled, so this is primarily for optimization.
         */
        bool isSnapshotEnabled() const;

        /**
         * @brief Increment snapshot counter (for metrics tracking)
         *
         * Call this in derived classes after each captureSnapshot() call
         * to track total snapshots for debugging and validation.
         */
        void incrementSnapshotCounter(PrefillMetrics &metrics);

    private:
        ModelConfig config_; ///< Model configuration
        MPIContext mpi_ctx_; ///< MPI context for distributed execution
    };

    /**
     * @brief Factory for creating PrefillProvider instances
     *
     * Selects the optimal provider based on:
     * - Model configuration (architecture, layer settings)
     * - Workload characteristics (sequence length, batch size)
     * - Environment variables (ADAPTIVE_DISABLE_COSMA, etc.)
     * - Hardware capabilities (MPI rank count, NUMA topology)
     *
     * Current selection logic:
     * - Small sequences (<4K tokens): OpenBLAS (avoids COSMA overhead)
     * - Large sequences (≥4K tokens): COSMA if multi-rank, else OpenBLAS
     * - ADAPTIVE_DISABLE_COSMA=1: Always OpenBLAS
     * - LLAMINAR_COSMA_FORCE_DIRECT=1: Always COSMA (for testing)
     */
    class PrefillProviderFactory
    {
    public:
        /**
         * @brief Create optimal provider for given configuration and workload
         *
         * @param config Model configuration
         * @param mpi_ctx MPI context
         * @param seq_len Sequence length (for workload-based selection)
         *
         * @return Unique pointer to created provider (never null)
         */
        static std::unique_ptr<PrefillProvider> create(
            const ModelConfig &config,
            const MPIContext &mpi_ctx,
            int seq_len);

        /**
         * @brief Create specific provider by name (for testing)
         *
         * @param provider_name "openblas", "cosma", or "gpu"
         * @param config Model configuration
         * @param mpi_ctx MPI context
         *
         * @return Unique pointer to created provider, or nullptr if name invalid
         */
        static std::unique_ptr<PrefillProvider> createByName(
            const std::string &provider_name,
            const ModelConfig &config,
            const MPIContext &mpi_ctx);
    };

} // namespace llaminar
