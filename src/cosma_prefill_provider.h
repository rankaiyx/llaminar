/**
 * @file cosma_prefill_provider.h
 * @brief COSMA-based distributed prefill provider
 * @author David Sanftenberg
 *
 * This provider implements prefill execution using COSMA for distributed matrix
 * multiplications. It's suitable for:
 * - Large sequence lengths (≥ 4K tokens)
 * - Multi-node distributed setups (2+ MPI ranks)
 * - Memory-bound workloads benefiting from distributed memory
 * - Peak throughput scenarios (can be 3.6x faster than OpenBLAS for large ops)
 *
 * Architecture:
 * - Leverages CosmaPrefillManager for distributed matmul coordination
 * - Uses fused RMSNorm + QKV projection for efficiency
 * - Applies optimized attention primitives (RoPE, scores, softmax)
 * - Falls back to adaptiveMatMul for FFN projections
 * - Captures snapshots at matching stages as OpenBLAS provider
 *
 * Key Differences from OpenBLAS:
 * - Attention: Fused norm+QKV via COSMA, then CPU attention primitives
 * - FFN: Uses adaptiveMatMul (may use COSMA for gate/up/down based on size)
 * - Memory: Distributed weight layout, higher communication overhead
 * - Performance: Better for large ops (≥64K tokens), worse for small (<4K)
 *
 * Snapshot Alignment:
 * - Captures at SAME stages as OpenBLASPrefillProvider for direct comparison
 * - Enables A/B testing: run both providers, compare snapshots stage-by-stage
 * - Identifies divergence source (COSMA matmul vs attention primitives vs etc.)
 */

#pragma once

#include "prefill_provider.h"
#include "qwen_pipeline.h"
#include "large_matmul_plan.h"
#include "cosma_prefill_manager.h"
#include <memory>
#include <vector>

namespace llaminar
{
    /**
     * @brief Timing breakdown for COSMA prefill attention
     *
     * Matches the structure used in QwenPipeline::executePrefillAttentionCosma
     */
    struct PrefillAttentionTiming
    {
        double norm_ms = 0.0;      ///< RMSNorm + QKV projection time
        double attention_ms = 0.0; ///< RoPE + attention computation time
        double linear_ms = 0.0;    ///< Output projection time
    };

    /**
     * @brief COSMA-based distributed prefill provider
     *
     * This provider extracts the COSMA prefill execution path from QwenPipeline.
     * It uses COSMA for distributed matrix multiplications and existing CPU
     * attention primitives for the attention computation.
     *
     * Key Features:
     * - Fused RMSNorm + QKV projection via CosmaPrefillManager
     * - Distributed weight streaming and layout management
     * - Optimized attention primitives (RoPE, scores, softmax)
     * - Comprehensive snapshot capture matching OpenBLAS provider
     * - Detailed timing metrics for each stage
     *
     * Execution Flow:
     * 1. Embedding: Token embedding lookup (host-side)
     * 2. For each layer:
     *    - Attention: Fused norm+QKV (COSMA) → RoPE → attention → output proj (adaptive)
     *    - FFN: norm → gate/up/down (adaptive matmul) → SwiGLU → residual
     * 3. Final: norm → LM head (adaptive matmul)
     *
     * Usage:
     * @code
     *   auto provider = std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
     *   PrefillMetrics metrics;
     *   bool success = provider->execute(tokens, weights, output, ctx, metrics);
     * @endcode
     */
    class COSMAPrefillProvider : public PrefillProvider
    {
    public:
        /**
         * @brief Construct COSMA prefill provider
         *
         * @param config Model configuration
         * @param mpi_ctx MPI context for distributed execution
         *
         * Note: Does not initialize kernels like OpenBLAS provider, as COSMA
         * uses CosmaPrefillManager singleton and adaptiveMatMul directly.
         */
        COSMAPrefillProvider(const ModelConfig &config, const MPIContext &mpi_ctx);

        /**
         * @brief Execute prefill for input tokens using COSMA
         *
         * @param tokens Input token IDs
         * @param weights Model weights (must be QwenPipeline::ModelWeights)
         * @param output Output tensor (final hidden states or logits)
         * @param ctx Stage context (sequence length, KV cache state)
         * @param metrics Output metrics (timing, FLOPs)
         *
         * @return true if execution succeeded, false on error
         *
         * Execution stages (with snapshot capture):
         * 1. Token embedding lookup (EMBEDDING)
         * 2. N transformer layers:
         *    - Attention: fused norm+QKV → RoPE → attention → output proj
         *    - FFN: norm → gate/up → SwiGLU → down → residual
         * 3. Final normalization (FINAL_NORM)
         * 4. LM head projection (LM_HEAD)
         *
         * Snapshots captured at SAME stages as OpenBLASPrefillProvider:
         * - EMBEDDING, ATTENTION_NORM, ATTENTION_OUTPUT, ATTENTION_RESIDUAL
         * - FFN_NORM, FFN_DOWN, FFN_RESIDUAL, FINAL_NORM, LM_HEAD
         */
        bool execute(
            const std::vector<int> &tokens,
            const IModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            StageContext &ctx,
            PrefillMetrics &metrics) override;

        /**
         * @brief Get provider name
         * @return "COSMA"
         */
        std::string name() const override { return "COSMA"; }

        /**
         * @brief Set sequence position for incremental decode
         *
         * @param n_past Number of tokens already processed (for RoPE position encoding)
         */
        void setSequencePosition(int n_past) { n_past_ = n_past; }

    private:
        /**
         * @brief Execute a single transformer layer using COSMA
         *
         * @param layer_idx Layer index (0 to n_layers-1)
         * @param input Input tensor (seq_len × d_model)
         * @param weights Model weights
         * @param output Output tensor (seq_len × d_model)
         * @param metrics Metrics to update with timing
         *
         * @return true if layer executed successfully
         *
         * Captures snapshots at:
         * - ATTENTION_NORM, ATTENTION_OUTPUT, ATTENTION_RESIDUAL
         * - FFN_NORM, FFN_DOWN, FFN_RESIDUAL
         *
         * Attention block:
         * 1. Fused RMSNorm + QKV projection (via CosmaPrefillManager)
         * 2. Materialize Q/K/V to row-major buffers
         * 3. Apply RoPE to Q and K
         * 4. Compute attention (scores, softmax, apply to V)
         * 5. Output projection (via adaptiveMatMul)
         * 6. Residual connection
         *
         * FFN block:
         * 1. RMSNorm (via kernel or adaptiveMatMul)
         * 2. Gate/Up projections (via adaptiveMatMul)
         * 3. SwiGLU activation (via kernel)
         * 4. Down projection (via adaptiveMatMul)
         * 5. Residual connection (via kernel)
         */
        bool executeTransformerLayer(
            int layer_idx,
            std::shared_ptr<TensorBase> &input,
            const QwenPipeline::ModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            PrefillMetrics &metrics);

        /**
         * @brief Execute COSMA-based attention for a single layer
         *
         * @param layer_idx Layer index
         * @param plan Large matmul plan with validated COSMA configuration
         * @param input Input tensor
         * @param weights Model weights
         * @param attn_norm_out Output of attention normalization
         * @param attn_out Output of attention (after output projection)
         * @param timing Timing breakdown to populate
         *
         * @return true if attention succeeded
         *
         * This method is extracted from QwenPipeline::executePrefillAttentionCosma
         * Uses CosmaPrefillManager for fused norm+QKV, then CPU attention primitives.
         */
        bool executeAttentionCosma(
            int layer_idx,
            const LargeMatmulPlan &plan,
            std::shared_ptr<TensorBase> &input,
            const QwenPipeline::ModelWeights &weights,
            std::shared_ptr<TensorBase> &attn_norm_out,
            std::shared_ptr<TensorBase> &attn_out,
            PrefillAttentionTiming &timing);

        /**
         * @brief Create local tensor with NUMA-aware allocation
         *
         * @param shape Tensor shape
         * @return Shared pointer to created tensor
         */
        std::shared_ptr<TensorBase> createLocalTensor(const std::vector<int> &shape);

        /**
         * @brief Execute kernel by name (for SwiGLU and residual)
         *
         * @param kernel_name Kernel name ("swiglu" or "residual")
         * @param inputs Input tensors
         * @param outputs Output tensors
         * @return true if kernel executed successfully
         *
         * Note: Only minimal kernels needed (SwiGLU, residual)
         * Most operations go through COSMA or adaptiveMatMul
         */
        bool executeKernel(
            const std::string &kernel_name,
            const std::vector<std::shared_ptr<TensorBase>> &inputs,
            std::vector<std::shared_ptr<TensorBase>> &outputs);

        /**
         * @brief Initialize minimal kernel set
         *
         * Registers only kernels not covered by COSMA/adaptiveMatMul:
         * - swiglu: SwiGLU activation
         * - residual: Residual connections
         * - rmsnorm: RMSNorm (fallback if COSMA fused path fails)
         */
        void initializeKernels();

        /**
         * @brief Get kernel by name
         * @param name Kernel name
         * @return Pointer to kernel or nullptr
         */
        MPIKernelBase *getKernel(const std::string &name);

        /**
         * @brief Register kernel in internal registry
         * @param name Kernel name
         * @param kernel Kernel instance
         * @return true if registration succeeded
         */
        bool registerKernel(const std::string &name, std::unique_ptr<MPIKernelBase> kernel);

        // Minimal kernel registry (only SwiGLU and residual needed)
        std::unordered_map<std::string, std::unique_ptr<MPIKernelBase>> kernels_;

        // Sequence position for RoPE
        int n_past_ = 0;
    };

} // namespace llaminar
