/**
 * @file openblas_prefill_provider.h
 * @brief OpenBLAS-based prefill provider (CPU, non-distributed matmuls)
 * @author David Sanftenberg
 *
 * This provider implements prefill execution using OpenBLAS for matrix multiplications
 * and existing MPI kernels for other operations. It represents the "baseline" execution
 * path suitable for:
 * - Small to medium sequence lengths (< 4K tokens)
 * - Single-node or small multi-node setups
 * - Development and debugging (well-tested, predictable behavior)
 * - Reference implementation for parity testing
 *
 * Architecture:
 * - Uses existing MPI kernel infrastructure (MPIRMSNormKernel, MPIAttentionKernel, etc.)
 * - Delegates matmuls to OpenBLAS (single-threaded or multi-threaded based on size)
 * - Captures snapshots at all standardized stages for PyTorch comparison
 * - Integrates with KV cache management
 *
 * Stage Flow:
 * 1. EMBEDDING: Token embedding lookup
 * 2. For each layer:
 *    - ATTENTION_NORM: RMSNorm before attention
 *    - Attention (via MPIAttentionKernel): Q/K/V proj, RoPE, scores, softmax, context, output proj
 *    - ATTENTION_OUTPUT: After output projection
 *    - ATTENTION_RESIDUAL: After residual connection
 *    - FFN_NORM: RMSNorm before FFN
 *    - FFN_GATE, FFN_UP: Linear projections
 *    - FFN_SWIGLU: SwiGLU activation
 *    - FFN_DOWN: Down projection
 *    - FFN_RESIDUAL: After FFN residual connection
 * 3. FINAL_NORM: RMSNorm after all layers
 * 4. LM_HEAD: Language model head projection
 */

#pragma once

#include "prefill_provider.h"
#include "qwen_pipeline.h" // For ModelWeights
#include "pipeline_base.h"
#include <memory>
#include <vector>

namespace llaminar
{
    /**
     * @brief OpenBLAS-based prefill provider
     *
     * This provider extracts the non-COSMA prefill execution path from QwenPipeline.
     * It uses OpenBLAS for matrix multiplications and existing MPI kernels for
     * other operations (attention, normalization, activations, etc.).
     *
     * Key Features:
     * - Kernel-based architecture (composition over implementation)
     * - Comprehensive snapshot capture at all stages
     * - KV cache integration
     * - Detailed timing metrics per stage
     * - MPI-aware execution (sequence-wise distribution)
     *
     * Usage:
     * @code
     *   auto provider = std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
     *   PrefillMetrics metrics;
     *   bool success = provider->execute(tokens, weights, output, ctx, metrics);
     * @endcode
     */
    class OpenBLASPrefillProvider : public PrefillProvider
    {
    public:
        /**
         * @brief Construct OpenBLAS prefill provider
         *
         * @param config Model configuration
         * @param mpi_ctx MPI context for distributed execution
         *
         * Initializes all required kernels:
         * - embedding: MPIEmbeddingKernel
         * - rmsnorm: MPIRMSNormKernel (sequence-wise distribution)
         * - attention: MPIAttentionKernel (handles Q/K/V/O projections internally)
         * - linear: MPILinearKernel (for FFN projections)
         * - swiglu: MPISwiGLUKernel (SwiGLU activation)
         * - residual: MPIResidualKernel (residual connections)
         */
        OpenBLASPrefillProvider(const ModelConfig &config, const MPIContext &mpi_ctx);

        /**
         * @brief Execute prefill for input tokens
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
         *    - Attention: norm → Q/K/V/RoPE/scores/softmax/context/output → residual
         *    - FFN: norm → gate/up → SwiGLU → down → residual
         * 3. Final normalization (FINAL_NORM)
         * 4. LM head projection (LM_HEAD)
         */
        bool execute(
            const std::vector<int> &tokens,
            const IModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            StageContext &ctx,
            PrefillMetrics &metrics) override;

        /**
         * @brief Get provider name
         * @return "OpenBLAS"
         */
        std::string name() const override { return "OpenBLAS"; }

        /**
         * @brief Set KV cache tensors for attention
         *
         * @param k_cache Vector of key cache tensors (one per layer)
         * @param v_cache Vector of value cache tensors (one per layer)
         *
         * Call this before execute() if you want to use KV caching.
         * If not set, temporary cache tensors will be allocated.
         */
        void setKVCache(
            const std::vector<std::shared_ptr<TensorBase>> &k_cache,
            const std::vector<std::shared_ptr<TensorBase>> &v_cache);

        /**
         * @brief Set sequence position for incremental decode
         *
         * @param n_past Number of tokens already processed (for RoPE position encoding)
         *
         * Set to 0 for prefill, >0 for incremental decode context.
         */
        void setSequencePosition(int n_past) { n_past_ = n_past; }

    private:
        /**
         * @brief Initialize all required kernels
         *
         * Registers kernels in internal registry:
         * - embedding, rmsnorm, attention, linear, swiglu, residual
         */
        void initializeKernels();

        /**
         * @brief Execute a single transformer layer
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
         */
        bool executeTransformerLayer(
            int layer_idx,
            std::shared_ptr<TensorBase> &input,
            const QwenPipeline::ModelWeights &weights,
            std::shared_ptr<TensorBase> &output,
            PrefillMetrics &metrics);

        /**
         * @brief Execute kernel by name
         *
         * @param kernel_name Name of registered kernel
         * @param inputs Input tensors
         * @param outputs Output tensors
         *
         * @return true if kernel execution succeeded
         *
         * Wrapper around kernel registry for cleaner code.
         */
        bool executeKernel(
            const std::string &kernel_name,
            const std::vector<std::shared_ptr<TensorBase>> &inputs,
            std::vector<std::shared_ptr<TensorBase>> &outputs);

        /**
         * @brief Get kernel by name from registry
         *
         * @param name Kernel name
         * @return Pointer to kernel, or nullptr if not found
         */
        MPIKernelBase *getKernel(const std::string &name);

        /**
         * @brief Register a kernel in internal registry
         *
         * @param name Unique kernel name
         * @param kernel Kernel instance
         * @return true if registration succeeded
         */
        bool registerKernel(const std::string &name, std::unique_ptr<MPIKernelBase> kernel);

        /**
         * @brief Create local tensor with NUMA-aware allocation
         *
         * @param shape Tensor shape
         * @return Shared pointer to created tensor
         */
        std::shared_ptr<TensorBase> createLocalTensor(const std::vector<int> &shape);

        // Kernel registry
        std::unordered_map<std::string, std::unique_ptr<MPIKernelBase>> kernels_;

        // KV cache state
        std::vector<std::shared_ptr<TensorBase>> k_cache_;
        std::vector<std::shared_ptr<TensorBase>> v_cache_;
        bool use_kv_cache_ = false;
        int n_past_ = 0; // Sequence position for RoPE
    };

} // namespace llaminar
