/**
 * @file PipelineBase.h
 * @brief Base class for transformer pipelines
 *
 * Provides common infrastructure for all model architectures:
 * - MPI context management
 * - Device placement
 * - Weight and activation management
 * - Common pipeline operations
 *
 * Derived classes implement architecture-specific details:
 * - Qwen2Pipeline, Qwen3Pipeline, Qwen3MoEPipeline, etc.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../utils/MPIContext.h"
#include "../backends/ComputeBackend.h"
#include "../loaders/ModelContext.h"
#include "../tensors/Tensors.h"
#include "../tensors/TensorKernels.h"
#include <vector>
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Base class for transformer pipelines
     *
     * Provides common infrastructure for model execution.
     * Derived classes implement architecture-specific logic.
     */
    class PipelineBase
    {
    public:
        /**
         * @brief Construct pipeline base
         *
         * @param model_ctx Model context with GGUF metadata and loader
         * @param mpi_ctx MPI context for distributed execution (nullptr = single node)
         * @param device_idx Default device for tensors (-1 = CPU, ≥0 = GPU device)
         */
        PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                     std::shared_ptr<MPIContext> mpi_ctx = nullptr,
                     int device_idx = -1);

        virtual ~PipelineBase() = default;

        /**
         * @brief Forward pass (prefill or decode)
         *
         * @param tokens Token IDs [seq_len]
         * @param seq_len Number of tokens
         * @return true on success, false on error
         */
        virtual bool forward(const int *tokens, int seq_len) = 0;

        /**
         * @brief Get output logits (FP32)
         *
         * @return Logits tensor [seq_len, vocab_size], or nullptr if not available
         */
        virtual const float *logits() const = 0;

        /**
         * @brief Get model architecture name
         *
         * @return Architecture string (e.g., "qwen2", "qwen3", "qwen3-moe")
         */
        virtual const char *architecture() const = 0;

        /**
         * @brief Get model context
         *
         * @return Model context with metadata and loader
         */
        std::shared_ptr<ModelContext> model_context() const { return model_ctx_; }

        /**
         * @brief Get MPI context
         *
         * @return MPI context pointer, or nullptr if not using MPI
         */
        std::shared_ptr<MPIContext> mpi_context() const { return mpi_ctx_; }

        /**
         * @brief Get default device index
         *
         * @return Device index (-1 = CPU, ≥0 = GPU device)
         */
        int device_index() const { return device_idx_; }

    protected:
        // Context management
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<MPIContext> mpi_ctx_;
        int device_idx_; // Default device (-1 = CPU)

        // Model path for convenience (from model_ctx_)
        std::string model_path_;

        // Common model parameters (set by derived classes)
        int n_layers_ = 0;
        int d_model_ = 0;
        int vocab_size_ = 0;

        /**
         * @brief Process a single transformer layer
         *
         * To be implemented by derived classes.
         *
         * @param layer_idx Layer index (0-indexed)
         * @param seq_len Sequence length
         * @return true on success, false on error
         */
        virtual bool transformer_layer(int layer_idx, int seq_len) = 0;

        /**
         * @brief Standard GQA (Grouped Query Attention) orchestration
         *
         * Default attention implementation supporting:
         * - GQA: n_heads > n_kv_heads (broadcast K/V heads)
         * - MHA: n_heads == n_kv_heads (no broadcasting)
         * - MQA: n_kv_heads == 1 (broadcast single K/V to all Q heads)
         * - Sliding window: Optional local attention window
         *
         * Handles ~95% of production models (Qwen, Llama, Mistral, Gemma, etc.).
         * Pipelines with custom attention (e.g., DeepSeek MLA) override attention_block().
         *
         * Algorithm:
         * 1. Broadcast K/V heads to match Q heads (if n_kv_heads < n_heads)
         * 2. Compute attention scores: Q @ K^T (per-head batched GEMM)
         * 3. Scale by 1/sqrt(head_dim)
         * 4. Apply causal mask (optional sliding window)
         * 5. Softmax over scores
         * 6. Compute context: scores @ V (per-head batched GEMM)
         * 7. Concatenate heads back to [seq_len, n_heads * head_dim]
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim]
         * @param K Key tensor [seq_len, n_kv_heads * head_dim]
         * @param V Value tensor [seq_len, n_kv_heads * head_dim]
         * @param output Output tensor [seq_len, n_heads * head_dim] (pre-allocated)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (GQA: ≤ n_heads)
         * @param head_dim Dimension per head
         * @param causal Apply causal masking for autoregressive generation
         * @param window_size Sliding window size (-1 = full attention, ≥0 = local window)
         * @return true on success, false on error
         *
         * @note Uses primitive kernels: ITensorGemm, ITensorSoftmax
         * @note Pipelines can override attention_block() for custom attention types
         */
        virtual bool attention_gqa(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = true, int window_size = -1);
    };

} // namespace llaminar2
