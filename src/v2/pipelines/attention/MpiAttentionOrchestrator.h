/**
 * @file MpiAttentionOrchestrator.h
 * @brief Distributed attention orchestrator with GQA/MHA support
 *
 * Provides reusable GQA attention logic extracted from PipelineBase.
 * Supports:
 * - MHA (Multi-Head Attention): n_heads == n_kv_heads
 * - GQA (Grouped Query Attention): n_heads > n_kv_heads
 * - MQA (Multi-Query Attention): n_kv_heads == 1
 * - Single-sequence and batched execution
 * - MPI-aware tensor-parallel execution
 * - Causal masking and sliding window attention
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../utils/MPIContext.h"
#include "../../tensors/Tensors.h"
#include "../MPIStrategy.h"
#include "../PipelineConfig.h"
#include <vector>
#include <memory>

namespace llaminar2
{

    /**
     * @brief Distributed attention configuration
     *
     * Encapsulates all parameters needed for attention computation.
     */
    struct MpiAttentionConfig
    {
        int n_heads;     // Number of query heads
        int n_kv_heads;  // Number of key/value heads (GQA: ≤ n_heads)
        int head_dim;    // Dimension per head
        bool causal;     // Apply causal masking for autoregressive generation
        int window_size; // Sliding window size (-1 = full attention)

        // Explicit sequence length (required when Q tensor is pre-allocated for max_seq_len)
        // If > 0, this is used instead of inferring from Q tensor shape.
        // If <= 0, seq_len is inferred from Q->shape()[0] / batch_size (legacy behavior)
        int seq_len; // Explicit sequence length (number of query tokens per batch)

        // Activation precision for attention operations
        ActivationPrecision precision; // FP32, BF16, FP16 (controls GEMM precision)

        // MPI configuration
        std::shared_ptr<MPIContext> mpi_ctx;
        MPIStrategy mpi_strategy;
        bool verbose_logging; // Enable verbose MPI logging

        // Workspace buffers (caller-allocated, reused across calls)
        // Eliminates per-call allocations in hot path
        std::shared_ptr<TensorBase> workspace_scores;     // [max_heads * max_seq, max_seq]
        std::shared_ptr<TensorBase> workspace_qkv_buffer; // [max_seq * head_dim * 3] per-head extraction
        std::shared_ptr<TensorBase> workspace_context;    // [max_seq * head_dim] per-head context
        std::shared_ptr<TensorBase> workspace_mask;       // [max_seq * max_seq] causal/padding mask

        // Constructor with defaults
        MpiAttentionConfig()
            : n_heads(0), n_kv_heads(0), head_dim(0),
              causal(true), window_size(-1),
              seq_len(0), // 0 = infer from Q tensor shape
              precision(ActivationPrecision::FP32),
              mpi_ctx(nullptr), mpi_strategy(MPIStrategy::None),
              verbose_logging(false),
              workspace_scores(nullptr), workspace_qkv_buffer(nullptr),
              workspace_context(nullptr), workspace_mask(nullptr)
        {
        }
    };

    /**
     * @brief MPI-aware attention orchestrator
     *
     * Provides static methods for GQA/MHA attention computation.
     * Can be used by any pipeline without coupling to PipelineBase.
     *
     * Design:
     * - Static methods (no state) for easy reuse
     * - Config object encapsulates parameters
     * - Automatic MPI dispatch based on config
     *
     * @deprecated Phase 9: Use KVCacheAppendStage + AttentionComputeStage instead.
     *             This class will be removed in v3.0. See MULTI_DEVICE_ARCHITECTURE.md.
     */
    class [[deprecated("Use KVCacheAppendStage + AttentionComputeStage for new code")]] MpiAttentionOrchestrator
    {
    public:
        /**
         * @brief Standard GQA attention (single sequence or batch)
         *
         * Default attention implementation supporting:
         * - GQA: n_heads > n_kv_heads (broadcast K/V heads)
         * - MHA: n_heads == n_kv_heads (no broadcasting)
         * - MQA: n_kv_heads == 1 (broadcast single K/V to all Q heads)
         * - Sliding window: Optional local attention window
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
         * @param config Attention configuration
         * @param batch_size Number of sequences in batch (default=1)
         * @param sequence_lengths Actual lengths per sequence for padding mask (nullptr = no padding)
         * @return true on success, false on error
         *
         * @note Single-rank implementation (no MPI coordination)
         * @note For MPI parallelization, use compute_mpi()
         */
        static bool compute(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            const MpiAttentionConfig &config,
            int batch_size = 1,
            const std::vector<int> *sequence_lengths = nullptr);

        /**
         * @brief Batched GQA attention with padding support
         *
         * Computes attention for multiple sequences simultaneously:
         *   For each batch b:
         *     attention(Q[b], K[b], V[b]) = softmax(Q·K^T / sqrt(head_dim) + mask[b]) · V
         *
         * Masking:
         * - Causal mask: Token i cannot attend to tokens j > i (if causal=true)
         * - Padding mask: Real tokens cannot attend to padding positions
         * - Combined mask applied before softmax
         *
         * Input shapes:
         * - Q: [batch_size * seq_len, n_heads * head_dim]
         * - K: [batch_size * seq_len, n_kv_heads * head_dim]
         * - V: [batch_size * seq_len, n_kv_heads * head_dim]
         * - output: [batch_size * seq_len, n_heads * head_dim]
         * - actual_lengths: [batch_size] (actual sequence lengths, not padded)
         *
         * @param Q Query tensor for all batches (flattened)
         * @param K Key tensor for all batches (flattened)
         * @param V Value tensor for all batches (flattened)
         * @param output Output tensor for all batches (pre-allocated, flattened)
         * @param actual_lengths Actual sequence lengths (before padding) [batch_size]
         * @param batch_size Number of sequences
         * @param seq_len Maximum sequence length (after padding)
         * @param config Attention configuration
         * @return true on success, false on error
         *
         * @note Single-rank implementation (no MPI coordination yet)
         * @note For MPI parallelization, extend with compute_batch_mpi()
         */
        static bool compute_batch(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            const std::vector<int> &actual_lengths,
            int batch_size, int seq_len,
            const MpiAttentionConfig &config);

        /**
         * @brief MPI-aware attention dispatcher
         *
         * Dispatches to appropriate attention implementation based on MPI strategy:
         * - MPIStrategy::None → compute() (single-rank fast path)
         * - MPIStrategy::TensorParallel → compute_tensor_parallel()
         * - MPIStrategy::SequenceParallel → compute_sequence_parallel() (TODO)
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim]
         * @param K Key tensor [seq_len, n_kv_heads * head_dim]
         * @param V Value tensor [seq_len, n_kv_heads * head_dim]
         * @param output Output tensor [seq_len, n_heads * head_dim] (pre-allocated)
         * @param config Attention configuration (must have mpi_ctx and mpi_strategy set)
         * @param batch_size Number of sequences in batch (default=1)
         * @param sequence_lengths Actual lengths per sequence for padding mask (nullptr = no padding)
         * @return true on success, false on error
         */
        static bool compute_mpi(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            const MpiAttentionConfig &config,
            int batch_size = 1,
            const std::vector<int> *sequence_lengths = nullptr);

        /**
         * @brief Tensor-parallel attention implementation
         *
         * Distributes attention heads across MPI ranks:
         * - Rank i computes heads [start_head, start_head + local_n_heads)
         * - Allreduce to sum outputs from all ranks
         *
         * Memory: O(seq_len * local_n_heads * head_dim) per rank
         * Communication: 1× allreduce (seq_len * n_heads * head_dim elements)
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim]
         * @param K Key tensor [seq_len, n_kv_heads * head_dim]
         * @param V Value tensor [seq_len, n_kv_heads * head_dim]
         * @param output Output tensor [seq_len, n_heads * head_dim] (pre-allocated)
         * @param config Attention configuration (must have mpi_ctx set)
         * @param batch_size Number of sequences in batch (default=1)
         * @param sequence_lengths Actual lengths per sequence for padding mask (nullptr = no padding)
         * @return true on success, false on error
         *
         * @note Requires n_heads % world_size == 0
         */
        static bool compute_tensor_parallel(
            TensorBase *Q, TensorBase *K, TensorBase *V, TensorBase *output,
            const MpiAttentionConfig &config,
            int batch_size = 1,
            const std::vector<int> *sequence_lengths = nullptr);

        // ========================================================================
        // Helper Functions (Public for Testing)
        // ========================================================================

        /**
         * @brief Validate attention inputs
         *
         * Checks:
         * - Non-null tensor pointers
         * - Valid head configuration (n_heads % n_kv_heads == 0)
         * - Compatible tensor shapes
         *
         * @return true if inputs valid, false otherwise
         */
        static bool validate_inputs(
            const TensorBase *Q, const TensorBase *K, const TensorBase *V,
            const TensorBase *output, const MpiAttentionConfig &config);

        /**
         * @brief Broadcast K/V heads to match Q heads (GQA)
         *
         * When n_kv_heads < n_heads (GQA or MQA):
         * - Each K/V head is replicated to (n_heads / n_kv_heads) query heads
         * - Example: 32 Q heads, 8 KV heads → each KV head used by 4 Q heads
         *
         * @param K_in Input K data [seq_len, n_kv_heads * head_dim]
         * @param V_in Input V data [seq_len, n_kv_heads * head_dim]
         * @param K_out Output K buffer [seq_len, n_heads * head_dim] (will resize)
         * @param V_out Output V buffer [seq_len, n_heads * head_dim] (will resize)
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         */
        static void broadcast_kv_heads_if_needed(
            const float *K_in, const float *V_in,
            std::vector<float> &K_out, std::vector<float> &V_out,
            int seq_len, int n_heads, int n_kv_heads, int head_dim);

        /**
         * @brief Extract single head from strided multi-head tensor
         *
         * Converts interleaved multi-head layout to contiguous single-head layout:
         *   Input:  [seq_len, n_heads * head_dim] (strided by n_heads)
         *   Output: [seq_len, head_dim] (contiguous for this head)
         *
         * @param strided_data Input tensor with interleaved heads
         * @param contiguous_out Output buffer for single head (must be pre-allocated)
         * @param seq_len Sequence length
         * @param head_dim Dimension per head
         * @param n_heads Total number of heads in input
         * @param head_idx Index of head to extract (0-based)
         * @param batch_offset Offset for batched tensors (in tokens)
         */
        static void extract_head_data(
            const float *strided_data, float *contiguous_out,
            int seq_len, int head_dim, int n_heads, int head_idx,
            int batch_offset = 0);

        /**
         * @brief Compute attention scores: Q @ K^T
         *
         * GEMM: scores = Q @ K^T
         *   Q: [seq_len, head_dim]
         *   K: [seq_len, head_dim]
         *   scores: [seq_len, seq_len]
         *
         * @param Q Query matrix (contiguous)
         * @param K Key matrix (contiguous)
         * @param scores Output scores matrix (pre-allocated)
         * @param seq_len Sequence length
         * @param head_dim Dimension per head
         * @param precision Activation precision (FP32, BF16, FP16)
         * @return true on success, false if GEMM fails
         */
        static bool compute_attention_scores(
            const float *Q, const float *K, float *scores,
            int seq_len, int head_dim, ActivationPrecision precision = ActivationPrecision::FP32);

        /**
         * @brief Scale attention scores by 1/sqrt(head_dim)
         *
         * Prevents softmax saturation for large head dimensions.
         * scores[i] *= 1.0 / sqrt(head_dim)
         *
         * @param scores Scores array to scale in-place
         * @param size Total number of elements
         * @param head_dim Dimension per head (for computing scale factor)
         */
        static void scale_scores_inplace(
            float *scores, int size, int head_dim);

        /**
         * @brief Create and apply attention mask
         *
         * Applies causal mask (autoregressive) and/or padding mask:
         * - Causal: scores[i][j] = -inf if j > i (future positions)
         * - Padding: scores[i][j] = -inf if j >= actual_len (padding tokens)
         * - Sliding window: scores[i][j] = -inf if j < i - window_size
         *
         * @param scores Scores array to mask in-place
         * @param seq_len Sequence length (per-batch)
         * @param batch_size Number of sequences (for batched attention)
         * @param seq_lengths Actual lengths per sequence (nullptr = no padding mask)
         * @param causal Apply causal mask
         * @param window_size Sliding window size (-1 = full attention)
         * @param config Full config (provides workspace_mask buffer)
         */
        static void apply_attention_mask(
            float *scores, int seq_len, int batch_size,
            const int *seq_lengths, bool causal, int window_size,
            const MpiAttentionConfig &config);

        /**
         * @brief Apply row-wise softmax to attention scores
         *
         * For each row i: scores[i] = softmax(scores[i])
         * Uses vectorized primitives (AVX512/AVX2/scalar fallback)
         *
         * @param scores Scores array to apply softmax in-place
         * @param rows Number of rows (typically seq_len or batch_size * seq_len)
         * @param cols Number of columns (typically seq_len)
         */
        static void apply_softmax(
            float *scores, int rows, int cols);

        /**
         * @brief Compute attention context: scores @ V
         *
         * GEMM: context = scores @ V
         *   scores: [seq_len, seq_len]
         *   V: [seq_len, head_dim]
         *   context: [seq_len, head_dim]
         *
         * @param scores Attention weights (post-softmax)
         * @param V Value matrix (contiguous)
         * @param context Output context matrix (pre-allocated)
         * @param seq_len Sequence length
         * @param head_dim Dimension per head
         * @param precision Activation precision (FP32, BF16, FP16)
         * @return true on success, false if GEMM fails
         */
        static bool compute_context_from_scores(
            const float *scores, const float *V, float *context,
            int seq_len, int head_dim, ActivationPrecision precision = ActivationPrecision::FP32);

        /**
         * @brief Write context to strided multi-head output
         *
         * Converts contiguous single-head context to interleaved multi-head output:
         *   Input:  [seq_len, head_dim] (contiguous for this head)
         *   Output: [seq_len, n_heads * head_dim] (strided by n_heads)
         *
         * @param context Input context for single head
         * @param output Output tensor with interleaved heads
         * @param seq_len Sequence length
         * @param head_dim Dimension per head
         * @param n_heads Total number of heads in output
         * @param head_idx Index of head to write (0-based)
         * @param batch_offset Offset for batched tensors (in tokens)
         */
        static void write_context_to_output(
            const float *context, float *output,
            int seq_len, int head_dim, int n_heads, int head_idx,
            int batch_offset = 0);

    private:
        // Disable instantiation (static-only class)
        MpiAttentionOrchestrator() = delete;
        ~MpiAttentionOrchestrator() = delete;
        MpiAttentionOrchestrator(const MpiAttentionOrchestrator &) = delete;
        MpiAttentionOrchestrator &operator=(const MpiAttentionOrchestrator &) = delete;
    };

} // namespace llaminar2
