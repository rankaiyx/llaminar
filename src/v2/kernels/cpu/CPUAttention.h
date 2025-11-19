/**
 * @file CPUAttention.h
 * @brief CPU implementation of attention kernel
 *
 * Multi-threaded CPU attention using OpenMP.
 * Supports MHA, GQA, and MQA with optional causal masking.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../tensors/TensorKernels.h"
#include "../../tensors/Tensors.h"
#include <memory>
#include <vector>

namespace llaminar2
{

    /**
     * @brief CPU implementation of ITensorAttention
     *
     * Uses OpenMP for multi-threaded execution:
     * - Broadcast: Parallel across heads
     * - Score computation (Q·K^T): Parallel GEMM per head
     * - Softmax: Parallel across rows
     * - Context (scores·V): Parallel GEMM per head
     *
     * Workspace management:
     * - If workspaces provided: Uses caller-allocated buffers (reduces allocation overhead)
     * - If nullptr: Allocates internally (convenience mode)
     *
     * Thread safety:
     * - Uses per-thread buffers in workspace_buffer and workspace_context
     * - Thread-safe for concurrent calls with different workspaces
     * - NOT thread-safe for concurrent calls sharing workspaces
     */
    class CPUAttention : public ITensorAttention
    {
    public:
        CPUAttention() = default;
        ~CPUAttention() override = default;

        /**
         * @brief Check if kernel supports specific device
         *
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @return true if device_idx == -1 (CPU only)
         */
        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        /**
         * @brief Compute attention (see ITensorAttention::compute for documentation)
         */
        bool compute(
            const float *Q, const float *K, const float *V,
            float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        /**
         * @brief Batch attention computation (not yet implemented)
         */
        bool compute_batch(
            const float *Q, const float *K, const float *V,
            float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

    private:
        // Helper methods (implementation details from MpiAttentionOrchestrator)

        /**
         * @brief Broadcast K/V from n_kv_heads to n_heads (for GQA/MQA)
         *
         * For GQA: Each KV head is replicated to multiple query heads
         * For MQA: Single KV head is replicated to all query heads
         * For MHA: No-op (n_heads == n_kv_heads)
         *
         * @param input K or V tensor [seq_len, n_kv_heads, head_dim]
         * @param output Broadcasted tensor [seq_len, n_heads, head_dim]
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Dimension per head
         */
        void broadcast_kv(
            const float *input, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim) const;

        /**
         * @brief Extract single head from multi-head tensor
         *
         * @param multi_head Source tensor [seq_len, n_heads, head_dim]
         * @param single_head Output tensor [seq_len, head_dim]
         * @param head_idx Index of head to extract
         * @param seq_len Sequence length
         * @param n_heads Total number of heads
         * @param head_dim Dimension per head
         */
        void extract_head(
            const float *multi_head, float *single_head,
            int head_idx, int seq_len, int n_heads, int head_dim) const;

        /**
         * @brief Write single head back to multi-head tensor
         *
         * @param single_head Source tensor [seq_len, head_dim]
         * @param multi_head Output tensor [seq_len, n_heads, head_dim]
         * @param head_idx Index of head to write
         * @param seq_len Sequence length
         * @param n_heads Total number of heads
         * @param head_dim Dimension per head
         */
        void write_head(
            const float *single_head, float *multi_head,
            int head_idx, int seq_len, int n_heads, int head_dim) const;

        /**
         * @brief Compute attention scores: scores = Q · K^T
         *
         * @param Q_head Query for one head [seq_len, head_dim]
         * @param K_head Key for one head [seq_len, head_dim]
         * @param scores Output scores [seq_len, seq_len]
         * @param seq_len Sequence length
         * @param head_dim Dimension per head
         */
        void compute_scores(
            const float *Q_head, const float *K_head,
            float *scores,
            int seq_len, int head_dim) const;

        /**
         * @brief Scale attention scores: scores /= sqrt(head_dim)
         *
         * @param scores Attention scores [seq_len, seq_len] (modified in-place)
         * @param seq_len Sequence length
         * @param head_dim Dimension per head
         */
        void scale_scores(float *scores, int seq_len, int head_dim) const;

        /**
         * @brief Apply causal mask to attention scores
         *
         * Sets scores[i,j] = -inf for j > i (future tokens)
         *
         * @param scores Attention scores [seq_len, seq_len] (modified in-place)
         * @param seq_len Sequence length
         */
        void apply_causal_mask(float *scores, int seq_len) const;

        /**
         * @brief Apply softmax normalization: weights = softmax(scores)
         *
         * @param scores Input scores [seq_len, seq_len]
         * @param weights Output weights [seq_len, seq_len]
         * @param seq_len Sequence length
         */
        void apply_softmax(const float *scores, float *weights, int seq_len) const;

        /**
         * @brief Compute weighted context: context = weights · V
         *
         * @param weights Attention weights [seq_len, seq_len]
         * @param V_head Value for one head [seq_len, head_dim]
         * @param context Output context [seq_len, head_dim]
         * @param seq_len Sequence length
         * @param head_dim Dimension per head
         */
        void compute_context(
            const float *weights, const float *V_head,
            float *context,
            int seq_len, int head_dim) const;
    };

} // namespace llaminar2
