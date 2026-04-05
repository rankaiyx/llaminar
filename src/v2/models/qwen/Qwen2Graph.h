/**
 * @file Qwen2Graph.h
 * @brief Qwen2 compute graph builder for standard multi-head attention
 * @author David Sanftenberg
 * @date December 2025
 *
 * Qwen2Graph implements the Qwen2-specific attention pattern:
 *   - Standard Q/K/V projections with optional biases
 *   - Optional QK normalization (Qwen3)
 *   - RoPE position embeddings
 *   - Multi-head/GQA attention with KV cache
 *   - Output projection (Wo) with optional TP allreduce
 *
 * Inherits shared infrastructure from QwenGraphBase:
 *   IGraphBuilder → QwenGraphBase → Qwen2Graph
 *
 * Also serves Qwen3 (which reuses Qwen2Graph with Qwen3SchemaFactory).
 */

#pragma once

#include "QwenGraphBase.h"
#include "Qwen2Schema.h"
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Qwen2 compute graph builder
     *
     * Implements standard multi-head attention for Qwen2/Qwen3 models.
     * Inherits all shared transformer infrastructure from QwenGraphBase.
     */
    class Qwen2Graph : public QwenGraphBase
    {
    public:
        /**
         * @brief Construct Qwen2Graph with full context
         *
         * @param model_ctx Model context with GGUF metadata
         * @param mpi_ctx MPI context (nullptr for single-rank)
         * @param config Graph configuration
         */
        Qwen2Graph(std::shared_ptr<ModelContext> model_ctx,
                   std::shared_ptr<MPIContext> mpi_ctx,
                   const GraphConfig &config);

        /**
         * @brief Construct Qwen2Graph for layer-level operations only
         *
         * @param config Graph configuration
         * @param mpi_ctx MPI context (nullptr for single-rank)
         */
        Qwen2Graph(const GraphConfig &config,
                   std::shared_ptr<MPIContext> mpi_ctx = nullptr);

        ~Qwen2Graph() = default;

        // Non-copyable
        Qwen2Graph(const Qwen2Graph &) = delete;
        Qwen2Graph &operator=(const Qwen2Graph &) = delete;

        // =====================================================================
        // Model-Specific Overrides
        // =====================================================================

        std::string architectureName() const override { return "qwen2"; }

        GraphSchema getSchema() const override;

        /**
         * @brief Build Qwen2 attention block graph
         *
         * Standard attention: norm → QKV projection → QK norm (Qwen3) → RoPE
         *                     → KV cache → attention → Wo → TP allreduce → residual
         */
        ComputeGraph buildAttentionGraph(
            const LayerWeights &layer,
            ActivationBuffers &buffers,
            int layer_idx,
            int seq_len,
            int batch_size,
            IKVCache *kv_cache,
            const int *position_ids,
            DeviceId device,
            const std::vector<int> *sequence_lengths = nullptr) override;
    };

} // namespace llaminar2
