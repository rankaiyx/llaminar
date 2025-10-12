#pragma once

/**
 * @file model_weights_provider.h
 * @brief MPI-aware model weights accessor with structured access patterns
 * @author David Sanftenberg
 *
 * Provides a clean interface for accessing model weights with:
 * - Type-safe getters for all weight categories
 * - MPI rank awareness for sliced weight metadata
 * - Separation of concerns (serving vs loading vs verification)
 * - Backward compatibility with existing ModelWeights struct
 *
 * Design Philosophy:
 * - Provider OWNS weights (via unique_ptr)
 * - Provider SERVES weights (via const shared_ptr getters)
 * - Provider DOCUMENTS slicing (via metadata methods)
 * - Provider does NOT verify or load weights (separate concerns)
 *
 * Usage:
 * @code
 * auto loader = std::make_unique<ModelLoader>();
 * loader->loadModel("model.gguf");
 * auto weights = loadModelWeights_impl_bridge(*loader, config);
 *
 * auto provider = std::make_unique<QwenModelWeightsProvider>(
 *     std::make_unique<QwenPipeline::ModelWeights>(std::move(weights)),
 *     mpi_ctx, config
 * );
 *
 * // Type-safe access
 * auto k_weight = provider->getKeyWeight(0);  // Layer 0 K weight
 *
 * // MPI metadata for verification
 * if (provider->isWeightSliced("K")) {
 *     auto [offset, count] = provider->getLocalSliceInfo("K");
 *     LOG_INFO("Rank " << provider->getRank()
 *              << " has K heads " << offset << "-" << (offset+count));
 * }
 * @endcode
 */

#include <memory>
#include <string>
#include <utility>
#include "qwen_pipeline.h"
#include "mpi_context.h"
#include "transformer_config.h"
#include "tensors/tensor_base.h"

namespace llaminar
{

    /**
     * @class QwenModelWeightsProvider
     * @brief MPI-aware accessor for Qwen model weights
     *
     * Wraps QwenPipeline::ModelWeights to provide:
     * - Structured access via named getters
     * - MPI slicing metadata for verification
     * - Bounds checking for layer indices
     * - Backward compatibility via rawWeights()
     *
     * **Weight Slicing Behavior:**
     *
     * ModelLoader automatically slices certain weights across MPI ranks:
     * - W_Q, W_K, W_V: Column-sliced by attention heads
     * - W_GATE, W_UP: Column-sliced by FFN hidden dimension
     * - W_DOWN: Row-sliced (transposed column partitioning)
     * - Embedding, W_O, norms: Replicated (not sliced)
     *
     * Getters return the LOCAL slice stored on this rank.
     * Use `isWeightSliced()` and `getLocalSliceInfo()` to determine
     * which portion of the full weight is stored locally.
     *
     * @see ModelLoader for weight slicing implementation details
     */
    class QwenModelWeightsProvider
    {
    public:
        /**
         * @brief Construct provider with ownership of weights
         * @param weights Unique pointer to loaded weights (provider takes ownership)
         * @param mpi_ctx MPI context (rank, world_size, communicator)
         * @param config Transformer configuration (layer counts, dimensions)
         *
         * @note Provider takes ownership of weights and will manage their lifecycle
         * @throws std::invalid_argument if weights is nullptr
         */
        QwenModelWeightsProvider(
            std::unique_ptr<QwenPipeline::ModelWeights> weights,
            const MPIContext &mpi_ctx,
            const TransformerLayerConfig &config);

        /**
         * @brief Default destructor
         */
        ~QwenModelWeightsProvider() = default;

        // Disable copy (owns unique_ptr)
        QwenModelWeightsProvider(const QwenModelWeightsProvider &) = delete;
        QwenModelWeightsProvider &operator=(const QwenModelWeightsProvider &) = delete;

        // Enable move
        QwenModelWeightsProvider(QwenModelWeightsProvider &&) = default;
        QwenModelWeightsProvider &operator=(QwenModelWeightsProvider &&) = default;

        // =====================================================================
        // Global Weights (not layer-specific)
        // =====================================================================

        /**
         * @brief Get token embedding matrix
         * @return Token embedding [vocab_size, d_model] (replicated, not sliced)
         */
        std::shared_ptr<TensorBase> getTokenEmbedding() const;

        /**
         * @brief Get output normalization weight (final RMSNorm)
         * @return Output norm weight [d_model] (replicated)
         */
        std::shared_ptr<TensorBase> getOutputNorm() const;

        /**
         * @brief Get language model head projection matrix
         * @return LM head [vocab_size, d_model] (replicated)
         * @note May be tied to token_embedding (weight sharing)
         */
        std::shared_ptr<TensorBase> getLMHead() const;

        // =====================================================================
        // Per-Layer Attention Weights
        // =====================================================================

        /**
         * @brief Get attention normalization weight (pre-attention RMSNorm)
         * @param layer Layer index [0, n_layers)
         * @return Attention norm weight [d_model] (replicated)
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getAttentionNorm(int layer) const;

        /**
         * @brief Get query projection weight
         * @param layer Layer index [0, n_layers)
         * @return Q weight (LOCAL SLICE) [local_q_heads * head_dim, d_model]
         * @throws std::out_of_range if layer index invalid
         * @note Returns rank-local slice. Use getLocalSliceInfo("Q") for metadata.
         */
        std::shared_ptr<TensorBase> getQueryWeight(int layer) const;

        /**
         * @brief Get key projection weight
         * @param layer Layer index [0, n_layers)
         * @return K weight (LOCAL SLICE) [local_kv_heads * head_dim, d_model]
         * @throws std::out_of_range if layer index invalid
         * @note Returns rank-local slice. Use getLocalSliceInfo("K") for metadata.
         */
        std::shared_ptr<TensorBase> getKeyWeight(int layer) const;

        /**
         * @brief Get value projection weight
         * @param layer Layer index [0, n_layers)
         * @return V weight (LOCAL SLICE) [local_kv_heads * head_dim, d_model]
         * @throws std::out_of_range if layer index invalid
         * @note Returns rank-local slice. Use getLocalSliceInfo("V") for metadata.
         */
        std::shared_ptr<TensorBase> getValueWeight(int layer) const;

        /**
         * @brief Get output projection weight
         * @param layer Layer index [0, n_layers)
         * @return O weight [d_model, n_heads * head_dim] (replicated, not sliced)
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getOutputWeight(int layer) const;

        /**
         * @brief Get query projection bias (Qwen-specific)
         * @param layer Layer index [0, n_layers)
         * @return Q bias (LOCAL SLICE) or nullptr if not present
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getQueryBias(int layer) const;

        /**
         * @brief Get key projection bias (Qwen-specific)
         * @param layer Layer index [0, n_layers)
         * @return K bias (LOCAL SLICE) or nullptr if not present
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getKeyBias(int layer) const;

        /**
         * @brief Get value projection bias (Qwen-specific)
         * @param layer Layer index [0, n_layers)
         * @return V bias (LOCAL SLICE) or nullptr if not present
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getValueBias(int layer) const;

        // =====================================================================
        // Per-Layer FFN Weights
        // =====================================================================

        /**
         * @brief Get FFN normalization weight (pre-FFN RMSNorm)
         * @param layer Layer index [0, n_layers)
         * @return FFN norm weight [d_model] (replicated)
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getFFNNorm(int layer) const;

        /**
         * @brief Get gate projection weight (SwiGLU gate path)
         * @param layer Layer index [0, n_layers)
         * @return Gate weight (LOCAL SLICE) [local_d_ff, d_model]
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getGateWeight(int layer) const;

        /**
         * @brief Get up projection weight (SwiGLU up path)
         * @param layer Layer index [0, n_layers)
         * @return Up weight (LOCAL SLICE) [local_d_ff, d_model]
         * @throws std::out_of_range if layer index invalid
         */
        std::shared_ptr<TensorBase> getUpWeight(int layer) const;

        /**
         * @brief Get down projection weight (SwiGLU down path)
         * @param layer Layer index [0, n_layers)
         * @return Down weight (LOCAL SLICE) [d_model, local_d_ff]
         * @throws std::out_of_range if layer index invalid
         * @note Row-sliced (transposed column partitioning)
         */
        std::shared_ptr<TensorBase> getDownWeight(int layer) const;

        // =====================================================================
        // MPI Metadata (for verification and debugging)
        // =====================================================================

        /**
         * @brief Get MPI rank of this provider
         * @return Rank in MPI_COMM_WORLD [0, world_size)
         */
        int getRank() const { return mpi_ctx_.rank; }

        /**
         * @brief Get MPI world size
         * @return Total number of MPI ranks
         */
        int getWorldSize() const { return mpi_ctx_.size; }

        /**
         * @brief Check if a weight type is sliced across ranks
         * @param weight_type Weight identifier: "Q", "K", "V", "O", "GATE", "UP", "DOWN"
         * @return true if weight is partitioned across ranks
         *
         * Sliced weights: Q, K, V, GATE, UP, DOWN
         * Replicated weights: O, embedding, norms, biases, lm_head
         */
        bool isWeightSliced(const std::string &weight_type) const;

        /**
         * @brief Get local slice metadata for a sliced weight
         * @param weight_type Weight identifier: "Q", "K", "V", "GATE", "UP", "DOWN"
         * @return Pair of (offset, count) in the full weight dimension
         *
         * For attention weights (Q, K, V):
         * - offset: starting head index for this rank
         * - count: number of heads owned by this rank
         *
         * For FFN weights (GATE, UP, DOWN):
         * - offset: starting column/row index for this rank
         * - count: number of columns/rows owned by this rank
         *
         * @throws std::invalid_argument if weight_type is not sliced
         */
        std::pair<int, int> getLocalSliceInfo(const std::string &weight_type) const;

        /**
         * @brief Get number of layers in model
         * @return Number of transformer layers
         */
        int getNumLayers() const { return config_.n_layers; }

        /**
         * @brief Get model configuration
         * @return Const reference to transformer layer config
         */
        const TransformerLayerConfig &getConfig() const { return config_; }

        // =====================================================================
        // Backward Compatibility
        // =====================================================================

        /**
         * @brief Access underlying ModelWeights struct (legacy compatibility)
         * @return Const reference to raw weights
         *
         * @note Prefer using typed getters above. This is for gradual migration.
         */
        const QwenPipeline::ModelWeights &rawWeights() const { return *weights_; }

    private:
        std::unique_ptr<QwenPipeline::ModelWeights> weights_; ///< Owned weights
        MPIContext mpi_ctx_;                                  ///< MPI rank/size info
        TransformerLayerConfig config_;                       ///< Model dimensions

        /**
         * @brief Validate layer index is within bounds
         * @param layer Layer index to check
         * @throws std::out_of_range if layer >= n_layers
         */
        void validateLayerIndex(int layer) const;
    };

} // namespace llaminar
