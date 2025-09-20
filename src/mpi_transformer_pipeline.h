#pragma once

#include "graph_compute.h"
#include "kernels/MPILinearKernel.h"
#include "kernels/MPIRMSNormKernel.h"
#include "kernels/MPIAttentionKernel.h"
#include "kernels/EmbeddingKernel.h"
#include "mpi_kernel_base.h"
#include "transformer_config.h"
#include "logger.h"
#include <memory>
#include <vector>
#include <mpi.h>

namespace llaminar
{

    /**
     * @brief MPI-enabled transformer pipeline for distributed LLM inference
     *
     * This class coordinates the execution of a complete transformer forward pass
     * using MPI-distributed kernels across multiple processes. It manages:
     * - Distributed embedding lookup
     * - MPI-parallel RMS normalization
     * - MPI-parallel multi-head attention with head-wise distribution
     * - MPI-parallel linear transformations with COSMA integration
     * - Inter-kernel communication and data synchronization
     */
    class MPITransformerPipeline : public MPIKernelBase
    {
    public:
        // Type alias for compatibility
        using LayerConfig = TransformerLayerConfig;

        /**
         * @brief Model weights for transformer layers
         */
        struct ModelWeights
        {
            // Embedding weights
            std::shared_ptr<TensorBase> token_embedding;

            // Per-layer weights (indexed by layer number)
            std::vector<std::shared_ptr<TensorBase>> attn_norm_weight;
            std::vector<std::shared_ptr<TensorBase>> wq;
            std::vector<std::shared_ptr<TensorBase>> wk;
            std::vector<std::shared_ptr<TensorBase>> wv;
            std::vector<std::shared_ptr<TensorBase>> wo;
            std::vector<std::shared_ptr<TensorBase>> ffn_norm_weight;
            std::vector<std::shared_ptr<TensorBase>> w_gate;
            std::vector<std::shared_ptr<TensorBase>> w_up;
            std::vector<std::shared_ptr<TensorBase>> w_down;

            // Output layer
            std::shared_ptr<TensorBase> output_norm_weight;
            std::shared_ptr<TensorBase> lm_head;
        };

        /**
         * @brief Constructor for MPI transformer pipeline
         * @param config Transformer layer configuration
         */
        explicit MPITransformerPipeline(const LayerConfig &config);

        ~MPITransformerPipeline() = default;

        /**
         * @brief Execute complete transformer forward pass
         * @param token_ids Input token sequence
         * @param weights Model weights
         * @param output Output logits tensor
         * @return Success status
         */
        bool execute(const std::vector<int> &token_ids,
                     const ModelWeights &weights,
                     std::shared_ptr<TensorBase> &output);

        /**
         * @brief Validate pipeline configuration and weights
         * @param weights Model weights to validate
         * @return Validation success
         */
        bool validate(const ModelWeights &weights) const;

        /**
         * @brief Get pipeline configuration
         * @return Layer configuration
         */
        const LayerConfig &getConfig() const { return config_; }

        /**
         * @brief Enable/disable attention caching for inference
         * @param enable Cache enable flag
         */
        void enableKVCache(bool enable) { use_kv_cache_ = enable; }

        /**
         * @brief Set current position for attention computation
         * @param pos Current sequence position
         */
        void setSequencePosition(int pos) { n_past_ = pos; }

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getKernelType() const override { return "MPITransformerPipeline"; }
        size_t getExpectedInputCount() const override { return 2; }  // token_ids, weights
        size_t getExpectedOutputCount() const override { return 1; } // logits

    private:
        /**
         * @brief Execute embedding lookup on rank 0 and broadcast
         * @param token_ids Input token sequence
         * @param embedding_weight Embedding weight matrix
         * @param embedded_output Output embedded sequence
         * @return Success status
         */
        bool executeEmbedding(const std::vector<int> &token_ids,
                              const std::shared_ptr<TensorBase> &embedding_weight,
                              std::shared_ptr<TensorBase> &embedded_output);

        /**
         * @brief Execute single transformer layer with MPI kernels
         * @param layer_idx Layer index
         * @param input Input tensor from previous layer
         * @param weights Layer-specific weights
         * @param output Output tensor for this layer
         * @return Success status
         */
        bool executeTransformerLayer(int layer_idx,
                                     std::shared_ptr<TensorBase> &input,
                                     const ModelWeights &weights,
                                     std::shared_ptr<TensorBase> &output);

        /**
         * @brief Execute final output projection and normalization
         * @param input Final layer output
         * @param weights Model weights (norm + lm_head)
         * @param output Final logits
         * @return Success status
         */
        bool executeOutputProjection(std::shared_ptr<TensorBase> &input,
                                     const ModelWeights &weights,
                                     std::shared_ptr<TensorBase> &output);

        /**
         * @brief Create intermediate tensors with proper shapes
         * @param seq_len Sequence length
         * @return Vector of intermediate tensors
         */
        std::vector<std::shared_ptr<TensorBase>> createIntermediateTensors(int seq_len);

        /**
         * @brief Validate tensor shapes for layer computation
         * @param layer_idx Layer index for error reporting
         * @param tensors Input tensors to validate
         * @return Validation success
         */
        bool validateLayerTensors(int layer_idx,
                                  const std::vector<std::shared_ptr<TensorBase>> &tensors) const;

        /**
         * @brief Initialize KV cache tensors for attention
         * @param seq_len Maximum sequence length
         */
        void initializeKVCache(int seq_len);

    private:
        LayerConfig config_; ///< Transformer configuration
        bool use_kv_cache_;  ///< KV cache enabled flag
        int n_past_;         ///< Current sequence position

        // MPI kernels for distributed computation
        std::unique_ptr<MPIRMSNormKernel> mpi_rmsnorm_kernel_;     ///< MPI RMS normalization
        std::unique_ptr<MPIAttentionKernel> mpi_attention_kernel_; ///< MPI attention computation
        std::unique_ptr<MPILinearKernel> mpi_linear_kernel_;       ///< MPI linear transformation
        std::unique_ptr<EmbeddingKernel> embedding_kernel_;        ///< Sequential embedding (rank 0 only)

        // KV cache tensors (if enabled)
        std::vector<std::shared_ptr<TensorBase>> k_cache_; ///< Key cache per layer
        std::vector<std::shared_ptr<TensorBase>> v_cache_; ///< Value cache per layer

        // Performance monitoring
        mutable std::chrono::high_resolution_clock::time_point start_time_;
        mutable double total_embedding_time_;
        mutable double total_attention_time_;
        mutable double total_linear_time_;
        mutable double total_norm_time_;
        mutable double total_communication_time_;
    };

    /**
     * @brief Factory function for creating MPI transformer pipeline
     * @param config Transformer configuration
     * @return Unique pointer to MPI transformer pipeline
     */
    std::unique_ptr<MPITransformerPipeline> createMPITransformerPipeline(
        const MPITransformerPipeline::LayerConfig &config);

    /**
     * @brief Utility function to load model weights from file/memory
     * @param model_path Path to model weights
     * @param config Transformer configuration
     * @return Model weights structure
     */
    MPITransformerPipeline::ModelWeights loadModelWeights(
        const std::string &model_path,
        const MPITransformerPipeline::LayerConfig &config);

} // namespace llaminar