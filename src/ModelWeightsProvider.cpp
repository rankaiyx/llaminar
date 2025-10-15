/**
 * @file ModelWeightsProvider.cpp
 * @brief Implementation of MPI-aware model weights provider
 * @author David Sanftenberg
 */

#include "ModelWeightsProvider.h"
#include "logger.h"
#include <stdexcept>
#include <sstream>

namespace llaminar
{

    QwenModelWeightsProvider::QwenModelWeightsProvider(
        std::unique_ptr<QwenPipeline::ModelWeights> weights,
        const MPIContext &mpi_ctx,
        const TransformerLayerConfig &config)
        : weights_(std::move(weights)), mpi_ctx_(mpi_ctx), config_(config)
    {
        if (!weights_)
        {
            throw std::invalid_argument("QwenModelWeightsProvider: weights cannot be nullptr");
        }

        // Validate that weights struct has expected layer counts
        if (weights_->wq.size() != static_cast<size_t>(config_.n_layers))
        {
            std::ostringstream oss;
            oss << "QwenModelWeightsProvider: wq.size()=" << weights_->wq.size()
                << " != config.n_layers=" << config_.n_layers;
            throw std::invalid_argument(oss.str());
        }

        LOG_DEBUG("[WeightsProvider] Initialized for rank " << mpi_ctx_.rank
                                                            << "/" << mpi_ctx_.size
                                                            << " with " << config_.n_layers << " layers");
    }

    // =========================================================================
    // Global Weights
    // =========================================================================

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getTokenEmbedding() const
    {
        return weights_->token_embedding;
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getOutputNorm() const
    {
        return weights_->output_norm_weight;
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getLMHead() const
    {
        return weights_->lm_head;
    }

    // =========================================================================
    // Per-Layer Attention Weights
    // =========================================================================

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getAttentionNorm(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->attn_norm_weight[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getQueryWeight(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->wq[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getKeyWeight(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->wk[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getValueWeight(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->wv[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getOutputWeight(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->wo[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getQueryBias(int layer) const
    {
        validateLayerIndex(layer);
        if (layer < static_cast<int>(weights_->bq.size()))
        {
            return weights_->bq[layer];
        }
        return nullptr;
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getKeyBias(int layer) const
    {
        validateLayerIndex(layer);
        if (layer < static_cast<int>(weights_->bk.size()))
        {
            return weights_->bk[layer];
        }
        return nullptr;
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getValueBias(int layer) const
    {
        validateLayerIndex(layer);
        if (layer < static_cast<int>(weights_->bv.size()))
        {
            return weights_->bv[layer];
        }
        return nullptr;
    }

    // =========================================================================
    // Per-Layer FFN Weights
    // =========================================================================

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getFFNNorm(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->ffn_norm_weight[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getGateWeight(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->w_gate[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getUpWeight(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->w_up[layer];
    }

    std::shared_ptr<TensorBase> QwenModelWeightsProvider::getDownWeight(int layer) const
    {
        validateLayerIndex(layer);
        return weights_->w_down[layer];
    }

    // =========================================================================
    // MPI Metadata
    // =========================================================================

    bool QwenModelWeightsProvider::isWeightSliced(const std::string &weight_type) const
    {
        // Sliced weights (partitioned across ranks)
        if (weight_type == "Q" || weight_type == "K" || weight_type == "V" || weight_type == "O")
        {
            return true; // Attention projections sliced by heads (Q/K/V row-sliced, O column-sliced)
        }
        if (weight_type == "GATE" || weight_type == "UP" || weight_type == "DOWN")
        {
            return true; // FFN weights sliced by hidden dimension
        }

        // Replicated weights (full copy on each rank)
        if (weight_type == "EMBEDDING" ||
            weight_type == "LM_HEAD" || weight_type == "NORM")
        {
            return false;
        }

        // Unknown weight type - conservative default
        LOG_WARN("[WeightsProvider] Unknown weight type: " << weight_type
                                                           << ", assuming replicated");
        return false;
    }

    std::pair<int, int> QwenModelWeightsProvider::getLocalSliceInfo(const std::string &weight_type) const
    {
        if (!isWeightSliced(weight_type))
        {
            throw std::invalid_argument("QwenModelWeightsProvider::getLocalSliceInfo: "
                                        "weight type '" +
                                        weight_type + "' is not sliced");
        }

        // Attention weights: sliced by heads
        if (weight_type == "Q" || weight_type == "O")
        {
            // Q and O are both sliced by Q heads (Q rows, O columns)
            int local_q_heads = config_.n_head / mpi_ctx_.size;
            int head_offset = mpi_ctx_.rank * local_q_heads;
            return {head_offset, local_q_heads};
        }

        if (weight_type == "K" || weight_type == "V")
        {
            int local_kv_heads = config_.n_head_kv / mpi_ctx_.size;
            int kv_head_offset = mpi_ctx_.rank * local_kv_heads;
            return {kv_head_offset, local_kv_heads};
        }

        // FFN weights: sliced by hidden dimension
        if (weight_type == "GATE" || weight_type == "UP")
        {
            int local_d_ff = config_.d_ff / mpi_ctx_.size;
            int d_ff_offset = mpi_ctx_.rank * local_d_ff;
            return {d_ff_offset, local_d_ff};
        }

        if (weight_type == "DOWN")
        {
            // Down projection is row-sliced (transpose of column partitioning)
            int local_d_ff = config_.d_ff / mpi_ctx_.size;
            int d_ff_offset = mpi_ctx_.rank * local_d_ff;
            return {d_ff_offset, local_d_ff};
        }

        throw std::invalid_argument("QwenModelWeightsProvider::getLocalSliceInfo: "
                                    "unknown sliced weight type '" +
                                    weight_type + "'");
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    void QwenModelWeightsProvider::validateLayerIndex(int layer) const
    {
        if (layer < 0 || layer >= config_.n_layers)
        {
            std::ostringstream oss;
            oss << "QwenModelWeightsProvider: layer index " << layer
                << " out of range [0, " << config_.n_layers << ")";
            throw std::out_of_range(oss.str());
        }
    }

} // namespace llaminar
