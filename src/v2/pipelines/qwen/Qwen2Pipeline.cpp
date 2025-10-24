/**
 * @file Qwen2Pipeline.cpp
 * @brief Qwen 2.x transformer pipeline implementation
 *
 * Greenfield V2 implementation with:
 * - Direct kernel orchestration (no operator layer)
 * - Streaming dequant in kernels (no slab cache)
 * - Per-tensor device placement
 * - Selective BF16 for bandwidth-bound ops
 *
 * @author David Sanftenberg
 */

#include "Qwen2Pipeline.h"
#include "../PipelineFactory.h"
#include "../../loaders/ModelLoader.h"
#include "../../tensors/TensorFactory.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace llaminar2
{

    // =============================================================================
    // Factory Registration
    // =============================================================================

    /**
     * @brief Creator function for Qwen2Pipeline
     */
    static std::unique_ptr<PipelineBase> createQwen2(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx)
    {
        return std::make_unique<Qwen2Pipeline>(model_ctx, mpi_ctx, device_idx);
    }

    /**
     * @brief Register Qwen2Pipeline with factory
     *
     * Made public so tests can force registration if needed
     */
    void ensureQwen2Registration()
    {
        static bool registered = false;
        if (!registered)
        {
            PipelineFactory::instance().registerCreator("qwen2", &createQwen2);
            registered = true;
        }
    }

    /**
     * @brief Automatic registration at startup
     */
    __attribute__((constructor)) static void initQwen2()
    {
        ensureQwen2Registration();
    }

    // =============================================================================
    // Pipeline Implementation
    // =============================================================================

    Qwen2Pipeline::Qwen2Pipeline(std::shared_ptr<ModelContext> model_ctx,
                                 std::shared_ptr<MPIContext> mpi_ctx,
                                 int device_idx)
        : PipelineBase(model_ctx, mpi_ctx, device_idx)
    {
        std::cout << "[Qwen2Pipeline] Initializing Qwen 2.x pipeline\n";

        // Read architecture from GGUF metadata
        const GGUFModel &model = model_ctx_->model();
        n_layers_ = static_cast<int>(model.block_count);
        d_model_ = static_cast<int>(model.embedding_length);
        vocab_size_ = static_cast<int>(model.vocab_size);
        n_heads_ = static_cast<int>(model.head_count);
        n_kv_heads_ = static_cast<int>(model.head_count_kv);

        // Calculate head_dim from d_model and n_heads
        head_dim_ = d_model_ / n_heads_;

        // Read FFN intermediate size from metadata
        if (model.hasMetadata("qwen2.feed_forward_length"))
        {
            d_ff_ = static_cast<int>(model.metadata.at("qwen2.feed_forward_length").asUInt32());
        }
        else
        {
            // Fallback: typical ratio for Qwen models
            d_ff_ = d_model_ * 4;
            std::cout << "[Qwen2Pipeline] Warning: feed_forward_length not in metadata, using " << d_ff_ << "\n";
        }

        std::cout << "[Qwen2Pipeline] Architecture: " << n_layers_ << " layers, "
                  << d_model_ << " d_model, " << vocab_size_ << " vocab\n";
        std::cout << "[Qwen2Pipeline] Attention: " << n_heads_ << " heads, "
                  << n_kv_heads_ << " KV heads (GQA), " << head_dim_ << " head_dim\n";
        std::cout << "[Qwen2Pipeline] FFN: " << d_ff_ << " intermediate_size (SwiGLU)\n";

        // Weights are loaded lazily via getLayerWeight() and model_ctx_->getWeight()
        // Resize layer weights vector for lazy loading
        layers_.resize(n_layers_);

        std::cout << "[Qwen2Pipeline] Pipeline initialized (weights loaded on-demand)\n";
    }

    bool Qwen2Pipeline::forward(const int *tokens, int seq_len)
    {
        std::cout << "[Qwen2Pipeline] Forward pass with seq_len=" << seq_len << "\n";

        // Allocate activation tensors if needed
        if (!current_hidden_ || static_cast<int>(current_hidden_->shape()[0]) != seq_len)
        {
            current_hidden_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
            std::cout << "[Qwen2Pipeline] Allocated hidden states: "
                      << seq_len << " x " << d_model_ << "\n";
        }

        // Validate hidden state dimensions
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "hidden_allocation");

        // Embedding lookup
        auto embed_table = getEmbeddingTable();
        if (!embed_table)
        {
            std::cerr << "[Qwen2Pipeline] Failed to load embedding table\n";
            return false;
        }

        // Manual embedding lookup: hidden[i] = embed_table[tokens[i]]
        const float *embed_data = embed_table->data();
        float *hidden_data = current_hidden_->mutable_data();
        for (int i = 0; i < seq_len; ++i)
        {
            int token_id = tokens[i];
            if (token_id < 0 || token_id >= vocab_size_)
            {
                std::cerr << "[Qwen2Pipeline] Invalid token " << token_id << " at position " << i << "\n";
                return false;
            }
            std::memcpy(hidden_data + i * d_model_,
                        embed_data + token_id * d_model_,
                        d_model_ * sizeof(float));
        }

        // Validate after embedding
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_embedding");

        // Process all transformer layers
        for (int i = 0; i < n_layers_; ++i)
        {
            if (!transformer_layer(i, seq_len))
            {
                std::cerr << "[Qwen2Pipeline] Layer " << i << " failed\n";
                return false;
            }

            // Validate hidden state dimensions unchanged between layers
            VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_layer_" + std::to_string(i));
        }

        // Final normalization
        auto final_norm = getFinalNorm();
        if (!final_norm)
        {
            std::cerr << "[Qwen2Pipeline] Failed to load final norm\n";
            return false;
        }

        auto norm_kernel = final_norm->createRMSNorm();
        if (!norm_kernel)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create RMSNorm kernel\n";
            return false;
        }

        if (!norm_kernel->apply(
                current_hidden_->data(), final_norm->data(), current_hidden_->mutable_data(),
                seq_len, d_model_, 1e-6f, false, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] Final RMSNorm failed\n";
            return false;
        }

        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_final_norm");

        // LM head projection
        if (!logits_ || static_cast<int>(logits_->shape()[0]) != seq_len)
        {
            logits_ = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(vocab_size_)});
            std::cout << "[Qwen2Pipeline] Allocated logits: "
                      << seq_len << " x " << vocab_size_ << "\n";
        }

        // Validate logits dimensions
        VALIDATE_TENSOR(logits_, spec_logits(seq_len), "logits_allocation");

        auto lm_head = getLMHead();
        if (!lm_head)
        {
            std::cerr << "[Qwen2Pipeline] Failed to load LM head\n";
            return false;
        }

        auto lm_gemm = lm_head->createGemm();
        if (!lm_gemm)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create LM head GEMM kernel\n";
            return false;
        }

        // LM head: logits = hidden @ lm_head^T
        // hidden: [seq_len, d_model], lm_head: [vocab_size, d_model]
        // output: [seq_len, vocab_size]
        if (!lm_gemm->multiply(
                current_hidden_->data(), logits_->mutable_data(),
                seq_len, vocab_size_, d_model_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] LM head projection failed\n";
            return false;
        }

        VALIDATE_TENSOR(logits_, spec_logits(seq_len), "after_lm_head");

        return true;
    }

    bool Qwen2Pipeline::transformer_layer(int layer_idx, int seq_len)
    {
        std::cout << "[Qwen2Pipeline] Processing layer " << layer_idx << "\n";

        // Get layer weights (loaded lazily on first access)
        auto &layer = getLayerWeights(layer_idx);

        // Attention block
        if (!attention_block(layer, seq_len))
        {
            std::cerr << "[Qwen2Pipeline] Attention block failed in layer "
                      << layer_idx << "\n";
            return false;
        }

        // FFN block
        if (!ffn_block(layer, seq_len))
        {
            std::cerr << "[Qwen2Pipeline] FFN block failed in layer "
                      << layer_idx << "\n";
            return false;
        }

        return true;
    }

    bool Qwen2Pipeline::attention_block(const LayerWeights &layer, int seq_len)
    {
        // Validate input dimensions
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "attn_input");
        VALIDATE_TENSOR_PTR(layer.attn_norm.get(), spec_norm_gamma(), "attn_norm_weight");

        // Save residual for later
        auto residual = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
        std::memcpy(residual->mutable_data(), current_hidden_->data(),
                    seq_len * d_model_ * sizeof(float));

        // 1. Pre-attention RMSNorm
        auto norm_kernel = layer.attn_norm->createRMSNorm();
        if (!norm_kernel ||
            !norm_kernel->apply(
                current_hidden_->data(), layer.attn_norm->data(), current_hidden_->mutable_data(),
                seq_len, d_model_, 1e-6f, false, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] Attention norm failed\n";
            return false;
        }
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_attn_norm");

        // 2. Q/K/V projections
        auto Q = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_ * head_dim_)});
        auto K = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)});
        auto V = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)});

        auto q_gemm = layer.wq->createGemm();
        auto k_gemm = layer.wk->createGemm();
        auto v_gemm = layer.wv->createGemm();

        if (!q_gemm || !k_gemm || !v_gemm)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create Q/K/V GEMM kernels\n";
            return false;
        }

        // Q = hidden @ wq^T
        if (!q_gemm->multiply(
                current_hidden_->data(), Q->mutable_data(),
                seq_len, n_heads_ * head_dim_, d_model_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] Q projection failed\n";
            return false;
        }
        VALIDATE_TENSOR(Q, spec_q(seq_len), "after_q_proj");

        // K = hidden @ wk^T
        if (!k_gemm->multiply(
                current_hidden_->data(), K->mutable_data(),
                seq_len, n_kv_heads_ * head_dim_, d_model_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] K projection failed\n";
            return false;
        }
        VALIDATE_TENSOR(K, spec_kv(seq_len), "after_k_proj");

        // V = hidden @ wv^T
        if (!v_gemm->multiply(
                current_hidden_->data(), V->mutable_data(),
                seq_len, n_kv_heads_ * head_dim_, d_model_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] V projection failed\n";
            return false;
        }
        VALIDATE_TENSOR(V, spec_kv(seq_len), "after_v_proj");

        // 3. Apply RoPE to Q and K
        std::vector<int> position_ids(seq_len);
        for (int i = 0; i < seq_len; ++i)
        {
            position_ids[i] = i; // TODO: Handle incremental decode position offset
        }

        auto rope_kernel = layer.wq->createRoPE(); // Any weight can create RoPE kernel
        if (!rope_kernel)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create RoPE kernel\n";
            return false;
        }

        if (!rope_kernel->apply(
                Q->mutable_data(), K->mutable_data(), position_ids.data(),
                seq_len, n_heads_, n_kv_heads_, head_dim_,
                false, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] RoPE application failed\n";
            return false;
        }
        VALIDATE_TENSOR(Q, spec_q(seq_len), "after_rope_q");
        VALIDATE_TENSOR(K, spec_kv(seq_len), "after_rope_k");

        // 4. GQA attention computation
        // Use default PipelineBase::attention_gqa() orchestration
        // Supports GQA (n_heads=14, n_kv_heads=2), MHA, MQA, and sliding window
        auto attn_output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_ * head_dim_)});

        if (!attention_gqa(
                Q->data(), K->data(), V->data(), attn_output->mutable_data(),
                seq_len, n_heads_, n_kv_heads_, head_dim_,
                /*causal=*/true, /*window_size=*/-1))
        {
            std::cerr << "[Qwen2Pipeline] GQA attention failed\n";
            return false;
        }

        VALIDATE_TENSOR(attn_output, spec_q(seq_len), "after_attention");

        // 5. Output projection
        auto o_gemm = layer.wo->createGemm();
        if (!o_gemm)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create output GEMM kernel\n";
            return false;
        }

        // output = attn_output @ wo^T
        auto attn_proj = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});

        if (!o_gemm->multiply(
                attn_output->data(), attn_proj->mutable_data(),
                seq_len, d_model_, n_heads_ * head_dim_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] Output projection failed\n";
            return false;
        }
        VALIDATE_TENSOR(attn_proj, spec_hidden(seq_len), "after_attn_out_proj");

        // 6. Residual connection
        for (size_t i = 0; i < seq_len * d_model_; ++i)
        {
            current_hidden_->mutable_data()[i] = residual->data()[i] + attn_proj->data()[i];
        }
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_attn_residual");

        return true;
    }

    bool Qwen2Pipeline::ffn_block(const LayerWeights &layer, int seq_len)
    {
        // Validate input dimensions
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "ffn_input");
        VALIDATE_TENSOR_PTR(layer.ffn_norm.get(), spec_norm_gamma(), "ffn_norm_weight");

        // Save residual for later
        auto residual = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
        std::memcpy(residual->mutable_data(), current_hidden_->data(),
                    seq_len * d_model_ * sizeof(float));

        // 1. Pre-FFN RMSNorm
        auto norm_kernel = layer.ffn_norm->createRMSNorm();
        if (!norm_kernel ||
            !norm_kernel->apply(
                current_hidden_->data(), layer.ffn_norm->data(), current_hidden_->mutable_data(),
                seq_len, d_model_, 1e-6f, false, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] FFN norm failed\n";
            return false;
        }
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_ffn_norm");

        // 2. Gate and up projections
        auto gate = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)});
        auto up = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)});

        auto gate_gemm = layer.gate_proj->createGemm();
        auto up_gemm = layer.up_proj->createGemm();

        if (!gate_gemm || !up_gemm)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create gate/up GEMM kernels\n";
            return false;
        }

        // gate = hidden @ gate_proj^T
        if (!gate_gemm->multiply(
                current_hidden_->data(), gate->mutable_data(),
                seq_len, d_ff_, d_model_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] Gate projection failed\n";
            return false;
        }
        VALIDATE_TENSOR(gate, spec_ffn_intermediate(seq_len), "after_gate_proj");

        // up = hidden @ up_proj^T
        if (!up_gemm->multiply(
                current_hidden_->data(), up->mutable_data(),
                seq_len, d_ff_, d_model_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] Up projection failed\n";
            return false;
        }
        VALIDATE_TENSOR(up, spec_ffn_intermediate(seq_len), "after_up_proj");

        // 3. SwiGLU activation: swiglu_out = gate * silu(up)
        auto swiglu_out = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_ff_)});

        auto swiglu_kernel = layer.gate_proj->createSwiGLU();
        if (!swiglu_kernel)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create SwiGLU kernel\n";
            return false;
        }

        if (!swiglu_kernel->apply(
                gate->data(), up->data(), swiglu_out->mutable_data(),
                seq_len, d_ff_, false, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] SwiGLU activation failed\n";
            return false;
        }
        VALIDATE_TENSOR(swiglu_out, spec_ffn_intermediate(seq_len), "after_swiglu");

        // 4. Down projection
        auto down_out = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});

        auto down_gemm = layer.down_proj->createGemm();
        if (!down_gemm)
        {
            std::cerr << "[Qwen2Pipeline] Failed to create down GEMM kernel\n";
            return false;
        }

        // down_out = swiglu_out @ down_proj^T
        if (!down_gemm->multiply(
                swiglu_out->data(), down_out->mutable_data(),
                seq_len, d_model_, d_ff_,
                true, 1.0f, 0.0f, mpi_ctx_.get(), device_idx_))
        {
            std::cerr << "[Qwen2Pipeline] Down projection failed\n";
            return false;
        }
        VALIDATE_TENSOR(down_out, spec_hidden(seq_len), "after_down_proj");

        // 5. Residual connection
        for (size_t i = 0; i < seq_len * d_model_; ++i)
        {
            current_hidden_->mutable_data()[i] = residual->data()[i] + down_out->data()[i];
        }
        VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_ffn_residual");

        return true;
    }

    const float *Qwen2Pipeline::logits() const
    {
        if (!logits_)
        {
            std::cerr << "[Qwen2Pipeline] logits() called before forward()\n";
            return nullptr;
        }
        return logits_->data();
    }

    // =============================================================================
    // Lazy Weight Accessors
    // =============================================================================

    std::shared_ptr<TensorBase> Qwen2Pipeline::getEmbeddingTable()
    {
        if (!embedding_table_)
        {
            embedding_table_ = model_ctx_->getWeight("token_embd.weight", device_idx_);
        }
        return embedding_table_;
    }

    std::shared_ptr<TensorBase> Qwen2Pipeline::getFinalNorm()
    {
        if (!final_norm_)
        {
            final_norm_ = model_ctx_->getWeight("output_norm.weight", device_idx_);
        }
        return final_norm_;
    }

    std::shared_ptr<TensorBase> Qwen2Pipeline::getLMHead()
    {
        if (!lm_head_)
        {
            lm_head_ = model_ctx_->getWeight("output.weight", device_idx_);
        }
        return lm_head_;
    }

    Qwen2Pipeline::LayerWeights &Qwen2Pipeline::getLayerWeights(int layer_idx)
    {
        auto &layer = layers_[layer_idx];
        std::string prefix = "blk." + std::to_string(layer_idx) + ".";

        // Lazy load on first access
        if (!layer.wq)
        {
            layer.wq = model_ctx_->getWeight(prefix + "attn_q.weight", device_idx_);
            layer.wk = model_ctx_->getWeight(prefix + "attn_k.weight", device_idx_);
            layer.wv = model_ctx_->getWeight(prefix + "attn_v.weight", device_idx_);
            layer.wo = model_ctx_->getWeight(prefix + "attn_output.weight", device_idx_);
            layer.attn_norm = model_ctx_->getWeight(prefix + "attn_norm.weight", device_idx_);
            layer.gate_proj = model_ctx_->getWeight(prefix + "ffn_gate.weight", device_idx_);
            layer.up_proj = model_ctx_->getWeight(prefix + "ffn_up.weight", device_idx_);
            layer.down_proj = model_ctx_->getWeight(prefix + "ffn_down.weight", device_idx_);
            layer.ffn_norm = model_ctx_->getWeight(prefix + "ffn_norm.weight", device_idx_);
        }

        return layer;
    }

    std::shared_ptr<TensorBase> Qwen2Pipeline::get_layer_weight(
        int layer_idx, const std::string &weight_name)
    {
        if (layer_idx < 0 || layer_idx >= n_layers_)
        {
            std::cerr << "[Qwen2Pipeline] Invalid layer index: " << layer_idx << "\n";
            return nullptr;
        }

        const auto &layer = layers_[layer_idx];

        if (weight_name == "wq")
            return layer.wq;
        if (weight_name == "wk")
            return layer.wk;
        if (weight_name == "wv")
            return layer.wv;
        if (weight_name == "wo")
            return layer.wo;
        if (weight_name == "gate")
            return layer.gate_proj;
        if (weight_name == "up")
            return layer.up_proj;
        if (weight_name == "down")
            return layer.down_proj;
        if (weight_name == "attn_norm")
            return layer.attn_norm;
        if (weight_name == "ffn_norm")
            return layer.ffn_norm;

        std::cerr << "[Qwen2Pipeline] Unknown weight name: " << weight_name << "\n";
        return nullptr;
    }

} // namespace llaminar2
