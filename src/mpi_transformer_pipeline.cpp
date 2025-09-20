#include "mpi_transformer_pipeline.h"
#include "model_loader.h"
#include "kernels/EmbeddingKernel.h"
#include "tensors/tensor_factory.h"
#include <chrono>
#include <iomanip>
#include <cmath>

namespace llaminar
{

    MPITransformerPipeline::MPITransformerPipeline(const LayerConfig &config)
        : MPIKernelBase(), config_(config), use_kv_cache_(true), n_past_(0),
          total_embedding_time_(0.0), total_attention_time_(0.0),
          total_linear_time_(0.0), total_norm_time_(0.0), total_communication_time_(0.0)
    {

        // Initialize MPI kernels
        mpi_rmsnorm_kernel_ = std::make_unique<MPIRMSNormKernel>();
        mpi_attention_kernel_ = std::make_unique<MPIAttentionKernel>(
            config_.n_head, config_.n_head_kv, config_.head_dim);
        mpi_linear_kernel_ = std::make_unique<MPILinearKernel>();

        // Embedding kernel only needed on rank 0
        if (getRank() == 0)
        {
            embedding_kernel_ = std::make_unique<EmbeddingKernel>(config_.vocab_size, config_.d_model);
        }

        // Initialize KV cache if enabled
        if (use_kv_cache_)
        {
            initializeKVCache(config_.max_seq_len);
        }

        LOG_INFO("MPITransformerPipeline initialized on rank " << getRank() << "/" << getSize()
                                                               << " with " << config_.n_layers << " layers, " << config_.n_head << " heads");
    }

    bool MPITransformerPipeline::execute(const std::vector<int> &token_ids,
                                         const ModelWeights &weights,
                                         std::shared_ptr<TensorBase> &output)
    {
        start_time_ = std::chrono::high_resolution_clock::now();

        if (!validate(weights))
        {
            LOG_ERROR("MPITransformerPipeline: Weight validation failed");
            return false;
        }

        int seq_len = token_ids.size();
        if (seq_len <= 0 || seq_len > config_.max_seq_len)
        {
            LOG_ERROR("MPITransformerPipeline: Invalid sequence length " << seq_len);
            return false;
        }

        // Create intermediate tensors for layer-by-layer processing
        auto intermediate_tensors = createIntermediateTensors(seq_len);
        auto current_input = intermediate_tensors[0];
        auto layer_output = intermediate_tensors[1];

        // 1. Embedding lookup (rank 0 computes, broadcasts to all)
        auto embedding_start = std::chrono::high_resolution_clock::now();
        if (!executeEmbedding(token_ids, weights.token_embedding, current_input))
        {
            LOG_ERROR("MPITransformerPipeline: Embedding execution failed");
            return false;
        }
        auto embedding_end = std::chrono::high_resolution_clock::now();
        total_embedding_time_ += std::chrono::duration<double, std::milli>(embedding_end - embedding_start).count();

        // 2. Execute transformer layers sequentially
        for (int layer_idx = 0; layer_idx < config_.n_layers; ++layer_idx)
        {
            if (!executeTransformerLayer(layer_idx, current_input, weights, layer_output))
            {
                LOG_ERROR("MPITransformerPipeline: Layer " << layer_idx << " execution failed");
                return false;
            }

            // Swap tensors for next layer
            std::swap(current_input, layer_output);
        }

        // 3. Final output projection
        if (!executeOutputProjection(current_input, weights, output))
        {
            LOG_ERROR("MPITransformerPipeline: Output projection failed");
            return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(end_time - start_time_).count();

        if (getRank() == 0)
        {
            LOG_INFO("MPITransformerPipeline completed: " << seq_len << " tokens, "
                                                          << config_.n_layers << " layers in " << std::fixed << std::setprecision(2)
                                                          << total_time << "ms");
            LOG_DEBUG("Breakdown - Embedding: " << total_embedding_time_ << "ms, "
                                                << "Attention: " << total_attention_time_ << "ms, "
                                                << "Linear: " << total_linear_time_ << "ms, "
                                                << "Norm: " << total_norm_time_ << "ms");
        }

        return true;
    }

    bool MPITransformerPipeline::executeEmbedding(const std::vector<int> &token_ids,
                                                  const std::shared_ptr<TensorBase> &embedding_weight,
                                                  std::shared_ptr<TensorBase> &embedded_output)
    {
        int seq_len = token_ids.size();

        // Rank 0 performs embedding lookup
        if (getRank() == 0)
        {
            if (!embedding_kernel_)
            {
                LOG_ERROR("Embedding kernel not initialized on rank 0");
                return false;
            }

            // Create token_ids tensor
            auto token_ids_tensor = createLocalTensor({static_cast<size_t>(seq_len)});
            for (int i = 0; i < seq_len; ++i)
            {
                token_ids_tensor->data()[i] = static_cast<float>(token_ids[i]);
            }

            std::vector<std::shared_ptr<TensorBase>> inputs = {token_ids_tensor, embedding_weight};
            std::vector<std::shared_ptr<TensorBase>> outputs = {embedded_output};

            if (!embedding_kernel_->execute(inputs, outputs))
            {
                LOG_ERROR("Embedding kernel execution failed");
                return false;
            }
        }

        // Broadcast embedded sequence to all ranks
        auto comm_start = std::chrono::high_resolution_clock::now();
        checkMPIError(MPI_Bcast(embedded_output->data(),
                                seq_len * config_.d_model,
                                MPI_FLOAT, 0, getComm()),
                      "Embedding broadcast");
        auto comm_end = std::chrono::high_resolution_clock::now();
        total_communication_time_ += std::chrono::duration<double, std::milli>(comm_end - comm_start).count();

        LOG_DEBUG("Embedding completed: " << seq_len << " tokens -> "
                                          << seq_len << "x" << config_.d_model << " on rank " << getRank());

        return true;
    }

    bool MPITransformerPipeline::executeTransformerLayer(int layer_idx,
                                                         std::shared_ptr<TensorBase> &input,
                                                         const ModelWeights &weights,
                                                         std::shared_ptr<TensorBase> &output)
    {
        int seq_len = input->shape()[0];

        // Create temporary tensors for layer computation
        auto attn_norm_out = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)});
        auto attn_out = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)});
        auto ffn_norm_out = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)});
        auto ffn_out = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)});
        auto residual_tmp = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)});

        // 1. Attention path: RMSNorm -> Attention -> Residual
        auto norm_start = std::chrono::high_resolution_clock::now();

        // Attention pre-norm
        std::vector<std::shared_ptr<TensorBase>> norm_inputs = {input, weights.attn_norm_weight[layer_idx]};
        std::vector<std::shared_ptr<TensorBase>> norm_outputs = {attn_norm_out};

        if (!mpi_rmsnorm_kernel_->execute(norm_inputs, norm_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " attention norm failed");
            return false;
        }

        auto norm_end = std::chrono::high_resolution_clock::now();
        total_norm_time_ += std::chrono::duration<double, std::milli>(norm_end - norm_start).count();

        // Multi-head attention with MPI distribution
        auto attn_start = std::chrono::high_resolution_clock::now();

        mpi_attention_kernel_->setSequencePosition(n_past_);

        std::vector<std::shared_ptr<TensorBase>> attn_inputs = {
            attn_norm_out,
            weights.wq[layer_idx],
            weights.wk[layer_idx],
            weights.wv[layer_idx],
            weights.wo[layer_idx],
            use_kv_cache_ ? k_cache_[layer_idx] : createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.n_head_kv * config_.head_dim)}),
            use_kv_cache_ ? v_cache_[layer_idx] : createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.n_head_kv * config_.head_dim)})};
        std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out};

        if (!mpi_attention_kernel_->execute(attn_inputs, attn_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " attention failed");
            return false;
        }

        auto attn_end = std::chrono::high_resolution_clock::now();
        total_attention_time_ += std::chrono::duration<double, std::milli>(attn_end - attn_start).count();

        // Attention residual connection: input + attn_out -> residual_tmp
        const float *input_data = input->data();
        const float *attn_data = attn_out->data();
        float *residual_data = residual_tmp->data();

        for (size_t i = 0; i < input->total_elements(); ++i)
        {
            residual_data[i] = input_data[i] + attn_data[i];
        }

        // 2. Feed-forward path: RMSNorm -> Gate/Up -> SwiGLU -> Down -> Residual
        norm_start = std::chrono::high_resolution_clock::now();

        // FFN pre-norm
        norm_inputs = {residual_tmp, weights.ffn_norm_weight[layer_idx]};
        norm_outputs = {ffn_norm_out};

        if (!mpi_rmsnorm_kernel_->execute(norm_inputs, norm_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " FFN norm failed");
            return false;
        }

        norm_end = std::chrono::high_resolution_clock::now();
        total_norm_time_ += std::chrono::duration<double, std::milli>(norm_end - norm_start).count();

        // FFN computation using MPI linear kernels
        auto linear_start = std::chrono::high_resolution_clock::now();

        // Gate and Up projections (parallel)
        auto gate_out = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_ff)});
        auto up_out = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_ff)});

        // Gate projection
        std::vector<std::shared_ptr<TensorBase>> gate_inputs = {ffn_norm_out, weights.w_gate[layer_idx]};
        std::vector<std::shared_ptr<TensorBase>> gate_outputs = {gate_out};

        if (!mpi_linear_kernel_->execute(gate_inputs, gate_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " gate projection failed");
            return false;
        }

        // Up projection
        std::vector<std::shared_ptr<TensorBase>> up_inputs = {ffn_norm_out, weights.w_up[layer_idx]};
        std::vector<std::shared_ptr<TensorBase>> up_outputs = {up_out};

        if (!mpi_linear_kernel_->execute(up_inputs, up_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " up projection failed");
            return false;
        }

        // SwiGLU activation: gate_out * silu(up_out)
        float *gate_data = gate_out->data();
        float *up_data = up_out->data();

        for (size_t i = 0; i < gate_out->total_elements(); ++i)
        {
            float x = up_data[i];
            float silu = x / (1.0f + std::exp(-x)); // SiLU activation
            gate_data[i] *= silu;                   // SwiGLU: gate * silu(up)
        }

        // Down projection
        std::vector<std::shared_ptr<TensorBase>> down_inputs = {gate_out, weights.w_down[layer_idx]};
        std::vector<std::shared_ptr<TensorBase>> down_outputs = {ffn_out};

        if (!mpi_linear_kernel_->execute(down_inputs, down_outputs))
        {
            LOG_ERROR("Layer " << layer_idx << " down projection failed");
            return false;
        }

        auto linear_end = std::chrono::high_resolution_clock::now();
        total_linear_time_ += std::chrono::duration<double, std::milli>(linear_end - linear_start).count();

        // Final residual connection: residual_tmp + ffn_out -> output
        const float *ffn_data = ffn_out->data();
        float *output_data = output->data();

        for (size_t i = 0; i < output->total_elements(); ++i)
        {
            output_data[i] = residual_data[i] + ffn_data[i];
        }

        LOG_DEBUG("Layer " << layer_idx << " completed on rank " << getRank());
        return true;
    }

    bool MPITransformerPipeline::executeOutputProjection(std::shared_ptr<TensorBase> &input,
                                                         const ModelWeights &weights,
                                                         std::shared_ptr<TensorBase> &output)
    {
        int seq_len = input->shape()[0];

        // Final RMS normalization
        auto norm_out = createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)});

        std::vector<std::shared_ptr<TensorBase>> norm_inputs = {input, weights.output_norm_weight};
        std::vector<std::shared_ptr<TensorBase>> norm_outputs = {norm_out};

        if (!mpi_rmsnorm_kernel_->execute(norm_inputs, norm_outputs))
        {
            LOG_ERROR("Output normalization failed");
            return false;
        }

        // Language model head projection
        std::vector<std::shared_ptr<TensorBase>> lm_inputs = {norm_out, weights.lm_head};
        std::vector<std::shared_ptr<TensorBase>> lm_outputs = {output};

        if (!mpi_linear_kernel_->execute(lm_inputs, lm_outputs))
        {
            LOG_ERROR("LM head projection failed");
            return false;
        }

        LOG_DEBUG("Output projection completed: " << seq_len << "x" << config_.d_model
                                                  << " -> " << seq_len << "x" << config_.vocab_size);

        return true;
    }

    bool MPITransformerPipeline::validate(const ModelWeights &weights) const
    {
        // Validate embedding weights
        if (!weights.token_embedding)
        {
            LOG_ERROR("Token embedding is null");
            return false;
        }

        auto shape = weights.token_embedding->shape();
        LOG_DEBUG("Token embedding shape: [" << shape[0] << ", " << shape[1] << "]");
        LOG_DEBUG("Expected vocab_size: " << config_.vocab_size << ", d_model: " << config_.d_model);

        if (shape.size() != 2)
        {
            LOG_ERROR("Token embedding has " << shape.size() << " dimensions, expected 2");
            return false;
        }

        if (shape[0] != config_.d_model)
        {
            LOG_ERROR("Token embedding model dimension mismatch: got " << shape[0] << ", expected " << config_.d_model);
            return false;
        }

        if (shape[1] != config_.vocab_size)
        {
            LOG_ERROR("Token embedding vocab dimension mismatch: got " << shape[1] << ", expected " << config_.vocab_size);
            return false;
        }

        // Validate layer weights
        if (weights.attn_norm_weight.size() != config_.n_layers ||
            weights.wq.size() != config_.n_layers ||
            weights.wk.size() != config_.n_layers ||
            weights.wv.size() != config_.n_layers ||
            weights.wo.size() != config_.n_layers ||
            weights.ffn_norm_weight.size() != config_.n_layers ||
            weights.w_gate.size() != config_.n_layers ||
            weights.w_up.size() != config_.n_layers ||
            weights.w_down.size() != config_.n_layers)
        {
            LOG_ERROR("Inconsistent layer weight count");
            return false;
        }

        // Validate attention weight shapes for each layer
        int total_head_dim = config_.n_head * config_.head_dim;
        int kv_head_dim = config_.n_head_kv * config_.head_dim;

        for (int i = 0; i < config_.n_layers; ++i)
        {
            // Attention norm weights
            if (!weights.attn_norm_weight[i] ||
                weights.attn_norm_weight[i]->shape().size() != 1 ||
                weights.attn_norm_weight[i]->shape()[0] != config_.d_model)
            {
                LOG_ERROR("Invalid attention norm weight shape at layer " << i);
                return false;
            }

            // Q, K, V weight matrices
            if (!weights.wq[i] || weights.wq[i]->shape().size() != 2 ||
                weights.wq[i]->shape()[0] != config_.d_model ||
                weights.wq[i]->shape()[1] != total_head_dim)
            {
                LOG_ERROR("Invalid query weight shape at layer " << i);
                return false;
            }

            if (!weights.wk[i] || weights.wk[i]->shape().size() != 2 ||
                weights.wk[i]->shape()[0] != config_.d_model ||
                weights.wk[i]->shape()[1] != kv_head_dim)
            {
                LOG_ERROR("Invalid key weight shape at layer " << i);
                return false;
            }

            if (!weights.wv[i] || weights.wv[i]->shape().size() != 2 ||
                weights.wv[i]->shape()[0] != config_.d_model ||
                weights.wv[i]->shape()[1] != kv_head_dim)
            {
                LOG_ERROR("Invalid value weight shape at layer " << i);
                return false;
            }

            if (!weights.wo[i] || weights.wo[i]->shape().size() != 2 ||
                weights.wo[i]->shape()[0] != total_head_dim ||
                weights.wo[i]->shape()[1] != config_.d_model)
            {
                LOG_ERROR("Invalid output weight shape at layer " << i);
                return false;
            }

            // FFN weights
            if (!weights.ffn_norm_weight[i] ||
                weights.ffn_norm_weight[i]->shape().size() != 1 ||
                weights.ffn_norm_weight[i]->shape()[0] != config_.d_model)
            {
                LOG_ERROR("Invalid FFN norm weight shape at layer " << i);
                return false;
            }

            if (!weights.w_gate[i] || weights.w_gate[i]->shape().size() != 2 ||
                weights.w_gate[i]->shape()[0] != config_.d_model ||
                weights.w_gate[i]->shape()[1] != config_.d_ff)
            {
                LOG_ERROR("Invalid gate weight shape at layer " << i);
                return false;
            }

            if (!weights.w_up[i] || weights.w_up[i]->shape().size() != 2 ||
                weights.w_up[i]->shape()[0] != config_.d_model ||
                weights.w_up[i]->shape()[1] != config_.d_ff)
            {
                LOG_ERROR("Invalid up weight shape at layer " << i);
                return false;
            }

            if (!weights.w_down[i] || weights.w_down[i]->shape().size() != 2 ||
                weights.w_down[i]->shape()[0] != config_.d_ff ||
                weights.w_down[i]->shape()[1] != config_.d_model)
            {
                LOG_ERROR("Invalid down weight shape at layer " << i);
                return false;
            }
        }

        // Validate output weights
        if (!weights.output_norm_weight ||
            weights.output_norm_weight->shape().size() != 1 ||
            weights.output_norm_weight->shape()[0] != config_.d_model)
        {
            LOG_ERROR("Invalid output norm weight shape");
            return false;
        }

        if (!weights.lm_head || weights.lm_head->shape().size() != 2 ||
            weights.lm_head->shape()[0] != config_.d_model ||
            weights.lm_head->shape()[1] != config_.vocab_size)
        {
            LOG_ERROR("Invalid LM head weight shape");
            return false;
        }

        return true;
    }

    std::vector<std::shared_ptr<TensorBase>> MPITransformerPipeline::createIntermediateTensors(int seq_len)
    {
        std::vector<std::shared_ptr<TensorBase>> tensors;

        // Create tensors for intermediate computation
        tensors.push_back(createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)})); // Layer input
        tensors.push_back(createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(config_.d_model)})); // Layer output

        return tensors;
    }

    void MPITransformerPipeline::initializeKVCache(int seq_len)
    {
        k_cache_.clear();
        v_cache_.clear();

        for (int i = 0; i < config_.n_layers; ++i)
        {
            int kv_dim = config_.n_head_kv * config_.head_dim;
            k_cache_.push_back(createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}));
            v_cache_.push_back(createLocalTensor({static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}));

            // Initialize cache to zeros
            k_cache_[i]->zero();
            v_cache_[i]->zero();
        }

        LOG_DEBUG("Initialized KV cache for " << config_.n_layers << " layers, "
                                              << seq_len << " max sequence length on rank " << getRank());
    }

    // Factory function
    std::unique_ptr<MPITransformerPipeline> createMPITransformerPipeline(
        const MPITransformerPipeline::LayerConfig &config)
    {
        return std::make_unique<MPITransformerPipeline>(config);
    }

    // Utility function for loading weights
    MPITransformerPipeline::ModelWeights loadModelWeights(
        const std::string &model_path,
        const MPITransformerPipeline::LayerConfig &config)
    {
        MPITransformerPipeline::ModelWeights weights;

        // Create and load model
        ModelLoader loader;
        if (!loader.loadModel(model_path))
        {
            LOG_ERROR("Failed to load model from: " << model_path);
            throw std::runtime_error("Failed to load model");
        }

        LOG_INFO("Loading model weights from: " << model_path);

        // Load token embedding
        weights.token_embedding = loader.loadTensor("token_embd.weight");
        if (!weights.token_embedding)
        {
            LOG_ERROR("Failed to load token embedding");
            throw std::runtime_error("Failed to load token embedding");
        }

        // Debug: Print token embedding shape
        auto token_emb_shape = weights.token_embedding->shape();
        LOG_INFO("Token embedding shape: [" << token_emb_shape[0] << ", " << token_emb_shape[1] << "]");
        LOG_INFO("Expected vocab_size: " << config.vocab_size << ", d_model: " << config.d_model);

        LOG_INFO("Loading output norm weights...");
        // Load output layer weights (Qwen2.5 uses tied embeddings - no separate lm_head)
        weights.output_norm_weight = loader.loadTensor("output_norm.weight");
        if (!weights.output_norm_weight)
        {
            LOG_ERROR("Failed to load output norm weights");
            throw std::runtime_error("Failed to load output norm weights");
        }
        LOG_INFO("Output norm weights loaded successfully");

        // Qwen2.5 uses tied embeddings: token_embd.weight is reused for output projection
        weights.lm_head = weights.token_embedding;
        LOG_INFO("Set lm_head to reuse token_embedding (tied embeddings)");

        LOG_INFO("Starting per-layer weight loading for " << config.n_layers << " layers...");
        for (int i = 0; i < config.n_layers; ++i)
        {
            std::string layer_prefix = "blk." + std::to_string(i) + ".";

            // Attention normalization
            std::string attn_norm_name = layer_prefix + "attn_norm.weight";
            auto attn_norm = loader.loadTensor(attn_norm_name);
            if (!attn_norm)
            {
                LOG_ERROR("Failed to load " << attn_norm_name);
                throw std::runtime_error("Failed to load attention norm weight");
            }
            weights.attn_norm_weight.push_back(attn_norm);

            // Attention projection weights
            auto wq = loader.loadTensor(layer_prefix + "attn_q.weight");
            auto wk = loader.loadTensor(layer_prefix + "attn_k.weight");
            auto wv = loader.loadTensor(layer_prefix + "attn_v.weight");
            auto wo = loader.loadTensor(layer_prefix + "attn_output.weight");

            if (!wq || !wk || !wv || !wo)
            {
                LOG_ERROR("Failed to load attention weights for layer " << i);
                throw std::runtime_error("Failed to load attention weights");
            }

            weights.wq.push_back(wq);
            weights.wk.push_back(wk);
            weights.wv.push_back(wv);
            weights.wo.push_back(wo);

            // Feed-forward normalization
            auto ffn_norm = loader.loadTensor(layer_prefix + "ffn_norm.weight");
            if (!ffn_norm)
            {
                LOG_ERROR("Failed to load FFN norm for layer " << i);
                throw std::runtime_error("Failed to load FFN norm weight");
            }
            weights.ffn_norm_weight.push_back(ffn_norm);

            // Feed-forward weights
            auto w_gate = loader.loadTensor(layer_prefix + "ffn_gate.weight");
            auto w_up = loader.loadTensor(layer_prefix + "ffn_up.weight");
            auto w_down = loader.loadTensor(layer_prefix + "ffn_down.weight");

            if (!w_gate || !w_up || !w_down)
            {
                LOG_ERROR("Failed to load FFN weights for layer " << i);
                throw std::runtime_error("Failed to load FFN weights");
            }

            weights.w_gate.push_back(w_gate);
            weights.w_up.push_back(w_up);
            weights.w_down.push_back(w_down);
        }

        LOG_INFO("Successfully loaded all model weights: " << config.n_layers << " layers");
        return weights;
    }

    // KernelBase interface implementation
    bool MPITransformerPipeline::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                         std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        // This interface is not used for our main pipeline - use the execute(tokens, weights, output) method instead
        LOG_ERROR("MPITransformerPipeline: KernelBase::execute not implemented - use execute(tokens, weights, output)");
        return false;
    }

    bool MPITransformerPipeline::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                          const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 2)
        {
            LOG_ERROR("MPITransformerPipeline: Expected 2 inputs (token_ids, weights), got " << inputs.size());
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("MPITransformerPipeline: Expected 1 output (logits), got " << outputs.size());
            return false;
        }

        // Validate token_ids tensor
        if (!inputs[0] || inputs[0]->shape().size() != 1)
        {
            LOG_ERROR("MPITransformerPipeline: Invalid token_ids tensor shape");
            return false;
        }

        // Validate output tensor
        if (!outputs[0])
        {
            LOG_ERROR("MPITransformerPipeline: Output tensor is null");
            return false;
        }

        return true;
    }

} // namespace llaminar