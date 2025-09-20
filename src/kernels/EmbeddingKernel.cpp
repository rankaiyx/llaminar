#include "EmbeddingKernel.h"
#include "../logger.h"
#include <algorithm>

namespace llaminar
{

    EmbeddingKernel::EmbeddingKernel(size_t vocab_size, size_t embedding_dim)
        : vocab_size_(vocab_size), embedding_dim_(embedding_dim)
    {
        LOG_DEBUG("EmbeddingKernel initialized with vocab_size=" << vocab_size << ", embedding_dim=" << embedding_dim);
    }

    bool EmbeddingKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                  std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("EmbeddingKernel validation failed");
            return false;
        }

        auto token_ids_tensor = inputs[0];
        auto embedding_table = inputs[1];
        auto output = outputs[0];

        // Extract token IDs
        std::vector<int> token_ids(token_ids_tensor->shape()[0]);
        std::copy(token_ids_tensor->data(), token_ids_tensor->data() + token_ids_tensor->shape()[0], token_ids.begin());

        // Perform embedding lookup
        computeEmbedding(token_ids.data(), embedding_table->data(),
                         output->data(), token_ids.size(), embedding_dim_);

        return true;
    }

    bool EmbeddingKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                   const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            LOG_ERROR("EmbeddingKernel: Expected 2 inputs and 1 output, got " << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto token_ids = inputs[0];
        auto embedding_table = inputs[1];
        auto output = outputs[0];

        if (!token_ids || !embedding_table || !output)
        {
            LOG_ERROR("EmbeddingKernel: Null tensor provided");
            return false;
        }

        // Check token_ids is 1D
        if (token_ids->shape().size() != 1)
        {
            LOG_ERROR("EmbeddingKernel: Token IDs must be 1D, got " << token_ids->shape().size() << " dimensions");
            return false;
        }

        // Check embedding table is 2D [embedding_dim, vocab_size] (GGUF format)
        if (embedding_table->shape().size() != 2 ||
            embedding_table->shape()[0] != embedding_dim_ ||
            embedding_table->shape()[1] != vocab_size_)
        {
            LOG_ERROR("EmbeddingKernel: Embedding table shape mismatch. Expected ["
                      << embedding_dim_ << ", " << vocab_size_ << "], got ["
                      << embedding_table->shape()[0] << ", " << embedding_table->shape()[1] << "]");
            return false;
        }

        // Check output is 2D [seq_len, embedding_dim]
        if (output->shape().size() != 2 ||
            output->shape()[0] != token_ids->shape()[0] ||
            output->shape()[1] != embedding_dim_)
        {
            LOG_ERROR("EmbeddingKernel: Output shape mismatch");
            return false;
        }

        return true;
    }

    void EmbeddingKernel::computeEmbedding(const int *token_ids, const float *embedding_table,
                                           float *output, size_t seq_len, size_t embedding_dim)
    {
        for (size_t i = 0; i < seq_len; ++i)
        {
            int token_id = token_ids[i];
            if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size_)
            {
                LOG_WARN("EmbeddingKernel: Token ID " << token_id << " out of range [0, " << vocab_size_ << ")");
                // Zero out the embedding for invalid tokens
                std::fill(output + i * embedding_dim, output + (i + 1) * embedding_dim, 0.0f);
                continue;
            }

            // Copy embedding for this token (tensor is [embedding_dim, vocab_size])
            for (size_t j = 0; j < embedding_dim; ++j)
            {
                output[i * embedding_dim + j] = embedding_table[j * vocab_size_ + token_id];
            }
        }
    }

} // namespace llaminar