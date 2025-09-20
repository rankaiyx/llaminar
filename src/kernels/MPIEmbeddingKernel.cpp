#include "MPIEmbeddingKernel.h"
#include "../logger.h"
#include <algorithm>
#include <cmath>

namespace llaminar
{

    MPIEmbeddingKernel::MPIEmbeddingKernel(size_t vocab_size, size_t embedding_dim)
        : vocab_size_(vocab_size), embedding_dim_(embedding_dim)
    {
        initializeMPI();

        // Calculate vocabulary partition for this rank
        size_t base_partition_size = vocab_size_ / size_;
        size_t remainder = vocab_size_ % size_;

        // Distribute remainder among first 'remainder' ranks
        if (rank_ < static_cast<int>(remainder))
        {
            local_vocab_size_ = base_partition_size + 1;
            local_vocab_start_ = rank_ * local_vocab_size_;
        }
        else
        {
            local_vocab_size_ = base_partition_size;
            local_vocab_start_ = remainder * (base_partition_size + 1) +
                                 (rank_ - remainder) * base_partition_size;
        }

        local_vocab_end_ = local_vocab_start_ + local_vocab_size_;

        LOG_DEBUG("MPIEmbeddingKernel initialized on rank " << rank_ << "/" << size_
                                                            << " with vocab_size=" << vocab_size << ", embedding_dim=" << embedding_dim
                                                            << ", local_vocab_range=[" << local_vocab_start_ << ", " << local_vocab_end_ << ")");
    }

    bool MPIEmbeddingKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        if (!validate(inputs, outputs))
        {
            LOG_ERROR("MPIEmbeddingKernel validation failed");
            return false;
        }

        auto token_ids_tensor = inputs[0];
        auto embedding_table = inputs[1];
        auto output = outputs[0];

        size_t seq_len = token_ids_tensor->shape()[0];

        // Extract token IDs
        std::vector<int> token_ids(seq_len);
        std::copy(token_ids_tensor->data(), token_ids_tensor->data() + seq_len, token_ids.begin());

        // Create local output buffer for this rank's contribution
        auto local_output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(embedding_dim_)});
        std::fill(local_output->data(), local_output->data() + seq_len * embedding_dim_, 0.0f);

        // Perform local embedding lookup for tokens owned by this rank
        computeLocalEmbedding(token_ids.data(), embedding_table->data(),
                              local_output->data(), seq_len, embedding_dim_);

        // Gather embeddings from all ranks
        gatherEmbeddings(local_output, output, seq_len, embedding_dim_);

        return true;
    }

    bool MPIEmbeddingKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 2 || outputs.size() != 1)
        {
            LOG_ERROR("MPIEmbeddingKernel: Expected 2 inputs and 1 output, got "
                      << inputs.size() << " inputs and " << outputs.size() << " outputs");
            return false;
        }

        auto token_ids = inputs[0];
        auto embedding_table = inputs[1];
        auto output = outputs[0];

        if (!token_ids || !embedding_table || !output)
        {
            LOG_ERROR("MPIEmbeddingKernel: Null tensor provided");
            return false;
        }

        // Check token_ids is 1D
        if (token_ids->shape().size() != 1)
        {
            LOG_ERROR("MPIEmbeddingKernel: Token IDs must be 1D, got "
                      << token_ids->shape().size() << " dimensions");
            return false;
        }

        // Check embedding table is 2D [local_vocab_size, embedding_dim]
        if (embedding_table->shape().size() != 2 ||
            embedding_table->shape()[0] != local_vocab_size_ ||
            embedding_table->shape()[1] != embedding_dim_)
        {
            LOG_ERROR("MPIEmbeddingKernel: Local embedding table shape mismatch. Expected ["
                      << local_vocab_size_ << ", " << embedding_dim_ << "], got ["
                      << embedding_table->shape()[0] << ", " << embedding_table->shape()[1] << "]");
            return false;
        }

        // Check output is 2D [seq_len, embedding_dim]
        size_t seq_len = token_ids->shape()[0];
        if (output->shape().size() != 2 ||
            output->shape()[0] != seq_len ||
            output->shape()[1] != embedding_dim_)
        {
            LOG_ERROR("MPIEmbeddingKernel: Output shape mismatch. Expected ["
                      << seq_len << ", " << embedding_dim_ << "], got ["
                      << output->shape()[0] << ", " << output->shape()[1] << "]");
            return false;
        }

        return true;
    }

    void MPIEmbeddingKernel::computeLocalEmbedding(const int *token_ids, const float *local_embedding_table,
                                                   float *output, size_t seq_len, size_t embedding_dim)
    {
        for (size_t i = 0; i < seq_len; ++i)
        {
            int token_id = token_ids[i];

            // Check if this token belongs to this rank's vocabulary partition
            if (token_id >= static_cast<int>(local_vocab_start_) &&
                token_id < static_cast<int>(local_vocab_end_))
            {
                // Convert global token ID to local token ID
                int local_token_id = getLocalTokenId(token_id);

                if (local_token_id >= 0 && local_token_id < static_cast<int>(local_vocab_size_))
                {
                    // Copy embedding for this token from local table
                    const float *embedding = local_embedding_table + local_token_id * embedding_dim;
                    std::copy(embedding, embedding + embedding_dim,
                              output + i * embedding_dim);

                    LOG_DEBUG("Rank " << rank_ << " processed token " << token_id
                                      << " (local_id=" << local_token_id << ") at position " << i);
                }
                else
                {
                    LOG_WARN("MPIEmbeddingKernel: Invalid local token ID " << local_token_id
                                                                           << " for global token " << token_id);
                }
            }
            // For tokens not owned by this rank, output remains zero (will be filled by other ranks)
        }
    }

    void MPIEmbeddingKernel::gatherEmbeddings(const std::shared_ptr<TensorBase> &local_output,
                                              std::shared_ptr<TensorBase> &global_output,
                                              size_t seq_len, size_t embedding_dim)
    {
        // Use MPI_Allreduce to sum contributions from all ranks
        // Each rank contributes zeros for tokens it doesn't own, and actual embeddings for tokens it owns
        checkMPIError(MPI_Allreduce(local_output->data(), global_output->data(),
                                    static_cast<int>(seq_len * embedding_dim), MPI_FLOAT, MPI_SUM, getComm()),
                      "MPI_Allreduce in gatherEmbeddings");

        LOG_DEBUG("Rank " << rank_ << " completed embedding gather for " << seq_len
                          << " tokens with " << embedding_dim << " dimensions");
    }

    int MPIEmbeddingKernel::getTokenRank(int token_id) const
    {
        if (token_id < 0 || token_id >= static_cast<int>(vocab_size_))
        {
            return -1; // Invalid token ID
        }

        size_t base_partition_size = vocab_size_ / size_;
        size_t remainder = vocab_size_ % size_;

        // Tokens 0 to (remainder * (base_partition_size + 1) - 1) are in ranks 0 to remainder-1
        size_t threshold = remainder * (base_partition_size + 1);

        if (token_id < static_cast<int>(threshold))
        {
            return token_id / (base_partition_size + 1);
        }
        else
        {
            return remainder + (token_id - threshold) / base_partition_size;
        }
    }

    int MPIEmbeddingKernel::getLocalTokenId(int global_token_id) const
    {
        if (global_token_id < static_cast<int>(local_vocab_start_) ||
            global_token_id >= static_cast<int>(local_vocab_end_))
        {
            return -1; // Token not owned by this rank
        }

        return global_token_id - static_cast<int>(local_vocab_start_);
    }

} // namespace llaminar