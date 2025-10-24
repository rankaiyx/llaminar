/**
 * @file PipelineBase.cpp
 * @brief Base pipeline implementation
 * @author David Sanftenberg
 */

#include "PipelineBase.h"
#include "AttentionUtils.h"
#include "../tensors/TensorFactory.h"
#include <iostream>
#include <cstring>
#include <vector>

namespace llaminar2
{

    PipelineBase::PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                               std::shared_ptr<MPIContext> mpi_ctx,
                               int device_idx)
        : model_ctx_(model_ctx), mpi_ctx_(mpi_ctx), device_idx_(device_idx)
    {
        if (!model_ctx_)
        {
            throw std::runtime_error("PipelineBase: model_ctx cannot be null");
        }

        model_path_ = model_ctx_->path();

        std::cout << "[PipelineBase] Initializing with model: " << model_path_ << "\n";

        if (mpi_ctx_)
        {
            std::cout << "[PipelineBase] MPI context provided, rank "
                      << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size() << "\n";
        }

        if (device_idx_ >= 0)
        {
            std::cout << "[PipelineBase] Device index: " << device_idx_ << " (GPU)\n";
        }
        else
        {
            std::cout << "[PipelineBase] Device index: " << device_idx_ << " (CPU)\n";
        }
    }

    bool PipelineBase::attention_gqa(
        const float *Q, const float *K, const float *V, float *output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size)
    {
        // Validate inputs
        if (!Q || !K || !V || !output)
        {
            std::cerr << "[PipelineBase] attention_gqa: null pointer\n";
            return false;
        }

        if (n_heads % n_kv_heads != 0)
        {
            std::cerr << "[PipelineBase] attention_gqa: n_heads (" << n_heads
                      << ") must be divisible by n_kv_heads (" << n_kv_heads << ")\n";
            return false;
        }

        // Broadcast K/V heads to match Q heads (if needed)
        std::vector<float> K_broadcast;
        std::vector<float> V_broadcast;
        const float *K_expanded = K;
        const float *V_expanded = V;

        if (n_kv_heads < n_heads)
        {
            // Need to broadcast K/V
            K_broadcast.resize(seq_len * n_heads * head_dim);
            V_broadcast.resize(seq_len * n_heads * head_dim);

            attention_utils::broadcast_kv_heads(
                K, K_broadcast.data(),
                seq_len, n_heads, n_kv_heads, head_dim);

            attention_utils::broadcast_kv_heads(
                V, V_broadcast.data(),
                seq_len, n_heads, n_kv_heads, head_dim);

            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // Allocate attention scores: [n_heads, seq_len, seq_len]
        std::vector<float> scores(n_heads * seq_len * seq_len);

        // Compute attention scores per head: Q @ K^T
        // For each head h:
        //   scores[h] = Q[h] @ K[h]^T
        //   Q[h]: [seq_len, head_dim]
        //   K[h]^T: [head_dim, seq_len]
        //   scores[h]: [seq_len, seq_len]

        for (int h = 0; h < n_heads; ++h)
        {
            const float *Q_h = Q + h * head_dim; // Stride: n_heads * head_dim
            const float *K_h = K_expanded + h * head_dim;
            float *scores_h = scores.data() + h * seq_len * seq_len;

            // Simple GEMM: scores_h = Q_h @ K_h^T
            // TODO: Replace with ITensorGemm kernel for better performance
            for (int i = 0; i < seq_len; ++i)
            {
                for (int j = 0; j < seq_len; ++j)
                {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        sum += Q_h[i * n_heads * head_dim + d] * K_h[j * n_heads * head_dim + d];
                    }
                    scores_h[i * seq_len + j] = sum;
                }
            }
        }

        // Scale scores by 1/sqrt(head_dim)
        attention_utils::scale_attention_scores(
            scores.data(), scores.size(), head_dim);

        // Apply causal mask (if enabled)
        if (causal)
        {
            std::vector<float> mask(seq_len * seq_len);
            attention_utils::create_causal_mask(mask.data(), seq_len, window_size);

            // Apply mask to each head
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores.data() + h * seq_len * seq_len;
                attention_utils::apply_attention_mask(scores_h, mask.data(), seq_len, seq_len);
            }
        }

        // Softmax over scores (per head, per row)
        // TODO: Replace with ITensorSoftmax kernel
        for (int h = 0; h < n_heads; ++h)
        {
            float *scores_h = scores.data() + h * seq_len * seq_len;

            for (int i = 0; i < seq_len; ++i)
            {
                float *row = scores_h + i * seq_len;

                // Find max for numerical stability
                float max_val = row[0];
                for (int j = 1; j < seq_len; ++j)
                {
                    if (row[j] > max_val)
                        max_val = row[j];
                }

                // Exp and sum
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j)
                {
                    row[j] = std::exp(row[j] - max_val);
                    sum += row[j];
                }

                // Normalize
                for (int j = 0; j < seq_len; ++j)
                {
                    row[j] /= sum;
                }
            }
        }

        // Compute context: scores @ V
        // For each head h:
        //   context[h] = scores[h] @ V[h]
        //   scores[h]: [seq_len, seq_len]
        //   V[h]: [seq_len, head_dim]
        //   context[h]: [seq_len, head_dim]

        std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

        for (int h = 0; h < n_heads; ++h)
        {
            const float *scores_h = scores.data() + h * seq_len * seq_len;
            const float *V_h = V_expanded + h * head_dim;
            float *output_h = output + h * head_dim;

            // Simple GEMM: output_h = scores_h @ V_h
            // TODO: Replace with ITensorGemm kernel
            for (int i = 0; i < seq_len; ++i)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j)
                    {
                        sum += scores_h[i * seq_len + j] * V_h[j * n_heads * head_dim + d];
                    }
                    output_h[i * n_heads * head_dim + d] = sum;
                }
            }
        }

        return true;
    }

} // namespace llaminar2
