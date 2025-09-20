#include "MPIAttentionKernel.h"
#include "../logger.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace llaminar
{

    MPIAttentionKernel::MPIAttentionKernel(int n_head, int n_head_kv, int head_dim,
                                           DistributionStrategy strategy)
        : MPIKernelBase(), n_head_(n_head), n_head_kv_(n_head_kv), head_dim_(head_dim),
          n_past_(0), strategy_(strategy)
    {
        if (n_head_ % getSize() != 0)
        {
            LOG_WARN("Number of heads (" << n_head_ << ") not evenly divisible by MPI size ("
                                         << getSize() << "). Load balancing may be suboptimal.");
        }
    }

    bool MPIAttentionKernel::execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                     std::vector<std::shared_ptr<TensorBase>> &outputs)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (!validate(inputs, outputs))
        {
            return false;
        }

        // Extract inputs: input, wq, wk, wv, wo, k_cache, v_cache
        auto global_input = inputs[0];
        auto global_wq = inputs[1];
        auto global_wk = inputs[2];
        auto global_wv = inputs[3];
        auto global_wo = inputs[4];
        auto k_cache = inputs[5]; // TODO: Handle KV cache in future version
        auto v_cache = inputs[6]; // TODO: Handle KV cache in future version
        auto global_output = outputs[0];

        size_t seq_len = static_cast<size_t>(global_input->shape()[0]);
        size_t d_model = static_cast<size_t>(global_input->shape()[1]);

        // Get local head distribution
        auto [local_heads, head_offset] = getHeadDistribution();
        size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);

        // Create local weight tensors for distributed heads
        auto local_wq = createLocalTensor({d_model, local_head_dim});
        auto local_wk = createLocalTensor({d_model, local_head_dim});
        auto local_wv = createLocalTensor({d_model, local_head_dim});
        auto local_wo = createLocalTensor({local_head_dim, d_model});

        // Distribute weights according to head assignment
        distributeInputs(global_input, global_wq, global_wk, global_wv, global_wo,
                         local_wq, local_wk, local_wv, local_wo, seq_len, d_model);

        // Create local projection tensors
        auto local_q = createLocalTensor({seq_len, local_head_dim});
        auto local_k = createLocalTensor({seq_len, local_head_dim});
        auto local_v = createLocalTensor({seq_len, local_head_dim});

        // Compute Q, K, V projections for local heads using COSMA
        computeLocalProjections(global_input, local_wq, local_wk, local_wv,
                                local_q, local_k, local_v, seq_len, d_model);

        // Apply RoPE to local Q and K
        applyLocalRoPE(local_q->data(), local_k->data(), seq_len, local_heads);

        // Create local attended output tensor
        auto local_attended_output = createLocalTensor({seq_len, local_head_dim});

        // Compute attention for local heads
        computeLocalAttention(local_q->data(), local_k->data(), local_v->data(),
                              local_attended_output->data(), seq_len, local_heads);

        // Create local final output tensor
        auto local_final_output = createLocalTensor({seq_len, d_model});

        // Compute output projection for local heads using COSMA
        computeLocalOutputProjection(local_attended_output, local_wo,
                                     local_final_output, seq_len, local_heads);

        // Gather final outputs from all processes
        gatherOutput(local_final_output, global_output, seq_len, d_model);

        auto end = std::chrono::high_resolution_clock::now();
        double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

        LOG_DEBUG("MPIAttention executed in " + std::to_string(execution_time) +
                  " ms on rank " + std::to_string(getRank()) +
                  " (local_heads=" + std::to_string(local_heads) + ")");
        return true;
    }

    bool MPIAttentionKernel::validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const
    {
        if (inputs.size() != 7)
        {
            LOG_ERROR("MPIAttentionKernel: Expected 7 inputs (input, wq, wk, wv, wo, k_cache, v_cache), got " << inputs.size());
            return false;
        }

        if (outputs.size() != 1)
        {
            LOG_ERROR("MPIAttentionKernel: Expected 1 output, got " << outputs.size());
            return false;
        }

        auto input = inputs[0];
        auto wq = inputs[1];
        auto wk = inputs[2];
        auto wv = inputs[3];
        auto wo = inputs[4];
        auto output = outputs[0];

        if (input->shape().size() != 2)
        {
            LOG_ERROR("MPIAttentionKernel: Input must be 2D [seq_len, d_model], got shape size " << input->shape().size());
            return false;
        }

        size_t seq_len = static_cast<size_t>(input->shape()[0]);
        size_t d_model = static_cast<size_t>(input->shape()[1]);
        size_t total_head_dim = static_cast<size_t>(n_head_ * head_dim_);

        // Validate weight dimensions
        if (wq->shape().size() != 2 || static_cast<size_t>(wq->shape()[0]) != d_model ||
            static_cast<size_t>(wq->shape()[1]) != total_head_dim)
        {
            LOG_ERROR("MPIAttentionKernel: Query weight dimension mismatch");
            return false;
        }

        if (wk->shape().size() != 2 || static_cast<size_t>(wk->shape()[0]) != d_model ||
            static_cast<size_t>(wk->shape()[1]) != static_cast<size_t>(n_head_kv_ * head_dim_))
        {
            LOG_ERROR("MPIAttentionKernel: Key weight dimension mismatch");
            return false;
        }

        if (wv->shape().size() != 2 || static_cast<size_t>(wv->shape()[0]) != d_model ||
            static_cast<size_t>(wv->shape()[1]) != static_cast<size_t>(n_head_kv_ * head_dim_))
        {
            LOG_ERROR("MPIAttentionKernel: Value weight dimension mismatch");
            return false;
        }

        if (wo->shape().size() != 2 || static_cast<size_t>(wo->shape()[0]) != total_head_dim ||
            static_cast<size_t>(wo->shape()[1]) != d_model)
        {
            LOG_ERROR("MPIAttentionKernel: Output weight dimension mismatch");
            return false;
        }

        // Validate output dimensions
        if (output->shape().size() != 2 || static_cast<size_t>(output->shape()[0]) != seq_len ||
            static_cast<size_t>(output->shape()[1]) != d_model)
        {
            LOG_ERROR("MPIAttentionKernel: Output dimension mismatch");
            return false;
        }

        return true;
    }

    void MPIAttentionKernel::setHeadDimensions(int n_head, int n_head_kv, int head_dim)
    {
        n_head_ = n_head;
        n_head_kv_ = n_head_kv;
        head_dim_ = head_dim;

        if (n_head_ % getSize() != 0)
        {
            LOG_WARN("Number of heads (" << n_head_ << ") not evenly divisible by MPI size ("
                                         << getSize() << "). Load balancing may be suboptimal.");
        }
    }

    std::pair<int, int> MPIAttentionKernel::getHeadDistribution() const
    {
        return getHeadDistribution(getRank());
    }

    std::pair<int, int> MPIAttentionKernel::getHeadDistribution(int rank) const
    {
        int heads_per_rank = n_head_ / getSize();
        int remainder = n_head_ % getSize();

        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
        int head_offset = rank * heads_per_rank + std::min(rank, remainder);

        return {local_heads, head_offset};
    }

    void MPIAttentionKernel::distributeInputs(const std::shared_ptr<TensorBase> &global_input,
                                              const std::shared_ptr<TensorBase> &global_wq,
                                              const std::shared_ptr<TensorBase> &global_wk,
                                              const std::shared_ptr<TensorBase> &global_wv,
                                              const std::shared_ptr<TensorBase> &global_wo,
                                              std::shared_ptr<TensorBase> &local_wq,
                                              std::shared_ptr<TensorBase> &local_wk,
                                              std::shared_ptr<TensorBase> &local_wv,
                                              std::shared_ptr<TensorBase> &local_wo,
                                              size_t seq_len, size_t d_model)
    {
        auto [local_heads, head_offset] = getHeadDistribution();
        size_t local_head_dim = static_cast<size_t>(local_heads * head_dim_);
        size_t head_offset_dim = static_cast<size_t>(head_offset * head_dim_);

        // Extract local query weights (columns for assigned heads)
        for (size_t i = 0; i < d_model; ++i)
        {
            const float *global_row = global_wq->data() + i * n_head_ * head_dim_;
            float *local_row = local_wq->data() + i * local_head_dim;
            memcpy(local_row, global_row + head_offset_dim, local_head_dim * sizeof(float));
        }

        // For grouped attention, handle KV heads distribution
        // For simplicity, assuming n_head_kv == n_head for now
        // TODO: Implement proper grouped attention support
        for (size_t i = 0; i < d_model; ++i)
        {
            const float *global_k_row = global_wk->data() + i * n_head_kv_ * head_dim_;
            const float *global_v_row = global_wv->data() + i * n_head_kv_ * head_dim_;
            float *local_k_row = local_wk->data() + i * local_head_dim;
            float *local_v_row = local_wv->data() + i * local_head_dim;
            memcpy(local_k_row, global_k_row + head_offset_dim, local_head_dim * sizeof(float));
            memcpy(local_v_row, global_v_row + head_offset_dim, local_head_dim * sizeof(float));
        }

        // Extract local output weights (rows for assigned heads)
        for (size_t i = 0; i < local_head_dim; ++i)
        {
            const float *global_row = global_wo->data() + (head_offset_dim + i) * d_model;
            float *local_row = local_wo->data() + i * d_model;
            memcpy(local_row, global_row, d_model * sizeof(float));
        }

        LOG_DEBUG("Distributed weights: local_heads=" << local_heads << ", head_offset=" << head_offset << " on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalProjections(const std::shared_ptr<TensorBase> &input,
                                                     const std::shared_ptr<TensorBase> &local_wq,
                                                     const std::shared_ptr<TensorBase> &local_wk,
                                                     const std::shared_ptr<TensorBase> &local_wv,
                                                     std::shared_ptr<TensorBase> &local_q,
                                                     std::shared_ptr<TensorBase> &local_k,
                                                     std::shared_ptr<TensorBase> &local_v,
                                                     size_t seq_len, size_t d_model)
    {
        // Compute Q = input @ local_wq using COSMA
        std::vector<std::shared_ptr<TensorBase>> q_inputs = {input, local_wq};
        std::vector<std::shared_ptr<TensorBase>> q_outputs = {local_q};
        if (!matmul_kernel_.execute(q_inputs, q_outputs))
        {
            LOG_ERROR("COSMA Q projection failed on rank " << getRank());
            return;
        }

        // Compute K = input @ local_wk using COSMA
        std::vector<std::shared_ptr<TensorBase>> k_inputs = {input, local_wk};
        std::vector<std::shared_ptr<TensorBase>> k_outputs = {local_k};
        if (!matmul_kernel_.execute(k_inputs, k_outputs))
        {
            LOG_ERROR("COSMA K projection failed on rank " << getRank());
            return;
        }

        // Compute V = input @ local_wv using COSMA
        std::vector<std::shared_ptr<TensorBase>> v_inputs = {input, local_wv};
        std::vector<std::shared_ptr<TensorBase>> v_outputs = {local_v};
        if (!matmul_kernel_.execute(v_inputs, v_outputs))
        {
            LOG_ERROR("COSMA V projection failed on rank " << getRank());
            return;
        }

        auto [local_heads, head_offset] = getHeadDistribution();
        LOG_DEBUG("Computed local projections using COSMA for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalAttention(const float *local_q, const float *local_k, const float *local_v,
                                                   float *local_output, size_t seq_len, int local_heads)
    {
        // Create temporary storage for attention scores
        size_t scores_size = seq_len * seq_len * static_cast<size_t>(local_heads);
        auto scores = std::make_unique<float[]>(scores_size);

        // Compute attention scores and apply softmax
        computeLocalAttentionScores(local_q, local_k, scores.get(), seq_len, local_heads);

        // Apply attention to values
        applyLocalAttention(scores.get(), local_v, local_output, seq_len, local_heads);

        LOG_DEBUG("Computed local attention for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::applyLocalRoPE(float *local_q, float *local_k, size_t seq_len, int local_heads)
    {
        // Simplified RoPE implementation - apply rotation to each head
        float theta_base = 10000.0f;

        for (int head = 0; head < local_heads; ++head)
        {
            for (size_t seq = 0; seq < seq_len; ++seq)
            {
                float *q_head = local_q + seq * local_heads * head_dim_ + head * head_dim_;
                float *k_head = local_k + seq * local_heads * head_dim_ + head * head_dim_;

                for (int dim_pair = 0; dim_pair < head_dim_ / 2; ++dim_pair)
                {
                    float theta = 1.0f / std::pow(theta_base, (2.0f * dim_pair) / head_dim_);
                    float cos_theta = std::cos((n_past_ + static_cast<float>(seq)) * theta);
                    float sin_theta = std::sin((n_past_ + static_cast<float>(seq)) * theta);

                    // Apply rotation to Q
                    float q0 = q_head[2 * dim_pair];
                    float q1 = q_head[2 * dim_pair + 1];
                    q_head[2 * dim_pair] = q0 * cos_theta - q1 * sin_theta;
                    q_head[2 * dim_pair + 1] = q0 * sin_theta + q1 * cos_theta;

                    // Apply rotation to K
                    float k0 = k_head[2 * dim_pair];
                    float k1 = k_head[2 * dim_pair + 1];
                    k_head[2 * dim_pair] = k0 * cos_theta - k1 * sin_theta;
                    k_head[2 * dim_pair + 1] = k0 * sin_theta + k1 * cos_theta;
                }
            }
        }

        LOG_DEBUG("Applied RoPE to " << local_heads << " local heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalAttentionScores(const float *local_q, const float *local_k, float *scores,
                                                         size_t seq_len, int local_heads)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        for (int head = 0; head < local_heads; ++head)
        {
            // Compute Q @ K^T for this head
            for (size_t i = 0; i < seq_len; ++i)
            {
                const float *q_row = local_q + i * local_heads * head_dim_ + head * head_dim_;

                for (size_t j = 0; j < seq_len; ++j)
                {
                    const float *k_row = local_k + j * local_heads * head_dim_ + head * head_dim_;

                    float score = 0.0f;
                    for (int d = 0; d < head_dim_; ++d)
                    {
                        score += q_row[d] * k_row[d];
                    }

                    scores[head * seq_len * seq_len + i * seq_len + j] = score * scale;
                }
            }

            // Apply causal mask and softmax for this head
            for (size_t i = 0; i < seq_len; ++i)
            {
                float *score_row = scores + head * seq_len * seq_len + i * seq_len;

                // Apply causal mask
                for (size_t j = i + 1; j < seq_len; ++j)
                {
                    score_row[j] = -INFINITY;
                }

                // Compute softmax
                float max_score = -INFINITY;
                for (size_t j = 0; j <= i; ++j)
                {
                    max_score = std::max(max_score, score_row[j]);
                }

                float sum_exp = 0.0f;
                for (size_t j = 0; j <= i; ++j)
                {
                    score_row[j] = std::exp(score_row[j] - max_score);
                    sum_exp += score_row[j];
                }

                for (size_t j = 0; j <= i; ++j)
                {
                    score_row[j] /= sum_exp;
                }

                // Set masked positions to 0
                for (size_t j = i + 1; j < seq_len; ++j)
                {
                    score_row[j] = 0.0f;
                }
            }
        }

        LOG_DEBUG("Computed attention scores for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::applyLocalAttention(const float *scores, const float *local_v, float *local_attended_output,
                                                 size_t seq_len, int local_heads)
    {
        // Compute attention @ values for each head
        for (int head = 0; head < local_heads; ++head)
        {
            for (size_t i = 0; i < seq_len; ++i)
            {
                const float *score_row = scores + head * seq_len * seq_len + i * seq_len;
                float *output_row = local_attended_output + i * local_heads * head_dim_ + head * head_dim_;

                // Initialize output row to zero
                for (int d = 0; d < head_dim_; ++d)
                {
                    output_row[d] = 0.0f;
                }

                // Compute weighted sum of values
                for (size_t j = 0; j < seq_len; ++j)
                {
                    const float *v_row = local_v + j * local_heads * head_dim_ + head * head_dim_;
                    float weight = score_row[j];

                    for (int d = 0; d < head_dim_; ++d)
                    {
                        output_row[d] += weight * v_row[d];
                    }
                }
            }
        }

        LOG_DEBUG("Applied attention to values for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::computeLocalOutputProjection(const std::shared_ptr<TensorBase> &local_attended_output,
                                                          const std::shared_ptr<TensorBase> &local_wo,
                                                          std::shared_ptr<TensorBase> &local_final_output,
                                                          size_t seq_len, int local_heads)
    {
        // Compute attended_output @ local_wo using COSMA
        std::vector<std::shared_ptr<TensorBase>> inputs = {local_attended_output, local_wo};
        std::vector<std::shared_ptr<TensorBase>> outputs = {local_final_output};

        if (!matmul_kernel_.execute(inputs, outputs))
        {
            LOG_ERROR("COSMA output projection failed on rank " << getRank());
            return;
        }

        LOG_DEBUG("Computed output projection using COSMA for " << local_heads << " heads on rank " << getRank());
    }

    void MPIAttentionKernel::gatherOutput(const std::shared_ptr<TensorBase> &local_output,
                                          std::shared_ptr<TensorBase> &global_output,
                                          size_t seq_len, size_t d_model)
    {
        // Sum all local outputs to get final result using MPI_Allreduce
        checkMPIError(MPI_Allreduce(local_output->data(), global_output->data(),
                                    static_cast<int>(seq_len * d_model), MPI_FLOAT, MPI_SUM, getComm()),
                      "MPI_Allreduce in gatherOutput");

        LOG_DEBUG("Gathered output: [" << seq_len << ", " << d_model << "] on rank " << getRank());
    }

} // namespace llaminar