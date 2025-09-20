#pragma once

#include "../kernel_base.h"
#include <string>
#include <vector>
#include <memory>

namespace llaminar
{

    /**
     * @brief Multi-head attention kernel for transformer attention mechanism
     */
    class AttentionKernel : public KernelBase
    {
    public:
        AttentionKernel(int n_head, int n_head_kv, int head_dim);
        ~AttentionKernel() = default;

        // KernelBase interface implementation
        bool execute(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                     std::vector<std::shared_ptr<TensorBase>> &outputs) override;

        bool validate(const std::vector<std::shared_ptr<TensorBase>> &inputs,
                      const std::vector<std::shared_ptr<TensorBase>> &outputs) const override;

        std::string getKernelType() const override { return "Attention"; }
        size_t getExpectedInputCount() const override { return 7; } // input, wq, wk, wv, wo, k_cache, v_cache
        size_t getExpectedOutputCount() const override { return 1; }

        // Configuration methods
        void setHeadDimensions(int n_head, int n_head_kv, int head_dim);
        void setSequencePosition(int n_past) { n_past_ = n_past; }

    private:
        /**
         * @brief Compute query projections
         * @param input Input data pointer
         * @param wq Query weight matrix
         * @param query Output query matrix
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void computeQueries(const float *input, const float *wq, float *query,
                            int seq_len, int d_model);

        /**
         * @brief Compute key projections
         * @param input Input data pointer
         * @param wk Key weight matrix
         * @param key Output key matrix
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void computeKeys(const float *input, const float *wk, float *key,
                         int seq_len, int d_model);

        /**
         * @brief Compute value projections
         * @param input Input data pointer
         * @param wv Value weight matrix
         * @param value Output value matrix
         * @param seq_len Sequence length
         * @param d_model Model dimension
         */
        void computeValues(const float *input, const float *wv, float *value,
                           int seq_len, int d_model);

        /**
         * @brief Apply rotary position embedding (RoPE)
         * @param tensor Input tensor to apply RoPE to
         * @param seq_len Sequence length
         * @param head_dim Head dimension
         * @param n_past Number of past tokens for position
         */
        void applyRoPE(float *tensor, int seq_len, int head_dim, int n_past);

        /**
         * @brief Compute attention scores and apply softmax
         * @param query Query matrix
         * @param key Key matrix
         * @param scores Output attention scores
         * @param seq_len Sequence length
         * @param head_dim Head dimension
         */
        void computeAttentionScores(const float *query, const float *key, float *scores,
                                    int seq_len, int head_dim);

        /**
         * @brief Apply attention scores to values
         * @param scores Attention scores
         * @param value Value matrix
         * @param output Output matrix
         * @param seq_len Sequence length
         * @param head_dim Head dimension
         */
        void applyAttention(const float *scores, const float *value, float *output,
                            int seq_len, int head_dim);

        int n_head_;    // Number of attention heads
        int n_head_kv_; // Number of key-value heads (for grouped attention)
        int head_dim_;  // Dimension per attention head
        int n_past_;    // Number of past tokens for position embedding
    };

} // namespace llaminar