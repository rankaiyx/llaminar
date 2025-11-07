/**
 * @file INT8AttentionKernel.cpp
 * @brief INT8 multi-head attention kernel implementation
 * @author David Sanftenberg
 * @date 2025-11-06
 */

#include "INT8AttentionKernel.h"
#include "../../utils/Logger.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cblas.h>

namespace llaminar2
{

    INT8AttentionKernel::INT8AttentionKernel(int n_heads, int d_head, int device_idx)
        : n_heads_(n_heads), d_head_(d_head), device_idx_(device_idx)
    {
    }

    bool INT8AttentionKernel::forward(
        const int8_t *q_int8,
        const float *q_row_scales,
        const int8_t *k_int8,
        const float *k_row_scales,
        const int8_t *v_int8,
        const float *v_row_scales,
        int8_t *output_int8,
        float *output_row_scales,
        int batch,
        int seq_len,
        bool use_causal_mask,
        float eps)
    {
        // Validate device
        if (device_idx_ != -1)
        {
            LOG_ERROR("INT8AttentionKernel only supports CPU (device_idx=-1), got " << device_idx_);
            return false;
        }

        // Validate pointers
        if (!q_int8 || !q_row_scales || !k_int8 || !k_row_scales ||
            !v_int8 || !v_row_scales || !output_int8 || !output_row_scales)
        {
            LOG_ERROR("INT8AttentionKernel: null pointer(s) detected");
            return false;
        }

        // Validate dimensions
        if (batch <= 0 || seq_len <= 0 || n_heads_ <= 0 || d_head_ <= 0)
        {
            LOG_ERROR("INT8AttentionKernel: invalid dimensions: batch=" << batch
                                                                        << " seq_len=" << seq_len
                                                                        << " n_heads=" << n_heads_
                                                                        << " d_head=" << d_head_);
            return false;
        }

        // Allocate temporary buffers
        size_t scores_size = batch * n_heads_ * seq_len * seq_len;
        size_t context_size = batch * n_heads_ * seq_len * d_head_;

        scores_buffer_.resize(scores_size);
        scores_fp32_buffer_.resize(scores_size);
        attn_weights_buffer_.resize(scores_size);
        attn_weights_scales_buffer_.resize(batch * n_heads_ * seq_len);
        context_buffer_.resize(context_size);

        // Initialize scores_fp32_buffer_ to zero (important!)
        std::fill(scores_fp32_buffer_.begin(), scores_fp32_buffer_.end(), 0.0f);

        // Step 1: Compute attention scores Q @ K^T (populates scores_fp32_buffer_)
        compute_scores(q_int8, q_row_scales, k_int8, k_row_scales, batch, seq_len);

        // Step 2: Apply softmax (FP32 → INT8)
        apply_softmax(attn_weights_buffer_.data(),
                      attn_weights_scales_buffer_.data(),
                      batch, seq_len, use_causal_mask, eps);

        // Step 3: Compute context attn_weights @ V
        compute_context(attn_weights_buffer_.data(),
                        attn_weights_scales_buffer_.data(),
                        v_int8, v_row_scales,
                        context_buffer_.data(),
                        batch, seq_len);

        // Step 4: Requantize output INT32 → INT8
        requantize_output(context_buffer_.data(), output_int8, output_row_scales,
                          batch, seq_len);

        return true;
    }

    void INT8AttentionKernel::compute_scores(
        const int8_t *q_int8,
        const float *q_row_scales,
        const int8_t *k_int8,
        const float *k_row_scales,
        int batch,
        int seq_len)
    {
        // For each batch and head, compute: Q[seq_len, d_head] @ K^T[d_head, seq_len]
        // Result: scores[seq_len, seq_len] per head
        // Populates scores_fp32_buffer_ directly

        int d_model = n_heads_ * d_head_;

        for (int b = 0; b < batch; ++b)
        {
            for (int h = 0; h < n_heads_; ++h)
            {
                // Extract Q and K for this head
                // Layout: [batch, seq_len, n_heads, d_head] (interleaved)
                for (int i = 0; i < seq_len; ++i)
                {
                    for (int j = 0; j < seq_len; ++j)
                    {
                        // Compute dot product: Q[i] · K[j]
                        int64_t dot = 0;

                        for (int d = 0; d < d_head_; ++d)
                        {
                            // Q index: [b, i, h * d_head + d]
                            int q_idx = (b * seq_len + i) * d_model + h * d_head_ + d;
                            // K index: [b, j, h * d_head + d]
                            int k_idx = (b * seq_len + j) * d_model + h * d_head_ + d;

                            dot += static_cast<int64_t>(q_int8[q_idx]) *
                                   static_cast<int64_t>(k_int8[k_idx]);
                        }

                        // Scale by row scales (Q[i] scale * K[j] scale)
                        float q_scale = q_row_scales[b * seq_len + i];
                        float k_scale = k_row_scales[b * seq_len + j];
                        float combined_scale = q_scale * k_scale;

                        // Store scaled FP32 score directly
                        int score_idx = ((b * n_heads_ + h) * seq_len + i) * seq_len + j;
                        scores_fp32_buffer_[score_idx] = static_cast<float>(dot) * combined_scale;
                    }
                }
            }
        }
    }

    void INT8AttentionKernel::apply_softmax(
        int8_t *attn_weights_int8,
        float *attn_weights_row_scales,
        int batch,
        int seq_len,
        bool use_causal_mask,
        float eps)
    {
        // Softmax is applied per-row: softmax(scores[i, :]) for each query position i
        // We operate in FP32 for numerical stability
        // Reads from scores_fp32_buffer_, outputs to attn_weights_int8

        float sqrt_d_head = std::sqrt(static_cast<float>(d_head_));

        for (int b = 0; b < batch; ++b)
        {
            for (int h = 0; h < n_heads_; ++h)
            {
                for (int i = 0; i < seq_len; ++i)
                {
                    // Get row base index
                    int row_base = ((b * n_heads_ + h) * seq_len + i) * seq_len;

                    // Step 1: Find max value in row (for numerical stability)
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < seq_len; ++j)
                    {
                        if (use_causal_mask && j > i)
                        {
                            continue; // Skip future positions
                        }
                        float score = scores_fp32_buffer_[row_base + j] / sqrt_d_head;
                        max_val = std::max(max_val, score);
                    }

                    // Step 2: Compute exp(score - max) and sum
                    float sum_exp = 0.0f;
                    std::vector<float> exp_scores(seq_len);

                    for (int j = 0; j < seq_len; ++j)
                    {
                        if (use_causal_mask && j > i)
                        {
                            exp_scores[j] = 0.0f; // Masked positions
                        }
                        else
                        {
                            float score = scores_fp32_buffer_[row_base + j] / sqrt_d_head;
                            exp_scores[j] = std::exp(score - max_val);
                            sum_exp += exp_scores[j];
                        }
                    }

                    // Step 3: Normalize to get probabilities
                    sum_exp = std::max(sum_exp, eps); // Avoid division by zero

                    for (int j = 0; j < seq_len; ++j)
                    {
                        exp_scores[j] /= sum_exp;
                    }

                    // Step 4: Requantize to INT8
                    // Find max value in row (for dynamic per-row scaling)
                    float max_abs = 0.0f;
                    for (int j = 0; j < seq_len; ++j)
                    {
                        max_abs = std::max(max_abs, std::abs(exp_scores[j]));
                    }

                    float scale = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;
                    attn_weights_row_scales[b * n_heads_ * seq_len + h * seq_len + i] = scale;

                    float inv_scale = 1.0f / scale;
                    for (int j = 0; j < seq_len; ++j)
                    {
                        float scaled = exp_scores[j] * inv_scale;
                        int quantized = static_cast<int>(std::round(scaled));
                        attn_weights_int8[row_base + j] = static_cast<int8_t>(
                            std::clamp(quantized, -127, 127));
                    }
                }
            }
        }
    }

    void INT8AttentionKernel::compute_context(
        const int8_t *attn_weights_int8,
        const float *attn_weights_row_scales,
        const int8_t *v_int8,
        const float *v_row_scales,
        float *context_fp32,
        int batch,
        int seq_len)
    {
        // For each batch and head, compute: attn_weights[seq_len, seq_len] @ V[seq_len, d_head]
        // Result: context[seq_len, d_head] per head (in FP32)

        int d_model = n_heads_ * d_head_;

        for (int b = 0; b < batch; ++b)
        {
            for (int h = 0; h < n_heads_; ++h)
            {
                for (int i = 0; i < seq_len; ++i)
                {
                    // Get attention weights row scale for this query position
                    float attn_scale = attn_weights_row_scales[b * n_heads_ * seq_len + h * seq_len + i];

                    for (int d = 0; d < d_head_; ++d)
                    {
                        // Compute weighted sum: sum_j(attn_weights[i,j] * V[j,d])
                        // Apply proper per-token V scaling
                        float sum_fp32 = 0.0f;

                        for (int j = 0; j < seq_len; ++j)
                        {
                            // Attn weights index: [b, h, i, j]
                            int attn_idx = ((b * n_heads_ + h) * seq_len + i) * seq_len + j;
                            // V index: [b, j, h * d_head + d]
                            int v_idx = (b * seq_len + j) * d_model + h * d_head_ + d;

                            // Use per-token V scale (critical for correctness!)
                            float v_scale = v_row_scales[b * seq_len + j];
                            float combined_scale = attn_scale * v_scale;

                            // Accumulate in FP32 with proper scaling
                            sum_fp32 += static_cast<float>(attn_weights_int8[attn_idx]) *
                                        static_cast<float>(v_int8[v_idx]) *
                                        combined_scale;
                        }

                        // Store as FP32 (properly scaled)
                        int context_idx = ((b * n_heads_ + h) * seq_len + i) * d_head_ + d;
                        context_fp32[context_idx] = sum_fp32;
                    }
                }
            }
        }
    }

    void INT8AttentionKernel::requantize_output(
        const float *context_fp32,
        int8_t *output_int8,
        float *output_row_scales,
        int batch,
        int seq_len)
    {
        // Requantize context from [batch, n_heads, seq_len, d_head] FP32
        // to [batch, seq_len, n_heads * d_head] INT8 (interleaved heads)

        int d_model = n_heads_ * d_head_;

        for (int b = 0; b < batch; ++b)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                // Find max abs across all heads for this position
                float max_abs = 0.0f;

                for (int h = 0; h < n_heads_; ++h)
                {
                    for (int d = 0; d < d_head_; ++d)
                    {
                        int context_idx = ((b * n_heads_ + h) * seq_len + i) * d_head_ + d;
                        max_abs = std::max(max_abs, std::abs(context_fp32[context_idx]));
                    }
                }

                // Compute scale for this row
                float scale = (max_abs > 1e-6f) ? (max_abs / 127.0f) : 1.0f;
                output_row_scales[b * seq_len + i] = scale;

                float inv_scale = 1.0f / scale;

                // Quantize and interleave heads
                for (int h = 0; h < n_heads_; ++h)
                {
                    for (int d = 0; d < d_head_; ++d)
                    {
                        int context_idx = ((b * n_heads_ + h) * seq_len + i) * d_head_ + d;
                        int output_idx = (b * seq_len + i) * d_model + h * d_head_ + d;

                        float scaled = context_fp32[context_idx] * inv_scale;
                        int quantized = static_cast<int>(std::round(scaled));
                        output_int8[output_idx] = static_cast<int8_t>(
                            std::clamp(quantized, -127, 127));
                    }
                }
            }
        }
    }

} // namespace llaminar2
