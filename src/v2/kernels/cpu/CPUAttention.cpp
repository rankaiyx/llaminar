/**
 * @file CPUAttention.cpp
 * @brief CPU implementation of attention kernel
 *
 * @author David Sanftenberg
 */

#include "CPUAttention.h"
#include "../../pipelines/AttentionUtils.h"
#include "../../utils/Logger.h"
#include "../../kernels/cpu/primitives/SoftmaxPrimitives.h"
#include "../../tensors/Tensors.h"
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

namespace llaminar2
{

    bool CPUAttention::compute(
        const float *Q, const float *K, const float *V,
        float *output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal,
        int window_size,
        TensorBase *workspace_scores,
        TensorBase *workspace_buffer,
        TensorBase *workspace_context,
        TensorBase *workspace_mask,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        // Validate device
        if (device_idx != -1)
        {
            LOG_ERROR("[CPUAttention] compute: device_idx must be -1 (CPU), got " << device_idx);
            return false;
        }

        // Validate inputs
        if (!Q || !K || !V || !output)
        {
            LOG_ERROR("[CPUAttention] compute: null tensor data");
            return false;
        }

        if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0)
        {
            LOG_ERROR("[CPUAttention] compute: invalid dimensions");
            return false;
        }

        if (n_heads % n_kv_heads != 0)
        {
            LOG_ERROR("[CPUAttention] compute: n_heads (" << n_heads
                                                          << ") must be divisible by n_kv_heads (" << n_kv_heads << ")");
            return false;
        }

        // Allocate workspaces if not provided
        std::shared_ptr<TensorBase> owned_scores;
        std::shared_ptr<TensorBase> owned_buffer;
        std::shared_ptr<TensorBase> owned_context;

        if (!workspace_scores)
        {
            owned_scores = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(n_heads * seq_len), static_cast<size_t>(seq_len)});
            workspace_scores = owned_scores.get();
        }

        // Get actual thread count for proper workspace sizing
        const int max_threads = omp_get_max_threads();
        const int actual_threads = (n_heads > 1) ? std::min(max_threads, n_heads) : 1;

        if (!workspace_buffer)
        {
            // Per-thread buffer: [actual_threads * seq_len * head_dim * 2] (Q_h + K_h/V_h)
            owned_buffer = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(actual_threads * seq_len * head_dim * 2)});
            workspace_buffer = owned_buffer.get();
        }

        if (!workspace_context)
        {
            owned_context = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(actual_threads * seq_len * head_dim)});
            workspace_context = owned_context.get();
        }

        float *scores = workspace_scores->mutable_data();

        // 1. Broadcast K/V heads if needed (GQA/MQA)
        std::vector<float> K_broadcast, V_broadcast;
        const float *K_expanded = K;
        const float *V_expanded = V;

        if (n_heads != n_kv_heads)
        {
            K_broadcast.resize(seq_len * n_heads * head_dim);
            V_broadcast.resize(seq_len * n_heads * head_dim);
            broadcast_kv(K, K_broadcast.data(), seq_len, n_heads, n_kv_heads, head_dim);
            broadcast_kv(V, V_broadcast.data(), seq_len, n_heads, n_kv_heads, head_dim);
            K_expanded = K_broadcast.data();
            V_expanded = V_broadcast.data();
        }

        // 2. Compute attention scores per head: Q @ K^T with fused scaling
        // Optimization 1: Use GEMM alpha parameter for attention scaling (1/sqrt(d_k))
        // Optimization 2: Use strided GEMM to eliminate Q/K head extraction (zero-copy)
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Create GEMM kernel once (reused across heads)
        // Use BF16 if requested for reduced memory bandwidth
        std::unique_ptr<ITensorGemm> gemm;
        if (use_bf16)
        {
            BF16Tensor bf16_dummy(std::vector<size_t>{1, 1});
            gemm = bf16_dummy.createGemm();
        }
        else
        {
            FP32Tensor fp32_dummy(std::vector<size_t>{1, 1});
            gemm = fp32_dummy.createGemm();
        }

#pragma omp parallel for if (n_heads > 1)
        for (int h = 0; h < n_heads; ++h)
        {
            float *scores_h = scores + h * seq_len * seq_len;

            // Strided Q@K^T with fused scaling
            // Q layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
            // K layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
            const float *Q_h = Q + h * head_dim;          // First element of head h
            const float *K_h = K_expanded + h * head_dim; // First element of head h

            const int lda = n_heads * head_dim; // Q: stride between rows (skip other heads)
            const int ldb = n_heads * head_dim; // K: stride between rows (skip other heads)
            const int ldc = seq_len;            // scores: contiguous [seq_len, seq_len]

            // Strided GEMM: scores = (1/sqrt(d_k)) * Q @ K^T
            gemm->multiply_activations_strided(
                Q_h, K_h, scores_h,
                seq_len, seq_len, head_dim, // m=seq_len, n=seq_len, k=head_dim
                lda, ldb, ldc,
                true,    // transpose_B=true (K^T)
                scale,   // alpha=1/sqrt(d_k) - FUSED SCALING!
                0.0f,    // beta
                nullptr, // mpi_ctx
                -1);     // device_idx (CPU)
        }

        // 3. Apply softmax with optional causal masking (scaling already done in GEMM)

#pragma omp parallel for if (n_heads > 1)
        for (int h = 0; h < n_heads; ++h)
        {
            float *scores_h = scores + h * seq_len * seq_len;

            // Apply softmax with optional causal masking (NO scaling - already fused in GEMM alpha)
            primitives::SoftmaxRowArgs softmax_args;
            softmax_args.causal = causal; // Fused causal masking in softmax
            softmax_args.scale = 1.0f;    // NO scaling (already done in GEMM alpha parameter)
            softmax_args.rows = seq_len;
            softmax_args.cols = seq_len;
            softmax_args.scores = scores_h;

            primitives::softmax_row_major_vectorized(softmax_args);
        }

        // 4. Compute context: weights @ V using strided GEMM (zero-copy multi-head)
        // Zero output once before parallel region
        std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

        // Reuse GEMM kernel from Q@K^T computation

#pragma omp parallel for if (n_heads > 1)
        for (int h = 0; h < n_heads; ++h)
        {
            const float *weights_h = scores + h * seq_len * seq_len;
            const float *V_h = V_expanded + h * head_dim; // First element of head h
            float *output_h = output + h * head_dim;      // First element of head h

            // Strided GEMM: context = weights @ V
            // V layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
            // Output layout: [seq_len, n_heads, head_dim] (head dimension interleaved)
            const int lda = seq_len;            // weights: contiguous [seq_len, seq_len]
            const int ldb = n_heads * head_dim; // V: stride between rows (skip other heads)
            const int ldc = n_heads * head_dim; // output: stride between rows

            gemm->multiply_activations_strided(
                weights_h, V_h, output_h,
                seq_len, head_dim, seq_len, // m=seq_len, n=head_dim, k=seq_len
                lda, ldb, ldc,
                false,   // transpose_B=false (weights @ V, not V^T)
                1.0f,    // alpha
                0.0f,    // beta (output already zeroed)
                nullptr, // mpi_ctx
                -1);     // device_idx (CPU)
        }

        return true;
    }

    bool CPUAttention::compute_batch(
        const float *Q, const float *K, const float *V,
        float *output,
        int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal,
        int window_size,
        TensorBase *workspace_scores,
        TensorBase *workspace_buffer,
        TensorBase *workspace_context,
        TensorBase *workspace_mask,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        LOG_ERROR("[CPUAttention] compute_batch: not yet implemented");
        return false;
    }

    void CPUAttention::broadcast_kv(
        const float *input, float *output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim) const
    {
        // Use AttentionUtils to broadcast KV heads from n_kv_heads to n_heads
        attention_utils::broadcast_kv_heads(
            input, output,
            seq_len, n_heads, n_kv_heads, head_dim);
    }

    void CPUAttention::extract_head(
        const float *multi_head, float *single_head,
        int head_idx, int seq_len, int n_heads, int head_dim) const
    {
        // Extract contiguous head data from strided multi-head layout
        for (int s = 0; s < seq_len; ++s)
        {
#pragma omp simd
            for (int d = 0; d < head_dim; ++d)
            {
                const int src_idx = s * n_heads * head_dim + head_idx * head_dim + d;
                const int dst_idx = s * head_dim + d;
                single_head[dst_idx] = multi_head[src_idx];
            }
        }
    }

    void CPUAttention::write_head(
        const float *single_head, float *multi_head,
        int head_idx, int seq_len, int n_heads, int head_dim) const
    {
        // Write contiguous single head to strided multi-head output
        for (int s = 0; s < seq_len; ++s)
        {
#pragma omp simd
            for (int d = 0; d < head_dim; ++d)
            {
                const int src_idx = s * head_dim + d;
                const int dst_idx = s * n_heads * head_dim + head_idx * head_dim + d;
                multi_head[dst_idx] = single_head[src_idx];
            }
        }
    }

    void CPUAttention::compute_scores(
        const float *Q_head, const float *K_head,
        float *scores,
        int seq_len, int head_dim) const
    {
        // Use ITensorGemm::multiply_activations for activation-activation GEMM
        // GEMM: scores = (1/sqrt(d_k)) * Q @ K^T with fused scaling
        // Q_head: [seq_len, head_dim], K_head: [seq_len, head_dim]
        // scores: [seq_len, seq_len]

        // Create a dummy FP32Tensor just to get the GEMM kernel
        // (We only need the kernel interface, not the tensor data)
        FP32Tensor dummy_tensor(std::vector<size_t>{1, 1});
        auto gemm = dummy_tensor.createGemm();

        // Compute attention scaling factor
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Execute activation-activation GEMM: scores = scale * Q @ K^T
        gemm->multiply_activations(
            Q_head, K_head, scores,
            seq_len, seq_len, head_dim,
            true,    // transpose_B (K^T)
            scale,   // alpha=1/sqrt(d_k) - FUSED SCALING!
            0.0f,    // beta
            nullptr, // mpi_ctx
            -1);     // device_idx (CPU)
    }

    void CPUAttention::scale_scores(float *scores, int seq_len, int head_dim) const
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int size = seq_len * seq_len;

#pragma omp parallel for if (size > 8192)
        for (int i = 0; i < size; ++i)
        {
            scores[i] *= scale;
        }
    }

    void CPUAttention::apply_causal_mask(float *scores, int seq_len) const
    {
        // Optimized: Apply causal mask in-place without temporary buffer
        // Set scores[i,j] = -inf for j > i (future tokens)
        for (int i = 0; i < seq_len; ++i)
        {
#pragma omp simd
            for (int j = i + 1; j < seq_len; ++j)
            {
                scores[i * seq_len + j] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    void CPUAttention::apply_softmax(const float *scores, float *weights, int seq_len) const
    {
        // Optimized: Apply softmax in-place if scores == weights
        const bool in_place = (scores == weights);

        if (!in_place)
        {
            // Copy only if different buffers
            const int size = seq_len * seq_len;
            std::memcpy(weights, scores, size * sizeof(float));
        }

        primitives::SoftmaxRowArgs softmax_args;
        softmax_args.causal = false; // Mask already applied
        softmax_args.scale = 1.0f;   // Scaling already done
        softmax_args.rows = seq_len;
        softmax_args.cols = seq_len;
        softmax_args.scores = weights;

        primitives::softmax_row_major_vectorized(softmax_args);
    }

    void CPUAttention::compute_context(
        const float *weights, const float *V_head,
        float *context,
        int seq_len, int head_dim) const
    {
        // Use ITensorGemm::multiply_activations for activation-activation GEMM
        // GEMM: context = weights @ V
        // weights: [seq_len, seq_len], V_head: [seq_len, head_dim]
        // context: [seq_len, head_dim]

        // Create a dummy FP32Tensor just to get the GEMM kernel
        FP32Tensor dummy_tensor(std::vector<size_t>{1, 1});
        auto gemm = dummy_tensor.createGemm();

        // Execute activation-activation GEMM: context = weights @ V (no transpose)
        gemm->multiply_activations(
            weights, V_head, context,
            seq_len, head_dim, seq_len,
            false,   // no transpose (V is already [seq_len, head_dim])
            1.0f,    // alpha
            0.0f,    // beta
            nullptr, // mpi_ctx
            -1);     // device_idx (CPU)
    }

} // namespace llaminar2
