/**
 * @file CPUAttentionKernelTyped.h
 * @brief Typed CPU attention kernel supporting activation precisions FP32, BF16, FP16, Q8_1
 *
 * Design Philosophy:
 * - Maintains native precision throughout the attention computation
 * - FP32/BF16/FP16: All operations (Q@K^T, softmax, scores@V) in native precision
 * - Q8_1: Dequantizes to FP32 (Q8_1 softmax requires FP32 intermediate per ActivationTraits)
 *
 * Benefits:
 * - Single implementation supports FP32, BF16, FP16, Q8_1 (zero code duplication)
 * - Compile-time dispatch via ActivationTraits (zero runtime overhead)
 * - Type-safe (compile errors for unsupported types)
 * - True typed residuals - maintains precision specified by template parameter
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../CPUKernelBase.h"
#include "../primitives/ActivationTraits.h"
#include "../primitives/SoftmaxPrimitivesImpl.h"
#include "../../../pipelines/AttentionUtils.h"
#include "../../../pipelines/PipelineConfig.h"
#include "../../../utils/Logger.h"
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

namespace llaminar2
{
    namespace detail
    {
        /**
         * @brief Map ActivationPrecision enum to tensor type
         *
         * Only supports FP32, BF16, FP16, Q8_1 - no INT32/Q8_0 support needed.
         */
        template <ActivationPrecision P>
        struct PrecisionToTensor;

        template <>
        struct PrecisionToTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };
        template <>
        struct PrecisionToTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };
        template <>
        struct PrecisionToTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };
        template <>
        struct PrecisionToTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
        };
    } // namespace detail

    /**
     * @brief Template-based CPU attention kernel with native precision support
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     *
     * Uses ActivationTraits<TensorT> for precision-specific operations:
     * - apply_softmax(): Native precision softmax (FP32/BF16/FP16) or FP32 for Q8_1
     * - create_activation_gemm(): Precision-specific GEMM kernel
     * - allocate_workspace(): Precision-specific workspace allocation
     *
     * Key Design:
     * - FP32/BF16/FP16: Entire computation in native precision
     * - Q8_1: Dequantizes Q/K/V to FP32, computes in FP32 (per ActivationTraits design)
     */
    template <ActivationPrecision Precision>
    class CPUAttentionKernelTyped : public ITensorAttention, public CPUKernelBase
    {
    public:
        using TensorT = typename detail::PrecisionToTensor<Precision>::Type;
        using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;
        using Traits = primitives::ActivationTraits<TensorT>;

        // For Q8_1, we compute in FP32 (dequantized)
        // For FP32/BF16/FP16, we compute in native precision
        static constexpr bool kRequiresDequant = std::is_same_v<TensorT, Q8_1Tensor>;
        using ComputeType = std::conditional_t<kRequiresDequant, float, ElementType>;

        CPUAttentionKernelTyped() = default;
        ~CPUAttentionKernelTyped() override = default;

        /**
         * @brief Check if kernel supports specific device
         */
        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        /**
         * @brief Compute attention (ITensorAttention interface implementation)
         *
         * Interface uses float* for compatibility. Internally casts to ElementType*.
         */
        bool compute(
            const float *Q, const float *K, const float *V,
            float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            return compute_impl(
                Q, K, V, output,
                seq_len, n_heads, n_kv_heads, head_dim,
                causal, window_size,
                workspace_scores, workspace_mask,
                device_idx, seq_len);
        }

        /**
         * @brief Batch attention computation
         */
        bool compute_batch(
            const float *Q, const float *K, const float *V,
            float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // Allocate per-batch scores workspace
            std::shared_ptr<TensorBase> scores_alloc;
            ComputeType *scores_ptr = nullptr;

            const size_t scores_size = batch_size * n_heads * seq_len * seq_len;
            if (!workspace_scores)
            {
                if constexpr (kRequiresDequant)
                {
                    scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{scores_size});
                }
                else
                {
                    scores_alloc = Traits::allocate_workspace(std::vector<size_t>{scores_size});
                }
                scores_ptr = reinterpret_cast<ComputeType *>(scores_alloc->mutable_data());
            }
            else
            {
                scores_ptr = reinterpret_cast<ComputeType *>(workspace_scores->mutable_data());
            }

            // Get mask if provided
            const float *mask_ptr = workspace_mask ? workspace_mask->data() : nullptr;
            const int total_tokens = batch_size * seq_len;

            // Mask tile buffer for extracting per-batch blocks
            std::vector<float> mask_tile;
            if (mask_ptr)
            {
                mask_tile.resize(seq_len * seq_len);
            }

            // Cast inputs to ElementType
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Loop over batch items
            for (int b = 0; b < batch_size; ++b)
            {
                // Offsets in element units
                size_t q_offset = b * seq_len * n_heads * head_dim;
                size_t k_offset = b * seq_len * n_kv_heads * head_dim;
                size_t v_offset = b * seq_len * n_kv_heads * head_dim;
                size_t out_offset = b * seq_len * n_heads * head_dim;
                size_t scores_offset = b * n_heads * seq_len * seq_len;

                // Extract per-sequence mask tile
                const float *mask_slice = nullptr;
                if (mask_ptr)
                {
                    const int row_offset = b * seq_len;
                    const int col_offset = b * seq_len;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        for (int j = 0; j < seq_len; ++j)
                        {
                            mask_tile[i * seq_len + j] = mask_ptr[(row_offset + i) * total_tokens + (col_offset + j)];
                        }
                    }
                    mask_slice = mask_tile.data();
                }

                bool success = compute_attention_native(
                    Q_typed + q_offset,
                    K_typed + k_offset,
                    V_typed + v_offset,
                    reinterpret_cast<ElementType *>(output) + out_offset,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    causal, scores_ptr + scores_offset,
                    mask_slice, seq_len);

                if (!success)
                    return false;
            }

            return true;
        }

    private:
        /**
         * @brief Implementation of compute() with workspace handling
         */
        bool compute_impl(
            const float *Q, const float *K, const float *V,
            float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size,
            TensorBase *workspace_scores,
            TensorBase *workspace_mask,
            int device_idx, int kv_len)
        {
            // Validate device
            if (device_idx != -1)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: device_idx must be -1 (CPU), got " << device_idx);
                return false;
            }

            // Allocate scores workspace if not provided
            std::shared_ptr<TensorBase> scores_alloc;
            ComputeType *scores_ptr = nullptr;

            const size_t scores_size = n_heads * seq_len * kv_len;
            if (!workspace_scores)
            {
                if constexpr (kRequiresDequant)
                {
                    scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{scores_size});
                }
                else
                {
                    scores_alloc = Traits::allocate_workspace(std::vector<size_t>{scores_size});
                }
                scores_ptr = reinterpret_cast<ComputeType *>(scores_alloc->mutable_data());
            }
            else
            {
                scores_ptr = reinterpret_cast<ComputeType *>(workspace_scores->mutable_data());
            }

            const float *mask_ptr = workspace_mask ? workspace_mask->data() : nullptr;

            // Cast inputs to ElementType
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            return compute_attention_native(
                Q_typed, K_typed, V_typed,
                reinterpret_cast<ElementType *>(output),
                seq_len, n_heads, n_kv_heads, head_dim,
                causal, scores_ptr, mask_ptr, kv_len);
        }

        /**
         * @brief Core attention computation in native precision
         *
         * For FP32/BF16/FP16: All operations in native ElementType
         * For Q8_1: Dequantizes to FP32, computes in FP32
         */
        bool compute_attention_native(
            const ElementType *Q, const ElementType *K, const ElementType *V,
            ElementType *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            ComputeType *scores,
            const float *mask,
            int kv_len) const
        {
            // Validate inputs
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: null tensor data");
                return false;
            }

            if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0 || kv_len <= 0)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: invalid dimensions");
                return false;
            }

            if (n_heads % n_kv_heads != 0)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: n_heads must be divisible by n_kv_heads");
                return false;
            }

            // Dispatch to precision-specific implementation
            if constexpr (kRequiresDequant)
            {
                return compute_q8_1_via_fp32(Q, K, V, output, seq_len, n_heads, n_kv_heads, head_dim, causal, scores, mask, kv_len);
            }
            else
            {
                return compute_native_precision(Q, K, V, output, seq_len, n_heads, n_kv_heads, head_dim, causal, scores, mask, kv_len);
            }
        }

        /**
         * @brief Native precision attention (FP32, BF16, FP16)
         *
         * Entire computation stays in ElementType precision.
         */
        bool compute_native_precision(
            const ElementType *Q, const ElementType *K, const ElementType *V,
            ElementType *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            ElementType *scores,
            const float *mask,
            int kv_len) const
        {
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int heads_per_kv = n_heads / n_kv_heads;

            // Create GEMM kernel for this precision
            auto gemm = Traits::create_activation_gemm();

            // Determine parallelization strategy
            const int max_threads = omp_get_max_threads();
            bool parallelize_heads = true;
            if (seq_len >= 128 && n_heads * 2 < max_threads)
            {
                parallelize_heads = false;
            }

            // Get activation format for GEMM calls
            ActivationFormat format = ActivationFormat::FP32;
            if constexpr (std::is_same_v<TensorT, BF16Tensor>)
                format = ActivationFormat::BF16;
            else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
                format = ActivationFormat::FP16;

            // Zero output
            std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(ElementType));

#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                // --- Step 1: Q @ K^T -> Scores ---
                ElementType *scores_h = scores + h * seq_len * kv_len;
                const ElementType *Q_h = Q + h * head_dim;
                int kv_h = h / heads_per_kv;
                const ElementType *K_h = K + kv_h * head_dim;

                const int lda_k = n_heads * head_dim;
                const int ldb_k = n_kv_heads * head_dim;
                const int ldc_k = kv_len;

                // Optimized decode path (seq_len == 1) for FP32
                if (seq_len == 1 && std::is_same_v<TensorT, FP32Tensor>)
                {
                    compute_decode_fp32_head(
                        reinterpret_cast<const float *>(Q_h),
                        reinterpret_cast<const float *>(K_h),
                        reinterpret_cast<const float *>(V + kv_h * head_dim),
                        reinterpret_cast<float *>(output + h * head_dim),
                        reinterpret_cast<float *>(scores_h),
                        kv_len, head_dim, n_kv_heads, n_heads,
                        scale, mask);
                    continue;
                }

                // General path: GEMM with fused softmax
                gemm->template multiply_with_softmax_strided_typed<ElementType, ElementType>(
                    Q_h, K_h, scores_h,
                    seq_len, kv_len, head_dim,
                    lda_k, ldb_k, ldc_k,
                    scale,
                    true, // transpose_B
                    1,    // softmax_axis
                    mask,
                    causal,
                    nullptr, -1, format);

                // --- Step 2: Scores @ V -> Output ---
                const ElementType *V_h = V + kv_h * head_dim;
                ElementType *output_h = output + h * head_dim;

                const int lda_v = kv_len;
                const int ldb_v = n_kv_heads * head_dim;
                const int ldc_v = n_heads * head_dim;

                // For native precision, scores are ElementType*
                // Need to handle the fact that multiply_activations_strided expects float*
                // This is where we need precision-aware GEMM
                if constexpr (std::is_same_v<ElementType, float>)
                {
                    gemm->multiply_activations_strided(
                        scores_h, V_h, output_h,
                        seq_len, head_dim, kv_len,
                        lda_v, ldb_v, ldc_v,
                        false, 1.0f, 0.0f, nullptr, -1);
                }
                else
                {
                    // BF16/FP16: Use typed GEMM
                    gemm->template multiply_activations_strided_typed<ElementType, ElementType>(
                        scores_h, V_h, output_h,
                        seq_len, head_dim, kv_len,
                        lda_v, ldb_v, ldc_v,
                        false, 1.0f, 0.0f, nullptr, -1, format);
                }
            }

            return true;
        }

        /**
         * @brief Q8_1 attention via FP32 dequantization
         *
         * Q8_1 requires dequantization for softmax (per ActivationTraits design).
         */
        bool compute_q8_1_via_fp32(
            const ElementType *Q, const ElementType *K, const ElementType *V,
            ElementType *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            float *scores,
            const float *mask,
            int kv_len) const
        {
            // Dequantize Q, K, V to FP32
            const size_t Q_size = seq_len * n_heads * head_dim;
            const size_t K_size = kv_len * n_kv_heads * head_dim;
            const size_t V_size = kv_len * n_kv_heads * head_dim;

            std::vector<float> Q_fp32(Q_size);
            std::vector<float> K_fp32(K_size);
            std::vector<float> V_fp32(V_size);

            // Q8_1: decode blocks
            const size_t Q_blocks = Q_size / 32;
            const size_t K_blocks = K_size / 32;
            const size_t V_blocks = V_size / 32;

            for (size_t i = 0; i < Q_blocks; ++i)
            {
                Q8_1Tensor::decodeBlock(Q[i], Q_fp32.data() + i * 32);
            }
            for (size_t i = 0; i < K_blocks; ++i)
            {
                Q8_1Tensor::decodeBlock(K[i], K_fp32.data() + i * 32);
            }
            for (size_t i = 0; i < V_blocks; ++i)
            {
                Q8_1Tensor::decodeBlock(V[i], V_fp32.data() + i * 32);
            }

            // Compute attention in FP32
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int heads_per_kv = n_heads / n_kv_heads;

            auto gemm = primitives::ActivationTraits<FP32Tensor>::create_activation_gemm();

            const int max_threads = omp_get_max_threads();
            bool parallelize_heads = (seq_len < 128) || (n_heads * 2 >= max_threads);

            // Allocate FP32 output buffer
            std::vector<float> output_fp32(seq_len * n_heads * head_dim, 0.0f);

#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * kv_len;
                const float *Q_h = Q_fp32.data() + h * head_dim;
                int kv_h = h / heads_per_kv;
                const float *K_h = K_fp32.data() + kv_h * head_dim;

                const int lda = n_heads * head_dim;
                const int ldb = n_kv_heads * head_dim;
                const int ldc = kv_len;

                // Q @ K^T
                gemm->multiply_activations_strided(
                    Q_h, K_h, scores_h,
                    seq_len, kv_len, head_dim,
                    lda, ldb, ldc,
                    true, scale, 0.0f, nullptr, -1);
            }

            // Apply mask
            if (mask)
            {
#pragma omp parallel for if (n_heads > 1)
                for (int h = 0; h < n_heads; ++h)
                {
                    float *scores_h = scores + h * seq_len * kv_len;
                    attention_utils::apply_attention_mask(scores_h, mask, seq_len, kv_len);
                }
            }

            // Softmax in FP32
#pragma omp parallel for if (n_heads > 1)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * kv_len;
                primitives::softmax_row_major_fp32(scores_h, seq_len, kv_len, causal, 1.0f, false);
            }

            // Scores @ V
#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                const float *weights_h = scores + h * seq_len * kv_len;
                int kv_h = h / heads_per_kv;
                const float *V_h = V_fp32.data() + kv_h * head_dim;
                float *output_h = output_fp32.data() + h * head_dim;

                const int lda = kv_len;
                const int ldb = n_kv_heads * head_dim;
                const int ldc = n_heads * head_dim;

                gemm->multiply_activations_strided(
                    weights_h, V_h, output_h,
                    seq_len, head_dim, kv_len,
                    lda, ldb, ldc,
                    false, 1.0f, 0.0f, nullptr, -1);
            }

            // Re-quantize output to Q8_1
            const size_t out_blocks = (seq_len * n_heads * head_dim) / 32;
            for (size_t i = 0; i < out_blocks; ++i)
            {
                Q8_1Tensor::encodeBlock(output_fp32.data() + i * 32, output[i]);
            }

            return true;
        }

        /**
         * @brief Optimized FP32 decode path (seq_len == 1)
         */
        void compute_decode_fp32_head(
            const float *Q_h, const float *K_h, const float *V_h,
            float *output_h, float *scores_h,
            int kv_len, int head_dim, int n_kv_heads, int n_heads,
            float scale, const float *mask) const
        {
            const int ldb_k = n_kv_heads * head_dim;
            const int ldb_v = n_kv_heads * head_dim;

            // Q @ K^T
            for (int t = 0; t < kv_len; ++t)
            {
                const float *k_ptr = K_h + t * ldb_k;
                float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += Q_h[d] * k_ptr[d];
                }
                scores_h[t] = dot * scale;
                if (mask)
                    scores_h[t] += mask[t];
            }

            // Softmax
            primitives::softmax_row_fp32(scores_h, kv_len, false, 1.0f, -1);

            // Scores @ V
            std::memset(output_h, 0, head_dim * sizeof(float));
            for (int t = 0; t < kv_len; ++t)
            {
                float s = scores_h[t];
                const float *v_ptr = V_h + t * ldb_v;
#pragma omp simd
                for (int d = 0; d < head_dim; ++d)
                {
                    output_h[d] += s * v_ptr[d];
                }
            }
        }
    };

    // Explicit instantiations (only FP32, BF16, FP16, Q8_1 - no INT32/Q8_0)
    extern template class CPUAttentionKernelTyped<ActivationPrecision::FP32>;
    extern template class CPUAttentionKernelTyped<ActivationPrecision::BF16>;
    extern template class CPUAttentionKernelTyped<ActivationPrecision::FP16>;
    extern template class CPUAttentionKernelTyped<ActivationPrecision::Q8_1>;

} // namespace llaminar2
