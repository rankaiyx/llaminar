/**
 * @file RoPEOp.h
 * @brief Typed Rotary Position Embedding operation
 * @author David Sanftenberg
 *
 * RoPE applies rotary position embeddings to Q and K tensors.
 * This is applied after Q/K projections, before attention computation.
 *
 * Provides precision-aware implementations for FP32, BF16, FP16, and Q8_1
 * activation precisions.
 */

#pragma once

#include "Op.h"
#include "../PipelineConfig.h"
#include "../../tensors/Tensors.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "../../utils/MPIContext.h"

#include <memory>
#include <cstring>

namespace llaminar2
{

    // ============================================================================
    // IRoPEOp Interface
    // ============================================================================

    /**
     * @brief Abstract interface for RoPE operations
     *
     * RoPE applies rotary embeddings based on position IDs:
     * - Q: [seq_len, n_heads * head_dim]
     * - K: [seq_len, n_kv_heads * head_dim]
     *
     * Position IDs of -1 signal padding tokens (RoPE is skipped for those positions).
     */
    class IRoPEOp
    {
    public:
        virtual ~IRoPEOp() = default;

        /**
         * @brief Apply RoPE to Q and K tensors
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim] - modified in-place
         * @param K Key tensor [seq_len, n_kv_heads * head_dim] - modified in-place
         * @param position_ids Position indices [seq_len] (-1 = padding, skip RoPE)
         * @param seq_len Sequence length (number of tokens)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (for GQA)
         * @param head_dim Dimension per head
         * @param rope_theta RoPE frequency base (10000.0 for LLaMA, 1000000.0 for Qwen2.5)
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on validation or execution failure
         */
        virtual bool apply(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx) = 0;

        /**
         * @brief Get the activation precision this op handles
         */
        virtual ActivationPrecision precision() const = 0;
    };

    // ============================================================================
    // RoPEOpTyped<P> Template Implementation
    // ============================================================================

    namespace detail
    {

        /**
         * @brief Map ActivationPrecision to tensor type for RoPE ops
         */
        template <ActivationPrecision P>
        struct RoPETensor;

        template <>
        struct RoPETensor<ActivationPrecision::FP32>
        {
            using type = FP32Tensor;
        };

        template <>
        struct RoPETensor<ActivationPrecision::BF16>
        {
            using type = BF16Tensor;
        };

        template <>
        struct RoPETensor<ActivationPrecision::FP16>
        {
            using type = FP16Tensor;
        };

        template <>
        struct RoPETensor<ActivationPrecision::Q8_1>
        {
            using type = Q8_1Tensor;
        };

    } // namespace detail

    /**
     * @brief Typed RoPE operation implementation
     *
     * Template specializations delegate to the appropriate activation tensor
     * interface for each precision:
     * - FP32/BF16/FP16: Use IActivationTensor::applyRoPE()
     * - Q8_1: Currently converts to FP32, applies RoPE, converts back
     *
     * @note Q8_1 RoPE is computationally expensive due to conversions.
     *       Future optimization: implement native Q8_1 RoPE kernel.
     */
    template <ActivationPrecision P>
    class RoPEOpTyped : public IRoPEOp, public OpBase
    {
    public:
        using TensorType = typename detail::RoPETensor<P>::type;

        RoPEOpTyped() = default;

        const char *name() const override { return "RoPEOpTyped"; }
        ActivationPrecision precision() const override { return P; }

        bool apply(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx) override
        {
            // Validate inputs
            if (!Q || !K)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: null Q or K tensor");
                return false;
            }
            if (!position_ids)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: null position_ids");
                return false;
            }
            if (seq_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: invalid dimensions (seq_len="
                                         << seq_len << ", n_heads=" << n_heads << ", n_kv_heads=" << n_kv_heads
                                         << ", head_dim=" << head_dim << ")");
                return false;
            }

            return applyImpl(Q, K, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, mpi_ctx, device_idx);
        }

    private:
        bool applyImpl(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx)
        {
            if constexpr (P == ActivationPrecision::FP32)
            {
                // FP32: Use standard IActivationTensor interface
                auto *activation_q = dynamic_cast<IActivationTensor *>(Q);
                if (!activation_q)
                {
                    LOG_ERROR("RoPEOpTyped<FP32>: Q tensor must implement IActivationTensor");
                    return false;
                }

                return activation_q->applyRoPE(
                    K->mutable_data(),
                    position_ids,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_theta,
                    false, // use_bf16
                    mpi_ctx,
                    device_idx);
            }
            else if constexpr (P == ActivationPrecision::BF16)
            {
                // BF16: Use IActivationTensor with use_bf16=true
                auto *activation_q = dynamic_cast<IActivationTensor *>(Q);
                if (!activation_q)
                {
                    LOG_ERROR("RoPEOpTyped<BF16>: Q tensor must implement IActivationTensor");
                    return false;
                }

                return activation_q->applyRoPE(
                    K->mutable_data(),
                    position_ids,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_theta,
                    true, // use_bf16
                    mpi_ctx,
                    device_idx);
            }
            else if constexpr (P == ActivationPrecision::FP16)
            {
                // FP16: Convert to FP32, apply RoPE, convert back
                // This is a fallback; ideally we'd have native FP16 RoPE
                auto *q_typed = dynamic_cast<TensorType *>(Q);
                auto *k_typed = dynamic_cast<TensorType *>(K);
                if (!q_typed || !k_typed)
                {
                    LOG_ERROR("RoPEOpTyped<FP16>: tensor type mismatch");
                    return false;
                }

                // Convert Q to FP32
                const size_t q_numel = static_cast<size_t>(seq_len) * n_heads * head_dim;
                const size_t k_numel = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;

                std::vector<float> q_fp32(q_numel);
                std::vector<float> k_fp32(k_numel);

                // Use native FP16 accessors (not dequantized data())
                const uint16_t *q_data = q_typed->fp16_data();
                const uint16_t *k_data = k_typed->fp16_data();

                for (size_t i = 0; i < q_numel; ++i)
                {
                    q_fp32[i] = fp16_to_fp32(q_data[i]);
                }
                for (size_t i = 0; i < k_numel; ++i)
                {
                    k_fp32[i] = fp16_to_fp32(k_data[i]);
                }

                // Create temporary FP32 tensor and apply RoPE
                FP32Tensor q_temp({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)}, -1);
                std::memcpy(q_temp.mutable_data(), q_fp32.data(), q_numel * sizeof(float));

                auto *activation_q = dynamic_cast<IActivationTensor *>(&q_temp);
                if (!activation_q)
                {
                    LOG_ERROR("RoPEOpTyped<FP16>: failed to get IActivationTensor from temp tensor");
                    return false;
                }

                if (!activation_q->applyRoPE(
                        k_fp32.data(),
                        position_ids,
                        seq_len,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        rope_theta,
                        false, // use_bf16
                        mpi_ctx,
                        device_idx))
                {
                    return false;
                }

                // Convert back to FP16
                // Use native FP16 accessors (not dequantized mutable_data())
                uint16_t *q_out = q_typed->mutable_fp16_data();
                uint16_t *k_out = k_typed->mutable_fp16_data();
                const float *q_result = q_temp.data();

                for (size_t i = 0; i < q_numel; ++i)
                {
                    q_out[i] = fp32_to_fp16(q_result[i]);
                }
                for (size_t i = 0; i < k_numel; ++i)
                {
                    k_out[i] = fp32_to_fp16(k_fp32[i]);
                }

                return true;
            }
            else if constexpr (P == ActivationPrecision::Q8_1)
            {
                // Q8_1: Use IActivationTensor interface (KernelFactory-backed)
                // The kernel is obtained via createRoPE() which uses KernelFactory
                auto *activation_q = dynamic_cast<IActivationTensor *>(Q);
                auto *k_typed = dynamic_cast<Q8_1Tensor *>(K);
                if (!activation_q || !k_typed)
                {
                    LOG_ERROR("RoPEOpTyped<Q8_1>: Q must implement IActivationTensor, K must be Q8_1Tensor");
                    return false;
                }

                // Q8_1Tensor::applyRoPE expects K as float* but reinterprets as Q8_1 blocks
                // This is a design quirk - K is passed as block pointer cast to float*
                float *k_as_float = reinterpret_cast<float *>(k_typed->mutable_q8_1_blocks());

                return activation_q->applyRoPE(
                    k_as_float,
                    position_ids,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    rope_theta,
                    false, // use_bf16 not applicable for Q8_1
                    mpi_ctx,
                    device_idx);
            }

            return false;
        }

        // FP16 conversion helpers
        static float fp16_to_fp32(uint16_t fp16)
        {
            uint32_t sign = (fp16 >> 15) & 0x1;
            uint32_t exp = (fp16 >> 10) & 0x1F;
            uint32_t mant = fp16 & 0x3FF;

            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    fp32_bits = sign << 31;
                }
                else
                {
                    // Denormalized
                    exp = 1;
                    while ((mant & 0x400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x3FF;
                    fp32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                }
            }
            else if (exp == 31)
            {
                fp32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
            }
            else
            {
                fp32_bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }

            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        static uint16_t fp32_to_fp16(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(uint32_t));

            uint32_t sign = (fp32_bits >> 31) & 0x1;
            int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (fp32_bits >> 13) & 0x3FF;

            if (exp <= 0)
            {
                return static_cast<uint16_t>(sign << 15);
            }
            else if (exp >= 31)
            {
                return static_cast<uint16_t>((sign << 15) | 0x7C00);
            }

            return static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
        }
    };

    // ============================================================================
    // Factory Function
    // ============================================================================

    /**
     * @brief Create a typed RoPE op for the given activation precision
     * @param precision The activation precision to use
     * @return Unique pointer to IRoPEOp implementation
     */
    inline std::unique_ptr<IRoPEOp> createRoPEOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::FP32>>();
        case ActivationPrecision::BF16:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::Q8_1:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::Q8_1>>();
        default:
            LOG_ERROR("createRoPEOp: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

    // ============================================================================
    // Legacy RoPEOp (Backward Compatibility)
    // ============================================================================

    /**
     * @brief Legacy RoPE op wrapper for backward compatibility
     *
     * @deprecated Use IRoPEOp with createRoPEOp() factory instead
     */
    class RoPEOp : public OpBase
    {
    public:
        RoPEOp() : impl_(std::make_unique<RoPEOpTyped<ActivationPrecision::FP32>>()) {}

        const char *name() const { return "RoPEOp"; }

        bool operator()(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            const char *snapshot_prefix = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            (void)snapshot_prefix; // Snapshot capture handled by calling pipeline
            return impl_->apply(Q, K, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, mpi_ctx, device_idx);
        }

    private:
        std::unique_ptr<IRoPEOp> impl_;
    };

} // namespace llaminar2
