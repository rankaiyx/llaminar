/**
 * @file CPURoPEKernelTyped.h
 * @brief Typed RoPE kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This kernel follows the "kernel black-box model" pattern from CPURMSNormTypedKernel:
 * - External interface uses configured ActivationPrecision (FP32, BF16, FP16, Q8_1)
 * - Internal computation varies by precision
 * - Q8_1 uses pure-integer operations without FP32 round-trips
 *
 * Usage:
 *   CPURoPEKernelTyped<ActivationPrecision::FP32> kernel_fp32;
 *   kernel_fp32.apply_typed(Q_fp32, K_fp32, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
 *
 *   CPURoPEKernelTyped<ActivationPrecision::Q8_1> kernel_q8;
 *   kernel_q8.apply_typed(Q_q8, K_q8, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../pipelines/PipelineConfig.h"
#include "../../../tensors/BlockStructures.h"

#include <cstdint>

namespace llaminar2
{

    // Forward declarations
    class MPIContext;

    // =========================================================================
    // Precision Metadata Traits (same pattern as CPURMSNormTypedKernel)
    // =========================================================================

    namespace detail
    {
        template <ActivationPrecision P>
        struct RoPEPrecisionMetadata;

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::FP32>
        {
            using StorageType = float;
            static constexpr const char *name = "FP32";
            static constexpr float compression_ratio = 1.0f;
        };

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::BF16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "BF16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::FP16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "FP16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::Q8_1>
        {
            using StorageType = Q8_1Block;
            static constexpr const char *name = "Q8_1";
            static constexpr float compression_ratio = 4.0f;
        };
    } // namespace detail

    // =========================================================================
    // Primary Template (Base Declaration)
    // =========================================================================

    /**
     * @brief Typed RoPE kernel template
     *
     * Primary template provides type traits and common metadata.
     * Specializations implement precision-specific apply_typed().
     */
    template <ActivationPrecision Precision>
    class CPURoPEKernelTyped : public CPUKernelBase
    {
    public:
        using Metadata = detail::RoPEPrecisionMetadata<Precision>;
        using StorageType = typename Metadata::StorageType;

        virtual ~CPURoPEKernelTyped() = default;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only
        }

        static constexpr ActivationPrecision precision() { return Precision; }
        static const char *precision_name() { return Metadata::name; }
        static constexpr float compression_ratio() { return Metadata::compression_ratio; }

        /**
         * @brief Apply RoPE with typed storage
         *
         * @param Q Q tensor data (modified in-place)
         * @param K K tensor data (modified in-place), or nullptr
         * @param position_ids Position indices [seq_len], nullptr = sequential, -1 = padding
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE base frequency (e.g., 10000.0f)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            StorageType *Q,
            StorageType *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // FP32 Specialization
    // =========================================================================

    template <>
    class CPURoPEKernelTyped<ActivationPrecision::FP32> : public CPUKernelBase
    {
    public:
        using StorageType = float;

        CPURoPEKernelTyped() = default;
        ~CPURoPEKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
        static const char *precision_name() { return "FP32"; }
        static constexpr float compression_ratio() { return 1.0f; }

        bool apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // BF16 Specialization
    // =========================================================================

    template <>
    class CPURoPEKernelTyped<ActivationPrecision::BF16> : public CPUKernelBase
    {
    public:
        using StorageType = uint16_t;

        CPURoPEKernelTyped() = default;
        ~CPURoPEKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
        static const char *precision_name() { return "BF16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        bool apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // FP16 Specialization
    // =========================================================================

    template <>
    class CPURoPEKernelTyped<ActivationPrecision::FP16> : public CPUKernelBase
    {
    public:
        using StorageType = uint16_t;

        CPURoPEKernelTyped() = default;
        ~CPURoPEKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
        static const char *precision_name() { return "FP16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        bool apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1);
    };

    // =========================================================================
    // Q8_1 Specialization (Pure-Integer)
    // =========================================================================

    /**
     * @brief Q8_1 specialization - pure-integer RoPE without FP32 round-trips
     *
     * This specialization uses:
     * - Sin/cos quantized to Q15 fixed-point format
     * - Integer rotation arithmetic on Q8_1 blocks
     * - No intermediate FP32 dequantization
     *
     * Constraint: head_dim must be divisible by 32 (Q8_1 block size)
     */
    template <>
    class CPURoPEKernelTyped<ActivationPrecision::Q8_1> : public CPUKernelBase
    {
    public:
        using StorageType = Q8_1Block;

        CPURoPEKernelTyped() = default;
        ~CPURoPEKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::Q8_1; }
        static const char *precision_name() { return "Q8_1"; }
        static constexpr float compression_ratio() { return 4.0f; }

        bool apply_typed(
            Q8_1Block *Q,
            Q8_1Block *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1);
    };

} // namespace llaminar2
