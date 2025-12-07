/**
 * @file CPUSwiGLUKernelTyped.h
 * @brief Typed SwiGLU kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This kernel follows the "kernel black-box model" pattern from CPUSoftmaxKernelTyped:
 * - External interface uses configured ActivationPrecision (FP32, BF16, FP16, Q8_1)
 * - Internal computation varies by precision
 * - Q8_1 uses integer-aware operations without full FP32 dequantization
 *
 * SwiGLU computation: output = silu(gate) * up
 * where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Usage:
 *   CPUSwiGLUKernelTyped<ActivationPrecision::FP32> kernel_fp32;
 *   kernel_fp32.apply_typed(gate_fp32, up_fp32, output_fp32, rows, cols);
 *
 *   CPUSwiGLUKernelTyped<ActivationPrecision::Q8_1> kernel_q8;
 *   kernel_q8.apply_typed(gate_q8, up_q8, output_q8, n_blocks);
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
    // Precision Metadata Traits (same pattern as CPUSoftmaxKernelTyped)
    // =========================================================================

    namespace detail
    {
        template <ActivationPrecision P>
        struct SwiGLUPrecisionMetadata;

        template <>
        struct SwiGLUPrecisionMetadata<ActivationPrecision::FP32>
        {
            using StorageType = float;
            static constexpr const char *name = "FP32";
            static constexpr float compression_ratio = 1.0f;
        };

        template <>
        struct SwiGLUPrecisionMetadata<ActivationPrecision::BF16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "BF16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct SwiGLUPrecisionMetadata<ActivationPrecision::FP16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "FP16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct SwiGLUPrecisionMetadata<ActivationPrecision::Q8_1>
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
     * @brief Typed SwiGLU kernel template
     *
     * Primary template provides type traits and common metadata.
     * Specializations implement precision-specific apply_typed().
     */
    template <ActivationPrecision Precision>
    class CPUSwiGLUKernelTyped : public CPUKernelBase
    {
    public:
        using Metadata = detail::SwiGLUPrecisionMetadata<Precision>;
        using StorageType = typename Metadata::StorageType;

        virtual ~CPUSwiGLUKernelTyped() = default;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only
        }

        static constexpr ActivationPrecision precision() { return Precision; }
        static const char *precision_name() { return Metadata::name; }
        static constexpr float compression_ratio() { return Metadata::compression_ratio; }

        /**
         * @brief Apply SwiGLU with typed storage
         *
         * @param gate Input gate tensor data
         * @param up Input up tensor data
         * @param output Output tensor data
         * @param size Number of elements (for Q8_1: total elements, not blocks)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            const StorageType *gate,
            const StorageType *up,
            StorageType *output,
            int size,
            int device_idx = -1);
    };

    // =========================================================================
    // FP32 Specialization
    // =========================================================================

    template <>
    class CPUSwiGLUKernelTyped<ActivationPrecision::FP32> : public CPUKernelBase
    {
    public:
        using StorageType = float;

        CPUSwiGLUKernelTyped() = default;
        ~CPUSwiGLUKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
        static const char *precision_name() { return "FP32"; }
        static constexpr float compression_ratio() { return 1.0f; }

        /**
         * @brief Apply SwiGLU to FP32 data
         *
         * @param gate Input gate tensor [size]
         * @param up Input up tensor [size]
         * @param output Output tensor [size]
         * @param size Number of elements
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            const float *gate,
            const float *up,
            float *output,
            int size,
            int device_idx = -1);
    };

    // =========================================================================
    // BF16 Specialization
    // =========================================================================

    template <>
    class CPUSwiGLUKernelTyped<ActivationPrecision::BF16> : public CPUKernelBase
    {
    public:
        using StorageType = uint16_t;

        CPUSwiGLUKernelTyped() = default;
        ~CPUSwiGLUKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
        static const char *precision_name() { return "BF16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        /**
         * @brief Apply SwiGLU to BF16 data
         *
         * @param gate Input gate tensor [size] (BF16 stored as uint16_t)
         * @param up Input up tensor [size] (BF16 stored as uint16_t)
         * @param output Output tensor [size] (BF16 stored as uint16_t)
         * @param size Number of elements
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx = -1);
    };

    // =========================================================================
    // FP16 Specialization
    // =========================================================================

    template <>
    class CPUSwiGLUKernelTyped<ActivationPrecision::FP16> : public CPUKernelBase
    {
    public:
        using StorageType = uint16_t;

        CPUSwiGLUKernelTyped() = default;
        ~CPUSwiGLUKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
        static const char *precision_name() { return "FP16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        /**
         * @brief Apply SwiGLU to FP16 data
         *
         * @param gate Input gate tensor [size] (FP16 stored as uint16_t)
         * @param up Input up tensor [size] (FP16 stored as uint16_t)
         * @param output Output tensor [size] (FP16 stored as uint16_t)
         * @param size Number of elements
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx = -1);
    };

    // =========================================================================
    // Q8_1 Specialization (Integer-Aware)
    // =========================================================================

    /**
     * @brief Q8_1 specialization - integer-aware SwiGLU
     *
     * This specialization uses the Q8_1 SwiGLU primitives:
     * - Dequantize gate and up blocks
     * - Compute silu(gate) * up
     * - Requantize result to output Q8_1 blocks
     *
     * @note size parameter is total element count (num_blocks * 32)
     */
    template <>
    class CPUSwiGLUKernelTyped<ActivationPrecision::Q8_1> : public CPUKernelBase
    {
    public:
        using StorageType = Q8_1Block;

        CPUSwiGLUKernelTyped() = default;
        ~CPUSwiGLUKernelTyped() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::Q8_1; }
        static const char *precision_name() { return "Q8_1"; }
        static constexpr float compression_ratio() { return 4.0f; }

        /**
         * @brief Apply SwiGLU to Q8_1 data
         *
         * @param gate Input gate blocks
         * @param up Input up blocks
         * @param output Output blocks
         * @param size Total element count (n_blocks * 32)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            const Q8_1Block *gate,
            const Q8_1Block *up,
            Q8_1Block *output,
            int size,
            int device_idx = -1);
    };

} // namespace llaminar2
