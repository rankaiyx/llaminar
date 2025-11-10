/**
 * @file ActivationTraits.h
 * @brief Activation storage format traits for generic GEMM kernels
 *
 * Provides compile-time abstraction over different activation tensor formats:
 * - FP32: Standard single-precision floating point
 * - BF16: Brain floating point (16-bit)
 * - FP16: IEEE half precision (16-bit)
 * - INT8: 8-bit integer (quantized activations)
 *
 * This allows GemmKernelTemplate to be generic over activation precision
 * while maintaining optimal performance through template specialization.
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "../../../tensors/FP16Utils.h"
#include "../../../tensors/SIMDHelpers.h"
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Activation precision enumeration
             *
             * Used for runtime dispatch and autotuner cache keys.
             */
            enum class ActivationPrecision
            {
                FP32, // 32-bit float
                BF16, // 16-bit bfloat16
                FP16, // 16-bit IEEE half
                INT8  // 8-bit signed integer (Q8_0 format)
            };

            /**
             * @brief Activation storage traits template (primary template)
             *
             * Specialize for each supported storage type.
             */
            template <typename T>
            struct ActivationStorageTraits;

            /**
             * @brief Traits for FP32 activation storage (baseline)
             *
             * Zero-copy path: storage_type == compute_type == float
             */
            template <>
            struct ActivationStorageTraits<float>
            {
                using storage_type = float;    // How data is stored in memory
                using compute_type = float;    // Type used for computation
                using accumulate_type = float; // Type for accumulators

                static constexpr ActivationPrecision precision = ActivationPrecision::FP32;
                static constexpr bool requires_conversion = false;

                /**
                 * @brief Load from storage format to compute format
                 */
                static inline float load(const float *src, size_t idx)
                {
                    return src[idx];
                }

                /**
                 * @brief Store from compute format to storage format
                 */
                static inline void store(float *dst, size_t idx, float value)
                {
                    dst[idx] = value;
                }

                /**
                 * @brief Pack panel from row-major source to packed layout
                 *
                 * For FP32, this is just a memcpy since no conversion needed.
                 *
                 * @param src Source matrix (row-major, lda stride)
                 * @param dst Destination packed buffer (contiguous)
                 * @param rows Number of rows to pack
                 * @param cols Number of columns to pack
                 * @param lda Leading dimension of source
                 */
                static inline void pack_panel(
                    const float *src, float *dst,
                    int rows, int cols, int lda)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        std::memcpy(dst + i * cols, src + i * lda, cols * sizeof(float));
                    }
                }
            };

            /**
             * @brief Traits for BF16 activation storage
             *
             * Requires conversion: storage (uint16_t) → compute (float)
             * Uses SIMD-optimized BF16→FP32 conversion from SIMDHelpers
             */
            template <>
            struct ActivationStorageTraits<uint16_t>
            {
                using storage_type = uint16_t; // BF16 or FP16 stored as uint16_t
                using compute_type = float;    // Always compute in FP32
                using accumulate_type = float;

                // Note: Caller must specify BF16 vs FP16 at runtime
                static constexpr ActivationPrecision precision = ActivationPrecision::BF16;
                static constexpr bool requires_conversion = true;

                /**
                 * @brief Load BF16/FP16 and convert to FP32
                 *
                 * Assumes BF16 format by default. For FP16, use fp16_to_fp32().
                 */
                static inline float load(const uint16_t *src, size_t idx)
                {
                    return llaminar2::simd::bf16_to_fp32(src[idx]);
                }

                /**
                 * @brief Store FP32 as BF16/FP16
                 */
                static inline void store(uint16_t *dst, size_t idx, float value)
                {
                    dst[idx] = llaminar2::simd::fp32_to_bf16(value);
                }

                /**
                 * @brief Pack panel with BF16→FP32 conversion
                 *
                 * Uses SIMD-optimized bulk conversion (8-16 elements at a time).
                 */
                static inline void pack_panel(
                    const uint16_t *src, float *dst,
                    int rows, int cols, int lda)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        const uint16_t *src_row = src + i * lda;
                        float *dst_row = dst + i * cols;

                        // Use SIMD bulk conversion
                        llaminar2::simd::convert_bf16_to_fp32(src_row, dst_row, cols);
                    }
                }
            }; /**
                * @brief Traits for INT8 activation storage (quantized)
                *
                * Used for integer GEMM with Q8_0 format:
                * - Storage: int8_t signed integers + FP16 scale per block
                * - Compute: int32_t accumulators (VNNI DPBUSD)
                * - Accumulate: double for final scaling
                */
            template <>
            struct ActivationStorageTraits<int8_t>
            {
                using storage_type = int8_t;     // Quantized int8 values
                using compute_type = int8_t;     // Compute in int8 domain
                using accumulate_type = int32_t; // Accumulate in int32 (VNNI output)

                static constexpr ActivationPrecision precision = ActivationPrecision::INT8;
                static constexpr bool requires_conversion = false; // No load/store conversion

                /**
                 * @brief Load int8 value (no conversion)
                 */
                static inline int8_t load(const int8_t *src, size_t idx)
                {
                    return src[idx];
                }

                /**
                 * @brief Store int8 value (no conversion)
                 */
                static inline void store(int8_t *dst, size_t idx, int8_t value)
                {
                    dst[idx] = value;
                }

                /**
                 * @brief Pack int8 panel (memcpy, no conversion needed)
                 */
                static inline void pack_panel(
                    const int8_t *src, int8_t *dst,
                    int rows, int cols, int lda)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        std::memcpy(dst + i * cols, src + i * lda, cols * sizeof(int8_t));
                    }
                }
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
