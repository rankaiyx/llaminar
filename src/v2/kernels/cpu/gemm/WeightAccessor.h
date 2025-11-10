/**
 * @file WeightAccessor.h
 * @brief Dual-mode weight access abstraction for GEMM kernels
 *
 * Provides two weight access strategies:
 * 1. FP32 decode path: ITensorGemmTileDataProvider (existing, for FP32 GEMM)
 * 2. Raw quantized path: IQuantizedTileAccessor (new, for INT8 GEMM)
 *
 * This allows GemmKernelTemplate to support both:
 * - Traditional FP32×FP32 GEMM (decode weights to FP32)
 * - Integer INT8×IQ4_NL GEMM (keep weights as int8, fuse dequant into output scaling)
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            /**
             * @brief Interface for raw quantized weight block access
             *
             * Unlike ITensorGemmTileDataProvider (which decodes to FP32),
             * this interface exposes raw quantized blocks + metadata for
             * integer GEMM kernels that want to defer dequantization.
             *
             * Example: IQ4_NL blocks remain as packed nibbles + FP16 scale,
             * allowing VNNI kernels to decode on-the-fly into registers.
             */
            class IQuantizedTileAccessor
            {
            public:
                virtual ~IQuantizedTileAccessor() = default;

                /**
                 * @brief Get raw pointer to quantized block
                 *
                 * @param row_idx Row index in weight tensor
                 * @param k_block_idx Block index along K dimension
                 * @return Const pointer to raw quantized block structure
                 *
                 * For IQ4_NL: Returns IQ4_NLBlock* with {uint8_t qs[16], uint16_t d}
                 * For Q6_K: Returns Q6_KBlock* with appropriate structure
                 */
                virtual const void *get_raw_block(size_t row_idx, size_t k_block_idx) const = 0;

                /**
                 * @brief Get dequantization scale for block
                 *
                 * @param row_idx Row index
                 * @param k_block_idx Block index
                 * @return FP32 scale factor (converted from FP16 if needed)
                 */
                virtual float get_block_scale(size_t row_idx, size_t k_block_idx) const = 0;

                /**
                 * @brief Get block size in elements
                 *
                 * For IQ4_NL: 32 (32 int8 values per block after decode)
                 * For Q6_K: 256
                 */
                virtual size_t block_size() const = 0;

                /**
                 * @brief Get tensor dimensions
                 */
                virtual size_t rows() const = 0;
                virtual size_t cols() const = 0;
            };

            /**
             * @brief FP32 weight accessor (existing decode path)
             *
             * Wraps ITensorGemmTileDataProvider for backward compatibility.
             * Used by existing FP32 GEMM micro-kernels.
             */
            class FP32WeightAccessor
            {
            public:
                using weight_type = float;
                static constexpr bool is_quantized = false;

                explicit FP32WeightAccessor(const ITensorGemmTileDataProvider *provider)
                    : provider_(provider)
                {
                }

                /**
                 * @brief Decode block to FP32 (existing path)
                 */
                void decode_block(size_t row_idx, size_t k_block_idx, float *output) const
                {
                    provider_->decode_block_at(row_idx, k_block_idx, output);
                }

                size_t block_size() const { return provider_->block_size(); }
                size_t rows() const { return provider_->decoder_rows(); }
                size_t cols() const { return provider_->decoder_cols(); }

            private:
                const ITensorGemmTileDataProvider *provider_;
            };

            /**
             * @brief Quantized weight accessor (new raw block path)
             *
             * Wraps IQuantizedTileAccessor for integer GEMM kernels.
             * Exposes raw quantized blocks instead of decoded FP32.
             */
            class QuantizedWeightAccessor
            {
            public:
                using weight_type = void; // Type-erased (raw blocks)
                static constexpr bool is_quantized = true;

                explicit QuantizedWeightAccessor(const IQuantizedTileAccessor *accessor)
                    : accessor_(accessor)
                {
                }

                /**
                 * @brief Get raw quantized block (no decode)
                 */
                const void *get_raw_block(size_t row_idx, size_t k_block_idx) const
                {
                    return accessor_->get_raw_block(row_idx, k_block_idx);
                }

                /**
                 * @brief Get block scale for output scaling
                 */
                float get_block_scale(size_t row_idx, size_t k_block_idx) const
                {
                    return accessor_->get_block_scale(row_idx, k_block_idx);
                }

                size_t block_size() const { return accessor_->block_size(); }
                size_t rows() const { return accessor_->rows(); }
                size_t cols() const { return accessor_->cols(); }

            private:
                const IQuantizedTileAccessor *accessor_;
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
