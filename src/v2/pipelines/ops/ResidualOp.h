/**
 * @file ResidualOp.h
 * @brief Self-validating residual connection operation
 *
 * Residual connections are used throughout transformers:
 * - After attention: hidden = residual + attn_output
 * - After FFN: hidden = residual + ffn_output
 *
 * Handles batch padding by zeroing out padding positions to prevent NaN propagation.
 *
 * Native Q8_1 Support (June 2025):
 * - When all three tensors are Q8_1, uses native q8_1_add_q8_1 SIMD primitive
 * - Avoids unnecessary FP32 dequant/requant overhead
 * - Falls back to FP32 for mixed precision operations
 *
 * Usage:
 * @code
 * ResidualOp residual;
 *
 * // Simple residual (no padding handling)
 * TRY_OP(residual(saved_residual, projection_output, current_hidden,
 *                 seq_len, d_model, "layer0_ATTN_RESIDUAL"));
 *
 * // Batched residual with padding mask
 * TRY_OP(residual.batched(saved_residual, projection_output, current_hidden,
 *                         batch_size, padded_seq_len, d_model, sequence_lengths,
 *                         "layer0_FFN_RESIDUAL"));
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "../../tensors/SIMDHelpers.h"
#include <omp.h>

namespace llaminar2
{

    /**
     * @brief Self-validating residual connection operation
     *
     * Replaces the verbose pattern:
     * @code
     * const size_t elements = seq_len * d_model;
     * #pragma omp parallel for
     * for (size_t i = 0; i < elements; ++i) {
     *     output->mutable_data()[i] = residual->data()[i] + projection->data()[i];
     * }
     * CAPTURE_SNAPSHOT("layer0_ATTN_RESIDUAL", output);
     * @endcode
     *
     * With:
     * @code
     * TRY_OP(residual(residual_buf, projection, output, seq_len, d_model, "ATTN_RESIDUAL"));
     * @endcode
     */
    class ResidualOp : public OpBase
    {
    public:
        const char *name() const override { return "ResidualOp"; }

        /**
         * @brief Execute simple residual connection: output = residual + input
         *
         * Supports native Q8_1 precision when all tensors are Q8_1:
         * - Uses q8_1_add_q8_1 SIMD primitive (AVX512/AVX2 optimized)
         * - Avoids FP32 dequant/requant overhead
         *
         * Falls back to FP32 for:
         * - Mixed precision (different tensor types)
         * - Non-Q8_1 tensors
         *
         * @param residual Saved residual tensor [rows, cols]
         * @param input Projection output to add [rows, cols]
         * @param output Destination tensor [rows, cols] (can be same as input)
         * @param rows Number of rows
         * @param cols Number of columns
         * @param snapshot_key Snapshot identifier (nullptr to skip)
         *
         * @return true on success, false on validation failure
         */
        bool operator()(
            const TensorBase *residual,
            const TensorBase *input,
            TensorBase *output,
            int rows,
            int cols,
            const char *snapshot_key = nullptr)
        {
            // 1. Validate inputs
            if (!validateTensor(residual, "residual"))
                return false;
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (!validateDimensions(rows, cols, "residual"))
                return false;

            const size_t elements = static_cast<size_t>(rows) * cols;

            // 2. Check for native Q8_1 path (all three tensors must be Q8_1)
            if (residual->native_type() == TensorType::Q8_1 &&
                input->native_type() == TensorType::Q8_1 &&
                output->native_type() == TensorType::Q8_1)
            {
                // Native Q8_1 path: no FP32 conversion
                auto *q8_1_residual = dynamic_cast<const Q8_1Tensor *>(residual);
                auto *q8_1_input = dynamic_cast<const Q8_1Tensor *>(input);
                auto *q8_1_output = dynamic_cast<Q8_1Tensor *>(output);

                if (q8_1_residual && q8_1_input && q8_1_output)
                {
                    // Validate element count is multiple of 32 (Q8_1 block size)
                    if (elements % 32 != 0)
                    {
                        LOG_ERROR(name() << ": Q8_1 residual requires element count multiple of 32, got " << elements);
                        return false;
                    }

                    // mutable_q8_1_blocks() already invalidates the FP32 dequant cache
                    simd::q8_1_add_q8_1(
                        q8_1_residual->q8_1_blocks(),
                        q8_1_input->q8_1_blocks(),
                        q8_1_output->mutable_q8_1_blocks(),
                        elements);

                    (void)snapshot_key;
                    return true;
                }
            }

            // 3. FP32 fallback path
            const float *res_data = residual->data();
            const float *in_data = input->data();
            float *out_data = output->mutable_data();

#pragma omp parallel for
            for (size_t i = 0; i < elements; ++i)
            {
                out_data[i] = res_data[i] + in_data[i];
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }

        /**
         * @brief Execute batched residual with padding mask
         *
         * Zeros out padding positions to prevent NaN propagation through layers.
         * Critical for batched inference with variable-length sequences.
         *
         * NOTE: Currently uses FP32 path even for Q8_1 tensors because:
         * - Padding mask handling requires per-element granularity
         * - Q8_1 blocks are 32 elements, making partial zeroing complex
         * - Could optimize with block-aligned padding in future
         *
         * @param residual Saved residual tensor [batch_size * padded_seq_len, cols]
         * @param input Projection output [batch_size * padded_seq_len, cols]
         * @param output Destination tensor [batch_size * padded_seq_len, cols]
         * @param batch_size Number of sequences in batch
         * @param padded_seq_len Maximum sequence length (padded)
         * @param cols Feature dimension (d_model)
         * @param sequence_lengths Actual lengths per sequence [batch_size]
         * @param snapshot_key Snapshot identifier (nullptr to skip)
         *
         * @return true on success, false on validation failure
         */
        bool batched(
            const TensorBase *residual,
            const TensorBase *input,
            TensorBase *output,
            int batch_size,
            int padded_seq_len,
            int cols,
            const std::vector<int> &sequence_lengths,
            const char *snapshot_key = nullptr)
        {
            // 1. Validate inputs
            if (!validateTensor(residual, "residual"))
                return false;
            if (!validateTensor(input, "input"))
                return false;
            if (!validateTensor(output, "output"))
                return false;
            if (batch_size <= 0 || padded_seq_len <= 0 || cols <= 0)
            {
                LOG_ERROR(name() << ": invalid dimensions (batch_size=" << batch_size
                                 << ", padded_seq_len=" << padded_seq_len << ", cols=" << cols << ")");
                return false;
            }
            if (static_cast<int>(sequence_lengths.size()) != batch_size)
            {
                LOG_ERROR(name() << ": sequence_lengths size mismatch (expected " << batch_size
                                 << ", got " << sequence_lengths.size() << ")");
                return false;
            }

            // 2. Execute batched residual with padding mask
            const size_t total_elements = static_cast<size_t>(batch_size) * padded_seq_len * cols;
            const float *res_data = residual->data();
            const float *in_data = input->data();
            float *out_data = output->mutable_data();

#pragma omp parallel for
            for (size_t i = 0; i < total_elements; ++i)
            {
                size_t token_idx = i / cols;
                size_t batch_idx = token_idx / padded_seq_len;
                size_t seq_idx = token_idx % padded_seq_len;

                // Zero out padding positions to prevent NaN propagation
                if (static_cast<int>(seq_idx) >= sequence_lengths[batch_idx])
                {
                    out_data[i] = 0.0f;
                }
                else
                {
                    out_data[i] = res_data[i] + in_data[i];
                }
            }

            // Note: Snapshot capture is handled by the calling pipeline
            (void)snapshot_key;

            return true;
        }
    };

} // namespace llaminar2
