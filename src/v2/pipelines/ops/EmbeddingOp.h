/**
 * @file EmbeddingOp.h
 * @brief Self-validating embedding lookup operation with typed tensor support
 *
 * Handles embedding table lookup for transformer input processing:
 * - FP32 output: Direct memcpy from embedding table
 * - Q8_1 output: Lookup FP32 embeddings, then quantize to Q8_1 format
 *
 * The embedding table is always FP32 (stored in model weights).
 * Output format is determined by activation precision configuration.
 *
 * Usage:
 * @code
 * EmbeddingOp embedding;
 *
 * // Batched embedding lookup
 * if (!embedding(embed_table, token_batches, padded_seq_len, d_model, output))
 *     return false;
 * @endcode
 *
 * @author David Sanftenberg
 */

#pragma once

#include "Op.h"
#include "../../tensors/Tensors.h"
#include "../../tensors/SIMDHelpers.h"
#include <cstring>
#include <omp.h>

namespace llaminar2
{

    /**
     * @brief Self-validating embedding lookup operation
     *
     * Supports typed output:
     * - FP32Tensor: Direct copy from embedding table
     * - Q8_1Tensor: Copy then quantize to Q8_1 format (uses SIMD primitives)
     */
    class EmbeddingOp : public OpBase
    {
    public:
        const char *name() const override { return "EmbeddingOp"; }

        /**
         * @brief Execute batched embedding lookup
         *
         * @param embed_table Embedding table tensor [vocab_size, d_model] (always FP32)
         * @param token_batches Vector of token sequences, one per batch element
         * @param padded_seq_len Padded sequence length (all sequences padded to this)
         * @param d_model Model dimension (embedding size)
         * @param output Output tensor [batch_size * padded_seq_len, d_model]
         *               Can be FP32Tensor or Q8_1Tensor
         *
         * @return true on success, false on validation or execution failure
         */
        bool operator()(
            const TensorBase *embed_table,
            const std::vector<std::vector<int>> &token_batches,
            int padded_seq_len,
            int d_model,
            TensorBase *output)
        {
            // 1. Validate inputs
            if (!validateTensor(embed_table, "embed_table"))
                return false;
            if (!validateTensor(output, "output"))
                return false;

            const int batch_size = static_cast<int>(token_batches.size());
            if (batch_size == 0)
            {
                logError("empty token_batches");
                return false;
            }

            // 2. Get embedding table data (always FP32)
            const float *embed_data = embed_table->data();
            if (!embed_data)
            {
                logError("embed_table->data() returned nullptr");
                return false;
            }

            // 3. Dispatch based on output tensor type
            const TensorType out_type = output->native_type();

            switch (out_type)
            {
            case TensorType::FP32:
                return execute_fp32(embed_data, token_batches, padded_seq_len, d_model, output);

            case TensorType::Q8_1:
            {
                auto *q8_output = dynamic_cast<Q8_1Tensor *>(output);
                if (!q8_output)
                {
                    logError("output has Q8_1 type but dynamic_cast to Q8_1Tensor failed");
                    return false;
                }
                return execute_q8_1(embed_data, token_batches, padded_seq_len, d_model, q8_output);
            }

            default:
            {
                std::string msg = "unsupported output tensor type for embedding: " +
                                  std::to_string(static_cast<int>(out_type));
                logError(msg.c_str());
                return false;
            }
            }
        }

    private:
        /**
         * @brief FP32 embedding lookup: direct memcpy
         */
        bool execute_fp32(
            const float *embed_data,
            const std::vector<std::vector<int>> &token_batches,
            int padded_seq_len,
            int d_model,
            TensorBase *output)
        {
            float *output_data = output->mutable_data();
            if (!output_data)
            {
                logError("output->mutable_data() returned nullptr");
                return false;
            }

            const int batch_size = static_cast<int>(token_batches.size());
            int global_idx = 0;

            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = token_batches[b];
                const int seq_len = static_cast<int>(tokens.size());

                // Lookup embeddings for this sequence
                for (int i = 0; i < seq_len; ++i)
                {
                    int token_id = tokens[i];
                    std::memcpy(output_data + global_idx * d_model,
                                embed_data + token_id * d_model,
                                d_model * sizeof(float));
                    global_idx++;
                }

                // Pad remaining positions with zeros
                for (int i = seq_len; i < padded_seq_len; ++i)
                {
                    std::memset(output_data + global_idx * d_model, 0, d_model * sizeof(float));
                    global_idx++;
                }
            }

            LOG_TRACE("[EmbeddingOp] FP32 lookup: " << batch_size << " batches, "
                                                    << global_idx << " total positions");
            return true;
        }

        /**
         * @brief Q8_1 embedding lookup: lookup FP32 then quantize to Q8_1
         *
         * Strategy:
         * 1. Look up FP32 embedding from table
         * 2. Quantize each row (d_model elements) to Q8_1 blocks
         *
         * This is the correct place to quantize - at the pipeline input boundary.
         * All subsequent operations work on Q8_1 data.
         */
        bool execute_q8_1(
            const float *embed_data,
            const std::vector<std::vector<int>> &token_batches,
            int padded_seq_len,
            int d_model,
            Q8_1Tensor *output)
        {
            // Get mutable Q8_1 blocks
            Q8_1Block *output_blocks = output->mutable_q8_1_blocks();
            if (!output_blocks)
            {
                logError("output->mutable_q8_1_blocks() returned nullptr");
                return false;
            }

            // Q8_1 block count per row: d_model / 32 (32 elements per block)
            const int blocks_per_row = d_model / 32;
            if (d_model % 32 != 0)
            {
                std::string msg = "d_model must be multiple of 32 for Q8_1 quantization, got " +
                                  std::to_string(d_model);
                logError(msg.c_str());
                return false;
            }

            const int batch_size = static_cast<int>(token_batches.size());
            int global_idx = 0;

            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = token_batches[b];
                const int seq_len = static_cast<int>(tokens.size());

                // Lookup and quantize embeddings for this sequence
                for (int i = 0; i < seq_len; ++i)
                {
                    int token_id = tokens[i];
                    const float *embed_row = embed_data + token_id * d_model;
                    Q8_1Block *out_row = output_blocks + global_idx * blocks_per_row;

                    // Quantize FP32 row to Q8_1 blocks using SIMD primitives
                    simd::quantize_fp32_to_q8_1_blocks(embed_row, out_row, d_model);

                    global_idx++;
                }

                // Pad remaining positions with zero blocks
                for (int i = seq_len; i < padded_seq_len; ++i)
                {
                    Q8_1Block *out_row = output_blocks + global_idx * blocks_per_row;

                    // Zero out all blocks in this row
                    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx)
                    {
                        Q8_1Block &block = out_row[block_idx];
                        block.d = 0;
                        block.sum_qs = 0;
                        std::memset(block.qs, 0, 32);
                    }

                    global_idx++;
                }
            }

            LOG_TRACE("[EmbeddingOp] Q8_1 lookup+quantize: " << batch_size << " batches, "
                                                             << global_idx << " total positions, " << blocks_per_row << " blocks/row");
            return true;
        }
    };

} // namespace llaminar2
