/**
 * @file TurboQuantDequantize.h
 * @brief TQ4/TQ3 → FP32 dequantization for TurboQuant KV cache
 * @author David Sanftenberg
 *
 * Dequantizes TurboQuant prod blocks back to FP32 vectors.
 *
 * Provides both single-vector and batch entry points.
 */

#pragma once

#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantQJL.h"

#include <cmath>
#include <cstddef>
#include <cstring>

namespace llaminar2
{

    inline void unpack_bitplane_8(const uint8_t *packed, uint8_t *out)
    {
        const uint8_t bits = *packed;
        for (int i = 0; i < 8; ++i)
            out[i] = static_cast<uint8_t>((bits >> i) & 0x1u);
    }

    // ========================================================================
    // Single-vector dequantize: TQ4 → FP32
    // ========================================================================

    /**
     * @brief Dequantize one TQ4 block to FP32 vector.
     *
     * @tparam D Dimension of the vector (head_dim)
     * @param block Input TQ4 block
     * @param ctx Pre-generated TurboQuant context
     * @param output FP32 output vector of length D
     * @param scratch Scratch buffer of at least D floats (for centroid vector)
     */
    template <int D>
    inline void turboquant_dequantize_tq4(
        const TQ4Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        if (block.norm < 1e-30f) {
            for (int i = 0; i < D; ++i)
                output[i] = 0.0f;
            return;
        }

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (int i = 0; i < D; i += 8) {
            uint8_t idx8[8];
            uint8_t high_bits[8];
            tq3_unpack_8(block.mse_indices + (i / 8) * 3, idx8);
            unpack_bitplane_8(block.qjl_signs + (i / 8), high_bits);
            for (int j = 0; j < 8; ++j)
                scratch[i + j] = TQ4_CENTROIDS[idx8[j] | static_cast<uint8_t>(high_bits[j] << 3)] * inv_scale;
        }

        apply_rotation_transpose(ctx.rotation(), scratch, output);

        for (int i = 0; i < D; ++i)
            output[i] *= block.norm;
    }

    // ========================================================================
    // Single-vector dequantize: TQ3 → FP32
    // ========================================================================

    /**
     * @brief Dequantize one TQ3 block to FP32 vector.
     *
     * @tparam D Dimension of the vector (head_dim)
     * @param block Input TQ3 block
     * @param ctx Pre-generated TurboQuant context
     * @param output FP32 output vector of length D
     * @param scratch Scratch buffer of at least D floats
     */
    template <int D>
    inline void turboquant_dequantize_tq3(
        const TQ3Block<D> &block,
        const TurboQuantContext &ctx,
        float *output,
        float *scratch)
    {
        if (block.norm < 1e-30f) {
            for (int i = 0; i < D; ++i)
                output[i] = 0.0f;
            return;
        }

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (int i = 0; i < D; i += 8) {
            uint8_t high_bits[8];
            unpack_bitplane_8(block.qjl_signs + (i / 8), high_bits);
            for (int j = 0; j < 8; j += 4) {
                uint8_t idx4[4];
                tq2_unpack_4(block.mse_indices + ((i + j) / 4), idx4);
                for (int k = 0; k < 4; ++k)
                    scratch[i + j + k] = TQ3_CENTROIDS[idx4[k] | static_cast<uint8_t>(high_bits[j + k] << 2)] * inv_scale;
            }
        }

        apply_rotation_transpose(ctx.rotation(), scratch, output);

        for (int i = 0; i < D; ++i)
            output[i] *= block.norm;
    }

    // ========================================================================
    // Batch dequantize
    // ========================================================================

    /**
     * @brief Dequantize a batch of TQ4 blocks to FP32.
     *
     * @tparam D Dimension of each vector (head_dim)
     * @param blocks Pointer to num_vectors TQ4 blocks
     * @param ctx Pre-generated TurboQuant context
     * @param output Pointer to num_vectors × D contiguous FP32 data (row-major)
     * @param num_vectors Number of vectors to dequantize
     */
    template <int D>
    inline void turboquant_dequantize_tq4_batch(
        const TQ4Block<D> *blocks,
        const TurboQuantContext &ctx,
        float *output,
        int num_vectors)
    {
        float scratch[D];

        for (int v = 0; v < num_vectors; ++v) {
            turboquant_dequantize_tq4<D>(
                blocks[v],
                ctx,
                output + v * D,
                scratch);
        }
    }

    /**
     * @brief Dequantize a batch of TQ3 blocks to FP32.
     *
     * @tparam D Dimension of each vector (head_dim)
     * @param blocks Pointer to num_vectors TQ3 blocks
     * @param ctx Pre-generated TurboQuant context
     * @param output Pointer to num_vectors × D contiguous FP32 data (row-major)
     * @param num_vectors Number of vectors to dequantize
     */
    template <int D>
    inline void turboquant_dequantize_tq3_batch(
        const TQ3Block<D> *blocks,
        const TurboQuantContext &ctx,
        float *output,
        int num_vectors)
    {
        float scratch[D];

        for (int v = 0; v < num_vectors; ++v) {
            turboquant_dequantize_tq3<D>(
                blocks[v],
                ctx,
                output + v * D,
                scratch);
        }
    }

} // namespace llaminar2
