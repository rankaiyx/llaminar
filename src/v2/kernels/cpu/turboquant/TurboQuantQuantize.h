/**
 * @file TurboQuantQuantize.h
 * @brief FP32 → TQ4/TQ3 quantization for TurboQuant KV cache
 * @author David Sanftenberg
 *
 * Quantizes FP32 vectors to the paper's inner-product-corrected TurboQuant
 * blocks.
 *
 * Provides both single-vector and batch (multi-head) entry points.
 */

#pragma once

#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantQJL.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace llaminar2
{

    inline void pack_bitplane_8(const uint8_t *bits, uint8_t *out)
    {
        uint8_t packed = 0;
        for (int i = 0; i < 8; ++i)
        {
            if (bits[i] & 0x1u)
                packed |= static_cast<uint8_t>(1u << i);
        }
        *out = packed;
    }

    // ========================================================================
    // Single-vector quantize: FP32 → TQ4
    // ========================================================================

    /**
     * @brief Quantize one FP32 vector to a TQ4 block.
     *
     * @tparam D Dimension of the vector (head_dim)
     * @param input FP32 vector of length D
     * @param ctx Pre-generated TurboQuant context (rotation + QJL projection)
     * @param out Output TQ4 block
     * @param scratch0 Scratch buffer of at least D floats
     * @param scratch1 Scratch buffer of at least D floats
     */
    template <int D>
    inline void turboquant_quantize_tq4(
        const float *input,
        const TurboQuantContext &ctx,
        TQ4Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
        float norm_sq = 0.0f;
        for (int i = 0; i < D; ++i)
            norm_sq += input[i] * input[i];
        float norm = std::sqrt(norm_sq);
        out.norm = norm;
        out.residual_norm = 0.0f;

        if (norm < 1e-30f) {
            std::memset(out.mse_indices, 0, TQ4Block<D>::MSE_BYTES);
            std::memset(out.qjl_signs, 0, TQ4Block<D>::QJL_BYTES);
            return;
        }

        float inv_norm = 1.0f / norm;
        float unit_vec[D];
        for (int i = 0; i < D; ++i)
            unit_vec[i] = input[i] * inv_norm;

        apply_rotation(ctx.rotation(), unit_vec, scratch0);

        float scale = std::sqrt(static_cast<float>(D));
        for (int i = 0; i < D; ++i)
            scratch0[i] *= scale;

        for (int i = 0; i < D; i += 8) {
            uint8_t idx8[8];
            uint8_t high_bits[8];
            for (int j = 0; j < 8; ++j)
            {
                idx8[j] = tq4_nearest_centroid(scratch0[i + j]);
                high_bits[j] = static_cast<uint8_t>((idx8[j] >> 3) & 0x1u);
                idx8[j] &= 0x7u;
            }
            tq3_pack_8(idx8, out.mse_indices + (i / 8) * 3);
            pack_bitplane_8(high_bits, out.qjl_signs + (i / 8));
        }
    }

    // ========================================================================
    // Single-vector quantize: FP32 → TQ3
    // ========================================================================

    /**
     * @brief Quantize one FP32 vector to a TQ3 block.
     *
     * @tparam D Dimension of the vector (head_dim)
     * @param input FP32 vector of length D
     * @param ctx Pre-generated TurboQuant context (rotation + QJL projection)
     * @param out Output TQ3 block
     * @param scratch0 Scratch buffer of at least D floats
     * @param scratch1 Scratch buffer of at least D floats
     */
    template <int D>
    inline void turboquant_quantize_tq3(
        const float *input,
        const TurboQuantContext &ctx,
        TQ3Block<D> &out,
        float *scratch0,
        float *scratch1)
    {
        float norm_sq = 0.0f;
        for (int i = 0; i < D; ++i)
            norm_sq += input[i] * input[i];
        float norm = std::sqrt(norm_sq);
        out.norm = norm;
        out.residual_norm = 0.0f;

        if (norm < 1e-30f) {
            std::memset(out.mse_indices, 0, TQ3Block<D>::MSE_BYTES);
            std::memset(out.qjl_signs, 0, TQ3Block<D>::QJL_BYTES);
            return;
        }

        float inv_norm = 1.0f / norm;
        float unit_vec[D];
        for (int i = 0; i < D; ++i)
            unit_vec[i] = input[i] * inv_norm;

        apply_rotation(ctx.rotation(), unit_vec, scratch0);

        float scale = std::sqrt(static_cast<float>(D));
        for (int i = 0; i < D; ++i)
            scratch0[i] *= scale;

        for (int i = 0; i < D; i += 8) {
            uint8_t high_bits[8];
            for (int j = 0; j < 8; j += 4)
            {
                uint8_t idx4[4];
                for (int k = 0; k < 4; ++k)
                {
                    idx4[k] = tq3_nearest_centroid(scratch0[i + j + k]);
                    high_bits[j + k] = static_cast<uint8_t>((idx4[k] >> 2) & 0x1u);
                    idx4[k] &= 0x3u;
                }
                tq2_pack_4(idx4, out.mse_indices + ((i + j) / 4));
            }
            pack_bitplane_8(high_bits, out.qjl_signs + (i / 8));
        }
    }

    // ========================================================================
    // Batch quantize: multiple vectors (e.g., all KV heads in a layer)
    // ========================================================================

    /**
     * @brief Quantize a batch of FP32 vectors to TQ4 blocks.
     *
    * Each vector is independently quantized using the same TurboQuant context.
     * Suitable for quantizing K or V vectors across all heads in a layer.
     *
     * @tparam D Dimension of each vector (head_dim)
     * @param input Pointer to num_vectors × D contiguous FP32 data (row-major)
     * @param ctx Pre-generated TurboQuant context
     * @param out Pointer to num_vectors TQ4 blocks
     * @param num_vectors Number of vectors to quantize
     */
    template <int D>
    inline void turboquant_quantize_tq4_batch(
        const float *input,
        const TurboQuantContext &ctx,
        TQ4Block<D> *out,
        int num_vectors)
    {
        float scratch0[D];
        float scratch1[D];

        for (int v = 0; v < num_vectors; ++v) {
            turboquant_quantize_tq4<D>(
                input + v * D,
                ctx,
                out[v],
                scratch0,
                scratch1);
        }
    }

    /**
     * @brief Quantize a batch of FP32 vectors to TQ3 blocks.
     *
     * @tparam D Dimension of each vector (head_dim)
     * @param input Pointer to num_vectors × D contiguous FP32 data (row-major)
     * @param ctx Pre-generated TurboQuant context
     * @param out Pointer to num_vectors TQ3 blocks
     * @param num_vectors Number of vectors to quantize
     */
    template <int D>
    inline void turboquant_quantize_tq3_batch(
        const float *input,
        const TurboQuantContext &ctx,
        TQ3Block<D> *out,
        int num_vectors)
    {
        float scratch0[D];
        float scratch1[D];

        for (int v = 0; v < num_vectors; ++v) {
            turboquant_quantize_tq3<D>(
                input + v * D,
                ctx,
                out[v],
                scratch0,
                scratch1);
        }
    }

} // namespace llaminar2
