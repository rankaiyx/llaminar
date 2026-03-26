/**
 * @file TurboQuantDequantize.h
 * @brief TQ4 → FP32 dequantization for TurboQuant KV cache
 * @author David Sanftenberg
 *
 * Dequantizes TQ4 scalar-full blocks back to FP32 vectors.
 *
 * Provides both single-vector and batch entry points.
 */

#pragma once

#include "tensors/BlockStructures.h"
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantRotation.h"
#include "utils/OpenMPUtils.h"

#include <cmath>
#include <cstddef>
#include <cstring>

namespace llaminar2
{

    /// Unpack one byte into 8 individual bits (LSB first).
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
     * Unpacks 4-bit centroid indices (3 low bits from mse_indices +
     * 1 high bit from high_bits), looks up TQ4_CENTROIDS, applies
     * inverse rotation, and scales by the stored norm.
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
        if (block.norm < 1e-30f)
        {
            for (int i = 0; i < D; ++i)
                output[i] = 0.0f;
            return;
        }

        const float inv_scale = 1.0f / std::sqrt(static_cast<float>(D));
        for (int i = 0; i < D; i += 8)
        {
            uint8_t idx8[8];
            uint8_t high_bits[8];
            tq3_unpack_8(block.mse_indices + (i / 8) * 3, idx8);
            unpack_bitplane_8(block.high_bits + (i / 8), high_bits);
            for (int j = 0; j < 8; ++j)
                scratch[i + j] = TQ4_CENTROIDS[idx8[j] | static_cast<uint8_t>(high_bits[j] << 3)] * inv_scale;
        }

        apply_rotation_transpose(ctx.rotation(), scratch, output);

        for (int i = 0; i < D; ++i)
            output[i] *= block.norm;
    }

    // ========================================================================
    // Row-range dequantization helpers for AttentionComputeStage
    //
    // These wrap the per-block dequantize in a row/head loop so that
    // the stage class only needs a single call per path.
    // ========================================================================

    /**
     * @brief Dequantize V rows from TQ4 to FP32.
     *
     * @param v_raw      Raw TQ4 block bytes (TQ4Tensor::typed_data())
     * @param ctx        Layer-level TurboQuant context (rotation matrices)
     * @param v_fp32     Output FP32 buffer, layout [kv_len × kv_dim]
     * @param from_row   First row to dequantize (inclusive)
     * @param to_row     Last row to dequantize (exclusive)
     * @param head_dim   Head dimension (must be 128 on this path)
     * @param n_kv_heads Number of KV heads
     * @param row_bytes  Bytes per row (blocks_per_row * block_bytes)
     * @param block_bytes Bytes per single TQ4 block
     */
    inline void turboquant_dequantize_v_rows(
        const uint8_t *v_raw,
        const TurboQuantContext &ctx,
        float *v_fp32,
        int from_row, int to_row,
        int head_dim, int n_kv_heads,
        size_t row_bytes, size_t block_bytes)
    {
        const int kv_dim = n_kv_heads * head_dim;
        const int num_rows = to_row - from_row;
        auto work = [&]()
        {
#pragma omp for schedule(static)
            for (int r = from_row; r < to_row; ++r)
            {
                float *v_dst = v_fp32 + r * kv_dim;
                const uint8_t *v_row = v_raw + static_cast<size_t>(r) * row_bytes;
                alignas(64) float scratch[128];

                for (int h = 0; h < n_kv_heads; ++h)
                {
                    const auto &head_ctx = ctx.for_layer(h);
                    const auto *vb = reinterpret_cast<const TQ4Block_128 *>(
                        v_row + static_cast<size_t>(h) * block_bytes);
                    turboquant_dequantize_tq4(*vb, head_ctx,
                                              v_dst + h * head_dim, scratch);
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work, num_rows >= 4);
    }

    /**
     * @brief Dequantize both K and V rows from TQ4 to FP32.
     *
     * @param k_raw      Raw TQ4 block bytes for K (TQ4Tensor::typed_data())
     * @param v_raw      Raw TQ4 block bytes for V (TQ4Tensor::typed_data())
     * @param ctx        Layer-level TurboQuant context (rotation matrices)
     * @param k_fp32     Output FP32 buffer for K, layout [kv_len × kv_dim]
     * @param v_fp32     Output FP32 buffer for V, layout [kv_len × kv_dim]
     * @param from_row   First row to dequantize (inclusive)
     * @param to_row     Last row to dequantize (exclusive)
     * @param head_dim   Head dimension (64 or 128)
     * @param n_kv_heads Number of KV heads
     * @param k_row_bytes Bytes per K row
     * @param v_row_bytes Bytes per V row
     * @param k_block_bytes Bytes per single K block
     * @param v_block_bytes Bytes per single V block
     */
    inline void turboquant_dequantize_kv_rows(
        const uint8_t *k_raw,
        const uint8_t *v_raw,
        const TurboQuantContext &ctx,
        float *k_fp32, float *v_fp32,
        int from_row, int to_row,
        int head_dim, int n_kv_heads,
        size_t k_row_bytes, size_t v_row_bytes,
        size_t k_block_bytes, size_t v_block_bytes)
    {
        const int kv_dim = n_kv_heads * head_dim;
        const int num_rows = to_row - from_row;
        auto work = [&]()
        {
#pragma omp for schedule(static)
            for (int r = from_row; r < to_row; ++r)
            {
                const uint8_t *k_row = k_raw + static_cast<size_t>(r) * k_row_bytes;
                const uint8_t *v_row = v_raw + static_cast<size_t>(r) * v_row_bytes;
                float *k_dst = k_fp32 + r * kv_dim;
                float *v_dst = v_fp32 + r * kv_dim;
                alignas(64) float scratch[128];

                for (int h = 0; h < n_kv_heads; ++h)
                {
                    const auto &head_ctx = ctx.for_layer(h);
                    if (head_dim == 128)
                    {
                        const auto *kb = reinterpret_cast<const TQ4Block_128 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq4(*kb, head_ctx,
                                                  k_dst + h * head_dim, scratch);
                        const auto *vb = reinterpret_cast<const TQ4Block_128 *>(
                            v_row + static_cast<size_t>(h) * v_block_bytes);
                        turboquant_dequantize_tq4(*vb, head_ctx,
                                                  v_dst + h * head_dim, scratch);
                    }
                    else
                    {
                        const auto *kb = reinterpret_cast<const TQ4Block_64 *>(
                            k_row + static_cast<size_t>(h) * k_block_bytes);
                        turboquant_dequantize_tq4(*kb, head_ctx,
                                                  k_dst + h * head_dim, scratch);
                        const auto *vb = reinterpret_cast<const TQ4Block_64 *>(
                            v_row + static_cast<size_t>(h) * v_block_bytes);
                        turboquant_dequantize_tq4(*vb, head_ctx,
                                                  v_dst + h * head_dim, scratch);
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION_IF(work, num_rows >= 4);
    }

} // namespace llaminar2
