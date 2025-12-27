/**
 * @file Int8RequantRef.cpp
 * @brief INT8 requantization utilities for Q16_1 attention
 */

#include "Int8RequantRef.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace llaminar2::kernels::q16_1::microkernels
{

    namespace
    {

        inline int8_t clamp_to_int8(int v)
        {
            if (v > 127)
                return 127;
            if (v < -127)
                return -127;
            return static_cast<int8_t>(v);
        }

    } // namespace

    float q16_maxabs_head_row(
        const Q16_1Block *blocks,
        int blocks_per_row,
        int start_col,
        int head_dim)
    {
        float max_abs = 0.0f;
        for (int i = 0; i < head_dim; ++i)
        {
            const int col = start_col + i;
            const int block_idx = col / Q16_1Block::BLOCK_SIZE;
            const int in_block = col % Q16_1Block::BLOCK_SIZE;
            if (block_idx < 0 || block_idx >= blocks_per_row)
                continue;
            const auto &b = blocks[block_idx];
            const float v = static_cast<float>(b.qs[in_block]) * b.d;
            max_abs = std::max(max_abs, std::fabs(v));
        }
        return max_abs;
    }

    float q16_maxabs_head_across_rows(
        const Q16_1Block *blocks,
        int num_rows,
        int blocks_per_row,
        int start_col,
        int head_dim)
    {
        float max_abs = 0.0f;
        for (int r = 0; r < num_rows; ++r)
        {
            const auto *row = blocks + r * blocks_per_row;
            max_abs = std::max(max_abs, q16_maxabs_head_row(row, blocks_per_row, start_col, head_dim));
        }
        return max_abs;
    }

    void q16_quantize_head_to_int8(
        const Q16_1Block *blocks,
        int blocks_per_row,
        int start_col,
        int head_dim,
        float scale,
        int8_t *out)
    {
        if (scale <= 0.0f)
        {
            std::memset(out, 0, static_cast<size_t>(head_dim));
            return;
        }

        const float inv = 1.0f / scale;
        for (int i = 0; i < head_dim; ++i)
        {
            const int col = start_col + i;
            const int block_idx = col / Q16_1Block::BLOCK_SIZE;
            const int in_block = col % Q16_1Block::BLOCK_SIZE;
            const auto &b = blocks[block_idx];
            const float v = static_cast<float>(b.qs[in_block]) * b.d;
            const int q = static_cast<int>(std::lrint(v * inv));
            out[i] = clamp_to_int8(q);
        }
    }

    int32_t dot_int8(const int8_t *a, const int8_t *b, int n)
    {
        int32_t acc = 0;
        for (int i = 0; i < n; ++i)
        {
            acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
        }
        return acc;
    }

    void compute_int8_requant_logits(
        const Int8RequantParams &params,
        int32_t *scores,
        float *alpha_out)
    {
        const int head_dim = params.head_dim;
        const int kv_end = params.kv_end;
        const int q_start_col = params.head * head_dim;
        const int k_start_col = params.kv_head * head_dim;

        std::vector<int8_t> q8(static_cast<size_t>(head_dim));
        std::vector<int8_t> k8(static_cast<size_t>(head_dim));

        // Compute Q scale and requantize
        const auto *q_row_blocks = params.Q + params.q_row * params.q_blocks_per_row;
        const float q_max = q16_maxabs_head_row(q_row_blocks, params.q_blocks_per_row, q_start_col, head_dim);
        const float sQ = (q_max > 0.0f) ? (q_max / 127.0f) : 1.0f;
        q16_quantize_head_to_int8(q_row_blocks, params.q_blocks_per_row, q_start_col, head_dim, sQ, q8.data());

        // Compute K scale (across all KV positions)
        const float k_max = q16_maxabs_head_across_rows(
            params.K, kv_end, params.kv_blocks_per_row, k_start_col, head_dim);
        const float sK = (k_max > 0.0f) ? (k_max / 127.0f) : 1.0f;

        // Combined alpha
        *alpha_out = sQ * sK * params.attention_scale;

        // Compute INT8 dot products
        for (int kv = 0; kv < kv_end; ++kv)
        {
            const auto *k_row_blocks = params.K + kv * params.kv_blocks_per_row;
            q16_quantize_head_to_int8(k_row_blocks, params.kv_blocks_per_row, k_start_col, head_dim, sK, k8.data());
            scores[kv] = dot_int8(q8.data(), k8.data(), head_dim);
        }
    }

} // namespace llaminar2::kernels::q16_1::microkernels
