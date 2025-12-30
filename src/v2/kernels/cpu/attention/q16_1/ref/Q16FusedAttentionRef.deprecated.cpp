/**
 * @file Q16FusedAttentionRef.cpp
 * @brief Scalar reference implementation of Q16_1 fused attention (FP32 scores + Q16 context)
 *
 * This file implements the FULL fused attention kernel with two execution paths:
 *
 * FLASH DECODE (seq_len_q=1):
 * - Single query against full KV cache
 * - Parallel over KV positions, no tiling
 * - FP32 softmax (standard exp-based)
 * - Steps: Q×K^T (FP32) → Softmax (FP32) → P×V (INT32) → Wo (Q16_1) → Residual (Q16_1)
 * - Optimized for latency in token generation
 *
 * FA2 PREFILL (seq_len_q>1):
 * - Batched queries with per-query processing
 * - FP32 softmax per query
 * - Steps: [Per-query: Q×K^T → Softmax → P×V] → Wo → Residual
 * - Optimized for throughput in prompt processing
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * FP32 SCORES + Q16 CONTEXT PIPELINE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * The attention pipeline uses FP32 for scores/softmax, Q16 for context/output:
 *
 * 1. Q×K^T → FP32 scores
 *    - Q16DotProductRef: Proper block-scale handling
 *    - INT16×INT16 → INT32 with FP32 block scale accumulation
 *    - Full Q16 precision preserved (no INT8 requant!)
 *
 * 2. FP32 Softmax → FP32 weights
 *    - Standard exp-based softmax with max subtraction
 *    - Outputs FP32 weights [0.0, 1.0]
 *
 * 3. P×V → INT32 accumulators
 *    - FP32 weights scaled to INT32 × INT16 V values
 *    - Tracks weighted v_scale for proper output scaling
 *
 * 4. Wo Projection (VPDPWSSD) → Q16_1 output
 *    - INT32→INT16 context requantization (keeps 16 bits!)
 *    - VPDPWSSD: INT16 context × INT16 weights → INT32
 *    - INT32→Q16_1 output requantization
 *
 * 5. Native Q16_1 Residual Add
 *    - Uses simd::q16_1_add_q16_1() for Q16_1 + Q16_1
 *    - No FP32 conversion in the residual path!
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * @see Q16FusedAttentionRef.h for API documentation
 * @see PROJECT_Q16_INTEGER_ATTENTION.md for design details
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "Q16FusedAttentionRef.deprecated.h"
#include "microkernels/Q16DotProductRef.deprecated.h"
#include "microkernels/Exp2FixedSoftmaxRef.deprecated.h"
#include "microkernels/Int8RequantRef.deprecated.h"
#include "microkernels/WoProjectionVNNIRef.deprecated.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Assertions.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <atomic>

namespace llaminar2::kernels::q16_1
{

    // Import microkernel namespace for convenience
    using namespace microkernels;

    namespace
    {
        struct TopKEntry
        {
            int idx = -1;
            float value = 0.0f;
        };

        static std::vector<TopKEntry> compute_topk(const float *values, int n, int k)
        {
            std::vector<TopKEntry> out;
            if (!values || n <= 0 || k <= 0)
            {
                return out;
            }
            out.reserve(static_cast<size_t>(k));

            for (int i = 0; i < n; ++i)
            {
                const float v = values[i];
                if (static_cast<int>(out.size()) < k)
                {
                    out.push_back({i, v});
                    // insertion sort step
                    for (int j = static_cast<int>(out.size()) - 1; j > 0 && out[j].value > out[j - 1].value; --j)
                    {
                        std::swap(out[j], out[j - 1]);
                    }
                }
                else if (v > out.back().value)
                {
                    out.back() = {i, v};
                    for (int j = static_cast<int>(out.size()) - 1; j > 0 && out[j].value > out[j - 1].value; --j)
                    {
                        std::swap(out[j], out[j - 1]);
                    }
                }
            }
            return out;
        }

        static std::vector<TopKEntry> compute_topk_i32(const int32_t *values, int n, int k)
        {
            std::vector<TopKEntry> out;
            if (!values || n <= 0 || k <= 0)
            {
                return out;
            }
            out.reserve(static_cast<size_t>(k));

            for (int i = 0; i < n; ++i)
            {
                const float v = static_cast<float>(values[i]);
                if (static_cast<int>(out.size()) < k)
                {
                    out.push_back({i, v});
                    for (int j = static_cast<int>(out.size()) - 1; j > 0 && out[j].value > out[j - 1].value; --j)
                    {
                        std::swap(out[j], out[j - 1]);
                    }
                }
                else if (v > out.back().value)
                {
                    out.back() = {i, v};
                    for (int j = static_cast<int>(out.size()) - 1; j > 0 && out[j].value > out[j - 1].value; --j)
                    {
                        std::swap(out[j], out[j - 1]);
                    }
                }
            }
            return out;
        }

        static float cosine_similarity(const float *a, const float *b, int n)
        {
            if (!a || !b || n <= 0)
            {
                return 0.0f;
            }
            double dot = 0.0;
            double na = 0.0;
            double nb = 0.0;
            for (int i = 0; i < n; ++i)
            {
                const double da = static_cast<double>(a[i]);
                const double db = static_cast<double>(b[i]);
                dot += da * db;
                na += da * da;
                nb += db * db;
            }
            if (na <= 0.0 || nb <= 0.0)
            {
                return 0.0f;
            }
            return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
        }

        static float dequant_q16_1_elem(const Q16_1Block *row, int blocks_per_row, int col)
        {
            constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
            const int block_idx = col / BLOCK_SIZE;
            const int elem_idx = col % BLOCK_SIZE;
            const Q16_1Block &b = row[block_idx];
            (void)blocks_per_row;
            return static_cast<float>(b.qs[elem_idx]) * b.d;
        }

        static void compute_slow_scores_qk(
            const Q16FusedAttentionWoResidualParams &params,
            int q_row,
            int head,
            int kv_end,
            float *out_scores)
        {
            if (!out_scores || kv_end <= 0)
            {
                return;
            }

            const int head_dim = params.head_dim;
            const int kv_head = params.get_kv_head(head);
            const int q_blocks_per_row = params.q_blocks_per_row();
            const int kv_blocks_per_row = params.kv_blocks_per_row();
            const float attention_scale = params.get_scale();

            const int q_start_col = head * head_dim;
            const int k_start_col = kv_head * head_dim;

            const Q16_1Block *q_row_ptr = params.Q + q_row * q_blocks_per_row;
            for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
            {
                const Q16_1Block *k_row_ptr = params.K + kv_pos * kv_blocks_per_row;
                double sum = 0.0;
                for (int d = 0; d < head_dim; ++d)
                {
                    const float qv = dequant_q16_1_elem(q_row_ptr, q_blocks_per_row, q_start_col + d);
                    const float kv = dequant_q16_1_elem(k_row_ptr, kv_blocks_per_row, k_start_col + d);
                    sum += static_cast<double>(qv) * static_cast<double>(kv);
                }
                out_scores[kv_pos] = static_cast<float>(sum) * attention_scale;
            }

            // Ultra-verbose element dump for debugging
            const auto &cfg = debugEnv().q16_attention_dump;
            if (cfg.enabled && params.layer_idx == cfg.layer && head == cfg.head)
            {
                LOG_DEBUG("[Q16AttnDump] SLOW_SCORE_ELEM layer=" << params.layer_idx << " head=" << head 
                          << " q_row=" << q_row << " kv_pos=0"
                          << " q_start_col=" << q_start_col << " k_start_col=" << k_start_col
                          << " q_blocks_per_row=" << q_blocks_per_row << " kv_blocks_per_row=" << kv_blocks_per_row);
                
                // Recompute for kv_pos=0 with element-level trace
                const Q16_1Block *k0_row_ptr = params.K + 0 * kv_blocks_per_row;
                double trace_sum = 0.0;
                std::ostringstream first8, last8;
                for (int d = 0; d < head_dim; ++d)
                {
                    const float qv = dequant_q16_1_elem(q_row_ptr, q_blocks_per_row, q_start_col + d);
                    const float kv = dequant_q16_1_elem(k0_row_ptr, kv_blocks_per_row, k_start_col + d);
                    trace_sum += static_cast<double>(qv) * static_cast<double>(kv);
                    if (d < 8)
                        first8 << "d" << d << ":(q=" << qv << ",k=" << kv << ",p=" << (qv * kv) << ") ";
                    if (d >= head_dim - 8)
                        last8 << "d" << d << ":(q=" << qv << ",k=" << kv << ",p=" << (qv * kv) << ") ";
                }
                LOG_DEBUG("[Q16AttnDump] SLOW_SCORE_ELEM first8: " << first8.str());
                LOG_DEBUG("[Q16AttnDump] SLOW_SCORE_ELEM last8: " << last8.str());
                LOG_DEBUG("[Q16AttnDump] SLOW_SCORE_ELEM raw_dot=" << trace_sum << " scaled=" << (trace_sum * attention_scale));
            }
        }

        static void compute_context_from_fp32_weights(
            const Q16FusedAttentionWoResidualParams &params,
            int kv_end,
            int kv_head,
            const float *weights,
            float *out_ctx)
        {
            const int head_dim = params.head_dim;
            const int kv_blocks_per_row = params.kv_blocks_per_row();
            constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

            if (!weights || !out_ctx || kv_end <= 0 || head_dim <= 0)
            {
                return;
            }

            std::memset(out_ctx, 0, static_cast<size_t>(head_dim) * sizeof(float));

            const int v_start_col = kv_head * head_dim;
            for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
            {
                const float w = weights[kv_pos];
                if (w == 0.0f)
                    continue;

                const Q16_1Block *v_row_ptr = params.V + kv_pos * kv_blocks_per_row;
                for (int d = 0; d < head_dim; ++d)
                {
                    const int v_col = v_start_col + d;
                    const int v_block_idx = v_col / BLOCK_SIZE;
                    const int v_elem_idx = v_col % BLOCK_SIZE;
                    const Q16_1Block &v_block = v_row_ptr[v_block_idx];
                    const float v = static_cast<float>(v_block.qs[v_elem_idx]) * v_block.d;
                    out_ctx[d] += w * v;
                }
            }
        }

        static bool should_dump_q16_attention(const Q16FusedAttentionWoResidualParams &params, int head, int q_row)
        {
            const auto &cfg = debugEnv().q16_attention_dump;
            if (!cfg.enabled)
            {
                return false;
            }
            if (cfg.layer >= 0 && params.layer_idx != cfg.layer)
            {
                return false;
            }
            if (head != cfg.head)
            {
                return false;
            }

            const int target_row = (cfg.row < 0) ? (params.seq_len_q - 1) : cfg.row;
            if (q_row != target_row)
            {
                return false;
            }

            if (cfg.once)
            {
                static std::atomic<bool> dumped{false};
                if (dumped.exchange(true))
                {
                    return false;
                }
            }

            return true;
        }
    }

    // ============================================================================
    // Full Fused Kernel Validation
    // ============================================================================

    bool q16_validate_full_params(const Q16FusedAttentionWoResidualParams &params)
    {
        // Validate attention inputs
        if (params.Q == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: Q tensor is null");
            return false;
        }
        if (params.K == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: K tensor is null");
            return false;
        }
        if (params.V == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: V tensor is null");
            return false;
        }

        // Validate Wo weights
        if (params.Wo_packed == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: Wo_packed weights are null");
            return false;
        }

        // Validate residual
        if (params.residual_in == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: residual_in is null");
            return false;
        }
        if (params.residual_out == nullptr)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: residual_out is null");
            return false;
        }

        // Validate dimensions
        if (params.seq_len_q <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: seq_len_q must be > 0");
            return false;
        }
        if (params.kv_len <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: kv_len must be > 0");
            return false;
        }
        if (params.num_heads <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: num_heads must be > 0");
            return false;
        }
        if (params.num_kv_heads <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: num_kv_heads must be > 0");
            return false;
        }
        if (params.head_dim <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: head_dim must be > 0");
            return false;
        }
        if (params.d_model <= 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: d_model must be > 0");
            return false;
        }
        if (params.num_heads % params.num_kv_heads != 0)
        {
            LOG_ERROR("Q16FusedAttentionWoResidual: num_heads must be divisible by num_kv_heads");
            return false;
        }

        return true;
    }

    // ============================================================================
    // Flash Decode: FP32 Attention Core Helper (Single Query)
    // ============================================================================

    static void quantize_fp32_weights_to_exact_sum(
        const float *fp32_weights,
        int n,
        int32_t weight_sum_target,
        std::vector<int32_t> &w_int)
    {
        w_int.assign(static_cast<size_t>(std::max(0, n)), 0);
        if (fp32_weights == nullptr || n <= 0)
        {
            return;
        }

        // Bucket fractional parts into 256 bins so we can distribute the remainder
        // deterministically in O(n) without sorting.
        std::vector<uint8_t> frac_bin(static_cast<size_t>(n), 0);
        int64_t sum_w = 0;
        int bin_counts[256] = {0};

        for (int i = 0; i < n; ++i)
        {
            float scaled = fp32_weights[i] * static_cast<float>(weight_sum_target);
            if (scaled <= 0.0f)
            {
                w_int[static_cast<size_t>(i)] = 0;
                frac_bin[static_cast<size_t>(i)] = 0;
                continue;
            }

            const int32_t wi = static_cast<int32_t>(scaled); // floor
            w_int[static_cast<size_t>(i)] = wi;
            sum_w += static_cast<int64_t>(wi);

            float frac = scaled - static_cast<float>(wi);
            if (frac < 0.0f)
                frac = 0.0f;
            if (frac >= 1.0f)
                frac = 0.999999f;

            const int b = std::min(255, std::max(0, static_cast<int>(frac * 256.0f)));
            frac_bin[static_cast<size_t>(i)] = static_cast<uint8_t>(b);
            bin_counts[b]++;
        }

        int64_t diff = static_cast<int64_t>(weight_sum_target) - sum_w;

        // Add leftover units to the largest fractional bins.
        if (diff > 0)
        {
            for (int b = 255; b >= 0 && diff > 0; --b)
            {
                if (bin_counts[b] == 0)
                    continue;
                for (int i = 0; i < n && diff > 0; ++i)
                {
                    if (frac_bin[static_cast<size_t>(i)] != static_cast<uint8_t>(b))
                        continue;
                    w_int[static_cast<size_t>(i)] += 1;
                    --diff;
                }
            }
        }
        // Remove excess units from the smallest fractional bins, keeping weights non-negative.
        else if (diff < 0)
        {
            for (int b = 0; b <= 255 && diff < 0; ++b)
            {
                for (int i = 0; i < n && diff < 0; ++i)
                {
                    if (frac_bin[static_cast<size_t>(i)] != static_cast<uint8_t>(b))
                        continue;
                    int32_t &wi = w_int[static_cast<size_t>(i)];
                    if (wi <= 0)
                        continue;
                    wi -= 1;
                    ++diff;
                }
            }
        }

        LLAMINAR_ASSERT(diff == 0, "quantize_fp32_weights_to_exact_sum: failed to reach exact sum");
    }

    /**
     * @brief Process attention for a single head in Flash Decode mode
     *
     * Uses FP32 scores and softmax for high precision (matching FP32 pipeline),
     * then accumulates weighted V into INT32 accumulators for downstream processing.
     *
     * Pipeline:
     * 1. Q×K^T → FP32 scores (proper Q16 block scale handling)
     * 2. FP32 Softmax → FP32 weights (standard exp-based)
     * 3. P×V → INT32 accumulators with FP32 scaling info
     *
     * @param params Full kernel parameters
     * @param head Query head index
     * @param int32_accum Output INT32 accumulator for this head [head_dim]
     * @param weight_sum Output sum of INT16 weights (scaled to 32767)
     * @param v_scale_product Output average V block scale
     */
    static void flash_decode_attention_head_fp32_scores(
        const Q16FusedAttentionWoResidualParams &params,
        int head,
        int32_t *int32_accum,
        int32_t *weight_sum,
        float *v_scales_out)
    {
        const int head_dim = params.head_dim;
        const int kv_len = params.kv_len;
        const int kv_head = params.get_kv_head(head);
        const int q_blocks_per_row = params.q_blocks_per_row();
        const int kv_blocks_per_row = params.kv_blocks_per_row();
        const float attention_scale = params.get_scale(); // 1/sqrt(head_dim)
        const bool causal = params.causal;
        const int position_offset = params.position_offset;

        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;
        constexpr int q_row = 0; // Decode: single query

        // Zero the accumulator
        std::memset(int32_accum, 0, head_dim * sizeof(int32_t));
        *weight_sum = 0;
        if (v_scales_out)
        {
            std::memset(v_scales_out, 0, head_dim * sizeof(float));
        }

        // Determine effective KV length (causal masking)
        int kv_end = kv_len;
        if (causal)
        {
            kv_end = std::min(kv_len, position_offset + 1);
        }

        if (kv_end <= 0)
        {
            return;
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 1: Q×K^T → FP32 scores (proper block scale handling)
        // This preserves full Q16 precision by computing:
        //   score = Σ(int_dot_block × q_scale × k_scale) × attention_scale
        // ════════════════════════════════════════════════════════════════════════

        std::vector<float> fp32_scores(kv_end);

        {
            microkernels::Q16DotProductParams dot_params{};
            dot_params.Q = params.Q;
            dot_params.K = params.K;
            dot_params.q_rows = 1;
            dot_params.k_rows = kv_end;
            dot_params.head_dim = head_dim;
            dot_params.q_head = head;
            dot_params.kv_head = kv_head;
            dot_params.q_blocks_per_row = q_blocks_per_row;
            dot_params.kv_blocks_per_row = kv_blocks_per_row;
            dot_params.attention_scale = attention_scale;

            microkernels::q16_dot_product_gemv(dot_params, q_row, 0, kv_end, fp32_scores.data());
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 2: FP32 Softmax → FP32 weights [0.0, 1.0]
        // Standard softmax: weights = exp(score - max) / sum(exp(score - max))
        // ════════════════════════════════════════════════════════════════════════

        // Find max for numerical stability
        float max_score = fp32_scores[0];
        for (int i = 1; i < kv_end; ++i)
        {
            max_score = std::max(max_score, fp32_scores[i]);
        }

        // Compute exp and sum
        std::vector<float> fp32_weights(kv_end);
        float exp_sum = 0.0f;
        for (int i = 0; i < kv_end; ++i)
        {
            fp32_weights[i] = std::exp(fp32_scores[i] - max_score);
            exp_sum += fp32_weights[i];
        }

        // Normalize weights
        float inv_sum = 1.0f / exp_sum;
        for (int i = 0; i < kv_end; ++i)
        {
            fp32_weights[i] *= inv_sum;
        }

        // Track weight sum in the same integer domain used for accumulation.
        // This must reflect the actual integer weights used in P×V.
        // Keep weights in INT16 range to avoid INT32 accumulator overflow in P×V.
        // The integer-domain design (Exp2FixedSoftmax) targets [0, 32767].
        constexpr int32_t kWeightMax = 32767;
        int64_t weight_sum_i64 = 0;

        // ════════════════════════════════════════════════════════════════════════
        // STEP 3: P×V → INT32 accumulators with proper scale tracking
        // FP32 weights × Q16 V values, accumulated in INT32 domain
        //
        // For each dimension d, we compute:
        //   context[d] = Σ(w[kv] × v[kv,d]) = Σ(w[kv] × v_int[kv,d] × v_scale[kv,d])
        //
        // We use INT32 accumulators scaled by kWeightMax to maintain precision:
        //   int32_accum[d] = Σ((w[kv]*kWeightMax) × v_int[kv,d])
        //   actual_value = int32_accum[d] / kWeightMax * avg_v_scale
        // ════════════════════════════════════════════════════════════════════════

        // Track attention-weighted V scales.
        // Q16_1 uses per-block scales, but reconstructing a single scale per block is
        // still noticeably lossy for some layers; prefer per-element effective scales
        // when numerically stable, and fall back to per-block scales otherwise.
        const int blocks_per_head = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<float> weighted_v_scale_sum(blocks_per_head, 0.0f);
        std::vector<double> weighted_v_d_qs_sum(head_dim, 0.0);

        const int v_start_col = kv_head * head_dim;
        const int v_start_block = v_start_col / BLOCK_SIZE;

        // Quantize FP32 weights into integer weights summing to kWeightMax.
        // This reduces rounding error (especially when attention is very peaky)
        // while keeping PV accumulation in INT32.
        std::vector<int32_t> w_int;
        quantize_fp32_weights_to_exact_sum(fp32_weights.data(), kv_end, kWeightMax, w_int);

        for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
        {
            const int32_t w_scaled = w_int[static_cast<size_t>(kv_pos)];
            if (w_scaled <= 0)
                continue;

            weight_sum_i64 += static_cast<int64_t>(w_scaled);

            // Update per-block V scales once per KV position.
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const Q16_1Block &v_block_for_scale = params.V[kv_pos * kv_blocks_per_row + (v_start_block + b)];
                weighted_v_scale_sum[b] += static_cast<float>(w_scaled) * v_block_for_scale.d;
            }

            // Accumulate weighted V values
            for (int d = 0; d < head_dim; ++d)
            {
                int v_col = v_start_col + d;
                int v_block_idx = v_col / BLOCK_SIZE;
                int v_elem_idx = v_col % BLOCK_SIZE;

                const Q16_1Block &v_block = params.V[kv_pos * kv_blocks_per_row + v_block_idx];
                int16_t v_val = v_block.qs[v_elem_idx];

                // INT32 × INT16 → INT32
                int32_accum[d] += w_scaled * static_cast<int32_t>(v_val);

                // Track Σ(w_scaled * v_block.d * v_val) for per-element effective scaling.
                weighted_v_d_qs_sum[d] += static_cast<double>(w_scaled) *
                                          static_cast<double>(v_block.d) *
                                          static_cast<double>(v_val);
            }
        }

        // Finalize weight sum and per-element V scales.
        if (weight_sum_i64 <= 0)
        {
            *weight_sum = 1;
            if (v_scales_out)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    v_scales_out[d] = 1.0f;
                }
            }
            return;
        }

        *weight_sum = static_cast<int32_t>(std::min<int64_t>(weight_sum_i64, static_cast<int64_t>(INT32_MAX)));
        const float inv_weight_sum = 1.0f / static_cast<float>(weight_sum_i64);

        if (v_scales_out)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const float v_scale_eff = weighted_v_scale_sum[b] * inv_weight_sum;
                const int start = b * BLOCK_SIZE;
                const int end = std::min(start + BLOCK_SIZE, head_dim);
                for (int d = start; d < end; ++d)
                {
                    const int32_t denom = int32_accum[d];
                    if (denom != 0 && std::abs(denom) >= 256)
                    {
                        v_scales_out[d] = static_cast<float>(weighted_v_d_qs_sum[d] /
                                                             static_cast<double>(denom));
                    }
                    else
                    {
                        // Fallback: stable per-block scale estimate.
                        v_scales_out[d] = v_scale_eff;
                    }
                }
            }
        }

        if (should_dump_q16_attention(params, head, q_row))
        {
            const auto &cfg = debugEnv().q16_attention_dump;
            const int k = std::max(1, cfg.topk);
            const auto top_scores = compute_topk(fp32_scores.data(), kv_end, k);
            const auto top_weights = compute_topk(fp32_weights.data(), kv_end, k);

            int32_t w_sum_check = 0;
            int32_t w_min = w_int[0];
            int32_t w_max = w_int[0];
            int zeros = 0;
            for (int i = 0; i < kv_end; ++i)
            {
                const int32_t wi = w_int[static_cast<size_t>(i)];
                w_sum_check += wi;
                w_min = std::min(w_min, wi);
                w_max = std::max(w_max, wi);
                zeros += (wi == 0) ? 1 : 0;
            }
            const auto top_w_int = compute_topk_i32(w_int.data(), kv_end, k);

            int32_t acc_min = int32_accum[0];
            int32_t acc_max = int32_accum[0];
            int small_denom = 0;
            for (int d = 0; d < head_dim; ++d)
            {
                acc_min = std::min(acc_min, int32_accum[d]);
                acc_max = std::max(acc_max, int32_accum[d]);
                small_denom += (std::abs(int32_accum[d]) < 256) ? 1 : 0;
            }

            float ctx_min = 0.0f;
            float ctx_max = 0.0f;
            float ctx_mean = 0.0f;
            if (head_dim > 0)
            {
                const float inv_wsum_f = 1.0f / static_cast<float>(std::max<int64_t>(1, weight_sum_i64));
                ctx_min = static_cast<float>(int32_accum[0]) * inv_wsum_f * (v_scales_out ? v_scales_out[0] : 1.0f);
                ctx_max = ctx_min;
                for (int d = 0; d < head_dim; ++d)
                {
                    const float v_scale = v_scales_out ? v_scales_out[d] : 1.0f;
                    const float ctx = static_cast<float>(int32_accum[d]) * inv_wsum_f * v_scale;
                    ctx_min = std::min(ctx_min, ctx);
                    ctx_max = std::max(ctx_max, ctx);
                    ctx_mean += ctx;
                }
                ctx_mean /= static_cast<float>(head_dim);
            }

            LOG_DEBUG("[Q16AttnDump] layer=" << params.layer_idx
                                             << " mode=decode"
                                             << " head=" << head
                                             << " q_row=" << q_row
                                             << " kv_end=" << kv_end
                                             << " scale=" << attention_scale
                                             << " weight_target=" << kWeightMax
                                             << " weight_sum=" << weight_sum_i64
                                             << " w_sum_check=" << w_sum_check
                                             << " w[min,max]=[" << w_min << "," << w_max << "]"
                                             << " w_zeros=" << zeros
                                             << " acc[min,max]=[" << acc_min << "," << acc_max << "]"
                                             << " small_denom(<256)=" << small_denom << "/" << head_dim
                                             << " ctx[min,max,mean]=[" << ctx_min << "," << ctx_max << "," << ctx_mean << "]");

            // Additional internal consistency checks to isolate divergence source.
            {
                std::vector<float> slow_scores(static_cast<size_t>(kv_end), 0.0f);
                compute_slow_scores_qk(params, q_row, head, kv_end, slow_scores.data());

                float max_abs_score_err = 0.0f;
                int worst_score_idx = 0;
                double mse = 0.0;
                for (int i = 0; i < kv_end; ++i)
                {
                    const float diff = std::abs(fp32_scores[i] - slow_scores[static_cast<size_t>(i)]);
                    if (diff > max_abs_score_err)
                    {
                        max_abs_score_err = diff;
                        worst_score_idx = i;
                    }
                    mse += static_cast<double>(diff) * static_cast<double>(diff);
                }
                const float rmse = (kv_end > 0) ? static_cast<float>(std::sqrt(mse / static_cast<double>(kv_end))) : 0.0f;

                // Softmax comparison: do the slow softmax and compare weights.
                std::vector<float> slow_weights(static_cast<size_t>(kv_end), 0.0f);
                {
                    float max_s = slow_scores[0];
                    for (int i = 1; i < kv_end; ++i)
                        max_s = std::max(max_s, slow_scores[static_cast<size_t>(i)]);
                    float sum = 0.0f;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        slow_weights[static_cast<size_t>(i)] = std::exp(slow_scores[static_cast<size_t>(i)] - max_s);
                        sum += slow_weights[static_cast<size_t>(i)];
                    }
                    const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                    for (int i = 0; i < kv_end; ++i)
                        slow_weights[static_cast<size_t>(i)] *= inv;
                }

                float max_abs_w_err = 0.0f;
                int worst_w_idx = 0;
                double mse_w = 0.0;
                for (int i = 0; i < kv_end; ++i)
                {
                    const float diff = std::abs(fp32_weights[static_cast<size_t>(i)] - slow_weights[static_cast<size_t>(i)]);
                    if (diff > max_abs_w_err)
                    {
                        max_abs_w_err = diff;
                        worst_w_idx = i;
                    }
                    mse_w += static_cast<double>(diff) * static_cast<double>(diff);
                }
                const float rmse_w = (kv_end > 0) ? static_cast<float>(std::sqrt(mse_w / static_cast<double>(kv_end))) : 0.0f;

                const auto top_slow_scores = compute_topk(slow_scores.data(), kv_end, std::min(1, kv_end));
                const auto top_fast_scores = compute_topk(fp32_scores.data(), kv_end, std::min(1, kv_end));

                // Compare context reconstructions:
                // - ctx_fp32: FP32 softmax weights * dequant(V)
                // - ctx_fp32_wint: (w_int/kWeightMax) * dequant(V)
                // - ctx_fast: current integer path reconstruction (int32_accum + v_scales_out)
                std::vector<float> ctx_fp32(static_cast<size_t>(head_dim), 0.0f);
                std::vector<float> ctx_fp32_wint(static_cast<size_t>(head_dim), 0.0f);
                std::vector<float> ctx_fast(static_cast<size_t>(head_dim), 0.0f);

                compute_context_from_fp32_weights(params, kv_end, kv_head, fp32_weights.data(), ctx_fp32.data());

                std::vector<float> w_int_as_fp32(static_cast<size_t>(kv_end), 0.0f);
                const float inv_weight_max = 1.0f / static_cast<float>(kWeightMax);
                for (int i = 0; i < kv_end; ++i)
                {
                    w_int_as_fp32[static_cast<size_t>(i)] = static_cast<float>(w_int[static_cast<size_t>(i)]) * inv_weight_max;
                }
                compute_context_from_fp32_weights(params, kv_end, kv_head, w_int_as_fp32.data(), ctx_fp32_wint.data());

                const float inv_wsum_f = 1.0f / static_cast<float>(std::max<int64_t>(1, weight_sum_i64));
                for (int d = 0; d < head_dim; ++d)
                {
                    const float v_scale = v_scales_out ? v_scales_out[d] : 1.0f;
                    ctx_fast[static_cast<size_t>(d)] = static_cast<float>(int32_accum[d]) * inv_wsum_f * v_scale;
                }

                // V saturation stats (helps identify quantization/clipping driven divergence).
                const int kv_blocks_per_row_local = kv_blocks_per_row;
                constexpr int BLOCK_SIZE_LOCAL = Q16_1Block::BLOCK_SIZE;
                const int v_start_col = kv_head * head_dim;
                int sat_count = 0;
                int total_count = 0;
                for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
                {
                    const Q16_1Block *v_row_ptr = params.V + kv_pos * kv_blocks_per_row_local;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        const int v_col = v_start_col + d;
                        const int v_block_idx = v_col / BLOCK_SIZE_LOCAL;
                        const int v_elem_idx = v_col % BLOCK_SIZE_LOCAL;
                        const int16_t vv = v_row_ptr[v_block_idx].qs[v_elem_idx];
                        sat_count += (vv == std::numeric_limits<int16_t>::min() || vv == std::numeric_limits<int16_t>::max()) ? 1 : 0;
                        total_count += 1;
                    }
                }

                LOG_DEBUG("[Q16AttnDump] score_check max_abs=" << max_abs_score_err << " rmse=" << rmse
                                                               << " w_check max_abs=" << max_abs_w_err << " rmse=" << rmse_w
                                                               << " top_fast=" << (top_fast_scores.empty() ? -1 : top_fast_scores[0].idx)
                                                               << ":" << (top_fast_scores.empty() ? 0.0f : top_fast_scores[0].value)
                                                               << " top_slow=" << (top_slow_scores.empty() ? -1 : top_slow_scores[0].idx)
                                                               << ":" << (top_slow_scores.empty() ? 0.0f : top_slow_scores[0].value)
                                                               << " ctx_cos(fast,fp32_wint)=" << cosine_similarity(ctx_fast.data(), ctx_fp32_wint.data(), head_dim)
                                                               << " ctx_cos(fp32_wint,fp32)=" << cosine_similarity(ctx_fp32_wint.data(), ctx_fp32.data(), head_dim)
                                                               << " v_sat=" << sat_count << "/" << total_count);

                if (kv_end > 0)
                {
                    LOG_DEBUG("[Q16AttnDump] worst_score idx=" << worst_score_idx
                                                             << " fast=" << fp32_scores[worst_score_idx]
                                                             << " slow=" << slow_scores[static_cast<size_t>(worst_score_idx)]
                                                             << " w_int=" << w_int[static_cast<size_t>(worst_score_idx)]
                                                             << " fast_w=" << fp32_weights[static_cast<size_t>(worst_score_idx)]
                                                             << " slow_w=" << slow_weights[static_cast<size_t>(worst_score_idx)]);

                    LOG_DEBUG("[Q16AttnDump] worst_weight idx=" << worst_w_idx
                                                               << " fast=" << fp32_scores[worst_w_idx]
                                                               << " slow=" << slow_scores[static_cast<size_t>(worst_w_idx)]
                                                               << " w_int=" << w_int[static_cast<size_t>(worst_w_idx)]
                                                               << " fast_w=" << fp32_weights[static_cast<size_t>(worst_w_idx)]
                                                               << " slow_w=" << slow_weights[static_cast<size_t>(worst_w_idx)]);
                }
            }

            {
                std::ostringstream msg;
                msg << "[Q16AttnDump] top_scores:";
                for (const auto &e : top_scores)
                    msg << " (" << e.idx << ":" << e.value << ")";
                LOG_DEBUG(msg.str());
            }
            {
                std::ostringstream msg;
                msg << "[Q16AttnDump] top_weights(fp32):";
                for (const auto &e : top_weights)
                    msg << " (" << e.idx << ":" << e.value << ")";
                LOG_DEBUG(msg.str());
            }
            {
                std::ostringstream msg;
                msg << "[Q16AttnDump] top_weights(int):";
                for (const auto &e : top_w_int)
                    msg << " (" << e.idx << ":" << static_cast<int32_t>(e.value) << ")";
                LOG_DEBUG(msg.str());
            }

            // When KV is small, print full distributions to spot truncated/zeroed weights.
            if (kv_end <= 16)
            {
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] all_scores:";
                    for (int i = 0; i < kv_end; ++i)
                        msg << " (" << i << ":" << fp32_scores[i] << ")";
                    LOG_DEBUG(msg.str());
                }
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] all_weights(fp32):";
                    double sum = 0.0;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        const float w = fp32_weights[static_cast<size_t>(i)];
                        sum += static_cast<double>(w);
                        msg << " (" << i << ":" << w << ")";
                    }
                    msg << " sum=" << sum;
                    LOG_DEBUG(msg.str());
                }
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] all_weights(int):";
                    int32_t sum = 0;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        const int32_t wi = w_int[static_cast<size_t>(i)];
                        sum += wi;
                        msg << " (" << i << ":" << wi << ")";
                    }
                    msg << " sum=" << sum << " target=" << kWeightMax;
                    LOG_DEBUG(msg.str());
                }
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] zeroed_nontrivial(fp32>1e-4):";
                    int count = 0;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        const float w = fp32_weights[static_cast<size_t>(i)];
                        const int32_t wi = w_int[static_cast<size_t>(i)];
                        if (wi == 0 && w > 1.0e-4f)
                        {
                            msg << " (" << i << ":" << w << ")";
                            ++count;
                        }
                    }
                    if (count == 0)
                        msg << " <none>";
                    LOG_DEBUG(msg.str());
                }
            }
        }
    }

    // ============================================================================
    // FA2 Prefill: FP32 Scores Attention Core Helper (One Query Row)
    // ============================================================================

    /**
     * @brief Process attention for a single query row in FA2 Prefill mode (FP32 scores)
     */
    static void fa2_prefill_attention_row_fp32_scores(
        const Q16FusedAttentionWoResidualParams &params,
        int q_row,
        int head,
        int32_t *int32_accum,
        int32_t *weight_sum,
        float *v_scales_out)
    {
        const int head_dim = params.head_dim;
        const int kv_len = params.kv_len;
        const int kv_head = params.get_kv_head(head);
        const int q_blocks_per_row = params.q_blocks_per_row();
        const int kv_blocks_per_row = params.kv_blocks_per_row();
        const float attention_scale = params.get_scale();
        const bool causal = params.causal;

        constexpr int BLOCK_SIZE = Q16_1Block::BLOCK_SIZE;

        // Zero the accumulator
        std::memset(int32_accum, 0, head_dim * sizeof(int32_t));
        *weight_sum = 0;
        if (v_scales_out)
        {
            std::memset(v_scales_out, 0, head_dim * sizeof(float));
        }

        // Determine effective KV length (causal masking)
        int kv_end = kv_len;
        if (causal)
        {
            kv_end = std::min(kv_len, q_row + 1);
        }

        if (kv_end <= 0)
        {
            return;
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 1: Q×K^T → FP32 scores (proper block scale handling)
        // ════════════════════════════════════════════════════════════════════════

        std::vector<float> fp32_scores(kv_end);

        {
            microkernels::Q16DotProductParams dot_params{};
            dot_params.Q = params.Q;
            dot_params.K = params.K;
            dot_params.q_rows = params.seq_len_q;
            dot_params.k_rows = kv_end;
            dot_params.head_dim = head_dim;
            dot_params.q_head = head;
            dot_params.kv_head = kv_head;
            dot_params.q_blocks_per_row = q_blocks_per_row;
            dot_params.kv_blocks_per_row = kv_blocks_per_row;
            dot_params.attention_scale = attention_scale;

            microkernels::q16_dot_product_gemv(dot_params, q_row, 0, kv_end, fp32_scores.data());
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 2: FP32 Softmax → FP32 weights [0.0, 1.0]
        // ════════════════════════════════════════════════════════════════════════

        // Find max for numerical stability
        float max_score = fp32_scores[0];
        for (int i = 1; i < kv_end; ++i)
        {
            max_score = std::max(max_score, fp32_scores[i]);
        }

        // Compute exp and sum
        std::vector<float> fp32_weights(kv_end);
        float exp_sum = 0.0f;
        for (int i = 0; i < kv_end; ++i)
        {
            fp32_weights[i] = std::exp(fp32_scores[i] - max_score);
            exp_sum += fp32_weights[i];
        }

        // Normalize weights
        float inv_sum = 1.0f / exp_sum;
        for (int i = 0; i < kv_end; ++i)
        {
            fp32_weights[i] *= inv_sum;
        }

        // Track weight sum in the same integer domain used for accumulation.
        // Keep weights in INT16 range to avoid INT32 accumulator overflow in P×V.
        // The integer-domain design (Exp2FixedSoftmax) targets [0, 32767].
        constexpr int32_t kWeightMax = 32767;
        int64_t weight_sum_i64 = 0;

        // ════════════════════════════════════════════════════════════════════════
        // STEP 3: P×V → INT32 accumulators
        // ════════════════════════════════════════════════════════════════════════

        // Track effective V scale per element.
        // This can become unstable when the signed denominator Σ(w_scaled * v_qs) is near zero
        // (cancellation between positive/negative int16 values). In that case, fall back to a
        // stable per-block scale estimate.
        const int blocks_per_head = (head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<float> weighted_v_scale_sum(blocks_per_head, 0.0f);
        std::vector<double> weighted_v_d_qs_sum(head_dim, 0.0);

        const int v_start_col = kv_head * head_dim;
        const int v_start_block = v_start_col / BLOCK_SIZE;

        // Quantize FP32 weights into integer weights summing to kWeightMax.
        std::vector<int32_t> w_int;
        quantize_fp32_weights_to_exact_sum(fp32_weights.data(), kv_end, kWeightMax, w_int);

        for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
        {
            const int32_t w_scaled = w_int[static_cast<size_t>(kv_pos)];
            if (w_scaled <= 0)
                continue;

            weight_sum_i64 += static_cast<int64_t>(w_scaled);

            // Update per-block V scales once per KV position.
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const Q16_1Block &v_block_for_scale = params.V[kv_pos * kv_blocks_per_row + (v_start_block + b)];
                weighted_v_scale_sum[b] += static_cast<float>(w_scaled) * v_block_for_scale.d;
            }

            for (int d = 0; d < head_dim; ++d)
            {
                int v_col = v_start_col + d;
                int v_block_idx = v_col / BLOCK_SIZE;
                int v_elem_idx = v_col % BLOCK_SIZE;

                const Q16_1Block &v_block = params.V[kv_pos * kv_blocks_per_row + v_block_idx];
                int16_t v_val = v_block.qs[v_elem_idx];

                // INT32 × INT16 → INT32
                int32_accum[d] += w_scaled * static_cast<int32_t>(v_val);

                // Track Σ(w_scaled * v_block.d * v_val) for per-element effective scaling.
                weighted_v_d_qs_sum[d] += static_cast<double>(w_scaled) *
                                          static_cast<double>(v_block.d) *
                                          static_cast<double>(v_val);
            }
        }

        if (weight_sum_i64 <= 0)
        {
            *weight_sum = 1;
            if (v_scales_out)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    v_scales_out[d] = 1.0f;
                }
            }
            return;
        }

        *weight_sum = static_cast<int32_t>(std::min<int64_t>(weight_sum_i64, static_cast<int64_t>(INT32_MAX)));
        const float inv_weight_sum = 1.0f / static_cast<float>(weight_sum_i64);
        if (v_scales_out)
        {
            for (int b = 0; b < blocks_per_head; ++b)
            {
                const float v_scale_eff = weighted_v_scale_sum[b] * inv_weight_sum;
                const int start = b * BLOCK_SIZE;
                const int end = std::min(start + BLOCK_SIZE, head_dim);
                for (int d = start; d < end; ++d)
                {
                    const int32_t denom = int32_accum[d];
                    if (denom != 0 && std::abs(denom) >= 256)
                    {
                        v_scales_out[d] = static_cast<float>(weighted_v_d_qs_sum[d] /
                                                             static_cast<double>(denom));
                    }
                    else
                    {
                        v_scales_out[d] = v_scale_eff;
                    }
                }
            }
        }

        if (should_dump_q16_attention(params, head, q_row))
        {
            const auto &cfg = debugEnv().q16_attention_dump;
            const int k = std::max(1, cfg.topk);
            const auto top_scores = compute_topk(fp32_scores.data(), kv_end, k);
            const auto top_weights = compute_topk(fp32_weights.data(), kv_end, k);

            int32_t w_sum_check = 0;
            int32_t w_min = w_int[0];
            int32_t w_max = w_int[0];
            int zeros = 0;
            for (int i = 0; i < kv_end; ++i)
            {
                const int32_t wi = w_int[static_cast<size_t>(i)];
                w_sum_check += wi;
                w_min = std::min(w_min, wi);
                w_max = std::max(w_max, wi);
                zeros += (wi == 0) ? 1 : 0;
            }
            const auto top_w_int = compute_topk_i32(w_int.data(), kv_end, k);

            int32_t acc_min = int32_accum[0];
            int32_t acc_max = int32_accum[0];
            int small_denom = 0;
            for (int d = 0; d < head_dim; ++d)
            {
                acc_min = std::min(acc_min, int32_accum[d]);
                acc_max = std::max(acc_max, int32_accum[d]);
                small_denom += (std::abs(int32_accum[d]) < 256) ? 1 : 0;
            }

            float ctx_min = 0.0f;
            float ctx_max = 0.0f;
            float ctx_mean = 0.0f;
            if (head_dim > 0)
            {
                const float inv_wsum_f = 1.0f / static_cast<float>(std::max<int64_t>(1, weight_sum_i64));
                ctx_min = static_cast<float>(int32_accum[0]) * inv_wsum_f * (v_scales_out ? v_scales_out[0] : 1.0f);
                ctx_max = ctx_min;
                for (int d = 0; d < head_dim; ++d)
                {
                    const float v_scale = v_scales_out ? v_scales_out[d] : 1.0f;
                    const float ctx = static_cast<float>(int32_accum[d]) * inv_wsum_f * v_scale;
                    ctx_min = std::min(ctx_min, ctx);
                    ctx_max = std::max(ctx_max, ctx);
                    ctx_mean += ctx;
                }
                ctx_mean /= static_cast<float>(head_dim);
            }

            LOG_DEBUG("[Q16AttnDump] layer=" << params.layer_idx
                                             << " mode=prefill"
                                             << " head=" << head
                                             << " q_row=" << q_row
                                             << " kv_end=" << kv_end
                                             << " scale=" << attention_scale
                                             << " weight_target=" << kWeightMax
                                             << " weight_sum=" << weight_sum_i64
                                             << " w_sum_check=" << w_sum_check
                                             << " w[min,max]=[" << w_min << "," << w_max << "]"
                                             << " w_zeros=" << zeros
                                             << " acc[min,max]=[" << acc_min << "," << acc_max << "]"
                                             << " small_denom(<256)=" << small_denom << "/" << head_dim
                                             << " ctx[min,max,mean]=[" << ctx_min << "," << ctx_max << "," << ctx_mean << "]");

            // Additional internal consistency checks to isolate divergence source.
            {
                std::vector<float> slow_scores(static_cast<size_t>(kv_end), 0.0f);
                compute_slow_scores_qk(params, q_row, head, kv_end, slow_scores.data());

                float max_abs_score_err = 0.0f;
                int worst_score_idx = 0;
                double mse = 0.0;
                for (int i = 0; i < kv_end; ++i)
                {
                    const float diff = std::abs(fp32_scores[i] - slow_scores[static_cast<size_t>(i)]);
                    if (diff > max_abs_score_err)
                    {
                        max_abs_score_err = diff;
                        worst_score_idx = i;
                    }
                    mse += static_cast<double>(diff) * static_cast<double>(diff);
                }
                const float rmse = (kv_end > 0) ? static_cast<float>(std::sqrt(mse / static_cast<double>(kv_end))) : 0.0f;

                // Softmax comparison: do the slow softmax and compare weights.
                std::vector<float> slow_weights(static_cast<size_t>(kv_end), 0.0f);
                if (kv_end > 0)
                {
                    float max_s = slow_scores[0];
                    for (int i = 1; i < kv_end; ++i)
                        max_s = std::max(max_s, slow_scores[static_cast<size_t>(i)]);
                    float sum = 0.0f;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        slow_weights[static_cast<size_t>(i)] = std::exp(slow_scores[static_cast<size_t>(i)] - max_s);
                        sum += slow_weights[static_cast<size_t>(i)];
                    }
                    const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
                    for (int i = 0; i < kv_end; ++i)
                        slow_weights[static_cast<size_t>(i)] *= inv;
                }

                float max_abs_w_err = 0.0f;
                int worst_w_idx = 0;
                double mse_w = 0.0;
                for (int i = 0; i < kv_end; ++i)
                {
                    const float diff = std::abs(fp32_weights[static_cast<size_t>(i)] - slow_weights[static_cast<size_t>(i)]);
                    if (diff > max_abs_w_err)
                    {
                        max_abs_w_err = diff;
                        worst_w_idx = i;
                    }
                    mse_w += static_cast<double>(diff) * static_cast<double>(diff);
                }
                const float rmse_w = (kv_end > 0) ? static_cast<float>(std::sqrt(mse_w / static_cast<double>(kv_end))) : 0.0f;

                const auto top_slow_scores = compute_topk(slow_scores.data(), kv_end, std::min(1, kv_end));
                const auto top_fast_scores = compute_topk(fp32_scores.data(), kv_end, std::min(1, kv_end));

                std::vector<float> ctx_fp32(static_cast<size_t>(head_dim), 0.0f);
                std::vector<float> ctx_fp32_wint(static_cast<size_t>(head_dim), 0.0f);
                std::vector<float> ctx_fast(static_cast<size_t>(head_dim), 0.0f);

                compute_context_from_fp32_weights(params, kv_end, kv_head, fp32_weights.data(), ctx_fp32.data());

                std::vector<float> w_int_as_fp32(static_cast<size_t>(kv_end), 0.0f);
                const float inv_weight_max = 1.0f / static_cast<float>(kWeightMax);
                for (int i = 0; i < kv_end; ++i)
                {
                    w_int_as_fp32[static_cast<size_t>(i)] = static_cast<float>(w_int[static_cast<size_t>(i)]) * inv_weight_max;
                }
                compute_context_from_fp32_weights(params, kv_end, kv_head, w_int_as_fp32.data(), ctx_fp32_wint.data());

                const float inv_wsum_f = 1.0f / static_cast<float>(std::max<int64_t>(1, weight_sum_i64));
                for (int d = 0; d < head_dim; ++d)
                {
                    const float v_scale = v_scales_out ? v_scales_out[d] : 1.0f;
                    ctx_fast[static_cast<size_t>(d)] = static_cast<float>(int32_accum[d]) * inv_wsum_f * v_scale;
                }

                const int kv_blocks_per_row_local = kv_blocks_per_row;
                constexpr int BLOCK_SIZE_LOCAL = Q16_1Block::BLOCK_SIZE;
                const int v_start_col = kv_head * head_dim;
                int sat_count = 0;
                int total_count = 0;
                for (int kv_pos = 0; kv_pos < kv_end; ++kv_pos)
                {
                    const Q16_1Block *v_row_ptr = params.V + kv_pos * kv_blocks_per_row_local;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        const int v_col = v_start_col + d;
                        const int v_block_idx = v_col / BLOCK_SIZE_LOCAL;
                        const int v_elem_idx = v_col % BLOCK_SIZE_LOCAL;
                        const int16_t vv = v_row_ptr[v_block_idx].qs[v_elem_idx];
                        sat_count += (vv == std::numeric_limits<int16_t>::min() || vv == std::numeric_limits<int16_t>::max()) ? 1 : 0;
                        total_count += 1;
                    }
                }

                LOG_DEBUG("[Q16AttnDump] score_check max_abs=" << max_abs_score_err << " rmse=" << rmse
                                                               << " w_check max_abs=" << max_abs_w_err << " rmse=" << rmse_w
                                                               << " top_fast=" << (top_fast_scores.empty() ? -1 : top_fast_scores[0].idx)
                                                               << ":" << (top_fast_scores.empty() ? 0.0f : top_fast_scores[0].value)
                                                               << " top_slow=" << (top_slow_scores.empty() ? -1 : top_slow_scores[0].idx)
                                                               << ":" << (top_slow_scores.empty() ? 0.0f : top_slow_scores[0].value)
                                                               << " ctx_cos(fast,fp32_wint)=" << cosine_similarity(ctx_fast.data(), ctx_fp32_wint.data(), head_dim)
                                                               << " ctx_cos(fp32_wint,fp32)=" << cosine_similarity(ctx_fp32_wint.data(), ctx_fp32.data(), head_dim)
                                                               << " v_sat=" << sat_count << "/" << total_count);

                if (kv_end > 0)
                {
                    LOG_DEBUG("[Q16AttnDump] worst_score idx=" << worst_score_idx
                                                             << " fast=" << fp32_scores[worst_score_idx]
                                                             << " slow=" << slow_scores[static_cast<size_t>(worst_score_idx)]
                                                             << " w_int=" << w_int[static_cast<size_t>(worst_score_idx)]
                                                             << " fast_w=" << fp32_weights[static_cast<size_t>(worst_score_idx)]
                                                             << " slow_w=" << slow_weights[static_cast<size_t>(worst_score_idx)]);

                    LOG_DEBUG("[Q16AttnDump] worst_weight idx=" << worst_w_idx
                                                               << " fast=" << fp32_scores[worst_w_idx]
                                                               << " slow=" << slow_scores[static_cast<size_t>(worst_w_idx)]
                                                               << " w_int=" << w_int[static_cast<size_t>(worst_w_idx)]
                                                               << " fast_w=" << fp32_weights[static_cast<size_t>(worst_w_idx)]
                                                               << " slow_w=" << slow_weights[static_cast<size_t>(worst_w_idx)]);
                }
            }

            {
                std::ostringstream msg;
                msg << "[Q16AttnDump] top_scores:";
                for (const auto &e : top_scores)
                    msg << " (" << e.idx << ":" << e.value << ")";
                LOG_DEBUG(msg.str());
            }
            {
                std::ostringstream msg;
                msg << "[Q16AttnDump] top_weights(fp32):";
                for (const auto &e : top_weights)
                    msg << " (" << e.idx << ":" << e.value << ")";
                LOG_DEBUG(msg.str());
            }
            {
                std::ostringstream msg;
                msg << "[Q16AttnDump] top_weights(int):";
                for (const auto &e : top_w_int)
                    msg << " (" << e.idx << ":" << static_cast<int32_t>(e.value) << ")";
                LOG_DEBUG(msg.str());
            }

            // When KV is small, print full distributions to spot truncated/zeroed weights.
            if (kv_end <= 16)
            {
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] all_scores:";
                    for (int i = 0; i < kv_end; ++i)
                        msg << " (" << i << ":" << fp32_scores[i] << ")";
                    LOG_DEBUG(msg.str());
                }
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] all_weights(fp32):";
                    double sum = 0.0;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        const float w = fp32_weights[static_cast<size_t>(i)];
                        sum += static_cast<double>(w);
                        msg << " (" << i << ":" << w << ")";
                    }
                    msg << " sum=" << sum;
                    LOG_DEBUG(msg.str());
                }
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] all_weights(int):";
                    int32_t sum = 0;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        const int32_t wi = w_int[static_cast<size_t>(i)];
                        sum += wi;
                        msg << " (" << i << ":" << wi << ")";
                    }
                    msg << " sum=" << sum << " target=" << kWeightMax;
                    LOG_DEBUG(msg.str());
                }
                {
                    std::ostringstream msg;
                    msg << "[Q16AttnDump] zeroed_nontrivial(fp32>1e-4):";
                    int count = 0;
                    for (int i = 0; i < kv_end; ++i)
                    {
                        const float w = fp32_weights[static_cast<size_t>(i)];
                        const int32_t wi = w_int[static_cast<size_t>(i)];
                        if (wi == 0 && w > 1.0e-4f)
                        {
                            msg << " (" << i << ":" << w << ")";
                            ++count;
                        }
                    }
                    if (count == 0)
                        msg << " <none>";
                    LOG_DEBUG(msg.str());
                }
            }
        }
    }

    // ============================================================================
    // Full Fused Kernel - Flash Decode Path (PURE INTEGER)
    // ============================================================================

    bool q16_fused_attention_wo_residual_decode(const Q16FusedAttentionWoResidualParams &params)
    {
        // Validate parameters
        if (!q16_validate_full_params(params))
        {
            return false;
        }

        if (!params.is_decode())
        {
            LOG_WARN("Q16FusedAttentionWoResidual: decode called with seq_len_q > 1, "
                     "redirecting to FA2 prefill");
            return q16_fused_attention_wo_residual_prefill(params);
        }

        LOG_DEBUG("Q16FusedAttentionWoResidual FLASH DECODE (PURE INTEGER): heads=" << params.num_heads
                                                                                    << "/" << params.num_kv_heads
                                                                                    << ", head_dim=" << params.head_dim
                                                                                    << ", d_model=" << params.d_model
                                                                                    << ", kv_len=" << params.kv_len);

        const int num_heads = params.num_heads;
        const int head_dim = params.head_dim;
        const int d_model = params.d_model;
        const int input_dim = num_heads * head_dim;
        const int output_blocks = (d_model + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;

        // ════════════════════════════════════════════════════════════════════════
        // STEPS 1-3: Attention core → INT32 context (all heads concatenated)
        // ════════════════════════════════════════════════════════════════════════

        std::vector<int32_t> int32_context(input_dim, 0);
        std::vector<int32_t> head_weight_sums(num_heads, 1);
        std::vector<float> context_v_scales(input_dim, 1.0f);

        for (int h = 0; h < num_heads; ++h)
        {
            int32_t *head_accum = int32_context.data() + h * head_dim;
            int32_t head_weight_sum = 0;
            float *head_v_scales = context_v_scales.data() + h * head_dim;

            flash_decode_attention_head_fp32_scores(
                params, h, head_accum, &head_weight_sum, head_v_scales);

            head_weight_sums[h] = (head_weight_sum > 0) ? head_weight_sum : 1;
        }

        // ════════════════════════════════════════════════════════════════════════
        // SNAPSHOT: Capture attention context (INT32 → FP32 for debugging)
        // ════════════════════════════════════════════════════════════════════════

#ifdef ENABLE_PIPELINE_SNAPSHOTS
        LOG_TRACE("[Q16FusedAttention DECODE] context_snapshot ptr=" << params.context_snapshot
                                                                     << " (snapshots " << (params.context_snapshot ? "ENABLED" : "DISABLED") << ")");
#endif

        if (params.context_snapshot)
        {
            // Convert INT32 context to FP32 for snapshot
            // The INT32 values need to be scaled by weight_sum and v_scale to get FP32 equivalents
            float *ctx_out = params.context_snapshot;
            for (int h = 0; h < num_heads; ++h)
            {
                const float inv_weight_sum = 1.0f / static_cast<float>(head_weight_sums[h]);
                for (int d = 0; d < head_dim; ++d)
                {
                    int idx = h * head_dim + d;
                    ctx_out[idx] = static_cast<float>(int32_context[idx]) * inv_weight_sum * context_v_scales[idx];
                }
            }

            LOG_DEBUG("[Q16FusedAttention DECODE] Captured context snapshot: "
                      << input_dim << " elements");
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 4: Wo Projection (VPDPWSSD) → Q16_1 output (ALL INTEGER)
        // ════════════════════════════════════════════════════════════════════════

        // Allocate Q16_1 projection output
        std::vector<Q16_1Block> projection_q16(output_blocks);

        // Create IntegerContext for Wo projection
        IntegerContext context;
        context.int32_data = int32_context.data();
        context.weight_sums = head_weight_sums.data();
        context.v_scales = context_v_scales.data();
        context.num_heads = num_heads;
        context.head_dim = head_dim;
        context.count = input_dim;

        // Create output structure
        Q16_1Projection projection_out;
        projection_out.blocks = projection_q16.data();
        projection_out.count = d_model;

        // Set up Wo projection params
        WoProjectionVNNIParams wo_params;
        wo_params.Wo_packed = params.Wo_packed;
        wo_params.input_dim = input_dim;
        wo_params.d_model = d_model;
        wo_params.bias = nullptr;

        // Execute VPDPWSSD projection → INT32 → requant → Q16_1 output
        wo_projection_vpdpwssd_to_q16_1_gemv(wo_params, context, projection_out);

        // ════════════════════════════════════════════════════════════════════════
        // SNAPSHOT: Capture attention output (Wo projection result, before residual)
        // ════════════════════════════════════════════════════════════════════════

        if (params.attention_output_snapshot)
        {
            // Dequantize Q16_1 projection output to FP32 for snapshot
            float *out_snapshot = params.attention_output_snapshot;
            for (int b = 0; b < output_blocks; ++b)
            {
                const Q16_1Block &block = projection_q16[b];
                const int start = b * Q16_1Block::BLOCK_SIZE;
                const int end = std::min(start + static_cast<int>(Q16_1Block::BLOCK_SIZE), d_model);
                for (int i = start; i < end; ++i)
                {
                    out_snapshot[i] = block.d * static_cast<float>(block.qs[i - start]);
                }
            }
            LOG_DEBUG("[Q16FusedAttention DECODE] Captured attention_output snapshot: " << d_model << " elements");
        }

        // ════════════════════════════════════════════════════════════════════════
        // STEP 5: Native Q16_1 Residual Add (ALL INTEGER: Q16_1 + Q16_1 → Q16_1)
        // ════════════════════════════════════════════════════════════════════════

        // Copy residual_in to residual_out if different pointers
        if (params.residual_in != params.residual_out)
        {
            std::memcpy(params.residual_out, params.residual_in,
                        output_blocks * sizeof(Q16_1Block));
        }

        // Q16_1 + Q16_1 → Q16_1 (in-place on residual_out) - NO FP32!
        simd::q16_1_add_q16_1(
            params.residual_out,   // Residual (accumulates result in-place)
            projection_out.blocks, // Projection to add
            params.residual_out,   // Output (same as residual for in-place)
            d_model);

        // ════════════════════════════════════════════════════════════════════════
        // SNAPSHOT: Capture attention residual (final output after residual add)
        // ════════════════════════════════════════════════════════════════════════

        if (params.attention_residual_snapshot)
        {
            // Dequantize Q16_1 residual output to FP32 for snapshot
            float *res_snapshot = params.attention_residual_snapshot;
            for (int b = 0; b < output_blocks; ++b)
            {
                const Q16_1Block &block = params.residual_out[b];
                const int start = b * Q16_1Block::BLOCK_SIZE;
                const int end = std::min(start + static_cast<int>(Q16_1Block::BLOCK_SIZE), d_model);
                for (int i = start; i < end; ++i)
                {
                    res_snapshot[i] = block.d * static_cast<float>(block.qs[i - start]);
                }
            }
            LOG_DEBUG("[Q16FusedAttention DECODE] Captured attention_residual snapshot: " << d_model << " elements");
        }

        return true;
    }

    // ============================================================================
    // Full Fused Kernel - FA2 Prefill Path (PURE INTEGER)
    // ============================================================================

    bool q16_fused_attention_wo_residual_prefill(const Q16FusedAttentionWoResidualParams &params)
    {
        // Validate parameters
        if (!q16_validate_full_params(params))
        {
            return false;
        }

        if (params.is_decode())
        {
            LOG_WARN("Q16FusedAttentionWoResidual: prefill called with seq_len_q == 1, "
                     "redirecting to Flash Decode");
            return q16_fused_attention_wo_residual_decode(params);
        }

        LOG_DEBUG("Q16FusedAttentionWoResidual FA2 PREFILL (PURE INTEGER): seq_len=" << params.seq_len_q
                                                                                     << ", heads=" << params.num_heads
                                                                                     << "/" << params.num_kv_heads
                                                                                     << ", head_dim=" << params.head_dim
                                                                                     << ", d_model=" << params.d_model
                                                                                     << ", kv_len=" << params.kv_len);

        const int seq_len_q = params.seq_len_q;
        const int num_heads = params.num_heads;
        const int head_dim = params.head_dim;
        const int d_model = params.d_model;
        const int input_dim = num_heads * head_dim;
        const int output_blocks = (d_model + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        const int residual_blocks_per_row = params.residual_blocks_per_row();

        // Process each query position
        for (int q = 0; q < seq_len_q; ++q)
        {
            // ════════════════════════════════════════════════════════════════════
            // STEPS 1-3: Attention core → INT32 context
            // ════════════════════════════════════════════════════════════════════

            std::vector<int32_t> int32_context(input_dim, 0);
            std::vector<int32_t> head_weight_sums(num_heads, 1);
            std::vector<float> context_v_scales(input_dim, 1.0f);

            for (int h = 0; h < num_heads; ++h)
            {
                int32_t *head_accum = int32_context.data() + h * head_dim;
                int32_t head_weight_sum = 0;
                float *head_v_scales = context_v_scales.data() + h * head_dim;

                fa2_prefill_attention_row_fp32_scores(
                    params, q, h, head_accum, &head_weight_sum, head_v_scales);

                head_weight_sums[h] = (head_weight_sum > 0) ? head_weight_sum : 1;
            }

            // ════════════════════════════════════════════════════════════════════
            // SNAPSHOT: Capture attention context for this query position
            // ════════════════════════════════════════════════════════════════════

#ifdef ENABLE_PIPELINE_SNAPSHOTS
            if (q == 0)
            {
                LOG_TRACE("[Q16FusedAttention PREFILL] context_snapshot ptr=" << params.context_snapshot
                                                                              << " (snapshots " << (params.context_snapshot ? "ENABLED" : "DISABLED") << ")");
            }
#endif

            if (params.context_snapshot)
            {
                // Convert INT32 context to FP32 for snapshot
                // Output layout: [seq_len_q × input_dim] where input_dim = num_heads × head_dim
                float *ctx_row = params.context_snapshot + q * input_dim;
                for (int h = 0; h < num_heads; ++h)
                {
                    const float inv_weight_sum = 1.0f / static_cast<float>(head_weight_sums[h]);
                    for (int d = 0; d < head_dim; ++d)
                    {
                        int idx = h * head_dim + d;
                        ctx_row[idx] = static_cast<float>(int32_context[idx]) * inv_weight_sum * context_v_scales[idx];
                    }
                }

                if (q == 0)
                {
                    LOG_DEBUG("[Q16FusedAttention PREFILL] Capturing context snapshot for "
                              << seq_len_q << " positions, input_dim=" << input_dim);
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // STEP 4: Wo Projection → Q16_1 output (ALL INTEGER)
            // ════════════════════════════════════════════════════════════════════

            std::vector<Q16_1Block> projection_q16(output_blocks);

            IntegerContext context;
            context.int32_data = int32_context.data();
            context.weight_sums = head_weight_sums.data();
            context.v_scales = context_v_scales.data();
            context.num_heads = num_heads;
            context.head_dim = head_dim;
            context.count = input_dim;

            Q16_1Projection projection_out;
            projection_out.blocks = projection_q16.data();
            projection_out.count = d_model;

            WoProjectionVNNIParams wo_params;
            wo_params.Wo_packed = params.Wo_packed;
            wo_params.input_dim = input_dim;
            wo_params.d_model = d_model;
            wo_params.bias = nullptr;

            wo_projection_vpdpwssd_to_q16_1_gemv(wo_params, context, projection_out);

            // ════════════════════════════════════════════════════════════════════
            // SNAPSHOT: Capture attention output (Wo projection, before residual) for this position
            // ════════════════════════════════════════════════════════════════════

            if (params.attention_output_snapshot)
            {
                // Output layout: [seq_len_q × d_model]
                float *out_row = params.attention_output_snapshot + q * d_model;
                for (int b = 0; b < output_blocks; ++b)
                {
                    const Q16_1Block &block = projection_q16[b];
                    const int start = b * Q16_1Block::BLOCK_SIZE;
                    const int end = std::min(start + static_cast<int>(Q16_1Block::BLOCK_SIZE), d_model);
                    for (int i = start; i < end; ++i)
                    {
                        out_row[i] = block.d * static_cast<float>(block.qs[i - start]);
                    }
                }
                if (q == 0)
                {
                    LOG_DEBUG("[Q16FusedAttention PREFILL] Capturing attention_output snapshot for "
                              << seq_len_q << " positions, d_model=" << d_model);
                }
            }

            // ════════════════════════════════════════════════════════════════════
            // STEP 5: Native Q16_1 Residual Add (ALL INTEGER)
            // ════════════════════════════════════════════════════════════════════

            Q16_1Block *residual_out_row = params.residual_out + q * residual_blocks_per_row;
            const Q16_1Block *residual_in_row = params.residual_in + q * residual_blocks_per_row;

            // Copy input to output if different
            if (residual_in_row != residual_out_row)
            {
                std::memcpy(residual_out_row, residual_in_row,
                            output_blocks * sizeof(Q16_1Block));
            }

            // Q16_1 + Q16_1 → Q16_1 - NO FP32!
            simd::q16_1_add_q16_1(
                residual_out_row,
                projection_out.blocks,
                residual_out_row,
                d_model);

            // ════════════════════════════════════════════════════════════════════
            // SNAPSHOT: Capture attention residual (final output) for this position
            // ════════════════════════════════════════════════════════════════════

            if (params.attention_residual_snapshot)
            {
                // Output layout: [seq_len_q × d_model]
                float *res_row = params.attention_residual_snapshot + q * d_model;
                for (int b = 0; b < output_blocks; ++b)
                {
                    const Q16_1Block &block = residual_out_row[b];
                    const int start = b * Q16_1Block::BLOCK_SIZE;
                    const int end = std::min(start + static_cast<int>(Q16_1Block::BLOCK_SIZE), d_model);
                    for (int i = start; i < end; ++i)
                    {
                        res_row[i] = block.d * static_cast<float>(block.qs[i - start]);
                    }
                }
                if (q == 0)
                {
                    LOG_DEBUG("[Q16FusedAttention PREFILL] Capturing attention_residual snapshot for "
                              << seq_len_q << " positions, d_model=" << d_model);
                }
            }
        }

        return true;
    }

} // namespace llaminar2::kernels::q16_1
