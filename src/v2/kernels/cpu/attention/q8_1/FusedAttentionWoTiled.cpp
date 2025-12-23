/**
 * @file FusedAttentionWoTiled.cpp
 * @brief Cache-blocked tiled implementation of fused attention + Wo projection.
 *
 * Implements the FlashAttention-style tiling algorithm with Wo fusion:
 * - K/V tiles loaded into L2, reused across multiple queries
 * - Per-query softmax state maintained across KV tiles
 * - Wo projection fused at end of each Q tile
 * - OpenMP parallelization across attention heads (FA2-style)
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include "FusedAttentionWoTiled.h"
#include "FusedAttentionWoRef.h"
#include "microkernels/Q8DotProduct.h"
#include "microkernels/OnlineSoftmax.h"
#include "microkernels/VWeightedAccum.h"
#include "microkernels/WoProjection.h"
#include "microkernels/FastExp.h"
#include "../../../../utils/Logger.h"
#include "../../../../utils/OpenMPUtils.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

#ifdef __x86_64__
#include <xmmintrin.h> // For _mm_prefetch
#endif

namespace llaminar::v2::kernels
{

    // ============================================================================
    // Tile Configuration
    // ============================================================================

    TileConfig compute_tile_config(int head_dim, int d_model)
    {
        using namespace llaminar2;

        TileConfig config;
        config.l2_size = cpu_l2_cache_size(); // Per-core L2 (e.g., 1MB on Xeon)
        config.l3_size = cpu_l3_cache_size(); // Shared L3 (e.g., 38MB on Xeon)

        // Q8_1Block layout: head_dim / 32 blocks per row
        // Each block: 32 int8 + 2 bytes (fp16 scale) + 2 bytes (int16 sum) = 36 bytes
        const int num_blocks = head_dim / 32;
        const int bytes_per_kv_row = num_blocks * 36 * 2; // K + V row together

        // Target: Use 50% of L2 for K/V tile (leave room for Q, context, scratch)
        const int l2_for_kv = config.l2_size / 2;
        config.kv_tile = std::max(32, std::min(512, l2_for_kv / bytes_per_kv_row));

        // Q_TILE: Each query needs head_dim * 4 bytes for context accumulator
        // Target: 25% of L2 for context buffers
        const int l2_for_context = config.l2_size / 4;
        const int bytes_per_context = head_dim * sizeof(float);
        config.q_tile = std::max(8, std::min(64, l2_for_context / bytes_per_context));

        // Wo tiling (optional, for very large d_model > 4096)
        // For Qwen2 models up to 72B (d_model=8192), single pass is fine
        config.wo_tile = d_model; // No Wo tiling needed for typical model sizes

        LOG_DEBUG("TileConfig: L2=" << config.l2_size / 1024 << "KB"
                                    << " L3=" << config.l3_size / (1024 * 1024) << "MB"
                                    << " kv_tile=" << config.kv_tile
                                    << " q_tile=" << config.q_tile
                                    << " (head_dim=" << head_dim << ", d_model=" << d_model << ")");

        return config;
    }

    // ============================================================================
    // Main Entry Points
    // ============================================================================

    bool FusedAttentionWoTiled::execute(const FusedAttentionWoParams &params)
    {
        // Compute optimal tile configuration
        TileConfig config = compute_tile_config(params.head_dim, params.d_model);
        return execute(params, config);
    }

    bool FusedAttentionWoTiled::execute(
        const FusedAttentionWoParams &params,
        const TileConfig &config)
    {
        // Validate parameters (reuse validation from reference)
        if (!FusedAttentionWoRef::validate_params(params))
        {
            return false;
        }

        // Check if tiling is beneficial
        if (!config.should_tile(params.kv_seq_len))
        {
            // Fall back to reference for short sequences
            LOG_DEBUG("FusedAttentionWoTiled: kv_seq_len=" << params.kv_seq_len
                                                           << " <= kv_tile=" << config.kv_tile << ", using reference impl");
            return FusedAttentionWoRef::execute(params);
        }

        const int batch_size = params.batch_size;
        const int seq_len = params.seq_len;
        const int d_model = params.d_model;

        // Zero entire output buffer
        std::memset(params.output, 0,
                    static_cast<size_t>(batch_size) * seq_len * d_model * sizeof(float));

        // Process each batch item with tiling
        for (int b = 0; b < batch_size; ++b)
        {
            process_batch_item_tiled(params, b, config);
        }

        return true;
    }

    TileConfig FusedAttentionWoTiled::get_tile_config(const FusedAttentionWoParams &params)
    {
        return compute_tile_config(params.head_dim, params.d_model);
    }

    // ============================================================================
    // Batch Item Processing (Tiled with OpenMP Head Parallelism)
    // ============================================================================

    void FusedAttentionWoTiled::process_batch_item_tiled(
        const FusedAttentionWoParams &params,
        int batch_idx,
        const TileConfig &config)
    {
        using namespace llaminar::v2;

        const int seq_len = params.seq_len;
        const int num_heads = params.num_heads;
        const int head_dim = params.head_dim;
        const int d_model = params.d_model;
        const int q_tile = config.q_tile;

        // Get KV length and position offset for this batch item
        const int kv_len = params.get_kv_len(batch_idx);
        const int pos_offset = params.get_position_offset(batch_idx);

        // Compute output offset for this batch item
        const int num_blocks = head_dim / 32;
        const size_t output_batch_stride = static_cast<size_t>(seq_len) * d_model;
        float *output_base = params.output + batch_idx * output_batch_stride;

        // Process query positions in tiles
        for (int q_start = 0; q_start < seq_len; q_start += q_tile)
        {
            const int q_end = std::min(q_start + q_tile, seq_len);
            const int q_count = q_end - q_start;

            // ================================================================
            // FA2-Style: Parallel head processing with per-head Wo outputs
            // ================================================================
            // Each thread processes one or more heads independently.
            // Wo projection outputs are accumulated into per-head buffers,
            // then reduced to final output to avoid thread contention.
            //
            // Uses OMP_WORKSHARE_REGION for nested-safe parallelism:
            // - If called from within an existing parallel region, reuses threads
            // - If called standalone, creates new parallel region
            // ================================================================

            // Per-head Wo output buffers: [num_heads][q_count][d_model]
            // Each head writes its contribution here, then we reduce.
            std::vector<float> head_outputs(num_heads * q_count * d_model, 0.0f);

            // Define all parallel work in a single lambda for OMP_WORKSHARE_REGION
            auto do_attention_work = [&]()
            {
                // Thread-local buffers for context and softmax state
                // Allocated per-thread inside the parallel region to avoid false sharing
                alignas(64) std::vector<float> context_buffers(q_count * head_dim);
                std::vector<OnlineSoftmaxStateTiled> softmax_states(q_count);

// Phase 1: Process heads in parallel
#pragma omp for schedule(static)
                for (int h = 0; h < num_heads; ++h)
                {
                    // Reset context buffers and softmax states for this head
                    std::memset(context_buffers.data(), 0, q_count * head_dim * sizeof(float));
                    for (int i = 0; i < q_count; ++i)
                    {
                        softmax_states[i] = OnlineSoftmaxStateTiled();
                    }

                    // Process this head with tiling
                    process_head_tiled(
                        params, batch_idx, h,
                        q_start, q_end,
                        kv_len, pos_offset,
                        config,
                        context_buffers.data(),
                        softmax_states.data());

                    // Finalize softmax and project through Wo for each query in tile
                    // Write to per-head output buffer (no contention)
                    float *head_output_base = head_outputs.data() + h * q_count * d_model;

                    for (int qi = 0; qi < q_count; ++qi)
                    {
                        float *context = context_buffers.data() + qi * head_dim;
                        const OnlineSoftmaxStateTiled &state = softmax_states[qi];

                        // Normalize context by 1/sum_exp
                        if (state.sum_exp > 0.0f)
                        {
                            const float inv_sum = 1.0f / state.sum_exp;
                            for (int d = 0; d < head_dim; ++d)
                            {
                                context[d] *= inv_sum;
                            }
                        }

                        // Project through Wo into per-head buffer (no accumulate - fresh buffer)
                        float *head_output_row = head_output_base + qi * d_model;

                        microkernels::WoProjectionParams wo_params;
                        wo_params.context = context;
                        wo_params.wo_weights = params.Wo;
                        wo_params.wo_type = params.wo_type;
                        wo_params.head_dim = head_dim;
                        wo_params.d_model = d_model;
                        wo_params.head_idx = h;
                        wo_params.n_heads = num_heads;
                        wo_params.output = head_output_row;
                        wo_params.accumulate = false; // Write directly (we'll reduce later)

                        microkernels::wo_projection_ref(wo_params);
                    }
                }

// Implicit barrier after #pragma omp for

// Phase 2: Reduce per-head outputs into final output (parallel reduction)
#pragma omp for schedule(static)
                for (int qi = 0; qi < q_count; ++qi)
                {
                    const int m = q_start + qi;
                    float *output_row = output_base + m * d_model;

                    // Sum contributions from all heads
                    for (int h = 0; h < num_heads; ++h)
                    {
                        const float *head_output_row = head_outputs.data() +
                                                       h * q_count * d_model + qi * d_model;
                        for (int d = 0; d < d_model; ++d)
                        {
                            output_row[d] += head_output_row[d];
                        }
                    }
                }
            }; // end do_attention_work lambda

            // Execute using OMP_WORKSHARE_REGION for nested-safe parallelism
            OMP_WORKSHARE_REGION(do_attention_work);
        }
    }

    // ============================================================================
    // Head Processing (Tiled Core Algorithm)
    // ============================================================================

    /**
     * @brief FA2-style tile-wise softmax batching
     *
     * Instead of updating softmax state per KV position, we:
     * 1. Compute ALL scores in the KV tile first
     * 2. Find the tile maximum in one pass
     * 3. Update softmax state ONCE per tile
     * 4. Apply correction and accumulate all V values
     *
     * This reduces per-position overhead by ~tile_size× (typically 64×).
     */
    void FusedAttentionWoTiled::process_kv_tile_batched(
        const FusedAttentionWoParams &params,
        int batch_idx,
        int head_idx,
        int kv_head,
        int q_idx,                // Which query position
        int kv_start,             // Start of KV tile
        int kv_end,               // End of KV tile
        const Q8_1Block *Q_row,   // Q for this query position
        const Q8_1Block *K_batch, // K tensor base for this batch
        const Q8_1Block *V_batch, // V tensor base for this batch
        int num_blocks,
        int num_kv_heads,
        float *context, // [head_dim] output accumulator
        OnlineSoftmaxStateTiled &state,
        float *tile_scores) // [kv_tile] scratch buffer for scores
    {
        using namespace microkernels;

        const int head_dim = params.head_dim;
        const int tile_len = kv_end - kv_start;

        // ================================================================
        // Phase 1: Compute all Q·K scores for this tile
        // ================================================================
        for (int t = 0; t < tile_len; ++t)
        {
            const int n = kv_start + t;
            const Q8_1Block *K_row = K_batch +
                                     (static_cast<size_t>(n) * num_kv_heads + kv_head) * num_blocks;

            Q8DotProductParams dot_params;
            dot_params.q_blocks = Q_row;
            dot_params.k_blocks = K_row;
            dot_params.num_blocks = num_blocks;
            dot_params.global_scale = params.scale;

            tile_scores[t] = q8_dot_product_ref(dot_params).score;
        }

        // ================================================================
        // Phase 2: Find tile maximum
        // ================================================================
        float tile_max = tile_scores[0];
        for (int t = 1; t < tile_len; ++t)
        {
            if (tile_scores[t] > tile_max)
            {
                tile_max = tile_scores[t];
            }
        }

        // ================================================================
        // Phase 3: Update softmax state ONCE for entire tile
        // ================================================================
        float correction = 1.0f;

        if (!state.initialized)
        {
            // First tile: initialize state with tile_max
            state.max_score = tile_max;
            state.initialized = true;
        }
        else if (tile_max > state.max_score)
        {
            // New global maximum: compute correction for existing context
            correction = fast_exp_poly(state.max_score - tile_max);
            state.sum_exp *= correction;
            state.max_score = tile_max;

            // Apply correction to context ONCE per tile (not per position!)
            for (int d = 0; d < head_dim; ++d)
            {
                context[d] *= correction;
            }
        }

        // ================================================================
        // Phase 4: Compute weights and accumulate V for entire tile
        // ================================================================
        for (int t = 0; t < tile_len; ++t)
        {
            const int n = kv_start + t;

            // Compute weight relative to current global max
            float weight = fast_exp_poly(tile_scores[t] - state.max_score);
            state.sum_exp += weight;

            // Get V row and accumulate weighted V
            const Q8_1Block *V_row = V_batch +
                                     (static_cast<size_t>(n) * num_kv_heads + kv_head) * num_blocks;

            VWeightedAccumParams accum_params;
            accum_params.v_blocks = V_row;
            accum_params.weight = weight;
            accum_params.correction = 1.0f; // Correction already applied above
            accum_params.context = context;
            accum_params.num_blocks = num_blocks;

            v_weighted_accum_ref(accum_params);
        }
    }

    void FusedAttentionWoTiled::process_head_tiled(
        const FusedAttentionWoParams &params,
        int batch_idx,
        int head_idx,
        int q_start,
        int q_end,
        int kv_len,
        int pos_offset,
        const TileConfig &config,
        float *context_buffers,                 // [q_count][head_dim]
        OnlineSoftmaxStateTiled *softmax_states // [q_count]
    )
    {
        using namespace microkernels;

        const int head_dim = params.head_dim;
        const int num_blocks = head_dim / 32;
        const int num_kv_heads = params.num_kv_heads;
        const int kv_head = head_idx / (params.num_heads / num_kv_heads); // GQA mapping
        const int kv_tile = config.kv_tile;
        const int q_count = q_end - q_start;

        // Thread-local scratch buffer for tile scores (FA2 batching)
        // Sized to hold one KV tile worth of scores
        alignas(64) std::vector<float> tile_scores(kv_tile);

        // Compute batch strides
        const size_t q_batch_stride = static_cast<size_t>(params.seq_len) *
                                      params.num_heads * num_blocks;
        const size_t kv_batch_stride = static_cast<size_t>(params.kv_seq_len) *
                                       num_kv_heads * num_blocks;

        // Get tensor bases for this batch
        const Q8_1Block *Q_batch = params.Q + batch_idx * q_batch_stride;
        const Q8_1Block *K_batch = params.K + batch_idx * kv_batch_stride;
        const Q8_1Block *V_batch = params.V + batch_idx * kv_batch_stride;

        // Process KV positions in tiles
        for (int kv_start = 0; kv_start < kv_len; kv_start += kv_tile)
        {
            const int kv_end = std::min(kv_start + kv_tile, kv_len);

            // Prefetch next KV tile while processing current
            if (kv_end < kv_len)
            {
                const int next_kv_end = std::min(kv_end + kv_tile, kv_len);
                prefetch_kv_tile(K_batch, V_batch, kv_head, kv_end, next_kv_end,
                                 num_kv_heads, num_blocks);
            }

            // Process each query in the Q tile against this KV tile
            for (int qi = 0; qi < q_count; ++qi)
            {
                const int m = q_start + qi;
                float *context = context_buffers + qi * head_dim;
                OnlineSoftmaxStateTiled &state = softmax_states[qi];

                // Get Q row for this query position
                const Q8_1Block *Q_row = Q_batch +
                                         (static_cast<size_t>(m) * params.num_heads + head_idx) * num_blocks;

                // Determine effective KV range for this query (causal masking)
                int effective_kv_start = kv_start;
                int effective_kv_end = kv_end;

                if (params.causal)
                {
                    // Can only attend to positions <= m + pos_offset
                    const int max_attend = m + pos_offset + 1;
                    if (max_attend <= kv_start)
                    {
                        // This query cannot attend to any position in this KV tile
                        continue;
                    }
                    effective_kv_end = std::min(effective_kv_end, max_attend);
                }

                // FA2-style: Process entire KV tile with batched softmax
                // This computes all scores first, finds tile max, then updates state ONCE
                process_kv_tile_batched(
                    params, batch_idx, head_idx, kv_head, qi,
                    effective_kv_start, effective_kv_end,
                    Q_row, K_batch, V_batch,
                    num_blocks, num_kv_heads,
                    context, state,
                    tile_scores.data());
            }
        }
    }

    // ============================================================================
    // Prefetch Utilities
    // ============================================================================

    void FusedAttentionWoTiled::prefetch_kv_tile(
        const Q8_1Block *K_base,
        const Q8_1Block *V_base,
        int kv_head,
        int kv_start,
        int kv_end,
        int num_kv_heads,
        int num_blocks)
    {
#ifdef __x86_64__
        // Prefetch K and V rows into L2 cache
        constexpr int CACHE_LINE = 64; // x86 cache line size

        for (int n = kv_start; n < kv_end; ++n)
        {
            const size_t row_offset = static_cast<size_t>(n) * num_kv_heads + kv_head;

            // Prefetch K row
            const char *k_ptr = reinterpret_cast<const char *>(K_base + row_offset * num_blocks);
            const int k_bytes = num_blocks * sizeof(Q8_1Block);
            for (int offset = 0; offset < k_bytes; offset += CACHE_LINE)
            {
                _mm_prefetch(k_ptr + offset, _MM_HINT_T1); // Prefetch to L2
            }

            // Prefetch V row
            const char *v_ptr = reinterpret_cast<const char *>(V_base + row_offset * num_blocks);
            const int v_bytes = num_blocks * sizeof(Q8_1Block);
            for (int offset = 0; offset < v_bytes; offset += CACHE_LINE)
            {
                _mm_prefetch(v_ptr + offset, _MM_HINT_T1); // Prefetch to L2
            }
        }
#else
        // Non-x86: no prefetch available, rely on hardware prefetcher
        (void)K_base;
        (void)V_base;
        (void)kv_head;
        (void)kv_start;
        (void)kv_end;
        (void)num_kv_heads;
        (void)num_blocks;
#endif
    }

} // namespace llaminar::v2::kernels
