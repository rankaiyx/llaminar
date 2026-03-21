/**
 * @file CPUNativeVNNITileConfig.h
 * @brief Cache-aware tile configuration for CPU NativeVNNI GEMV/GEMM.
 *
 * Uses detected L1/L2/L3 cache sizes (via CPUFeatures.h) to select optimal
 * tile sizes per shape category (Attention, FFN, LM_Head).
 *
 * ## Shape Categories
 *
 * | Category   | N range        | K range        | Example (Qwen 3B)     |
 * |------------|----------------|----------------|-----------------------|
 * | Attention  | 128–2048       | 896–3584       | K_proj [256×2048]     |
 * | FFN        | 4864–18944     | 896–18944      | FFN_Gate [11008×2048] |
 * | LM_Head    | 32000–152000   | 896–3584       | LM_Head [151936×3584] |
 *
 * ## Tile Strategy
 *
 * For GEMV (M=1), the weight matrix is streamed once (no reuse). The key
 * parameters are:
 * - **N_chunk**: Fixed at 64 (AVX-512 ZMM width).
 * - **N_block**: Number of N_chunks processed per task block. Controls
 *   L2 residency for scales/comp data.
 * - **K_tile**: Number of K-blocks per tile. Controls L1 residency for
 *   activation quantized blocks. For M=1, full K usually fits.
 *
 * For GEMM (M>1), B-tile reuse across M rows matters:
 * - **n_block_size**: Chosen so B_tile fits in L2 (nblock × K × payload_bytes)
 * - **k_tile_blocks**: Chosen so one K-tile of B fits in 50% of L2
 * - **m_unroll**: 2 for M≥2 (process 2 rows to amortize B loads)
 */

#pragma once

#include <algorithm>
#include <cstdint>

#include "utils/CPUFeatures.h"

namespace llaminar2::cpu::native_vnni
{

    /**
     * @brief Shape category for tile selection.
     */
    enum class ShapeCategory
    {
        ATTENTION, ///< Small N projections (Q/K/V/Wo)
        FFN,       ///< Medium-large N (Gate/Up/Down)
        LM_HEAD,   ///< Very large N (vocabulary projection)
        GENERIC    ///< Fallback
    };

    /**
     * @brief Tile configuration for NativeVNNI GEMV/GEMM.
     */
    struct NativeVNNITileConfig
    {
        int n_block_chunks; ///< N-chunks per parallel task (each chunk = 64 cols)
        int k_tile_blocks;  ///< K-blocks per tile (0 = full K, no tiling)
        int m_unroll;       ///< M-loop unroll factor (1 for GEMV, 2 for GEMM)
        int omp_min_tasks;  ///< Minimum parallel tasks before falling back to serial
        ShapeCategory category;
        int k_tiles; ///< K-parallel tiles for GEMV (0 = N-parallel only)
    };

    /**
     * @brief Compute tile configuration based on shape, cache sizes, and format.
     *
     * @param N           Output dimension (number of weight rows)
     * @param K           Input dimension (weight row length)
     * @param M           Batch size (1 = GEMV, >1 = GEMM)
     * @param payload_bytes  Bytes per native block (e.g., 16 for Q4_0)
     * @param num_threads Number of OpenMP threads
     * @return Optimal tile configuration
     */
    inline NativeVNNITileConfig computeTileConfig(
        int N, int K, int M, int payload_bytes, int num_threads)
    {
        CacheInfo cache;
        NativeVNNITileConfig cfg{};

        int N_chunks = (N + 63) / 64;
        int blocks_per_row = (K + 31) / 32;

        // Classify shape
        if (N <= 2048)
            cfg.category = ShapeCategory::ATTENTION;
        else if (N <= 20000)
            cfg.category = ShapeCategory::FFN;
        else
            cfg.category = ShapeCategory::LM_HEAD;

        // ---------------------------------------------------------------
        // GEMV (M=1): Weight matrix streamed once, no B reuse
        // ---------------------------------------------------------------
        if (M <= 1)
        {
            cfg.m_unroll = 1;
            cfg.k_tile_blocks = 0; // Full K (no K-tiling for M=1)
            cfg.k_tiles = 0;       // Default: N-parallel only

            // Compute bytes per N-chunk per K-block iteration:
            //   vnni_b: 2048 bytes (8 subs × 4 zmms × 64 bytes)
            //   scales: 256 bytes (64 floats)
            //   comp:   256 bytes (64 int32s)
            //   = 2560 bytes per K-block per chunk
            //
            // For streaming, we want each thread's working set to stay in L2.
            // L2 typically: 1MB (Cascade Lake), 1.25MB (Ice Lake), 2MB (Sapphire Rapids)
            //
            // Working set per chunk = blocks_per_row * 2560 bytes
            // Max chunks fitting in 75% of L2:
            long long l2_usable = (long long)cache.l2_size * 3 / 4;
            long long bytes_per_chunk = (long long)blocks_per_row * 2560;
            int max_chunks_l2 = (int)(l2_usable / std::max(bytes_per_chunk, 1LL));
            max_chunks_l2 = std::max(max_chunks_l2, 1);

            // Target parallelism: at least 4× threads for good load balance
            int target_tasks = num_threads * 4;
            int chunks_per_task = std::max(1, N_chunks / target_tasks);

            // For Attention (small N), use 1 chunk per task for maximum parallelism
            // For FFN/LM_Head (large N), use larger blocks up to L2 limit
            switch (cfg.category)
            {
            case ShapeCategory::ATTENTION:
                // Small N (128-2048): 2-32 chunks total
                // Use 1 chunk per task to maximize parallelism
                cfg.n_block_chunks = 1;
                break;

            case ShapeCategory::FFN:
                // Medium N (4864-18944): 76-296 chunks total
                // Use cache-limited block size for streaming efficiency
                cfg.n_block_chunks = std::min(chunks_per_task, max_chunks_l2);
                cfg.n_block_chunks = std::max(cfg.n_block_chunks, 1);
                break;

            case ShapeCategory::LM_HEAD:
                // Large N (32000+): 500+ chunks total
                // More chunks per task to amortize OpenMP overhead
                cfg.n_block_chunks = std::min(chunks_per_task, max_chunks_l2);
                cfg.n_block_chunks = std::max(cfg.n_block_chunks, 2);
                break;

            default:
                cfg.n_block_chunks = std::min(chunks_per_task, max_chunks_l2);
                cfg.n_block_chunks = std::max(cfg.n_block_chunks, 1);
                break;
            }

            // ---------------------------------------------------------------
            // K-parallel: when N-parallelism yields fewer tasks than threads,
            // split the K dimension across multiple threads per N-chunk.
            // Each thread computes partial sums over a K-block range, then a
            // reduction phase combines them.
            //
            // Guards:
            //  1. K dimension must be large (bpr >= 256 ≈ K>=8192) so the
            //     reduction overhead is amortized by the DRAM bandwidth gain.
            //     Medium-K shapes (attention projections) are better served by
            //     N-parallel only — their weights fit in L3 and the reduction
            //     cost exceeds the extra-thread benefit.
            //  2. Each K-tile must have >= 4 K-blocks of work so per-task
            //     overhead doesn't dominate.
            // ---------------------------------------------------------------
            constexpr int MIN_BPR_FOR_K_PARALLEL = 256;
            int total_n_tasks = (N_chunks + cfg.n_block_chunks - 1) / cfg.n_block_chunks;
            if (total_n_tasks < num_threads && blocks_per_row >= MIN_BPR_FOR_K_PARALLEL)
            {
                // Compute k_tiles so that total 2D tasks ≈ num_threads
                int desired_k_tiles = (num_threads + total_n_tasks - 1) / total_n_tasks;
                // Each K-tile should have at least 4 K-blocks to amortize overhead
                constexpr int MIN_K_BLOCKS_PER_TILE = 4;
                int max_k_tiles = std::max(1, blocks_per_row / MIN_K_BLOCKS_PER_TILE);
                cfg.k_tiles = std::clamp(desired_k_tiles, 1, max_k_tiles);

                // Guard: each K-tile must process enough K-blocks
                int k_blocks_per_tile = (blocks_per_row + cfg.k_tiles - 1) / cfg.k_tiles;
                if (k_blocks_per_tile < MIN_K_BLOCKS_PER_TILE)
                    cfg.k_tiles = 0;
                // Don't bother if we'd only get 1 tile
                if (cfg.k_tiles <= 1)
                    cfg.k_tiles = 0;
            }

            // Minimum tasks threshold: below this, overhead > benefit
            cfg.omp_min_tasks = std::max(2, num_threads / 2);
            return cfg;
        }

        // ---------------------------------------------------------------
        // GEMM (M>1): B-tile reuse across M rows matters
        // ---------------------------------------------------------------
        cfg.m_unroll = (M >= 2) ? 2 : 1;
        cfg.k_tiles = 0; // K-parallel only for GEMV

        // B-tile must fit in L2 for reuse across M rows
        // B-tile size = n_block_size × blocks_per_row × (2048 per chunk + 256 scales + 256 comp)
        long long l2_for_b = (long long)cache.l2_size * 3 / 4;
        long long bytes_per_chunk_full_k = (long long)blocks_per_row * 2560;
        int max_chunks_for_b = (int)(l2_for_b / std::max(bytes_per_chunk_full_k, 1LL));
        max_chunks_for_b = std::max(max_chunks_for_b, 1);

        // Check if full K fits; if not, tile K
        if (bytes_per_chunk_full_k > l2_for_b)
        {
            // K-tile so that one chunk stays in L2
            long long target_k_bytes = l2_for_b / 2; // Leave room for activations
            cfg.k_tile_blocks = (int)(target_k_bytes / 2560);
            cfg.k_tile_blocks = std::clamp(cfg.k_tile_blocks, 4, blocks_per_row);
            cfg.n_block_chunks = 1;
        }
        else
        {
            cfg.k_tile_blocks = 0; // Full K
            // Choose n_block_chunks: balance parallelism vs cache
            int target_tasks = num_threads * 2;
            int m_tasks = (M + cfg.m_unroll - 1) / cfg.m_unroll;
            int needed_n_tasks = std::max(1, target_tasks / m_tasks);
            int calc_chunks = std::max(1, N_chunks / needed_n_tasks);
            cfg.n_block_chunks = std::min(calc_chunks, max_chunks_for_b);
            cfg.n_block_chunks = std::max(cfg.n_block_chunks, 1);
        }

        cfg.omp_min_tasks = std::max(2, num_threads / 2);
        return cfg;
    }

} // namespace llaminar2::cpu::native_vnni
