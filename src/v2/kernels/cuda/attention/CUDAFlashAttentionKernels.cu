/**
 * @file CUDAFlashAttentionKernels.cu
 * @brief CUDA device kernels for Flash Attention 2 and Flash Decoding
 *
 * This file contains the CUDA device kernels and extern "C" wrapper functions
 * called from the C++ implementation file.
 *
 * Algorithms implemented:
 * - Flash Attention 2: Tiled attention with online softmax for prefill
 * - Flash Decoding: Split-K parallelism for single-token decode
 *
 * @author David Sanftenberg
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstdio>

namespace
{
    // =========================================================================
    // BF16/FP16 Conversion Helpers (same as CUDAOpsKernels.cu)
    // =========================================================================

    __device__ __forceinline__ float bf16_to_float(uint16_t bf16)
    {
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float result;
        memcpy(&result, &bits, sizeof(float));
        return result;
    }

    __device__ __forceinline__ uint16_t float_to_bf16(float f)
    {
        uint32_t bits;
        memcpy(&bits, &f, sizeof(float));
        // Round to nearest even
        bits += 0x7FFF + ((bits >> 16) & 1);
        return static_cast<uint16_t>(bits >> 16);
    }

    __device__ __forceinline__ float fp16_to_float(uint16_t fp16)
    {
        __half h;
        memcpy(&h, &fp16, sizeof(__half));
        return __half2float(h);
    }

    __device__ __forceinline__ uint16_t float_to_fp16(float f)
    {
        __half h = __float2half(f);
        uint16_t result;
        memcpy(&result, &h, sizeof(uint16_t));
        return result;
    }

    // =========================================================================
    // Constants
    // =========================================================================

    constexpr int WARP_SIZE = 32;

    // Tile sizes for Flash Attention 2
    // These are tuned for common GPU architectures (Ampere, Ada)
    // Note: Actual tile sizes are computed at runtime based on head_dim
    // to stay within shared memory limits (100KB on Ampere/Ada)
    constexpr int MAX_TILE_Q = 64;  // Max rows of Q per tile (Br)
    constexpr int MAX_TILE_KV = 64; // Max rows of K/V per tile (Bc)

    // For head_dim > 64, we use smaller tiles to fit in shared memory
    // Formula: smem = (TILE_Q + 2*TILE_KV) * head_dim * 4 + TILE_Q * TILE_KV * 4
    // Target: smem <= 99KB (to leave some margin on 100KB GPUs)
    __host__ __device__ inline int computeTileQ(int head_dim)
    {
        if (head_dim <= 64)
            return 64;
        if (head_dim <= 128)
            return 32; // 32*128*3*4 + 32*32*4 = 49152 + 4096 = 53KB
        return 16;     // For even larger head_dim
    }

    __host__ __device__ inline int computeTileKV(int head_dim)
    {
        if (head_dim <= 64)
            return 64;
        if (head_dim <= 128)
            return 32;
        return 16;
    }

    // Number of splits for Flash Decoding
    constexpr int DEFAULT_NUM_SPLITS = 8;

    // =========================================================================
    // Warp Reduction Utilities
    // =========================================================================

    __device__ __forceinline__ float warpReduceMax(float val)
    {
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        }
        return val;
    }

    __device__ __forceinline__ float warpReduceSum(float val)
    {
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            val += __shfl_xor_sync(0xffffffff, val, offset);
        }
        return val;
    }

    // =========================================================================
    // Flash Attention 2 - Prefill Kernel (FP32)
    // =========================================================================

    /**
     * @brief Flash Attention 2 forward kernel - FP32 version
     *
     * Each thread block handles one (batch, head) pair.
     * Tiles over Q rows (Br) and K/V rows (Bc) with online softmax.
     *
     * Memory layout assumptions:
     *   Q: [batch_size, seq_len, n_heads, head_dim] or [batch_size * seq_len, n_heads, head_dim]
     *   K: [batch_size, kv_len, n_kv_heads, head_dim]
     *   V: [batch_size, kv_len, n_kv_heads, head_dim]
     *   O: [batch_size, seq_len, n_heads, head_dim]
     *
     * For simplicity, we use: [total_tokens, n_heads, head_dim] layout
     * where total_tokens = batch_size * seq_len for Q/O and batch_size * kv_len for K/V.
     */
    __global__ void flash_attention_2_fp32_kernel(
        const float *__restrict__ Q,
        const float *__restrict__ K,
        const float *__restrict__ V,
        float *__restrict__ O,
        int batch_size,
        int seq_len,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float softmax_scale,
        bool causal,
        int window_size,
        int position_offset,
        int tile_q,  // Runtime tile size for Q
        int tile_kv) // Runtime tile size for K/V
    {
        // Each block processes one head for one batch element
        const int batch_idx = blockIdx.z;
        const int head_idx = blockIdx.x;
        const int q_tile_idx = blockIdx.y;

        // GQA: map query head to KV head
        const int kv_head_idx = (n_heads == n_kv_heads) ? head_idx : (head_idx / (n_heads / n_kv_heads));

        // Q row range for this tile
        const int q_start = q_tile_idx * tile_q;
        const int q_end = min(q_start + tile_q, seq_len);
        if (q_start >= seq_len)
            return;

        // Thread position within block
        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;

        // Shared memory for tiles (dynamically sized based on tile_q, tile_kv)
        extern __shared__ char smem[];
        float *Q_tile = reinterpret_cast<float *>(smem);
        float *K_tile = Q_tile + tile_q * head_dim;
        float *V_tile = K_tile + tile_kv * head_dim;
        float *scores = V_tile + tile_kv * head_dim; // [tile_q, tile_kv]

        // Per-row accumulators (in registers)
        // Each thread handles one row of Q (if tid < q_end - q_start)
        const int my_q_row = q_start + tid;
        const bool valid_row = (my_q_row < seq_len) && (tid < tile_q);

        float O_acc[128]; // Max head_dim = 128, adjust if needed
        float m_i = -FLT_MAX;
        float l_i = 0.0f;

        // Initialize output accumulator
        for (int d = 0; d < head_dim; d++)
        {
            O_acc[d] = 0.0f;
        }

        // Pointers for this batch/head
        const float *Q_batch = Q + batch_idx * seq_len * n_heads * head_dim;
        const float *K_batch = K + batch_idx * kv_len * n_kv_heads * head_dim;
        const float *V_batch = V + batch_idx * kv_len * n_kv_heads * head_dim;

        // Load Q tile into shared memory
        // Q layout: [seq_len, n_heads, head_dim]
        for (int i = tid; i < (q_end - q_start) * head_dim; i += num_threads)
        {
            int local_row = i / head_dim;
            int d = i % head_dim;
            int global_row = q_start + local_row;
            Q_tile[local_row * head_dim + d] = Q_batch[global_row * n_heads * head_dim + head_idx * head_dim + d];
        }
        __syncthreads();

        // Iterate over K/V tiles
        const int num_kv_tiles = (kv_len + tile_kv - 1) / tile_kv;

        for (int kv_tile_iter = 0; kv_tile_iter < num_kv_tiles; kv_tile_iter++)
        {
            const int kv_start = kv_tile_iter * tile_kv;
            const int kv_end_tile = min(kv_start + tile_kv, kv_len);
            const int actual_tile_kv_len = kv_end_tile - kv_start;

            // Causal masking: skip KV tiles that are entirely after Q positions
            if (causal)
            {
                // For causal attention, K[j] can only attend to Q[i] if j <= i + position_offset
                // Skip if all K positions are beyond all Q positions in this tile
                if (kv_start > (q_end - 1) + position_offset)
                {
                    continue;
                }
            }

            // Sliding window: skip tiles outside window
            if (window_size > 0)
            {
                // Skip if kv_start is beyond window from all Q positions
                if (kv_start > q_end - 1 + position_offset + window_size)
                {
                    continue;
                }
                // Skip if kv_end is before window starts
                if (kv_end_tile <= q_start + position_offset - window_size)
                {
                    continue;
                }
            }

            // Load K tile
            // K layout: [kv_len, n_kv_heads, head_dim]
            for (int i = tid; i < actual_tile_kv_len * head_dim; i += num_threads)
            {
                int local_row = i / head_dim;
                int d = i % head_dim;
                int global_row = kv_start + local_row;
                K_tile[local_row * head_dim + d] = K_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
            }

            // Load V tile
            for (int i = tid; i < actual_tile_kv_len * head_dim; i += num_threads)
            {
                int local_row = i / head_dim;
                int d = i % head_dim;
                int global_row = kv_start + local_row;
                V_tile[local_row * head_dim + d] = V_batch[global_row * n_kv_heads * head_dim + kv_head_idx * head_dim + d];
            }
            __syncthreads();

            // Compute attention scores for this Q row
            if (valid_row)
            {
                const int local_q_row = tid;
                const float *Q_row = Q_tile + local_q_row * head_dim;

                // Compute scores: Q[i] @ K[j]^T for j in this tile
                float m_ij = -FLT_MAX; // Max in this tile

                for (int j = 0; j < actual_tile_kv_len; j++)
                {
                    const int kv_pos = kv_start + j;

                    // Causal mask: Q at position (my_q_row + position_offset) can attend to K at position kv_pos
                    // if kv_pos <= my_q_row + position_offset
                    bool masked = false;
                    if (causal && kv_pos > my_q_row + position_offset)
                    {
                        masked = true;
                    }

                    // Sliding window mask
                    if (window_size > 0)
                    {
                        int q_pos = my_q_row + position_offset;
                        if (kv_pos < q_pos - window_size || kv_pos > q_pos + window_size)
                        {
                            masked = true;
                        }
                    }

                    float score;
                    if (masked)
                    {
                        score = -FLT_MAX;
                    }
                    else
                    {
                        // Dot product Q[i] @ K[j]
                        score = 0.0f;
                        const float *K_row = K_tile + j * head_dim;
                        for (int d = 0; d < head_dim; d++)
                        {
                            score += Q_row[d] * K_row[d];
                        }
                        score *= softmax_scale;
                    }

                    scores[local_q_row * tile_kv + j] = score;
                    m_ij = fmaxf(m_ij, score);
                }

                // Online softmax update
                float m_i_new = fmaxf(m_i, m_ij);
                float scale_old = expf(m_i - m_i_new);

                // Rescale previous accumulator
                for (int d = 0; d < head_dim; d++)
                {
                    O_acc[d] *= scale_old;
                }
                l_i *= scale_old;

                // Accumulate softmax(scores) @ V
                float l_ij = 0.0f;
                for (int j = 0; j < actual_tile_kv_len; j++)
                {
                    float s = scores[local_q_row * tile_kv + j];
                    if (s > -FLT_MAX / 2)
                    { // Not masked
                        float p = expf(s - m_i_new);
                        l_ij += p;

                        const float *V_row = V_tile + j * head_dim;
                        for (int d = 0; d < head_dim; d++)
                        {
                            O_acc[d] += p * V_row[d];
                        }
                    }
                }

                l_i += l_ij;
                m_i = m_i_new;
            }

            __syncthreads();
        }

        // Final normalization and write output
        if (valid_row)
        {
            float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

            float *O_batch = O + batch_idx * seq_len * n_heads * head_dim;
            float *O_row = O_batch + my_q_row * n_heads * head_dim + head_idx * head_dim;

            for (int d = 0; d < head_dim; d++)
            {
                O_row[d] = O_acc[d] * inv_l;
            }
        }
    }

    // =========================================================================
    // Flash Decoding - Split-K Kernel (FP32)
    // =========================================================================

    /**
     * @brief Flash Decoding kernel for single-query decode
     *
     * Parallelizes over KV cache using split-K pattern.
     * Each block computes partial attention over a range of K/V.
     *
     * Grid: (n_heads, num_splits, batch_size)
     * Block: (256,) threads
     */
    __global__ void flash_decoding_fp32_kernel(
        const float *__restrict__ Q,
        const float *__restrict__ K_cache,
        const float *__restrict__ V_cache,
        float *__restrict__ O_partial,
        float *__restrict__ lse_partial,
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int num_splits,
        float softmax_scale)
    {
        const int head_idx = blockIdx.x;
        const int split_idx = blockIdx.y;
        const int batch_idx = blockIdx.z;

        const int kv_head_idx = (n_heads == n_kv_heads) ? head_idx : (head_idx / (n_heads / n_kv_heads));

        // Determine this split's KV range
        const int split_size = (kv_len + num_splits - 1) / num_splits;
        const int kv_start = split_idx * split_size;
        const int kv_end = min(kv_start + split_size, kv_len);

        if (kv_start >= kv_len)
        {
            // This split has no work - write sentinel values
            if (threadIdx.x == 0)
            {
                lse_partial[(batch_idx * n_heads + head_idx) * num_splits + split_idx] = -FLT_MAX;
            }
            return;
        }

        const int tid = threadIdx.x;
        const int num_threads = blockDim.x;

        // Load Q into shared memory (single query vector)
        extern __shared__ char smem[];
        float *Q_shared = reinterpret_cast<float *>(smem);

        const float *Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += num_threads)
        {
            Q_shared[d] = Q_ptr[d];
        }
        __syncthreads();

        // Per-thread partial sums
        float O_local[128] = {0}; // Max head_dim
        float m_local = -FLT_MAX;
        float l_local = 0.0f;

        // K/V pointers
        const float *K_batch = K_cache + batch_idx * kv_len * n_kv_heads * head_dim;
        const float *V_batch = V_cache + batch_idx * kv_len * n_kv_heads * head_dim;

        // Each thread processes multiple KV positions
        for (int kv_pos = kv_start + tid; kv_pos < kv_end; kv_pos += num_threads)
        {
            // Compute Q @ K^T for this position
            const float *K_ptr = K_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
            {
                score += Q_shared[d] * K_ptr[d];
            }
            score *= softmax_scale;

            // Online softmax update
            float m_new = fmaxf(m_local, score);
            float scale = expf(m_local - m_new);

            l_local = l_local * scale + expf(score - m_new);

            for (int d = 0; d < head_dim; d++)
            {
                O_local[d] *= scale;
            }

            // Accumulate V
            const float *V_ptr = V_batch + kv_pos * n_kv_heads * head_dim + kv_head_idx * head_dim;
            float p = expf(score - m_new);
            for (int d = 0; d < head_dim; d++)
            {
                O_local[d] += p * V_ptr[d];
            }

            m_local = m_new;
        }

        // Warp reduction
        float m_warp = warpReduceMax(m_local);

        // Rescale based on warp max
        float scale_to_warp = expf(m_local - m_warp);
        l_local *= scale_to_warp;
        for (int d = 0; d < head_dim; d++)
        {
            O_local[d] *= scale_to_warp;
        }

        // Reduce l across warp
        float l_warp = warpReduceSum(l_local);

        // Block reduction using shared memory
        __shared__ float block_m[8];       // Max 8 warps
        __shared__ float block_l[8];       // Max 8 warps
        __shared__ float block_O[8 * 128]; // Max 8 warps * max head_dim

        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;
        const int num_warps = num_threads / WARP_SIZE;

        if (lane_id == 0)
        {
            block_m[warp_id] = m_warp;
            block_l[warp_id] = l_warp;
        }

        // Store O to shared (only first thread in warp)
        // Note: This is simplified - proper implementation needs O reduction
        if (lane_id == 0)
        {
            for (int d = 0; d < head_dim; d++)
            {
                block_O[warp_id * head_dim + d] = O_local[d];
            }
        }
        __syncthreads();

        // Final reduction by first warp
        if (warp_id == 0 && lane_id < num_warps)
        {
            float my_m = block_m[lane_id];
            float my_l = block_l[lane_id];

            // Find global max across warps
            float global_m = warpReduceMax(my_m);

            // Rescale and sum
            float scale = expf(my_m - global_m);
            float scaled_l = my_l * scale;
            float global_l = warpReduceSum(scaled_l);

            // Output reduction (simplified - first thread writes)
            if (lane_id == 0)
            {
                float *O_out = O_partial + ((batch_idx * n_heads + head_idx) * num_splits + split_idx) * head_dim;

                // Combine O from all warps with proper scaling
                for (int d = 0; d < head_dim; d++)
                {
                    float sum = 0.0f;
                    for (int w = 0; w < num_warps; w++)
                    {
                        float w_scale = expf(block_m[w] - global_m);
                        sum += block_O[w * head_dim + d] * w_scale;
                    }
                    O_out[d] = sum;
                }

                // Store logsumexp for this split
                lse_partial[(batch_idx * n_heads + head_idx) * num_splits + split_idx] =
                    global_m + logf(global_l + 1e-10f);
            }
        }
    }

    /**
     * @brief Flash Decoding reduction kernel
     *
     * Combines partial outputs from all splits using LSE correction.
     */
    __global__ void flash_decoding_reduce_fp32_kernel(
        const float *__restrict__ O_partial,
        const float *__restrict__ lse_partial,
        float *__restrict__ O,
        int n_heads,
        int head_dim,
        int num_splits)
    {
        const int head_idx = blockIdx.x;
        const int batch_idx = blockIdx.y;

        const int tid = threadIdx.x;

        // Find global max LSE across splits
        float lse_max = -FLT_MAX;
        for (int s = 0; s < num_splits; s++)
        {
            float lse = lse_partial[(batch_idx * n_heads + head_idx) * num_splits + s];
            lse_max = fmaxf(lse_max, lse);
        }

        // Combine outputs with LSE correction
        float l_combined = 0.0f;
        float O_combined[128] = {0}; // Max head_dim

        for (int s = 0; s < num_splits; s++)
        {
            float lse = lse_partial[(batch_idx * n_heads + head_idx) * num_splits + s];
            if (lse > -FLT_MAX / 2)
            { // Valid split
                float scale = expf(lse - lse_max);
                l_combined += scale;

                const float *O_s = O_partial + ((batch_idx * n_heads + head_idx) * num_splits + s) * head_dim;
                for (int d = tid; d < head_dim; d += blockDim.x)
                {
                    O_combined[d] += scale * O_s[d];
                }
            }
        }

        // Normalize and write output
        float inv_l = (l_combined > 0.0f) ? (1.0f / l_combined) : 0.0f;
        float *O_out = O + (batch_idx * n_heads + head_idx) * head_dim;

        for (int d = tid; d < head_dim; d += blockDim.x)
        {
            O_out[d] = O_combined[d] * inv_l;
        }
    }

} // anonymous namespace

// =============================================================================
// Extern "C" Wrapper Functions
// =============================================================================

extern "C"
{
    /**
     * @brief Launch Flash Attention 2 prefill kernel (FP32)
     */
    int cudaFlashAttn_prefill_fp32(
        const float *Q, const float *K, const float *V, float *O,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset,
        void *stream)
    {
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Compute adaptive tile sizes based on head_dim to fit shared memory
        const int tile_q = computeTileQ(head_dim);
        const int tile_kv = computeTileKV(head_dim);

        // Grid: (n_heads, num_q_tiles, batch_size)
        int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
        dim3 grid(n_heads, num_q_tiles, batch_size);

        // Block: enough threads for tile_q rows
        int block_size = tile_q; // One thread per Q row in tile
        block_size = max(block_size, 64);
        block_size = min(block_size, 256);

        // Shared memory: Q_tile + K_tile + V_tile + scores
        size_t smem_size = (tile_q * head_dim + tile_kv * head_dim + tile_kv * head_dim + tile_q * tile_kv) * sizeof(float);

        // Configure shared memory carveout for larger shared memory
        // RTX 3090 supports up to 100KB per SM, but default is 48KB
        cudaFuncSetAttribute(flash_attention_2_fp32_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);

        flash_attention_2_fp32_kernel<<<grid, block_size, smem_size, cuda_stream>>>(
            Q, K, V, O,
            batch_size, seq_len, kv_len,
            n_heads, n_kv_heads, head_dim,
            softmax_scale, causal, window_size, position_offset,
            tile_q, tile_kv); // Pass tile sizes to kernel

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("[cudaFlashAttn_prefill_fp32] CUDA error: %s (smem=%zu bytes, grid=(%d,%d,%d), block=%d)\n",
                   cudaGetErrorString(err), smem_size, grid.x, grid.y, grid.z, block_size);
            return -1;
        }
        return 0;
    }

    /**
     * @brief Launch Flash Decoding kernel (FP32)
     */
    int cudaFlashAttn_decode_fp32(
        const float *Q, const float *K_cache, const float *V_cache, float *O,
        float *O_partial, float *lse_partial,
        int batch_size, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        int num_splits,
        void *stream)
    {
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        float softmax_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

        // Phase 1: Compute partial attention per split
        {
            dim3 grid(n_heads, num_splits, batch_size);
            int block_size = 256;

            size_t smem_size = head_dim * sizeof(float); // Q_shared

            flash_decoding_fp32_kernel<<<grid, block_size, smem_size, cuda_stream>>>(
                Q, K_cache, V_cache,
                O_partial, lse_partial,
                kv_len, n_heads, n_kv_heads, head_dim,
                num_splits, softmax_scale);
        }

        // Phase 2: Reduce partials to final output
        {
            dim3 grid(n_heads, batch_size);
            int block_size = min(head_dim, 256);

            flash_decoding_reduce_fp32_kernel<<<grid, block_size, 0, cuda_stream>>>(
                O_partial, lse_partial, O,
                n_heads, head_dim, num_splits);
        }

        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Allocate workspace for Flash Decoding
     */
    int cudaFlashAttn_allocWorkspace(
        void **partial_output, void **partial_lse,
        int batch_size, int n_heads, int head_dim, int num_splits)
    {
        size_t output_size = static_cast<size_t>(batch_size) * n_heads * num_splits * head_dim * sizeof(float);
        size_t lse_size = static_cast<size_t>(batch_size) * n_heads * num_splits * sizeof(float);

        cudaError_t err1 = cudaMalloc(partial_output, output_size);
        cudaError_t err2 = cudaMalloc(partial_lse, lse_size);

        if (err1 != cudaSuccess || err2 != cudaSuccess)
        {
            if (*partial_output)
                cudaFree(*partial_output);
            if (*partial_lse)
                cudaFree(*partial_lse);
            *partial_output = nullptr;
            *partial_lse = nullptr;
            return -1;
        }

        return 0;
    }

    /**
     * @brief Free workspace
     */
    void cudaFlashAttn_freeWorkspace(void *partial_output, void *partial_lse)
    {
        if (partial_output)
            cudaFree(partial_output);
        if (partial_lse)
            cudaFree(partial_lse);
    }

    /**
     * @brief Set CUDA device
     */
    int cudaFlashAttn_setDevice(int device_idx)
    {
        return cudaSetDevice(device_idx) == cudaSuccess ? 0 : -1;
    }

    /**
     * @brief Synchronize device
     */
    int cudaFlashAttn_synchronize()
    {
        return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
    }

} // extern "C"
