/**
 * @file CUDARoPEKernels.cu
 * @brief CUDA RoPE (Rotary Position Embedding) kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32, BF16, and FP16 RoPE kernels with extern "C" wrapper functions.
 * Uses SPLIT-HALF layout matching the CPU implementation.
 *
 * OPTIMIZATIONS (v3 - CPU strategy adaptation):
 * - Pre-computed inverse frequency table cached in device memory
 * - Shared memory for sin/cos table per thread block
 * - Main rotation is pure FMA (no per-thread transcendentals)
 * - Single fused kernel for Q+K to reduce launch overhead
 *
 * OPTIMIZATION (v4 - Fused Q+K kernel):
 * - Process both Q and K in a single kernel launch to reduce overhead
 */

#include "CUDAHelpers.cuh"
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>

// =========================================================================
// Inverse Frequency Cache (CPU-side, mirrors RoPEPrimitives.cpp)
// =========================================================================

namespace
{
    // Device memory cache for inverse frequencies
    struct InvFreqCache
    {
        float *d_inv_freq = nullptr;
        int head_dim = 0;
        float freq_base = 0.0f;
        int device_idx = -1;

        ~InvFreqCache()
        {
            if (d_inv_freq)
            {
                cudaFree(d_inv_freq);
            }
        }
    };

    // Global cache - one per (head_dim, freq_base, device) combination
    static std::unordered_map<uint64_t, InvFreqCache> g_inv_freq_cache;
    static std::mutex g_cache_mutex;

    uint64_t make_cache_key(int head_dim, float freq_base, int device_idx)
    {
        uint32_t freq_bits;
        std::memcpy(&freq_bits, &freq_base, sizeof(float));
        // Pack: device_idx (8 bits) | head_dim (24 bits) | freq_bits (32 bits)
        return ((uint64_t)(device_idx & 0xFF) << 56) | ((uint64_t)(head_dim & 0xFFFFFF) << 32) | freq_bits;
    }

    /**
     * @brief Get or create cached inverse frequency table on device
     * @return Device pointer to inv_freq array of size head_dim/2
     */
    float *get_inv_freq_device(int head_dim, float freq_base, int device_idx)
    {
        uint64_t key = make_cache_key(head_dim, freq_base, device_idx);

        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto it = g_inv_freq_cache.find(key);
        if (it != g_inv_freq_cache.end())
        {
            return it->second.d_inv_freq;
        }

        // Compute inverse frequencies on host
        const int half_dim = head_dim / 2;
        std::vector<float> h_inv_freq(half_dim);
        const float log_base = std::log(freq_base);

        for (int i = 0; i < half_dim; ++i)
        {
            float exponent = (2.0f * i) / head_dim;
            h_inv_freq[i] = std::exp(-log_base * exponent);
        }

        // Allocate and copy to device
        cudaSetDevice(device_idx);
        float *d_inv_freq = nullptr;
        cudaMalloc(&d_inv_freq, half_dim * sizeof(float));
        cudaMemcpy(d_inv_freq, h_inv_freq.data(), half_dim * sizeof(float), cudaMemcpyHostToDevice);

        // Cache it
        InvFreqCache cache;
        cache.d_inv_freq = d_inv_freq;
        cache.head_dim = head_dim;
        cache.freq_base = freq_base;
        cache.device_idx = device_idx;
        g_inv_freq_cache[key] = std::move(cache);

        return d_inv_freq;
    }
} // anonymous namespace

// =========================================================================
// Optimized RoPE CUDA Kernels (v3 - with precomputed inv_freq)
// =========================================================================

// Maximum head_dim/2 we support in shared memory (covers up to head_dim=256)
constexpr int MAX_HALF_DIM = 128;

/**
 * @brief FP32 RoPE kernel with shared memory sin/cos table
 *
 * Strategy (adapted from CPU RoPEPrimitives.cpp):
 * 1. Each thread block loads inv_freq into shared memory once
 * 2. For each position in the block, compute sin/cos table in shared memory
 * 3. Apply rotation using cached sin/cos (pure FMA, no transcendentals)
 *
 * This kernel processes one (seq_idx, head_idx) pair per thread block.
 * Threads within the block cooperatively process the half_dim pairs.
 */
__global__ void rope_fp32_kernel_v3(
    float *__restrict__ data,
    const float *__restrict__ inv_freq, // Pre-computed inverse frequencies [half_dim]
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    // Shared memory for sin/cos cache (one set per position in this block)
    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    // Each block handles one (seq_idx, head_idx) pair
    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    // Get actual position
    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Step 1: Cooperatively compute sin/cos table into shared memory
    // Each thread handles multiple pairs if half_dim > blockDim.x
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Step 2: Apply rotation using cached sin/cos (pure FMA)
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = data[i0];
        float x1 = data[i1];
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = x0 * cos_val - x1 * sin_val;
        data[i1] = x0 * sin_val + x1 * cos_val;
    }
}

/**
 * @brief Fused FP32 RoPE kernel for Q and K in single launch
 *
 * Processes both Q (first n_q_heads blocks) and K (next n_kv_heads blocks)
 * in a single kernel launch, reducing overhead by 50%.
 */
__global__ void rope_fp32_fused_qk_kernel(
    float *__restrict__ Q,
    float *__restrict__ K, // Can be nullptr
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    // Total blocks: seq_len * n_q_heads (for Q) + seq_len * n_kv_heads (for K)
    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    float *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        // This block processes Q
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        // This block processes K
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = data[i0];
        float x1 = data[i1];

        data[i0] = x0 * s_cos[i] - x1 * s_sin[i];
        data[i1] = x0 * s_sin[i] + x1 * s_cos[i];
    }
}

/**
 * @brief BF16 RoPE kernel with shared memory sin/cos table
 */
__global__ void rope_bf16_kernel_v3(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = float_to_bf16(x0 * cos_val - x1 * sin_val);
        data[i1] = float_to_bf16(x0 * sin_val + x1 * cos_val);
    }
}

/**
 * @brief Fused BF16 RoPE kernel for Q and K in single launch
 */
__global__ void rope_bf16_fused_qk_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);

        data[i0] = float_to_bf16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_bf16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief FP16 RoPE kernel with shared memory sin/cos table
 */
__global__ void rope_fp16_kernel_v3(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = float_to_fp16(x0 * cos_val - x1 * sin_val);
        data[i1] = float_to_fp16(x0 * sin_val + x1 * cos_val);
    }
}

/**
 * @brief Fused FP16 RoPE kernel for Q and K in single launch
 */
__global__ void rope_fp16_fused_qk_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);

        data[i0] = float_to_fp16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_fp16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    bool cudaOps_rope_fp32(
        float *Q,
        float *K, // Can be nullptr
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        cudaSetDevice(device_idx);

        // Get cached inverse frequency table
        float *d_inv_freq = get_inv_freq_device(head_dim, rope_theta, device_idx);

        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);      // At least one thread per pair
        const size_t smem_size = 2 * half_dim * sizeof(float); // sin + cos tables

        if (K != nullptr)
        {
            // Fused Q+K kernel - single launch for both
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            rope_fp32_fused_qk_kernel<<<total_blocks, threads_per_block, smem_size>>>(
                Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
        }
        else
        {
            // Q only
            int num_blocks_q = seq_len * n_heads;
            rope_fp32_kernel_v3<<<num_blocks_q, threads_per_block, smem_size>>>(
                Q, d_inv_freq, position_ids, seq_len, n_heads, head_dim);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RoPE FP32 v3 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_rope_bf16(
        uint16_t *Q,
        uint16_t *K, // Can be nullptr
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        cudaSetDevice(device_idx);

        float *d_inv_freq = get_inv_freq_device(head_dim, rope_theta, device_idx);

        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            rope_bf16_fused_qk_kernel<<<total_blocks, threads_per_block, smem_size>>>(
                Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
        }
        else
        {
            int num_blocks_q = seq_len * n_heads;
            rope_bf16_kernel_v3<<<num_blocks_q, threads_per_block, smem_size>>>(
                Q, d_inv_freq, position_ids, seq_len, n_heads, head_dim);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RoPE BF16 v3 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool cudaOps_rope_fp16(
        uint16_t *Q,
        uint16_t *K, // Can be nullptr
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        cudaSetDevice(device_idx);

        float *d_inv_freq = get_inv_freq_device(head_dim, rope_theta, device_idx);

        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            rope_fp16_fused_qk_kernel<<<total_blocks, threads_per_block, smem_size>>>(
                Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
        }
        else
        {
            int num_blocks_q = seq_len * n_heads;
            rope_fp16_kernel_v3<<<num_blocks_q, threads_per_block, smem_size>>>(
                Q, d_inv_freq, position_ids, seq_len, n_heads, head_dim);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RoPE FP16 v3 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
