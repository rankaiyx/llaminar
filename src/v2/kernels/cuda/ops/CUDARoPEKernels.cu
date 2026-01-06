/**
 * @file CUDARoPEKernels.cu
 * @brief CUDA RoPE (Rotary Position Embedding) kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32, BF16, and FP16 RoPE kernels with extern "C" wrapper functions.
 * Uses SPLIT-HALF layout matching the CPU implementation.
 */

#include "CUDAHelpers.cuh"
#include <cmath>
#include <cstdio>

// =========================================================================
// RoPE CUDA Kernels
// =========================================================================

/**
 * @brief FP32 RoPE kernel - applies rotary position embeddings
 *
 * Each thread handles one pair of dimensions for one head at one position.
 * Layout: [seq_len, n_heads, head_dim] with SPLIT-HALF pairing:
 *   - pair(i) = (data[i], data[i + half_dim]) for i in 0..half_dim-1
 * This matches the CPU implementation in RoPEPrimitives.cpp
 */
__global__ void rope_fp32_kernel(
    float *__restrict__ data,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    float rope_theta)
{
    // Each thread handles one (position, head, dim_pair) triplet
    const int half_dim = head_dim / 2;
    int total_pairs = seq_len * n_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs)
        return;

    // Decode indices
    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % n_heads;
    int seq_idx = idx / (n_heads * half_dim);

    // Get actual position
    int pos = position_ids ? position_ids[seq_idx] : seq_idx;

    // Skip padding positions
    if (pos < 0)
        return;

    // Compute frequency: inv_freq[i] = pow(rope_theta, -2*i/head_dim)
    float freq = powf(rope_theta, -2.0f * pair_idx / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Get data indices using SPLIT-HALF layout (matching CPU)
    // First element at pair_idx, second element at pair_idx + half_dim
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int i0 = base_idx + pair_idx;            // First half
    int i1 = base_idx + pair_idx + half_dim; // Second half

    // Apply rotation
    float x0 = data[i0];
    float x1 = data[i1];
    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
}

/**
 * @brief BF16 RoPE kernel - FP32 computation with BF16 I/O
 * Uses SPLIT-HALF layout matching CPU implementation
 */
__global__ void rope_bf16_kernel(
    uint16_t *__restrict__ data,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    float rope_theta)
{
    const int half_dim = head_dim / 2;
    int total_pairs = seq_len * n_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs)
        return;

    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % n_heads;
    int seq_idx = idx / (n_heads * half_dim);

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    float freq = powf(rope_theta, -2.0f * pair_idx / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // SPLIT-HALF layout (matching CPU)
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int i0 = base_idx + pair_idx;            // First half
    int i1 = base_idx + pair_idx + half_dim; // Second half

    float x0 = bf16_to_float(data[i0]);
    float x1 = bf16_to_float(data[i1]);
    data[i0] = float_to_bf16(x0 * cos_val - x1 * sin_val);
    data[i1] = float_to_bf16(x0 * sin_val + x1 * cos_val);
}

/**
 * @brief FP16 RoPE kernel - FP32 computation with FP16 I/O
 * Uses SPLIT-HALF layout matching CPU implementation
 */
__global__ void rope_fp16_kernel(
    uint16_t *__restrict__ data,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    float rope_theta)
{
    const int half_dim = head_dim / 2;
    int total_pairs = seq_len * n_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs)
        return;

    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % n_heads;
    int seq_idx = idx / (n_heads * half_dim);

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    float freq = powf(rope_theta, -2.0f * pair_idx / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // SPLIT-HALF layout (matching CPU)
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int i0 = base_idx + pair_idx;            // First half
    int i1 = base_idx + pair_idx + half_dim; // Second half

    float x0 = fp16_to_float(data[i0]);
    float x1 = fp16_to_float(data[i1]);
    data[i0] = float_to_fp16(x0 * cos_val - x1 * sin_val);
    data[i1] = float_to_fp16(x0 * sin_val + x1 * cos_val);
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

        int threads_per_block = 256;

        // Process Q
        int total_pairs_q = seq_len * n_heads * (head_dim / 2);
        int num_blocks_q = (total_pairs_q + threads_per_block - 1) / threads_per_block;
        rope_fp32_kernel<<<num_blocks_q, threads_per_block>>>(
            Q, position_ids, seq_len, n_heads, head_dim, rope_theta);

        // Process K if provided
        if (K != nullptr)
        {
            int total_pairs_k = seq_len * n_kv_heads * (head_dim / 2);
            int num_blocks_k = (total_pairs_k + threads_per_block - 1) / threads_per_block;
            rope_fp32_kernel<<<num_blocks_k, threads_per_block>>>(
                K, position_ids, seq_len, n_kv_heads, head_dim, rope_theta);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RoPE FP32 kernel launch failed: %s\n", cudaGetErrorString(err));
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

        int threads_per_block = 256;

        int total_pairs_q = seq_len * n_heads * (head_dim / 2);
        int num_blocks_q = (total_pairs_q + threads_per_block - 1) / threads_per_block;
        rope_bf16_kernel<<<num_blocks_q, threads_per_block>>>(
            Q, position_ids, seq_len, n_heads, head_dim, rope_theta);

        if (K != nullptr)
        {
            int total_pairs_k = seq_len * n_kv_heads * (head_dim / 2);
            int num_blocks_k = (total_pairs_k + threads_per_block - 1) / threads_per_block;
            rope_bf16_kernel<<<num_blocks_k, threads_per_block>>>(
                K, position_ids, seq_len, n_kv_heads, head_dim, rope_theta);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RoPE BF16 kernel launch failed: %s\n", cudaGetErrorString(err));
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

        int threads_per_block = 256;

        int total_pairs_q = seq_len * n_heads * (head_dim / 2);
        int num_blocks_q = (total_pairs_q + threads_per_block - 1) / threads_per_block;
        rope_fp16_kernel<<<num_blocks_q, threads_per_block>>>(
            Q, position_ids, seq_len, n_heads, head_dim, rope_theta);

        if (K != nullptr)
        {
            int total_pairs_k = seq_len * n_kv_heads * (head_dim / 2);
            int num_blocks_k = (total_pairs_k + threads_per_block - 1) / threads_per_block;
            rope_fp16_kernel<<<num_blocks_k, threads_per_block>>>(
                K, position_ids, seq_len, n_kv_heads, head_dim, rope_theta);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA RoPE FP16 kernel launch failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
