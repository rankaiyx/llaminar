/**
 * @file Test__AttentionWithKVCacheStage_GPU.cpp
 * @brief GPU dispatch tests for AttentionWithKVCacheStage
 *
 * Tests GPU path using CUDAFlashAttentionKernelT with CUDA Ring KV Cache.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

#include "execution/compute_stages/stages/AttentionWithKVCacheStage.h"
#include "tensors/cuda/CUDATypedTensor.h"
#include "kernels/cuda/CUDARingKVCache.h"
#include "utils/Logger.h"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace llaminar2;

namespace
{
    // =============================================================================
    // Helper Functions
    // =============================================================================

    bool hasCUDA()
    {
#ifdef HAVE_CUDA
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
#else
        return false;
#endif
    }

    std::vector<float> generateRandomFP32(size_t count, unsigned seed)
    {
        std::vector<float> data(count);
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : data)
        {
            v = dist(gen);
        }
        return data;
    }

    float computeMaxError(const std::vector<float> &a, const std::vector<float> &b)
    {
        if (a.size() != b.size())
            return std::numeric_limits<float>::infinity();
        float max_err = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

    // Compute reference attention (CPU, single head for simplicity)
    void computeReferenceAttention(
        const float *Q, const float *K, const float *V,
        float *output,
        int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal)
    {
        // Simple reference: for each head, compute QK^T, softmax, then V
        const int gqa_ratio = n_heads / n_kv_heads;

        for (int h = 0; h < n_heads; ++h)
        {
            const int kv_h = h / gqa_ratio;

            for (int q = 0; q < seq_len; ++q)
            {
                // Compute attention scores for this query position
                std::vector<float> scores(kv_len);
                float max_score = -std::numeric_limits<float>::infinity();

                for (int k = 0; k < kv_len; ++k)
                {
                    if (causal && k > q)
                    {
                        scores[k] = -std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                        {
                            float q_val = Q[q * n_heads * head_dim + h * head_dim + d];
                            float k_val = K[k * n_kv_heads * head_dim + kv_h * head_dim + d];
                            dot += q_val * k_val;
                        }
                        scores[k] = dot / std::sqrt(static_cast<float>(head_dim));
                    }
                    max_score = std::max(max_score, scores[k]);
                }

                // Softmax
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_len; ++k)
                {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum_exp += scores[k];
                }
                for (int k = 0; k < kv_len; ++k)
                {
                    scores[k] /= sum_exp;
                }

                // Output = scores @ V
                for (int d = 0; d < head_dim; ++d)
                {
                    float out_val = 0.0f;
                    for (int k = 0; k < kv_len; ++k)
                    {
                        float v_val = V[k * n_kv_heads * head_dim + kv_h * head_dim + d];
                        out_val += scores[k] * v_val;
                    }
                    output[q * n_heads * head_dim + h * head_dim + d] = out_val;
                }
            }
        }
    }

} // anonymous namespace

// =============================================================================
// Test: AttentionWithKVCacheStage supports GPU backend
// =============================================================================

TEST(Test__AttentionWithKVCacheStage_GPU, SupportsGPUBackend)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Create CUDA KV cache
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 64, 4, 64);
    ASSERT_NE(cuda_cache, nullptr);

    // Stage with no cache - supports CPU (cache-free attention)
    AttentionWithKVCacheStage::Params no_cache_params;
    no_cache_params.kv_cache = nullptr;
    AttentionWithKVCacheStage no_cache_stage(no_cache_params);
    EXPECT_TRUE(no_cache_stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_FALSE(no_cache_stage.supportsBackend(ComputeBackendType::GPU_CUDA)); // No CUDA cache

    // Stage with CUDA cache - supports GPU only
    AttentionWithKVCacheStage::Params gpu_params;
    gpu_params.kv_cache = cuda_cache.get();
    AttentionWithKVCacheStage gpu_stage(gpu_params);
    EXPECT_FALSE(gpu_stage.supportsBackend(ComputeBackendType::CPU));     // CUDA cache, not CPU
    EXPECT_TRUE(gpu_stage.supportsBackend(ComputeBackendType::GPU_CUDA)); // Has CUDA cache
#endif
}

// =============================================================================
// Test: Prefill mode with GPU tensors and CUDA KV cache
// =============================================================================

TEST(Test__AttentionWithKVCacheStage_GPU, PrefillMode_GPUTensors)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Model parameters (Qwen2-0.5B-like)
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 128;
    const int seq_len = 8;
    const int n_heads = 4;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    // Create CUDA KV cache
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cuda_cache, nullptr);

    // Generate test data on host
    auto h_Q = generateRandomFP32(seq_len * q_dim, 111);
    auto h_K = generateRandomFP32(seq_len * kv_dim, 222);
    auto h_V = generateRandomFP32(seq_len * kv_dim, 333);

    // Create GPU tensors
    auto gpu_Q = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)}, 0);
    auto gpu_K = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_V = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_output = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(q_dim)}, 0);

    // Copy data to GPU
    cudaMemcpy(gpu_Q->raw_mutable_data(), h_Q.data(), seq_len * q_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_K->raw_mutable_data(), h_K.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_V->raw_mutable_data(), h_V.data(), seq_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Verify tensors are on GPU
    ASSERT_TRUE(gpu_Q->is_on_gpu());
    ASSERT_TRUE(gpu_K->is_on_gpu());
    ASSERT_TRUE(gpu_V->is_on_gpu());
    ASSERT_TRUE(gpu_output->is_on_gpu());

    // Create stage params with CUDA cache
    AttentionWithKVCacheStage::Params params;
    params.Q = gpu_Q.get();
    params.K = gpu_K.get();
    params.V = gpu_V.get();
    params.output = gpu_output.get();
    params.kv_cache = nullptr;
    params.kv_cache = cuda_cache.get();
    params.layer_idx = 0;
    params.mode = AttentionWithKVCacheStage::Mode::PREFILL;
    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.causal = true;
    params.device_idx = 0;

    AttentionWithKVCacheStage stage(params);

    // Execute - should dispatch to CUDA Flash Attention
    bool result = stage.execute(nullptr);
    EXPECT_TRUE(result);

    // Verify KV cache was populated
    EXPECT_EQ(cuda_cache->get_cached_tokens(0, 0), seq_len);

    // Copy output back and verify against reference
    std::vector<float> h_output(seq_len * q_dim);
    cudaMemcpy(h_output.data(), gpu_output->raw_data(), seq_len * q_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference attention
    std::vector<float> h_reference(seq_len * q_dim);
    computeReferenceAttention(h_Q.data(), h_K.data(), h_V.data(), h_reference.data(),
                              seq_len, seq_len, n_heads, n_kv_heads, head_dim, true);

    float max_err = computeMaxError(h_output, h_reference);
    LOG_INFO("[PrefillMode_GPUTensors] max_err=" << max_err);

    // Flash Attention should be accurate to ~1e-3 for FP32
    EXPECT_LT(max_err, 1e-2f) << "Attention output should match reference";

    LOG_INFO("[PrefillMode_GPUTensors] PASSED");
#endif
}

// =============================================================================
// Test: Decode mode with GPU tensors and CUDA KV cache
// =============================================================================

TEST(Test__AttentionWithKVCacheStage_GPU, DecodeMode_GPUTensors)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Model parameters
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 128;
    const int prefill_len = 16;
    const int n_heads = 4;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    // Create CUDA KV cache
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cuda_cache, nullptr);

    // Pre-populate KV cache with prefill data (via direct API for setup)
    auto h_K_prefill = generateRandomFP32(prefill_len * kv_dim, 444);
    auto h_V_prefill = generateRandomFP32(prefill_len * kv_dim, 555);

    float *d_K_prefill, *d_V_prefill;
    cudaMalloc(&d_K_prefill, prefill_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V_prefill, prefill_len * kv_dim * sizeof(float));
    cudaMemcpy(d_K_prefill, h_K_prefill.data(), prefill_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_prefill, h_V_prefill.data(), prefill_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    ASSERT_TRUE(cuda_cache->append(0, 0, d_K_prefill, d_V_prefill, prefill_len));
    EXPECT_EQ(cuda_cache->get_cached_tokens(0, 0), prefill_len);

    cudaFree(d_K_prefill);
    cudaFree(d_V_prefill);

    // Generate decode token data
    const int decode_len = 1;
    auto h_Q = generateRandomFP32(decode_len * q_dim, 666);
    auto h_K = generateRandomFP32(decode_len * kv_dim, 777);
    auto h_V = generateRandomFP32(decode_len * kv_dim, 888);

    // Create GPU tensors for decode
    auto gpu_Q = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(decode_len), static_cast<size_t>(q_dim)}, 0);
    auto gpu_K = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(decode_len), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_V = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(decode_len), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_output = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(decode_len), static_cast<size_t>(q_dim)}, 0);

    cudaMemcpy(gpu_Q->raw_mutable_data(), h_Q.data(), decode_len * q_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_K->raw_mutable_data(), h_K.data(), decode_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_V->raw_mutable_data(), h_V.data(), decode_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Create stage params for decode
    AttentionWithKVCacheStage::Params params;
    params.Q = gpu_Q.get();
    params.K = gpu_K.get();
    params.V = gpu_V.get();
    params.output = gpu_output.get();
    params.kv_cache = nullptr;
    params.kv_cache = cuda_cache.get();
    params.layer_idx = 0;
    params.mode = AttentionWithKVCacheStage::Mode::DECODE;
    params.batch_size = batch_size;
    params.seq_len = decode_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.causal = true;
    params.device_idx = 0;
    params.position_offset = prefill_len;

    AttentionWithKVCacheStage stage(params);

    // Execute - should dispatch to CUDA Flash Decoding
    bool result = stage.execute(nullptr);
    EXPECT_TRUE(result);

    // Verify KV cache grew by 1
    EXPECT_EQ(cuda_cache->get_cached_tokens(0, 0), prefill_len + 1);

    // Copy output back
    std::vector<float> h_output(decode_len * q_dim);
    cudaMemcpy(h_output.data(), gpu_output->raw_data(), decode_len * q_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify output is not all zeros (sanity check)
    float sum = 0.0f;
    for (float v : h_output)
        sum += std::abs(v);
    EXPECT_GT(sum, 0.0f) << "Output should not be all zeros";

    LOG_INFO("[DecodeMode_GPUTensors] PASSED - output sum=" << sum);
#endif
}

// =============================================================================
// Test: Multi-decode iterations
// =============================================================================

TEST(Test__AttentionWithKVCacheStage_GPU, MultiDecodeIterations)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int prefill_len = 8;
    const int decode_steps = 5;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;

    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cuda_cache, nullptr);

    // Pre-populate with prefill
    auto h_K_prefill = generateRandomFP32(prefill_len * kv_dim, 100);
    auto h_V_prefill = generateRandomFP32(prefill_len * kv_dim, 200);
    float *d_K, *d_V;
    cudaMalloc(&d_K, prefill_len * kv_dim * sizeof(float));
    cudaMalloc(&d_V, prefill_len * kv_dim * sizeof(float));
    cudaMemcpy(d_K, h_K_prefill.data(), prefill_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_prefill.data(), prefill_len * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    ASSERT_TRUE(cuda_cache->append(0, 0, d_K, d_V, prefill_len));
    cudaFree(d_K);
    cudaFree(d_V);

    // Run multiple decode steps
    for (int step = 0; step < decode_steps; ++step)
    {
        auto h_Q = generateRandomFP32(q_dim, 300 + step);
        auto h_K = generateRandomFP32(kv_dim, 400 + step);
        auto h_V = generateRandomFP32(kv_dim, 500 + step);

        auto gpu_Q = std::make_unique<CUDAFp32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(q_dim)}, 0);
        auto gpu_K = std::make_unique<CUDAFp32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, 0);
        auto gpu_V = std::make_unique<CUDAFp32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(kv_dim)}, 0);
        auto gpu_output = std::make_unique<CUDAFp32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(q_dim)}, 0);

        cudaMemcpy(gpu_Q->raw_mutable_data(), h_Q.data(), q_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_K->raw_mutable_data(), h_K.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_V->raw_mutable_data(), h_V.data(), kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        AttentionWithKVCacheStage::Params params;
        params.Q = gpu_Q.get();
        params.K = gpu_K.get();
        params.V = gpu_V.get();
        params.output = gpu_output.get();
        params.kv_cache = cuda_cache.get();
        params.layer_idx = 0;
        params.mode = AttentionWithKVCacheStage::Mode::DECODE;
        params.batch_size = 1;
        params.seq_len = 1;
        params.n_heads = n_heads;
        params.n_kv_heads = n_kv_heads;
        params.head_dim = head_dim;
        params.causal = true;
        params.device_idx = 0;
        params.position_offset = prefill_len + step;

        AttentionWithKVCacheStage stage(params);
        bool result = stage.execute(nullptr);
        EXPECT_TRUE(result) << "Decode step " << step << " failed";

        int expected_len = prefill_len + step + 1;
        EXPECT_EQ(cuda_cache->get_cached_tokens(0, 0), expected_len)
            << "Cache length mismatch at step " << step;
    }

    LOG_INFO("[MultiDecodeIterations] PASSED - " << decode_steps << " decode steps");
#endif
}
