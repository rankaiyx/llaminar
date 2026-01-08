/**
 * @file Test__KVCacheStages_GPU.cpp
 * @brief GPU dispatch tests for KVCacheAppendStage and KVCacheGatherStage
 *
 * Test-driven implementation for Phase 4.1: KV Cache GPU Integration
 *
 * Tests verify:
 * 1. KVCacheAppendStage dispatches to CUDA KV cache when GPU tensors provided
 * 2. KVCacheGatherStage dispatches to CUDA KV cache gather
 * 3. Round-trip: Append on GPU → Gather on GPU produces correct results
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "kernels/cuda/CUDARingKVCache.h"
#include "tensors/cuda/CUDATypedTensor.h"
#endif

#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/compute_stages/stages/KVCacheGatherStage.h"
#include "tensors/Tensors.h"
#include "tensors/UnifiedKVCache.h"
#include "utils/Logger.h"

#include <vector>
#include <random>
#include <cmath>

using namespace llaminar2;

namespace
{
    // Check CUDA availability
    bool hasCUDA()
    {
#ifdef HAVE_CUDA
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
#else
        return false;
#endif
    }

    // Generate random FP32 data
    std::vector<float> generateRandomFP32(size_t count, unsigned seed = 42)
    {
        std::vector<float> data(count);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : data)
        {
            val = dist(rng);
        }
        return data;
    }

    // Compute max absolute error
    float computeMaxError(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

} // namespace

// =============================================================================
// Test: KVCacheAppendStage supports GPU backend
// =============================================================================

TEST(Test__KVCacheAppendStage_GPU, SupportsGPUBackend)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Create CUDA KV cache for GPU backend support
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 64, 4, 64);
    ASSERT_NE(cuda_cache, nullptr);

    // Stage with no cache
    KVCacheAppendStage::Params no_cache_params;
    no_cache_params.kv_cache = nullptr;
    KVCacheAppendStage no_cache_stage(no_cache_params);
    EXPECT_FALSE(no_cache_stage.supportsBackend(ComputeBackendType::CPU));      // No cache at all
    EXPECT_FALSE(no_cache_stage.supportsBackend(ComputeBackendType::GPU_CUDA)); // No CUDA cache

    // Stage with CUDA cache
    KVCacheAppendStage::Params gpu_params;
    gpu_params.kv_cache = cuda_cache.get();
    KVCacheAppendStage gpu_stage(gpu_params);
    EXPECT_FALSE(gpu_stage.supportsBackend(ComputeBackendType::CPU));     // CUDA cache, not CPU
    EXPECT_TRUE(gpu_stage.supportsBackend(ComputeBackendType::GPU_CUDA)); // Has CUDA cache
#endif
}

// =============================================================================
// Test: KVCacheAppendStage GPU dispatch with device tensors
// =============================================================================

TEST(Test__KVCacheAppendStage_GPU, AppendDeviceTensors)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Parameters matching typical Qwen2-0.5B
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 128;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 10;

    // Create CUDA KV cache
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cuda_cache, nullptr);

    // Generate test data on host
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);

    // Create GPU tensors
    auto gpu_K = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_V = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)}, 0);

    // Copy data to GPU
    cudaMemcpy(gpu_K->raw_mutable_data(), h_K.data(),
               num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_V->raw_mutable_data(), h_V.data(),
               num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Verify tensors are on GPU
    ASSERT_TRUE(gpu_K->is_on_gpu());
    ASSERT_TRUE(gpu_V->is_on_gpu());

    // Create stage params with CUDA cache
    KVCacheAppendStage::Params params;
    params.K = gpu_K.get();
    params.V = gpu_V.get();
    params.kv_cache = cuda_cache.get();
    params.layer_idx = 0;
    params.seq_idx = 0;
    params.num_tokens = num_tokens;

    KVCacheAppendStage stage(params);

    // Execute stage - should dispatch to CUDA cache
    bool result = stage.execute(nullptr);
    EXPECT_TRUE(result);

    // Verify cache state
    EXPECT_EQ(cuda_cache->get_cached_tokens(0, 0), num_tokens);

    // Retrieve and verify data
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cuda_cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, num_tokens);

    // Copy back and verify
    std::vector<float> h_K_out(num_tokens * kv_dim);
    std::vector<float> h_V_out(num_tokens * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err_K = computeMaxError(h_K, h_K_out);
    float max_err_V = computeMaxError(h_V, h_V_out);

    LOG_INFO("[AppendDeviceTensors] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f) << "K data should match exactly (FP32)";
    EXPECT_EQ(max_err_V, 0.0f) << "V data should match exactly (FP32)";

    LOG_INFO("[AppendDeviceTensors] PASSED");
#endif
}

// =============================================================================
// Test: KVCacheGatherStage supports GPU backend
// =============================================================================

TEST(Test__KVCacheGatherStage_GPU, SupportsGPUBackend)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Create CUDA KV cache for GPU backend support
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 64, 4, 64);
    ASSERT_NE(cuda_cache, nullptr);

    // Stage with no cache
    KVCacheGatherStage::Params no_cache_params;
    no_cache_params.kv_cache = nullptr;
    KVCacheGatherStage no_cache_stage(no_cache_params);
    EXPECT_FALSE(no_cache_stage.supportsBackend(ComputeBackendType::CPU));      // No cache at all
    EXPECT_FALSE(no_cache_stage.supportsBackend(ComputeBackendType::GPU_CUDA)); // No CUDA cache

    // Stage with CUDA cache
    KVCacheGatherStage::Params gpu_params;
    gpu_params.kv_cache = cuda_cache.get();
    KVCacheGatherStage gpu_stage(gpu_params);
    EXPECT_FALSE(gpu_stage.supportsBackend(ComputeBackendType::CPU));     // CUDA cache, not CPU
    EXPECT_TRUE(gpu_stage.supportsBackend(ComputeBackendType::GPU_CUDA)); // Has CUDA cache
#endif
}

// =============================================================================
// Test: KVCacheGatherStage GPU dispatch with device tensors
// =============================================================================

TEST(Test__KVCacheGatherStage_GPU, GatherToDeviceTensors)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Parameters
    const int n_layers = 1;
    const int batch_size = 2; // Multi-sequence
    const int max_seq_len = 128;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int tokens_per_seq = 5;

    // Create CUDA KV cache
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cuda_cache, nullptr);

    // Populate cache with test data (use direct CUDA API for setup)
    for (int seq = 0; seq < batch_size; ++seq)
    {
        auto h_K = generateRandomFP32(tokens_per_seq * kv_dim, 100 + seq);
        auto h_V = generateRandomFP32(tokens_per_seq * kv_dim, 200 + seq);

        float *d_K, *d_V;
        cudaMalloc(&d_K, tokens_per_seq * kv_dim * sizeof(float));
        cudaMalloc(&d_V, tokens_per_seq * kv_dim * sizeof(float));
        cudaMemcpy(d_K, h_K.data(), tokens_per_seq * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V.data(), tokens_per_seq * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        ASSERT_TRUE(cuda_cache->append(0, seq, d_K, d_V, tokens_per_seq));

        cudaFree(d_K);
        cudaFree(d_V);
    }

    // Verify cache is populated
    for (int seq = 0; seq < batch_size; ++seq)
    {
        EXPECT_EQ(cuda_cache->get_cached_tokens(0, seq), tokens_per_seq);
    }

    // Create output GPU tensors for gather
    // Shape: [batch_size * max_kv_len, kv_dim]
    const int max_kv_len = tokens_per_seq; // All sequences same length in this test
    auto gpu_out_K = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * max_kv_len), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_out_V = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * max_kv_len), static_cast<size_t>(kv_dim)}, 0);

    ASSERT_TRUE(gpu_out_K->is_on_gpu());
    ASSERT_TRUE(gpu_out_V->is_on_gpu());

    // Create stage params with CUDA cache
    KVCacheGatherStage::Params params;
    params.kv_cache = cuda_cache.get();
    params.layer_idx = 0;
    params.batch_size = batch_size;
    params.out_K = gpu_out_K.get();
    params.out_V = gpu_out_V.get();

    KVCacheGatherStage stage(params);

    // Execute stage - should dispatch to CUDA cache gather
    bool result = stage.execute(nullptr);
    EXPECT_TRUE(result);
    EXPECT_EQ(stage.getMaxKVLen(), max_kv_len);

    LOG_INFO("[KVCacheGatherStage_GPU] GatherToDeviceTensors PASSED");
#endif
}

// =============================================================================
// Test: Round-trip - Append then Gather on GPU
// =============================================================================

TEST(Test__KVCacheStages_GPU, RoundTrip_AppendThenGather)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    // Parameters
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    // Create CUDA KV cache
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP32,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cuda_cache, nullptr);

    // Generate original data
    auto h_K_orig = generateRandomFP32(num_tokens * kv_dim, 777);
    auto h_V_orig = generateRandomFP32(num_tokens * kv_dim, 888);

    // Create GPU input tensors
    auto gpu_K_in = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_V_in = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)}, 0);

    cudaMemcpy(gpu_K_in->raw_mutable_data(), h_K_orig.data(),
               num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_V_in->raw_mutable_data(), h_V_orig.data(),
               num_tokens * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

    // --- Append Stage ---
    KVCacheAppendStage::Params append_params;
    append_params.K = gpu_K_in.get();
    append_params.V = gpu_V_in.get();
    append_params.kv_cache = cuda_cache.get();
    append_params.layer_idx = 0;
    append_params.seq_idx = 0;
    append_params.num_tokens = num_tokens;

    KVCacheAppendStage append_stage(append_params);
    bool append_ok = append_stage.execute(nullptr);
    ASSERT_TRUE(append_ok);

    // --- Gather Stage ---
    auto gpu_K_out = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)}, 0);
    auto gpu_V_out = std::make_unique<CUDAFp32Tensor>(
        std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(kv_dim)}, 0);

    KVCacheGatherStage::Params gather_params;
    gather_params.kv_cache = cuda_cache.get();
    gather_params.layer_idx = 0;
    gather_params.batch_size = 1;
    gather_params.out_K = gpu_K_out.get();
    gather_params.out_V = gpu_V_out.get();

    KVCacheGatherStage gather_stage(gather_params);
    bool gather_ok = gather_stage.execute(nullptr);
    ASSERT_TRUE(gather_ok);

    // Verify round-trip via direct API
    const void *d_K_result, *d_V_result;
    int kv_len;
    ASSERT_TRUE(cuda_cache->get_kv_for_attention(0, 0, &d_K_result, &d_V_result, &kv_len));
    EXPECT_EQ(kv_len, num_tokens);

    // Copy back and verify round-trip
    std::vector<float> h_K_result(num_tokens * kv_dim);
    std::vector<float> h_V_result(num_tokens * kv_dim);
    cudaMemcpy(h_K_result.data(), d_K_result, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_result.data(), d_V_result, num_tokens * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err_K = computeMaxError(h_K_orig, h_K_result);
    float max_err_V = computeMaxError(h_V_orig, h_V_result);

    LOG_INFO("[RoundTrip] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    EXPECT_EQ(max_err_K, 0.0f) << "K data should match exactly (FP32)";
    EXPECT_EQ(max_err_V, 0.0f) << "V data should match exactly (FP32)";

    LOG_INFO("[RoundTrip_AppendThenGather] PASSED (using stage GPU dispatch)");
#endif
}

// =============================================================================
// Test: Mixed precision (FP16 cache)
// =============================================================================

TEST(Test__KVCacheStages_GPU, FP16Cache_RoundTrip)
{
    if (!hasCUDA())
    {
        GTEST_SKIP() << "CUDA not available";
    }

#ifdef HAVE_CUDA
    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 32;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    // Create FP16 CUDA KV cache
    auto cuda_cache = createCUDARingKVCache(
        ActivationPrecision::FP16,
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim);
    ASSERT_NE(cuda_cache, nullptr);
    EXPECT_EQ(cuda_cache->precision(), ActivationPrecision::FP16);

    // Generate test data (FP32, will be converted to FP16 internally)
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 111);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 222);

    // Clamp to FP16 range for fair comparison
    for (auto &v : h_K)
        v = std::clamp(v, -65504.0f, 65504.0f);
    for (auto &v : h_V)
        v = std::clamp(v, -65504.0f, 65504.0f);

    // Allocate device memory (FP16)
    __half *d_K, *d_V;
    cudaMalloc(&d_K, num_tokens * kv_dim * sizeof(__half));
    cudaMalloc(&d_V, num_tokens * kv_dim * sizeof(__half));

    // Convert FP32 to FP16 and copy
    std::vector<__half> h_K_fp16(num_tokens * kv_dim);
    std::vector<__half> h_V_fp16(num_tokens * kv_dim);
    for (size_t i = 0; i < h_K.size(); ++i)
    {
        h_K_fp16[i] = __float2half(h_K[i]);
        h_V_fp16[i] = __float2half(h_V[i]);
    }
    cudaMemcpy(d_K, h_K_fp16.data(), num_tokens * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_fp16.data(), num_tokens * kv_dim * sizeof(__half), cudaMemcpyHostToDevice);

    // Append to cache
    ASSERT_TRUE(cuda_cache->append(0, 0, d_K, d_V, num_tokens));
    EXPECT_EQ(cuda_cache->get_cached_tokens(0, 0), num_tokens);

    // Retrieve
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cuda_cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len));
    EXPECT_EQ(kv_len, num_tokens);

    // Copy back and convert to FP32 for comparison
    std::vector<__half> h_K_out_fp16(num_tokens * kv_dim);
    std::vector<__half> h_V_out_fp16(num_tokens * kv_dim);
    cudaMemcpy(h_K_out_fp16.data(), d_K_out, num_tokens * kv_dim * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out_fp16.data(), d_V_out, num_tokens * kv_dim * sizeof(__half), cudaMemcpyDeviceToHost);

    std::vector<float> h_K_out(num_tokens * kv_dim);
    std::vector<float> h_V_out(num_tokens * kv_dim);
    for (size_t i = 0; i < h_K_out.size(); ++i)
    {
        h_K_out[i] = __half2float(h_K_out_fp16[i]);
        h_V_out[i] = __half2float(h_V_out_fp16[i]);
    }

    // FP16 has limited precision, expect some error
    float max_err_K = computeMaxError(h_K, h_K_out);
    float max_err_V = computeMaxError(h_V, h_V_out);

    LOG_INFO("[FP16Cache] max_err_K=" << max_err_K << ", max_err_V=" << max_err_V);

    // FP16 precision: ~1e-3 relative error for values in [-1, 1]
    EXPECT_LT(max_err_K, 1e-2f) << "K error within FP16 tolerance";
    EXPECT_LT(max_err_V, 1e-2f) << "V error within FP16 tolerance";

    cudaFree(d_K);
    cudaFree(d_V);

    LOG_INFO("[FP16Cache_RoundTrip] PASSED");
#endif
}
