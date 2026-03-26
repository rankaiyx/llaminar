/**
 * @file Test__CPURingKVCache_TurboQuant.cpp
 * @brief Tests for TQ4 and TQ3 KV cache precision modes in CPURingKVCache.
 *
 * Validates the full round-trip: FP32 → TQ4/TQ3 quantize → ring buffer append →
 * ring buffer gather → TQ4/TQ3 dequantize → FP32. Ensures the ring buffer mechanics
 * (wrap-around, eviction, multi-layer) work correctly with block-quantized data.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>

#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "tensors/TQ4Tensor.h"
#include "tensors/TQ3Tensor.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ─────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────

class Test__CPURingKVCache_TurboQuant : public ::testing::Test
{
  protected:
    static constexpr int HEAD_DIM = 64;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 128

    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};

    std::unique_ptr<TurboQuantContext> turboquant_ctx_;

    void SetUp() override
    {
        turboquant_ctx_ = std::make_unique<TurboQuantContext>(HEAD_DIM, /*rotation_seed=*/42, /*projection_seed=*/42);
    }

    /// Create a random FP32 tensor with shape [num_tokens, KV_DIM].
    static std::shared_ptr<FP32Tensor> makeRandomFP32(int num_tokens, unsigned seed)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(num_tokens), static_cast<size_t>(KV_DIM)});
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(rng);
        return t;
    }

    /// Quantize FP32 tensor to TQ4.
    std::shared_ptr<TQ4Tensor> quantizeTQ4(const FP32Tensor &src)
    {
        return TQ4Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Quantize FP32 tensor to TQ3.
    std::shared_ptr<TQ3Tensor> quantizeTQ3(const FP32Tensor &src)
    {
        return TQ3Tensor::quantize_from_fp32(
            src.data(), src.shape(), HEAD_DIM, *turboquant_ctx_);
    }

    /// Dequantize TQ4 tensor to a new FP32 buffer.
    std::vector<float> dequantizeTQ4(const TQ4Tensor &t)
    {
        std::vector<float> buf(t.rows() * KV_DIM);
        t.dequantize_to_fp32(buf.data(), *turboquant_ctx_);
        return buf;
    }

    /// Dequantize TQ3 tensor to a new FP32 buffer.
    std::vector<float> dequantizeTQ3(const TQ3Tensor &t)
    {
        std::vector<float> buf(t.rows() * KV_DIM);
        t.dequantize_to_fp32(buf.data(), *turboquant_ctx_);
        return buf;
    }

    /// Compute MSE between two float buffers.
    static double computeMSE(const float *a, const float *b, size_t n)
    {
        double acc = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            acc += d * d;
        }
        return acc / static_cast<double>(n);
    }
};

// ─────────────────────────────────────────────────────────────────────
// TQ4 tests
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_PositionMajor_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheTQ4 cache(mpi_ctx_, /*n_layers=*/1, /*batch_size=*/1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu(),
                            KVCacheLayoutMode::POSITION_MAJOR);

    // Create FP32 data, quantize, append to cache
    auto fp32_k = makeRandomFP32(N_TOKENS, 100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 200);
    auto tq4_k = quantizeTQ4(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(/*layer=*/0, /*seq=*/0, tq4_k.get(), tq4_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    // Gather from cache into fresh TQ4 output tensors
    auto out_k = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{MAX_SEQ, KV_DIM}, HEAD_DIM, DeviceId::cpu());
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{MAX_SEQ, KV_DIM}, HEAD_DIM, DeviceId::cpu());
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(/*layer=*/0, /*num_seq=*/1,
                                         out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);
    ASSERT_EQ(kv_lens.size(), 1u);
    EXPECT_EQ(kv_lens[0], N_TOKENS);

    // Dequantize both original and gathered, compare
    auto orig_k_fp32 = dequantizeTQ4(*tq4_k);
    auto orig_v_fp32 = dequantizeTQ4(*tq4_v);

    // Set rotation on gathered tensors so dequantize works
    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());

    // Dequantize gathered data — only first N_TOKENS rows are valid
    std::vector<float> gathered_k_fp32(N_TOKENS * KV_DIM);
    std::vector<float> gathered_v_fp32(N_TOKENS * KV_DIM);

    // Use to_fp32_row for per-row dequantization of the gathered output
    for (int r = 0; r < N_TOKENS; ++r)
    {
        out_k->to_fp32_row(r, gathered_k_fp32.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v_fp32.data() + r * KV_DIM);
    }

    // Round-trip through the ring cache should be bit-exact at the quantized level
    double mse_k = computeMSE(orig_k_fp32.data(), gathered_k_fp32.data(), N_TOKENS * KV_DIM);
    double mse_v = computeMSE(orig_v_fp32.data(), gathered_v_fp32.data(), N_TOKENS * KV_DIM);

    EXPECT_EQ(mse_k, 0.0) << "TQ4 K data changed after cache round-trip";
    EXPECT_EQ(mse_v, 0.0) << "TQ4 V data changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_HeadMajor_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 4;

    CPURingKVCacheTQ4 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu(),
                            KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, 300);
    auto fp32_v = makeRandomFP32(N_TOKENS, 400);
    auto tq4_k = quantizeTQ4(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq4_k.get(), tq4_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    auto out_k = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);

    auto orig_k_fp32 = dequantizeTQ4(*tq4_k);
    auto orig_v_fp32 = dequantizeTQ4(*tq4_v);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());

    std::vector<float> gathered_k(N_TOKENS * KV_DIM);
    std::vector<float> gathered_v(N_TOKENS * KV_DIM);
    for (int r = 0; r < N_TOKENS; ++r)
    {
        out_k->to_fp32_row(r, gathered_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v.data() + r * KV_DIM);
    }

    EXPECT_EQ(computeMSE(orig_k_fp32.data(), gathered_k.data(), N_TOKENS * KV_DIM), 0.0)
        << "HEAD_MAJOR TQ4 K data changed after cache round-trip";
    EXPECT_EQ(computeMSE(orig_v_fp32.data(), gathered_v.data(), N_TOKENS * KV_DIM), 0.0)
        << "HEAD_MAJOR TQ4 V data changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_RingWrap_PreservesNewestTokens)
{
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheTQ4 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Append 6 tokens (overflows a capacity-4 ring → oldest 2 evicted)
    auto fp32_k = makeRandomFP32(6, 500);
    auto fp32_v = makeRandomFP32(6, 600);
    auto tq4_k = quantizeTQ4(*fp32_k);
    auto tq4_v = quantizeTQ4(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq4_k.get(), tq4_v.get(), 6));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);

    // Gather — should contain tokens 2,3,4,5 in logical order
    auto out_k = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, MAX_SEQ);

    // Directly quantize only the last 4 tokens as reference
    auto ref_k_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, KV_DIM});
    auto ref_v_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, KV_DIM});
    std::memcpy(ref_k_slice->mutable_data(), fp32_k->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    std::memcpy(ref_v_slice->mutable_data(), fp32_v->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    auto ref_tq4_k = quantizeTQ4(*ref_k_slice);
    auto ref_tq4_v = quantizeTQ4(*ref_v_slice);
    auto ref_k_fp32 = dequantizeTQ4(*ref_tq4_k);
    auto ref_v_fp32 = dequantizeTQ4(*ref_tq4_v);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> gathered_k(MAX_SEQ * KV_DIM), gathered_v(MAX_SEQ * KV_DIM);
    for (int r = 0; r < MAX_SEQ; ++r)
    {
        out_k->to_fp32_row(r, gathered_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v.data() + r * KV_DIM);
    }

    EXPECT_EQ(computeMSE(ref_k_fp32.data(), gathered_k.data(), MAX_SEQ * KV_DIM), 0.0)
        << "After ring wrap, gathered TQ4 K data doesn't match expected newest tokens";
    EXPECT_EQ(computeMSE(ref_v_fp32.data(), gathered_v.data(), MAX_SEQ * KV_DIM), 0.0)
        << "After ring wrap, gathered TQ4 V data doesn't match expected newest tokens";
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_IncrementalAppend_DecodeLike)
{
    constexpr int MAX_SEQ = 16;

    CPURingKVCacheTQ4 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Simulate prefill (5 tokens) + 3 decode steps (1 token each)
    auto prefill_k = makeRandomFP32(5, 700);
    auto prefill_v = makeRandomFP32(5, 800);
    ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ4(*prefill_k).get(),
                                quantizeTQ4(*prefill_v).get(), 5));
    EXPECT_EQ(cache.ring_size(0, 0), 5);

    for (int step = 0; step < 3; ++step)
    {
        auto dec_k = makeRandomFP32(1, 900 + step);
        auto dec_v = makeRandomFP32(1, 1000 + step);
        ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ4(*dec_k).get(),
                                    quantizeTQ4(*dec_v).get(), 1));
    }
    EXPECT_EQ(cache.ring_size(0, 0), 8); // 5 prefill + 3 decode

    // Gather and verify no NaN/Inf
    auto out_k = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ4Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, 8);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> fp32_k(8 * KV_DIM), fp32_v(8 * KV_DIM);
    for (int r = 0; r < 8; ++r)
    {
        out_k->to_fp32_row(r, fp32_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, fp32_v.data() + r * KV_DIM);
    }

    for (size_t i = 0; i < fp32_k.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(fp32_k[i])) << "NaN in gathered K at index " << i;
        ASSERT_FALSE(std::isinf(fp32_k[i])) << "Inf in gathered K at index " << i;
        ASSERT_FALSE(std::isnan(fp32_v[i])) << "NaN in gathered V at index " << i;
        ASSERT_FALSE(std::isinf(fp32_v[i])) << "Inf in gathered V at index " << i;
    }
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_MultiLayer_IndependentData)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_LAYERS = 3;

    CPURingKVCacheTQ4 cache(mpi_ctx_, N_LAYERS, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Append different data to each layer
    for (int l = 0; l < N_LAYERS; ++l)
    {
        auto k = makeRandomFP32(2, 1100 + l * 100);
        auto v = makeRandomFP32(2, 1200 + l * 100);
        ASSERT_TRUE(cache.append_kv(l, 0, quantizeTQ4(*k).get(),
                                    quantizeTQ4(*v).get(), 2));
        EXPECT_EQ(cache.ring_size(l, 0), 2);
    }

    // Gather each layer and verify they're different
    std::vector<std::vector<float>> layer_k_data(N_LAYERS);
    for (int l = 0; l < N_LAYERS; ++l)
    {
        auto out_k = std::make_shared<TQ4Tensor>(
            std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
        auto out_v = std::make_shared<TQ4Tensor>(
            std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
        std::vector<int> kv_lens;
        int max_kv = cache.gather_kv_batched(l, 1, out_k.get(), out_v.get(), kv_lens);
        ASSERT_EQ(max_kv, 2);

        out_k->set_turboquant_context(turboquant_ctx_.get());
        layer_k_data[l].resize(2 * KV_DIM);
        for (int r = 0; r < 2; ++r)
            out_k->to_fp32_row(r, layer_k_data[l].data() + r * KV_DIM);
    }

    // Verify layers contain different data (different seeds → different values)
    double mse_01 = computeMSE(layer_k_data[0].data(), layer_k_data[1].data(), 2 * KV_DIM);
    double mse_02 = computeMSE(layer_k_data[0].data(), layer_k_data[2].data(), 2 * KV_DIM);
    EXPECT_GT(mse_01, 0.0) << "Layer 0 and 1 should have different data";
    EXPECT_GT(mse_02, 0.0) << "Layer 0 and 2 should have different data";
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_Clear_ResetsAllLayers)
{
    CPURingKVCacheTQ4 cache(mpi_ctx_, 2, 1, 8, N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto k = makeRandomFP32(3, 1400);
    auto v = makeRandomFP32(3, 1500);
    ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ4(*k).get(), quantizeTQ4(*v).get(), 3));
    ASSERT_TRUE(cache.append_kv(1, 0, quantizeTQ4(*k).get(), quantizeTQ4(*v).get(), 3));

    EXPECT_EQ(cache.ring_size(0, 0), 3);
    EXPECT_EQ(cache.ring_size(1, 0), 3);

    cache.clear();

    EXPECT_EQ(cache.ring_size(0, 0), 0);
    EXPECT_EQ(cache.ring_size(1, 0), 0);
}

// ─────────────────────────────────────────────────────────────────────
// TQ3 tests
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_TurboQuant, TQ3_PositionMajor_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 3;

    CPURingKVCacheTQ3 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu(),
                            KVCacheLayoutMode::POSITION_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, 2100);
    auto fp32_v = makeRandomFP32(N_TOKENS, 2200);
    auto tq3_k = quantizeTQ3(*fp32_k);
    auto tq3_v = quantizeTQ3(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq3_k.get(), tq3_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    auto out_k = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);

    auto orig_k_fp32 = dequantizeTQ3(*tq3_k);
    auto orig_v_fp32 = dequantizeTQ3(*tq3_v);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> gathered_k(N_TOKENS * KV_DIM), gathered_v(N_TOKENS * KV_DIM);
    for (int r = 0; r < N_TOKENS; ++r)
    {
        out_k->to_fp32_row(r, gathered_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v.data() + r * KV_DIM);
    }

    EXPECT_EQ(computeMSE(orig_k_fp32.data(), gathered_k.data(), N_TOKENS * KV_DIM), 0.0)
        << "TQ3 K data changed after cache round-trip";
    EXPECT_EQ(computeMSE(orig_v_fp32.data(), gathered_v.data(), N_TOKENS * KV_DIM), 0.0)
        << "TQ3 V data changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ3_HeadMajor_AppendAndGather_RoundTrip)
{
    constexpr int MAX_SEQ = 8;
    constexpr int N_TOKENS = 4;

    CPURingKVCacheTQ3 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu(),
                            KVCacheLayoutMode::HEAD_MAJOR);

    auto fp32_k = makeRandomFP32(N_TOKENS, 2300);
    auto fp32_v = makeRandomFP32(N_TOKENS, 2400);
    auto tq3_k = quantizeTQ3(*fp32_k);
    auto tq3_v = quantizeTQ3(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq3_k.get(), tq3_v.get(), N_TOKENS));
    EXPECT_EQ(cache.ring_size(0, 0), N_TOKENS);

    auto out_k = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;

    int max_kv = cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);
    ASSERT_EQ(max_kv, N_TOKENS);

    auto orig_k_fp32 = dequantizeTQ3(*tq3_k);
    auto orig_v_fp32 = dequantizeTQ3(*tq3_v);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> gathered_k(N_TOKENS * KV_DIM), gathered_v(N_TOKENS * KV_DIM);
    for (int r = 0; r < N_TOKENS; ++r)
    {
        out_k->to_fp32_row(r, gathered_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v.data() + r * KV_DIM);
    }

    EXPECT_EQ(computeMSE(orig_k_fp32.data(), gathered_k.data(), N_TOKENS * KV_DIM), 0.0)
        << "HEAD_MAJOR TQ3 K data changed after cache round-trip";
    EXPECT_EQ(computeMSE(orig_v_fp32.data(), gathered_v.data(), N_TOKENS * KV_DIM), 0.0)
        << "HEAD_MAJOR TQ3 V data changed after cache round-trip";
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ3_RingWrap_PreservesNewestTokens)
{
    constexpr int MAX_SEQ = 4;

    CPURingKVCacheTQ3 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    auto fp32_k = makeRandomFP32(6, 2500);
    auto fp32_v = makeRandomFP32(6, 2600);
    auto tq3_k = quantizeTQ3(*fp32_k);
    auto tq3_v = quantizeTQ3(*fp32_v);

    ASSERT_TRUE(cache.append_kv(0, 0, tq3_k.get(), tq3_v.get(), 6));
    EXPECT_EQ(cache.ring_size(0, 0), MAX_SEQ);

    auto out_k = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;
    cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);

    // Reference: quantize only the last 4 tokens
    auto ref_k_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, KV_DIM});
    auto ref_v_slice = std::make_shared<FP32Tensor>(std::vector<size_t>{4, KV_DIM});
    std::memcpy(ref_k_slice->mutable_data(), fp32_k->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    std::memcpy(ref_v_slice->mutable_data(), fp32_v->data() + 2 * KV_DIM, 4 * KV_DIM * sizeof(float));
    auto ref_tq3_k = quantizeTQ3(*ref_k_slice);
    auto ref_tq3_v = quantizeTQ3(*ref_v_slice);
    auto ref_k_fp32 = dequantizeTQ3(*ref_tq3_k);
    auto ref_v_fp32 = dequantizeTQ3(*ref_tq3_v);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> gathered_k(MAX_SEQ * KV_DIM), gathered_v(MAX_SEQ * KV_DIM);
    for (int r = 0; r < MAX_SEQ; ++r)
    {
        out_k->to_fp32_row(r, gathered_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, gathered_v.data() + r * KV_DIM);
    }

    EXPECT_EQ(computeMSE(ref_k_fp32.data(), gathered_k.data(), MAX_SEQ * KV_DIM), 0.0);
    EXPECT_EQ(computeMSE(ref_v_fp32.data(), gathered_v.data(), MAX_SEQ * KV_DIM), 0.0);
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ3_IncrementalAppend_DecodeLike)
{
    constexpr int MAX_SEQ = 16;

    CPURingKVCacheTQ3 cache(mpi_ctx_, 1, 1, MAX_SEQ,
                            N_KV_HEADS, HEAD_DIM, DeviceId::cpu());

    // Prefill + decode
    auto prefill_k = makeRandomFP32(5, 2700);
    auto prefill_v = makeRandomFP32(5, 2800);
    ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ3(*prefill_k).get(),
                                quantizeTQ3(*prefill_v).get(), 5));

    for (int step = 0; step < 3; ++step)
    {
        auto dk = makeRandomFP32(1, 2900 + step);
        auto dv = makeRandomFP32(1, 3000 + step);
        ASSERT_TRUE(cache.append_kv(0, 0, quantizeTQ3(*dk).get(),
                                    quantizeTQ3(*dv).get(), 1));
    }
    EXPECT_EQ(cache.ring_size(0, 0), 8);

    // Gather and verify no NaN/Inf
    auto out_k = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    auto out_v = std::make_shared<TQ3Tensor>(
        std::vector<size_t>{static_cast<size_t>(MAX_SEQ), KV_DIM}, HEAD_DIM);
    std::vector<int> kv_lens;
    cache.gather_kv_batched(0, 1, out_k.get(), out_v.get(), kv_lens);

    out_k->set_turboquant_context(turboquant_ctx_.get());
    out_v->set_turboquant_context(turboquant_ctx_.get());
    std::vector<float> fp32_k(8 * KV_DIM), fp32_v(8 * KV_DIM);
    for (int r = 0; r < 8; ++r)
    {
        out_k->to_fp32_row(r, fp32_k.data() + r * KV_DIM);
        out_v->to_fp32_row(r, fp32_v.data() + r * KV_DIM);
    }

    for (size_t i = 0; i < fp32_k.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(fp32_k[i])) << "NaN in gathered TQ3 K at " << i;
        ASSERT_FALSE(std::isinf(fp32_k[i])) << "Inf in gathered TQ3 K at " << i;
        ASSERT_FALSE(std::isnan(fp32_v[i])) << "NaN in gathered TQ3 V at " << i;
        ASSERT_FALSE(std::isinf(fp32_v[i])) << "Inf in gathered TQ3 V at " << i;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cross-precision comparison
// ─────────────────────────────────────────────────────────────────────

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_vs_TQ3_QualityOrdering)
{
    // TQ4 (4-bit) should have lower quantization error than TQ3 (3-bit)
    constexpr int N_TOKENS = 8;

    auto fp32_k = makeRandomFP32(N_TOKENS, 3100);

    auto tq4_k = quantizeTQ4(*fp32_k);
    auto tq3_k = quantizeTQ3(*fp32_k);

    auto tq4_fp32 = dequantizeTQ4(*tq4_k);
    auto tq3_fp32 = dequantizeTQ3(*tq3_k);

    double mse_tq4 = computeMSE(fp32_k->data(), tq4_fp32.data(), N_TOKENS * KV_DIM);
    double mse_tq3 = computeMSE(fp32_k->data(), tq3_fp32.data(), N_TOKENS * KV_DIM);

    EXPECT_LT(mse_tq4, mse_tq3)
        << "TQ4 (4-bit) should have lower MSE than TQ3 (3-bit). "
        << "TQ4 MSE=" << mse_tq4 << ", TQ3 MSE=" << mse_tq3;
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ4_QuantizationError_WithinBounds)
{
    // TurboQuant prod prioritizes inner-product quality over pure reconstruction MSE.
    constexpr int N_TOKENS = 32;
    auto fp32 = makeRandomFP32(N_TOKENS, 3200);
    auto tq4 = quantizeTQ4(*fp32);
    auto roundtrip = dequantizeTQ4(*tq4);

    double mse = computeMSE(fp32->data(), roundtrip.data(), N_TOKENS * KV_DIM);
    EXPECT_LT(mse, 0.06) << "TQ4 prod MSE " << mse << " exceeds expected quality floor";
}

TEST_F(Test__CPURingKVCache_TurboQuant, TQ3_QuantizationError_WithinBounds)
{
    // TurboQuant prod prioritizes inner-product quality over pure reconstruction MSE.
    constexpr int N_TOKENS = 32;
    auto fp32 = makeRandomFP32(N_TOKENS, 3300);
    auto tq3 = quantizeTQ3(*fp32);
    auto roundtrip = dequantizeTQ3(*tq3);

    double mse = computeMSE(fp32->data(), roundtrip.data(), N_TOKENS * KV_DIM);
    EXPECT_LT(mse, 0.20) << "TQ3 prod MSE " << mse << " exceeds expected quality floor";
}
