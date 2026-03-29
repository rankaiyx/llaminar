/**
 * @file Test__TQFusedAttention.cpp
 * @brief Tests for the fused TQ8-K / TQ4-V attention decode path.
 *
 * Validates that compute_decode_tqkv (which operates directly on quantized
 * TQ8/TQ4 blocks via the rotation orthogonality trick) produces results
 * matching the dequant-first path within acceptable tolerance.
 *
 * The fused path:
 *   1. Pre-rotates Q: Q_rot = Π·Q  (once per head, O(D²))
 *   2. QK dot product: dot(Q_rot, centroids(K)) × norm × (1/D)  [O(D) per pos]
 *   3. V accumulation in rotated space: Σ(w × norm × centroids(V))  [O(D) per pos]
 *   4. Inverse rotation: output = Πᵀ × rotated_accum  (once per head, O(D²))
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ4.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────
static double cosine_similarity(const float *a, const float *b, size_t n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        dot += double(a[i]) * double(b[i]);
        na += double(a[i]) * double(a[i]);
        nb += double(b[i]) * double(b[i]);
    }
    if (na < 1e-30 || nb < 1e-30)
        return 0.0;
    return dot / std::sqrt(na * nb);
}

// ─────────────────────────────────────────────────────────────────────
// Test fixture
// ─────────────────────────────────────────────────────────────────────
class Test__TQFusedAttention : public ::testing::Test
{
protected:
    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};
    std::unique_ptr<TurboQuantContext> tq_ctx_;

    int head_dim_ = 0;
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int kv_dim_ = 0;
    int q_dim_ = 0;

    void SetUpDims(int head_dim, int n_heads, int n_kv_heads)
    {
        head_dim_ = head_dim;
        n_heads_ = n_heads;
        n_kv_heads_ = n_kv_heads;
        kv_dim_ = n_kv_heads * head_dim;
        q_dim_ = n_heads * head_dim;
        tq_ctx_ = std::make_unique<TurboQuantContext>(head_dim, /*rotation_seed=*/42);
    }

    static std::shared_ptr<FP32Tensor> makeRandomFP32(int rows, int cols, unsigned seed)
    {
        auto t = std::make_shared<FP32Tensor>(std::vector<size_t>{
            static_cast<size_t>(rows), static_cast<size_t>(cols)});
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float *d = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
            d[i] = dist(rng);
        return t;
    }

    std::shared_ptr<TQ8Tensor> quantizeTQ8(const FP32Tensor &src)
    {
        return TQ8Tensor::quantize_from_fp32(src.data(), src.shape(), head_dim_, *tq_ctx_);
    }

    std::shared_ptr<TQ4Tensor> quantizeTQ4(const FP32Tensor &src)
    {
        return TQ4Tensor::quantize_from_fp32(src.data(), src.shape(), head_dim_, *tq_ctx_);
    }

    /**
     * Run attention via the dequant-first path:
     *   FP32 Q, dequant(TQ8 K) → FP32, dequant(TQ4 V) → FP32 → kernel.
     */
    std::vector<float> runDequantPath(
        FP32Tensor *Q,
        const TQ8Tensor *K_tq8,
        const TQ4Tensor *V_tq4,
        int kv_len, bool causal)
    {
        // Dequant K and V
        std::vector<float> dk(kv_len * kv_dim_);
        std::vector<float> dv(kv_len * kv_dim_);
        K_tq8->dequantize_to_fp32(dk.data(), *tq_ctx_);
        V_tq4->dequantize_to_fp32(dv.data(), *tq_ctx_);

        auto K_fp32 = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim_)});
        auto V_fp32 = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(kv_dim_)});
        std::memcpy(K_fp32->mutable_data(), dk.data(), dk.size() * sizeof(float));
        std::memcpy(V_fp32->mutable_data(), dv.data(), dv.size() * sizeof(float));

        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(q_dim_)});

        auto kernel = Q->createAttention();
        EXPECT_NE(kernel, nullptr);

        auto ws = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads_ * kv_len)});

        bool ok = kernel->compute_tensor(
            Q, K_fp32.get(), V_fp32.get(), output.get(),
            /*batch_size=*/1, /*seq_len=*/1, kv_len,
            n_heads_, n_kv_heads_, head_dim_,
            causal, /*window_size=*/-1,
            ws.get(), nullptr, nullptr, -1);
        EXPECT_TRUE(ok);

        std::vector<float> result(q_dim_);
        std::memcpy(result.data(), output->data(), q_dim_ * sizeof(float));
        return result;
    }

    /**
     * Run attention via the fused TQ path:
     *   FP32 Q, raw TQ8 K, raw TQ4 V → kernel (no dequant).
     */
    std::vector<float> runFusedPath(
        FP32Tensor *Q,
        const TQ8Tensor *K_tq8,
        const TQ4Tensor *V_tq4,
        int kv_len, bool causal)
    {
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{1, static_cast<size_t>(q_dim_)});

        auto kernel = Q->createAttention();
        EXPECT_NE(kernel, nullptr);

        auto ws = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_heads_ * kv_len)});

        // Pass TQ tensors directly — the kernel should detect TQ8/TQ4 and
        // dispatch to compute_decode_tqkv.
        bool ok = kernel->compute_tensor(
            Q, K_tq8, V_tq4, output.get(),
            /*batch_size=*/1, /*seq_len=*/1, kv_len,
            n_heads_, n_kv_heads_, head_dim_,
            causal, /*window_size=*/-1,
            ws.get(), nullptr, nullptr, -1);
        EXPECT_TRUE(ok);

        std::vector<float> result(q_dim_);
        std::memcpy(result.data(), output->data(), q_dim_ * sizeof(float));
        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────
// Test 1: Fused vs Dequant path — head_dim=128, short context
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__TQFusedAttention, FusedVsDequant_HeadDim128_ShortContext)
{
    SetUpDims(/*head_dim=*/128, /*n_heads=*/16, /*n_kv_heads=*/8);
    constexpr int KV_LEN = 10;

    auto Q = makeRandomFP32(1, q_dim_, 1000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 1001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 1002);

    auto K_tq8 = quantizeTQ8(*K_fp32);
    auto V_tq4 = quantizeTQ4(*V_fp32);

    // Set TQ context on the tensors (the fused path reads it)
    K_tq8->set_turboquant_context(tq_ctx_.get());
    V_tq4->set_turboquant_context(tq_ctx_.get());

    auto dequant_result = runDequantPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, false);
    auto fused_result = runFusedPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Fused vs Dequant (h128, kv=10): cosine=" << cos << std::endl;

    // The fused path should match the dequant path very closely.
    // Both use the same TQ data; the only difference is compute order.
    EXPECT_GT(cos, 0.99) << "Fused path should closely match dequant path";

    // Per-head analysis
    double worst = 1.0;
    int worst_h = -1;
    for (int h = 0; h < n_heads_; ++h)
    {
        double hcos = cosine_similarity(
            dequant_result.data() + h * head_dim_,
            fused_result.data() + h * head_dim_,
            head_dim_);
        if (hcos < worst)
        {
            worst = hcos;
            worst_h = h;
        }
    }
    std::cout << "  Worst per-head cosine: head " << worst_h << " = " << worst << std::endl;
    EXPECT_GT(worst, 0.95) << "Per-head cosine too low";
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: Fused vs Dequant path — head_dim=128, longer context
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__TQFusedAttention, FusedVsDequant_HeadDim128_LongContext)
{
    SetUpDims(128, 16, 8);
    constexpr int KV_LEN = 200;

    auto Q = makeRandomFP32(1, q_dim_, 2000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 2001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 2002);

    auto K_tq8 = quantizeTQ8(*K_fp32);
    auto V_tq4 = quantizeTQ4(*V_fp32);
    K_tq8->set_turboquant_context(tq_ctx_.get());
    V_tq4->set_turboquant_context(tq_ctx_.get());

    auto dequant_result = runDequantPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, false);
    auto fused_result = runFusedPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Fused vs Dequant (h128, kv=200): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.99) << "Fused path should closely match dequant path for long context";
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: Fused vs Dequant path — head_dim=64 (Qwen2-like)
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__TQFusedAttention, FusedVsDequant_HeadDim64)
{
    SetUpDims(/*head_dim=*/64, /*n_heads=*/16, /*n_kv_heads=*/4);
    constexpr int KV_LEN = 50;

    auto Q = makeRandomFP32(1, q_dim_, 3000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 3001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 3002);

    auto K_tq8 = quantizeTQ8(*K_fp32);
    auto V_tq4 = quantizeTQ4(*V_fp32);
    K_tq8->set_turboquant_context(tq_ctx_.get());
    V_tq4->set_turboquant_context(tq_ctx_.get());

    auto dequant_result = runDequantPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, false);
    auto fused_result = runFusedPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, false);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Fused vs Dequant (h64, kv=50): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.99) << "Fused path should match dequant path for head_dim=64";
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: Causal masking in fused path
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__TQFusedAttention, FusedVsDequant_CausalMask)
{
    SetUpDims(128, 16, 8);
    constexpr int KV_LEN = 30;

    auto Q = makeRandomFP32(1, q_dim_, 4000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 4001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 4002);

    auto K_tq8 = quantizeTQ8(*K_fp32);
    auto V_tq4 = quantizeTQ4(*V_fp32);
    K_tq8->set_turboquant_context(tq_ctx_.get());
    V_tq4->set_turboquant_context(tq_ctx_.get());

    auto dequant_result = runDequantPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, /*causal=*/true);
    auto fused_result = runFusedPath(Q.get(), K_tq8.get(), V_tq4.get(), KV_LEN, /*causal=*/true);

    double cos = cosine_similarity(dequant_result.data(), fused_result.data(), q_dim_);
    std::cout << "Fused vs Dequant (causal, h128, kv=30): cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.99) << "Fused path with causal mask should match dequant path";
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: Per-layer rotation context
// Ensure different layers produce the same fused output when both paths
// use the same per-layer rotation.
// ─────────────────────────────────────────────────────────────────────
TEST_F(Test__TQFusedAttention, FusedVsDequant_PerLayerContext)
{
    SetUpDims(128, 8, 4);
    constexpr int KV_LEN = 20;

    // Use layer 5's derived context
    const int layer_idx = 5;
    const TurboQuantContext &layer_ctx = tq_ctx_->for_layer(layer_idx);

    auto Q = makeRandomFP32(1, q_dim_, 5000);
    auto K_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 5001);
    auto V_fp32 = makeRandomFP32(KV_LEN, kv_dim_, 5002);

    // Quantize with the per-layer context
    auto K_tq8 = TQ8Tensor::quantize_from_fp32(
        K_fp32->data(), K_fp32->shape(), head_dim_, layer_ctx);
    auto V_tq4 = TQ4Tensor::quantize_from_fp32(
        V_fp32->data(), V_fp32->shape(), head_dim_, layer_ctx);

    // Set per-layer context on tensors (fused path reads from K_tq8->turboquant_context())
    K_tq8->set_turboquant_context(&layer_ctx);
    V_tq4->set_turboquant_context(&layer_ctx);

    // Dequant path also needs the same per-layer context
    std::vector<float> dk(KV_LEN * kv_dim_);
    std::vector<float> dv(KV_LEN * kv_dim_);
    K_tq8->dequantize_to_fp32(dk.data(), layer_ctx);
    V_tq4->dequantize_to_fp32(dv.data(), layer_ctx);

    auto K_deq = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(kv_dim_)});
    auto V_deq = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(KV_LEN), static_cast<size_t>(kv_dim_)});
    std::memcpy(K_deq->mutable_data(), dk.data(), dk.size() * sizeof(float));
    std::memcpy(V_deq->mutable_data(), dv.data(), dv.size() * sizeof(float));

    auto output_deq = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(q_dim_)});
    auto output_fused = std::make_shared<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(q_dim_)});

    auto kernel = Q->createAttention();
    ASSERT_NE(kernel, nullptr);

    auto ws = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(n_heads_ * KV_LEN)});

    // Dequant path
    ASSERT_TRUE(kernel->compute_tensor(
        Q.get(), K_deq.get(), V_deq.get(), output_deq.get(),
        1, 1, KV_LEN, n_heads_, n_kv_heads_, head_dim_,
        false, -1, ws.get(), nullptr, nullptr, -1));

    // Fused path
    ASSERT_TRUE(kernel->compute_tensor(
        Q.get(), K_tq8.get(), V_tq4.get(), output_fused.get(),
        1, 1, KV_LEN, n_heads_, n_kv_heads_, head_dim_,
        false, -1, ws.get(), nullptr, nullptr, -1));

    double cos = cosine_similarity(
        output_deq->data(), output_fused->data(), q_dim_);
    std::cout << "Per-layer (layer=" << layer_idx << ") fused vs dequant: cosine=" << cos << std::endl;
    EXPECT_GT(cos, 0.99) << "Per-layer rotation context fused path should match";
}
