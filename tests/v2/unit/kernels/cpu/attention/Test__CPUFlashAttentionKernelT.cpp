/**
 * @file Test__CPUFlashAttentionKernelT.cpp
 * @brief Comprehensive accuracy tests for CPUFlashAttentionKernelT
 *
 * Exercises every code path in the flash attention kernel:
 *   - FP32 decode (seq_len=1, dot_fp32_avx512)
 *   - FP32 prefill (dot_fp32_avx512, online softmax over KV tiles)
 *   - AVX512-VNNI i16/i12 prefill (quantise + packed-pair dot products)
 *   - Causal masking, window masking, GQA head broadcasting
 *   - Edge cases: non-multiple-of-32 head_dim, odd kv_len, single token
 *
 * All tests compare against a scalar reference implementation of
 * softmax(Q·K^T / sqrt(d)) · V to validate numerical accuracy.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "v2/kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/CPUFeatures.h"
#include "v2/utils/DebugEnv.h"

using namespace llaminar2;

// ---------------------------------------------------------------------------
// Scalar reference implementation
// ---------------------------------------------------------------------------
namespace ref
{
    /// Scalar softmax attention:  output = softmax(Q·K^T / sqrt(d)) · V
    /// Layout: Q  [seq_len,  n_heads * head_dim]
    ///         K  [kv_len,   n_kv_heads * head_dim]
    ///         V  [kv_len,   n_kv_heads * head_dim]
    ///         O  [seq_len,  n_heads * head_dim]
    static void attention(
        const float *Q, const float *K, const float *V, float *O,
        int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, int position_offset)
    {
        const int heads_per_kv = n_heads / n_kv_heads;
        const int q_stride = n_heads * head_dim;
        const int kv_stride = n_kv_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int h = 0; h < n_heads; ++h)
        {
            const int kv_h = h / heads_per_kv;
            for (int q_pos = 0; q_pos < seq_len; ++q_pos)
            {
                const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                float *out = O + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                const int q_abs = position_offset + q_pos;

                // Compute scores
                std::vector<float> scores(kv_len);
                for (int k = 0; k < kv_len; ++k)
                {
                    const float *k_ptr = K + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;

                    // Check masking
                    bool masked = false;
                    if (causal && k > q_abs)
                        masked = true;
                    if (window_size > 0 && k < q_abs - window_size + 1)
                        masked = true;

                    if (masked)
                    {
                        scores[k] = -std::numeric_limits<float>::infinity();
                    }
                    else
                    {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot += q_ptr[d] * k_ptr[d];
                        scores[k] = dot * scale;
                    }
                }

                // Softmax
                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_len; ++k)
                {
                    if (std::isfinite(scores[k]))
                    {
                        scores[k] = std::exp(scores[k] - max_s);
                        sum_exp += scores[k];
                    }
                    else
                    {
                        scores[k] = 0.0f;
                    }
                }
                if (sum_exp > 0.0f)
                {
                    for (int k = 0; k < kv_len; ++k)
                        scores[k] /= sum_exp;
                }

                // Weighted V
                std::fill(out, out + head_dim, 0.0f);
                for (int k = 0; k < kv_len; ++k)
                {
                    if (scores[k] == 0.0f)
                        continue;
                    const float *v_ptr = V + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                    for (int d = 0; d < head_dim; ++d)
                        out[d] += scores[k] * v_ptr[d];
                }
            }
        }
    }
} // namespace ref

// ---------------------------------------------------------------------------
// Test utilities
// ---------------------------------------------------------------------------
namespace
{
    /// Seed a deterministic RNG per test for reproducibility.
    std::mt19937 &rng()
    {
        static std::mt19937 gen(42);
        return gen;
    }

    void fill_random(float *buf, size_t n, float lo = -1.0f, float hi = 1.0f)
    {
        std::uniform_real_distribution<float> dist(lo, hi);
        auto &g = rng();
        for (size_t i = 0; i < n; ++i)
            buf[i] = dist(g);
    }

    /// Max absolute error between two buffers.
    float max_abs_error(const float *a, const float *b, size_t n)
    {
        float mx = 0.0f;
        for (size_t i = 0; i < n; ++i)
            mx = std::max(mx, std::abs(a[i] - b[i]));
        return mx;
    }

    /// Root-mean-square error.
    float rms_error(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += d * d;
        }
        return static_cast<float>(std::sqrt(sum / static_cast<double>(n)));
    }

    /// Cosine similarity.
    float cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom < 1e-12)
            return 1.0f; // both zero → identical
        return static_cast<float>(dot / denom);
    }

    /// Check no NaN or Inf in buffer.
    bool is_finite(const float *buf, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(buf[i]))
                return false;
        return true;
    }

    /// Set env var helper (uses setenv on Linux).
    struct ScopedEnv
    {
        std::string key_;
        std::string old_val_;
        bool had_old_ = false;

        ScopedEnv(const char *key, const char *val) : key_(key)
        {
            const char *old = std::getenv(key);
            if (old)
            {
                had_old_ = true;
                old_val_ = old;
            }
            setenv(key, val, 1);
        }

        ~ScopedEnv()
        {
            if (had_old_)
                setenv(key_.c_str(), old_val_.c_str(), 1);
            else
                unsetenv(key_.c_str());
        }
    };

    /// FP32 tolerance for the FP32-only path (flash vs reference).
    /// Online softmax introduces small rounding differences vs. 2-pass softmax.
    constexpr float FP32_TOLERANCE = 1e-4f;
    constexpr float FP32_COSINE_THRESHOLD = 0.99999f;

    /// VNNI i16 quantised path — now the DEFAULT production prefill path.
    /// INT16 quantisation introduces measurable but bounded error.
    /// Tolerances here are tight enough to catch regressions while allowing
    /// the inherent quantisation noise from 12-bit effective precision.
    constexpr float VNNI_MAX_ABS_TOLERANCE = 0.02f;
    constexpr float VNNI_COSINE_THRESHOLD = 0.9995f;

} // anonymous namespace

// ===========================================================================
// Test fixture
// ===========================================================================
class Test__CPUFlashAttentionKernelT : public ::testing::Test
{
protected:
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;

    void SetUp() override
    {
        rng().seed(42); // deterministic per test

        // Explicitly disable VNNI so these tests isolate the FP32 code path.
        // VNNI accuracy is tested separately in Test__CPUFlashAttentionKernelT_VNNI.
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12", "0", 1);
        mutableDebugEnv().attention.reload();
    }

    void TearDown() override
    {
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12");
        mutableDebugEnv().attention.reload();
    }

    /// Run kernel compute() and compare to reference.
    /// Suitable for prefill-like calls where seq_len == kv_len.
    void runAndCompare(
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        float max_abs_tol, float cosine_tol,
        const char *label)
    {
        const int kv_len = seq_len;
        const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

        std::vector<float> Q(q_size), K(kv_size), V(kv_size);
        std::vector<float> out_kernel(q_size, 0.0f);
        std::vector<float> out_ref(q_size, 0.0f);

        fill_random(Q.data(), q_size);
        fill_random(K.data(), kv_size);
        fill_random(V.data(), kv_size);

        // Kernel under test
        bool ok = kernel_.compute(
            Q.data(), K.data(), V.data(), out_kernel.data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            causal, window_size,
            nullptr, nullptr, nullptr, nullptr,
            false, nullptr, -1);
        ASSERT_TRUE(ok) << label << ": kernel.compute() failed";

        // Reference
        ref::attention(Q.data(), K.data(), V.data(), out_ref.data(),
                       seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                       causal, window_size, 0);

        ASSERT_TRUE(is_finite(out_kernel.data(), q_size)) << label << ": kernel output has NaN/Inf";
        ASSERT_TRUE(is_finite(out_ref.data(), q_size)) << label << ": reference output has NaN/Inf";

        float mae = max_abs_error(out_kernel.data(), out_ref.data(), q_size);
        float cos = cosine_similarity(out_kernel.data(), out_ref.data(), q_size);

        EXPECT_LE(mae, max_abs_tol)
            << label << ": max abs error " << mae << " exceeds tolerance " << max_abs_tol;
        EXPECT_GE(cos, cosine_tol)
            << label << ": cosine sim " << cos << " below threshold " << cosine_tol;
    }

    /// Run kernel compute_decode() and compare to reference.
    /// This is the decode path where seq_len < kv_len.
    void runDecodeAndCompare(
        int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal, int position_offset,
        float max_abs_tol, float cosine_tol,
        const char *label)
    {
        const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

        std::vector<float> Q(q_size), K(kv_size), V(kv_size);
        std::vector<float> out_kernel(q_size, 0.0f);
        std::vector<float> out_ref(q_size, 0.0f);

        fill_random(Q.data(), q_size);
        fill_random(K.data(), kv_size);
        fill_random(V.data(), kv_size);

        // Kernel
        bool ok = kernel_.compute_decode(
            Q.data(), K.data(), V.data(), out_kernel.data(),
            seq_len, kv_len, n_heads, n_kv_heads, head_dim,
            causal, position_offset);
        ASSERT_TRUE(ok) << label << ": kernel.compute_decode() failed";

        // Reference
        ref::attention(Q.data(), K.data(), V.data(), out_ref.data(),
                       seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                       causal, -1, position_offset);

        ASSERT_TRUE(is_finite(out_kernel.data(), q_size)) << label << ": kernel output has NaN/Inf";

        float mae = max_abs_error(out_kernel.data(), out_ref.data(), q_size);
        float cos = cosine_similarity(out_kernel.data(), out_ref.data(), q_size);

        EXPECT_LE(mae, max_abs_tol)
            << label << ": max abs error " << mae << " > " << max_abs_tol;
        EXPECT_GE(cos, cosine_tol)
            << label << ": cosine sim " << cos << " < " << cosine_tol;
    }
};

// ===========================================================================
// 1. FP32 Decode Path Tests (seq_len=1, kv_len varies)
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, Decode_SingleKV)
{
    runDecodeAndCompare(1, 1, 2, 2, 64, true, 0,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_SingleKV");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_Short_KV16)
{
    runDecodeAndCompare(1, 16, 4, 4, 64, true, 15,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_Short_KV16");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_Medium_KV128)
{
    runDecodeAndCompare(1, 128, 4, 4, 64, true, 127,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_Medium_KV128");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_Long_KV512)
{
    runDecodeAndCompare(1, 512, 4, 2, 64, true, 511,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_Long_KV512");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_Long_KV1024)
{
    runDecodeAndCompare(1, 1024, 4, 2, 128, true, 1023,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_Long_KV1024");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_NoCausal)
{
    // Non-causal decode: all KV positions are visible
    runDecodeAndCompare(1, 64, 4, 4, 64, false, 0,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_NoCausal");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_OddKVLen)
{
    // kv_len not a power of 2 / not a multiple of tile size
    runDecodeAndCompare(1, 37, 2, 2, 64, true, 36,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_OddKVLen");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_HeadDim128)
{
    runDecodeAndCompare(1, 256, 4, 2, 128, true, 255,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_HeadDim128");
}

// ===========================================================================
// 2. FP32 Prefill Path Tests (seq_len > 1, VNNI disabled)
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_Tiny_2x4)
{
    runAndCompare(2, 1, 1, 4, false, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_Tiny_2x4");
}

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_Small_8x64_Causal)
{
    runAndCompare(8, 4, 4, 64, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_Small_8x64_Causal");
}

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_Medium_32x64)
{
    runAndCompare(32, 4, 2, 64, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_Medium_32x64");
}

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_64x128_Causal)
{
    runAndCompare(64, 4, 2, 128, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_64x128_Causal");
}

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_NoCausal)
{
    // All positions see all positions
    runAndCompare(16, 4, 4, 64, false, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_NoCausal");
}

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_HeadDim4)
{
    // Very small head_dim (scalar tail path in AVX512 dot)
    runAndCompare(8, 2, 2, 4, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_HeadDim4");
}

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_HeadDim48)
{
    // head_dim not a multiple of 16 (AVX512 tail handling)
    runAndCompare(16, 2, 2, 48, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_HeadDim48");
}

TEST_F(Test__CPUFlashAttentionKernelT, Prefill_HeadDim96)
{
    // head_dim = 96 (3 AVX512 iterations, no tail)
    runAndCompare(16, 2, 2, 96, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "Prefill_HeadDim96");
}

// ===========================================================================
// 3. GQA (Grouped Query Attention) Tests
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, GQA_4HeadsTo2KV_Decode)
{
    runDecodeAndCompare(1, 64, 4, 2, 64, true, 63,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "GQA_4HeadsTo2KV_Decode");
}

TEST_F(Test__CPUFlashAttentionKernelT, GQA_8HeadsTo1KV_Decode)
{
    // MQA: single KV head shared by all query heads
    runDecodeAndCompare(1, 64, 8, 1, 64, true, 63,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "GQA_8HeadsTo1KV_Decode");
}

TEST_F(Test__CPUFlashAttentionKernelT, GQA_4HeadsTo2KV_Prefill)
{
    runAndCompare(16, 4, 2, 64, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "GQA_4HeadsTo2KV_Prefill");
}

TEST_F(Test__CPUFlashAttentionKernelT, GQA_8HeadsTo2KV_Prefill)
{
    runAndCompare(32, 8, 2, 64, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "GQA_8HeadsTo2KV_Prefill");
}

// ===========================================================================
// 4. Window (Sliding) Attention Tests
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, WindowAttention_Small)
{
    runAndCompare(16, 4, 4, 64, true, 4,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "WindowAttention_Small");
}

TEST_F(Test__CPUFlashAttentionKernelT, WindowAttention_LargerSeq)
{
    runAndCompare(64, 4, 2, 64, true, 16,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "WindowAttention_LargerSeq");
}

TEST_F(Test__CPUFlashAttentionKernelT, WindowAttention_Decode)
{
    // Window attention during decode
    const int seq_len = 1;
    const int kv_len = 128;
    const int n_heads = 4;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int window_size = 32;
    const int position_offset = kv_len - 1;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K(kv_size), V(kv_size);
    std::vector<float> out_kernel(q_size, 0.0f);
    std::vector<float> out_ref(q_size, 0.0f);

    fill_random(Q.data(), q_size);
    fill_random(K.data(), kv_size);
    fill_random(V.data(), kv_size);

    // compute() takes window_size; but compute_decode() does not have window_size param.
    // Use compute() with seq_len=kv_len then compare reference.
    // Actually, compute_decode doesn't accept window_size. Test via compute() pathway instead
    // by using the compute() interface directly with window_size.
    // For the decode case we just invoke compute() with seq_len=1, kv_len= kv_len via compute_decode.
    // compute_decode hardcodes window_size=-1. So window attention only works via compute() path.

    // Test a prefill-like call with window
    const int prefill_len = 64;
    const size_t pq_size = static_cast<size_t>(prefill_len) * n_heads * head_dim;
    const size_t pkv_size = static_cast<size_t>(prefill_len) * n_kv_heads * head_dim;
    std::vector<float> PQ(pq_size), PK(pkv_size), PV(pkv_size);
    std::vector<float> pout_kernel(pq_size, 0.0f);
    std::vector<float> pout_ref(pq_size, 0.0f);

    fill_random(PQ.data(), pq_size);
    fill_random(PK.data(), pkv_size);
    fill_random(PV.data(), pkv_size);

    bool ok = kernel_.compute(
        PQ.data(), PK.data(), PV.data(), pout_kernel.data(),
        prefill_len, n_heads, n_kv_heads, head_dim,
        true, window_size,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);
    ASSERT_TRUE(ok);

    ref::attention(PQ.data(), PK.data(), PV.data(), pout_ref.data(),
                   prefill_len, prefill_len, n_heads, n_kv_heads, head_dim,
                   true, window_size, 0);

    ASSERT_TRUE(is_finite(pout_kernel.data(), pq_size));
    float mae = max_abs_error(pout_kernel.data(), pout_ref.data(), pq_size);
    float cos = cosine_similarity(pout_kernel.data(), pout_ref.data(), pq_size);

    EXPECT_LE(mae, FP32_TOLERANCE) << "WindowAttention_Decode: max abs err " << mae;
    EXPECT_GE(cos, FP32_COSINE_THRESHOLD) << "WindowAttention_Decode: cosine " << cos;
}

// ===========================================================================
// 5. Causal Masking Correctness
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, CausalMask_FirstTokenSeesOnlySelf)
{
    // With causal masking, position 0 should only attend to position 0.
    // So output[0] should equal V[0].
    const int seq_len = 4;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 8;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = q_size;

    std::vector<float> Q(q_size), K(kv_size), V(kv_size);
    std::vector<float> out(q_size, 0.0f);

    // Set Q so that Q[0] · K[0] has a specific score, but it doesn't matter
    // since position 0 can only attend to position 0.
    fill_random(Q.data(), q_size);
    fill_random(K.data(), kv_size);
    fill_random(V.data(), kv_size);

    bool ok = kernel_.compute(
        Q.data(), K.data(), V.data(), out.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);
    ASSERT_TRUE(ok);

    // The output for position 0 = softmax([score(Q0,K0)]) * V[0] = V[0]
    // (single-element softmax is always 1.0)
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_NEAR(out[d], V[d], 1e-5f)
            << "Position 0 should see only V[0], but differs at dim " << d;
    }
}

TEST_F(Test__CPUFlashAttentionKernelT, CausalMask_LastTokenSeesAll)
{
    // The last token in causal should see all prior tokens → same as non-causal
    // for that specific row.
    const int seq_len = 8;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 16;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = q_size;

    std::vector<float> Q(q_size), K(kv_size), V(kv_size);
    fill_random(Q.data(), q_size);
    fill_random(K.data(), kv_size);
    fill_random(V.data(), kv_size);

    std::vector<float> out_causal(q_size, 0.0f);
    std::vector<float> out_noncausal(q_size, 0.0f);

    kernel_.compute(Q.data(), K.data(), V.data(), out_causal.data(),
                    seq_len, n_heads, n_kv_heads, head_dim,
                    true, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1);

    kernel_.compute(Q.data(), K.data(), V.data(), out_noncausal.data(),
                    seq_len, n_heads, n_kv_heads, head_dim,
                    false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1);

    // Last row of causal should match last row of non-causal
    const float *last_causal = out_causal.data() + static_cast<size_t>(seq_len - 1) * head_dim;
    const float *last_nocausal = out_noncausal.data() + static_cast<size_t>(seq_len - 1) * head_dim;
    float mae = max_abs_error(last_causal, last_nocausal, head_dim);
    EXPECT_LE(mae, 1e-6f)
        << "Last token output should be identical for causal/non-causal, but mae=" << mae;
}

// ===========================================================================
// 6. compute_batch() Path
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, Batch_MatchesLooped)
{
    const int batch_size = 3;
    const int seq_len = 8;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 32;

    const size_t per_batch_q = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t per_batch_kv = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;
    const size_t total_q = per_batch_q * batch_size;
    const size_t total_kv = per_batch_kv * batch_size;

    std::vector<float> Q(total_q), K(total_kv), V(total_kv);
    fill_random(Q.data(), total_q);
    fill_random(K.data(), total_kv);
    fill_random(V.data(), total_kv);

    // Batched call
    std::vector<float> out_batch(total_q, 0.0f);
    bool ok = kernel_.compute_batch(
        Q.data(), K.data(), V.data(), out_batch.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);
    ASSERT_TRUE(ok);

    // Looped single calls
    std::vector<float> out_loop(total_q, 0.0f);
    for (int b = 0; b < batch_size; ++b)
    {
        bool ok2 = kernel_.compute(
            Q.data() + b * per_batch_q,
            K.data() + b * per_batch_kv,
            V.data() + b * per_batch_kv,
            out_loop.data() + b * per_batch_q,
            seq_len, n_heads, n_kv_heads, head_dim,
            true, -1,
            nullptr, nullptr, nullptr, nullptr,
            false, nullptr, -1);
        ASSERT_TRUE(ok2);
    }

    float mae = max_abs_error(out_batch.data(), out_loop.data(), total_q);
    EXPECT_LE(mae, 1e-6f)
        << "Batch output differs from looped single calls, mae = " << mae;
}

// ===========================================================================
// 7. Edge Cases
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, EdgeCase_SingleTokenNoCausal)
{
    runAndCompare(1, 1, 1, 64, false, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "EdgeCase_SingleTokenNoCausal");
}

TEST_F(Test__CPUFlashAttentionKernelT, EdgeCase_HeadDimNotMultipleOf16)
{
    // head_dim=17 forces scalar tail on every dot product
    runAndCompare(8, 2, 2, 17, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "EdgeCase_HeadDimNotMultipleOf16");
}

TEST_F(Test__CPUFlashAttentionKernelT, EdgeCase_HeadDim1)
{
    // Degenerate: head_dim=1
    runAndCompare(4, 2, 2, 1, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "EdgeCase_HeadDim1");
}

TEST_F(Test__CPUFlashAttentionKernelT, EdgeCase_LargeHead_256)
{
    // Llama-style large head dim
    runAndCompare(8, 2, 1, 256, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "EdgeCase_LargeHead_256");
}

TEST_F(Test__CPUFlashAttentionKernelT, EdgeCase_ManyHeads_32)
{
    runAndCompare(4, 32, 4, 64, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "EdgeCase_ManyHeads_32");
}

TEST_F(Test__CPUFlashAttentionKernelT, EdgeCase_PrimeSeqLen)
{
    // seq_len=13, a prime number that doesn't divide evenly into tiles
    runAndCompare(13, 2, 2, 64, true, -1,
                  FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                  "EdgeCase_PrimeSeqLen");
}

// ===========================================================================
// 8. VNNI i16/i12 Prefill Path Tests
//    These require setting environment variables to enable the VNNI path
//    and lowering thresholds so short sequences trigger it.
// ===========================================================================

class Test__CPUFlashAttentionKernelT_VNNI : public ::testing::Test
{
protected:
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> kernel_;

    void SetUp() override
    {
        rng().seed(42);

        // Set env vars to enable VNNI i16/i12 path with zero thresholds
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12", "1", 1);
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_SEQ", "1", 1);
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_KV", "1", 1);
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_WORK", "0", 1);
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12_MAX_HEAD_DIM", "256", 1);
        setenv("LLAMINAR_FLASH_PREFILL_I16_I12_QMAX", "2047", 1);

        // Force DebugEnv singleton to re-read the attention env vars
        mutableDebugEnv().attention.reload();
    }

    void TearDown() override
    {
        // Restore defaults
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12");
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_SEQ");
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_KV");
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12_MIN_WORK");
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12_MAX_HEAD_DIM");
        unsetenv("LLAMINAR_FLASH_PREFILL_I16_I12_QMAX");
        mutableDebugEnv().attention.reload();
    }

    void runVNNIAndCompare(
        int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal,
        float max_abs_tol, float cosine_tol,
        const char *label)
    {
        const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
        const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

        std::vector<float> Q(q_size), K(kv_size), V(kv_size);
        std::vector<float> out_kernel(q_size, 0.0f);
        std::vector<float> out_ref(q_size, 0.0f);

        fill_random(Q.data(), q_size, -0.5f, 0.5f);
        fill_random(K.data(), kv_size, -0.5f, 0.5f);
        fill_random(V.data(), kv_size, -0.5f, 0.5f);

        // Use compute() which feeds into compute_flash_fp32 with is_decode=false
        bool ok = kernel_.compute(
            Q.data(), K.data(), V.data(), out_kernel.data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            causal, -1,
            nullptr, nullptr, nullptr, nullptr,
            false, nullptr, -1);
        ASSERT_TRUE(ok) << label << ": kernel.compute() failed";

        ref::attention(Q.data(), K.data(), V.data(), out_ref.data(),
                       seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                       causal, -1, 0);

        ASSERT_TRUE(is_finite(out_kernel.data(), q_size)) << label << ": kernel output has NaN/Inf";
        ASSERT_TRUE(is_finite(out_ref.data(), q_size)) << label << ": reference output has NaN/Inf";

        float mae = max_abs_error(out_kernel.data(), out_ref.data(), q_size);
        float rmse = rms_error(out_kernel.data(), out_ref.data(), q_size);
        float cos = cosine_similarity(out_kernel.data(), out_ref.data(), q_size);

        EXPECT_LE(mae, max_abs_tol)
            << label << ": max abs error " << mae << " > " << max_abs_tol
            << " (rmse=" << rmse << ", cosine=" << cos << ")";
        EXPECT_GE(cos, cosine_tol)
            << label << ": cosine sim " << cos << " < " << cosine_tol;
    }
};

// Note: VNNI i16/i12 prefill is now ON by default. The VNNI fixture forces
// the thresholds low (min_seq=1, min_kv=1) so even small test cases exercise
// the VNNI code path. The runtime guard cpu_supports_avx512_vnni() still
// applies — on machines without VNNI these tests fall back to FP32 and pass
// with even tighter margins.
//
// The base fixture (Test__CPUFlashAttentionKernelT) explicitly disables VNNI
// to isolate the FP32 code path.

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_Small_4x64_Causal)
{
    // 4 tokens, 64 head_dim — exactly 2 AVX512 iterations per row
    // 4-row VNNI fast path can fire (4 positions >= k+3 threshold)
    runVNNIAndCompare(4, 4, 2, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_Small_4x64_Causal");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_Medium_32x64_Causal)
{
    runVNNIAndCompare(32, 32, 4, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_Medium_32x64_Causal");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_128x64_Causal)
{
    // Large enough to exercise multi-tile KV processing
    runVNNIAndCompare(128, 128, 4, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_128x64_Causal");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_64x128_Causal)
{
    // head_dim=128 (Llama-style), forces i16_row_stride=128 (4 blocks of 32)
    runVNNIAndCompare(64, 64, 4, 2, 128, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_64x128_Causal");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_NoCausal)
{
    // Non-causal prefill with VNNI
    runVNNIAndCompare(32, 32, 4, 4, 64, false,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_NoCausal");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_OddKVLen_31)
{
    // kv_len=31 is odd → last K row goes into a packed pair with zero padding
    // Tests single-from-packedpair fallback path for the last KV position
    runVNNIAndCompare(31, 31, 2, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_OddKVLen_31");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_OddKVLen_5)
{
    // kv_len=5: 2 pairs + 1 single → exercises 4-row(fail), 2-row, then 1-row paths
    runVNNIAndCompare(5, 5, 2, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_OddKVLen_5");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_HeadDimNotMultipleOf32)
{
    // head_dim=48: padded to 64 in i16 quantisation → tests zero-padding
    runVNNIAndCompare(16, 16, 2, 2, 48, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_HeadDimNotMultipleOf32");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_HeadDim96)
{
    // 96 = 3*32, aligned but 3 blocks per row
    runVNNIAndCompare(16, 16, 2, 2, 96, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_HeadDim96");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_GQA_8To2)
{
    // GQA with VNNI path: 8 q heads, 2 kv heads
    runVNNIAndCompare(32, 32, 8, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_GQA_8To2");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_MQA_8To1)
{
    // MQA with VNNI: 8 q heads, 1 kv head
    runVNNIAndCompare(32, 32, 8, 1, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_MQA_8To1");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_256Tokens_64HeadDim)
{
    // Larger stress test
    runVNNIAndCompare(256, 256, 4, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_256Tokens_64HeadDim");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_KV3_AllDispatchPaths)
{
    // kv_len=3: pair0 has rows 0,1; pair1 has row 2 + zero pad
    // 4-row path needs k+3<valid_end → fails for kv_len=3 causal where valid_end<=3
    // Exercises 2-row + 1-row fallback only
    runVNNIAndCompare(3, 3, 2, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_KV3_AllDispatchPaths");
}

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_KV2_PairOnly)
{
    // kv_len=2: exactly 1 pair, exercises 2-row path
    runVNNIAndCompare(2, 2, 2, 2, 64, true,
                      VNNI_MAX_ABS_TOLERANCE, VNNI_COSINE_THRESHOLD,
                      "VNNI_KV2_PairOnly");
}

// ===========================================================================
// 9. A/B comparison: same data, VNNI path vs FP32 path
//    Runs the kernel twice with identical inputs — once with VNNI enabled
//    (already set by the VNNI fixture), once with VNNI explicitly disabled.
//    This proves the VNNI code path is actually executing and measures its
//    error relative to both the FP32 kernel path and the scalar reference.
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT_VNNI, VNNI_vs_FP32_AB_Comparison)
{
    const int seq_len = 128;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 64;

    const size_t q_size = static_cast<size_t>(seq_len) * n_heads * head_dim;
    const size_t kv_size = static_cast<size_t>(seq_len) * n_kv_heads * head_dim;

    std::vector<float> Q(q_size), K(kv_size), V(kv_size);
    // Use [-1, 1] range to make quantisation error more visible
    fill_random(Q.data(), q_size, -1.0f, 1.0f);
    fill_random(K.data(), kv_size, -1.0f, 1.0f);
    fill_random(V.data(), kv_size, -1.0f, 1.0f);

    // --- Run A: VNNI path (env is already set by fixture SetUp) ---
    std::vector<float> out_vnni(q_size, 0.0f);
    ASSERT_TRUE(kernel_.compute(
        Q.data(), K.data(), V.data(), out_vnni.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1));

    // --- Run B: FP32 path (temporarily disable VNNI) ---
    setenv("LLAMINAR_FLASH_PREFILL_I16_I12", "0", 1);
    mutableDebugEnv().attention.reload();

    std::vector<float> out_fp32(q_size, 0.0f);
    ASSERT_TRUE(kernel_.compute(
        Q.data(), K.data(), V.data(), out_fp32.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1));

    // Re-enable VNNI for subsequent tests
    setenv("LLAMINAR_FLASH_PREFILL_I16_I12", "1", 1);
    mutableDebugEnv().attention.reload();

    // --- Scalar reference ---
    std::vector<float> out_ref(q_size, 0.0f);
    ref::attention(Q.data(), K.data(), V.data(), out_ref.data(),
                   seq_len, seq_len, n_heads, n_kv_heads, head_dim,
                   true, -1, 0);

    ASSERT_TRUE(is_finite(out_vnni.data(), q_size));
    ASSERT_TRUE(is_finite(out_fp32.data(), q_size));

    // VNNI vs reference
    float vnni_mae = max_abs_error(out_vnni.data(), out_ref.data(), q_size);
    float vnni_rmse = rms_error(out_vnni.data(), out_ref.data(), q_size);
    float vnni_cos = cosine_similarity(out_vnni.data(), out_ref.data(), q_size);

    // FP32 vs reference
    float fp32_mae = max_abs_error(out_fp32.data(), out_ref.data(), q_size);
    float fp32_rmse = rms_error(out_fp32.data(), out_ref.data(), q_size);

    // VNNI vs FP32 kernel (direct comparison)
    float ab_mae = max_abs_error(out_vnni.data(), out_fp32.data(), q_size);

    std::cout << "[A/B] VNNI vs ref:  mae=" << vnni_mae
              << " rmse=" << vnni_rmse
              << " cosine=" << vnni_cos << std::endl;
    std::cout << "[A/B] FP32 vs ref:  mae=" << fp32_mae
              << " rmse=" << fp32_rmse << std::endl;
    std::cout << "[A/B] VNNI vs FP32: mae=" << ab_mae << std::endl;

    // On AVX512-VNNI hardware, the two paths MUST produce different outputs.
    // The int16 quantisation introduces measurable error vs FP32.
    if (cpu_supports_avx512_vnni())
    {
        EXPECT_GT(ab_mae, 0.0f)
            << "VNNI and FP32 outputs are identical — VNNI path is NOT executing!";
        // VNNI error should be meaningfully larger than FP32 rounding error
        EXPECT_GT(vnni_mae, fp32_mae)
            << "VNNI error is not larger than FP32 error — VNNI path may not be active";
    }

    // Accuracy bounds for production VNNI path
    EXPECT_LE(vnni_mae, VNNI_MAX_ABS_TOLERANCE)
        << "VNNI output deviates too much from reference";
    EXPECT_GE(vnni_cos, VNNI_COSINE_THRESHOLD)
        << "VNNI output cosine similarity too low";
}

// ===========================================================================
// 10. Null / Invalid Input Handling
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, NullPointers_ReturnFalse)
{
    float dummy[64] = {};
    EXPECT_FALSE(kernel_.compute(nullptr, dummy, dummy, dummy,
                                 1, 1, 1, 64, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
    EXPECT_FALSE(kernel_.compute(dummy, nullptr, dummy, dummy,
                                 1, 1, 1, 64, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
    EXPECT_FALSE(kernel_.compute(dummy, dummy, nullptr, dummy,
                                 1, 1, 1, 64, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
    EXPECT_FALSE(kernel_.compute(dummy, dummy, dummy, nullptr,
                                 1, 1, 1, 64, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
}

TEST_F(Test__CPUFlashAttentionKernelT, ZeroDimensions_ReturnFalse)
{
    float dummy[64] = {};
    // seq_len=0
    EXPECT_FALSE(kernel_.compute(dummy, dummy, dummy, dummy,
                                 0, 1, 1, 64, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
    // head_dim=0
    EXPECT_FALSE(kernel_.compute(dummy, dummy, dummy, dummy,
                                 1, 1, 1, 0, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
    // n_heads=0
    EXPECT_FALSE(kernel_.compute(dummy, dummy, dummy, dummy,
                                 1, 0, 1, 64, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
}

TEST_F(Test__CPUFlashAttentionKernelT, InvalidGQARatio_ReturnFalse)
{
    // n_heads not divisible by n_kv_heads
    float dummy[256] = {};
    EXPECT_FALSE(kernel_.compute(dummy, dummy, dummy, dummy,
                                 2, 3, 2, 64, false, -1,
                                 nullptr, nullptr, nullptr, nullptr,
                                 false, nullptr, -1));
}

// ===========================================================================
// 11. supports_device() Tests
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, SupportsDevice)
{
    EXPECT_TRUE(kernel_.supports_device(-1)) << "Should support CPU (device_idx=-1)";
    EXPECT_FALSE(kernel_.supports_device(0)) << "Should NOT support GPU (device_idx=0)";
    EXPECT_FALSE(kernel_.supports_device(1)) << "Should NOT support GPU (device_idx=1)";
}

// ===========================================================================
// 12. Decode with position_offset (KV cache scenario)
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, Decode_PositionOffset_Causal)
{
    // Simulate decode at step 50 with 50 KV entries
    // position_offset = 49 means Q position 0 maps to absolute position 49
    runDecodeAndCompare(1, 50, 4, 4, 64, true, 49,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_PositionOffset_Causal");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_PositionOffset_Large)
{
    // Simulate decode at step 500 with 500 KV entries
    runDecodeAndCompare(1, 500, 4, 2, 128, true, 499,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_PositionOffset_Large");
}

// ===========================================================================
// 13. compute_decode() with multi-token query (prefill continuation)
// ===========================================================================

TEST_F(Test__CPUFlashAttentionKernelT, Decode_MultiTokenQuery)
{
    // Simulate prefill continuation: 4 new Q tokens against 20 KV tokens
    runDecodeAndCompare(4, 20, 4, 2, 64, true, 16,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_MultiTokenQuery");
}

TEST_F(Test__CPUFlashAttentionKernelT, Decode_MultiTokenQuery_NoCausal)
{
    runDecodeAndCompare(4, 20, 4, 2, 64, false, 0,
                        FP32_TOLERANCE, FP32_COSINE_THRESHOLD,
                        "Decode_MultiTokenQuery_NoCausal");
}
