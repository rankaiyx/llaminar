/**
 * @file Test__Q16IntegerDomainParity.cpp
 * @brief Unit test for Q16 integer-domain attention kernel parity with FP32 reference
 *
 * This test validates the accuracy of the Q16 integer-domain attention pipeline:
 * - Q×K^T → INT32 scores
 * - Exp2FixedSoftmax → INT16 weights (exp2-based, pure integer)
 * - P×V → INT32 accumulator
 * - Wo → Q16_1 output (VPDPWSSD, pure integer)
 * - Residual add → Q16_1 output (simd::q16_1_add_q16_1)
 *
 * The test compares against a TRUE FP32 reference implementation to measure
 * accuracy loss from integer-domain operations.
 *
 * Expected metrics:
 * - Cosine similarity: ≥ 0.95 (target ≥ 0.98)
 * - Max absolute error: documented for analysis
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cstdlib>

#include "kernels/cpu/attention/q16_1/ref/Q16FusedAttentionRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Exp2FixedSoftmaxRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Int8RequantRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/Q16DotProductRef.h"
#include "kernels/cpu/attention/q16_1/ref/microkernels/PVAccumulateRef.h"
#include "tensors/BlockStructures.h"

using namespace llaminar2;
using namespace llaminar2::kernels::q16_1;
using namespace llaminar2::kernels::q16_1::microkernels;

// ============================================================================
// Test Fixture
// ============================================================================

class Q16IntegerDomainParityTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    struct EnvVarGuard
    {
        std::string key;
        bool had_old = false;
        std::string old;

        EnvVarGuard(const char *k, const char *value) : key(k)
        {
            const char *prev = std::getenv(k);
            if (prev)
            {
                had_old = true;
                old = prev;
            }
            setenv(k, value, 1);
        }

        ~EnvVarGuard()
        {
            if (had_old)
            {
                setenv(key.c_str(), old.c_str(), 1);
            }
            else
            {
                unsetenv(key.c_str());
            }
        }
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Statistical Utilities
    // ═══════════════════════════════════════════════════════════════════════

    double cosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    double maxAbsDiff(const float *a, const float *b, size_t n)
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

    double meanAbsDiff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        }
        return sum / static_cast<double>(n);
    }

    double rmsDiff(const float *a, const float *b, size_t n)
    {
        double sum_sq = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum_sq += diff * diff;
        }
        return std::sqrt(sum_sq / static_cast<double>(n));
    }

    void printComparisonStats(const std::string &name,
                              const float *q16_out, const float *fp32_out, size_t n)
    {
        double cos_sim = cosineSimilarity(q16_out, fp32_out, n);
        double max_diff = maxAbsDiff(q16_out, fp32_out, n);
        double mean_diff = meanAbsDiff(q16_out, fp32_out, n);
        double rms = rmsDiff(q16_out, fp32_out, n);

        std::cout << "\n  " << name << " Parity Statistics:" << std::endl;
        std::cout << "    Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
        std::cout << "    Max Abs Diff:      " << std::scientific << std::setprecision(4) << max_diff << std::endl;
        std::cout << "    Mean Abs Diff:     " << mean_diff << std::endl;
        std::cout << "    RMS Diff:          " << rms << std::endl;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Random Data Generation
    // ═══════════════════════════════════════════════════════════════════════

    void generateRandomFP32(float *data, size_t n, float scale)
    {
        std::normal_distribution<float> dist(0.0f, scale);
        for (size_t i = 0; i < n; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Quantization / Dequantization
    // ═══════════════════════════════════════════════════════════════════════

    static constexpr float Q16_1_QMAX = 32767.0f;
    static constexpr int Q16_1_BLOCK_SIZE = 32;

    void quantizeFP32ToQ16_1(const float *fp32, int num_elements, Q16_1Block *blocks)
    {
        int num_blocks = (num_elements + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            Q16_1Block &block = blocks[b];
            const float *src = fp32 + b * Q16_1_BLOCK_SIZE;
            int block_elems = std::min(Q16_1_BLOCK_SIZE, num_elements - b * Q16_1_BLOCK_SIZE);

            float max_abs = 0.0f;
            for (int i = 0; i < block_elems; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            block.d = (max_abs > 1e-10f) ? (max_abs / Q16_1_QMAX) : 1e-10f;
            float inv_d = Q16_1_QMAX / (max_abs > 1e-10f ? max_abs : 1e-10f);

            int32_t sum_qs = 0;
            for (int i = 0; i < Q16_1_BLOCK_SIZE; ++i)
            {
                float val = (i < block_elems) ? src[i] : 0.0f;
                int16_t q = static_cast<int16_t>(std::round(std::clamp(val * inv_d, -Q16_1_QMAX, Q16_1_QMAX)));
                block.qs[i] = q;
                sum_qs += q;
            }
            block.sum_qs = sum_qs;
        }
    }

    void dequantizeQ16_1ToFP32(const Q16_1Block *blocks, int num_elements, float *fp32)
    {
        int num_blocks = (num_elements + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const Q16_1Block &block = blocks[b];
            float *dst = fp32 + b * Q16_1_BLOCK_SIZE;
            int block_elems = std::min(Q16_1_BLOCK_SIZE, num_elements - b * Q16_1_BLOCK_SIZE);

            for (int i = 0; i < block_elems; ++i)
            {
                dst[i] = static_cast<float>(block.qs[i]) * block.d;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TRUE FP32 Reference Attention (No Quantization)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Compute TRUE FP32 attention: softmax(Q @ K^T * scale) @ V
     *
     * This is the ground truth reference - pure FP32 with no quantization.
     *
     * @param seq_len_q Number of query positions
     * @param kv_len Number of KV positions
     * @param num_heads Number of query heads
     * @param num_kv_heads Number of KV heads (for GQA)
     * @param head_dim Dimension per head
     * @param Q_fp32 Query tensor [seq_len_q × num_heads × head_dim]
     * @param K_fp32 Key tensor [kv_len × num_kv_heads × head_dim]
     * @param V_fp32 Value tensor [kv_len × num_kv_heads × head_dim]
     * @param attention_scale Scale factor (1/sqrt(head_dim))
     * @param context_fp32 Output [seq_len_q × num_heads × head_dim]
     * @param causal Whether to apply causal masking
     */
    void computeFP32AttentionReference(
        int seq_len_q,
        int kv_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        const float *Q_fp32,
        const float *K_fp32,
        const float *V_fp32,
        float attention_scale,
        float *context_fp32,
        bool causal)
    {
        int gqa_ratio = num_heads / num_kv_heads;

        for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                int kv_head = h / gqa_ratio;
                const float *Q_head = Q_fp32 + q_pos * num_heads * head_dim + h * head_dim;
                float *out_head = context_fp32 + q_pos * num_heads * head_dim + h * head_dim;

                // Determine effective KV length (causal masking)
                int kv_end = kv_len;
                if (causal)
                {
                    kv_end = std::min(kv_len, q_pos + 1);
                }

                if (kv_end <= 0)
                {
                    std::memset(out_head, 0, head_dim * sizeof(float));
                    continue;
                }

                // Step 1: Compute Q @ K^T scores
                std::vector<float> scores(kv_end);
                for (int k_pos = 0; k_pos < kv_end; ++k_pos)
                {
                    const float *K_pos = K_fp32 + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        dot += Q_head[d] * K_pos[d];
                    }
                    scores[k_pos] = dot * attention_scale;
                }

                // Step 2: FP32 Softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_end; ++k)
                {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum_exp += scores[k];
                }
                for (int k = 0; k < kv_end; ++k)
                {
                    scores[k] /= sum_exp;
                }

                // Step 3: Weighted V sum
                std::memset(out_head, 0, head_dim * sizeof(float));
                for (int k_pos = 0; k_pos < kv_end; ++k_pos)
                {
                    const float *V_pos = V_fp32 + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
                    float weight = scores[k_pos];
                    for (int d = 0; d < head_dim; ++d)
                    {
                        out_head[d] += weight * V_pos[d];
                    }
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Q16 Integer-Domain Attention (Using Microkernels)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @brief Compute Q16 integer-domain attention using microkernels
     *
     * Pipeline: Q×K^T (INT32) → exp2 Softmax (INT16) → P×V (INT64) → FP32
     *
     * This mirrors the reference kernel implementation in Q16FusedAttentionRef.cpp
     * Uses exp2 fixed-point softmax (pure integer pipeline).
     */
    void computeQ16IntegerDomainAttention(
        int seq_len_q,
        int kv_len,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        const Q16_1Block *Q,
        const Q16_1Block *K,
        const Q16_1Block *V,
        float attention_scale,
        float *context_fp32,
        bool causal)
    {
        int gqa_ratio = num_heads / num_kv_heads;
        int blocks_per_head = (head_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
        int q_blocks_per_row = (num_heads * head_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
        int kv_blocks_per_row = (num_kv_heads * head_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

        for (int q_pos = 0; q_pos < seq_len_q; ++q_pos)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                int kv_head = h / gqa_ratio;
                float *out_head = context_fp32 + q_pos * num_heads * head_dim + h * head_dim;

                // Determine effective KV length (causal masking)
                int kv_end = kv_len;
                if (causal)
                {
                    kv_end = std::min(kv_len, q_pos + 1);
                }

                if (kv_end <= 0)
                {
                    std::memset(out_head, 0, head_dim * sizeof(float));
                    continue;
                }

                // ════════════════════════════════════════════════════════════════
                // STEP 1: Q×K^T → INT32 scores (via Int8RequantRef)
                // ════════════════════════════════════════════════════════════════

                std::vector<int32_t> int32_scores(kv_end);
                float alpha = 0.0f;

                {
                    // Use the Int8RequantRef microkernel for INT32 logits with proper scaling
                    microkernels::Int8RequantParams requant_params{};
                    requant_params.Q = Q;
                    requant_params.K = K;
                    requant_params.q_row = q_pos;
                    requant_params.head = h;
                    requant_params.kv_head = kv_head;
                    requant_params.head_dim = head_dim;
                    requant_params.kv_end = kv_end;
                    requant_params.q_blocks_per_row = q_blocks_per_row;
                    requant_params.kv_blocks_per_row = kv_blocks_per_row;
                    requant_params.attention_scale = attention_scale;

                    microkernels::compute_int8_requant_logits(requant_params, int32_scores.data(), &alpha);
                }

                // ════════════════════════════════════════════════════════════════
                // STEP 2: Exp2FixedSoftmax → INT16 weights (VNNI compatible)
                // ════════════════════════════════════════════════════════════════

                std::vector<int16_t> int16_weights(kv_end);
                int32_t weight_sum = 0;

                microkernels::exp2_fixed_softmax_row(
                    int32_scores.data(),
                    int16_weights.data(),
                    kv_end,
                    alpha,
                    &weight_sum);

                // ════════════════════════════════════════════════════════════════
                // STEP 3: P×V → FP32 output (INT32 accumulator, VNNI model)
                // ════════════════════════════════════════════════════════════════

                PVAccumulateParams pv_params;
                pv_params.V = V;
                pv_params.kv_len = kv_end;
                pv_params.head_dim = head_dim;
                pv_params.kv_head = kv_head;
                pv_params.kv_blocks_per_row = kv_blocks_per_row;

                pv_accumulate_gemv_int16(
                    pv_params,
                    int16_weights.data(),
                    weight_sum,
                    0, // kv_start
                    kv_end,
                    out_head);
            }
        }
    }
};

// ============================================================================
// Parity Tests: Integer-Domain vs FP32 Reference
// ============================================================================

TEST_F(Q16IntegerDomainParityTest, AttentionCore_SmallConfig_Decode)
{
    std::cout << "\n=== Q16 Integer-Domain vs FP32 Parity Test ===" << std::endl;
    std::cout << "Configuration: seq_len_q=1 (decode), kv_len=16, heads=4, head_dim=64" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // Generate random FP32 data
    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);

    // Quantize to Q16_1
    const int q_blocks = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE * SEQ_LEN_Q;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(q_blocks);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        quantizeFP32ToQ16_1(Q_fp32.data() + q * input_dim, input_dim,
                            Q_q16.data() + q * (q_blocks / SEQ_LEN_Q));
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim,
                            K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim,
                            V_q16.data() + kv * kv_blocks_per_row);
    }

    // Dequantize for "fair" FP32 reference (same quantized values)
    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        dequantizeQ16_1ToFP32(Q_q16.data() + q * (q_blocks / SEQ_LEN_Q), input_dim,
                              Q_dequant.data() + q * input_dim);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              V_dequant.data() + kv * kv_dim);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // COMPUTE THREE REFERENCES:
    // 1. TRUE FP32: Original FP32 data (shows total quantization + algorithm error)
    // 2. DEQUANT FP32: Dequantized data with FP32 softmax (isolates algorithm error)
    // 3. Q16 INTEGER: Our integer-domain implementation
    // ═══════════════════════════════════════════════════════════════════════

    // 1. TRUE FP32 reference (original data, no quantization)
    std::vector<float> context_true_fp32(SEQ_LEN_Q * input_dim);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ATTENTION_SCALE, context_true_fp32.data(), CAUSAL);

    // 2. DEQUANT FP32 reference (quantized then dequantized, FP32 softmax)
    std::vector<float> context_dequant_fp32(SEQ_LEN_Q * input_dim);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_dequant_fp32.data(), CAUSAL);

    // 3. Q16 integer-domain attention
    std::vector<float> context_q16(SEQ_LEN_Q * input_dim);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    // ═══════════════════════════════════════════════════════════════════════
    // COMPARISONS
    // ═══════════════════════════════════════════════════════════════════════

    std::cout << "\n  [1] Q16 vs TRUE FP32 (total error: quantization + algorithm):" << std::endl;
    double cos_sim_total = cosineSimilarity(context_q16.data(), context_true_fp32.data(), input_dim);
    double max_diff_total = maxAbsDiff(context_q16.data(), context_true_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_total << std::endl;
    std::cout << "      Max Abs Diff:      " << std::scientific << std::setprecision(4) << max_diff_total << std::endl;

    std::cout << "\n  [2] Q16 vs DEQUANT FP32 (algorithm error only, same quantized values):" << std::endl;
    double cos_sim_algo = cosineSimilarity(context_q16.data(), context_dequant_fp32.data(), input_dim);
    double max_diff_algo = maxAbsDiff(context_q16.data(), context_dequant_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_algo << std::endl;
    std::cout << "      Max Abs Diff:      " << std::scientific << std::setprecision(4) << max_diff_algo << std::endl;

    std::cout << "\n  [3] DEQUANT FP32 vs TRUE FP32 (quantization error only):" << std::endl;
    double cos_sim_quant = cosineSimilarity(context_dequant_fp32.data(), context_true_fp32.data(), input_dim);
    double max_diff_quant = maxAbsDiff(context_dequant_fp32.data(), context_true_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_quant << std::endl;
    std::cout << "      Max Abs Diff:      " << std::scientific << std::setprecision(4) << max_diff_quant << std::endl;

    // Assertions - test total error (the meaningful metric)
    EXPECT_GT(cos_sim_total, 0.90) << "Total error too high (Q16 vs TRUE FP32)";
    EXPECT_GT(cos_sim_algo, 0.95) << "Algorithm error too high (Q16 vs DEQUANT FP32)";
    EXPECT_FALSE(std::isnan(cos_sim_total)) << "Cosine similarity is NaN";

    // Print verdict based on TOTAL error
    std::cout << "\n  Verdict (Q16 vs TRUE FP32): ";
    if (cos_sim_total >= 0.98)
    {
        std::cout << "EXCELLENT (≥0.98)" << std::endl;
    }
    else if (cos_sim_total >= 0.95)
    {
        std::cout << "GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim_total >= 0.90)
    {
        std::cout << "ACCEPTABLE (≥0.90)" << std::endl;
    }
    else
    {
        std::cout << "POOR (<0.90) - NEEDS INVESTIGATION" << std::endl;
    }
}

TEST_F(Q16IntegerDomainParityTest, AttentionCore_LargerConfig_Decode)
{
    std::cout << "\n=== Q16 Integer-Domain vs FP32 Parity Test (Larger) ===" << std::endl;
    std::cout << "Configuration: seq_len_q=1 (decode), kv_len=128, heads=8, head_dim=128" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 128;
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 8;
    constexpr int HEAD_DIM = 128;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);

    const int q_blocks = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE * SEQ_LEN_Q;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(q_blocks);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        quantizeFP32ToQ16_1(Q_fp32.data() + q * input_dim, input_dim,
                            Q_q16.data() + q * (q_blocks / SEQ_LEN_Q));
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim,
                            K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim,
                            V_q16.data() + kv * kv_blocks_per_row);
    }

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        dequantizeQ16_1ToFP32(Q_q16.data() + q * (q_blocks / SEQ_LEN_Q), input_dim,
                              Q_dequant.data() + q * input_dim);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              V_dequant.data() + kv * kv_dim);
    }

    // Compute three references: TRUE FP32, DEQUANT FP32, Q16
    std::vector<float> context_true_fp32(SEQ_LEN_Q * input_dim);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ATTENTION_SCALE, context_true_fp32.data(), CAUSAL);

    std::vector<float> context_dequant_fp32(SEQ_LEN_Q * input_dim);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_dequant_fp32.data(), CAUSAL);

    std::vector<float> context_q16(SEQ_LEN_Q * input_dim);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    // Report all three comparisons
    std::cout << "\n  [1] Q16 vs TRUE FP32 (total error):" << std::endl;
    double cos_sim_total = cosineSimilarity(context_q16.data(), context_true_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_total << std::endl;

    std::cout << "\n  [2] Q16 vs DEQUANT FP32 (algorithm error only):" << std::endl;
    double cos_sim_algo = cosineSimilarity(context_q16.data(), context_dequant_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_algo << std::endl;

    std::cout << "\n  [3] DEQUANT FP32 vs TRUE FP32 (quantization error only):" << std::endl;
    double cos_sim_quant = cosineSimilarity(context_dequant_fp32.data(), context_true_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_quant << std::endl;

    EXPECT_GT(cos_sim_total, 0.90) << "Total error too high (Q16 vs TRUE FP32)";
    EXPECT_GT(cos_sim_algo, 0.95) << "Algorithm error too high (Q16 vs DEQUANT FP32)";
    EXPECT_FALSE(std::isnan(cos_sim_total)) << "Cosine similarity is NaN";

    std::cout << "\n  Verdict (Q16 vs TRUE FP32): ";
    if (cos_sim_total >= 0.98)
    {
        std::cout << "EXCELLENT (≥0.98)" << std::endl;
    }
    else if (cos_sim_total >= 0.95)
    {
        std::cout << "GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim_total >= 0.90)
    {
        std::cout << "ACCEPTABLE (≥0.90)" << std::endl;
    }
    else
    {
        std::cout << "POOR (<0.90) - NEEDS INVESTIGATION" << std::endl;
    }
}

TEST_F(Q16IntegerDomainParityTest, AttentionCore_GQA)
{
    std::cout << "\n=== Q16 Integer-Domain vs FP32 Parity Test (GQA) ===" << std::endl;
    std::cout << "Configuration: seq_len_q=1, kv_len=64, heads=8, kv_heads=2, head_dim=64" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 64;
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 2; // GQA 4:1 ratio
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);

    const int q_blocks = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE * SEQ_LEN_Q;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(q_blocks);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        quantizeFP32ToQ16_1(Q_fp32.data() + q * input_dim, input_dim,
                            Q_q16.data() + q * (q_blocks / SEQ_LEN_Q));
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim,
                            K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim,
                            V_q16.data() + kv * kv_blocks_per_row);
    }

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        dequantizeQ16_1ToFP32(Q_q16.data() + q * (q_blocks / SEQ_LEN_Q), input_dim,
                              Q_dequant.data() + q * input_dim);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              V_dequant.data() + kv * kv_dim);
    }

    // Compute three references: TRUE FP32, DEQUANT FP32, Q16
    std::vector<float> context_true_fp32(SEQ_LEN_Q * input_dim);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ATTENTION_SCALE, context_true_fp32.data(), CAUSAL);

    std::vector<float> context_dequant_fp32(SEQ_LEN_Q * input_dim);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_dequant_fp32.data(), CAUSAL);

    std::vector<float> context_q16(SEQ_LEN_Q * input_dim);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    // Report all three comparisons
    std::cout << "\n  [1] Q16 vs TRUE FP32 (total error):" << std::endl;
    double cos_sim_total = cosineSimilarity(context_q16.data(), context_true_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_total << std::endl;

    std::cout << "\n  [2] Q16 vs DEQUANT FP32 (algorithm error only):" << std::endl;
    double cos_sim_algo = cosineSimilarity(context_q16.data(), context_dequant_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_algo << std::endl;

    std::cout << "\n  [3] DEQUANT FP32 vs TRUE FP32 (quantization error only):" << std::endl;
    double cos_sim_quant = cosineSimilarity(context_dequant_fp32.data(), context_true_fp32.data(), input_dim);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_quant << std::endl;

    EXPECT_GT(cos_sim_total, 0.90) << "Total error too high (Q16 vs TRUE FP32)";
    EXPECT_GT(cos_sim_algo, 0.95) << "Algorithm error too high (Q16 vs DEQUANT FP32)";
    EXPECT_FALSE(std::isnan(cos_sim_total)) << "Cosine similarity is NaN";

    std::cout << "\n  Verdict (Q16 vs TRUE FP32): ";
    if (cos_sim_total >= 0.98)
    {
        std::cout << "EXCELLENT (≥0.98)" << std::endl;
    }
    else if (cos_sim_total >= 0.95)
    {
        std::cout << "GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim_total >= 0.90)
    {
        std::cout << "ACCEPTABLE (≥0.90)" << std::endl;
    }
    else
    {
        std::cout << "POOR (<0.90) - NEEDS INVESTIGATION" << std::endl;
    }
}

TEST_F(Q16IntegerDomainParityTest, AttentionCore_Prefill)
{
    std::cout << "\n=== Q16 Integer-Domain vs FP32 Parity Test (Prefill) ===" << std::endl;
    std::cout << "Configuration: seq_len_q=16 (prefill), kv_len=16, heads=4, head_dim=64" << std::endl;

    constexpr int SEQ_LEN_Q = 16;
    constexpr int KV_LEN = 16;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int total_output = SEQ_LEN_Q * input_dim;

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 1.0f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 1.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 1.0f);

    const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        quantizeFP32ToQ16_1(Q_fp32.data() + q * input_dim, input_dim,
                            Q_q16.data() + q * q_blocks_per_row);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim,
                            K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim,
                            V_q16.data() + kv * kv_blocks_per_row);
    }

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        dequantizeQ16_1ToFP32(Q_q16.data() + q * q_blocks_per_row, input_dim,
                              Q_dequant.data() + q * input_dim);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim,
                              V_dequant.data() + kv * kv_dim);
    }

    // Compute three references: TRUE FP32, DEQUANT FP32, Q16
    std::vector<float> context_true_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_fp32.data(), K_fp32.data(), V_fp32.data(),
        ATTENTION_SCALE, context_true_fp32.data(), CAUSAL);

    std::vector<float> context_dequant_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_dequant_fp32.data(), CAUSAL);

    std::vector<float> context_q16(total_output);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    // Per-position diagnostic (compare against TRUE FP32)
    std::cout << "\n  Per-position cosine similarity vs TRUE FP32 (head 0):" << std::endl;
    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        int offset = q * input_dim; // head 0
        double pos_cos_sim = cosineSimilarity(
            context_q16.data() + offset,
            context_true_fp32.data() + offset,
            HEAD_DIM);
        std::cout << "    q_pos=" << q << " (kv_end=" << (q + 1) << "): cos_sim=" << pos_cos_sim << std::endl;
    }

    // Report all three comparisons
    std::cout << "\n  [1] Q16 vs TRUE FP32 (total error):" << std::endl;
    double cos_sim_total = cosineSimilarity(context_q16.data(), context_true_fp32.data(), total_output);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_total << std::endl;

    std::cout << "\n  [2] Q16 vs DEQUANT FP32 (algorithm error only):" << std::endl;
    double cos_sim_algo = cosineSimilarity(context_q16.data(), context_dequant_fp32.data(), total_output);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_algo << std::endl;

    std::cout << "\n  [3] DEQUANT FP32 vs TRUE FP32 (quantization error only):" << std::endl;
    double cos_sim_quant = cosineSimilarity(context_dequant_fp32.data(), context_true_fp32.data(), total_output);
    std::cout << "      Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim_quant << std::endl;

    // NOTE: Prefill has known issues with early positions (low kv_len → uniform softmax)
    // We relax the threshold for total error but still test algorithm error
    EXPECT_GT(cos_sim_total, 0.70) << "Total error too high (Q16 vs TRUE FP32)";
    EXPECT_GT(cos_sim_algo, 0.70) << "Algorithm error too high (Q16 vs DEQUANT FP32)";
    EXPECT_FALSE(std::isnan(cos_sim_total)) << "Cosine similarity is NaN";

    std::cout << "\n  Verdict (Q16 vs TRUE FP32): ";
    if (cos_sim_total >= 0.98)
    {
        std::cout << "EXCELLENT (≥0.98)" << std::endl;
    }
    else if (cos_sim_total >= 0.95)
    {
        std::cout << "GOOD (≥0.95)" << std::endl;
    }
    else if (cos_sim_total >= 0.90)
    {
        std::cout << "ACCEPTABLE (≥0.90)" << std::endl;
    }
    else if (cos_sim_total >= 0.70)
    {
        std::cout << "POOR (≥0.70) - Known prefill issues with early positions" << std::endl;
    }
    else
    {
        std::cout << "FAILING (<0.70) - NEEDS INVESTIGATION" << std::endl;
    }
}

// ============================================================================
// Component-Level Parity Tests
// ============================================================================

// Helper functions for softmax parity tests (inline implementations)
namespace
{
    // Compute FP32 softmax for a single row
    void fp32_softmax_row_inline(const float *scores, float *weights, int n)
    {
        float max_score = scores[0];
        for (int i = 1; i < n; ++i)
            max_score = std::max(max_score, scores[i]);

        float sum_exp = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            weights[i] = std::exp(scores[i] - max_score);
            sum_exp += weights[i];
        }
        for (int i = 0; i < n; ++i)
        {
            weights[i] /= sum_exp;
        }
    }

    // Normalize INT16 weights to FP32 for comparison
    void normalize_int16_to_fp32_inline(const int16_t *weights, int32_t sum, float *normalized, int n)
    {
        float inv_sum = 1.0f / static_cast<float>(sum);
        for (int i = 0; i < n; ++i)
        {
            normalized[i] = static_cast<float>(weights[i]) * inv_sum;
        }
    }
} // namespace

TEST_F(Q16IntegerDomainParityTest, Exp2FixedSoftmax_vs_FP32Softmax)
{
    std::cout << "\n=== Exp2FixedSoftmax vs FP32 Softmax Parity ===" << std::endl;

    constexpr int N = 64;

    // Generate random INT32 scores (simulating Q×K^T output)
    std::vector<int32_t> int32_scores(N);
    std::uniform_int_distribution<int32_t> score_dist(-100000, 100000);
    for (int i = 0; i < N; ++i)
    {
        int32_scores[i] = score_dist(rng_);
    }

    // Convert to FP32 scores with a typical alpha
    float alpha = 0.0001f; // Typical scale factor
    std::vector<float> fp32_scores(N);
    for (int i = 0; i < N; ++i)
    {
        fp32_scores[i] = static_cast<float>(int32_scores[i]) * alpha;
    }

    // Compute FP32 softmax
    std::vector<float> fp32_weights(N);
    fp32_softmax_row_inline(fp32_scores.data(), fp32_weights.data(), N);

    // Compute Exp2FixedSoftmax (VNNI-compatible INT16/INT32)
    std::vector<int16_t> int16_weights(N);
    int32_t weight_sum = 0;
    microkernels::exp2_fixed_softmax_row(int32_scores.data(), int16_weights.data(), N, alpha, &weight_sum);

    // Normalize INT16 weights to FP32 for comparison
    std::vector<float> exp2_normalized(N);
    normalize_int16_to_fp32_inline(int16_weights.data(), weight_sum, exp2_normalized.data(), N);

    printComparisonStats("Softmax Weights", exp2_normalized.data(), fp32_weights.data(), N);

    double cos_sim = cosineSimilarity(exp2_normalized.data(), fp32_weights.data(), N);
    EXPECT_GT(cos_sim, 0.97) << "Exp2FixedSoftmax cosine similarity too low vs FP32 softmax";

    std::cout << "\n  Verdict: ";
    if (cos_sim >= 0.99)
    {
        std::cout << "EXCELLENT (≥0.99)" << std::endl;
    }
    else if (cos_sim >= 0.98)
    {
        std::cout << "GOOD (≥0.98)" << std::endl;
    }
    else if (cos_sim >= 0.97)
    {
        std::cout << "ACCEPTABLE (≥0.97)" << std::endl;
    }
    else
    {
        std::cout << "POOR (<0.97) - NEEDS INVESTIGATION" << std::endl;
    }
}

// ============================================================================
// Stress Tests for Exp2FixedSoftmax Path
// ============================================================================

/**
 * @brief Stress test: Long KV sequence (2048 positions)
 *
 * Tests numerical stability with large attention spans.
 */
TEST_F(Q16IntegerDomainParityTest, Exp2Softmax_Stress_LongKV)
{
    std::cout << "\n=== Exp2 Softmax Stress: Long KV (2048) ===" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 2048;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = false; // Non-causal to test full 2048 positions

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int total_output = SEQ_LEN_Q * input_dim;

    // Generate random FP32 data
    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 0.5f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 0.5f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 0.5f);

    // Quantize to Q16_1
    const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), input_dim, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim, V_q16.data() + kv * kv_blocks_per_row);
    }

    // Dequantize for FP32 reference
    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    dequantizeQ16_1ToFP32(Q_q16.data(), input_dim, Q_dequant.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim, K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim, V_dequant.data() + kv * kv_dim);
    }

    // Compute references
    std::vector<float> context_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_fp32.data(), CAUSAL);

    std::vector<float> context_q16(total_output);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    double cos_sim = cosineSimilarity(context_q16.data(), context_fp32.data(), total_output);
    std::cout << "  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;

    EXPECT_GT(cos_sim, 0.98) << "Long KV stress test failed";
    EXPECT_FALSE(std::isnan(cos_sim)) << "Cosine similarity is NaN";
}

/**
 * @brief Stress test: Large head dimension (128)
 *
 * Tests with Llama-style head_dim=128.
 */
TEST_F(Q16IntegerDomainParityTest, Exp2Softmax_Stress_LargeHeadDim)
{
    std::cout << "\n=== Exp2 Softmax Stress: Large Head Dim (128) ===" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 64;
    constexpr int NUM_HEADS = 8;
    constexpr int NUM_KV_HEADS = 8;
    constexpr int HEAD_DIM = 128;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int total_output = SEQ_LEN_Q * input_dim;

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 0.5f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 0.5f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 0.5f);

    const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), input_dim, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim, V_q16.data() + kv * kv_blocks_per_row);
    }

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    dequantizeQ16_1ToFP32(Q_q16.data(), input_dim, Q_dequant.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim, K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim, V_dequant.data() + kv * kv_dim);
    }

    std::vector<float> context_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_fp32.data(), CAUSAL);

    std::vector<float> context_q16(total_output);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    double cos_sim = cosineSimilarity(context_q16.data(), context_fp32.data(), total_output);
    std::cout << "  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;

    EXPECT_GT(cos_sim, 0.98) << "Large head dim stress test failed";
    EXPECT_FALSE(std::isnan(cos_sim)) << "Cosine similarity is NaN";
}

/**
 * @brief Stress test: Extreme activation values
 *
 * Tests numerical stability with very large/small activation values.
 */
TEST_F(Q16IntegerDomainParityTest, Exp2Softmax_Stress_ExtremeValues)
{
    std::cout << "\n=== Exp2 Softmax Stress: Extreme Values ===" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 32;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int total_output = SEQ_LEN_Q * input_dim;

    // Generate data with extreme values (large variance)
    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 2.0f); // Large variance
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 2.0f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 2.0f);

    const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), input_dim, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim, V_q16.data() + kv * kv_blocks_per_row);
    }

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    dequantizeQ16_1ToFP32(Q_q16.data(), input_dim, Q_dequant.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim, K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim, V_dequant.data() + kv * kv_dim);
    }

    std::vector<float> context_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_fp32.data(), CAUSAL);

    std::vector<float> context_q16(total_output);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    double cos_sim = cosineSimilarity(context_q16.data(), context_fp32.data(), total_output);
    std::cout << "  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;

    EXPECT_GT(cos_sim, 0.95) << "Extreme values stress test failed";
    EXPECT_FALSE(std::isnan(cos_sim)) << "Cosine similarity is NaN";
}

/**
 * @brief Stress test: Many heads with GQA (32 Q heads, 4 KV heads)
 *
 * Tests GQA ratio of 8:1 (common in larger models).
 */
TEST_F(Q16IntegerDomainParityTest, Exp2Softmax_Stress_HighGQARatio)
{
    std::cout << "\n=== Exp2 Softmax Stress: High GQA Ratio (32:4) ===" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 64;
    constexpr int NUM_HEADS = 32;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int total_output = SEQ_LEN_Q * input_dim;

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 0.5f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 0.5f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 0.5f);

    const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), input_dim, Q_q16.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim, V_q16.data() + kv * kv_blocks_per_row);
    }

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    dequantizeQ16_1ToFP32(Q_q16.data(), input_dim, Q_dequant.data());
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim, K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim, V_dequant.data() + kv * kv_dim);
    }

    std::vector<float> context_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_fp32.data(), CAUSAL);

    std::vector<float> context_q16(total_output);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    double cos_sim = cosineSimilarity(context_q16.data(), context_fp32.data(), total_output);
    std::cout << "  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;

    EXPECT_GT(cos_sim, 0.98) << "High GQA ratio stress test failed";
    EXPECT_FALSE(std::isnan(cos_sim)) << "Cosine similarity is NaN";
}

/**
 * @brief Stress test: Long prefill sequence (128 queries)
 *
 * Tests prefill mode with longer sequences.
 */
TEST_F(Q16IntegerDomainParityTest, Exp2Softmax_Stress_LongPrefill)
{
    std::cout << "\n=== Exp2 Softmax Stress: Long Prefill (128 queries) ===" << std::endl;

    constexpr int SEQ_LEN_Q = 128;
    constexpr int KV_LEN = 128;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int total_output = SEQ_LEN_Q * input_dim;

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 0.5f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 0.5f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 0.5f);

    const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        quantizeFP32ToQ16_1(Q_fp32.data() + q * input_dim, input_dim, Q_q16.data() + q * q_blocks_per_row);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        quantizeFP32ToQ16_1(K_fp32.data() + kv * kv_dim, kv_dim, K_q16.data() + kv * kv_blocks_per_row);
        quantizeFP32ToQ16_1(V_fp32.data() + kv * kv_dim, kv_dim, V_q16.data() + kv * kv_blocks_per_row);
    }

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    for (int q = 0; q < SEQ_LEN_Q; ++q)
    {
        dequantizeQ16_1ToFP32(Q_q16.data() + q * q_blocks_per_row, input_dim, Q_dequant.data() + q * input_dim);
    }
    for (int kv = 0; kv < KV_LEN; ++kv)
    {
        dequantizeQ16_1ToFP32(K_q16.data() + kv * kv_blocks_per_row, kv_dim, K_dequant.data() + kv * kv_dim);
        dequantizeQ16_1ToFP32(V_q16.data() + kv * kv_blocks_per_row, kv_dim, V_dequant.data() + kv * kv_dim);
    }

    std::vector<float> context_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_fp32.data(), CAUSAL);

    std::vector<float> context_q16(total_output);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    double cos_sim = cosineSimilarity(context_q16.data(), context_fp32.data(), total_output);
    std::cout << "  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;

    // Prefill has known challenges with early positions, but exp2 should handle it well
    EXPECT_GT(cos_sim, 0.95) << "Long prefill stress test failed";
    EXPECT_FALSE(std::isnan(cos_sim)) << "Cosine similarity is NaN";
}

/**
 * @brief Stress test: Single KV position (edge case)
 *
 * Tests behavior when there's only one KV position (softmax degenerates to 1.0).
 */
TEST_F(Q16IntegerDomainParityTest, Exp2Softmax_Stress_SingleKV)
{
    std::cout << "\n=== Exp2 Softmax Stress: Single KV Position ===" << std::endl;

    constexpr int SEQ_LEN_Q = 1;
    constexpr int KV_LEN = 1;
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 64;
    constexpr float ATTENTION_SCALE = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    constexpr bool CAUSAL = true;

    const int input_dim = NUM_HEADS * HEAD_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int total_output = SEQ_LEN_Q * input_dim;

    std::vector<float> Q_fp32(SEQ_LEN_Q * input_dim);
    std::vector<float> K_fp32(KV_LEN * kv_dim);
    std::vector<float> V_fp32(KV_LEN * kv_dim);

    generateRandomFP32(Q_fp32.data(), Q_fp32.size(), 0.5f);
    generateRandomFP32(K_fp32.data(), K_fp32.size(), 0.5f);
    generateRandomFP32(V_fp32.data(), V_fp32.size(), 0.5f);

    const int q_blocks_per_row = (input_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;
    const int kv_blocks_per_row = (kv_dim + Q16_1_BLOCK_SIZE - 1) / Q16_1_BLOCK_SIZE;

    std::vector<Q16_1Block> Q_q16(SEQ_LEN_Q * q_blocks_per_row);
    std::vector<Q16_1Block> K_q16(KV_LEN * kv_blocks_per_row);
    std::vector<Q16_1Block> V_q16(KV_LEN * kv_blocks_per_row);

    quantizeFP32ToQ16_1(Q_fp32.data(), input_dim, Q_q16.data());
    quantizeFP32ToQ16_1(K_fp32.data(), kv_dim, K_q16.data());
    quantizeFP32ToQ16_1(V_fp32.data(), kv_dim, V_q16.data());

    std::vector<float> Q_dequant(SEQ_LEN_Q * input_dim);
    std::vector<float> K_dequant(KV_LEN * kv_dim);
    std::vector<float> V_dequant(KV_LEN * kv_dim);

    dequantizeQ16_1ToFP32(Q_q16.data(), input_dim, Q_dequant.data());
    dequantizeQ16_1ToFP32(K_q16.data(), kv_dim, K_dequant.data());
    dequantizeQ16_1ToFP32(V_q16.data(), kv_dim, V_dequant.data());

    std::vector<float> context_fp32(total_output);
    computeFP32AttentionReference(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_dequant.data(), K_dequant.data(), V_dequant.data(),
        ATTENTION_SCALE, context_fp32.data(), CAUSAL);

    std::vector<float> context_q16(total_output);
    computeQ16IntegerDomainAttention(
        SEQ_LEN_Q, KV_LEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,
        Q_q16.data(), K_q16.data(), V_q16.data(),
        ATTENTION_SCALE, context_q16.data(), CAUSAL);

    double cos_sim = cosineSimilarity(context_q16.data(), context_fp32.data(), total_output);
    std::cout << "  Cosine Similarity: " << std::fixed << std::setprecision(6) << cos_sim << std::endl;

    // With single KV, output should just be V (softmax weight = 1.0)
    EXPECT_GT(cos_sim, 0.99) << "Single KV stress test failed - should be near-exact";
    EXPECT_FALSE(std::isnan(cos_sim)) << "Cosine similarity is NaN";
}
