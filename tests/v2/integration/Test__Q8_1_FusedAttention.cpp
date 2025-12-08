/**
 * @file Test__Q8_1_FusedAttention.cpp
 * @brief Integration tests for fully fused Q8_1 attention kernel correctness
 *
 * Validates the QuantisedAttentionJit_Q8_1_Fused kernel against a pure FP32
 * reference implementation. The fused kernel performs:
 *   1. Q8_1 × Q8_1 dot product (Q @ K^T)
 *   2. Online softmax (no intermediate score storage)
 *   3. Online weighted V accumulation
 *   4. Online requantization to Q8_1 output
 *
 * The FP32 reference uses:
 *   1. FP32 GEMM for Q @ K^T
 *   2. FP32 Softmax (row-wise)
 *   3. FP32 GEMM for scores @ V
 *   4. Final FP32 output (then quantized for comparison)
 *
 * Metrics computed:
 *   - Relative L2 error: ||fused - ref||_2 / ||ref||_2
 *   - Cosine similarity: (fused · ref) / (||fused|| × ||ref||)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h"
#include "kernels/cpu/ops/CPUSoftmaxKernelTyped.h"
#include "utils/CPUFeatures.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

/**
 * @brief Model attention configuration for test cases
 */
struct TestModelConfig
{
    std::string name;
    int head_dim; ///< Dimension per head (must be multiple of 32)
};

// Model configurations
static const TestModelConfig QWEN_0_5B = {"Qwen-0.5B", 64};
static const TestModelConfig QWEN_7B = {"Qwen-7B", 128};

/**
 * @brief Test fixture for Q8_1 Fused Attention kernel correctness
 */
class Test__Q8_1_FusedAttention : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    // Correctness thresholds for Q8_1 quantization
    // Q8_1 has ~0.4% quantization error per operation, compounded through attention
    static constexpr double MIN_COSINE_SIM = 0.98;   // Allow for quantization noise
    static constexpr double MAX_REL_L2_ERROR = 0.15; // 15% relative L2 (quantization accumulates)

    // Stricter thresholds for small sequences (less error accumulation)
    static constexpr double MIN_COSINE_SIM_SMALL = 0.99;
    static constexpr double MAX_REL_L2_ERROR_SMALL = 0.10;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    /**
     * @brief Compute pure FP32 reference attention
     *
     * Implements: output = softmax(Q @ K^T * scale) @ V
     *
     * @param M Sequence length (Q rows)
     * @param N KV length (K/V rows)
     * @param D Head dimension
     * @param Q_fp32 Query [M, D]
     * @param K_fp32 Key [N, D]
     * @param V_fp32 Value [N, D]
     * @param scale Attention scale (1/sqrt(D))
     * @param output_fp32 Output [M, D]
     */
    void compute_fp32_reference(
        int M, int N, int D,
        const float *Q_fp32, const float *K_fp32, const float *V_fp32,
        float scale,
        float *output_fp32)
    {
        // Step 1: Compute Q @ K^T -> scores [M, N]
        std::vector<float> scores(M * N);
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                {
                    dot += Q_fp32[i * D + d] * K_fp32[j * D + d];
                }
                scores[i * N + j] = dot * scale;
            }
        }

        // Step 2: Row-wise softmax
        for (int i = 0; i < M; ++i)
        {
            // Find max for numerical stability
            float max_val = scores[i * N];
            for (int j = 1; j < N; ++j)
            {
                max_val = std::max(max_val, scores[i * N + j]);
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                scores[i * N + j] = std::exp(scores[i * N + j] - max_val);
                sum_exp += scores[i * N + j];
            }

            // Normalize
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j < N; ++j)
            {
                scores[i * N + j] *= inv_sum;
            }
        }

        // Step 3: scores @ V -> output [M, D]
        for (int i = 0; i < M; ++i)
        {
            for (int d = 0; d < D; ++d)
            {
                float acc = 0.0f;
                for (int j = 0; j < N; ++j)
                {
                    acc += scores[i * N + j] * V_fp32[j * D + d];
                }
                output_fp32[i * D + d] = acc;
            }
        }
    }

    /**
     * @brief Dequantize Q8_1 blocks to FP32 for comparison
     */
    void dequant_q8_1_to_fp32(const Q8_1Block *blocks, int num_rows, int head_dim, float *out_fp32)
    {
        int num_blocks = head_dim / 32;
        for (int row = 0; row < num_rows; ++row)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                const Q8_1Block &block = blocks[row * num_blocks + b];
                float d = simd::fp16_to_fp32(block.d);
                for (int i = 0; i < 32; ++i)
                {
                    out_fp32[row * head_dim + b * 32 + i] = static_cast<float>(block.qs[i]) * d;
                }
            }
        }
    }

    /**
     * @brief Quantize FP32 to Q8_1 blocks
     */
    void quantize_fp32_to_q8_1(const float *fp32, int num_rows, int head_dim, Q8_1Block *blocks)
    {
        int num_blocks = head_dim / 32;
        for (int row = 0; row < num_rows; ++row)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                Q8_1Block &block = blocks[row * num_blocks + b];
                const float *src = fp32 + row * head_dim + b * 32;

                // Find max abs value
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(src[i]));
                }

                // Compute scale
                float d = max_abs / 127.0f;
                float inv_d = (d > 1e-10f) ? (1.0f / d) : 0.0f;
                block.d = simd::fp32_to_fp16(d);

                // Quantize and compute sum
                int16_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(std::clamp(src[i] * inv_d, -127.0f, 127.0f)));
                    block.qs[i] = q;
                    sum_qs += q;
                }
                block.sum_qs = sum_qs;
            }
        }
    }

    /**
     * @brief Compute relative L2 error and cosine similarity
     * @return pair of (rel_l2_error, cosine_similarity)
     */
    std::pair<double, double> compute_metrics(
        const float *actual, const float *reference, int count)
    {
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        double dot_prod = 0.0;
        double norm_act = 0.0;
        double norm_ref = 0.0;

        for (int i = 0; i < count; ++i)
        {
            double a = actual[i];
            double r = reference[i];
            double diff = a - r;

            sum_sq_diff += diff * diff;
            sum_sq_ref += r * r;

            dot_prod += a * r;
            norm_act += a * a;
            norm_ref += r * r;
        }

        double rel_l2 = (sum_sq_ref > 1e-12) ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;
        double cosine = (norm_act > 1e-12 && norm_ref > 1e-12)
                            ? dot_prod / (std::sqrt(norm_act) * std::sqrt(norm_ref))
                            : 0.0;

        return {rel_l2, cosine};
    }

    /**
     * @brief Run fused attention test
     */
    void run_fused_attention_test(const TestModelConfig &model, int seq_len, int kv_len)
    {
        const int M = seq_len;
        const int N = kv_len;
        const int D = model.head_dim;
        const int num_blocks = D / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        // Generate random FP32 data
        std::mt19937 gen(42 + seq_len * 1000 + kv_len * 10 + D);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> Q_fp32(M * D);
        std::vector<float> K_fp32(N * D);
        std::vector<float> V_fp32(N * D);

        for (auto &x : Q_fp32)
            x = dist(gen);
        for (auto &x : K_fp32)
            x = dist(gen);
        for (auto &x : V_fp32)
            x = dist(gen);

        // Compute FP32 reference
        std::vector<float> ref_output(M * D);
        compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

        // Quantize inputs to Q8_1
        std::vector<Q8_1Block> Q_q8(M * num_blocks);
        std::vector<Q8_1Block> K_q8(N * num_blocks);
        std::vector<Q8_1Block> V_q8(N * num_blocks);

        quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

        // Output buffer for fused kernel (Q8_1)
        std::vector<Q8_1Block> output_q8(M * num_blocks);
        std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

        // Strides in bytes
        const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

        // Run fused kernel using reference implementation
        // (JIT kernel requires specific head_dim, use reference for general testing)
        fused_q8_1_attention_reference(
            Q_q8.data(), K_q8.data(), V_q8.data(), output_q8.data(),
            M, N, D,
            num_blocks, num_blocks, num_blocks, num_blocks,
            scale, nullptr, 0);

        // Dequantize output for comparison
        std::vector<float> fused_output(M * D);
        dequant_q8_1_to_fp32(output_q8.data(), M, D, fused_output.data());

        // Compute metrics
        auto [rel_l2, cosine] = compute_metrics(fused_output.data(), ref_output.data(), M * D);

        // Choose thresholds based on sequence length
        double min_cosine = (M * N < 256) ? MIN_COSINE_SIM_SMALL : MIN_COSINE_SIM;
        double max_l2 = (M * N < 256) ? MAX_REL_L2_ERROR_SMALL : MAX_REL_L2_ERROR;

        // Print results
        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "  " << model.name << " M=" << M << " N=" << N << " D=" << D
                      << " | Cosine: " << cosine
                      << " | Rel L2: " << rel_l2 << std::endl;
        }

        // Assert correctness
        EXPECT_GE(cosine, min_cosine)
            << model.name << " M=" << M << " N=" << N << " D=" << D
            << ": Cosine similarity " << cosine << " below threshold " << min_cosine;

        EXPECT_LE(rel_l2, max_l2)
            << model.name << " M=" << M << " N=" << N << " D=" << D
            << ": Relative L2 error " << rel_l2 << " above threshold " << max_l2;
    }

    /**
     * @brief Test the JIT kernel specifically (only for supported head_dim)
     */
    void run_jit_kernel_test(int head_dim, int seq_len, int kv_len)
    {
        const int M = seq_len;
        const int N = kv_len;
        const int D = head_dim;
        const int num_blocks = D / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        // Generate random FP32 data
        std::mt19937 gen(12345 + seq_len * 100 + kv_len);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<float> Q_fp32(M * D);
        std::vector<float> K_fp32(N * D);
        std::vector<float> V_fp32(N * D);

        for (auto &x : Q_fp32)
            x = dist(gen);
        for (auto &x : K_fp32)
            x = dist(gen);
        for (auto &x : V_fp32)
            x = dist(gen);

        // Compute FP32 reference
        std::vector<float> ref_output(M * D);
        compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

        // Quantize inputs to Q8_1
        std::vector<Q8_1Block> Q_q8(M * num_blocks);
        std::vector<Q8_1Block> K_q8(N * num_blocks);
        std::vector<Q8_1Block> V_q8(N * num_blocks);

        quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

        // Create JIT kernel
        QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
        auto kernel_func = jit_kernel.get_kernel();

        // Output buffer
        std::vector<Q8_1Block> output_q8(M * num_blocks);
        std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

        // Strides in bytes
        const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

        // Build params
        FusedQ8_1AttentionParams params;
        params.Q = Q_q8.data();
        params.K = K_q8.data();
        params.V = V_q8.data();
        params.output = output_q8.data();
        params.M = M;
        params.N = N;
        params.head_dim = D;
        params.Q_stride_bytes = stride_bytes;
        params.K_stride_bytes = stride_bytes;
        params.V_stride_bytes = stride_bytes;
        params.output_stride_bytes = stride_bytes;
        params.scale = scale;
        params.mask = nullptr;
        params.mask_stride = 0;

        // Execute JIT kernel
        kernel_func(&params);

        // Dequantize output for comparison
        std::vector<float> jit_output(M * D);
        dequant_q8_1_to_fp32(output_q8.data(), M, D, jit_output.data());

        // Compute metrics
        auto [rel_l2, cosine] = compute_metrics(jit_output.data(), ref_output.data(), M * D);

        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "  JIT D=" << D << " M=" << M << " N=" << N
                      << " | Cosine: " << cosine
                      << " | Rel L2: " << rel_l2 << std::endl;
        }

        // JIT kernel should match reference implementation closely
        EXPECT_GE(cosine, 0.95)
            << "JIT D=" << D << " M=" << M << " N=" << N
            << ": Cosine similarity " << cosine << " below threshold 0.95";

        EXPECT_LE(rel_l2, 0.20)
            << "JIT D=" << D << " M=" << M << " N=" << N
            << ": Relative L2 error " << rel_l2 << " above threshold 0.20";
    }
};

// =============================================================================
// Reference Implementation Tests (all head_dim supported)
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_SingleToken)
{
    // Single token decode scenario
    run_fused_attention_test(QWEN_0_5B, 1, 64);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_SmallPrefill)
{
    // Small prefill
    run_fused_attention_test(QWEN_0_5B, 8, 8);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_MediumPrefill)
{
    // Medium prefill
    run_fused_attention_test(QWEN_0_5B, 32, 32);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_LargePrefill)
{
    // Larger prefill
    run_fused_attention_test(QWEN_0_5B, 64, 64);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_Decode_LongContext)
{
    // Decode with long KV cache
    run_fused_attention_test(QWEN_0_5B, 1, 256);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen7B_SingleToken)
{
    // Single token with larger head_dim
    run_fused_attention_test(QWEN_7B, 1, 64);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen7B_SmallPrefill)
{
    run_fused_attention_test(QWEN_7B, 8, 8);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen7B_MediumPrefill)
{
    run_fused_attention_test(QWEN_7B, 32, 32);
}

// =============================================================================
// JIT vs C++ Reference Debug Test
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_Debug)
{
    // Compare JIT kernel against C++ reference with random data
    const int M = 1;
    const int N = 4;
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Use random data like the failing tests
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Q_fp32(M * D);
    std::vector<float> K_fp32(N * D);
    std::vector<float> V_fp32(N * D);

    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Quantize inputs to Q8_1
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Debug: Print Q8_1 block info
    std::cerr << "\n=== Q8_1 Block Debug ===" << std::endl;
    for (int b = 0; b < num_blocks; ++b)
    {
        std::cerr << "Q Block " << b << ": d=" << simd::fp16_to_fp32(Q_q8[b].d)
                  << " sum_qs=" << Q_q8[b].sum_qs;
        std::cerr << " qs[0..3]=" << (int)Q_q8[b].qs[0] << "," << (int)Q_q8[b].qs[1]
                  << "," << (int)Q_q8[b].qs[2] << "," << (int)Q_q8[b].qs[3] << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        for (int b = 0; b < num_blocks; ++b)
        {
            std::cerr << "K[" << n << "] Block " << b << ": d="
                      << simd::fp16_to_fp32(K_q8[n * num_blocks + b].d)
                      << " sum_qs=" << K_q8[n * num_blocks + b].sum_qs << std::endl;
        }
    }

    // Manually compute expected dot product for Q[0] · K[0]
    std::cerr << "\n=== Manual dot computation ===" << std::endl;
    float ref_dot = 0.0f;
    float jit_dot_no_scale = 0.0f;
    for (int b = 0; b < num_blocks; ++b)
    {
        float d_q = simd::fp16_to_fp32(Q_q8[b].d);
        float d_k = simd::fp16_to_fp32(K_q8[b].d);

        // Reference: signed × signed
        int32_t ref_acc = 0;
        for (int i = 0; i < 32; ++i)
        {
            ref_acc += static_cast<int32_t>(Q_q8[b].qs[i]) * static_cast<int32_t>(K_q8[b].qs[i]);
        }
        float ref_block = static_cast<float>(ref_acc) * d_q * d_k;
        ref_dot += ref_block;

        // JIT simulation: unsigned × signed with correction
        int32_t jit_acc = 0;
        for (int i = 0; i < 32; ++i)
        {
            uint8_t q_unsigned = static_cast<uint8_t>(static_cast<int>(Q_q8[b].qs[i]) + 128);
            int8_t k_signed = K_q8[b].qs[i];
            jit_acc += static_cast<int32_t>(q_unsigned) * static_cast<int32_t>(k_signed);
        }
        float correction = 128.0f * static_cast<float>(K_q8[b].sum_qs);
        float jit_block = (static_cast<float>(jit_acc) - correction) * d_q * d_k;
        jit_dot_no_scale += jit_block;

        std::cerr << "Block " << b << ": ref_acc=" << ref_acc << " jit_acc=" << jit_acc
                  << " correction=" << correction << " d_q*d_k=" << (d_q * d_k)
                  << " ref_block=" << ref_block << " jit_block=" << jit_block << std::endl;
    }
    std::cerr << "Total: ref_dot=" << ref_dot << " jit_dot=" << jit_dot_no_scale << std::endl;
    std::cerr << "After scale: ref=" << (ref_dot * scale) << " jit=" << (jit_dot_no_scale * scale) << std::endl;

    // Run C++ reference
    std::vector<Q8_1Block> ref_output_q8(M * num_blocks);
    std::memset(ref_output_q8.data(), 0, ref_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), ref_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> ref_output(M * D);
    dequant_q8_1_to_fp32(ref_output_q8.data(), M, D, ref_output.data());

    // Run JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    std::cerr << "\n=== Output Comparison ===" << std::endl;
    std::cerr << "Ref output[0..7]: ";
    for (int i = 0; i < 8; ++i)
        std::cerr << ref_output[i] << " ";
    std::cerr << std::endl;
    std::cerr << "JIT output[0..7]: ";
    for (int i = 0; i < 8; ++i)
        std::cerr << jit_output[i] << " ";
    std::cerr << std::endl;

    auto [rel_l2, cosine] = compute_metrics(jit_output.data(), ref_output.data(), M * D);
    std::cerr << "\nCosine: " << cosine << " Rel L2: " << rel_l2 << std::endl;

    EXPECT_GE(cosine, 0.99) << "JIT vs C++ reference: Cosine too low";
    EXPECT_LE(rel_l2, 0.05) << "JIT vs C++ reference: L2 error too high";
}

// =============================================================================
// JIT Kernel Tests (specific head_dim)
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, JIT_D64_SingleToken)
{
    run_jit_kernel_test(64, 1, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D64_SmallBatch)
{
    run_jit_kernel_test(64, 4, 16);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D64_MediumBatch)
{
    run_jit_kernel_test(64, 8, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D128_SingleToken)
{
    run_jit_kernel_test(128, 1, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D128_SmallBatch)
{
    run_jit_kernel_test(128, 4, 16);
}

// Test D=256 (8 blocks) - validates dynamic stack spilling for arbitrary head_dim
TEST_F(Test__Q8_1_FusedAttention, JIT_D256_SingleToken)
{
    run_jit_kernel_test(256, 1, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D256_SmallBatch)
{
    run_jit_kernel_test(256, 4, 16);
}

// Test D=192 (6 blocks) - odd number of spilled blocks
TEST_F(Test__Q8_1_FusedAttention, JIT_D192_SingleToken)
{
    run_jit_kernel_test(192, 1, 32);
}

// Test D=512 (16 blocks) - stress test for large head dimensions
TEST_F(Test__Q8_1_FusedAttention, JIT_D512_SingleToken)
{
    run_jit_kernel_test(512, 1, 16); // Smaller N for faster test
}

// Diagnostic test: Compare JIT vs C++ Q8_1 reference (not FP32)
// This isolates JIT correctness from quantization error
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_D128_Batch)
{
    const int M = 4;
    const int N = 16;
    const int D = 128;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Generate random FP32 data
    std::mt19937 gen(98765);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Q_fp32(M * D);
    std::vector<float> K_fp32(N * D);
    std::vector<float> V_fp32(N * D);

    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Quantize inputs to Q8_1
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // Run C++ Q8_1 reference
    std::vector<Q8_1Block> cpp_output_q8(M * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> cpp_output(M * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), M, D, cpp_output.data());

    // Run JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = stride_bytes;
    params.K_stride_bytes = stride_bytes;
    params.V_stride_bytes = stride_bytes;
    params.output_stride_bytes = stride_bytes;
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // Compare JIT vs C++ reference (both use Q8_1, should match very closely)
    auto [rel_l2_jit_cpp, cosine_jit_cpp] = compute_metrics(jit_output.data(), cpp_output.data(), M * D);

    // Also compute vs FP32 for context
    std::vector<float> fp32_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, fp32_output.data());
    auto [rel_l2_jit_fp32, cosine_jit_fp32] = compute_metrics(jit_output.data(), fp32_output.data(), M * D);
    auto [rel_l2_cpp_fp32, cosine_cpp_fp32] = compute_metrics(cpp_output.data(), fp32_output.data(), M * D);

    if (rank_ == 0)
    {
        std::cerr << "\n=== JIT vs C++ vs FP32 Comparison (D=" << D << " M=" << M << " N=" << N << ") ===" << std::endl;
        std::cerr << "  JIT vs C++ Ref:  Cosine=" << std::fixed << std::setprecision(6) << cosine_jit_cpp
                  << "  RelL2=" << rel_l2_jit_cpp << std::endl;
        std::cerr << "  JIT vs FP32:     Cosine=" << cosine_jit_fp32 << "  RelL2=" << rel_l2_jit_fp32 << std::endl;
        std::cerr << "  C++ vs FP32:     Cosine=" << cosine_cpp_fp32 << "  RelL2=" << rel_l2_cpp_fp32 << std::endl;

        // Per-row analysis
        std::cerr << "\n  Per-row analysis:" << std::endl;
        for (int m = 0; m < M; ++m)
        {
            auto [row_l2, row_cos] = compute_metrics(
                jit_output.data() + m * D, cpp_output.data() + m * D, D);
            std::cerr << "    Row " << m << ": Cosine=" << row_cos << " RelL2=" << row_l2 << std::endl;
        }
    }

    // JIT should match C++ reference very closely (both Q8_1)
    // Note: RelL2 tolerance relaxed from 0.001 to 0.005 after fixing the zmm22/zmm23
    // clobber bug. The JIT uses faster exp approximation which introduces small errors.
    EXPECT_GE(cosine_jit_cpp, 0.9999) << "JIT vs C++ reference should be near-identical";
    EXPECT_LE(rel_l2_jit_cpp, 0.005) << "JIT vs C++ reference L2 should be small";
}

// Debug test for D=128 to identify which blocks are incorrect
TEST_F(Test__Q8_1_FusedAttention, JIT_D128_BlockAnalysis)
{
    const int M = 1;
    const int N = 4; // Small N for simpler analysis
    const int D = 128;
    const int num_blocks = D / 32; // 4 blocks
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Use uniform value for simpler analysis
    std::vector<float> Q_fp32(M * D, 0.1f);
    std::vector<float> K_fp32(N * D, 0.1f);
    std::vector<float> V_fp32(N * D, 0.1f); // All same value

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Run reference
    std::vector<Q8_1Block> ref_output_q8(M * num_blocks);
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), ref_output_q8.data(),
        M, N, D, num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> ref_output(M * D);
    dequant_q8_1_to_fp32(ref_output_q8.data(), M, D, ref_output.data());

    // Run JIT
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // Analyze per-block
    std::cerr << "\n=== D=128 Block-by-Block Analysis ===" << std::endl;
    for (int b = 0; b < num_blocks; ++b)
    {
        std::cerr << "\nBlock " << b << " (elements " << (b * 32) << "-" << (b * 32 + 31) << "):" << std::endl;
        std::cerr << "  Ref[" << b * 32 << "..+3]: "
                  << ref_output[b * 32] << " " << ref_output[b * 32 + 1] << " "
                  << ref_output[b * 32 + 2] << " " << ref_output[b * 32 + 3] << std::endl;
        std::cerr << "  JIT[" << b * 32 << "..+3]: "
                  << jit_output[b * 32] << " " << jit_output[b * 32 + 1] << " "
                  << jit_output[b * 32 + 2] << " " << jit_output[b * 32 + 3] << std::endl;

        // Compute per-block metrics
        float block_sum_sq_diff = 0, block_sum_sq_ref = 0;
        for (int i = 0; i < 32; ++i)
        {
            float diff = jit_output[b * 32 + i] - ref_output[b * 32 + i];
            block_sum_sq_diff += diff * diff;
            block_sum_sq_ref += ref_output[b * 32 + i] * ref_output[b * 32 + i];
        }
        float block_rel_l2 = std::sqrt(block_sum_sq_diff / (block_sum_sq_ref + 1e-10f));
        std::cerr << "  Block Rel L2: " << block_rel_l2 << std::endl;
    }

    // Overall metrics
    auto [rel_l2, cosine] = compute_metrics(jit_output.data(), ref_output.data(), M * D);
    std::cerr << "\nOverall Cosine: " << cosine << " Rel L2: " << rel_l2 << std::endl;

    EXPECT_GE(cosine, 0.95) << "D=128 JIT: Cosine too low";
}

// =============================================================================
// Numerical Properties Tests
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, SoftmaxRowSumProperty)
{
    // Verify that attention weights sum to 1.0 (approximately)
    // by checking that output is bounded

    const int M = 4;
    const int N = 16;
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Quantize
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    std::vector<Q8_1Block> output_q8(M * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Run fused kernel
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    // Dequantize and check bounds
    std::vector<float> output(M * D);
    dequant_q8_1_to_fp32(output_q8.data(), M, D, output.data());

    // Find V value range
    float v_min = *std::min_element(V_fp32.begin(), V_fp32.end());
    float v_max = *std::max_element(V_fp32.begin(), V_fp32.end());

    // Output should be within V range (weighted average property)
    for (int i = 0; i < M * D; ++i)
    {
        // Allow some slack for quantization noise
        EXPECT_GE(output[i], v_min - 0.5f)
            << "Output[" << i << "] = " << output[i] << " below V_min - 0.5 = " << (v_min - 0.5f);
        EXPECT_LE(output[i], v_max + 0.5f)
            << "Output[" << i << "] = " << output[i] << " above V_max + 0.5 = " << (v_max + 0.5f);
    }
}

TEST_F(Test__Q8_1_FusedAttention, IdentityKVTest)
{
    // When K = Q and V is identity-like, output should roughly equal input
    // This tests the softmax concentration property

    const int M = 2;
    const int N = 2;
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Create Q with distinct patterns
    std::vector<float> Q_fp32(M * D, 0.0f);
    for (int i = 0; i < D; ++i)
    {
        Q_fp32[i] = 1.0f;      // First row: all 1s
        Q_fp32[D + i] = -1.0f; // Second row: all -1s
    }

    // K = Q (same as queries)
    std::vector<float> K_fp32 = Q_fp32;

    // V = identity pattern
    std::vector<float> V_fp32(N * D, 0.0f);
    for (int i = 0; i < D; ++i)
    {
        V_fp32[i] = 1.0f;      // First V row
        V_fp32[D + i] = -1.0f; // Second V row
    }

    // Quantize
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    std::vector<Q8_1Block> output_q8(M * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Run fused kernel
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    // Dequantize
    std::vector<float> output(M * D);
    dequant_q8_1_to_fp32(output_q8.data(), M, D, output.data());

    // When Q[0] · K[0] >> Q[0] · K[1], softmax should concentrate on first V row
    // So output[0] should be close to V[0] = all 1s
    float avg_row0 = 0.0f;
    for (int i = 0; i < D; ++i)
    {
        avg_row0 += output[i];
    }
    avg_row0 /= D;

    // Should be closer to 1.0 than -1.0
    EXPECT_GT(avg_row0, 0.0f)
        << "First output row average should be positive (close to V[0]=1), got " << avg_row0;

    float avg_row1 = 0.0f;
    for (int i = 0; i < D; ++i)
    {
        avg_row1 += output[D + i];
    }
    avg_row1 /= D;

    // Should be closer to -1.0 than 1.0
    EXPECT_LT(avg_row1, 0.0f)
        << "Second output row average should be negative (close to V[1]=-1), got " << avg_row1;
}

// Diagnostic: M=2 N=1 test to isolate row-by-row correctness
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_M2_N1)
{
    const int M = 2;
    const int N = 1; // Single K/V position - no softmax complexity
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::mt19937 gen(55555);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Q_fp32(M * D);
    std::vector<float> K_fp32(N * D);
    std::vector<float> V_fp32(N * D);

    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // C++ Q8_1 reference
    std::vector<Q8_1Block> cpp_output_q8(M * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> cpp_output(M * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), M, D, cpp_output.data());

    // JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = stride_bytes;
    params.K_stride_bytes = stride_bytes;
    params.V_stride_bytes = stride_bytes;
    params.output_stride_bytes = stride_bytes;
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    if (rank_ == 0)
    {
        std::cerr << "\n=== M=2 N=1 per-row analysis ===" << std::endl;
        for (int m = 0; m < M; ++m)
        {
            auto [row_l2, row_cos] = compute_metrics(
                jit_output.data() + m * D, cpp_output.data() + m * D, D);
            std::cerr << "  Row " << m << ": Cosine=" << std::fixed << std::setprecision(6)
                      << row_cos << " RelL2=" << row_l2 << std::endl;

            // Print first few values
            std::cerr << "    CPP[" << m << "][0..7]: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << cpp_output[m * D + i] << " ";
            std::cerr << std::endl;
            std::cerr << "    JIT[" << m << "][0..7]: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << jit_output[m * D + i] << " ";
            std::cerr << std::endl;
        }
    }

    // Both rows should match very closely
    for (int m = 0; m < M; ++m)
    {
        auto [row_l2, row_cos] = compute_metrics(
            jit_output.data() + m * D, cpp_output.data() + m * D, D);
        EXPECT_GE(row_cos, 0.9999) << "Row " << m << " should match";
    }
}

// Diagnostic: M=2 N=2 test - minimum softmax case
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_M2_N2)
{
    const int M = 2;
    const int N = 2; // Two K/V positions - softmax is meaningful
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Use deterministic values for easier debugging
    // Q: row 0 = [1,0,0,...], row 1 = [0,1,0,...]
    // K: row 0 = [1,0,0,...], row 1 = [0,1,0,...]
    // V: row 0 = [1,1,1,...], row 1 = [2,2,2,...]
    std::vector<float> Q_fp32(M * D, 0.0f);
    std::vector<float> K_fp32(N * D, 0.0f);
    std::vector<float> V_fp32(N * D, 0.0f);

    // Q[0] and Q[1] both have 1.0 at position 0 (IDENTICAL rows)
    Q_fp32[0] = 1.0f;
    Q_fp32[D + 0] = 1.0f; // Same as Q[0]!

    // K[0] has 1.0 at position 0, K[1] has 1.0 at position 1
    K_fp32[0] = 1.0f;
    K_fp32[D + 1] = 1.0f;

    // V[0] = all 0s, V[1] = all 1s (distinct values to check V selection)
    for (int i = 0; i < D; ++i)
    {
        V_fp32[i] = 0.0f;     // V[0] = 0
        V_fp32[D + i] = 1.0f; // V[1] = 1
    }

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // C++ Q8_1 reference
    std::vector<Q8_1Block> cpp_output_q8(M * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> cpp_output(M * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), M, D, cpp_output.data());

    // JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = stride_bytes;
    params.K_stride_bytes = stride_bytes;
    params.V_stride_bytes = stride_bytes;
    params.output_stride_bytes = stride_bytes;
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // Both rows should match very closely
    for (int m = 0; m < M; ++m)
    {
        auto [row_l2, row_cos] = compute_metrics(
            jit_output.data() + m * D, cpp_output.data() + m * D, D);
        EXPECT_GE(row_cos, 0.9999) << "Row " << m << " should match";
    }
}

// Diagnostic: Test M=2 but call kernel TWICE with M=1 each
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_M1_TwoCalls)
{
    const int D = 64;
    const int N = 2;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Same setup as other tests
    std::vector<float> Q_fp32(2 * D, 0.0f);
    std::vector<float> K_fp32(N * D, 0.0f);
    std::vector<float> V_fp32(N * D, 0.0f);

    Q_fp32[0] = 1.0f;     // Q[0]: nonzero at position 0
    Q_fp32[D + 1] = 1.0f; // Q[1]: nonzero at position 1
    K_fp32[0] = 1.0f;     // K[0]: nonzero at position 0
    K_fp32[D + 1] = 1.0f; // K[1]: nonzero at position 1

    for (int i = 0; i < D; ++i)
    {
        V_fp32[i] = 0.0f;
        V_fp32[D + i] = 1.0f;
    }

    std::vector<Q8_1Block> Q_q8(2 * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), 2, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // Create SINGLE JIT kernel instance
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    // C++ reference for both rows
    std::vector<Q8_1Block> cpp_output_q8(2 * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        2, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);
    std::vector<float> cpp_output(2 * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), 2, D, cpp_output.data());

    // Call JIT kernel TWICE with M=1 each time
    std::vector<Q8_1Block> jit_output_q8(2 * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    // First call: Q[0]
    FusedQ8_1AttentionParams params0;
    params0.Q = Q_q8.data(); // Start at Q[0]
    params0.K = K_q8.data();
    params0.V = V_q8.data();
    params0.output = jit_output_q8.data(); // Output to row 0
    params0.M = 1;
    params0.N = N;
    params0.head_dim = D;
    params0.Q_stride_bytes = stride_bytes;
    params0.K_stride_bytes = stride_bytes;
    params0.V_stride_bytes = stride_bytes;
    params0.output_stride_bytes = stride_bytes;
    params0.scale = scale;
    params0.mask = nullptr;
    params0.mask_stride = 0;

    kernel_func(&params0);

    // Second call: Q[1] (using same kernel instance!)
    FusedQ8_1AttentionParams params1;
    params1.Q = Q_q8.data() + num_blocks; // Start at Q[1]
    params1.K = K_q8.data();
    params1.V = V_q8.data();
    params1.output = jit_output_q8.data() + num_blocks; // Output to row 1
    params1.M = 1;
    params1.N = N;
    params1.head_dim = D;
    params1.Q_stride_bytes = stride_bytes;
    params1.K_stride_bytes = stride_bytes;
    params1.V_stride_bytes = stride_bytes;
    params1.output_stride_bytes = stride_bytes;
    params1.scale = scale;
    params1.mask = nullptr;
    params1.mask_stride = 0;

    kernel_func(&params1);

    std::vector<float> jit_output(2 * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), 2, D, jit_output.data());

    if (rank_ == 0)
    {
        std::cerr << "\n=== Two separate M=1 calls ===" << std::endl;
        for (int m = 0; m < 2; ++m)
        {
            auto [row_l2, row_cos] = compute_metrics(
                jit_output.data() + m * D, cpp_output.data() + m * D, D);
            std::cerr << "  Row " << m << ": Cosine=" << std::fixed << std::setprecision(6)
                      << row_cos << " RelL2=" << row_l2 << std::endl;
            std::cerr << "    CPP: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << cpp_output[m * D + i] << " ";
            std::cerr << std::endl;
            std::cerr << "    JIT: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << jit_output[m * D + i] << " ";
            std::cerr << std::endl;
        }
    }

    for (int m = 0; m < 2; ++m)
    {
        auto [row_l2, row_cos] = compute_metrics(
            jit_output.data() + m * D, cpp_output.data() + m * D, D);
        EXPECT_GE(row_cos, 0.9999) << "Row " << m << " should match";
    }
}

// Main function for GoogleTest with MPI
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
