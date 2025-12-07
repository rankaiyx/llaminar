/**
 * @file Test__Q8_1_OnlineSoftmax_Unit.cpp
 * @brief Unit tests for QuantisedGemmJit_Q8_1_OnlineSoftmax JIT kernel
 *
 * Tests the Q8_1 x Q8_1 attention kernel (Q @ K^T with fused online softmax)
 * without loading models. Validates:
 * - JIT kernel instantiation and code generation
 * - OnlineSoftmaxParams structure
 * - Numerical correctness for small matrices
 * - Edge cases (M=1, M=4, odd K blocks, tail N handling)
 * - Softmax properties (non-negative, row sums ≈ 1.0)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>

#include "kernels/cpu/gemm_v4/QuantisedGemmJit_Q8_1_OnlineSoftmax.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

/**
 * @brief Test fixture for Q8_1 OnlineSoftmax unit tests
 */
class Test__Q8_1_OnlineSoftmax_Unit : public ::testing::Test
{
protected:
    // Fixed seed for reproducibility
    std::mt19937 gen_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    // Thresholds for correctness
    static constexpr double MIN_COSINE_SIM = 0.999;
    static constexpr double MAX_ROW_SUM_ERROR = 0.05; // 5% tolerance for quantized softmax

    /**
     * @brief Quantize a single row of FP32 data to Q8_1 block format
     *
     * Q8_1 format: [d: fp16, sum_qs: int16, qs: int8[32]]
     * - d = scale = max_abs / 127
     * - sum_qs = sum of quantized values (for correction term)
     * - qs = quantized int8 values
     */
    void quantize_row_to_q8_1(const float *fp32_row, Q8_1Block *blocks, int k)
    {
        int k_blocks = (k + 31) / 32;

        for (int kb = 0; kb < k_blocks; ++kb)
        {
            int start = kb * 32;
            int end = std::min(start + 32, k);

            // Find max absolute value
            float max_abs = 0.0f;
            for (int i = start; i < end; ++i)
            {
                max_abs = std::max(max_abs, std::fabs(fp32_row[i]));
            }

            // Compute scale
            float scale = max_abs / 127.0f;
            if (scale == 0.0f)
                scale = 1.0f; // Avoid division by zero

            // Quantize
            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                float val = (start + i < end) ? fp32_row[start + i] : 0.0f;
                int8_t q = static_cast<int8_t>(std::round(val / scale));
                blocks[kb].qs[i] = q;
                sum_qs += q;
            }

            // Store scale as FP16 and sum
            blocks[kb].d = fp32_to_fp16(scale);
            blocks[kb].sum_qs = static_cast<int16_t>(sum_qs);
        }
    }

    /**
     * @brief Compute FP32 reference: softmax(Q * K^T * scale)
     */
    void compute_reference(
        int M, int N, int K,
        const float *Q, const float *K_data,
        float scale,
        float *C_ref)
    {
        // Compute Q * K^T (row-major)
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int kk = 0; kk < K; ++kk)
                {
                    sum += Q[i * K + kk] * K_data[j * K + kk];
                }
                C_ref[i * N + j] = sum * scale;
            }
        }

        // Apply softmax per row
        for (int i = 0; i < M; ++i)
        {
            float max_val = C_ref[i * N];
            for (int j = 1; j < N; ++j)
            {
                max_val = std::max(max_val, C_ref[i * N + j]);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                C_ref[i * N + j] = std::exp(C_ref[i * N + j] - max_val);
                sum_exp += C_ref[i * N + j];
            }

            for (int j = 0; j < N; ++j)
            {
                C_ref[i * N + j] /= sum_exp;
            }
        }
    }

    /**
     * @brief Verify kernel output against reference
     * @return pair of (max_abs_error, cosine_similarity)
     */
    std::pair<double, double> verify_correctness(
        int M, int N,
        const float *C_actual, const float *C_ref)
    {
        double max_abs_error = 0.0;
        double dot_prod = 0.0;
        double norm_act = 0.0;
        double norm_ref = 0.0;

        for (int i = 0; i < M * N; ++i)
        {
            double diff = std::fabs(C_actual[i] - C_ref[i]);
            max_abs_error = std::max(max_abs_error, diff);

            dot_prod += C_actual[i] * C_ref[i];
            norm_act += C_actual[i] * C_actual[i];
            norm_ref += C_ref[i] * C_ref[i];
        }

        double cosine_sim = (norm_act > 0.0 && norm_ref > 0.0)
                                ? dot_prod / (std::sqrt(norm_act) * std::sqrt(norm_ref))
                                : 0.0;

        return {max_abs_error, cosine_sim};
    }

    /**
     * @brief Run a test case with given dimensions
     */
    void run_test(int M, int N, int K, bool use_m4_kernel = true)
    {
        ASSERT_EQ(K % 32, 0) << "K must be multiple of 32 for Q8_1 blocks";

        float scale = 1.0f / std::sqrt(static_cast<float>(K));
        int k_blocks = K / 32;
        int Q_stride_bytes = k_blocks * sizeof(Q8_1Block);
        int K_stride_bytes = k_blocks * sizeof(Q8_1Block);

        // Generate random FP32 data
        std::vector<float> Q_fp32(M * K);
        std::vector<float> K_fp32(N * K);
        for (auto &x : Q_fp32)
            x = dist_(gen_);
        for (auto &x : K_fp32)
            x = dist_(gen_);

        // Quantize to Q8_1
        std::vector<Q8_1Block> Q_blocks(M * k_blocks);
        std::vector<Q8_1Block> K_blocks(N * k_blocks);

        for (int i = 0; i < M; ++i)
        {
            quantize_row_to_q8_1(Q_fp32.data() + i * K, Q_blocks.data() + i * k_blocks, K);
        }
        for (int j = 0; j < N; ++j)
        {
            quantize_row_to_q8_1(K_fp32.data() + j * K, K_blocks.data() + j * k_blocks, K);
        }

        // Output buffer
        std::vector<float> C(M * N, 0.0f);

        // Build params
        OnlineSoftmaxParams params;
        params.Q = Q_blocks.data();
        params.K = K_blocks.data();
        params.C = C.data();
        params.M = M;
        params.N = N;
        params.K_blocks = k_blocks;
        params.Q_stride_bytes = Q_stride_bytes;
        params.K_stride_bytes = K_stride_bytes;
        params.C_stride_bytes = N * sizeof(float);
        params.scale = scale;
        params.mask = nullptr;
        params.mask_stride_bytes = 0;

        // Run kernel
        static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m4(4);
        static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m1(1);

        if (use_m4_kernel && M >= 4)
        {
            int m_blocking = 4;
            for (int i = 0; i < M; i += m_blocking)
            {
                int current_m = std::min(m_blocking, M - i);

                OnlineSoftmaxParams p = params;
                p.Q = reinterpret_cast<const char *>(params.Q) + i * Q_stride_bytes;
                p.C = params.C + i * N;
                p.M = current_m;

                if (current_m == 4)
                {
                    kernel_m4.get_kernel()(&p);
                }
                else
                {
                    auto func = kernel_m1.get_kernel();
                    for (int j = 0; j < current_m; ++j)
                    {
                        OnlineSoftmaxParams p1 = p;
                        p1.Q = reinterpret_cast<const char *>(p.Q) + j * Q_stride_bytes;
                        p1.C = p.C + j * N;
                        p1.M = 1;
                        func(&p1);
                    }
                }
            }
        }
        else
        {
            // Use M1 kernel for all rows
            auto func = kernel_m1.get_kernel();
            for (int i = 0; i < M; ++i)
            {
                OnlineSoftmaxParams p = params;
                p.Q = reinterpret_cast<const char *>(params.Q) + i * Q_stride_bytes;
                p.C = params.C + i * N;
                p.M = 1;
                func(&p);
            }
        }

        // Compute FP32 reference
        std::vector<float> C_ref(M * N);
        compute_reference(M, N, K, Q_fp32.data(), K_fp32.data(), scale, C_ref.data());

        // Verify correctness
        auto [max_error, cosine_sim] = verify_correctness(M, N, C.data(), C_ref.data());

        EXPECT_GE(cosine_sim, MIN_COSINE_SIM)
            << "M=" << M << " N=" << N << " K=" << K
            << ": Cosine similarity " << cosine_sim << " below threshold " << MIN_COSINE_SIM;

        // Verify softmax properties
        for (int i = 0; i < M; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                EXPECT_GE(C[i * N + j], 0.0f)
                    << "M=" << M << " N=" << N << " K=" << K
                    << ": Negative softmax value at row " << i << ", col " << j
                    << " value=" << C[i * N + j];
                row_sum += C[i * N + j];
            }
            EXPECT_NEAR(row_sum, 1.0f, MAX_ROW_SUM_ERROR)
                << "M=" << M << " N=" << N << " K=" << K
                << ": Row " << i << " does not sum to 1.0 (got " << row_sum << ")";
        }
    }
};

// =============================================================================
// JIT Kernel Instantiation Tests
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, JIT_M4_KernelInstantiation)
{
    // Test that M=4 kernel can be instantiated and generates valid code
    QuantisedGemmJit_Q8_1_OnlineSoftmax kernel(4);
    auto func = kernel.get_kernel();
    ASSERT_NE(func, nullptr) << "M=4 JIT kernel should generate valid function pointer";
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, JIT_M1_KernelInstantiation)
{
    // Test that M=1 kernel can be instantiated and generates valid code
    QuantisedGemmJit_Q8_1_OnlineSoftmax kernel(1);
    auto func = kernel.get_kernel();
    ASSERT_NE(func, nullptr) << "M=1 JIT kernel should generate valid function pointer";
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, JIT_StaticKernels_Reusable)
{
    // Test that static kernels are reusable
    static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m4(4);
    static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m1(1);

    auto func4_1 = kernel_m4.get_kernel();
    auto func4_2 = kernel_m4.get_kernel();
    auto func1_1 = kernel_m1.get_kernel();
    auto func1_2 = kernel_m1.get_kernel();

    EXPECT_EQ(func4_1, func4_2) << "Static M4 kernel should return same function pointer";
    EXPECT_EQ(func1_1, func1_2) << "Static M1 kernel should return same function pointer";
    EXPECT_NE(func4_1, func1_1) << "M4 and M1 kernels should be different";
}

// =============================================================================
// OnlineSoftmaxParams Structure Tests
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, Params_SizeAndAlignment)
{
    OnlineSoftmaxParams params;
    // Verify struct size is reasonable (should be around 72 bytes)
    EXPECT_LE(sizeof(params), 128) << "OnlineSoftmaxParams too large";

    // Verify pointer offsets match JIT expectations
    // The JIT code accesses fields by offset from struct pointer
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, Q), 0);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, K), 8);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, C), 16);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, M), 24);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, N), 28);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, K_blocks), 32);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, Q_stride_bytes), 36);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, K_stride_bytes), 40);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, C_stride_bytes), 44);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, scale), 48);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, mask), 56);
    EXPECT_EQ(offsetof(OnlineSoftmaxParams, mask_stride_bytes), 64);
}

// =============================================================================
// Basic Correctness Tests
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, SingleRow_SmallN)
{
    // M=1, N=8, K=32 (1 block)
    run_test(1, 8, 32, false);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, SingleRow_MediumN)
{
    // M=1, N=64, K=64 (2 blocks)
    run_test(1, 64, 64, false);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, SingleRow_LargeK)
{
    // M=1, N=32, K=128 (4 blocks) - typical head_dim
    run_test(1, 32, 128, false);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, FourRows_SmallN)
{
    // M=4, N=16, K=64 - test M4 kernel
    run_test(4, 16, 64, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, FourRows_LargeN)
{
    // M=4, N=128, K=64 - more KV positions
    run_test(4, 128, 64, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, FourRows_LargeK)
{
    // M=4, N=32, K=128 - typical Qwen 7B head_dim
    run_test(4, 32, 128, true);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, EdgeCase_M3_TailHandling)
{
    // M=3 requires M4 kernel + M1 tail
    run_test(3, 32, 64, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, EdgeCase_M5_TailHandling)
{
    // M=5 = 4 + 1
    run_test(5, 32, 64, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, EdgeCase_M7_TailHandling)
{
    // M=7 = 4 + 3 (partial second block)
    run_test(7, 32, 64, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, EdgeCase_OddN)
{
    // N not multiple of unroll_n (tests scalar tail in N loop)
    run_test(4, 37, 64, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, EdgeCase_SingleKBlock)
{
    // Only 1 K block (K=32)
    run_test(4, 32, 32, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, EdgeCase_OddKBlocks)
{
    // 3 K blocks (tests K loop unroll tail)
    run_test(4, 32, 96, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, EdgeCase_MinimalDimensions)
{
    // Smallest valid practical case (N=1 is degenerate - softmax of single element is trivial)
    // Use N=2 which tests the minimal multi-element softmax
    run_test(1, 2, 32, false);
}

// =============================================================================
// Stress Tests (Still Fast - No Model Loading)
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, Stress_MediumBatch)
{
    // M=32, N=256, K=64 - medium prefill
    run_test(32, 256, 64, true);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, Stress_LargeBatch)
{
    // M=64, N=512, K=128 - larger prefill
    run_test(64, 512, 128, true);
}

// =============================================================================
// Determinism Tests
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, Determinism_MultipleRuns)
{
    // Run same test twice and verify identical results
    gen_.seed(12345);

    int M = 4, N = 64, K = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(K));
    int k_blocks = K / 32;

    // Generate data
    std::vector<float> Q_fp32(M * K);
    std::vector<float> K_fp32(N * K);
    for (auto &x : Q_fp32)
        x = dist_(gen_);
    for (auto &x : K_fp32)
        x = dist_(gen_);

    // Quantize
    std::vector<Q8_1Block> Q_blocks(M * k_blocks);
    std::vector<Q8_1Block> K_blocks(N * k_blocks);
    for (int i = 0; i < M; ++i)
        quantize_row_to_q8_1(Q_fp32.data() + i * K, Q_blocks.data() + i * k_blocks, K);
    for (int j = 0; j < N; ++j)
        quantize_row_to_q8_1(K_fp32.data() + j * K, K_blocks.data() + j * k_blocks, K);

    std::vector<float> C1(M * N), C2(M * N);

    // Run kernel twice
    for (int run = 0; run < 2; ++run)
    {
        float *C = (run == 0) ? C1.data() : C2.data();

        OnlineSoftmaxParams params;
        params.Q = Q_blocks.data();
        params.K = K_blocks.data();
        params.C = C;
        params.M = M;
        params.N = N;
        params.K_blocks = k_blocks;
        params.Q_stride_bytes = k_blocks * sizeof(Q8_1Block);
        params.K_stride_bytes = k_blocks * sizeof(Q8_1Block);
        params.C_stride_bytes = N * sizeof(float);
        params.scale = scale;
        params.mask = nullptr;
        params.mask_stride_bytes = 0;

        static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m4(4);
        kernel_m4.get_kernel()(&params);
    }

    // Compare
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_EQ(C1[i], C2[i])
            << "Non-deterministic result at index " << i
            << ": run1=" << C1[i] << " run2=" << C2[i];
    }
}

// =============================================================================
// M1 Kernel Specific Tests (Used for Decode)
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, M1_Decode_SmallKV)
{
    // Typical decode: M=1, small KV cache
    run_test(1, 16, 64, false);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, M1_Decode_MediumKV)
{
    // M=1, medium KV cache
    run_test(1, 128, 64, false);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, M1_Decode_LargeKV)
{
    // M=1, large KV cache (typical after long context)
    run_test(1, 512, 128, false);
}

TEST_F(Test__Q8_1_OnlineSoftmax_Unit, M1_Decode_VeryLargeKV)
{
    // M=1, very large KV cache
    run_test(1, 1024, 128, false);
}

// =============================================================================
// Main (No MPI Required for Unit Tests)
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
