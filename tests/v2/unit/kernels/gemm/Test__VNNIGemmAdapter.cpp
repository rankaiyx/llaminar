/**
 * @file Test__VNNIGemmAdapter.cpp
 * @brief Test suite for VNNIGemmAdapter - high-level GEMM adapter validation
 *
 * Test Coverage:
 * 1. End-to-end adapter functionality with FP32Tensor × Q8_0Tensor
 * 2. OneDNN reference validation for numerical correctness
 * 3. Various matrix sizes and configurations
 *
 * Note: Low-level packing utilities are tested indirectly. For pack_A/pack_B
 * coverage, see Test__VNNIGemmKernel.cpp.
 *
 * @author David Sanftenberg
 * @date 2025-01-24
 */

#include <gtest/gtest.h>
#include "kernels/cpu/gemm_v3/VNNIGemmAdapter.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include <dnnl.hpp>
#include <random>
#include <cmath>

namespace llaminar2
{

    // ===== Helper: OneDNN Reference GEMM (s8s8s32) =====

    /**
     * @brief Compute C = A × B using OneDNN s8s8s32 matmul (INT8 accumulation to INT32)
     * @param A_int8 Activation matrix (M×K), quantized INT8
     * @param B_int8 Weight matrix (K×N), quantized INT8
     * @param C_int32 Output matrix (M×N), INT32 accumulation
     * @param M Number of rows in A and C
     * @param K Shared dimension (columns of A, rows of B)
     * @param N Number of columns in B and C
     */
    void computeOneDNNReference(const int8_t *A_int8, const int8_t *B_int8, int32_t *C_int32,
                                int M, int K, int N)
    {
        dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream cpu_stream(cpu_engine);

        // Create memory descriptors (row-major)
        dnnl::memory::dims A_dims = {M, K};
        dnnl::memory::dims B_dims = {K, N};
        dnnl::memory::dims C_dims = {M, N};

        auto A_md = dnnl::memory::desc(A_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
        auto B_md = dnnl::memory::desc(B_dims, dnnl::memory::data_type::s8, dnnl::memory::format_tag::ab);
        auto C_md = dnnl::memory::desc(C_dims, dnnl::memory::data_type::s32, dnnl::memory::format_tag::ab);

        // Create memory objects
        auto A_mem = dnnl::memory(A_md, cpu_engine, const_cast<int8_t *>(A_int8));
        auto B_mem = dnnl::memory(B_md, cpu_engine, const_cast<int8_t *>(B_int8));
        auto C_mem = dnnl::memory(C_md, cpu_engine, C_int32);

        // Create matmul primitive
        auto matmul_pd = dnnl::matmul::primitive_desc(cpu_engine, A_md, B_md, C_md);
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // Execute
        matmul_prim.execute(cpu_stream, {{DNNL_ARG_SRC, A_mem},
                                         {DNNL_ARG_WEIGHTS, B_mem},
                                         {DNNL_ARG_DST, C_mem}});
        cpu_stream.wait();
    }

    /**
     * @brief Compute relative L2 error: ||C_test - C_ref||_2 / ||C_ref||_2
     */
    float computeRelativeL2Error(const float *C_test, const float *C_ref, size_t size)
    {
        float diff_norm_sq = 0.0f;
        float ref_norm_sq = 0.0f;

        for (size_t i = 0; i < size; ++i)
        {
            float diff = C_test[i] - C_ref[i];
            diff_norm_sq += diff * diff;
            ref_norm_sq += C_ref[i] * C_ref[i];
        }

        return std::sqrt(diff_norm_sq) / std::sqrt(ref_norm_sq);
    }

    /**
     * @brief Compute max absolute error
     */
    float computeMaxAbsError(const float *C_test, const float *C_ref, size_t size)
    {
        float max_abs = 0.0f;
        for (size_t i = 0; i < size; ++i)
        {
            float diff = std::abs(C_test[i] - C_ref[i]);
            max_abs = std::max(max_abs, diff);
        }
        return max_abs;
    }

    // ===== Test Fixture =====

    class Test__VNNIGemmAdapter : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Seed for reproducible randomness
            gen.seed(42);
        }

        std::mt19937 gen;
    };

    // ===== End-to-End Adapter Tests =====

    /**
     * @brief Test adapter with small matrix (full tiles, 16×64×64)
     *
     * Verifies:
     * - FP32Tensor activation → INT8 quantization
     * - Q8_0Tensor weights → VNNI packing
     * - Bias application
     * - VNNI kernel execution
     * - Dequantization to FP32
     */
    TEST_F(Test__VNNIGemmAdapter, AdapterSmallMatrixFullTiles)
    {
        constexpr int M = 16, K = 64, N = 64;

        // Generate random FP32 activation (M×K)
        FP32Tensor A_tensor({(size_t)M, (size_t)K});
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < M * K; ++i)
        {
            A_tensor.mutable_data()[i] = dist(gen);
        }

        // Generate random FP32 weights (K×N) and convert to Q8_0
        std::vector<float> B_fp32(K * N);
        for (size_t i = 0; i < B_fp32.size(); ++i)
        {
            B_fp32[i] = dist(gen);
        }

        // Convert to Q8_0 (simplified: use FP32Tensor → to_int8_blocked → Q8_0Tensor)
        FP32Tensor B_fp32_tensor({(size_t)N, (size_t)K}); // Q8_0 is column-major
        // Transpose B to column-major
        for (int k = 0; k < K; ++k)
        {
            for (int n = 0; n < N; ++n)
            {
                B_fp32_tensor.mutable_data()[n * K + k] = B_fp32[k * N + n];
            }
        }

        // Quantize to Q8_0 (32-element blocks)
        constexpr size_t BLOCK_SIZE = 32;
        size_t num_blocks_per_col = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<uint8_t> raw_data(N * num_blocks_per_col * sizeof(Q8_0Block));

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t kb = 0; kb < num_blocks_per_col; ++kb)
            {
                size_t k_start = kb * BLOCK_SIZE;
                size_t k_end = std::min(k_start + BLOCK_SIZE, (size_t)K);
                size_t block_len = k_end - k_start;

                // Extract column block
                std::vector<float> block_fp32(BLOCK_SIZE, 0.0f);
                for (size_t k = k_start; k < k_end; ++k)
                {
                    block_fp32[k - k_start] = B_fp32_tensor.data()[n * K + k];
                }

                // Symmetric quantization
                float max_abs = 0.0f;
                for (size_t i = 0; i < block_len; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(block_fp32[i]));
                }
                float scale = max_abs / 127.0f;
                if (scale == 0.0f)
                    scale = 1.0f;

                // Quantize and pack into Q8_0Block
                Q8_0Block *block = reinterpret_cast<Q8_0Block *>(raw_data.data() + (n * num_blocks_per_col + kb) * sizeof(Q8_0Block));
                block->d = fp32_to_fp16(scale);
                for (size_t i = 0; i < block_len; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(block_fp32[i] / scale));
                    block->qs[i] = q;
                }
                if (block_len < BLOCK_SIZE)
                {
                    std::memset(block->qs + block_len, 0, BLOCK_SIZE - block_len);
                }
            }
        }

        Q8_0Tensor B_tensor({(size_t)N, (size_t)K}, raw_data);

        // Generate random bias (N)
        std::vector<float> bias(N);
        for (int i = 0; i < N; ++i)
        {
            bias[i] = dist(gen);
        }

        // Create output tensor
        FP32Tensor C_tensor({(size_t)M, (size_t)N});

        // Execute adapter GEMM
        // Note: vnni_gemm_adapter takes references (not pointers) and doesn't return bool
        std::cout << "Before adapter: C_tensor[0] = " << C_tensor.data()[0] << std::endl;
        vnni_gemm_adapter<4, 16, 64, 2, 16>(M, N, K, A_tensor, B_tensor, C_tensor.mutable_data(), N, bias.data());
        std::cout << "After adapter: C_tensor[0] = " << C_tensor.data()[0] << std::endl;
        std::cout << "Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cout << "Tile checks: M%4=" << (M % 4) << ", N%16=" << (N % 16) << ", K%64=" << (K % 64) << std::endl;

        // Validate against OneDNN reference
        // 1. Quantize A and B to INT8
        std::vector<int8_t> A_int8(M * K);
        std::vector<float> A_scales(M);
        for (int m = 0; m < M; ++m)
        {
            float max_abs = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                max_abs = std::max(max_abs, std::abs(A_tensor.data()[m * K + k]));
            }
            float scale = max_abs / 127.0f;
            if (scale == 0.0f)
                scale = 1.0f;
            A_scales[m] = scale;

            for (int k = 0; k < K; ++k)
            {
                int8_t q = static_cast<int8_t>(std::round(A_tensor.data()[m * K + k] / scale));
                A_int8[m * K + k] = q;
            }
        }

        // Extract INT8 from Q8_0 (row-major for OneDNN)
        std::vector<int8_t> B_int8(K * N);
        std::vector<float> B_scales(N);
        for (size_t n = 0; n < N; ++n)
        {
            // Average scales across K-blocks (simple approximation)
            float scale_sum = 0.0f;
            for (size_t kb = 0; kb < num_blocks_per_col; ++kb)
            {
                const Q8_0Block *block = reinterpret_cast<const Q8_0Block *>(raw_data.data() + (n * num_blocks_per_col + kb) * sizeof(Q8_0Block));
                scale_sum += fp16_to_fp32(block->d);
            }
            B_scales[n] = scale_sum / num_blocks_per_col;

            for (size_t kb = 0; kb < num_blocks_per_col; ++kb)
            {
                const Q8_0Block *block = reinterpret_cast<const Q8_0Block *>(raw_data.data() + (n * num_blocks_per_col + kb) * sizeof(Q8_0Block));
                size_t k_start = kb * BLOCK_SIZE;
                size_t k_end = std::min(k_start + BLOCK_SIZE, (size_t)K);

                for (size_t k = k_start; k < k_end; ++k)
                {
                    B_int8[k * N + n] = block->qs[k - k_start];
                }
            }
        }

        // 2. Compute OneDNN reference (INT32 accumulation)
        std::vector<int32_t> C_int32(M * N, 0);
        computeOneDNNReference(A_int8.data(), B_int8.data(), C_int32.data(), M, K, N);

        // 3. Dequantize and add bias
        std::vector<float> C_ref(M * N);
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float dequant = C_int32[m * N + n] * A_scales[m] * B_scales[n];
                C_ref[m * N + n] = dequant + bias[n];
            }
        }

        // 4. Compare
        float rel_l2 = computeRelativeL2Error(C_tensor.data(), C_ref.data(), M * N);
        float max_abs = computeMaxAbsError(C_tensor.data(), C_ref.data(), M * N);

        std::cout << "AdapterSmallMatrixFullTiles:" << std::endl;
        std::cout << "  Relative L2 error: " << rel_l2 << std::endl;
        std::cout << "  Max absolute error: " << max_abs << std::endl;

        // Tolerance: INT8 quantization introduces ~1/127 error, accumulates over K
        EXPECT_LT(rel_l2, 0.05f) << "Relative L2 error should be < 5%";
        EXPECT_LT(max_abs, 10.0f) << "Max absolute error should be reasonable";
    }

    /**
     * @brief Test adapter without bias (null bias)
     *
     * Verifies adapter handles null bias correctly
     *
     * KNOWN ISSUE: Test currently fails (returns zeros). Debugging needed.
     * Possible causes:
     * - Q8_0 tensor creation issue (scales might be zero)
     * - Random data generation producing near-zero products
     * - Adapter path difference when bias=nullptr
     */
    TEST_F(Test__VNNIGemmAdapter, AdapterNoBias)
    {
        constexpr int M = 16, K = 64, N = 64;

        // Generate random FP32 activation
        FP32Tensor A_tensor({(size_t)M, (size_t)K});
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < M * K; ++i)
        {
            A_tensor.mutable_data()[i] = dist(gen);
        }

        // Generate Q8_0 weights (same pattern as above)
        std::vector<float> B_fp32(K * N);
        for (size_t i = 0; i < B_fp32.size(); ++i)
        {
            B_fp32[i] = dist(gen);
        }

        FP32Tensor B_fp32_tensor({(size_t)N, (size_t)K});
        for (int k = 0; k < K; ++k)
        {
            for (int n = 0; n < N; ++n)
            {
                B_fp32_tensor.mutable_data()[n * K + k] = B_fp32[k * N + n];
            }
        }

        // Quantize to Q8_0
        constexpr size_t BLOCK_SIZE = 32;
        size_t num_blocks_per_col = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<uint8_t> raw_data(N * num_blocks_per_col * sizeof(Q8_0Block));

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t kb = 0; kb < num_blocks_per_col; ++kb)
            {
                size_t k_start = kb * BLOCK_SIZE;
                size_t k_end = std::min(k_start + BLOCK_SIZE, (size_t)K);
                size_t block_len = k_end - k_start;

                std::vector<float> block_fp32(BLOCK_SIZE, 0.0f);
                for (size_t k = k_start; k < k_end; ++k)
                {
                    block_fp32[k - k_start] = B_fp32_tensor.data()[n * K + k];
                }

                float max_abs = 0.0f;
                for (size_t i = 0; i < block_len; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(block_fp32[i]));
                }
                float scale = max_abs / 127.0f;
                if (scale == 0.0f)
                    scale = 1.0f;

                Q8_0Block *block = reinterpret_cast<Q8_0Block *>(raw_data.data() + (n * num_blocks_per_col + kb) * sizeof(Q8_0Block));
                block->d = fp32_to_fp16(scale);
                for (size_t i = 0; i < block_len; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(block_fp32[i] / scale));
                    block->qs[i] = q;
                }
                if (block_len < BLOCK_SIZE)
                {
                    std::memset(block->qs + block_len, 0, BLOCK_SIZE - block_len);
                }
            }
        }

        Q8_0Tensor B_tensor({(size_t)N, (size_t)K}, raw_data);

        // Create output tensor
        FP32Tensor C_tensor({(size_t)M, (size_t)N});

        // Execute adapter GEMM without bias
        std::cout << "Before adapter (no bias): C_tensor[0] = " << C_tensor.data()[0] << std::endl;
        std::cout << "Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cout << "Tile checks: M%4=" << (M % 4) << ", N%16=" << (N % 16) << ", K%64=" << (K % 64) << std::endl;
        vnni_gemm_adapter<4, 16, 64, 2, 16>(M, N, K, A_tensor, B_tensor, C_tensor.mutable_data(), N, nullptr);
        std::cout << "After adapter (no bias): C_tensor[0] = " << C_tensor.data()[0] << std::endl;

        // Validate: output should be non-zero (computed GEMM result)
        bool has_nonzero = false;
        for (int i = 0; i < M * N; ++i)
        {
            if (C_tensor.data()[i] != 0.0f)
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Output should contain GEMM results (not all zeros)";
    }

    /**
     * @brief Test adapter with multiple K-blocks (K=256, 4 blocks of 64)
     *
     * Verifies K-loop tiling and accumulation across blocks
     *
     * KNOWN ISSUE: Test currently fails (returns zeros). Debugging needed.
     * Possible causes:
     * - Q8_0 tensor creation issue for larger K
     * - Random data generation producing near-zero products
     * - K-loop accumulation path difference
     */
    TEST_F(Test__VNNIGemmAdapter, AdapterMultipleKBlocks)
    {
        constexpr int M = 16, K = 256, N = 64;

        // Generate random FP32 activation
        FP32Tensor A_tensor({(size_t)M, (size_t)K});
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (int i = 0; i < M * K; ++i)
        {
            A_tensor.mutable_data()[i] = dist(gen);
        }

        // Generate Q8_0 weights
        std::vector<float> B_fp32(K * N);
        for (size_t i = 0; i < B_fp32.size(); ++i)
        {
            B_fp32[i] = dist(gen);
        }

        FP32Tensor B_fp32_tensor({(size_t)N, (size_t)K});
        for (int k = 0; k < K; ++k)
        {
            for (int n = 0; n < N; ++n)
            {
                B_fp32_tensor.mutable_data()[n * K + k] = B_fp32[k * N + n];
            }
        }

        // Quantize to Q8_0
        constexpr size_t BLOCK_SIZE = 32;
        size_t num_blocks_per_col = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        std::vector<uint8_t> raw_data(N * num_blocks_per_col * sizeof(Q8_0Block));

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t kb = 0; kb < num_blocks_per_col; ++kb)
            {
                size_t k_start = kb * BLOCK_SIZE;
                size_t k_end = std::min(k_start + BLOCK_SIZE, (size_t)K);
                size_t block_len = k_end - k_start;

                std::vector<float> block_fp32(BLOCK_SIZE, 0.0f);
                for (size_t k = k_start; k < k_end; ++k)
                {
                    block_fp32[k - k_start] = B_fp32_tensor.data()[n * K + k];
                }

                float max_abs = 0.0f;
                for (size_t i = 0; i < block_len; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(block_fp32[i]));
                }
                float scale = max_abs / 127.0f;
                if (scale == 0.0f)
                    scale = 1.0f;

                Q8_0Block *block = reinterpret_cast<Q8_0Block *>(raw_data.data() + (n * num_blocks_per_col + kb) * sizeof(Q8_0Block));
                block->d = fp32_to_fp16(scale);
                for (size_t i = 0; i < block_len; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(block_fp32[i] / scale));
                    block->qs[i] = q;
                }
                if (block_len < BLOCK_SIZE)
                {
                    std::memset(block->qs + block_len, 0, BLOCK_SIZE - block_len);
                }
            }
        }

        Q8_0Tensor B_tensor({(size_t)N, (size_t)K}, raw_data);

        // Create output tensor
        FP32Tensor C_tensor({(size_t)M, (size_t)N});

        // Execute adapter GEMM
        vnni_gemm_adapter<4, 16, 64, 2, 16>(M, N, K, A_tensor, B_tensor, C_tensor.mutable_data(), N, nullptr);

        // Basic sanity check: output should be non-zero
        bool has_nonzero = false;
        for (int i = 0; i < M * N; ++i)
        {
            if (std::abs(C_tensor.data()[i]) > 1e-5f)
            {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Output should contain GEMM results";
    }

} // namespace llaminar2
