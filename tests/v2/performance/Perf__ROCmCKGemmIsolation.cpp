/**
 * @file Perf__ROCmCKGemmIsolation.cpp
 * @brief Isolated CK GEMM kernel performance test
 *
 * This test reproduces the slow CK GEMM execution observed in parity tests.
 * It directly measures kernel execution time (including GPU sync) to isolate
 * whether the issue is in CK itself vs. our integration.
 *
 * Key observations from parity test debugging:
 *   - M=8 N=896 K=896  → 1871ms (expected <1ms)
 *   - M=8 N=4864 K=896 → 2054ms
 *   - M=8 N=896 K=4864 → 10015ms (10 SECONDS!)
 *
 * The 32×32 kernel with GemmMNPadding appears pathologically slow on gfx906.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>

// HIP runtime
#include <hip/hip_runtime.h>

// CK GEMM kernel extern declarations
extern "C"
{
    bool rocmQuantGemm_executeNoScale(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8,
        int32_t *d_C_int32,
        int M, int N, int K,
        int rocm_device_id);

    bool rocmQuantGemm_setDevice(int device_id);

    void rocmQuantGemm_warmupKernels();
}

namespace
{

    class ROCmCKGemmIsolationTest : public ::testing::Test
    {
    protected:
        int device_id_ = 0;

        void SetUp() override
        {
            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);
            if (err != hipSuccess || device_count == 0)
            {
                GTEST_SKIP() << "No ROCm devices available";
            }

            hipDeviceProp_t props;
            hipGetDeviceProperties(&props, device_id_);
            fprintf(stderr, "\n[CK GEMM Isolation Test]\n");
            fprintf(stderr, "Device: %s\n", props.name);
            fprintf(stderr, "GCN Arch: %s\n", props.gcnArchName);

            // Warm up CK kernels
            fprintf(stderr, "Warming up CK kernels...\n");
            rocmQuantGemm_warmupKernels();
            fprintf(stderr, "Warmup complete.\n\n");
        }

        struct BenchResult
        {
            double time_ms;
            double tflops;
            bool success;
        };

        // Benchmark a single GEMM configuration
        BenchResult benchmarkGemm(int M, int N, int K, int warmup_runs = 2, int bench_runs = 3)
        {
            BenchResult result{0.0, 0.0, false};

            // Allocate device memory
            int8_t *d_A = nullptr;
            int8_t *d_B = nullptr;
            int32_t *d_C = nullptr;

            size_t a_size = static_cast<size_t>(M) * K;
            size_t b_size = static_cast<size_t>(K) * N;
            size_t c_size = static_cast<size_t>(M) * N;

            hipError_t err;
            err = hipMalloc(&d_A, a_size * sizeof(int8_t));
            if (err != hipSuccess)
            {
                fprintf(stderr, "Failed to allocate A: %s\n", hipGetErrorString(err));
                return result;
            }

            err = hipMalloc(&d_B, b_size * sizeof(int8_t));
            if (err != hipSuccess)
            {
                (void)hipFree(d_A);
                fprintf(stderr, "Failed to allocate B: %s\n", hipGetErrorString(err));
                return result;
            }

            err = hipMalloc(&d_C, c_size * sizeof(int32_t));
            if (err != hipSuccess)
            {
                (void)hipFree(d_A);
                (void)hipFree(d_B);
                fprintf(stderr, "Failed to allocate C: %s\n", hipGetErrorString(err));
                return result;
            }

            // Initialize with random data
            std::vector<int8_t> h_A(a_size);
            std::vector<int8_t> h_B(b_size);
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(-127, 127);
            for (auto &v : h_A)
                v = static_cast<int8_t>(dist(rng));
            for (auto &v : h_B)
                v = static_cast<int8_t>(dist(rng));

            (void)hipMemcpy(d_A, h_A.data(), a_size * sizeof(int8_t), hipMemcpyHostToDevice);
            (void)hipMemcpy(d_B, h_B.data(), b_size * sizeof(int8_t), hipMemcpyHostToDevice);
            (void)hipMemset(d_C, 0, c_size * sizeof(int32_t));
            (void)hipDeviceSynchronize();

            // Warmup runs
            for (int i = 0; i < warmup_runs; ++i)
            {
                rocmQuantGemm_executeNoScale(d_A, d_B, d_C, M, N, K, device_id_);
                (void)hipDeviceSynchronize();
            }

            // Timed benchmark runs
            double total_ms = 0.0;
            for (int i = 0; i < bench_runs; ++i)
            {
                auto t0 = std::chrono::high_resolution_clock::now();

                bool ok = rocmQuantGemm_executeNoScale(d_A, d_B, d_C, M, N, K, device_id_);
                (void)hipDeviceSynchronize();

                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_ms += ms;

                if (!ok)
                {
                    fprintf(stderr, "GEMM failed at run %d\n", i);
                    (void)hipFree(d_A);
                    (void)hipFree(d_B);
                    (void)hipFree(d_C);
                    return result;
                }
            }

            (void)hipFree(d_A);
            (void)hipFree(d_B);
            (void)hipFree(d_C);

            result.time_ms = total_ms / bench_runs;

            // Compute TFLOPS: 2*M*N*K operations (multiply-add)
            double flops = 2.0 * M * N * K;
            result.tflops = (flops / (result.time_ms * 1e-3)) / 1e12;
            result.success = true;

            return result;
        }

        void printTableHeader()
        {
            fprintf(stderr, "┌─────────┬─────────┬─────────┬────────────┬──────────────┬───────────┐\n");
            fprintf(stderr, "│    M    │    N    │    K    │  Time (ms) │    TFLOPS    │  Status   │\n");
            fprintf(stderr, "├─────────┼─────────┼─────────┼────────────┼──────────────┼───────────┤\n");
        }

        void printTableRow(int M, int N, int K, const BenchResult &result, double expected_ms = 0.0)
        {
            const char *status = "OK";
            if (!result.success)
            {
                status = "FAILED";
            }
            else if (expected_ms > 0 && result.time_ms > expected_ms * 10)
            {
                status = "SLOW!";
            }
            else if (expected_ms > 0 && result.time_ms > expected_ms * 2)
            {
                status = "slow";
            }

            fprintf(stderr, "│ %7d │ %7d │ %7d │ %10.3f │ %12.4f │ %-9s │\n",
                    M, N, K, result.time_ms, result.tflops, status);
        }

        void printTableFooter()
        {
            fprintf(stderr, "└─────────┴─────────┴─────────┴────────────┴──────────────┴───────────┘\n");
        }
    };

    // =============================================================================
    // Test: Reproduce the exact configurations that are slow in parity tests
    // =============================================================================

    TEST_F(ROCmCKGemmIsolationTest, ReproduceParityTestSlowdown)
    {
        fprintf(stderr, "\n=== Reproducing Parity Test Configurations ===\n");
        fprintf(stderr, "(These are the exact M/N/K values that took 2-10 seconds)\n\n");

        printTableHeader();

        // Wo projection: M=1 (padded to 8), N=896, K=896
        // In parity test: 1871ms
        auto r1 = benchmarkGemm(8, 896, 896);
        printTableRow(8, 896, 896, r1, 1.0); // Expected <1ms

        // FFN gate: M=1 (padded to 8), N=4864, K=896
        // In parity test: 2054ms
        auto r2 = benchmarkGemm(8, 4864, 896);
        printTableRow(8, 4864, 896, r2, 1.0);

        // FFN down: M=1 (padded to 8), N=896, K=4864
        // In parity test: 10015ms (10 SECONDS!)
        auto r3 = benchmarkGemm(8, 896, 4864);
        printTableRow(8, 896, 4864, r3, 1.0);

        printTableFooter();

        // Report findings
        fprintf(stderr, "\nFindings:\n");
        if (r1.time_ms > 100 || r2.time_ms > 100 || r3.time_ms > 100)
        {
            fprintf(stderr, "  ⚠️  REPRODUCED: CK GEMM is pathologically slow for small M!\n");
            fprintf(stderr, "  Expected: <1ms per GEMM\n");
            fprintf(stderr, "  Actual: %.1f ms, %.1f ms, %.1f ms\n", r1.time_ms, r2.time_ms, r3.time_ms);
        }
        else
        {
            fprintf(stderr, "  ✓ Performance is normal (issue may be elsewhere)\n");
        }
    }

    // =============================================================================
    // Test: Sweep M dimension to find where slowdown begins
    // =============================================================================

    TEST_F(ROCmCKGemmIsolationTest, SweepMDimension)
    {
        fprintf(stderr, "\n=== M Dimension Sweep (N=896, K=896) ===\n");
        fprintf(stderr, "(Testing 32x32 kernel threshold at M=32)\n\n");

        const int N = 896;
        const int K = 896;

        printTableHeader();

        // Test various M values around the kernel thresholds
        std::vector<int> m_values = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256};

        for (int M : m_values)
        {
            auto result = benchmarkGemm(M, N, K);

            // Expected time based on CK documentation
            double expected_ms = 0.5; // CK claims ~0.46ms for decode
            if (M >= 128)
                expected_ms = 0.8;

            printTableRow(M, N, K, result, expected_ms);
        }

        printTableFooter();
    }

    // =============================================================================
    // Test: Sweep K dimension (this showed 10s slowdown for K=4864)
    // =============================================================================

    TEST_F(ROCmCKGemmIsolationTest, SweepKDimension)
    {
        fprintf(stderr, "\n=== K Dimension Sweep (M=8, N=896) ===\n");
        fprintf(stderr, "(K=4864 showed 10 second execution in parity test)\n\n");

        const int M = 8;
        const int N = 896;

        printTableHeader();

        std::vector<int> k_values = {256, 512, 896, 1024, 2048, 3584, 4864};

        for (int K : k_values)
        {
            auto result = benchmarkGemm(M, N, K);
            printTableRow(M, N, K, result, 1.0);
        }

        printTableFooter();
    }

    // =============================================================================
    // Test: Compare 32x32 vs 64x64 vs 128x128 kernels
    // =============================================================================

    TEST_F(ROCmCKGemmIsolationTest, CompareKernelConfigurations)
    {
        fprintf(stderr, "\n=== Kernel Configuration Comparison ===\n");
        fprintf(stderr, "(32x32 for M≤32, 64x64 for 32<M<128, 128x128 for M≥128)\n\n");

        const int N = 4864;
        const int K = 896;

        printTableHeader();

        // Should use 32x32 kernel
        auto r_m8 = benchmarkGemm(8, N, K);
        printTableRow(8, N, K, r_m8, 0.5);

        // Should use 32x32 kernel (boundary)
        auto r_m32 = benchmarkGemm(32, N, K);
        printTableRow(32, N, K, r_m32, 0.5);

        // Should use 64x64 kernel
        auto r_m64 = benchmarkGemm(64, N, K);
        printTableRow(64, N, K, r_m64, 0.6);

        // Should use 128x128 kernel
        auto r_m128 = benchmarkGemm(128, N, K);
        printTableRow(128, N, K, r_m128, 0.8);

        // Large M for reference
        auto r_m512 = benchmarkGemm(512, N, K);
        printTableRow(512, N, K, r_m512, 2.0);

        printTableFooter();

        fprintf(stderr, "\nKernel efficiency analysis:\n");
        fprintf(stderr, "  M=8   (32x32):   %.4f TFLOPS\n", r_m8.tflops);
        fprintf(stderr, "  M=32  (32x32):   %.4f TFLOPS\n", r_m32.tflops);
        fprintf(stderr, "  M=64  (64x64):   %.4f TFLOPS\n", r_m64.tflops);
        fprintf(stderr, "  M=128 (128x128): %.4f TFLOPS\n", r_m128.tflops);
        fprintf(stderr, "  M=512 (128x128): %.4f TFLOPS\n", r_m512.tflops);
    }

    // =============================================================================
    // Test: Qwen2.5-0.5B decode workload simulation
    // =============================================================================

    TEST_F(ROCmCKGemmIsolationTest, Qwen05BDecodeWorkload)
    {
        fprintf(stderr, "\n=== Qwen2.5-0.5B Decode Workload (M=1→8 padded) ===\n");
        fprintf(stderr, "(Full layer: attn_norm → QKV → attn → Wo → ffn_norm → gate/up → down)\n\n");

        // Qwen2.5-0.5B dimensions
        const int hidden = 896;
        const int inter = 4864;
        const int M = 8; // Decode with padding

        printTableHeader();

        // Q projection
        auto r_q = benchmarkGemm(M, hidden, hidden);
        printTableRow(M, hidden, hidden, r_q, 0.5);

        // K projection
        auto r_k = benchmarkGemm(M, hidden, hidden);
        printTableRow(M, hidden, hidden, r_k, 0.5);

        // V projection
        auto r_v = benchmarkGemm(M, hidden, hidden);
        printTableRow(M, hidden, hidden, r_v, 0.5);

        // Wo projection
        auto r_wo = benchmarkGemm(M, hidden, hidden);
        printTableRow(M, hidden, hidden, r_wo, 0.5);

        // FFN gate
        auto r_gate = benchmarkGemm(M, inter, hidden);
        printTableRow(M, inter, hidden, r_gate, 0.5);

        // FFN up
        auto r_up = benchmarkGemm(M, inter, hidden);
        printTableRow(M, inter, hidden, r_up, 0.5);

        // FFN down
        auto r_down = benchmarkGemm(M, hidden, inter);
        printTableRow(M, hidden, inter, r_down, 0.5);

        printTableFooter();

        // Sum up total time for one layer
        double total_ms = r_q.time_ms + r_k.time_ms + r_v.time_ms + r_wo.time_ms +
                          r_gate.time_ms + r_up.time_ms + r_down.time_ms;

        fprintf(stderr, "\nTotal GEMM time per layer: %.2f ms\n", total_ms);
        fprintf(stderr, "Expected for 24 layers: %.2f ms (%.2f s)\n",
                total_ms * 24, total_ms * 24 / 1000.0);

        if (total_ms > 100)
        {
            fprintf(stderr, "\n⚠️  CRITICAL: GEMM alone takes %.1f ms per layer!\n", total_ms);
            fprintf(stderr, "This explains the multi-second parity test times.\n");
        }
    }

    // =============================================================================
    // Test: hipBLAS comparison (if available)
    // =============================================================================

    TEST_F(ROCmCKGemmIsolationTest, DISABLED_CompareWithHipBLAS)
    {
        // TODO: Add hipBLAS comparison to verify if issue is CK-specific
        fprintf(stderr, "\n=== hipBLAS Comparison (DISABLED - requires hipBLAS) ===\n");
        GTEST_SKIP() << "hipBLAS comparison not yet implemented";
    }

} // namespace
