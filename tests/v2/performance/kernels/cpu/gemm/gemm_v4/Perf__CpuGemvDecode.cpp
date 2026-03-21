/**
 * @file Perf__CpuGemvDecode.cpp
 * @brief Performance benchmark for CPU GEMV (M=1 decode) and GEMM (prefill) paths
 *
 * Benchmarks QuantisedGemmKernel with realistic Qwen model layer shapes:
 *   - QKV projection (fused 3-way)
 *   - Attention output (Wo) projection
 *   - FFN gate+up (fused 2-way)
 *   - FFN down projection
 *
 * Covers 4 model sizes: Qwen 0.5B, 1.5B, 3B, 7B
 * Tests both M=1 (decode/GEMV) and M=2,4,8,16 (small prefill)
 *
 * Uses high iteration counts (1000+) for stable M=1 microsecond-level measurements.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <string>

#include "fort.hpp"
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

// ============================================================================
// Model configurations (matching real Qwen architectures)
// ============================================================================

struct ModelConfig
{
    std::string name;
    int d_model;    // hidden size
    int n_heads;    // attention heads
    int n_kv_heads; // KV heads (GQA)
    int head_dim;   // head dimension
    int d_ff;       // FFN intermediate size
    int n_layers;   // transformer layers
};

static const ModelConfig kQwen05B = {"Qwen2.5-0.5B", 896, 14, 2, 64, 4864, 24};
static const ModelConfig kQwen15B = {"Qwen2.5-1.5B", 1536, 12, 2, 128, 8960, 28};
static const ModelConfig kQwen3B = {"Qwen2.5-3B", 2048, 16, 2, 128, 11008, 36};
static const ModelConfig kQwen7B = {"Qwen2.5-7B", 3584, 28, 4, 128, 18944, 28};

// Layer projection shapes for a given model
struct LayerShapes
{
    std::string model_name;
    int k_qkv;         // input dim for QKV (= d_model)
    int n_q;           // Q output dim (= n_heads * head_dim)
    int n_kv;          // K or V output dim (= n_kv_heads * head_dim)
    int n_qkv;         // total QKV output (= n_q + 2*n_kv)
    int k_wo;          // Wo input dim (= n_heads * head_dim)
    int n_wo;          // Wo output dim (= d_model)
    int k_ffn_gate_up; // FFN gate/up input (= d_model)
    int n_ffn_gate;    // FFN gate output (= d_ff)
    int n_ffn_up;      // FFN up output (= d_ff)
    int k_ffn_down;    // FFN down input (= d_ff)
    int n_ffn_down;    // FFN down output (= d_model)
};

static LayerShapes shapesFor(const ModelConfig &m)
{
    LayerShapes s;
    s.model_name = m.name;
    s.k_qkv = m.d_model;
    s.n_q = m.n_heads * m.head_dim;
    s.n_kv = m.n_kv_heads * m.head_dim;
    s.n_qkv = s.n_q + 2 * s.n_kv;
    s.k_wo = m.n_heads * m.head_dim;
    s.n_wo = m.d_model;
    s.k_ffn_gate_up = m.d_model;
    s.n_ffn_gate = m.d_ff;
    s.n_ffn_up = m.d_ff;
    s.k_ffn_down = m.d_ff;
    s.n_ffn_down = m.d_model;
    return s;
}

// ============================================================================
// Benchmark infrastructure
// ============================================================================

struct BenchResult
{
    std::string label;
    int M, N, K;
    double min_us;
    double mean_us;
    double stddev_us;
    double gflops;       // based on min_us
    double bandwidth_gb; // effective memory bandwidth (GB/s) based on min_us
    double l2_error;
};

class CpuGemvDecode : public ::testing::Test
{
protected:
    int rank_ = 0;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        if (rank_ == 0)
        {
            std::cout << "[GEMV Decode Perf] OpenMP threads: " << omp_get_max_threads() << std::endl;
        }
    }

    // Create Q8_1 weights quantized from random FP32
    std::shared_ptr<Q8_1Tensor> create_weights(int N, int K, std::mt19937 &gen)
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> fp32(static_cast<size_t>(N) * K);
        for (auto &x : fp32)
            x = dist(gen);
        return Q8_1Tensor::quantize_from_fp32(fp32.data(),
                                              {static_cast<size_t>(N), static_cast<size_t>(K)});
    }

    // Dequantize weights for reference calculation
    std::vector<float> dequantize(const Q8_1Tensor *t, int N, int K)
    {
        std::vector<float> out(static_cast<size_t>(N) * K);
        t->to_fp32(out.data());
        return out;
    }

    // Reference GEMM: C[M,N] = A[M,K] @ B^T[K,N]  (B stored as [N,K])
    void reference_gemv(const float *A, const float *B_deq, float *C, int M, int N, int K)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                double acc = 0.0;
                for (int p = 0; p < K; ++p)
                {
                    acc += static_cast<double>(A[i * K + p]) * static_cast<double>(B_deq[j * K + p]);
                }
                C[i * N + j] = static_cast<float>(acc);
            }
        }
    }

    double compute_l2_error(const float *actual, const float *ref, size_t n)
    {
        double sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double d = actual[i] - ref[i];
            sum_sq_diff += d * d;
            sum_sq_ref += static_cast<double>(ref[i]) * ref[i];
        }
        return (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;
    }

    BenchResult bench_single(const std::string &label, int M, int N, int K,
                             int warmup = 200, int iters = 1000)
    {
        std::mt19937 gen(42);
        auto weights = create_weights(N, K, gen);
        auto deq = dequantize(weights.get(), N, K);

        QuantisedGemmKernel kernel(weights.get());

        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> A(static_cast<size_t>(M) * K);
        for (auto &x : A)
            x = dist(gen);

        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);

        // Verify accuracy (once)
        kernel.multiply(A.data(), C.data(), M, N, K, true, 1.0f, 0.0f, nullptr, -1);
        std::vector<float> C_ref(static_cast<size_t>(M) * N);
        reference_gemv(A.data(), deq.data(), C_ref.data(), M, N, K);
        double l2 = compute_l2_error(C.data(), C_ref.data(), static_cast<size_t>(M) * N);

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            kernel.multiply(A.data(), C.data(), M, N, K, true, 1.0f, 0.0f, nullptr, -1);
        }

        // Benchmark - collect per-iteration times
        std::vector<double> times_us;
        times_us.reserve(iters);

        for (int i = 0; i < iters; ++i)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            kernel.multiply(A.data(), C.data(), M, N, K, true, 1.0f, 0.0f, nullptr, -1);
            auto t1 = std::chrono::high_resolution_clock::now();
            times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }

        std::sort(times_us.begin(), times_us.end());

        // Use p10 as "min" to filter out measurement noise
        int p10_idx = std::max(0, static_cast<int>(iters * 0.10));
        double min_us = times_us[p10_idx];

        double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
        double mean_us = sum / iters;
        double sq_sum = 0.0;
        for (auto t : times_us)
            sq_sum += (t - mean_us) * (t - mean_us);
        double stddev_us = std::sqrt(sq_sum / iters);

        // GFLOPS = 2*M*N*K / (time_s * 1e9)
        double ops = 2.0 * M * N * K;
        double gflops = ops / (min_us * 1e-6) / 1e9;

        // Effective bandwidth: weights = N*K bytes (int8 packed), A = M*K*4 bytes, C = M*N*4 bytes
        double bytes = static_cast<double>(N) * K + static_cast<double>(M) * K * 4.0 +
                       static_cast<double>(M) * N * 4.0;
        double bw_gb = bytes / (min_us * 1e-6) / 1e9;

        return {label, M, N, K, min_us, mean_us, stddev_us, gflops, bw_gb, l2};
    }

    void print_table(const std::string &title, const std::vector<BenchResult> &results)
    {
        if (rank_ != 0)
            return;

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        // Title row spanning all columns
        table << fort::header
              << "Projection" << "M" << "N" << "K"
              << "p10 (us)" << "Mean (us)" << "Stddev"
              << "GFLOPS" << "BW (GB/s)" << "L2 Err"
              << fort::endr;

        table.column(0).set_cell_text_align(fort::text_align::left);
        for (int c = 1; c <= 9; ++c)
            table.column(c).set_cell_text_align(fort::text_align::right);

        for (const auto &r : results)
        {
            std::ostringstream p10, mean, sd, gf, bw, l2;
            p10 << std::fixed << std::setprecision(1) << r.min_us;
            mean << std::fixed << std::setprecision(1) << r.mean_us;
            sd << std::fixed << std::setprecision(1) << r.stddev_us;
            gf << std::fixed << std::setprecision(1) << r.gflops;
            bw << std::fixed << std::setprecision(1) << r.bandwidth_gb;
            l2 << std::scientific << std::setprecision(2) << r.l2_error;

            table << r.label << r.M << r.N << r.K
                  << p10.str() << mean.str() << sd.str()
                  << gf.str() << bw.str() << l2.str()
                  << fort::endr;
        }

        std::cout << "\n=== " << title << " ===\n"
                  << table.to_string() << std::endl;
    }
};

// Custom main for MPI
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

// ============================================================================
// TEST 1: GEMV Decode (M=1) - All projections, all model sizes
// ============================================================================

TEST_F(CpuGemvDecode, Decode_AllModels)
{
    const ModelConfig *models[] = {&kQwen05B, &kQwen15B, &kQwen3B, &kQwen7B};

    for (const auto *model : models)
    {
        auto s = shapesFor(*model);
        std::vector<BenchResult> results;

        // QKV projection (as separate GEMM for now - the fused test is separate)
        results.push_back(bench_single("Q proj", 1, s.n_q, s.k_qkv, 200, 1000));
        results.push_back(bench_single("K proj", 1, s.n_kv, s.k_qkv, 200, 1000));
        results.push_back(bench_single("V proj", 1, s.n_kv, s.k_qkv, 200, 1000));
        results.push_back(bench_single("Wo proj", 1, s.n_wo, s.k_wo, 200, 1000));
        results.push_back(bench_single("FFN gate", 1, s.n_ffn_gate, s.k_ffn_gate_up, 200, 1000));
        results.push_back(bench_single("FFN up", 1, s.n_ffn_up, s.k_ffn_gate_up, 200, 1000));
        results.push_back(bench_single("FFN down", 1, s.n_ffn_down, s.k_ffn_down, 200, 1000));

        // Sum up total decode time for one full layer
        double total_us = 0.0;
        for (const auto &r : results)
            total_us += r.min_us;

        print_table(s.model_name + " Decode (M=1)", results);

        if (rank_ == 0)
        {
            std::cout << "  Total layer decode time: " << std::fixed << std::setprecision(1)
                      << total_us << " us ("
                      << std::setprecision(1) << (1e6 / total_us) << " layers/s)\n";
        }

        // Accuracy check
        for (const auto &r : results)
        {
            EXPECT_LT(r.l2_error, 0.01) << r.label << " L2 error too high";
        }
    }
}

// ============================================================================
// TEST 2: Small-batch prefill (M=2,4,8,16) for the most critical projections
// ============================================================================

TEST_F(CpuGemvDecode, SmallBatch_Prefill)
{
    const ModelConfig *models[] = {&kQwen05B, &kQwen7B};
    const int batch_sizes[] = {2, 4, 8, 16};

    for (const auto *model : models)
    {
        auto s = shapesFor(*model);
        std::vector<BenchResult> results;

        for (int M : batch_sizes)
        {
            results.push_back(bench_single("FFN down M=" + std::to_string(M),
                                           M, s.n_ffn_down, s.k_ffn_down, 100, 500));
            results.push_back(bench_single("Wo proj M=" + std::to_string(M),
                                           M, s.n_wo, s.k_wo, 100, 500));
        }

        print_table(s.model_name + " Small-Batch Prefill", results);

        for (const auto &r : results)
        {
            EXPECT_LT(r.l2_error, 0.01) << r.label;
        }
    }
}

// ============================================================================
// TEST 3: Large-batch prefill (M=32,128,512) for throughput measurement
// ============================================================================

TEST_F(CpuGemvDecode, LargeBatch_Prefill)
{
    const ModelConfig *models[] = {&kQwen05B, &kQwen7B};
    const int batch_sizes[] = {32, 128, 512};

    for (const auto *model : models)
    {
        auto s = shapesFor(*model);
        std::vector<BenchResult> results;

        for (int M : batch_sizes)
        {
            // FFN down (largest GEMM in the pipeline)
            results.push_back(bench_single("FFN down M=" + std::to_string(M),
                                           M, s.n_ffn_down, s.k_ffn_down, 10, 50));
        }

        print_table(s.model_name + " Large-Batch Prefill", results);

        for (const auto &r : results)
        {
            EXPECT_LT(r.l2_error, 0.01) << r.label;
        }
    }
}

// ============================================================================
// TEST 4: Bandwidth efficiency analysis (decode is bandwidth-bound)
// ============================================================================

TEST_F(CpuGemvDecode, BandwidthEfficiency)
{
    // For M=1 GEMV, the theoretical limit is memory bandwidth.
    // We read: N*K bytes of packed weights + K*36/32 bytes of A (Q8_1) + N*4 bytes for C write
    // On this system with ~90 GB/s per socket DDR4, we expect:
    //   time_us = bytes / (bandwidth * 1e-6)

    auto s = shapesFor(kQwen7B);
    std::vector<BenchResult> results;

    // Test different K sizes to see bandwidth scaling
    struct TestCase
    {
        std::string name;
        int N;
        int K;
    };
    std::vector<TestCase> cases = {
        {"Small (Wo 0.5B)", 896, 896},
        {"Medium (Wo 7B)", 3584, 3584},
        {"Large (FFN 7B)", 3584, 18944},
        {"XLarge (FFN 32B)", 5120, 27392},
    };

    for (const auto &tc : cases)
    {
        results.push_back(bench_single(tc.name, 1, tc.N, tc.K, 200, 2000));
    }

    print_table("Bandwidth Efficiency (M=1 GEMV)", results);

    if (rank_ == 0)
    {
        std::cout << "\n  Theoretical peak DDR4 bandwidth (single socket): ~90 GB/s\n";
        for (const auto &r : results)
        {
            double eff = r.bandwidth_gb / 90.0 * 100.0;
            std::cout << "  " << std::setw(25) << std::left << r.label
                      << " BW efficiency: " << std::fixed << std::setprecision(1) << eff << "%\n";
        }
    }
}

// ============================================================================
// TEST 5: Accuracy verification across all model configs
// ============================================================================

TEST_F(CpuGemvDecode, Accuracy_AllConfigs)
{
    const ModelConfig *models[] = {&kQwen05B, &kQwen15B, &kQwen3B, &kQwen7B};

    fort::utf8_table table;
    table.set_border_style(FT_DOUBLE2_STYLE);
    table << fort::header << "Model" << "Projection" << "M" << "N" << "K" << "L2 Error" << "Status" << fort::endr;

    bool all_pass = true;
    for (const auto *model : models)
    {
        auto s = shapesFor(*model);

        struct TestCase
        {
            std::string name;
            int M;
            int N;
            int K;
        };
        std::vector<TestCase> cases = {
            {"Q proj", 1, s.n_q, s.k_qkv},
            {"K proj", 1, s.n_kv, s.k_qkv},
            {"Wo proj", 1, s.n_wo, s.k_wo},
            {"FFN gate", 1, s.n_ffn_gate, s.k_ffn_gate_up},
            {"FFN down", 1, s.n_ffn_down, s.k_ffn_down},
            {"FFN down", 8, s.n_ffn_down, s.k_ffn_down},
            {"FFN down", 32, s.n_ffn_down, s.k_ffn_down},
        };

        for (const auto &tc : cases)
        {
            // Just compute accuracy, minimal iterations
            auto result = bench_single(tc.name, tc.M, tc.N, tc.K, 0, 1);
            bool pass = result.l2_error < 0.01;
            if (!pass)
                all_pass = false;

            std::ostringstream l2;
            l2 << std::scientific << std::setprecision(2) << result.l2_error;

            table << model->name << tc.name << tc.M << tc.N << tc.K
                  << l2.str() << (pass ? "PASS" : "FAIL") << fort::endr;
        }
    }

    if (rank_ == 0)
    {
        std::cout << "\n=== Accuracy Verification ===\n"
                  << table.to_string() << std::endl;
    }
    EXPECT_TRUE(all_pass) << "Some accuracy checks failed";
}
