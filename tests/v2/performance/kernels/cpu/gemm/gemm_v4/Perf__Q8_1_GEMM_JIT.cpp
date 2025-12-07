/**
 * @file Perf__Q8_1_GEMM_JIT.cpp
 * @brief Performance benchmark for Q8_1 x Q8_1 JIT GEMM operations
 *
 * This test benchmarks Q8_1 quantized matrix multiplication performance
 * where BOTH weights and activations are Q8_1 quantized.
 * It measures:
 *   - Throughput (GFLOPS)
 *   - Time per iteration (ms)
 *   - Correctness vs FP32 reference
 *
 * @author David Sanftenberg
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
#include <oneapi/dnnl/dnnl.hpp>

// V2 includes
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

using namespace llaminar2;

struct BenchmarkConfig
{
    int seq_len;             ///< Sequence length (m dimension)
    int in_features;         ///< Input feature dimension (k dimension)
    int out_features;        ///< Output feature dimension (n dimension)
    int warmup_iters;        ///< Number of warmup iterations
    int bench_iters;         ///< Number of timed benchmark iterations per trial
    int num_trials;          ///< Number of independent trials for statistics
    std::string description; ///< Human-readable description
};

struct BenchmarkStats
{
    double mean_ms;       ///< Mean time per iteration (ms)
    double stddev_ms;     ///< Standard deviation (ms)
    double min_ms;        ///< Minimum time (ms)
    double max_ms;        ///< Maximum time (ms)
    double mean_gflops;   ///< Mean throughput (GFLOPS)
    double stddev_gflops; ///< Standard deviation of throughput
    double l2_error;      ///< L2 error from verification
    double cosine_sim;    ///< Cosine similarity
};

class Q8_1_GEMM_JIT_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        int max_threads = omp_get_max_threads();
        if (rank_ == 0)
        {
            std::cout << "[Performance Test] OpenMP max threads: " << max_threads << std::endl;
        }
    }

    // Compute reference GEMM using OneDNN (FP32)
    void compute_reference_gemm(int M, int N, int K, const float *A, const float *B, float *C)
    {
        using namespace dnnl;
        engine eng(engine::kind::cpu, 0);
        stream s(eng);

        // A: M x K, row-major
        memory::dims a_dims = {M, K};
        memory::dims a_strides = {K, 1};
        auto a_md = memory::desc(a_dims, memory::data_type::f32, a_strides);
        auto a_mem = memory(a_md, eng, (void *)A);

        // B: K x N, column-major (because B is passed as N x K row-major)
        // We want to compute A * B^T.
        // If B is N x K row-major, it is effectively K x N column-major.
        // So we treat it as K x N matrix.
        memory::dims b_dims = {K, N};
        memory::dims b_strides = {1, K}; // Column major
        auto b_md = memory::desc(b_dims, memory::data_type::f32, b_strides);
        auto b_mem = memory(b_md, eng, (void *)B);

        // C: M x N, row-major
        memory::dims c_dims = {M, N};
        memory::dims c_strides = {N, 1};
        auto c_md = memory::desc(c_dims, memory::data_type::f32, c_strides);
        auto c_mem = memory(c_md, eng, (void *)C);

        auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
        auto matmul_prim = matmul(matmul_pd);

        matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                {DNNL_ARG_WEIGHTS, b_mem},
                                {DNNL_ARG_DST, c_mem}});
        s.wait();
    }

    std::pair<double, double> verify_correctness(int M, int N, int K, const float *A_ref, const float *B_ref, const float *C_act)
    {
        std::vector<float> C_ref(M * N);
        compute_reference_gemm(M, N, K, A_ref, B_ref, C_ref.data());

        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        double dot_prod = 0.0;
        double norm_act = 0.0;
        double norm_ref = 0.0;

        for (size_t i = 0; i < C_ref.size(); ++i)
        {
            double diff = C_act[i] - C_ref[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += C_ref[i] * C_ref[i];

            dot_prod += C_act[i] * C_ref[i];
            norm_act += C_act[i] * C_act[i];
            norm_ref += C_ref[i] * C_ref[i];
        }

        double l2_error = (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_diff) / std::sqrt(sum_sq_ref) : 0.0;
        double cosine_sim = (norm_act > 0.0 && norm_ref > 0.0) ? dot_prod / (std::sqrt(norm_act) * std::sqrt(norm_ref)) : 0.0;

        // Tolerance for Q8_1 x Q8_1 vs FP32
        // This involves double quantization (activations AND weights), so error will be higher than Q8_1 x FP32.
        // However, Q8_1 is high precision (8-bit + block scales), so it should still be good.
        // EXPECT_LT(l2_error, 0.05) << "Large divergence detected! L2 Error > 5%";
        // EXPECT_GT(cosine_sim, 0.99) << "Low cosine similarity!";

        return {l2_error, cosine_sim};
    }

    BenchmarkStats run_benchmark(const BenchmarkConfig &config)
    {
        int M = config.seq_len;
        int K = config.in_features;
        int N = config.out_features;

        // 1. Create random weights (N x K)
        std::vector<float> weights_fp32(N * K);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &x : weights_fp32)
            x = dist(gen);

        // 2. Quantize weights to Q8_1Tensor
        auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

        // 2b. Dequantize weights for reference calculation
        std::vector<float> weights_dequantized(N * K);
        weights_tensor->to_fp32(weights_dequantized.data());

        // 3. Create kernel
        auto kernel = weights_tensor->createGemm();
        if (!kernel)
        {
            throw std::runtime_error("Failed to create GEMM kernel");
        }

        // 4. Create random input A (M x K)
        std::vector<float> A_fp32(M * K);
        for (auto &x : A_fp32)
            x = dist(gen);

        // 5. Quantize activations to Q8_1Tensor
        auto A_tensor = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});

        // 5b. Dequantize activations for reference calculation
        std::vector<float> A_dequantized(M * K);
        A_tensor->to_fp32(A_dequantized.data());

        // 6. Output buffer C (M x N)
        auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

        // Verify correctness (once)
        kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
        auto [l2_error, cosine_sim] = verify_correctness(M, N, K, A_dequantized.data(), weights_dequantized.data(), C_tensor->data());

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
        }

        // Benchmark trials
        std::vector<double> trial_times_ms;
        trial_times_ms.reserve(config.num_trials);

        for (int t = 0; t < config.num_trials; ++t)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.bench_iters; ++i)
            {
                kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();

            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            trial_times_ms.push_back(total_ms / config.bench_iters);
        }

        // Calculate stats
        double sum = std::accumulate(trial_times_ms.begin(), trial_times_ms.end(), 0.0);
        double mean_ms = sum / config.num_trials;

        double sq_sum = std::inner_product(trial_times_ms.begin(), trial_times_ms.end(), trial_times_ms.begin(), 0.0);
        double stddev_ms = std::sqrt(sq_sum / config.num_trials - mean_ms * mean_ms);

        double min_ms = *std::min_element(trial_times_ms.begin(), trial_times_ms.end());
        double max_ms = *std::max_element(trial_times_ms.begin(), trial_times_ms.end());

        // GFLOPS = 2 * M * N * K / (time_s * 1e9)
        double ops = 2.0 * M * N * K;
        double mean_gflops = (ops / (mean_ms * 1e-3)) / 1e9;

        double stddev_gflops = mean_gflops * (stddev_ms / mean_ms);

        return {mean_ms, stddev_ms, min_ms, max_ms, mean_gflops, stddev_gflops, l2_error, cosine_sim};
    }

    void print_results(const BenchmarkConfig &config, const BenchmarkStats &stats)
    {
        if (rank_ != 0)
            return;

        std::cout << std::left << std::setw(40) << config.description
                  << " | M=" << std::setw(4) << config.seq_len
                  << " N=" << std::setw(5) << config.out_features
                  << " K=" << std::setw(5) << config.in_features
                  << " | Time: " << std::fixed << std::setprecision(3) << stats.mean_ms << " ms"
                  << " | T-put: " << std::setprecision(2) << stats.mean_gflops << " GFLOPS"
                  << " | L2 Err: " << std::scientific << std::setprecision(2) << stats.l2_error
                  << " | Cos Sim: " << std::fixed << std::setprecision(5) << stats.cosine_sim
                  << std::endl;
    }
};

// Incremental Decode (M=1)
TEST_F(Q8_1_GEMM_JIT_Perf, IncrementalDecode_Small)
{
    BenchmarkConfig config{
        .seq_len = 1,
        .in_features = 4096,
        .out_features = 4096,
        .warmup_iters = 10,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Incremental Decode (M=1, 4Kx4K)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    // Correctness assertions
    EXPECT_LT(stats.l2_error, 0.05);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_GEMM_JIT_Perf, IncrementalDecode_Large)
{
    BenchmarkConfig config{
        .seq_len = 1,
        .in_features = 11008, // MLP size for 7B
        .out_features = 4096,
        .warmup_iters = 10,
        .bench_iters = 100,
        .num_trials = 5,
        .description = "Incremental Decode (M=1, 11Kx4K)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.05);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

// Batched Prefill
TEST_F(Q8_1_GEMM_JIT_Perf, BatchedPrefill_Small)
{
    BenchmarkConfig config{
        .seq_len = 128,
        .in_features = 4096,
        .out_features = 4096,
        .warmup_iters = 5,
        .bench_iters = 20,
        .num_trials = 3,
        .description = "Batched Prefill (M=128, 4Kx4K)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.05);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

TEST_F(Q8_1_GEMM_JIT_Perf, BatchedPrefill_Medium)
{
    BenchmarkConfig config{
        .seq_len = 512,
        .in_features = 4096,
        .out_features = 4096,
        .warmup_iters = 2,
        .bench_iters = 10,
        .num_trials = 3,
        .description = "Batched Prefill (M=512, 4Kx4K)"};

    auto stats = run_benchmark(config);
    print_results(config, stats);

    EXPECT_LT(stats.l2_error, 0.05);
    EXPECT_GT(stats.cosine_sim, 0.999);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
