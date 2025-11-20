/**
 * @file Perf__OneDNNGemm_FusedSoftmax.cpp
 * @brief Benchmarks the fused matmul+softmax path backed by OneDNN.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"

using namespace llaminar2::gemm_v4;
using namespace std::chrono;

namespace
{
    std::vector<float> make_random_tensor(size_t elements, unsigned seed)
    {
        std::vector<float> data(elements);
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto &v : data)
        {
            v = dist(rng);
        }
        return data;
    }

    struct ErrorStats
    {
        double max_abs = 0.0;
        double rel_l2 = 0.0;
    };

    ErrorStats compute_error(const std::vector<float> &reference, const std::vector<float> &candidate)
    {
        ErrorStats stats;
        double sum_sq = 0.0;
        double ref_sq = 0.0;
        for (size_t idx = 0; idx < reference.size(); ++idx)
        {
            const double ref = static_cast<double>(reference[idx]);
            const double diff = static_cast<double>(candidate[idx]) - ref;
            stats.max_abs = std::max(stats.max_abs, std::abs(diff));
            sum_sq += diff * diff;
            ref_sq += ref * ref;
        }

        const double denom = ref_sq > 0.0 ? ref_sq : 1.0;
        stats.rel_l2 = std::sqrt(sum_sq / denom);
        return stats;
    }
}

TEST(OneDNNGemmPerformance, FusedMatmulSoftmaxQwenProfile)
{
    // Approximate a single-head attention profile: seq_len^2 softmax with head_dim inputs
    const std::vector<int> batch_sizes = {128, 256, 512, 1024, 2048, 4096, 8192};
    const int head_dim = 128;
    const int K = head_dim;
    const int warmup_iters = 5;
    const int bench_iters = 20;
    constexpr int kSoftmaxAxis = 1; // row-wise softmax

    const int max_seq_len = batch_sizes.back();
    auto activations = make_random_tensor(static_cast<size_t>(max_seq_len) * static_cast<size_t>(K), 42);
    auto keys_transposed = make_random_tensor(static_cast<size_t>(K) * static_cast<size_t>(max_seq_len), 1337);

    auto print_metrics = [](const char *label, std::pair<double, double> metrics)
    {
        std::cout << "    " << std::setw(12) << std::left << label << " : "
                  << std::right << std::fixed << std::setprecision(2)
                  << metrics.first << " ms, " << metrics.second << " GOPS\n";
    };

    std::cout << "======================================================================================================\n";
    std::cout << "==                               OneDNN Fused GEMM + Softmax Diagnostic                            ==\n";
    std::cout << "======================================================================================================\n";
    std::cout << "Head_dim=" << head_dim << " | Warmup=" << warmup_iters << " | Timed iterations per size=" << bench_iters
              << "\n";
    std::cout << "Batch sizes (seq_len):";
    for (size_t i = 0; i < batch_sizes.size(); ++i)
    {
        std::cout << (i == 0 ? " " : ", ") << batch_sizes[i];
    }
    std::cout << "\n\n";

    constexpr double kMaxAbsTolerance = 5e-4;
    constexpr double kRelL2Tolerance = 5e-4;

    for (const int seq_len : batch_sizes)
    {
        const int M = seq_len; // queries
        const int N = seq_len; // keys (after transpose)
        std::vector<float> fused_output(static_cast<size_t>(M) * static_cast<size_t>(N));
        std::vector<float> baseline_output(static_cast<size_t>(M) * static_cast<size_t>(N));

        auto run_fused = [&]() -> bool
        {
            return run_onednn_fp32_matmul_softmax(activations.data(),
                                                  keys_transposed.data(),
                                                  fused_output.data(),
                                                  M,
                                                  N,
                                                  K,
                                                  kSoftmaxAxis);
        };

        auto run_unfused = [&]() -> bool
        {
            if (!run_onednn_fp32_matmul(activations.data(),
                                        keys_transposed.data(),
                                        baseline_output.data(),
                                        M,
                                        N,
                                        K))
            {
                return false;
            }
            return apply_softmax_inplace(baseline_output.data(), M, N, kSoftmaxAxis);
        };

        if (!run_fused())
        {
            GTEST_SKIP() << "OneDNN build does not expose matmul+softmax fusion";
        }
        ASSERT_TRUE(run_unfused());

        for (int i = 1; i < warmup_iters; ++i)
        {
            ASSERT_TRUE(run_fused());
        }
        for (int i = 1; i < warmup_iters; ++i)
        {
            ASSERT_TRUE(run_unfused());
        }

        auto benchmark = [&](auto &&fn) -> std::pair<double, double>
        {
            auto start = high_resolution_clock::now();
            for (int i = 0; i < bench_iters; ++i)
            {
                if (!fn())
                {
                    ADD_FAILURE() << "Benchmark iteration failed";
                    return {0.0, 0.0};
                }
            }
            auto end = high_resolution_clock::now();
            const double total_ns = duration_cast<nanoseconds>(end - start).count();
            const double avg_ms = (total_ns / 1e6) / static_cast<double>(bench_iters);
            const double gops = (2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K)) /
                                (avg_ms * 1e6);
            return {avg_ms, gops};
        };

        const auto fused_metrics = benchmark(run_fused);
        const auto unfused_metrics = benchmark(run_unfused);

        // Produce a fresh set of outputs for correctness comparison
        ASSERT_TRUE(run_fused());
        ASSERT_TRUE(run_unfused());
        const auto stats = compute_error(baseline_output, fused_output);

        ASSERT_LT(stats.max_abs, kMaxAbsTolerance)
            << "Fused path deviates from unfused baseline (max abs diff " << stats.max_abs << ")";
        ASSERT_LT(stats.rel_l2, kRelL2Tolerance)
            << "Fused path deviates from unfused baseline (relative L2 " << stats.rel_l2 << ")";

        const double speedup = unfused_metrics.first / fused_metrics.first;

        std::cout << "Batch size " << std::setw(4) << seq_len << " (M=N=" << M << ")\n";
        std::cout << "  Timing\n";
        print_metrics("Fused", fused_metrics);
        print_metrics("Unfused", unfused_metrics);
        std::cout << "    Speedup        : " << std::fixed << std::setprecision(2) << speedup << "x\n";
        std::cout << "  Accuracy\n";
        std::cout << "    Max abs diff   : " << std::scientific << std::setprecision(3) << stats.max_abs << "\n";
        std::cout << "    Relative L2    : " << std::scientific << std::setprecision(3) << stats.rel_l2 << "\n\n";
    }

    std::cout << "======================================================================================================\n";
}
