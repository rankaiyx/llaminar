/**
 * @file Perf__OneDNNGemm_Int8StreamingSoftmax.cpp
 * @brief Benchmarks INT8 GEMM followed by Llaminar softmax vs. baseline.
 */

#include <gtest/gtest.h>

#include <algorithm>
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
    struct QuantizedRows
    {
        std::vector<int8_t> data;
        std::vector<float> scales;
    };

    QuantizedRows quantize_rows(const std::vector<float> &src, int rows, int cols)
    {
        QuantizedRows q;
        q.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        q.scales.resize(static_cast<size_t>(rows));

        for (int r = 0; r < rows; ++r)
        {
            const float *row_ptr = src.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
            float max_abs = 0.0f;
            for (int c = 0; c < cols; ++c)
            {
                max_abs = std::max(max_abs, std::abs(row_ptr[c]));
            }

            const float scale = max_abs / 127.0f;
            q.scales[static_cast<size_t>(r)] = scale;
            if (scale > 0.0f)
            {
                const float inv_scale = 127.0f / max_abs;
                for (int c = 0; c < cols; ++c)
                {
                    float val = row_ptr[c] * inv_scale;
                    val = std::max(-127.0f, std::min(127.0f, val));
                    q.data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                        static_cast<int8_t>(std::round(val));
                }
            }
            else
            {
                std::fill_n(q.data.data() + static_cast<size_t>(r) * static_cast<size_t>(cols), cols, int8_t(0));
            }
        }

        return q;
    }

    struct QuantizedCols
    {
        std::vector<int8_t> data;
        std::vector<float> scales;
    };

    QuantizedCols quantize_cols(const std::vector<float> &src, int rows, int cols)
    {
        QuantizedCols q;
        q.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
        q.scales.resize(static_cast<size_t>(cols));

        for (int c = 0; c < cols; ++c)
        {
            float max_abs = 0.0f;
            for (int r = 0; r < rows; ++r)
            {
                max_abs = std::max(max_abs, std::abs(src[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)]));
            }

            const float scale = max_abs / 127.0f;
            q.scales[static_cast<size_t>(c)] = scale;
            if (scale > 0.0f)
            {
                const float inv_scale = 127.0f / max_abs;
                for (int r = 0; r < rows; ++r)
                {
                    float val = src[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] * inv_scale;
                    val = std::max(-127.0f, std::min(127.0f, val));
                    q.data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                        static_cast<int8_t>(std::round(val));
                }
            }
            else
            {
                for (int r = 0; r < rows; ++r)
                {
                    q.data[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] = 0;
                }
            }
        }

        return q;
    }

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

TEST(OneDNNGemmPerformance, Int8StreamingSoftmaxQwenProfile)
{
    const int seq_len = 1024;
    const int head_dim = 128;
    const int M = seq_len;
    const int N = seq_len;
    const int K = head_dim;
    const int warmup_iters = 3;
    const int bench_iters = 20;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto activations = make_random_tensor(static_cast<size_t>(M) * static_cast<size_t>(K), 7);
    auto weights = make_random_tensor(static_cast<size_t>(K) * static_cast<size_t>(N), 99);

    const auto q_rows = quantize_rows(activations, M, K);
    const auto q_cols = quantize_cols(weights, K, N);

    std::vector<float> baseline_scores(static_cast<size_t>(M) * static_cast<size_t>(N));
    std::vector<float> fused_scores(static_cast<size_t>(M) * static_cast<size_t>(N));
    static thread_local std::vector<int32_t> accum;
    accum.resize(static_cast<size_t>(M) * static_cast<size_t>(N));

    auto run_baseline = [&]() -> bool
    {
        if (!run_onednn_int8_matmul(q_rows.data.data(),
                                    q_cols.data.data(),
                                    accum.data(),
                                    M,
                                    N,
                                    K))
        {
            return false;
        }

        ActivationView view(baseline_scores.data(), static_cast<size_t>(M), static_cast<size_t>(N));
        if (!view.from_int32_with_scales(accum.data(),
                                         M,
                                         N,
                                         q_rows.scales.data(),
                                         q_cols.scales.data(),
                                         nullptr))
        {
            return false;
        }

        llaminar2::primitives::softmax_row_major_fp32(baseline_scores.data(),
                                                      M,
                                                      N,
                                                      /*causal=*/false,
                                                      softmax_scale,
                                                      /*parallel=*/true);
        return true;
    };

    auto run_fused = [&]() -> bool
    {
        Int8MatmulSoftmaxParams params;
        params.M = M;
        params.N = N;
        params.K = K;
        params.row_scales = q_rows.scales.data();
        params.col_scales = q_cols.scales.data();
        params.softmax_scale = softmax_scale;
        params.causal = false;
        params.parallel_softmax = true;
        return run_onednn_int8_matmul_with_softmax(q_rows.data.data(),
                                                   q_cols.data.data(),
                                                   fused_scores.data(),
                                                   params);
    };

    ASSERT_TRUE(run_baseline());
    ASSERT_TRUE(run_fused());

    for (int i = 1; i < warmup_iters; ++i)
    {
        ASSERT_TRUE(run_baseline());
        ASSERT_TRUE(run_fused());
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

    const auto baseline_metrics = benchmark(run_baseline);
    const auto fused_metrics = benchmark(run_fused);

    ASSERT_TRUE(run_baseline());
    ASSERT_TRUE(run_fused());

    const auto stats = compute_error(baseline_scores, fused_scores);
    constexpr double kMaxAbsTolerance = 5e-3;
    constexpr double kRelL2Tolerance = 1e-3;
    ASSERT_LT(stats.max_abs, kMaxAbsTolerance);
    ASSERT_LT(stats.rel_l2, kRelL2Tolerance);

    const double speedup = baseline_metrics.first / fused_metrics.first;

    auto print_metrics = [](const char *label, std::pair<double, double> metrics)
    {
        std::cout << "  " << std::setw(12) << std::left << label << " : "
                  << std::right << std::fixed << std::setprecision(2)
                  << metrics.first << " ms, " << metrics.second << " GOPS\n";
    };

    std::cout << "======================================================================================================\n";
    std::cout << "==                        OneDNN INT8 GEMM + Llaminar Softmax Diagnostic                         ==\n";
    std::cout << "======================================================================================================\n";
    std::cout << "Configuration: seq_len=" << seq_len << " head_dim=" << head_dim
              << " (M=N=" << M << ", K=" << K << ")\n";
    std::cout << "Warmup iterations: " << warmup_iters << ", Timed iterations: " << bench_iters << "\n\n";
    std::cout << "Step 1: Timing\n";
    print_metrics("Baseline", baseline_metrics);
    print_metrics("Streaming", fused_metrics);
    std::cout << "  Speedup        : " << std::fixed << std::setprecision(2) << speedup << "x\n\n";
    std::cout << "Step 2: Accuracy\n";
    std::cout << "  Max abs diff   : " << std::scientific << std::setprecision(3) << stats.max_abs << "\n";
    std::cout << "  Relative L2    : " << std::scientific << std::setprecision(3) << stats.rel_l2 << "\n";
    std::cout << "======================================================================================================\n";
}
