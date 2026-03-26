/**
 * @file Perf__TurboQuant.cpp
 * @brief Performance benchmark for TQ4 quantize / dequantize
 *
 * Measures throughput for the two TQ4 hot paths:
 *   1. Quantize:   FP32 → TQ4 (TQ4)
 *   2. Dequantize: TQ4  → FP32 (TQ4)
 *
 * Each operation is benchmarked with warmup, then timed over many iterations.
 * Reports: ops/sec, throughput (GB/s for data-movement-bound ops), and
 * per-vector latency in nanoseconds.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantDequantize.h"
#include "kernels/cpu/turboquant/TurboQuantQuantize.h"

using namespace llaminar2;

// ============================================================================
// Helpers
// ============================================================================

static std::vector<float> make_random_fp32(int count, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(count);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

struct BenchResult
{
    double total_sec;
    int64_t total_ops;
    double ops_per_sec() const { return total_ops / total_sec; }
    double ns_per_op() const { return total_sec * 1e9 / total_ops; }
};

// ============================================================================
// D = 128  (Qwen3 / Llama-3 head_dim)
// ============================================================================

static constexpr int D = 128;
using Block = TQ4Block<D>;

TEST(Perf__TurboQuant, Quantize_ScalarFull_D128)
{
    const int num_vectors = 1024;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_vectors * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks(num_vectors);
    alignas(64) float scratch0[D], scratch1[D];

    // Warmup
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], scratch0, scratch1);

    // Bench
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], scratch0, scratch1);
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_vectors};

    std::cout << "\n=== TQ4 Quantize (D=" << D << ") ===" << std::endl;
    std::cout << "  Vectors:     " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " vec/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/vec" << std::endl;

    double input_gb = static_cast<double>(r.total_ops) * D * 4 / 1e9;
    std::cout << "  Input B/W:   " << std::fixed << std::setprecision(2) << input_gb / r.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

TEST(Perf__TurboQuant, Dequant_ScalarFull_D128)
{
    const int num_blocks = 1024;
    const int iterations = 500;
    const int warmup = 100;

    // Generate blocks by quantizing random data
    auto fp32_data = make_random_fp32(num_blocks * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks(num_blocks);
    alignas(64) float s0[D], s1[D];
    for (int i = 0; i < num_blocks; ++i)
        turboquant_quantize_tq4<D>(
            fp32_data.data() + i * D, ctx, blocks[i], s0, s1);

    std::vector<float> output(num_blocks * D);
    alignas(64) float scratch[D];

    // Warmup
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_blocks; ++i)
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_blocks};

    std::cout << "\n=== TQ4 Dequant (D=" << D << ") ===" << std::endl;
    std::cout << "  Blocks:      " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " vec/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/vec" << std::endl;

    // Block read bandwidth: 72 bytes per block
    double block_gb = static_cast<double>(r.total_ops) * sizeof(Block) / 1e9;
    std::cout << "  Block B/W:   " << std::fixed << std::setprecision(2) << block_gb / r.total_sec << " GB/s" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}

TEST(Perf__TurboQuant, RoundTrip_QuantDequant_D128)
{
    // Full quantize + dequantize roundtrip (e.g., KV cache append + attention dequant)
    const int num_vectors = 1024;
    const int iterations = 200;
    const int warmup = 50;

    auto fp32_data = make_random_fp32(num_vectors * D, 42);
    TurboQuantContext ctx(D, 31, 131);

    std::vector<Block> blocks(num_vectors);
    std::vector<float> output(num_vectors * D);

    alignas(64) float s0[D], s1[D], scratch[D];

    // Warmup
    for (int w = 0; w < warmup; ++w)
        for (int i = 0; i < num_vectors; ++i)
        {
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], s0, s1);
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
        }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
        for (int i = 0; i < num_vectors; ++i)
        {
            turboquant_quantize_tq4<D>(
                fp32_data.data() + i * D, ctx, blocks[i], s0, s1);
            turboquant_dequantize_tq4<D>(
                blocks[i], ctx, output.data() + i * D, scratch);
        }
    auto t1 = std::chrono::high_resolution_clock::now();

    BenchResult r{std::chrono::duration<double>(t1 - t0).count(),
                  static_cast<int64_t>(iterations) * num_vectors};

    std::cout << "\n=== TQ4 Roundtrip (quant + dequant, D=" << D << ") ===" << std::endl;
    std::cout << "  Roundtrips:  " << r.total_ops << std::endl;
    std::cout << "  Time:        " << std::fixed << std::setprecision(3) << r.total_sec * 1e3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << std::fixed << std::setprecision(0) << r.ops_per_sec() << " rt/s" << std::endl;
    std::cout << "  Latency:     " << std::fixed << std::setprecision(0) << r.ns_per_op() << " ns/rt" << std::endl;

    EXPECT_GT(r.ops_per_sec(), 0);
}
