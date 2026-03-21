#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"

using namespace llaminar2;
using namespace llaminar2::simd;

TEST(Perf__Q5_1_Unpack, SIMD_Tier_Benchmark)
{
    const int num_blocks = 10000;
    const int iterations = 1000;
    const int warmup = 100;

    std::vector<Q5_1Block> blocks(num_blocks);
    std::vector<int8_t> output(num_blocks * 32);

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    for (auto &block : blocks)
    {
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.1f);
        for (int i = 0; i < 4; ++i)
            block.qh[i] = dist(gen);
        for (int i = 0; i < 16; ++i)
            block.qs[i] = dist(gen);
    }

    auto benchmark = [&](auto fn, const char *label) -> double {
        for (int w = 0; w < warmup; ++w)
            for (int i = 0; i < num_blocks; ++i)
                fn(i);

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter)
            for (int i = 0; i < num_blocks; ++i)
                fn(i);
        auto end = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(end - start).count();
        double total_bytes = (double)num_blocks * 32 * iterations;
        double gb_s = (total_bytes / 1e9) / sec;

        std::cout << "  " << std::setw(10) << label << ": "
                  << std::fixed << std::setprecision(2) << gb_s << " GB/s  ("
                  << std::setprecision(3) << (sec * 1000.0) << " ms)" << std::endl;
        return sec;
    };

    std::cout << "\n=== Q5_1 Unpack Performance ===" << std::endl;

    double scalar_time = benchmark([&](int i) {
        unpack_q5_1_to_int8_scalar(blocks[i], output.data() + i * 32);
        asm volatile("" ::: "memory");
    }, "Scalar");

    double avx2_time = scalar_time;
#if defined(__AVX2__)
    if (cpu_supports_avx2())
    {
        avx2_time = benchmark([&](int i) {
            unpack_q5_1_to_int8_avx2(blocks[i], output.data() + i * 32);
            asm volatile("" ::: "memory");
        }, "AVX2");
    }
#endif

    double avx512_time = avx2_time;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (cpu_supports_avx512())
    {
        avx512_time = benchmark([&](int i) {
            unpack_q5_1_to_int8_avx512(blocks[i], output.data() + i * 32);
            asm volatile("" ::: "memory");
        }, "AVX-512");
    }
#endif

    std::cout << "\n  Speedup ratios:" << std::endl;
    std::cout << "    AVX2/Scalar:    " << std::fixed << std::setprecision(2) << scalar_time / avx2_time << "x" << std::endl;
    std::cout << "    AVX512/Scalar:  " << std::fixed << std::setprecision(2) << scalar_time / avx512_time << "x" << std::endl;
    std::cout << "    AVX512/AVX2:    " << std::fixed << std::setprecision(2) << avx2_time / avx512_time << "x" << std::endl;

    EXPECT_LE(avx2_time, scalar_time * 1.10) << "AVX2 should be at least as fast as scalar";
    EXPECT_LE(avx512_time, scalar_time * 1.10) << "AVX-512 should be at least as fast as scalar";
    if (avx512_time > avx2_time * 1.05) {
        std::cout << "  NOTE: AVX-512 was slower than AVX2 (" << std::fixed << std::setprecision(2) << avx2_time / avx512_time << "x). This may be expected due to AVX-512 clock throttling on this CPU." << std::endl;
    }
}
