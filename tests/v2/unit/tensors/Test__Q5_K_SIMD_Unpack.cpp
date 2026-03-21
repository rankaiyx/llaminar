#include <gtest/gtest.h>
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"
#include <vector>
#include <cstring>
#include <random>
#include <cmath>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q5_K_SIMD_Unpack : public ::testing::Test
{
protected:
    std::mt19937 gen_{42};

    Q5_KBlock create_random_block()
    {
        Q5_KBlock block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        block.d = fp32_to_fp16(0.5f + 0.5f * (dist(gen_) / 255.0f));
        block.dmin = fp32_to_fp16(0.1f * (dist(gen_) / 255.0f));
        for (int i = 0; i < 12; ++i)
            block.scales[i] = dist(gen_);
        for (int i = 0; i < 32; ++i)
            block.qh[i] = dist(gen_);
        for (int i = 0; i < 128; ++i)
            block.qs[i] = dist(gen_);
        return block;
    }

    void verify_arrays_near(const int8_t *expected, const int8_t *actual, int count,
                            int tolerance, const std::string &context)
    {
        for (int i = 0; i < count; ++i)
        {
            int diff = std::abs(static_cast<int>(expected[i]) - static_cast<int>(actual[i]));
            EXPECT_LE(diff, tolerance)
                << context << " mismatch at index " << i
                << ": expected=" << static_cast<int>(expected[i])
                << " actual=" << static_cast<int>(actual[i])
                << " diff=" << diff;
        }
    }

    void verify_float_arrays_near(const float *expected, const float *actual, int count,
                                  float tolerance, const std::string &context)
    {
        for (int i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            EXPECT_LE(diff, tolerance)
                << context << " mismatch at index " << i
                << ": expected=" << expected[i]
                << " actual=" << actual[i]
                << " diff=" << diff;
        }
    }
};

TEST_F(Q5_K_SIMD_Unpack, ScalarReference_Random)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_KBlock block = create_random_block();
        alignas(64) int8_t output[256];
        alignas(64) float scales[8];
        alignas(64) float mins[8];

        unpack_q5_k_superblock_to_int8_scalar(block, output, scales, mins);

        for (int i = 0; i < 256; ++i)
        {
            EXPECT_GE(static_cast<int>(output[i]), -128)
                << "Trial " << trial << " index " << i;
            EXPECT_LE(static_cast<int>(output[i]), 127)
                << "Trial " << trial << " index " << i;
        }
    }
}

#if defined(__AVX2__)

TEST_F(Q5_K_SIMD_Unpack, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_KBlock block = create_random_block();

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8];
        alignas(64) float scalar_mins[8];
        unpack_q5_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);

        alignas(64) int8_t avx2_output[256];
        alignas(64) float avx2_scales[8];
        alignas(64) float avx2_mins[8];
        unpack_q5_k_superblock_to_int8_avx2(block, avx2_output, avx2_scales, avx2_mins);

        verify_arrays_near(scalar_output, avx2_output, 256, 1,
                           "AVX2 vs Scalar trial " + std::to_string(trial) + " output");
        verify_float_arrays_near(scalar_scales, avx2_scales, 8, 1e-5f,
                                 "AVX2 vs Scalar trial " + std::to_string(trial) + " scales");
        verify_float_arrays_near(scalar_mins, avx2_mins, 8, 1e-5f,
                                 "AVX2 vs Scalar trial " + std::to_string(trial) + " mins");
    }
}

TEST_F(Q5_K_SIMD_Unpack, AVX2_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    // Edge case 1: all zeros
    {
        Q5_KBlock block;
        std::memset(&block, 0, sizeof(block));

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8];
        alignas(64) float scalar_mins[8];
        unpack_q5_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);

        alignas(64) int8_t avx2_output[256];
        alignas(64) float avx2_scales[8];
        alignas(64) float avx2_mins[8];
        unpack_q5_k_superblock_to_int8_avx2(block, avx2_output, avx2_scales, avx2_mins);

        verify_arrays_near(scalar_output, avx2_output, 256, 1, "AVX2 all-zeros output");
        verify_float_arrays_near(scalar_scales, avx2_scales, 8, 1e-5f, "AVX2 all-zeros scales");
        verify_float_arrays_near(scalar_mins, avx2_mins, 8, 1e-5f, "AVX2 all-zeros mins");
    }

    // Edge case 2: all 0xFF (set d/dmin to valid FP16 to avoid NaN)
    {
        Q5_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.dmin = fp32_to_fp16(0.5f);

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8];
        alignas(64) float scalar_mins[8];
        unpack_q5_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);

        alignas(64) int8_t avx2_output[256];
        alignas(64) float avx2_scales[8];
        alignas(64) float avx2_mins[8];
        unpack_q5_k_superblock_to_int8_avx2(block, avx2_output, avx2_scales, avx2_mins);

        verify_arrays_near(scalar_output, avx2_output, 256, 1, "AVX2 all-0xFF output");
        verify_float_arrays_near(scalar_scales, avx2_scales, 8, 1e-5f, "AVX2 all-0xFF scales");
        verify_float_arrays_near(scalar_mins, avx2_mins, 8, 1e-5f, "AVX2 all-0xFF mins");
    }
}

#endif // __AVX2__

#if defined(__AVX512F__) && defined(__AVX512BW__)

TEST_F(Q5_K_SIMD_Unpack, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_KBlock block = create_random_block();

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8];
        alignas(64) float scalar_mins[8];
        unpack_q5_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);

        alignas(64) int8_t avx512_output[256];
        alignas(64) float avx512_scales[8];
        alignas(64) float avx512_mins[8];
        unpack_q5_k_superblock_to_int8_avx512(block, avx512_output, avx512_scales, avx512_mins);

        verify_arrays_near(scalar_output, avx512_output, 256, 1,
                           "AVX512 vs Scalar trial " + std::to_string(trial) + " output");
        verify_float_arrays_near(scalar_scales, avx512_scales, 8, 1e-5f,
                                 "AVX512 vs Scalar trial " + std::to_string(trial) + " scales");
        verify_float_arrays_near(scalar_mins, avx512_mins, 8, 1e-5f,
                                 "AVX512 vs Scalar trial " + std::to_string(trial) + " mins");
    }
}

TEST_F(Q5_K_SIMD_Unpack, AVX512_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    // Edge case 1: all zeros
    {
        Q5_KBlock block;
        std::memset(&block, 0, sizeof(block));

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8];
        alignas(64) float scalar_mins[8];
        unpack_q5_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);

        alignas(64) int8_t avx512_output[256];
        alignas(64) float avx512_scales[8];
        alignas(64) float avx512_mins[8];
        unpack_q5_k_superblock_to_int8_avx512(block, avx512_output, avx512_scales, avx512_mins);

        verify_arrays_near(scalar_output, avx512_output, 256, 1, "AVX512 all-zeros output");
        verify_float_arrays_near(scalar_scales, avx512_scales, 8, 1e-5f, "AVX512 all-zeros scales");
        verify_float_arrays_near(scalar_mins, avx512_mins, 8, 1e-5f, "AVX512 all-zeros mins");
    }

    // Edge case 2: all 0xFF
    {
        Q5_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.dmin = fp32_to_fp16(0.5f);

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8];
        alignas(64) float scalar_mins[8];
        unpack_q5_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);

        alignas(64) int8_t avx512_output[256];
        alignas(64) float avx512_scales[8];
        alignas(64) float avx512_mins[8];
        unpack_q5_k_superblock_to_int8_avx512(block, avx512_output, avx512_scales, avx512_mins);

        verify_arrays_near(scalar_output, avx512_output, 256, 1, "AVX512 all-0xFF output");
        verify_float_arrays_near(scalar_scales, avx512_scales, 8, 1e-5f, "AVX512 all-0xFF scales");
        verify_float_arrays_near(scalar_mins, avx512_mins, 8, 1e-5f, "AVX512 all-0xFF mins");
    }
}

#endif // __AVX512F__ && __AVX512BW__
