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

class Q6_K_SIMD_Unpack : public ::testing::Test
{
protected:
    std::mt19937 gen_{42};

    Q6_KBlock create_random_block()
    {
        Q6_KBlock block;
        std::uniform_int_distribution<uint8_t> dist_u8(0, 255);
        std::uniform_int_distribution<int> dist_s8(-128, 127);
        for (int i = 0; i < 128; ++i)
            block.ql[i] = dist_u8(gen_);
        for (int i = 0; i < 64; ++i)
            block.qh[i] = dist_u8(gen_);
        for (int i = 0; i < 16; ++i)
            block.scales[i] = static_cast<int8_t>(dist_s8(gen_));
        block.d = fp32_to_fp16(0.5f + 0.5f * (dist_u8(gen_) / 255.0f));
        return block;
    }

    void verify_arrays_near(const int8_t *expected, const int8_t *actual, int count,
                            int tolerance, const std::string &context)
    {
        for (int i = 0; i < count; ++i)
        {
            int diff = std::abs(static_cast<int>(expected[i]) - static_cast<int>(actual[i]));
            ASSERT_LE(diff, tolerance)
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
            float diff = std::fabs(expected[i] - actual[i]);
            ASSERT_LE(diff, tolerance)
                << context << " mismatch at index " << i
                << ": expected=" << expected[i]
                << " actual=" << actual[i]
                << " diff=" << diff;
        }
    }
};

TEST_F(Q6_K_SIMD_Unpack, ScalarReference_Random)
{
    for (int iter = 0; iter < 100; ++iter)
    {
        Q6_KBlock block = create_random_block();

        alignas(64) int8_t output[256];
        alignas(64) float scales[8];
        alignas(64) float mins[8];
        std::memset(output, 0, sizeof(output));
        std::memset(scales, 0, sizeof(scales));
        std::memset(mins, 0, sizeof(mins));

        unpack_q6_k_superblock_to_int8_scalar(block, output, scales, mins);

        for (int i = 0; i < 256; ++i)
        {
            ASSERT_GE(static_cast<int>(output[i]), -128)
                << "iter=" << iter << " index=" << i;
            ASSERT_LE(static_cast<int>(output[i]), 127)
                << "iter=" << iter << " index=" << i;
        }
    }
}

#if defined(__AVX2__)

TEST_F(Q6_K_SIMD_Unpack, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported at runtime";
    }

    for (int iter = 0; iter < 100; ++iter)
    {
        Q6_KBlock block = create_random_block();

        alignas(64) int8_t output_scalar[256];
        alignas(64) float scales_scalar[8];
        alignas(64) float mins_scalar[8];
        std::memset(output_scalar, 0, sizeof(output_scalar));
        std::memset(scales_scalar, 0, sizeof(scales_scalar));
        std::memset(mins_scalar, 0, sizeof(mins_scalar));

        alignas(64) int8_t output_avx2[256];
        alignas(64) float scales_avx2[8];
        alignas(64) float mins_avx2[8];
        std::memset(output_avx2, 0, sizeof(output_avx2));
        std::memset(scales_avx2, 0, sizeof(scales_avx2));
        std::memset(mins_avx2, 0, sizeof(mins_avx2));

        unpack_q6_k_superblock_to_int8_scalar(block, output_scalar, scales_scalar, mins_scalar);
        unpack_q6_k_superblock_to_int8_avx2(block, output_avx2, scales_avx2, mins_avx2);

        std::string ctx = "iter=" + std::to_string(iter);
        verify_arrays_near(output_scalar, output_avx2, 256, 1, ctx + " output");
        verify_float_arrays_near(scales_scalar, scales_avx2, 8, 1e-5f, ctx + " scales");
        verify_float_arrays_near(mins_scalar, mins_avx2, 8, 1e-5f, ctx + " mins");
    }
}

TEST_F(Q6_K_SIMD_Unpack, AVX2_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported at runtime";
    }

    // Edge case 1: all zeros
    {
        Q6_KBlock block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t output_scalar[256], output_avx2[256];
        alignas(64) float scales_scalar[8], scales_avx2[8];
        alignas(64) float mins_scalar[8], mins_avx2[8];
        std::memset(output_scalar, 0, sizeof(output_scalar));
        std::memset(scales_scalar, 0, sizeof(scales_scalar));
        std::memset(mins_scalar, 0, sizeof(mins_scalar));
        std::memset(output_avx2, 0, sizeof(output_avx2));
        std::memset(scales_avx2, 0, sizeof(scales_avx2));
        std::memset(mins_avx2, 0, sizeof(mins_avx2));

        unpack_q6_k_superblock_to_int8_scalar(block, output_scalar, scales_scalar, mins_scalar);
        unpack_q6_k_superblock_to_int8_avx2(block, output_avx2, scales_avx2, mins_avx2);

        verify_arrays_near(output_scalar, output_avx2, 256, 1, "all_zeros output");
        verify_float_arrays_near(scales_scalar, scales_avx2, 8, 1e-5f, "all_zeros scales");
        verify_float_arrays_near(mins_scalar, mins_avx2, 8, 1e-5f, "all_zeros mins");
    }

    // Edge case 2: all 0xFF
    {
        Q6_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t output_scalar[256], output_avx2[256];
        alignas(64) float scales_scalar[8], scales_avx2[8];
        alignas(64) float mins_scalar[8], mins_avx2[8];
        std::memset(output_scalar, 0, sizeof(output_scalar));
        std::memset(scales_scalar, 0, sizeof(scales_scalar));
        std::memset(mins_scalar, 0, sizeof(mins_scalar));
        std::memset(output_avx2, 0, sizeof(output_avx2));
        std::memset(scales_avx2, 0, sizeof(scales_avx2));
        std::memset(mins_avx2, 0, sizeof(mins_avx2));

        unpack_q6_k_superblock_to_int8_scalar(block, output_scalar, scales_scalar, mins_scalar);
        unpack_q6_k_superblock_to_int8_avx2(block, output_avx2, scales_avx2, mins_avx2);

        verify_arrays_near(output_scalar, output_avx2, 256, 1, "all_0xFF output");
        verify_float_arrays_near(scales_scalar, scales_avx2, 8, 1e-5f, "all_0xFF scales");
        verify_float_arrays_near(mins_scalar, mins_avx2, 8, 1e-5f, "all_0xFF mins");
    }
}

#endif // __AVX2__

#if defined(__AVX512F__) && defined(__AVX512BW__)

TEST_F(Q6_K_SIMD_Unpack, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported at runtime";
    }

    for (int iter = 0; iter < 100; ++iter)
    {
        Q6_KBlock block = create_random_block();

        alignas(64) int8_t output_scalar[256];
        alignas(64) float scales_scalar[8];
        alignas(64) float mins_scalar[8];
        std::memset(output_scalar, 0, sizeof(output_scalar));
        std::memset(scales_scalar, 0, sizeof(scales_scalar));
        std::memset(mins_scalar, 0, sizeof(mins_scalar));

        alignas(64) int8_t output_avx512[256];
        alignas(64) float scales_avx512[8];
        alignas(64) float mins_avx512[8];
        std::memset(output_avx512, 0, sizeof(output_avx512));
        std::memset(scales_avx512, 0, sizeof(scales_avx512));
        std::memset(mins_avx512, 0, sizeof(mins_avx512));

        unpack_q6_k_superblock_to_int8_scalar(block, output_scalar, scales_scalar, mins_scalar);
        unpack_q6_k_superblock_to_int8_avx512(block, output_avx512, scales_avx512, mins_avx512);

        std::string ctx = "iter=" + std::to_string(iter);
        verify_arrays_near(output_scalar, output_avx512, 256, 1, ctx + " output");
        verify_float_arrays_near(scales_scalar, scales_avx512, 8, 1e-5f, ctx + " scales");
        verify_float_arrays_near(mins_scalar, mins_avx512, 8, 1e-5f, ctx + " mins");
    }
}

TEST_F(Q6_K_SIMD_Unpack, AVX512_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported at runtime";
    }

    // Edge case 1: all zeros
    {
        Q6_KBlock block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t output_scalar[256], output_avx512[256];
        alignas(64) float scales_scalar[8], scales_avx512[8];
        alignas(64) float mins_scalar[8], mins_avx512[8];
        std::memset(output_scalar, 0, sizeof(output_scalar));
        std::memset(scales_scalar, 0, sizeof(scales_scalar));
        std::memset(mins_scalar, 0, sizeof(mins_scalar));
        std::memset(output_avx512, 0, sizeof(output_avx512));
        std::memset(scales_avx512, 0, sizeof(scales_avx512));
        std::memset(mins_avx512, 0, sizeof(mins_avx512));

        unpack_q6_k_superblock_to_int8_scalar(block, output_scalar, scales_scalar, mins_scalar);
        unpack_q6_k_superblock_to_int8_avx512(block, output_avx512, scales_avx512, mins_avx512);

        verify_arrays_near(output_scalar, output_avx512, 256, 1, "all_zeros output");
        verify_float_arrays_near(scales_scalar, scales_avx512, 8, 1e-5f, "all_zeros scales");
        verify_float_arrays_near(mins_scalar, mins_avx512, 8, 1e-5f, "all_zeros mins");
    }

    // Edge case 2: all 0xFF
    {
        Q6_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t output_scalar[256], output_avx512[256];
        alignas(64) float scales_scalar[8], scales_avx512[8];
        alignas(64) float mins_scalar[8], mins_avx512[8];
        std::memset(output_scalar, 0, sizeof(output_scalar));
        std::memset(scales_scalar, 0, sizeof(scales_scalar));
        std::memset(mins_scalar, 0, sizeof(mins_scalar));
        std::memset(output_avx512, 0, sizeof(output_avx512));
        std::memset(scales_avx512, 0, sizeof(scales_avx512));
        std::memset(mins_avx512, 0, sizeof(mins_avx512));

        unpack_q6_k_superblock_to_int8_scalar(block, output_scalar, scales_scalar, mins_scalar);
        unpack_q6_k_superblock_to_int8_avx512(block, output_avx512, scales_avx512, mins_avx512);

        verify_arrays_near(output_scalar, output_avx512, 256, 1, "all_0xFF output");
        verify_float_arrays_near(scales_scalar, scales_avx512, 8, 1e-5f, "all_0xFF scales");
        verify_float_arrays_near(mins_scalar, mins_avx512, 8, 1e-5f, "all_0xFF mins");
    }
}

#endif // __AVX512F__ && __AVX512BW__
