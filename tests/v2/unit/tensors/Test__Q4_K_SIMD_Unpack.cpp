/**
 * @file Test__Q4_K_SIMD_Unpack.cpp
 * @brief Unit tests for Q4_K SIMD unpacking functions
 *
 * Tests verify that AVX2 and AVX-512 implementations produce results
 * matching the scalar reference for Q4_K super-block unpacking (256 elements).
 */

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

class Q4_K_SIMD_Unpack : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42);
    }

    Q4_KBlock create_random_block()
    {
        Q4_KBlock block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        block.d = fp32_to_fp16(0.5f + 0.5f * (dist(gen_) / 255.0f));
        block.dmin = fp32_to_fp16(0.1f * (dist(gen_) / 255.0f));
        for (int i = 0; i < 12; ++i)
            block.scales[i] = dist(gen_);
        for (int i = 0; i < 128; ++i)
            block.qs[i] = dist(gen_);
        return block;
    }

    void verify_arrays_near(const int8_t *expected, const int8_t *actual,
                            size_t count, int tolerance, const std::string &context)
    {
        for (size_t i = 0; i < count; ++i)
        {
            int diff = std::abs(static_cast<int>(expected[i]) - static_cast<int>(actual[i]));
            EXPECT_LE(diff, tolerance)
                << context << ": Mismatch at index " << i
                << " (expected " << static_cast<int>(expected[i])
                << ", got " << static_cast<int>(actual[i])
                << ", diff " << diff << ")";
        }
    }

    std::mt19937 gen_;
};

// ============================================================================
// Scalar Reference Tests
// ============================================================================

TEST_F(Q4_K_SIMD_Unpack, ScalarReference_Random)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        Q4_KBlock block = create_random_block();
        alignas(64) int8_t output[256];
        alignas(64) float scales[8], mins[8];

        unpack_q4_k_superblock_to_int8_scalar(block, output, scales, mins);

        for (int i = 0; i < 256; ++i)
        {
            EXPECT_GE(output[i], -128)
                << "Value below range at index " << i << " trial " << trial;
            EXPECT_LE(output[i], 127)
                << "Value above range at index " << i << " trial " << trial;
        }
    }
}

// ============================================================================
// AVX2 vs Scalar Parity
// ============================================================================

#if defined(__AVX2__)
TEST_F(Q4_K_SIMD_Unpack, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q4_KBlock block = create_random_block();

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8], scalar_mins[8];

        alignas(64) int8_t avx2_output[256];
        alignas(64) float avx2_scales[8], avx2_mins[8];

        unpack_q4_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);
        unpack_q4_k_superblock_to_int8_avx2(block, avx2_output, avx2_scales, avx2_mins);

        verify_arrays_near(scalar_output, avx2_output, 256, 1,
                           "AVX2 vs Scalar (random trial " + std::to_string(trial) + ")");

        for (int s = 0; s < 8; ++s)
        {
            EXPECT_NEAR(scalar_scales[s], avx2_scales[s], 1e-3f)
                << "Scale mismatch at subblock " << s << " trial " << trial;
            EXPECT_NEAR(scalar_mins[s], avx2_mins[s], 1e-3f)
                << "Min mismatch at subblock " << s << " trial " << trial;
        }
    }
}

TEST_F(Q4_K_SIMD_Unpack, AVX2_vs_Scalar_AllZeros)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    Q4_KBlock block;
    std::memset(&block, 0, sizeof(block));
    block.d = fp32_to_fp16(1.0f);

    alignas(64) int8_t scalar_output[256];
    alignas(64) float scalar_scales[8], scalar_mins[8];

    alignas(64) int8_t avx2_output[256];
    alignas(64) float avx2_scales[8], avx2_mins[8];

    unpack_q4_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);
    unpack_q4_k_superblock_to_int8_avx2(block, avx2_output, avx2_scales, avx2_mins);

    verify_arrays_near(scalar_output, avx2_output, 256, 1, "AVX2 vs Scalar (all zeros)");

    for (int s = 0; s < 8; ++s)
    {
        EXPECT_NEAR(scalar_scales[s], avx2_scales[s], 1e-3f)
            << "Scale mismatch at subblock " << s;
        EXPECT_NEAR(scalar_mins[s], avx2_mins[s], 1e-3f)
            << "Min mismatch at subblock " << s;
    }
}

TEST_F(Q4_K_SIMD_Unpack, AVX2_vs_Scalar_AllOnes)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    Q4_KBlock block;
    std::memset(&block, 0, sizeof(block));
    block.d = fp32_to_fp16(1.0f);
    block.dmin = fp32_to_fp16(0.5f);
    std::memset(block.scales, 0xFF, 12);
    std::memset(block.qs, 0xFF, 128);

    alignas(64) int8_t scalar_output[256];
    alignas(64) float scalar_scales[8], scalar_mins[8];

    alignas(64) int8_t avx2_output[256];
    alignas(64) float avx2_scales[8], avx2_mins[8];

    unpack_q4_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);
    unpack_q4_k_superblock_to_int8_avx2(block, avx2_output, avx2_scales, avx2_mins);

    verify_arrays_near(scalar_output, avx2_output, 256, 1, "AVX2 vs Scalar (all ones)");

    for (int s = 0; s < 8; ++s)
    {
        EXPECT_NEAR(scalar_scales[s], avx2_scales[s], 1e-3f)
            << "Scale mismatch at subblock " << s;
        EXPECT_NEAR(scalar_mins[s], avx2_mins[s], 1e-3f)
            << "Min mismatch at subblock " << s;
    }
}
#endif

// ============================================================================
// AVX-512 vs Scalar Parity
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q4_K_SIMD_Unpack, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q4_KBlock block = create_random_block();

        alignas(64) int8_t scalar_output[256];
        alignas(64) float scalar_scales[8], scalar_mins[8];

        alignas(64) int8_t avx512_output[256];
        alignas(64) float avx512_scales[8], avx512_mins[8];

        unpack_q4_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);
        unpack_q4_k_superblock_to_int8_avx512(block, avx512_output, avx512_scales, avx512_mins);

        verify_arrays_near(scalar_output, avx512_output, 256, 1,
                           "AVX512 vs Scalar (random trial " + std::to_string(trial) + ")");

        for (int s = 0; s < 8; ++s)
        {
            EXPECT_NEAR(scalar_scales[s], avx512_scales[s], 1e-3f)
                << "Scale mismatch at subblock " << s << " trial " << trial;
            EXPECT_NEAR(scalar_mins[s], avx512_mins[s], 1e-3f)
                << "Min mismatch at subblock " << s << " trial " << trial;
        }
    }
}

TEST_F(Q4_K_SIMD_Unpack, AVX512_vs_Scalar_AllZeros)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    Q4_KBlock block;
    std::memset(&block, 0, sizeof(block));
    block.d = fp32_to_fp16(1.0f);

    alignas(64) int8_t scalar_output[256];
    alignas(64) float scalar_scales[8], scalar_mins[8];

    alignas(64) int8_t avx512_output[256];
    alignas(64) float avx512_scales[8], avx512_mins[8];

    unpack_q4_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);
    unpack_q4_k_superblock_to_int8_avx512(block, avx512_output, avx512_scales, avx512_mins);

    verify_arrays_near(scalar_output, avx512_output, 256, 1, "AVX512 vs Scalar (all zeros)");

    for (int s = 0; s < 8; ++s)
    {
        EXPECT_NEAR(scalar_scales[s], avx512_scales[s], 1e-3f)
            << "Scale mismatch at subblock " << s;
        EXPECT_NEAR(scalar_mins[s], avx512_mins[s], 1e-3f)
            << "Min mismatch at subblock " << s;
    }
}

TEST_F(Q4_K_SIMD_Unpack, AVX512_vs_Scalar_AllOnes)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    Q4_KBlock block;
    std::memset(&block, 0, sizeof(block));
    block.d = fp32_to_fp16(1.0f);
    block.dmin = fp32_to_fp16(0.5f);
    std::memset(block.scales, 0xFF, 12);
    std::memset(block.qs, 0xFF, 128);

    alignas(64) int8_t scalar_output[256];
    alignas(64) float scalar_scales[8], scalar_mins[8];

    alignas(64) int8_t avx512_output[256];
    alignas(64) float avx512_scales[8], avx512_mins[8];

    unpack_q4_k_superblock_to_int8_scalar(block, scalar_output, scalar_scales, scalar_mins);
    unpack_q4_k_superblock_to_int8_avx512(block, avx512_output, avx512_scales, avx512_mins);

    verify_arrays_near(scalar_output, avx512_output, 256, 1, "AVX512 vs Scalar (all ones)");

    for (int s = 0; s < 8; ++s)
    {
        EXPECT_NEAR(scalar_scales[s], avx512_scales[s], 1e-3f)
            << "Scale mismatch at subblock " << s;
        EXPECT_NEAR(scalar_mins[s], avx512_mins[s], 1e-3f)
            << "Min mismatch at subblock " << s;
    }
}
#endif
