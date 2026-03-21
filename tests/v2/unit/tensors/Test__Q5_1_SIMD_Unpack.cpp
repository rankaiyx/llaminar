/**
 * @file Test__Q5_1_SIMD_Unpack.cpp
 * @brief Unit tests for Q5_1 SIMD unpacking functions
 *
 * Tests verify that all SIMD implementations (AVX-512, AVX2, scalar)
 * produce exactly identical results for Q5_1 unpacking operations.
 *
 * Q5_1 block layout (32 elements per block):
 *   uint16_t d       - scale (FP16)
 *   uint16_t m       - min (FP16)
 *   uint8_t  qh[4]   - high bits (5th bit for each of 32 elements)
 *   uint8_t  qs[16]  - low 4 bits (two nibbles per byte)
 *
 * Unpacking: value = low_nibble | (high_bit << 4), range [0, 31] (unsigned)
 */

#include <gtest/gtest.h>
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include "utils/CPUFeatures.h"
#include <vector>
#include <cstring>
#include <random>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q5_1_SIMD_Unpack : public ::testing::Test
{
protected:
    std::mt19937 gen_{42};

    Q5_1Block create_random_block()
    {
        Q5_1Block block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);
        for (int i = 0; i < 4; ++i)
            block.qh[i] = dist(gen_);
        for (int i = 0; i < 16; ++i)
            block.qs[i] = dist(gen_);
        return block;
    }

    void verify_arrays_equal(const int8_t *expected, const int8_t *actual, size_t count,
                             const std::string &context)
    {
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_EQ(expected[i], actual[i])
                << context << ": Mismatch at index " << i
                << " (expected " << static_cast<int>(expected[i])
                << ", got " << static_cast<int>(actual[i]) << ")";
        }
    }
};

// ============================================================================
// Scalar Reference Tests
// ============================================================================

TEST_F(Q5_1_SIMD_Unpack, ScalarReference_Range)
{
    // Verify all unpacked values are in [0, 31] for random blocks
    // Q5_1 is unsigned: value = low_nibble | (high_bit << 4)
    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_1Block block = create_random_block();
        alignas(64) int8_t output[32];

        unpack_q5_1_to_int8_scalar(block, output);

        for (int i = 0; i < 32; ++i)
        {
            // Cast to unsigned for range check: values are 5-bit unsigned [0, 31]
            int val = static_cast<uint8_t>(output[i]);
            EXPECT_GE(val, 0)
                << "Trial " << trial << ", index " << i
                << ": value " << val << " < 0";
            EXPECT_LE(val, 31)
                << "Trial " << trial << ", index " << i
                << ": value " << val << " > 31";
        }
    }
}

TEST_F(Q5_1_SIMD_Unpack, ScalarReference_AllZeros)
{
    Q5_1Block block;
    std::memset(&block, 0, sizeof(block));
    block.d = fp32_to_fp16(1.0f);
    block.m = fp32_to_fp16(0.0f);

    alignas(64) int8_t output[32];
    unpack_q5_1_to_int8_scalar(block, output);

    // All nibbles = 0, all high bits = 0 → value = 0
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output[i], 0)
            << "Index " << i << ": expected 0, got " << static_cast<int>(output[i]);
    }
}

// ============================================================================
// AVX2 vs Scalar Parity
// ============================================================================

#if defined(__AVX2__)
TEST_F(Q5_1_SIMD_Unpack, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_1Block block = create_random_block();
        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx2_output[32];

        unpack_q5_1_to_int8_scalar(block, scalar_output);
        unpack_q5_1_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32,
                            "AVX2 vs Scalar (random trial " + std::to_string(trial) + ")");
    }
}

TEST_F(Q5_1_SIMD_Unpack, AVX2_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    // Edge case 1: All zeros
    {
        Q5_1Block block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx2_output[32];

        unpack_q5_1_to_int8_scalar(block, scalar_output);
        unpack_q5_1_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (all zeros)");
    }

    // Edge case 2: All ones (0xFF in qs and qh)
    {
        Q5_1Block block;
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);
        std::memset(block.qh, 0xFF, 4);
        std::memset(block.qs, 0xFF, 16);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx2_output[32];

        unpack_q5_1_to_int8_scalar(block, scalar_output);
        unpack_q5_1_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (all 0xFF)");
    }
}
#endif

// ============================================================================
// AVX-512 vs Scalar Parity
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q5_1_SIMD_Unpack, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_1Block block = create_random_block();
        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx512_output[32];

        unpack_q5_1_to_int8_scalar(block, scalar_output);
        unpack_q5_1_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32,
                            "AVX-512 vs Scalar (random trial " + std::to_string(trial) + ")");
    }
}

TEST_F(Q5_1_SIMD_Unpack, AVX512_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    // Edge case 1: All zeros
    {
        Q5_1Block block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx512_output[32];

        unpack_q5_1_to_int8_scalar(block, scalar_output);
        unpack_q5_1_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32, "AVX-512 vs Scalar (all zeros)");
    }

    // Edge case 2: All ones (0xFF in qs and qh)
    {
        Q5_1Block block;
        block.d = fp32_to_fp16(1.0f);
        block.m = fp32_to_fp16(0.0f);
        std::memset(block.qh, 0xFF, 4);
        std::memset(block.qs, 0xFF, 16);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx512_output[32];

        unpack_q5_1_to_int8_scalar(block, scalar_output);
        unpack_q5_1_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32, "AVX-512 vs Scalar (all 0xFF)");
    }
}
#endif
