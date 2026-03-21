/**
 * @file Test__Q5_0_SIMD_Unpack.cpp
 * @brief Unit tests for Q5_0 SIMD unpacking functions
 *
 * Tests verify that all SIMD implementations (AVX-512, AVX2, scalar)
 * produce identical results for Q5_0 unpacking operations.
 *
 * Q5_0 block layout (32 elements per block):
 *   uint16_t d       - scale (FP16)
 *   uint8_t  qh[4]   - high bits (5th bit for each of 32 elements)
 *   uint8_t  qs[16]  - low 4 bits (two nibbles per byte)
 *
 * Unpacking: value = (low_nibble | (high_bit << 4)) - 16, range [-16, 15]
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

class Q5_0_SIMD_Unpack : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42);
    }

    Q5_0Block create_random_block()
    {
        Q5_0Block block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        block.d = fp32_to_fp16(1.0f);
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

    std::mt19937 gen_{42};
};

// ============================================================================
// Scalar Reference Tests
// ============================================================================

TEST_F(Q5_0_SIMD_Unpack, ScalarReference_Range)
{
    // Verify all unpacked values are in [-16, 15] for random blocks
    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_0Block block = create_random_block();
        alignas(64) int8_t output[32];

        unpack_q5_0_to_int8_scalar(block, output);

        for (int i = 0; i < 32; ++i)
        {
            EXPECT_GE(output[i], -16)
                << "Trial " << trial << ", index " << i
                << ": value " << static_cast<int>(output[i]) << " < -16";
            EXPECT_LE(output[i], 15)
                << "Trial " << trial << ", index " << i
                << ": value " << static_cast<int>(output[i]) << " > 15";
        }
    }
}

TEST_F(Q5_0_SIMD_Unpack, ScalarReference_AllZeros)
{
    Q5_0Block block;
    std::memset(&block, 0, sizeof(block));
    block.d = fp32_to_fp16(1.0f);

    alignas(64) int8_t output[32];
    unpack_q5_0_to_int8_scalar(block, output);

    // All nibbles = 0, all high bits = 0 → (0 | 0) - 16 = -16
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output[i], -16)
            << "Index " << i << ": expected -16, got " << static_cast<int>(output[i]);
    }
}

// ============================================================================
// AVX2 vs Scalar Parity
// ============================================================================

#if defined(__AVX2__)
TEST_F(Q5_0_SIMD_Unpack, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_0Block block = create_random_block();
        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx2_output[32];

        unpack_q5_0_to_int8_scalar(block, scalar_output);
        unpack_q5_0_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32,
                            "AVX2 vs Scalar (random trial " + std::to_string(trial) + ")");
    }
}

TEST_F(Q5_0_SIMD_Unpack, AVX2_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx2())
    {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }

    // Edge case 1: All zeros
    {
        Q5_0Block block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx2_output[32];

        unpack_q5_0_to_int8_scalar(block, scalar_output);
        unpack_q5_0_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (all zeros)");
    }

    // Edge case 2: All ones (0xFF in qs and qh)
    {
        Q5_0Block block;
        std::memset(block.qs, 0xFF, 16);
        std::memset(block.qh, 0xFF, 4);
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx2_output[32];

        unpack_q5_0_to_int8_scalar(block, scalar_output);
        unpack_q5_0_to_int8_avx2(block, avx2_output);

        verify_arrays_equal(scalar_output, avx2_output, 32, "AVX2 vs Scalar (all 0xFF)");
    }
}
#endif

// ============================================================================
// AVX-512 vs Scalar Parity
// ============================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q5_0_SIMD_Unpack, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    for (int trial = 0; trial < 100; ++trial)
    {
        Q5_0Block block = create_random_block();
        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx512_output[32];

        unpack_q5_0_to_int8_scalar(block, scalar_output);
        unpack_q5_0_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32,
                            "AVX-512 vs Scalar (random trial " + std::to_string(trial) + ")");
    }
}

TEST_F(Q5_0_SIMD_Unpack, AVX512_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx512())
    {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }

    // Edge case 1: All zeros
    {
        Q5_0Block block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx512_output[32];

        unpack_q5_0_to_int8_scalar(block, scalar_output);
        unpack_q5_0_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32, "AVX-512 vs Scalar (all zeros)");
    }

    // Edge case 2: All ones (0xFF in qs and qh)
    {
        Q5_0Block block;
        std::memset(block.qs, 0xFF, 16);
        std::memset(block.qh, 0xFF, 4);
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_output[32];
        alignas(64) int8_t avx512_output[32];

        unpack_q5_0_to_int8_scalar(block, scalar_output);
        unpack_q5_0_to_int8_avx512(block, avx512_output);

        verify_arrays_equal(scalar_output, avx512_output, 32, "AVX-512 vs Scalar (all 0xFF)");
    }
}
#endif
