#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/FP16Utils.h"
#include "v2/utils/CPUFeatures.h"

using namespace llaminar2;

class Test__Q2_K_SIMD_Unpack : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(Test__Q2_K_SIMD_Unpack, DecodeSubBlock_Simple)
{
    Q2_KBlock block;
    std::memset(&block, 0, sizeof(block));

    // Set super-block scales
    block.d = fp32_to_fp16(1.0f);
    block.dmin = fp32_to_fp16(0.0f);

    // We test sub-block 0 (first 32 elements)
    // This uses chunk 0, j=0.
    // Scales index 0 and 1.
    // q bytes 0..31.
    // shift = 0.

    // Set scales for sub-block 0
    // sc = (ml_idx << 4) | dl_idx
    // We want dl=1.0, ml=0.0. So sc = 0x01.
    // dl = d * (sc & 0xF) = 1.0 * 1 = 1.0
    // ml = dmin * ... = 0.0
    block.scales[0] = 0x01; // First 16 elements
    block.scales[1] = 0x01; // Second 16 elements

    // Set q values
    // q[l] has 4 values packed (2 bits each).
    // For j=0, shift=0. We use bits 0-1.
    // Let's set q[0] = 0x03 (value 3).
    // q[1] = 0x02 (value 2).
    // ...
    for (int i = 0; i < 32; ++i)
    {
        block.qs[i] = (i % 4); // Set bits 0-1 to 0, 1, 2, 3 repeating
    }

    float output[32];
    simd::decode_q2k_subblock_to_float(block, 0, output);

    for (int i = 0; i < 32; ++i)
    {
        float expected = (float)(i % 4);
        EXPECT_FLOAT_EQ(output[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(Test__Q2_K_SIMD_Unpack, DecodeSubBlock_ComplexScales)
{
    Q2_KBlock block;
    std::memset(&block, 0, sizeof(block));

    block.d = fp32_to_fp16(2.0f);
    block.dmin = fp32_to_fp16(1.0f);

    // Sub-block 1 (j=1). shift=2.
    // Scales index 2 and 3.
    // q bytes 0..31 (reused!).

    // Set scales
    // sc[2] = 0x12 (dl_idx=2, ml_idx=1)
    // dl = 2.0 * 2 = 4.0
    // ml = 1.0 * 1 = 1.0
    block.scales[2] = 0x12;

    // sc[3] = 0x23 (dl_idx=3, ml_idx=2)
    // dl = 2.0 * 3 = 6.0
    // ml = 1.0 * 2 = 2.0
    block.scales[3] = 0x23;

    // Set q values
    // We need to set bits 2-3.
    // q[i] |= (val << 2)
    for (int i = 0; i < 32; ++i)
    {
        int val = (i % 4);
        block.qs[i] |= (val << 2);
    }

    float output[32];
    simd::decode_q2k_subblock_to_float(block, 1, output);

    // First 16 elements
    for (int i = 0; i < 16; ++i)
    {
        int q = i % 4;
        float expected = 4.0f * q - 1.0f;
        EXPECT_FLOAT_EQ(output[i], expected) << "Mismatch at index " << i;
    }

    // Second 16 elements
    for (int i = 0; i < 16; ++i)
    {
        int q = (i + 16) % 4;
        float expected = 6.0f * q - 2.0f;
        EXPECT_FLOAT_EQ(output[i + 16], expected) << "Mismatch at index " << (i + 16);
    }
}

TEST_F(Test__Q2_K_SIMD_Unpack, RequantizeAffine)
{
    float input[32];
    // Create a range of values
    for (int i = 0; i < 32; ++i)
    {
        input[i] = -10.0f + i * 1.0f; // -10.0 to 21.0
    }

    int8_t output[32];
    float scale, min_val;
    simd::requantize_to_int8_affine(input, 32, output, &scale, &min_val);

    // Verify reconstruction
    // val ~= scale * q + min_val
    // Note: q is int8 (-128 to 127).
    // Our implementation maps min_val to -128.
    // So reconstructed = scale * (q + 128) + min_val_original?
    // Wait, let's check implementation:
    // *out_min = scale * 128.0f + min_val;
    // reconstructed = scale * q + *out_min
    // = scale * q + scale * 128 + min_val
    // = scale * (q + 128) + min_val
    // Correct.

    for (int i = 0; i < 32; ++i)
    {
        float reconstructed = scale * output[i] + min_val;
        EXPECT_NEAR(reconstructed, input[i], scale / 2.0f + 1e-5f) << "Mismatch at index " << i;
    }
}

// ============================================================================
// Superblock SIMD Parity Tests
// ============================================================================

class Q2_K_Superblock_SIMD_Parity : public ::testing::Test
{
protected:
    std::mt19937 gen_{42};

    Q2_KBlock create_random_block()
    {
        Q2_KBlock block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (int i = 0; i < 16; ++i) block.scales[i] = dist(gen_);
        for (int i = 0; i < 64; ++i) block.qs[i] = dist(gen_);
        block.d = fp32_to_fp16(0.5f + 0.5f * (dist(gen_) / 255.0f));
        block.dmin = fp32_to_fp16(0.1f * (dist(gen_) / 255.0f));
        return block;
    }

    void verify_arrays_near(const int8_t *expected, const int8_t *actual, size_t count, int tolerance, const std::string &context)
    {
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_NEAR(static_cast<int>(expected[i]), static_cast<int>(actual[i]), tolerance)
                << context << ": Mismatch at index " << i
                << " (expected " << static_cast<int>(expected[i])
                << ", got " << static_cast<int>(actual[i]) << ")";
        }
    }
};

TEST_F(Q2_K_Superblock_SIMD_Parity, ScalarReference_Random)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        Q2_KBlock block = create_random_block();
        alignas(64) int8_t output[256];
        alignas(64) float scales[8], mins[8];

        simd::unpack_q2_k_superblock_to_int8_scalar(block, output, scales, mins);

        for (int i = 0; i < 256; ++i)
        {
            EXPECT_GE(output[i], -128) << "Value below range at " << i << " trial " << trial;
            EXPECT_LE(output[i], 127) << "Value above range at " << i << " trial " << trial;
        }
    }
}

#if defined(__AVX2__)
TEST_F(Q2_K_Superblock_SIMD_Parity, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2()) GTEST_SKIP() << "AVX2 not supported";

    for (int trial = 0; trial < 100; ++trial)
    {
        Q2_KBlock block = create_random_block();
        alignas(64) int8_t scalar_out[256], avx2_out[256];
        alignas(64) float scalar_scales[8], scalar_mins[8];
        alignas(64) float avx2_scales[8], avx2_mins[8];

        simd::unpack_q2_k_superblock_to_int8_scalar(block, scalar_out, scalar_scales, scalar_mins);
        simd::unpack_q2_k_superblock_to_int8_avx2(block, avx2_out, avx2_scales, avx2_mins);

        verify_arrays_near(scalar_out, avx2_out, 256, 1, "AVX2 vs Scalar trial " + std::to_string(trial));

        for (int i = 0; i < 8; ++i)
        {
            EXPECT_NEAR(scalar_scales[i], avx2_scales[i], 1e-3f) << "Scale mismatch at " << i;
            EXPECT_NEAR(scalar_mins[i], avx2_mins[i], 1e-3f) << "Min mismatch at " << i;
        }
    }
}

TEST_F(Q2_K_Superblock_SIMD_Parity, AVX2_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx2()) GTEST_SKIP() << "AVX2 not supported";

    {
        Q2_KBlock block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.dmin = fp32_to_fp16(0.0f);

        alignas(64) int8_t scalar_out[256], avx2_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q2_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q2_k_superblock_to_int8_avx2(block, avx2_out, as, am);

        verify_arrays_near(scalar_out, avx2_out, 256, 1, "AVX2 vs Scalar (zeros)");
    }

    {
        Q2_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.dmin = fp32_to_fp16(0.5f);

        alignas(64) int8_t scalar_out[256], avx2_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q2_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q2_k_superblock_to_int8_avx2(block, avx2_out, as, am);

        verify_arrays_near(scalar_out, avx2_out, 256, 1, "AVX2 vs Scalar (0xFF)");
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q2_K_Superblock_SIMD_Parity, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512()) GTEST_SKIP() << "AVX-512 not supported";

    for (int trial = 0; trial < 100; ++trial)
    {
        Q2_KBlock block = create_random_block();
        alignas(64) int8_t scalar_out[256], avx512_out[256];
        alignas(64) float scalar_scales[8], scalar_mins[8];
        alignas(64) float avx512_scales[8], avx512_mins[8];

        simd::unpack_q2_k_superblock_to_int8_scalar(block, scalar_out, scalar_scales, scalar_mins);
        simd::unpack_q2_k_superblock_to_int8_avx512(block, avx512_out, avx512_scales, avx512_mins);

        verify_arrays_near(scalar_out, avx512_out, 256, 1, "AVX512 vs Scalar trial " + std::to_string(trial));

        for (int i = 0; i < 8; ++i)
        {
            EXPECT_NEAR(scalar_scales[i], avx512_scales[i], 1e-3f) << "Scale mismatch at " << i;
            EXPECT_NEAR(scalar_mins[i], avx512_mins[i], 1e-3f) << "Min mismatch at " << i;
        }
    }
}

TEST_F(Q2_K_Superblock_SIMD_Parity, AVX512_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx512()) GTEST_SKIP() << "AVX-512 not supported";

    {
        Q2_KBlock block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.dmin = fp32_to_fp16(0.0f);

        alignas(64) int8_t scalar_out[256], avx512_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q2_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q2_k_superblock_to_int8_avx512(block, avx512_out, as, am);

        verify_arrays_near(scalar_out, avx512_out, 256, 1, "AVX512 vs Scalar (zeros)");
    }

    {
        Q2_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);
        block.dmin = fp32_to_fp16(0.5f);

        alignas(64) int8_t scalar_out[256], avx512_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q2_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q2_k_superblock_to_int8_avx512(block, avx512_out, as, am);

        verify_arrays_near(scalar_out, avx512_out, 256, 1, "AVX512 vs Scalar (0xFF)");
    }
}
#endif
