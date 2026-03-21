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

class Test__Q3_K_SIMD_Unpack : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }

    // Helper to pack scales
    void pack_scales(Q3_KBlock &block, const uint8_t *unpacked_scales)
    {
        // Reverse of unpack_q3k_scales
        // This is tricky. Let's just set them manually if possible or implement pack logic.
        // Or just use the fact that we know how they map.
        // scales[0] -> aux[0] bits 0-3 and aux[2] bits 0-1?
        // Let's look at unpack logic:
        // aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        // aux[0] low 4 bits come from aux[0] low 4 bits.
        // aux[0] high 4 bits come from tmp bits 0-1 (shifted to 4-5).
        // So unpacked[0] = (packed[0] & 0xF) | ((packed[8] & 3) << 4).
        // unpacked[1] = (packed[1] & 0xF) | ((packed[8] >> 2 & 3) << 4).
        // ...
        // unpacked[8] = (packed[0] >> 4) | ((packed[9] & 3) << 4).
        // Wait, this is getting complicated.
        // Let's just set the packed bytes directly to achieve desired unpacked values.
        // If we want unpacked[i] = 33 (0x21 = 100001 binary).
        // Low 4 bits = 1. High 2 bits = 2.

        // Let's implement a simple packer for testing.
        uint8_t aux[16];
        std::memcpy(aux, unpacked_scales, 16);

        // We need to pack 16 6-bit values into 12 bytes.
        // The unpacking logic was:
        // uint32_t aux[4]; memcpy(aux, packed, 12); ...
        // It's easier to just reverse the bit operations if we understand them.
        // But for testing, maybe we can just set all scales to the same value?
        // If we set all bytes to 0, all scales are 0.
        // If we set all bytes to 0xFF, all scales are 63.
        // If we want specific values, we need to be careful.

        // Let's try to set specific scales.
        // unpacked[0] comes from packed[0] (low 4) and packed[8] (low 2).
        // unpacked[1] comes from packed[1] (low 4) and packed[8] (bits 2-3).
        // ...
        // unpacked[7] comes from packed[7] (low 4) and packed[9] (bits 6-7).
        // unpacked[8] comes from packed[0] (high 4) and packed[10] (low 2).
        // ...

        // Actually, let's just use a known pattern or trial and error in the test?
        // No, let's write a proper packer.

        // Mapping based on unpack_q3k_scales:
        // unpacked[0] = (packed[0] & 0xF) | ((packed[8] & 0x3) << 4)
        // unpacked[1] = (packed[1] & 0xF) | ((packed[8] >> 2 & 0x3) << 4)
        // unpacked[2] = (packed[2] & 0xF) | ((packed[8] >> 4 & 0x3) << 4)
        // unpacked[3] = (packed[3] & 0xF) | ((packed[8] >> 6 & 0x3) << 4)
        // unpacked[4] = (packed[4] & 0xF) | ((packed[9] & 0x3) << 4)
        // ...
        // unpacked[7] = (packed[7] & 0xF) | ((packed[9] >> 6 & 0x3) << 4)
        // unpacked[8] = (packed[0] >> 4) | ((packed[10] & 0x3) << 4)
        // ...
        // unpacked[15] = (packed[7] >> 4) | ((packed[11] >> 6 & 0x3) << 4)

        // So:
        // packed[0] = (unpacked[0] & 0xF) | ((unpacked[8] & 0xF) << 4)
        // ...
        // packed[7] = (unpacked[7] & 0xF) | ((unpacked[15] & 0xF) << 4)
        // packed[8] = (unpacked[0] >> 4) | ((unpacked[1] >> 4) << 2) | ((unpacked[2] >> 4) << 4) | ((unpacked[3] >> 4) << 6)
        // packed[9] = (unpacked[4] >> 4) | ...
        // packed[10] = (unpacked[8] >> 4) | ...
        // packed[11] = (unpacked[12] >> 4) | ...

        for (int i = 0; i < 8; ++i)
        {
            block.scales[i] = (unpacked_scales[i] & 0xF) | ((unpacked_scales[i + 8] & 0xF) << 4);
        }
        for (int i = 0; i < 4; ++i)
        {
            block.scales[8] |= ((unpacked_scales[i] >> 4) & 0x3) << (i * 2);
            block.scales[9] |= ((unpacked_scales[i + 4] >> 4) & 0x3) << (i * 2);
            block.scales[10] |= ((unpacked_scales[i + 8] >> 4) & 0x3) << (i * 2);
            block.scales[11] |= ((unpacked_scales[i + 12] >> 4) & 0x3) << (i * 2);
        }
    }
};

TEST_F(Test__Q3_K_SIMD_Unpack, DecodeSubBlock_Simple)
{
    Q3_KBlock block;
    std::memset(&block, 0, sizeof(block));

    block.d = fp32_to_fp16(1.0f);

    // Set scales to 33 (dl = 1.0 * (33-32) = 1.0)
    uint8_t scales[16];
    for (int i = 0; i < 16; ++i)
        scales[i] = 33;
    pack_scales(block, scales);

    // Sub-block 0 (chunk 0, group 0)
    // Element 0: q=3, h=1 -> val=3
    block.qs[0] = 0x03;    // bits 0-1
    block.hmask[0] = 0x01; // bit 0

    // Element 1: q=0, h=0 -> val=-4
    // qs[1] bits 0-1 are 0. hmask[1] bit 0 is 0.

    float output[32];
    simd::decode_q3k_subblock_to_float(block, 0, output);

    EXPECT_FLOAT_EQ(output[0], 3.0f);
    EXPECT_FLOAT_EQ(output[1], -4.0f);
}

TEST_F(Test__Q3_K_SIMD_Unpack, DecodeSubBlock_Group1)
{
    Q3_KBlock block;
    std::memset(&block, 0, sizeof(block));

    block.d = fp32_to_fp16(1.0f);
    uint8_t scales[16];
    for (int i = 0; i < 16; ++i)
        scales[i] = 33;
    pack_scales(block, scales);

    // Sub-block 1 (chunk 0, group 1)
    // Uses bits 2-3 of qs, bit 1 of hmask.

    // Element 0: q=2 (10 binary), h=1 -> val=2
    // qs[0] |= (2 << 2) = 8
    block.qs[0] = 0x08;
    // hmask[0] |= (1 << 1) = 2
    block.hmask[0] = 0x02;

    // Element 1: q=1 (01 binary), h=0 -> val=1-4=-3
    // qs[1] |= (1 << 2) = 4
    block.qs[1] = 0x04;
    // hmask[1] bit 1 is 0.

    float output[32];
    simd::decode_q3k_subblock_to_float(block, 1, output);

    EXPECT_FLOAT_EQ(output[0], 2.0f);
    EXPECT_FLOAT_EQ(output[1], -3.0f);
}

TEST_F(Test__Q3_K_SIMD_Unpack, DecodeSubBlock_Scales)
{
    Q3_KBlock block;
    std::memset(&block, 0, sizeof(block));

    block.d = fp32_to_fp16(2.0f);

    // Set scales for sub-block 0 (indices 0 and 1)
    // scale[0] = 34 (dl = 2.0 * (34-32) = 4.0)
    // scale[1] = 35 (dl = 2.0 * (35-32) = 6.0)
    uint8_t scales[16];
    std::memset(scales, 32, 16); // Default 0
    scales[0] = 34;
    scales[1] = 35;
    pack_scales(block, scales);

    // Element 0 (first half): q=3, h=1 -> val=3 -> out=4.0*3=12.0
    block.qs[0] = 0x03;
    block.hmask[0] = 0x01;

    // Element 16 (second half): q=3, h=1 -> val=3 -> out=6.0*3=18.0
    block.qs[16] = 0x03;
    block.hmask[16] = 0x01;

    float output[32];
    simd::decode_q3k_subblock_to_float(block, 0, output);

    EXPECT_FLOAT_EQ(output[0], 12.0f);
    EXPECT_FLOAT_EQ(output[16], 18.0f);
}

// ============================================================================
// Superblock SIMD Parity Tests
// ============================================================================

class Q3_K_Superblock_SIMD_Parity : public ::testing::Test
{
protected:
    std::mt19937 gen_{42};

    Q3_KBlock create_random_block()
    {
        Q3_KBlock block;
        std::uniform_int_distribution<uint8_t> dist(0, 255);
        for (int i = 0; i < 32; ++i) block.hmask[i] = dist(gen_);
        for (int i = 0; i < 64; ++i) block.qs[i] = dist(gen_);
        for (int i = 0; i < 12; ++i) block.scales[i] = dist(gen_);
        block.d = fp32_to_fp16(0.5f + 0.5f * (dist(gen_) / 255.0f));
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

TEST_F(Q3_K_Superblock_SIMD_Parity, ScalarReference_Random)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        Q3_KBlock block = create_random_block();
        alignas(64) int8_t output[256];
        alignas(64) float scales[8], mins[8];

        simd::unpack_q3_k_superblock_to_int8_scalar(block, output, scales, mins);

        for (int i = 0; i < 256; ++i)
        {
            EXPECT_GE(output[i], -128) << "Value below range at " << i << " trial " << trial;
            EXPECT_LE(output[i], 127) << "Value above range at " << i << " trial " << trial;
        }
    }
}

#if defined(__AVX2__)
TEST_F(Q3_K_Superblock_SIMD_Parity, AVX2_vs_Scalar_Random)
{
    if (!cpu_supports_avx2()) GTEST_SKIP() << "AVX2 not supported";

    for (int trial = 0; trial < 100; ++trial)
    {
        Q3_KBlock block = create_random_block();
        alignas(64) int8_t scalar_out[256], avx2_out[256];
        alignas(64) float scalar_scales[8], scalar_mins[8];
        alignas(64) float avx2_scales[8], avx2_mins[8];

        simd::unpack_q3_k_superblock_to_int8_scalar(block, scalar_out, scalar_scales, scalar_mins);
        simd::unpack_q3_k_superblock_to_int8_avx2(block, avx2_out, avx2_scales, avx2_mins);

        verify_arrays_near(scalar_out, avx2_out, 256, 1, "AVX2 vs Scalar trial " + std::to_string(trial));

        for (int i = 0; i < 8; ++i)
        {
            EXPECT_NEAR(scalar_scales[i], avx2_scales[i], 1e-3f) << "Scale mismatch at " << i;
            EXPECT_NEAR(scalar_mins[i], avx2_mins[i], 1e-3f) << "Min mismatch at " << i;
        }
    }
}

TEST_F(Q3_K_Superblock_SIMD_Parity, AVX2_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx2()) GTEST_SKIP() << "AVX2 not supported";

    // All zeros
    {
        Q3_KBlock block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_out[256], avx2_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q3_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q3_k_superblock_to_int8_avx2(block, avx2_out, as, am);

        verify_arrays_near(scalar_out, avx2_out, 256, 1, "AVX2 vs Scalar (zeros)");
    }

    // All 0xFF
    {
        Q3_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_out[256], avx2_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q3_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q3_k_superblock_to_int8_avx2(block, avx2_out, as, am);

        verify_arrays_near(scalar_out, avx2_out, 256, 1, "AVX2 vs Scalar (0xFF)");
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q3_K_Superblock_SIMD_Parity, AVX512_vs_Scalar_Random)
{
    if (!cpu_supports_avx512()) GTEST_SKIP() << "AVX-512 not supported";

    for (int trial = 0; trial < 100; ++trial)
    {
        Q3_KBlock block = create_random_block();
        alignas(64) int8_t scalar_out[256], avx512_out[256];
        alignas(64) float scalar_scales[8], scalar_mins[8];
        alignas(64) float avx512_scales[8], avx512_mins[8];

        simd::unpack_q3_k_superblock_to_int8_scalar(block, scalar_out, scalar_scales, scalar_mins);
        simd::unpack_q3_k_superblock_to_int8_avx512(block, avx512_out, avx512_scales, avx512_mins);

        verify_arrays_near(scalar_out, avx512_out, 256, 1, "AVX512 vs Scalar trial " + std::to_string(trial));

        for (int i = 0; i < 8; ++i)
        {
            EXPECT_NEAR(scalar_scales[i], avx512_scales[i], 1e-3f) << "Scale mismatch at " << i;
            EXPECT_NEAR(scalar_mins[i], avx512_mins[i], 1e-3f) << "Min mismatch at " << i;
        }
    }
}

TEST_F(Q3_K_Superblock_SIMD_Parity, AVX512_vs_Scalar_EdgeCases)
{
    if (!cpu_supports_avx512()) GTEST_SKIP() << "AVX-512 not supported";

    // All zeros
    {
        Q3_KBlock block;
        std::memset(&block, 0, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_out[256], avx512_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q3_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q3_k_superblock_to_int8_avx512(block, avx512_out, as, am);

        verify_arrays_near(scalar_out, avx512_out, 256, 1, "AVX512 vs Scalar (zeros)");
    }

    // All 0xFF
    {
        Q3_KBlock block;
        std::memset(&block, 0xFF, sizeof(block));
        block.d = fp32_to_fp16(1.0f);

        alignas(64) int8_t scalar_out[256], avx512_out[256];
        alignas(64) float ss[8], sm[8], as[8], am[8];

        simd::unpack_q3_k_superblock_to_int8_scalar(block, scalar_out, ss, sm);
        simd::unpack_q3_k_superblock_to_int8_avx512(block, avx512_out, as, am);

        verify_arrays_near(scalar_out, avx512_out, 256, 1, "AVX512 vs Scalar (0xFF)");
    }
}
#endif
