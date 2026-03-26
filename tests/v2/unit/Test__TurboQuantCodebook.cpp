/**
 * @file Test__TurboQuantCodebook.cpp
 * @brief Unit tests for TurboQuant Lloyd-Max codebook centroids
 *
 * Validates that the pre-computed codebook tables match Lloyd-Max
 * optimal quantization for N(0,1). Tests:
 * - Centroid values match re-derived Lloyd-Max iteration
 * - Thresholds are midpoints of adjacent centroids
 * - Symmetry: centroid[k] = -centroid[K-1-k]
 * - Nearest centroid returns correct indices
 * - 3-bit packing/unpacking round-trips correctly (used by TQ4)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "kernels/cpu/turboquant/TurboQuantCodebook.h"
#include "tensors/BlockStructures.h"
#include <cmath>
#include <random>
#include <vector>

using namespace llaminar2;

// ============================================================================
// 4-bit codebook verification
// ============================================================================

TEST(Test__TurboQuantCodebook, TQ4_CentroidsAreFixedPoint)
{
    // One Lloyd-Max step from stored centroids should barely move them.
    // Movement is bounded by float32→double promotion error (~1e-4).
    double movement = verify_codebook<16>(TQ4_CENTROIDS);
    EXPECT_LT(movement, 1e-4)
        << "TQ4 centroids are not a Lloyd-Max fixed point (moved by " << movement << ")";
}

TEST(Test__TurboQuantCodebook, TQ4_CentroidsAreSorted)
{
    for (int i = 0; i < 15; ++i)
    {
        EXPECT_LT(TQ4_CENTROIDS[i], TQ4_CENTROIDS[i + 1])
            << "TQ4 centroids not sorted at index " << i;
    }
}

TEST(Test__TurboQuantCodebook, TQ4_CentroidsAreSymmetric)
{
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_NEAR(TQ4_CENTROIDS[i], -TQ4_CENTROIDS[15 - i], 1e-5f)
            << "TQ4 symmetry violated at index " << i;
    }
}

TEST(Test__TurboQuantCodebook, TQ4_ThresholdsAreMidpoints)
{
    for (int i = 0; i < 15; ++i)
    {
        float expected = 0.5f * (TQ4_CENTROIDS[i] + TQ4_CENTROIDS[i + 1]);
        EXPECT_NEAR(TQ4_THRESHOLDS[i], expected, 1e-4f)
            << "TQ4 threshold[" << i << "] is not midpoint of centroids";
    }
}

TEST(Test__TurboQuantCodebook, TQ4_ThresholdsAreSorted)
{
    for (int i = 0; i < 14; ++i)
    {
        EXPECT_LT(TQ4_THRESHOLDS[i], TQ4_THRESHOLDS[i + 1])
            << "TQ4 thresholds not sorted at index " << i;
    }
}

// ============================================================================
// 3-bit codebook verification
// ============================================================================

TEST(Test__TurboQuantCodebook, TQ3_CentroidsAreFixedPoint)
{
    double movement = verify_codebook<8>(TQ3_CENTROIDS);
    EXPECT_LT(movement, 1e-4)
        << "TQ3 centroids are not a Lloyd-Max fixed point (moved by " << movement << ")";
}

TEST(Test__TurboQuantCodebook, TQ3_CentroidsAreSorted)
{
    for (int i = 0; i < 7; ++i)
    {
        EXPECT_LT(TQ3_CENTROIDS[i], TQ3_CENTROIDS[i + 1])
            << "TQ3 centroids not sorted at index " << i;
    }
}

TEST(Test__TurboQuantCodebook, TQ3_CentroidsAreSymmetric)
{
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(TQ3_CENTROIDS[i], -TQ3_CENTROIDS[7 - i], 1e-5f)
            << "TQ3 symmetry violated at index " << i;
    }
}

TEST(Test__TurboQuantCodebook, TQ3_ThresholdsAreMidpoints)
{
    for (int i = 0; i < 7; ++i)
    {
        float expected = 0.5f * (TQ3_CENTROIDS[i] + TQ3_CENTROIDS[i + 1]);
        EXPECT_NEAR(TQ3_THRESHOLDS[i], expected, 1e-4f)
            << "TQ3 threshold[" << i << "] is not midpoint of centroids";
    }
}

// ============================================================================
// Nearest centroid lookup
// ============================================================================

TEST(Test__TurboQuantCodebook, TQ4_NearestCentroid_ExactValues)
{
    // Each centroid should map to itself
    for (int k = 0; k < 16; ++k)
    {
        uint8_t idx = tq4_nearest_centroid(TQ4_CENTROIDS[k]);
        EXPECT_EQ(idx, k)
            << "TQ4 centroid[" << k << "] = " << TQ4_CENTROIDS[k]
            << " maps to index " << static_cast<int>(idx);
    }
}

TEST(Test__TurboQuantCodebook, TQ4_NearestCentroid_Extremes)
{
    // Very negative → index 0
    EXPECT_EQ(tq4_nearest_centroid(-10.0f), 0);
    // Very positive → index 15
    EXPECT_EQ(tq4_nearest_centroid(10.0f), 15);
    // Zero → one of the middle indices (7 or 8)
    uint8_t idx_zero = tq4_nearest_centroid(0.0f);
    EXPECT_TRUE(idx_zero == 7 || idx_zero == 8)
        << "Zero maps to index " << static_cast<int>(idx_zero);
}

TEST(Test__TurboQuantCodebook, TQ3_NearestCentroid_ExactValues)
{
    for (int k = 0; k < 8; ++k)
    {
        uint8_t idx = tq3_nearest_centroid(TQ3_CENTROIDS[k]);
        EXPECT_EQ(idx, k)
            << "TQ3 centroid[" << k << "] = " << TQ3_CENTROIDS[k]
            << " maps to index " << static_cast<int>(idx);
    }
}

TEST(Test__TurboQuantCodebook, TQ3_NearestCentroid_Extremes)
{
    EXPECT_EQ(tq3_nearest_centroid(-10.0f), 0);
    EXPECT_EQ(tq3_nearest_centroid(10.0f), 7);
    uint8_t idx_zero = tq3_nearest_centroid(0.0f);
    EXPECT_TRUE(idx_zero == 3 || idx_zero == 4)
        << "Zero maps to index " << static_cast<int>(idx_zero);
}

// ============================================================================
// TQ3 bit packing / unpacking
// ============================================================================

TEST(Test__TurboQuantCodebook, TQ3_PackUnpack_AllValues)
{
    // Test all 8^8 possible 8-element groups? No — test representative patterns.
    // Test all-same values
    for (uint8_t val = 0; val < 8; ++val)
    {
        uint8_t input[8] = {val, val, val, val, val, val, val, val};
        uint8_t packed[3];
        uint8_t unpacked[8];
        tq3_pack_8(input, packed);
        tq3_unpack_8(packed, unpacked);
        for (int i = 0; i < 8; ++i)
            EXPECT_EQ(unpacked[i], val) << "i=" << i << " val=" << static_cast<int>(val);
    }
}

TEST(Test__TurboQuantCodebook, TQ3_PackUnpack_Sequential)
{
    uint8_t input[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t packed[3];
    uint8_t unpacked[8];
    tq3_pack_8(input, packed);
    tq3_unpack_8(packed, unpacked);
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(unpacked[i], input[i]) << "Sequential pack/unpack failed at i=" << i;
}

TEST(Test__TurboQuantCodebook, TQ3_PackUnpack_Random)
{
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 7);

    for (int trial = 0; trial < 1000; ++trial)
    {
        uint8_t input[8], packed[3], unpacked[8];
        for (int i = 0; i < 8; ++i)
            input[i] = static_cast<uint8_t>(dist(rng));

        tq3_pack_8(input, packed);
        tq3_unpack_8(packed, unpacked);

        for (int i = 0; i < 8; ++i)
            EXPECT_EQ(unpacked[i], input[i])
                << "Random pack/unpack trial " << trial << " failed at i=" << i;
    }
}

// ============================================================================
// Theoretical MSE bounds
// ============================================================================

TEST(Test__TurboQuantCodebook, TQ4_TheoreticalMSE)
{
    // For N(0,1) with 16 levels, Lloyd-Max MSE ≈ 0.009497
    double mse = lloyd_max_mse(16, 1.0);
    EXPECT_GT(mse, 0.0) << "MSE should be positive";
    EXPECT_LT(mse, 0.015) << "4-bit MSE unusually high: " << mse;
    // Literature value is ~0.009497 for N(0,1) 16-level Lloyd-Max
    EXPECT_NEAR(mse, 0.009497, 0.002)
        << "4-bit MSE deviates from literature value";
}
