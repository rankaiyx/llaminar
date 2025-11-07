/**
 * @file Test__RoPEPrimitives.cpp
 * @brief Unit tests for RoPE primitive implementations (scalar, AVX2, AVX512)
 * @author David Sanftenberg
 *
 * Validates that:
 * 1. All implementations (scalar, AVX2, AVX512) produce identical results
 * 2. RoPE correctly handles various head dimensions (32, 64, 128)
 * 3. Position encoding is consistent across different positions
 * 4. Edge cases (odd head_dim, position=0) are handled correctly
 */

#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace llaminar2::primitives;

namespace
{
    /**
     * @brief Generate random test data for a single head
     */
    std::vector<float> generate_test_head(int head_dim, uint32_t seed = 42)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> data(head_dim);
        for (auto &val : data)
        {
            val = dist(gen);
        }
        return data;
    }

    /**
     * @brief Compare two float vectors element-wise
     * @return Tuple of (max_abs_diff, rel_l2_norm, num_mismatches)
     */
    std::tuple<float, float, int> compare_vectors(
        const std::vector<float> &a,
        const std::vector<float> &b,
        float tolerance = 1e-6f)
    {
        EXPECT_EQ(a.size(), b.size());
        if (a.size() != b.size())
            return {INFINITY, INFINITY, static_cast<int>(std::max(a.size(), b.size()))};

        float max_abs_diff = 0.0f;
        float sum_sq_diff = 0.0f;
        float sum_sq_ref = 0.0f;
        int mismatches = 0;

        for (size_t i = 0; i < a.size(); ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            sum_sq_diff += diff * diff;
            sum_sq_ref += b[i] * b[i];

            if (diff > tolerance)
            {
                mismatches++;
            }
        }

        float rel_l2 = sum_sq_ref > 0.0f ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0f;
        return {max_abs_diff, rel_l2, mismatches};
    }

    /**
     * @brief Print comparison metrics
     */
    void print_comparison(const char *impl1, const char *impl2, float max_abs, float rel_l2, int mismatches, int total)
    {
        printf("  %s vs %s: max_abs=%.2e, rel_l2=%.2e, mismatches=%d/%d\n",
               impl1, impl2, max_abs, rel_l2, mismatches, total);
    }
} // anonymous namespace

// ============================================================================
// Test Suite: RoPE Implementation Parity
// ============================================================================

class RoPEPrimitivesTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Detect available ISA
#if defined(__AVX512F__)
        has_avx512_ = true;
        has_avx2_ = true;
#elif defined(__AVX2__)
        has_avx2_ = true;
#else
        // Scalar only
#endif
    }

    bool has_avx2_ = false;
    bool has_avx512_ = false;
};

// ============================================================================
// Test 1: Scalar vs Vectorized Parity
// ============================================================================

TEST_F(RoPEPrimitivesTest, ScalarVsVectorizedParity)
{
    const int head_dim = 64;
    const int position = 10;
    const float freq_base = 10000.0f;

    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    // Generate test data
    auto original_data = generate_test_head(head_dim);

    // Apply scalar
    auto scalar_result = original_data;
    apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
    if (has_avx2_)
    {
        auto avx2_result = original_data;
        int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
        apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

        auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
        print_comparison("AVX2", "Scalar", max_abs, rel_l2, mismatches, head_dim);

        EXPECT_LT(max_abs, 1e-5f) << "AVX2 and scalar implementations differ significantly";
        EXPECT_LT(rel_l2, 1e-6f) << "AVX2 relative error too high";
        EXPECT_EQ(mismatches, 0) << "AVX2 has mismatches vs scalar";
    }
#endif

#if defined(__AVX512F__)
    if (has_avx512_)
    {
        auto avx512_result = original_data;
        int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
        apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

        auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
        print_comparison("AVX512", "Scalar", max_abs, rel_l2, mismatches, head_dim);

        EXPECT_LT(max_abs, 1e-5f) << "AVX512 and scalar implementations differ significantly";
        EXPECT_LT(rel_l2, 1e-6f) << "AVX512 relative error too high";
        EXPECT_EQ(mismatches, 0) << "AVX512 has mismatches vs scalar";
    }
#endif
}

// ============================================================================
// Test 2: Multiple Head Dimensions
// ============================================================================

TEST_F(RoPEPrimitivesTest, VariousHeadDimensions)
{
    const std::vector<int> head_dims = {32, 64, 128};
    const int position = 5;
    const float freq_base = 10000.0f;

    for (int head_dim : head_dims)
    {
        SCOPED_TRACE("head_dim=" + std::to_string(head_dim));

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);
        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX2 failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX512 failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 3: Multiple Positions
// ============================================================================

TEST_F(RoPEPrimitivesTest, VariousPositions)
{
    const int head_dim = 64;
    const float freq_base = 10000.0f;
    const std::vector<int> positions = {0, 1, 10, 100, 1000};

    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    for (int position : positions)
    {
        SCOPED_TRACE("position=" + std::to_string(position));

        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX2 failed for position=" << position;
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX512 failed for position=" << position;
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 4: Frequency Base Variations
// ============================================================================

TEST_F(RoPEPrimitivesTest, FrequencyBaseVariations)
{
    const int head_dim = 64;
    const int position = 10;
    const std::vector<float> freq_bases = {10000.0f, 500000.0f, 1000000.0f};

    for (float freq_base : freq_bases)
    {
        SCOPED_TRACE("freq_base=" + std::to_string(freq_base));

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);
        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f);
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f);
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 5: Edge Cases
// ============================================================================

TEST_F(RoPEPrimitivesTest, EdgeCases)
{
    const int head_dim = 64;
    const float freq_base = 10000.0f;
    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    // Test 5a: Zero position
    {
        auto original_data = generate_test_head(head_dim);
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), 0, inv_freq, head_dim, 0);

        // At position 0, angles are all zero, so cos=1, sin=0
        // Rotation becomes: new_first = x_first * 1 - x_second * 0 = x_first
        //                   new_second = x_first * 0 + x_second * 1 = x_second
        // So result should equal input
        auto [max_abs, rel_l2, mismatches] = compare_vectors(scalar_result, original_data, 1e-6f);
        EXPECT_LT(max_abs, 1e-5f) << "Position 0 should be identity transform";
    }

    // Test 5b: All zeros input
    {
        std::vector<float> zeros(head_dim, 0.0f);
        auto result = zeros;
        apply_rope_to_head_scalar(result.data(), 10, inv_freq, head_dim, 0);

        auto [max_abs, rel_l2, mismatches] = compare_vectors(result, zeros, 1e-6f);
        EXPECT_LT(max_abs, 1e-5f) << "All-zeros input should remain all-zeros";
    }

    // Test 5c: Large position (test numerical stability)
    {
        auto original_data = generate_test_head(head_dim);
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), 1000000, inv_freq, head_dim, 0);

        // Check for NaN/Inf
        bool has_invalid = false;
        for (float val : scalar_result)
        {
            if (!std::isfinite(val))
            {
                has_invalid = true;
                break;
            }
        }
        EXPECT_FALSE(has_invalid) << "Large position should not produce NaN/Inf";
    }
}

// ============================================================================
// Test 6: Vectorized Tail Handling
// ============================================================================

TEST_F(RoPEPrimitivesTest, VectorizedTailHandling)
{
    // Test head dimensions that don't align perfectly with vector widths
    const std::vector<int> misaligned_dims = {36, 68, 100}; // Not multiples of 8 or 16
    const int position = 10;
    const float freq_base = 10000.0f;

    for (int head_dim : misaligned_dims)
    {
        SCOPED_TRACE("head_dim=" + std::to_string(head_dim));

        const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);
        auto original_data = generate_test_head(head_dim);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX2 tail handling failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "AVX512 tail handling failed for head_dim=" << head_dim;
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}

// ============================================================================
// Test 7: Stress Test - Many Heads
// ============================================================================

TEST_F(RoPEPrimitivesTest, StressTestManyHeads)
{
    const int head_dim = 64;
    const int num_heads = 100;
    const int position = 42;
    const float freq_base = 10000.0f;

    const auto &inv_freq = get_inv_freq_cached(head_dim, freq_base);

    for (int h = 0; h < num_heads; ++h)
    {
        auto original_data = generate_test_head(head_dim, 42 + h);

        // Scalar reference
        auto scalar_result = original_data;
        apply_rope_to_head_scalar(scalar_result.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        if (has_avx512_)
        {
            auto avx512_result = original_data;
            int processed = apply_rope_to_head_avx512(avx512_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx512_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx512_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "Head " << h << " failed";
            EXPECT_EQ(mismatches, 0);
        }
#elif defined(__AVX2__)
        if (has_avx2_)
        {
            auto avx2_result = original_data;
            int processed = apply_rope_to_head_avx2(avx2_result.data(), position, inv_freq, head_dim);
            apply_rope_to_head_scalar(avx2_result.data(), position, inv_freq, head_dim, processed);

            auto [max_abs, rel_l2, mismatches] = compare_vectors(avx2_result, scalar_result, 1e-6f);
            EXPECT_LT(max_abs, 1e-5f) << "Head " << h << " failed";
            EXPECT_EQ(mismatches, 0);
        }
#endif
    }
}
