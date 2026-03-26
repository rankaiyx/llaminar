/**
 * @file TurboQuantCodebook.h
 * @brief Lloyd-Max optimal codebook centroids for TurboQuant quantization
 * @author David Sanftenberg
 *
 * Provides pre-computed Lloyd-Max optimal codebook centroids for the standard
 * normal distribution N(0,1). These are used by TurboQuant to quantize each
 * rotated coordinate of a unit-norm vector.
 *
 * The centroids are optimal in the MSE sense for N(0,1). For the unit-sphere
 * marginal N(0, 1/d), the input is scaled by sqrt(d) before quantization
 * and the codebook values are divided by sqrt(d) during dequantization.
 *
 * Reference: Max (1960), "Quantizing for minimum distortion", IRE Trans. Info. Theory.
 *
 * The codebook provides:
 *   - Sorted centroid tables (for dequantization lookup)
 *   - Sorted threshold tables (for fast nearest-centroid via interval lookup)
 *   - Nearest-centroid functions (scalar + SIMD)
 *   - Lloyd-Max verification function (re-derives centroids from first principles)
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <array>
#include <algorithm>

namespace llaminar2
{

    // ========================================================================
    // 2-bit codebook (4 centroids for N(0,1))
    // ========================================================================

    inline constexpr std::array<float, 4> TQ2_CENTROIDS = {
        -1.510418f, -0.452780f, 0.452780f, 1.510418f
    };

    inline constexpr std::array<float, 3> TQ2_THRESHOLDS = {
        -0.981599f, 0.000000f, 0.981599f
    };

    // ========================================================================
    // 4-bit codebook (16 centroids for N(0,1))
    // ========================================================================

    /**
     * @brief Lloyd-Max optimal 16-level centroids for N(0,1), sorted ascending.
     *
     * Index 0 = most negative, index 15 = most positive.
     * Symmetric about 0: centroid[k] = -centroid[15-k].
     */
    inline constexpr std::array<float, 16> TQ4_CENTROIDS = {
        -2.732897f, -2.069364f, -1.618400f, -1.256565f,
        -0.942629f, -0.656982f, -0.388189f, -0.128443f,
         0.128443f,  0.388189f,  0.656982f,  0.942629f,
         1.256565f,  1.618400f,  2.069364f,  2.732897f
    };

    /**
     * @brief Decision thresholds for 4-bit quantization.
     *
     * 15 thresholds partition the real line into 16 intervals.
     * If x falls in interval [threshold[k-1], threshold[k]), assign index k.
     * threshold[-1] = -inf, threshold[15] = +inf.
     *
     * Each threshold = midpoint of adjacent centroids.
     */
    inline constexpr std::array<float, 15> TQ4_THRESHOLDS = {
        -2.401131f, -1.843882f, -1.437483f, -1.099597f,
        -0.799805f, -0.522585f, -0.258316f,  0.000000f,
         0.258316f,  0.522585f,  0.799805f,  1.099597f,
         1.437483f,  1.843882f,  2.401131f
    };

    // ========================================================================
    // 3-bit codebook (8 centroids for N(0,1))
    // ========================================================================

    /**
     * @brief Lloyd-Max optimal 8-level centroids for N(0,1), sorted ascending.
     *
     * Index 0 = most negative, index 7 = most positive.
     * Symmetric about 0: centroid[k] = -centroid[7-k].
     */
    inline constexpr std::array<float, 8> TQ3_CENTROIDS = {
        -2.151946f, -1.343909f, -0.756005f, -0.245094f,
         0.245094f,  0.756005f,  1.343909f,  2.151946f
    };

    /**
     * @brief Decision thresholds for 3-bit quantization.
     *
     * 7 thresholds partition the real line into 8 intervals.
     */
    inline constexpr std::array<float, 7> TQ3_THRESHOLDS = {
        -1.747928f, -1.049957f, -0.500550f, 0.000000f,
         0.500550f,  1.049957f,  1.747928f
    };

    // ========================================================================
    // Nearest-centroid lookup (scalar)
    // ========================================================================

    /**
     * @brief Find nearest 4-bit centroid index for a scalar value.
     *
     * Uses threshold-based interval lookup (4 comparisons via binary search).
     * Input should be pre-scaled by sqrt(d) for unit-sphere quantization.
     *
     * @param x Scalar value (in N(0,1) scale after sqrt(d) scaling)
     * @return Index 0-15 of nearest centroid
     */
    inline uint8_t tq4_nearest_centroid(float x)
    {
        // Binary search through 15 thresholds to find interval
        // Unrolled for performance (exactly 4 comparisons for 16 levels)
        uint8_t idx = 0;
        idx += (x > TQ4_THRESHOLDS[7])  ? 8 : 0;
        idx += (x > TQ4_THRESHOLDS[idx + 3]) ? 4 : 0;
        idx += (x > TQ4_THRESHOLDS[idx + 1]) ? 2 : 0;
        // Final comparison: idx is now in {0..14}, check if we need +1
        if (idx < 15 && x > TQ4_THRESHOLDS[idx]) idx++;
        return idx;
    }

    /**
     * @brief Find nearest 3-bit centroid index for a scalar value.
     *
     * Uses threshold-based interval lookup (3 comparisons via binary search).
     *
     * @param x Scalar value (in N(0,1) scale after sqrt(d) scaling)
     * @return Index 0-7 of nearest centroid
     */
    inline uint8_t tq3_nearest_centroid(float x)
    {
        // Binary search through 7 thresholds (3 comparisons for 8 levels)
        uint8_t idx = 0;
        idx += (x > TQ3_THRESHOLDS[3]) ? 4 : 0;
        idx += (x > TQ3_THRESHOLDS[idx + 1]) ? 2 : 0;
        if (idx < 7 && x > TQ3_THRESHOLDS[idx]) idx++;
        return idx;
    }

    inline uint8_t tq2_nearest_centroid(float x)
    {
        uint8_t idx = 0;
        idx += (x > TQ2_THRESHOLDS[1]) ? 2 : 0;
        if (idx < 3 && x > TQ2_THRESHOLDS[idx]) idx++;
        return idx;
    }

    // ========================================================================
    // Lloyd-Max verification utilities
    // ========================================================================

    /**
     * @brief Standard normal PDF φ(x) = exp(-x²/2) / √(2π)
     */
    inline double gaussian_pdf(double x)
    {
        constexpr double INV_SQRT_2PI = 0.3989422804014327;
        return INV_SQRT_2PI * std::exp(-0.5 * x * x);
    }

    /**
     * @brief Standard normal CDF Φ(x) using erfc
     */
    inline double gaussian_cdf(double x)
    {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

    /**
     * @brief Conditional expectation E[X | a ≤ X ≤ b] for X ~ N(0,1)
     *
     * = (φ(a) - φ(b)) / (Φ(b) - Φ(a))
     */
    inline double gaussian_conditional_mean(double a, double b)
    {
        double prob = gaussian_cdf(b) - gaussian_cdf(a);
        if (prob < 1e-15) return 0.5 * (a + b);
        return (gaussian_pdf(a) - gaussian_pdf(b)) / prob;
    }

    /**
     * @brief Apply one Lloyd-Max iteration to a set of centroids.
     *
     * Given current centroids, computes thresholds as midpoints, then updates
     * each centroid to the conditional expectation E[X | interval] for N(0,1).
     *
     * @tparam K Number of quantization levels (must be even for symmetry)
     * @param centroids Current centroids (updated in-place)
     * @return Maximum absolute centroid movement from this iteration
     */
    template <int K>
    inline double lloyd_max_iteration(std::array<double, K> &centroids)
    {
        static_assert(K >= 2 && K % 2 == 0, "K must be even and >= 2");

        // Step 1: thresholds = midpoints of adjacent centroids
        std::array<double, K - 1> thresholds;
        for (int k = 0; k < K - 1; ++k)
            thresholds[k] = 0.5 * (centroids[k] + centroids[k + 1]);

        // Step 2: centroids = conditional expectations over each interval
        std::array<double, K> new_centroids;
        new_centroids[0] = gaussian_conditional_mean(-10.0, thresholds[0]);
        for (int k = 1; k < K - 1; ++k)
            new_centroids[k] = gaussian_conditional_mean(thresholds[k - 1], thresholds[k]);
        new_centroids[K - 1] = gaussian_conditional_mean(thresholds[K - 2], 10.0);

        double max_delta = 0.0;
        for (int k = 0; k < K; ++k)
            max_delta = std::max(max_delta, std::abs(new_centroids[k] - centroids[k]));

        centroids = new_centroids;
        return max_delta;
    }

    /**
     * @brief Compute Lloyd-Max optimal centroids for N(0,1) with K levels.
     *
     * Iterates the Lloyd-Max algorithm from a uniform initialization until
     * convergence (centroid movement < tolerance). No arbitrary iteration cap.
     *
     * @tparam K Number of quantization levels (must be even for symmetry)
     * @param tolerance Convergence tolerance on centroid movement (default 1e-12)
     * @return Array of K sorted centroid values
     */
    template <int K>
    inline std::array<double, K> compute_lloyd_max_centroids(double tolerance = 1e-12)
    {
        static_assert(K >= 2 && K % 2 == 0, "K must be even and >= 2");

        std::array<double, K> centroids;
        for (int k = 0; k < K; ++k)
            centroids[k] = -3.0 + (k + 0.5) * 6.0 / K;

        while (lloyd_max_iteration<K>(centroids) >= tolerance) {}

        return centroids;
    }

    /**
     * @brief Verify that pre-computed codebook centroids are a Lloyd-Max fixed point.
     *
     * Seeds one Lloyd-Max iteration with the stored centroids and measures how
     * much they move. A true fixed point moves by 0 (limited by float32 precision).
     * This avoids iterating from scratch and any dependence on iteration counts.
     *
     * @tparam K Number of centroids
     * @param stored_centroids Pre-computed centroid table to verify
     * @return Maximum centroid movement after one Lloyd-Max iteration
     */
    template <int K>
    inline double verify_codebook(const std::array<float, K> &stored_centroids)
    {
        // Promote to double and apply one Lloyd-Max step
        std::array<double, K> centroids;
        for (int k = 0; k < K; ++k)
            centroids[k] = static_cast<double>(stored_centroids[k]);

        return lloyd_max_iteration<K>(centroids);
    }

    /**
     * @brief Theoretical MSE of Lloyd-Max quantizer for N(0, σ²) with K levels.
     *
     * Computes the distortion D = E[(X - Q(X))²] for the optimal K-level quantizer.
     * Uses the converged centroids and thresholds.
     *
     * @param num_levels Number of quantization levels (8 or 16)
     * @param sigma Standard deviation of the Gaussian
     * @return Expected MSE per element
     */
    inline double lloyd_max_mse(int num_levels, double sigma = 1.0)
    {
        if (num_levels == 16) {
            auto centroids_d = compute_lloyd_max_centroids<16>();
            std::array<double, 15> thresholds_d;
            for (int k = 0; k < 15; ++k)
                thresholds_d[k] = 0.5 * (centroids_d[k] + centroids_d[k + 1]);

            double mse = 0.0;
            for (int k = 0; k < 16; ++k) {
                double a = (k == 0) ? -10.0 : thresholds_d[k - 1];
                double b = (k == 15) ? 10.0 : thresholds_d[k];
                double c = centroids_d[k];
                // E[(X - c)² | a ≤ X ≤ b] * P(a ≤ X ≤ b)
                // = E[X² | a ≤ X ≤ b] * p - 2c * E[X | a ≤ X ≤ b] * p + c² * p
                double p = gaussian_cdf(b) - gaussian_cdf(a);
                double ex = gaussian_conditional_mean(a, b) * p;
                // E[X² | a ≤ X ≤ b] * p = (x*φ(x) evaluated at boundaries) + p
                // Actually: E[X²] in interval = integral of x² * φ(x)
                // = [a·φ(a) - b·φ(b)] + p  for N(0,1)
                double ex2 = (a * gaussian_pdf(a) - b * gaussian_pdf(b)) + p;
                mse += ex2 - 2.0 * c * ex + c * c * p;
            }
            return mse * sigma * sigma;
        }
        if (num_levels == 8) {
            auto centroids_d = compute_lloyd_max_centroids<8>();
            std::array<double, 7> thresholds_d;
            for (int k = 0; k < 7; ++k)
                thresholds_d[k] = 0.5 * (centroids_d[k] + centroids_d[k + 1]);

            double mse = 0.0;
            for (int k = 0; k < 8; ++k) {
                double a = (k == 0) ? -10.0 : thresholds_d[k - 1];
                double b = (k == 7) ? 10.0 : thresholds_d[k];
                double c = centroids_d[k];
                double p = gaussian_cdf(b) - gaussian_cdf(a);
                double ex = gaussian_conditional_mean(a, b) * p;
                double ex2 = (a * gaussian_pdf(a) - b * gaussian_pdf(b)) + p;
                mse += ex2 - 2.0 * c * ex + c * c * p;
            }
            return mse * sigma * sigma;
        }
        return -1.0; // unsupported
    }

} // namespace llaminar2
