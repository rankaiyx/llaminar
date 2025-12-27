/**
 * @file Exp2FixedSoftmaxRef.h
 * @brief Reference microkernel for exp2 fixed-point softmax (integer-domain)
 *
 * This is an alternative to Index16Softmax for Q16_1 attention experiments.
 * It approximates exp(x) via exp2(x * log2(e)) in fixed-point:
 *   exp(-t) = 2^{-t * log2(e)}
 *
 * Design goals:
 * - Pure integer runtime (no per-element FP ops)
 * - Higher effective resolution than 32-entry index LUT when score scaling is stable
 * - Output INT16 weights in [0, 32767] for AVX512-VNNI compatibility (VPDPWSSD)
 *
 * Notes:
 * - This microkernel expects INT32 scores where a single alpha scale is meaningful
 *   (e.g., INT8 requantized Q/K scores with alpha = sQ*sK/sqrt(d)).
 */
#pragma once

#include <cstdint>

namespace llaminar2::kernels::q16_1::microkernels
{

    /**
     * @brief Configuration for exp2 fixed-point softmax.
     */
    struct Exp2FixedSoftmaxConfig
    {
        // Fraction bits for the exp2 input "t" in log2 domain.
        // We use 8 bits so we can index a 256-entry LUT for the fractional part.
        int frac_bits = 8;

        // Internal scaling bits for converting (delta * beta) to fixed-point.
        // beta = alpha * log2(e).
        int beta_scale_bits = 24;

        // LUT scaling for 2^{-u} values. Larger improves normalization precision.
        // Values are stored as uint32 in [0, 2^lut_value_bits].
        int lut_value_bits = 30;

        // Output weight maximum (INT16, VNNI-friendly).
        int16_t weight_max = 32767;
    };

    /**
     * @brief Apply exp2 fixed-point softmax to a single row of INT32 scores.
     *
     * Algorithm:
     * 1) max-subtract in INT32 for stability: delta = max - score
     * 2) Convert to log2 domain fixed-point: t ~= delta * (alpha * log2(e))
     * 3) Approximate exp(-alpha*delta) as exp2(-t/log2(e)) using:
     *    exp2(-t) = 2^{-ip} * 2^{-frac}
     * 4) Normalize to INT16 weights in [0, weight_max]
     *
     * Masking:
     * - scores==INT32_MIN are treated as masked and receive weight 0.
     *
     * @param scores Input INT32 scores [n] (INT32_MIN = masked)
     * @param weights Output INT16 weights [n] (range [0, 32767])
     * @param n Number of positions
     * @param alpha Score scale factor in FP domain (e.g. sQ*sK/sqrt(d))
     * @param sum_out Output sum of weights (INT32)
     * @param config Configuration parameters
     */
    void exp2_fixed_softmax_row(
        const int32_t *scores,
        int16_t *weights,
        int n,
        float alpha,
        int32_t *sum_out,
        const Exp2FixedSoftmaxConfig &config = Exp2FixedSoftmaxConfig{});

} // namespace llaminar2::kernels::q16_1::microkernels
