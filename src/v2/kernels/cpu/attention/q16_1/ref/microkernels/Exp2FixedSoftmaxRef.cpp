#include "Exp2FixedSoftmaxRef.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace llaminar2::kernels::q16_1::microkernels
{

    namespace
    {
        // log2(e)
        constexpr double LOG2E = 1.4426950408889634073599246810018921;

        struct Exp2LUT
        {
            std::array<uint32_t, 256> lut{};
            bool initialized = false;

            void ensure_initialized(int lut_value_bits)
            {
                if (initialized)
                    return;

                const double scale = static_cast<double>(1ull << lut_value_bits);
                for (int i = 0; i < 256; ++i)
                {
                    const double u = static_cast<double>(i) / 256.0; // [0, 1)
                    const double v = std::pow(2.0, -u);
                    const double q = std::round(v * scale);
                    uint64_t qi = static_cast<uint64_t>(q);
                    if (qi > 0xFFFFFFFFull)
                        qi = 0xFFFFFFFFull;
                    lut[static_cast<size_t>(i)] = static_cast<uint32_t>(qi);
                }

                initialized = true;
            }
        };

        Exp2LUT &get_exp2_lut()
        {
            static Exp2LUT table;
            return table;
        }

    } // namespace

    void exp2_fixed_softmax_row(
        const int32_t *scores,
        int16_t *weights,
        int n,
        float alpha,
        int32_t *sum_out,
        const Exp2FixedSoftmaxConfig &config)
    {
        if (sum_out)
        {
            *sum_out = 0;
        }
        if (scores == nullptr || weights == nullptr || n <= 0)
        {
            return;
        }

        // Find max over unmasked positions.
        int32_t max_score = std::numeric_limits<int32_t>::min();
        int unmasked = 0;
        for (int i = 0; i < n; ++i)
        {
            const int32_t s = scores[i];
            if (s == std::numeric_limits<int32_t>::min())
            {
                continue;
            }
            max_score = std::max(max_score, s);
            ++unmasked;
        }

        if (unmasked == 0)
        {
            std::fill(weights, weights + n, static_cast<int16_t>(0));
            return;
        }

        // Convert alpha (exp domain) to beta (log2 domain): exp(-x) = 2^{-x * log2(e)}.
        const double beta = static_cast<double>(alpha) * LOG2E;

        // If beta is too small to represent after scaling, fall back to uniform weights.
        const int frac_bits = config.frac_bits;
        const int beta_scale_bits = config.beta_scale_bits;
        const int lut_value_bits = config.lut_value_bits;

        // M ~= beta * 2^{beta_scale_bits}
        const int64_t M = static_cast<int64_t>(std::llround(beta * static_cast<double>(1ull << beta_scale_bits)));

        if (M <= 0)
        {
            const int16_t w = config.weight_max;
            int64_t sum64 = 0;
            for (int i = 0; i < n; ++i)
            {
                if (scores[i] == std::numeric_limits<int32_t>::min())
                {
                    weights[i] = 0;
                }
                else
                {
                    weights[i] = w;
                    sum64 += static_cast<int64_t>(w);
                }
            }
            if (sum_out)
            {
                *sum_out = static_cast<int32_t>(std::min<int64_t>(sum64, std::numeric_limits<int32_t>::max()));
            }
            return;
        }

        auto &table = get_exp2_lut();
        table.ensure_initialized(lut_value_bits);

        // First pass: compute unnormalized exp2 values in a widened integer domain.
        std::vector<uint32_t> e_vals(static_cast<size_t>(n), 0u);
        uint64_t sum_e = 0;

        const int shift_for_t = beta_scale_bits - frac_bits; // t_fixed = round(delta*beta*2^{frac_bits})

        for (int i = 0; i < n; ++i)
        {
            const int32_t s = scores[i];
            if (s == std::numeric_limits<int32_t>::min())
            {
                e_vals[static_cast<size_t>(i)] = 0u;
                continue;
            }

            const int32_t delta = max_score - s;
            if (delta <= 0)
            {
                // exp(0) = 1
                const uint32_t one = static_cast<uint32_t>(1u << lut_value_bits);
                e_vals[static_cast<size_t>(i)] = one;
                sum_e += static_cast<uint64_t>(one);
                continue;
            }

            // t_fixed (Q(frac_bits))
            const int64_t prod = static_cast<int64_t>(delta) * M;
            const int64_t t_fixed = (shift_for_t >= 0) ? (prod >> shift_for_t) : (prod << (-shift_for_t));

            const int64_t ip = t_fixed >> frac_bits;
            const int frac = static_cast<int>(t_fixed & ((1 << frac_bits) - 1));

            if (ip >= 31)
            {
                e_vals[static_cast<size_t>(i)] = 0u;
                continue;
            }

            // LUT for 2^{-frac/256}
            const uint32_t frac_scale = table.lut[static_cast<size_t>(frac)];
            const uint32_t ev = frac_scale >> static_cast<uint32_t>(ip);
            e_vals[static_cast<size_t>(i)] = ev;
            sum_e += static_cast<uint64_t>(ev);
        }

        if (sum_e == 0)
        {
            // Underflow (or all masked) - produce uniform weights for unmasked positions.
            const int16_t w = config.weight_max;
            int64_t sum64 = 0;
            for (int i = 0; i < n; ++i)
            {
                if (scores[i] == std::numeric_limits<int32_t>::min())
                {
                    weights[i] = 0;
                }
                else
                {
                    weights[i] = w;
                    sum64 += static_cast<int64_t>(w);
                }
            }
            if (sum_out)
            {
                *sum_out = static_cast<int32_t>(std::min<int64_t>(sum64, std::numeric_limits<int32_t>::max()));
            }
            return;
        }

        // Second pass: normalize to INT16 weights.
        int64_t sum_w = 0;
        const uint64_t denom = sum_e;
        const uint64_t half = denom / 2;

        for (int i = 0; i < n; ++i)
        {
            if (scores[i] == std::numeric_limits<int32_t>::min())
            {
                weights[i] = 0;
                continue;
            }

            const uint64_t num = static_cast<uint64_t>(e_vals[static_cast<size_t>(i)]) *
                                 static_cast<uint64_t>(config.weight_max);
            const uint64_t w_u = (num + half) / denom;
            const uint16_t w16 = static_cast<uint16_t>(std::min<uint64_t>(w_u, static_cast<uint64_t>(config.weight_max)));
            weights[i] = static_cast<int16_t>(w16);
            sum_w += static_cast<int64_t>(w16);
        }

        if (sum_out)
        {
            *sum_out = static_cast<int32_t>(std::min<int64_t>(sum_w, std::numeric_limits<int32_t>::max()));
        }
    }

} // namespace llaminar2::kernels::q16_1::microkernels
