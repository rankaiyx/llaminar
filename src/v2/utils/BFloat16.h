/**
 * @file BFloat16.h
 * @brief BFloat16 conversion utilities for V2
 * @author David Sanftenberg
 */

#pragma once

#ifndef LLAMINAR2_BFLOAT16_H
#define LLAMINAR2_BFLOAT16_H

#include <cstdint>
#include <cstring>

namespace llaminar2
{
    /**
     * @brief BFloat16 type (Brain Floating Point - 16 bit)
     *
     * BF16 stores the upper 16 bits of IEEE 754 float:
     * - 1 sign bit
     * - 8 exponent bits (same range as FP32)
     * - 7 mantissa bits (reduced precision vs FP32's 23 bits)
     *
     * Advantages:
     * - Same dynamic range as FP32 (prevents overflow/underflow)
     * - Direct truncation for conversion
     * - Hardware acceleration on Ice Lake+, Zen 4+
     * - 2× memory reduction vs FP32
     */
    struct bfloat16
    {
        uint16_t data;

        // Constructors
        bfloat16() : data(0) {}
        explicit bfloat16(float f) { *this = from_float(f); }

        /**
         * @brief Convert FP32 to BF16 with round-to-nearest-even
         */
        static bfloat16 from_float(float f)
        {
            bfloat16 bf;
            uint32_t u;
            std::memcpy(&u, &f, sizeof(float));

            // Round to nearest even: add 0x7FFF + lsb of top 16 bits
            uint32_t lsb = (u >> 16) & 1u; // LSB of truncated part for tie-to-even
            uint32_t rounding_bias = 0x7FFFu + lsb;
            u += rounding_bias;

            bf.data = static_cast<uint16_t>(u >> 16);
            return bf;
        }

        /**
         * @brief Convert BF16 to FP32
         */
        operator float() const
        {
            uint32_t u = static_cast<uint32_t>(data) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(float));
            return f;
        }

        /**
         * @brief Explicit conversion to float
         */
        float to_float() const
        {
            return static_cast<float>(*this);
        }
    };

} // namespace llaminar2

#endif // LLAMINAR2_BFLOAT16_H
