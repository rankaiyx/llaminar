/**
 * @file FP16.h
 * @brief FP16 conversion utilities for V2
 * @author David Sanftenberg
 */

#pragma once

#ifndef LLAMINAR2_FP16_H
#define LLAMINAR2_FP16_H

#include <cstdint>
#include <cstring>

namespace llaminar2
{
    /**
     * @brief Convert FP16 (uint16_t) to FP32 (float)
     *
     * IEEE 754 half-precision to single-precision conversion.
     * Handles normal, subnormal, infinity, and NaN values.
     */
    inline float fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exponent = (h & 0x7C00) >> 10;
        uint32_t mantissa = (h & 0x03FF);

        if (exponent == 0)
        {
            // Subnormal or zero
            if (mantissa == 0)
            {
                uint32_t zero = sign;
                float result;
                std::memcpy(&result, &zero, sizeof(float));
                return result;
            }
            else
            {
                // Normalize subnormal
                exponent = 1;
                while ((mantissa & 0x0400) == 0)
                {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x03FF;
                exponent += (127 - 15);
            }
        }
        else if (exponent == 0x1F)
        {
            // Inf or NaN
            exponent = 0xFF;
        }
        else
        {
            // Normalized
            exponent += (127 - 15);
        }

        uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
        float result;
        std::memcpy(&result, &bits, sizeof(float));
        return result;
    }

} // namespace llaminar2

#endif // LLAMINAR2_FP16_H