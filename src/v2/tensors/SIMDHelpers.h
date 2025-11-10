#pragma once

/**
 * @file SIMDHelpers.h
 * @brief SIMD helper functions for IQ4_NL quantization decode
 * @author David Sanftenberg
 *
 * Provides AVX512/AVX2 optimized int8→float32 conversion for IQ4_NL lookup table values.
 * Functions are inline and header-only for zero-cost abstraction.
 */

#include "../utils/CPUFeatures.h"
#include "FP16Utils.h"
#include <cstdint>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace simd
    {

        // Re-export CPU feature detection for compatibility
        using llaminar2::cpu_supports_avx;
        using llaminar2::cpu_supports_avx2;
        using llaminar2::cpu_supports_avx512;
        using llaminar2::cpu_supports_sse41;

        // Re-export FP16 conversion for convenience
        using llaminar2::fp16_to_fp32;
        using llaminar2::fp32_to_fp16;

        // ========================================================================
        // BF16 Conversion Helpers
        // ========================================================================

        /**
         * @brief Convert BF16 to FP32 (scalar)
         *
         * BF16 format: sign(1) + exponent(8) + mantissa(7) = 16 bits
         * FP32 format: sign(1) + exponent(8) + mantissa(23) = 32 bits
         * Conversion: Just shift left by 16 bits (zero-extend mantissa)
         *
         * @param bf16 BF16 value (16-bit)
         * @return FP32 value
         */
        inline float bf16_to_fp32(uint16_t bf16)
        {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16) << 16;
            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
        }

        /**
         * @brief Convert FP32 to BF16 (scalar, round-to-nearest-even)
         *
         * @param fp32 FP32 value
         * @return BF16 value (16-bit)
         */
        inline uint16_t fp32_to_bf16(float fp32)
        {
            uint32_t fp32_bits;
            std::memcpy(&fp32_bits, &fp32, sizeof(float));

            // Round-to-nearest-even: Add 0x7FFF + LSB of result
            uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
            uint32_t rounded = fp32_bits + rounding_bias;

            return static_cast<uint16_t>(rounded >> 16);
        }

        // ========================================================================
        // BF16 → FP32 Array Conversion (SIMD-optimized)
        // ========================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Convert BF16 array to FP32 using AVX-512 (32 elements per iteration)
         *
         * BF16→FP32 is lossless: just zero-extend mantissa by shifting left 16 bits.
         * No rounding needed, making this simpler than FP32→BF16.
         *
         * Note: Regular stores used instead of streaming stores. Empirical testing showed
         * streaming stores (_mm512_stream_ps) cause 40% performance regression in GEMM
         * workloads despite 15% improvement in pure conversion benchmarks. This is because:
         * - GEMM immediately reads the converted data (cache hit important)
         * - Streaming stores bypass cache, forcing memory reads
         * - Only beneficial for write-only workloads
         *
         * @param src Source BF16 array (must have at least count elements)
         * @param dst Destination FP32 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 32)
         */
        inline void convert_bf16_to_fp32_avx512(const uint16_t *src, float *dst, size_t count)
        {
            for (size_t i = 0; i < count; i += 32)
            {
                // Load 32 BF16 values (512 bits)
                __m512i bf16_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + i));

                // Split into two halves for processing (16 elements each)
                __m256i bf16_lo = _mm512_extracti64x4_epi64(bf16_vec, 0); // Lower 16 BF16
                __m256i bf16_hi = _mm512_extracti64x4_epi64(bf16_vec, 1); // Upper 16 BF16

                // Zero-extend 16-bit to 32-bit and shift left by 16
                __m512i fp32_lo = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_lo), 16);
                __m512i fp32_hi = _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_hi), 16);

                // Use regular stores (keep in cache for immediate GEMM consumption)
                // AlignedVector provides 64-byte alignment, so we could use _mm512_store_ps,
                // but _mm512_storeu_ps works fine and is safer
                _mm512_storeu_ps(dst + i, _mm512_castsi512_ps(fp32_lo));
                _mm512_storeu_ps(dst + i + 16, _mm512_castsi512_ps(fp32_hi));
            }
        }
#else
        inline void convert_bf16_to_fp32_avx512(const uint16_t *, float *, size_t) {}
#endif

#if defined(__AVX2__)
        /**
         * @brief Convert BF16 array to FP32 using AVX2 (16 elements per iteration)
         *
         * BF16→FP32 is lossless: just zero-extend mantissa by shifting left 16 bits.
         *
         * @param src Source BF16 array (must have at least count elements)
         * @param dst Destination FP32 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 16)
         */
        inline void convert_bf16_to_fp32_avx2(const uint16_t *src, float *dst, size_t count)
        {
            for (size_t i = 0; i < count; i += 16)
            {
                // Load 16 BF16 values (256 bits)
                __m256i bf16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));

                // Split into two halves (8 elements each)
                __m128i bf16_lo = _mm256_extracti128_si256(bf16_vec, 0); // Lower 8 BF16
                __m128i bf16_hi = _mm256_extracti128_si256(bf16_vec, 1); // Upper 8 BF16

                // Zero-extend 16-bit to 32-bit and shift left by 16
                __m256i fp32_lo = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_lo), 16);
                __m256i fp32_hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_hi), 16);

                // Store as FP32
                _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(fp32_lo));
                _mm256_storeu_ps(dst + i + 8, _mm256_castsi256_ps(fp32_hi));
            }
        }
#else
        inline void convert_bf16_to_fp32_avx2(const uint16_t *, float *, size_t) {}
#endif

        /**
         * @brief Convert BF16 array to FP32 using scalar operations (portable fallback)
         *
         * Uses bf16_to_fp32() for each element.
         * Always available, works on any architecture.
         *
         * @param src Source BF16 array
         * @param dst Destination FP32 array
         * @param count Number of elements to convert
         */
        inline void convert_bf16_to_fp32_scalar(const uint16_t *src, float *dst, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                dst[i] = bf16_to_fp32(src[i]);
            }
        }

        // ========================================================================
        // FP32 → BF16 Array Conversion (SIMD-optimized)
        // ========================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
        /**
         * @brief Convert FP32 array to BF16 using AVX-512 (32 elements per iteration)
         *
         * Uses AVX-512 instructions for maximum throughput:
         * - Processes 32 FP32 values → 32 BF16 values per iteration
         * - Round-to-nearest-even using vpaddw + vpsrlw
         * - ~8x faster than scalar for large arrays
         *
         * @param src Source FP32 array (must have at least count elements)
         * @param dst Destination BF16 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 32)
         */
        inline void convert_fp32_to_bf16_avx512(const float *src, uint16_t *dst, size_t count)
        {
            const __m512i rounding_bias = _mm512_set1_epi32(0x7FFF);

            for (size_t i = 0; i < count; i += 32)
            {
                // Load 32 FP32 values (2 zmm registers)
                __m512 vec0 = _mm512_loadu_ps(src + i);
                __m512 vec1 = _mm512_loadu_ps(src + i + 16);

                // Reinterpret as int32 for bit manipulation
                __m512i int0 = _mm512_castps_si512(vec0);
                __m512i int1 = _mm512_castps_si512(vec1);

                // Add rounding bias + LSB for round-to-nearest-even
                __m512i lsb0 = _mm512_srli_epi32(int0, 16);
                __m512i lsb1 = _mm512_srli_epi32(int1, 16);
                lsb0 = _mm512_and_si512(lsb0, _mm512_set1_epi32(1));
                lsb1 = _mm512_and_si512(lsb1, _mm512_set1_epi32(1));

                __m512i rounded0 = _mm512_add_epi32(int0, rounding_bias);
                __m512i rounded1 = _mm512_add_epi32(int1, rounding_bias);
                rounded0 = _mm512_add_epi32(rounded0, lsb0);
                rounded1 = _mm512_add_epi32(rounded1, lsb1);

                // Shift right to extract BF16 (upper 16 bits)
                __m512i bf16_32bit_0 = _mm512_srli_epi32(rounded0, 16);
                __m512i bf16_32bit_1 = _mm512_srli_epi32(rounded1, 16);

                // Pack 32-bit values to 16-bit (32 BF16 values)
                __m512i packed = _mm512_packus_epi32(bf16_32bit_0, bf16_32bit_1);

                // Permute to correct order and store
                const __m512i perm_idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
                packed = _mm512_permutexvar_epi64(perm_idx, packed);

                _mm512_storeu_si512(reinterpret_cast<__m512i *>(dst + i), packed);
            }
        }
#else
        inline void convert_fp32_to_bf16_avx512(const float *, uint16_t *, size_t) {}
#endif

#if defined(__AVX2__)
        /**
         * @brief Convert FP32 array to BF16 using AVX2 (16 elements per iteration)
         *
         * Uses AVX2 instructions for good throughput on older CPUs:
         * - Processes 16 FP32 values → 16 BF16 values per iteration
         * - Round-to-nearest-even using paddw + psrlw
         * - ~4x faster than scalar
         *
         * @param src Source FP32 array (must have at least count elements)
         * @param dst Destination BF16 array (must have at least count elements)
         * @param count Number of elements to convert (should be multiple of 16)
         */
        inline void convert_fp32_to_bf16_avx2(const float *src, uint16_t *dst, size_t count)
        {
            const __m256i rounding_bias = _mm256_set1_epi32(0x7FFF);

            for (size_t i = 0; i < count; i += 16)
            {
                // Load 16 FP32 values (2 ymm registers)
                __m256 vec0 = _mm256_loadu_ps(src + i);
                __m256 vec1 = _mm256_loadu_ps(src + i + 8);

                // Reinterpret as int32
                __m256i int0 = _mm256_castps_si256(vec0);
                __m256i int1 = _mm256_castps_si256(vec1);

                // Add rounding bias + LSB
                __m256i lsb0 = _mm256_srli_epi32(int0, 16);
                __m256i lsb1 = _mm256_srli_epi32(int1, 16);
                lsb0 = _mm256_and_si256(lsb0, _mm256_set1_epi32(1));
                lsb1 = _mm256_and_si256(lsb1, _mm256_set1_epi32(1));

                __m256i rounded0 = _mm256_add_epi32(int0, rounding_bias);
                __m256i rounded1 = _mm256_add_epi32(int1, rounding_bias);
                rounded0 = _mm256_add_epi32(rounded0, lsb0);
                rounded1 = _mm256_add_epi32(rounded1, lsb1);

                // Shift to extract BF16
                __m256i bf16_32bit_0 = _mm256_srli_epi32(rounded0, 16);
                __m256i bf16_32bit_1 = _mm256_srli_epi32(rounded1, 16);

                // Pack to 16-bit
                __m256i packed = _mm256_packus_epi32(bf16_32bit_0, bf16_32bit_1);

                // Permute lanes to correct order
                packed = _mm256_permute4x64_epi64(packed, 0xD8); // 0b11011000

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), packed);
            }
        }
#else
        inline void convert_fp32_to_bf16_avx2(const float *, uint16_t *, size_t) {}
#endif

        /**
         * @brief Convert FP32 array to BF16 using scalar operations (portable fallback)
         *
         * Uses fp32_to_bf16() for each element.
         * Always available, works on any architecture.
         *
         * @param src Source FP32 array
         * @param dst Destination BF16 array
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_bf16_scalar(const float *src, uint16_t *dst, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                dst[i] = fp32_to_bf16(src[i]);
            }
        }

        // ========================================================================
        // AVX-512 Conversion Helpers (16 elements at a time)
        // ========================================================================

#ifdef __AVX512F__

        /**
         * @brief Convert 16 int8 values to float32 with scale (AVX512)
         *
         * Used by IQ4_NL decode: lookup_values[16] (int8) → output[16] (float32)
         * Formula: output[i] = input[i] * scale
         *
         * @param input Pointer to 16 int8 values
         * @param scale Scale factor to multiply
         * @param output Pointer to 16 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_avx512(const int8_t *input, float scale, float *output)
        {
            __m128i i8_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(input));
            __m512i i32_vec = _mm512_cvtepi8_epi32(i8_vec);
            __m512 f32_vec = _mm512_cvtepi32_ps(i32_vec);
            __m512 result = _mm512_mul_ps(f32_vec, _mm512_set1_ps(scale));
            _mm512_storeu_ps(output, result);
        }

#else
        // Fallback: not available without AVX512
        inline void convert_i8_to_f32_scaled_avx512(const int8_t *, float, float *)
        {
            // Should never be called (guarded by cpu_supports_avx512() checks)
        }
#endif

        // ========================================================================
        // AVX2 Conversion Helpers (8 elements at a time)
        // ========================================================================

#ifdef __AVX2__

        /**
         * @brief Convert 8 int8 values to float32 with scale (AVX2)
         *
         * Used by IQ4_NL decode: lookup_values[8] (int8) → output[8] (float32)
         * Formula: output[i] = input[i] * scale
         *
         * @param input Pointer to 8 int8 values
         * @param scale Scale factor to multiply
         * @param output Pointer to 8 float32 output buffer
         */
        inline void convert_i8_to_f32_scaled_avx2(const int8_t *input, float scale, float *output)
        {
            // Load 8 int8 values (8 bytes)
            __m128i i8_vec = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(input));

            // Sign-extend int8 → int32 (8 elements)
            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);

            // Convert int32 → float32
            __m256 f32_vec = _mm256_cvtepi32_ps(i32_vec);

            // Multiply by scale
            __m256 result = _mm256_mul_ps(f32_vec, _mm256_set1_ps(scale));

            // Store result
            _mm256_storeu_ps(output, result);
        }

#else
        // Fallback: not available without AVX2
        inline void convert_i8_to_f32_scaled_avx2(const int8_t *, float, float *)
        {
            // Should never be called (guarded by cpu_supports_avx2() checks)
        }
#endif

        // ========================================================================
        // Scalar Fallback (portable, no SIMD)
        // ========================================================================

        /**
         * @brief Convert N int8 values to float32 with scale (scalar fallback)
         *
         * Portable implementation for platforms without AVX512/AVX2.
         * Used when SIMD is not available or for small counts.
         *
         * @param input Pointer to int8 values
         * @param scale Scale factor to multiply
         * @param output Pointer to float32 output buffer
         * @param count Number of elements to convert
         */
        inline void convert_i8_to_f32_scaled_scalar(const int8_t *input, float scale, float *output, size_t count)
        {
            for (size_t i = 0; i < count; ++i)
            {
                output[i] = static_cast<float>(input[i]) * scale;
            }
        }

        // ========================================================================
        // High-Level Auto-Dispatch Wrappers
        // ========================================================================

        /**
         * @brief Convert BF16 array to FP32 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512 if available (32 elements/iter)
         *   - AVX2 if available (16 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source BF16 array (uint16_t*)
         * @param dst Destination FP32 array (float*)
         * @param count Number of elements to convert
         */
        inline void convert_bf16_to_fp32(const uint16_t *src, float *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
            // AVX512: Process 32 elements at a time
            constexpr size_t VEC_SIZE = 32;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_bf16_to_fp32_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__)
            // AVX2: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_bf16_to_fp32_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = bf16_to_fp32(src[i]);
            }
        }

        /**
         * @brief Convert FP32 array to BF16 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512 if available (32 elements/iter)
         *   - AVX2 if available (16 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source FP32 array (float*)
         * @param dst Destination BF16 array (uint16_t*)
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_bf16(const float *src, uint16_t *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
            // AVX512: Process 32 elements at a time
            constexpr size_t VEC_SIZE = 32;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_bf16_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__)
            // AVX2: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_bf16_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = fp32_to_bf16(src[i]);
            }
        }

        // ====================================================================
        // FP16 ↔ FP32 Conversion (Vectorized with Auto-Dispatch)
        // ====================================================================

        /**
         * @brief Convert FP16 array to FP32 using AVX512/AVX2/F16C (vectorized)
         *
         * Automatically dispatches to best available ISA:
         *   - AVX512F: 16 elements/iteration (native _mm512_cvtph_ps)
         *   - AVX2+F16C: 8 elements/iteration (_mm256_cvtph_ps)
         *   - Scalar fallback: 1 element/iteration
         *
         * @param src Source FP16 array (uint16_t*)
         * @param dst Destination FP32 array (float*)
         * @param count Number of elements to convert
         */
        inline void convert_fp16_to_fp32_avx512(const uint16_t *src, float *dst, size_t count)
        {
#if defined(__AVX512F__)
            constexpr size_t VEC_SIZE = 16; // 16 FP16s → 16 FP32s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 16 FP16 values (256 bits)
                __m256i fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));

                // Convert FP16 → FP32 (native AVX512 instruction)
                __m512 fp32_vec = _mm512_cvtph_ps(fp16_vec);

                // Store 16 FP32 values
                _mm512_storeu_ps(dst + i, fp32_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        inline void convert_fp16_to_fp32_avx2(const uint16_t *src, float *dst, size_t count)
        {
#if defined(__AVX2__) && defined(__F16C__)
            constexpr size_t VEC_SIZE = 8; // 8 FP16s → 8 FP32s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 8 FP16 values (128 bits)
                __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));

                // Convert FP16 → FP32 (F16C instruction)
                __m256 fp32_vec = _mm256_cvtph_ps(fp16_vec);

                // Store 8 FP32 values
                _mm256_storeu_ps(dst + i, fp32_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        /**
         * @brief Convert FP32 array to FP16 using AVX512/AVX2/F16C (vectorized)
         *
         * Automatically dispatches to best available ISA:
         *   - AVX512F: 16 elements/iteration (native _mm512_cvtps_ph)
         *   - AVX2+F16C: 8 elements/iteration (_mm256_cvtps_ph)
         *   - Scalar fallback: 1 element/iteration
         *
         * @param src Source FP32 array (float*)
         * @param dst Destination FP16 array (uint16_t*)
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_fp16_avx512(const float *src, uint16_t *dst, size_t count)
        {
#if defined(__AVX512F__)
            constexpr size_t VEC_SIZE = 16; // 16 FP32s → 16 FP16s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 16 FP32 values
                __m512 fp32_vec = _mm512_loadu_ps(src + i);

                // Convert FP32 → FP16 (native AVX512 instruction, round to nearest)
                __m256i fp16_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Store 16 FP16 values (256 bits)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), fp16_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        inline void convert_fp32_to_fp16_avx2(const float *src, uint16_t *dst, size_t count)
        {
#if defined(__AVX2__) && defined(__F16C__)
            constexpr size_t VEC_SIZE = 8; // 8 FP32s → 8 FP16s
            for (size_t i = 0; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                // Load 8 FP32 values
                __m256 fp32_vec = _mm256_loadu_ps(src + i);

                // Convert FP32 → FP16 (F16C instruction, round to nearest)
                __m128i fp16_vec = _mm256_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

                // Store 8 FP16 values (128 bits)
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i), fp16_vec);
            }
#else
            (void)src;
            (void)dst;
            (void)count; // Suppress unused parameter warnings
#endif
        }

        /**
         * @brief Convert FP16 array to FP32 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512F if available (16 elements/iter)
         *   - AVX2+F16C if available (8 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source FP16 array (uint16_t*)
         * @param dst Destination FP32 array (float*)
         * @param count Number of elements to convert
         */
        inline void convert_fp16_to_fp32(const uint16_t *src, float *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__)
            // AVX512: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp16_to_fp32_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__) && defined(__F16C__)
            // AVX2+F16C: Process 8 elements at a time
            constexpr size_t VEC_SIZE = 8;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp16_to_fp32_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = fp16_to_fp32(src[i]);
            }
        }

        /**
         * @brief Convert FP32 array to FP16 (auto-dispatched to best ISA)
         *
         * Automatically selects:
         *   - AVX512F if available (16 elements/iter)
         *   - AVX2+F16C if available (8 elements/iter)
         *   - Scalar fallback (1 element/iter)
         *
         * @param src Source FP32 array (float*)
         * @param dst Destination FP16 array (uint16_t*)
         * @param count Number of elements to convert
         */
        inline void convert_fp32_to_fp16(const float *src, uint16_t *dst, size_t count)
        {
            size_t i = 0;

#if defined(__AVX512F__)
            // AVX512: Process 16 elements at a time
            constexpr size_t VEC_SIZE = 16;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_fp16_avx512(src + i, dst + i, VEC_SIZE);
            }
#elif defined(__AVX2__) && defined(__F16C__)
            // AVX2+F16C: Process 8 elements at a time
            constexpr size_t VEC_SIZE = 8;
            for (; i + VEC_SIZE <= count; i += VEC_SIZE)
            {
                convert_fp32_to_fp16_avx2(src + i, dst + i, VEC_SIZE);
            }
#endif

            // Scalar fallback for remainder
            for (; i < count; ++i)
            {
                dst[i] = fp32_to_fp16(src[i]);
            }
        }

    } // namespace simd
} // namespace llaminar2
