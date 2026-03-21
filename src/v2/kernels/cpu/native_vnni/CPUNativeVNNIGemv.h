/**
 * @file CPUNativeVNNIGemv.h
 * @brief Optimized M=1 GEMV kernel with native-interleaved VNNI weights + AVX-512.
 *
 * ## Architecture
 *
 * The weight packer stores native payload bytes (Q4_0 nibbles, IQ4_NL nibbles)
 * in VNNI-interleaved order at pack time:
 *   native_interleaved: [N_chunks][bpr][4 groups][4 ZMMs][64 bytes]
 *
 * Each 64-byte ZMM holds 16 columns × 4 consecutive native bytes.
 * Total weight memory = native payload size (zero expansion).
 *
 * At runtime, AVX-512 vpshufb decodes 4-bit nibbles to signed INT8 directly
 * in VNNI lane order. This gives native-size bandwidth (0.5 byte/element
 * for Q4_0) with vectorized VNNI accumulation — matching the GPU approach.
 *
 * ## Inner Loop (64 columns per N-chunk, 32 K-elements per block)
 *
 * Per K-block:
 *   4 groups × (4 ZMM loads + 8 vpshufb + 8 vpdpbusd) = 16 loads + 32 decode + 32 VNNI
 *   Memory traffic: 1024 bytes (native size — half of pre-decoded INT8)
 *
 * Post-K-block:
 *   4 comp loads + 4 mullo + 4 sub (bias correction)
 *   4 scale loads + 4 mul + 4 fmadd (FP32 accumulation)
 *
 * ## Cache-Aware Tiling
 *
 * Uses NativeVNNITileConfig (from CPUNativeVNNITileConfig.h) to select
 * N-block size based on detected L1/L2/L3 and shape category.
 */

#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>

#include "CPUNativeVNNIDecode.h"
#include "CPUNativeVNNIWeightPacker.h"
#include "CPUNativeVNNITileConfig.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/OpenMPUtils.h"

namespace llaminar2::cpu::native_vnni
{

    // =========================================================================
    // Scalar reference GEMV (any format, for correctness verification)
    // =========================================================================

    /**
     * @brief Scalar NativeVNNI GEMV for correctness reference.
     *
     * For nibble-LUT formats (Q4_0, IQ4_NL, Q4_1, IQ4_XS): decodes from payload.
     * For INT8 pre-decoded formats: reads from int8_flat buffer.
     */
    inline void gemv_native_vnni_scalar(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int N,
        int K_blocks)
    {
        const int N_chunks = (N + 63) / 64;

        for (int chunk = 0; chunk < N_chunks; ++chunk)
        {
            const int n_start = chunk * 64;
            const int n_end = std::min(n_start + 64, N);

            for (int n_local = 0; n_local < (n_end - n_start); ++n_local)
            {
                const int n = n_start + n_local;
                float acc = 0.0f;

                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    const Q8_1Block &a_blk = A_q8[kb];
                    float a_scale = simd::fp16_to_fp32(a_blk.d);

                    int8_t b_vals[32];
                    if (packed.is_nibble_lut)
                    {
                        const uint8_t *payload = packed.blockPayload(chunk, kb, n_local);
                        decode_native_block(packed.codebook_id, payload, b_vals);
                    }
                    else
                    {
                        std::memcpy(b_vals, packed.blockInt8(chunk, kb, n_local), 32);
                    }

                    float b_scale = packed.blockScale(chunk, kb, n_local);
                    float b_min = packed.blockMin(chunk, kb, n_local);

                    int32_t dot = 0;
                    int32_t b_comp = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        uint8_t a_u8 = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[i]) + 128);
                        dot += static_cast<int32_t>(a_u8) * static_cast<int32_t>(b_vals[i]);
                        b_comp += b_vals[i];
                    }

                    int32_t corrected = dot - 128 * b_comp;
                    acc += static_cast<float>(corrected) * a_scale * b_scale;

                    if (packed.is_asymmetric && b_min != 0.0f)
                    {
                        acc += static_cast<float>(a_blk.sum_qs) * a_scale * b_min;
                    }
                }

                C[n] = acc;
            }
        }
    }

    // =========================================================================
    // AVX-512 VNNI GEMV with native-interleaved weights (optimized hot path)
    // =========================================================================
    //
    // The weight packer stores native payload bytes in VNNI-interleaved order:
    //   native_interleaved: [N_chunks][bpr][4 groups][4 ZMMs][64 bytes]
    //
    // Each 64-byte ZMM holds 16 columns × 4 consecutive native bytes.
    // At runtime, vpshufb decodes nibbles→INT8 directly in VNNI lane order:
    //   - Low nibbles  → K-elements [group*4 .. group*4+3]
    //   - High nibbles → K-elements [group*4+16 .. group*4+19]
    //
    // Memory traffic per K-block per 64-col chunk: 1024 bytes (= native size!)
    // vs 2048 bytes for the old pre-decoded INT8 path.
    // =========================================================================

    // Decode LUT tables for nibble→INT8 conversion via vpshufb.
    // Each 16-entry table maps a 4-bit nibble to a signed INT8 value.
    // Q4_0: nibble → (nibble - 8) = [-8, -7, ..., +7]
    alignas(16) static constexpr int8_t Q4_0_DECODE_LUT[16] = {
        -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
    // IQ4_NL / IQ4_XS: nibble → kvalues_iq4nl[nibble]
    alignas(16) static constexpr int8_t IQ4_NL_DECODE_LUT[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    // Q4_1: nibble → nibble (unsigned identity [0..15])
    alignas(16) static constexpr int8_t Q4_1_DECODE_LUT[16] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

    /**
     * @brief Build the 512-bit decode LUT for a given codebook_id.
     *
     * Broadcasts a 16-byte LUT to all four 128-bit lanes of a ZMM register.
     * Used with vpshufb to decode 4-bit nibbles to signed INT8 in one instruction.
     * Only valid for nibble-LUT formats (Q4_0, IQ4_NL, Q4_1, IQ4_XS).
     */
    inline __m512i build_decode_lut(uint8_t codebook_id)
    {
        const int8_t *lut_data;
        switch (codebook_id)
        {
        case 4: // IQ4_NL / IQ4_XS (both use kvalues_iq4nl LUT)
            lut_data = IQ4_NL_DECODE_LUT;
            break;
        case 5: // Q4_1
            lut_data = Q4_1_DECODE_LUT;
            break;
        default: // Q4_0 (codebook 0)
            lut_data = Q4_0_DECODE_LUT;
            break;
        }
        __m128i lut_128 = _mm_load_si128(reinterpret_cast<const __m128i *>(lut_data));
        return _mm512_broadcast_i32x4(lut_128);
    }

    /**
     * @brief AVX-512 VNNI GEMV for one N-chunk of 64 columns.
     *
     * Reads native-interleaved bytes (1024 B/K-block = native payload size),
     * decodes nibbles to INT8 via vpshufb, and feeds directly into vpdpbusd.
     *
     * Per K-block inner loop (4 groups × 2 subs each = 8 sub-iterations):
     *   16 ZMM loads (native data) + 32 vpshufb (decode) + 32 vpdpbusd
     *   Memory traffic: 1024 bytes (native size — zero expansion)
     */
    inline void gemv_native_vnni_avx512_chunk_native(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        const __m512i decode_lut)
    {
        __m512 fp_acc0 = _mm512_setzero_ps();
        __m512 fp_acc1 = _mm512_setzero_ps();
        __m512 fp_acc2 = _mm512_setzero_ps();
        __m512 fp_acc3 = _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);
        const __m512i mask_0F = _mm512_set1_epi8(0x0F);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            __m512i int_acc0 = _mm512_setzero_si512();
            __m512i int_acc1 = _mm512_setzero_si512();
            __m512i int_acc2 = _mm512_setzero_si512();
            __m512i int_acc3 = _mm512_setzero_si512();

            // 4 groups: each loads 4 native bytes per column and extracts lo+hi nibbles.
            // Group g covers native bytes [g*4..g*4+3]:
            //   low  nibbles → K-elements [g*4 .. g*4+3]     (1st sub)
            //   high nibbles → K-elements [g*4+16 .. g*4+19]  (2nd sub)
            for (int group = 0; group < 4; ++group)
            {
                // Load 4 ZMMs of interleaved native bytes (64 cols × 4 bytes)
                __m512i raw0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i raw1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i raw2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i raw3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // Decode low nibbles via vpshufb LUT → signed INT8 in VNNI lane order
                __m512i lo0 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(raw0, mask_0F));
                __m512i lo1 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(raw1, mask_0F));
                __m512i lo2 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(raw2, mask_0F));
                __m512i lo3 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(raw3, mask_0F));

                // A broadcast for low-nibble sub (K-elements group*4..group*4+3)
                uint8_t a_lo[4];
                a_lo[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_lo[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_lo[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_lo[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_lo_i32;
                std::memcpy(&a_lo_i32, a_lo, 4);
                __m512i a_lo_bcast = _mm512_set1_epi32(a_lo_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_lo_bcast, lo0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_lo_bcast, lo1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_lo_bcast, lo2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_lo_bcast, lo3);

                // Decode high nibbles: shift right 4, mask, then vpshufb LUT
                __m512i hi0 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(_mm512_srli_epi16(raw0, 4), mask_0F));
                __m512i hi1 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(_mm512_srli_epi16(raw1, 4), mask_0F));
                __m512i hi2 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(_mm512_srli_epi16(raw2, 4), mask_0F));
                __m512i hi3 = _mm512_shuffle_epi8(decode_lut, _mm512_and_si512(_mm512_srli_epi16(raw3, 4), mask_0F));

                // A broadcast for high-nibble sub (K-elements group*4+16..group*4+19)
                uint8_t a_hi[4];
                a_hi[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 16]) + 128);
                a_hi[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 17]) + 128);
                a_hi[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 18]) + 128);
                a_hi[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 19]) + 128);
                int32_t a_hi_i32;
                std::memcpy(&a_hi_i32, a_hi, 4);
                __m512i a_hi_bcast = _mm512_set1_epi32(a_hi_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_hi_bcast, hi0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_hi_bcast, hi1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_hi_bcast, hi2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_hi_bcast, hi3);
            }

            // Bias correction: corrected = int_acc - 128 * comp
            // comp is INT16 (halved metadata); load via sign-extend to INT32
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i comp0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i comp1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i comp2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i comp3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            int_acc0 = _mm512_sub_epi32(int_acc0, _mm512_mullo_epi32(bias_128_i32, comp0));
            int_acc1 = _mm512_sub_epi32(int_acc1, _mm512_mullo_epi32(bias_128_i32, comp1));
            int_acc2 = _mm512_sub_epi32(int_acc2, _mm512_mullo_epi32(bias_128_i32, comp2));
            int_acc3 = _mm512_sub_epi32(int_acc3, _mm512_mullo_epi32(bias_128_i32, comp3));

            // Convert to FP32 and scale: fp_val = int32_val * a_scale * b_scale[n]
            // scales are FP16 (halved metadata); load via F16C convert to FP32
            __m512 a_scale_v = _mm512_set1_ps(a_scale);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 cs0 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales))));
            __m512 cs1 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16))));
            __m512 cs2 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32))));
            __m512 cs3 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48))));

            fp_acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc0), cs0, fp_acc0);
            fp_acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc1), cs1, fp_acc1);
            fp_acc2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc2), cs2, fp_acc2);
            fp_acc3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc3), cs3, fp_acc3);

            // Asymmetric correction: acc += a_scale * sum_qs * b_min[n]
            // Formula: weight = scale * int_val + min, so the offset term
            // contributes min * Σ A[k] ≈ min * a_scale * sum_qs per block.
            // mins are FP16; load via F16C convert to FP32
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 a_corr_v = _mm512_set1_ps(static_cast<float>(a_sum) * a_scale);
                fp_acc0 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins))), fp_acc0);
                fp_acc1 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16))), fp_acc1);
                fp_acc2 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32))), fp_acc2);
                fp_acc3 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48))), fp_acc3);
            }
        }

        _mm512_storeu_ps(C, fp_acc0);
        _mm512_storeu_ps(C + 16, fp_acc1);
        _mm512_storeu_ps(C + 32, fp_acc2);
        _mm512_storeu_ps(C + 48, fp_acc3);
    }

    // =========================================================================
    // AVX-512 VNNI GEMV with pre-decoded INT8 weights (non-4-bit formats)
    // =========================================================================
    //
    // For formats that cannot use vpshufb LUT decode (Q5_0, Q5_1, Q6_K, Q3_K,
    // Q2_K, IQ2/3/1 formats), the weight packer pre-decodes to INT8 at pack
    // time and stores them in VNNI-interleaved order:
    //   native_interleaved: [N_chunks][bpr][8 groups][4 ZMMs][64 bytes]
    //
    // Each group covers 4 consecutive K-elements (vs the nibble path which
    // covers 8 K-elements per group via lo/hi nibble split).
    //
    // Memory traffic per K-block: 2048 bytes (1.0 byte/element INT8)
    // This is 2× the native nibble path, but universally applicable.
    // =========================================================================

    /**
     * @brief AVX-512 VNNI GEMV for one N-chunk of 64 columns (INT8 pre-decoded path).
     *
     * Loads pre-decoded INT8 values directly from the interleaved buffer.
     * 8 groups × 4 K-elements per group = 32 K-elements per block.
     */
    inline void gemv_native_vnni_avx512_chunk_int8(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end)
    {
        __m512 fp_acc0 = _mm512_setzero_ps();
        __m512 fp_acc1 = _mm512_setzero_ps();
        __m512 fp_acc2 = _mm512_setzero_ps();
        __m512 fp_acc3 = _mm512_setzero_ps();

        const __m512i bias_128_i32 = _mm512_set1_epi32(128);

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const Q8_1Block &a_blk = A_q8[kb];
            float a_scale = simd::fp16_to_fp32(a_blk.d);
            int16_t a_sum = a_blk.sum_qs;

            __m512i int_acc0 = _mm512_setzero_si512();
            __m512i int_acc1 = _mm512_setzero_si512();
            __m512i int_acc2 = _mm512_setzero_si512();
            __m512i int_acc3 = _mm512_setzero_si512();

            // 8 groups × 4 K-elements each = 32 K-elements per block.
            // Each group loads pre-decoded signed INT8 in VNNI-interleaved order.
            for (int group = 0; group < 8; ++group)
            {
                // Load 4 ZMMs of pre-decoded INT8 (16 cols × 4 INT8 values each)
                __m512i b0 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 0));
                __m512i b1 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 1));
                __m512i b2 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 2));
                __m512i b3 = _mm512_load_si512(packed.interleavedB(chunk, kb, group, 3));

                // A broadcast for this group's 4 consecutive K-elements [group*4 .. group*4+3]
                uint8_t a_u8[4];
                a_u8[0] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 0]) + 128);
                a_u8[1] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 1]) + 128);
                a_u8[2] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 2]) + 128);
                a_u8[3] = static_cast<uint8_t>(static_cast<int16_t>(a_blk.qs[group * 4 + 3]) + 128);
                int32_t a_i32;
                std::memcpy(&a_i32, a_u8, 4);
                __m512i a_bcast = _mm512_set1_epi32(a_i32);

                int_acc0 = _mm512_dpbusd_epi32(int_acc0, a_bcast, b0);
                int_acc1 = _mm512_dpbusd_epi32(int_acc1, a_bcast, b1);
                int_acc2 = _mm512_dpbusd_epi32(int_acc2, a_bcast, b2);
                int_acc3 = _mm512_dpbusd_epi32(int_acc3, a_bcast, b3);
            }

            // Bias correction: corrected = int_acc - 128 * comp
            // comp is INT16 (halved metadata); load via sign-extend to INT32
            const int16_t *comp_ptr = packed.chunkComp(chunk, kb);
            __m512i comp0 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr)));
            __m512i comp1 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 16)));
            __m512i comp2 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 32)));
            __m512i comp3 = _mm512_cvtepi16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i *>(comp_ptr + 48)));

            int_acc0 = _mm512_sub_epi32(int_acc0, _mm512_mullo_epi32(bias_128_i32, comp0));
            int_acc1 = _mm512_sub_epi32(int_acc1, _mm512_mullo_epi32(bias_128_i32, comp1));
            int_acc2 = _mm512_sub_epi32(int_acc2, _mm512_mullo_epi32(bias_128_i32, comp2));
            int_acc3 = _mm512_sub_epi32(int_acc3, _mm512_mullo_epi32(bias_128_i32, comp3));

            // Convert to FP32 and scale: fp_val = int32_val * a_scale * b_scale[n]
            // scales are FP16 (halved metadata); load via F16C convert to FP32
            __m512 a_scale_v = _mm512_set1_ps(a_scale);

            const uint16_t *b_scales = packed.chunkScales(chunk, kb);
            __m512 cs0 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales))));
            __m512 cs1 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 16))));
            __m512 cs2 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 32))));
            __m512 cs3 = _mm512_mul_ps(a_scale_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_scales + 48))));

            fp_acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc0), cs0, fp_acc0);
            fp_acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc1), cs1, fp_acc1);
            fp_acc2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc2), cs2, fp_acc2);
            fp_acc3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(int_acc3), cs3, fp_acc3);

            // Asymmetric correction: acc += a_scale * sum_qs * b_min[n]
            // mins are FP16; load via F16C convert to FP32
            if (packed.is_asymmetric)
            {
                const uint16_t *b_mins = packed.chunkMins(chunk, kb);
                __m512 a_corr_v = _mm512_set1_ps(static_cast<float>(a_sum) * a_scale);
                fp_acc0 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins))), fp_acc0);
                fp_acc1 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 16))), fp_acc1);
                fp_acc2 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 32))), fp_acc2);
                fp_acc3 = _mm512_fmadd_ps(a_corr_v, _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i *>(b_mins + 48))), fp_acc3);
            }
        }

        _mm512_storeu_ps(C, fp_acc0);
        _mm512_storeu_ps(C + 16, fp_acc1);
        _mm512_storeu_ps(C + 32, fp_acc2);
        _mm512_storeu_ps(C + 48, fp_acc3);
    }

    /**
     * @brief Multi-chunk GEMV processing a block of consecutive N-chunks.
     *
     * Processes n_block_chunks consecutive 64-column chunks with L2 prefetch.
     */
    inline void gemv_native_vnni_avx512_block(
        const CPUNativeVNNIPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk_start,
        int chunk_count,
        int K_blocks,
        int N,
        const __m512i decode_lut)
    {
        const bool use_nibble_lut = packed.is_nibble_lut;

        for (int ci = 0; ci < chunk_count; ++ci)
        {
            int chunk = chunk_start + ci;
            int n_start = chunk * 64;
            int n_cols = std::min(64, N - n_start);

            // Prefetch next chunk's first group data into L2
            if (ci + 1 < chunk_count)
            {
                _mm_prefetch(reinterpret_cast<const char *>(packed.interleavedB(chunk + 1, 0, 0, 0)),
                             _MM_HINT_T1);
                _mm_prefetch(reinterpret_cast<const char *>(packed.chunkScales(chunk + 1, 0)),
                             _MM_HINT_T1);
            }

            if (use_nibble_lut)
                gemv_native_vnni_avx512_chunk_native(packed, A_q8, C + n_start, chunk, 0, K_blocks, decode_lut);
            else
                gemv_native_vnni_avx512_chunk_int8(packed, A_q8, C + n_start, chunk, 0, K_blocks);

            // Mask out padding columns in tail chunk
            if (n_cols < 64)
            {
                for (int i = n_cols; i < 64; ++i)
                    C[n_start + i] = 0.0f;
            }
        }
    }

#endif // __AVX512F__ && __AVX512VNNI__ && __AVX512BW__

    // =========================================================================
    // Full GEMV dispatcher (M=1) with cache-aware tiling
    // =========================================================================

    inline void gemv_native_vnni(
        const CPUNativeVNNIPackedWeights &packed,
        const float *A_fp32,
        float *C)
    {
        const int N = packed.N;
        const int K = packed.K;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;
        const int N_padded = N_chunks * 64;

        // Step 1: Quantize activations to Q8_1 (thread-local buffer avoids heap alloc per call)
        thread_local std::vector<Q8_1Block> A_q8_tls;
        if (static_cast<int>(A_q8_tls.size()) < K_blocks)
            A_q8_tls.resize(K_blocks);
        Q8_1Block *A_q8 = A_q8_tls.data();
        for (int kb = 0; kb < K_blocks; ++kb)
        {
            int block_start = kb * 32;
            int block_len = std::min(32, K - block_start);
            simd::quantize_single_block(A_fp32 + block_start, A_q8[kb], block_len);
        }

        // Step 2: Compute tile configuration
        int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(N, K, 1, packed.payload_bytes, num_threads);

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        // Build decode LUT once (only used for nibble-LUT formats, but harmless otherwise)
        __m512i decode_lut = packed.is_nibble_lut
                                 ? build_decode_lut(packed.codebook_id)
                                 : _mm512_setzero_si512();
#endif

        // Step 3: K-parallel GEMV when N-parallelism is insufficient.
        // Creates a 2D task grid (chunk, k_tile) so more threads participate.
        // Each thread computes partial sums over a K-block range, then a
        // vectorized reduction phase combines them.
        if (cfg.k_tiles > 1)
        {
            int k_tiles = cfg.k_tiles;
            int k_blocks_per_tile = (K_blocks + k_tiles - 1) / k_tiles;

            // Partial sums buffer: [N_chunks][k_tiles][64] layout
            // Each (chunk, k_tile) writes a contiguous 64-float block.
            // Reduction reads k_tiles × 64 = contiguous per chunk → cache-friendly.
            // No zero-init needed: each slot is fully written by Phase 1 before Phase 2 reads it.
            std::vector<float> partial_sums(static_cast<size_t>(N_chunks) * k_tiles * 64);

            auto do_gemv_kpar = [&]()
            {
                int total_2d = N_chunks * k_tiles;

                // Phase 1: parallel over (chunk, k_tile) 2D grid
#pragma omp for schedule(static)
                for (int task = 0; task < total_2d; ++task)
                {
                    int chunk_idx = task / k_tiles;
                    int kt = task % k_tiles;
                    int kb_start = kt * k_blocks_per_tile;
                    int kb_end = std::min(kb_start + k_blocks_per_tile, K_blocks);
                    float *dest = partial_sums.data() +
                                  (static_cast<size_t>(chunk_idx) * k_tiles + kt) * 64;

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                    if (packed.is_nibble_lut)
                        gemv_native_vnni_avx512_chunk_native(packed, A_q8, dest,
                                                             chunk_idx, kb_start, kb_end, decode_lut);
                    else
                        gemv_native_vnni_avx512_chunk_int8(packed, A_q8, dest,
                                                           chunk_idx, kb_start, kb_end);
#else
                    int n_cols = std::min(64, N - chunk_idx * 64);
                    // Scalar fallback: compute partial K-range
                    gemv_native_vnni_scalar(packed, A_q8, dest, n_cols, K_blocks);
                    // Note: scalar path doesn't support K-range; falls back to full K
#endif
                }
                // implicit barrier

                // Phase 2: vectorized reduction of partial sums into C
#pragma omp for schedule(static)
                for (int chunk_idx = 0; chunk_idx < N_chunks; ++chunk_idx)
                {
                    int n_start = chunk_idx * 64;
                    int n_cols = std::min(64, N - n_start);
                    const float *base = partial_sums.data() +
                                        static_cast<size_t>(chunk_idx) * k_tiles * 64;

#if defined(__AVX512F__)
                    // Sum k_tiles partial vectors using AVX-512 (4 ZMM accumulators × 16 floats = 64)
                    __m512 sum0 = _mm512_loadu_ps(base);
                    __m512 sum1 = _mm512_loadu_ps(base + 16);
                    __m512 sum2 = _mm512_loadu_ps(base + 32);
                    __m512 sum3 = _mm512_loadu_ps(base + 48);
                    for (int kt = 1; kt < k_tiles; ++kt)
                    {
                        const float *src = base + kt * 64;
                        sum0 = _mm512_add_ps(sum0, _mm512_loadu_ps(src));
                        sum1 = _mm512_add_ps(sum1, _mm512_loadu_ps(src + 16));
                        sum2 = _mm512_add_ps(sum2, _mm512_loadu_ps(src + 32));
                        sum3 = _mm512_add_ps(sum3, _mm512_loadu_ps(src + 48));
                    }
                    _mm512_storeu_ps(C + n_start, sum0);
                    _mm512_storeu_ps(C + n_start + 16, sum1);
                    _mm512_storeu_ps(C + n_start + 32, sum2);
                    _mm512_storeu_ps(C + n_start + 48, sum3);
#else
                    for (int j = 0; j < 64; ++j)
                    {
                        float sum = base[j];
                        for (int kt = 1; kt < k_tiles; ++kt)
                            sum += base[kt * 64 + j];
                        if (j < n_cols)
                            C[n_start + j] = sum;
                    }
#endif
                    // Zero out padding columns in tail chunk
                    if (n_cols < 64)
                    {
                        for (int i = n_cols; i < 64; ++i)
                            C[n_start + i] = 0.0f;
                    }
                }
            };

            OMP_WORKSHARE_REGION(do_gemv_kpar);
            return;
        }

        // Step 3b: Standard N-parallel GEMV (when N-parallelism is sufficient)
        auto do_gemv = [&]()
        {
            int n_block_chunks = cfg.n_block_chunks;
            int total_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;

#pragma omp for schedule(static)
            for (int block_idx = 0; block_idx < total_blocks; ++block_idx)
            {
                int chunk_start = block_idx * n_block_chunks;
                int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                gemv_native_vnni_avx512_block(packed, A_q8, C,
                                              chunk_start, chunk_count, K_blocks, N,
                                              decode_lut);
#else
                // Scalar fallback
                for (int ci = 0; ci < chunk_count; ++ci)
                {
                    int chunk = chunk_start + ci;
                    int n_start = chunk * 64;
                    int n_cols = std::min(64, N - n_start);
                    gemv_native_vnni_scalar(packed, A_q8, C + n_start, n_cols, K_blocks);
                }
#endif
            }
        };

        OMP_WORKSHARE_REGION(do_gemv);
    }

    // =========================================================================
    // Full GEMM dispatcher (M>1) with cache-aware tiling
    // =========================================================================

    inline void gemm_native_vnni(
        const CPUNativeVNNIPackedWeights &packed,
        const float *A_fp32,
        float *C,
        int M,
        int ldc)
    {
        const int K = packed.K;
        const int N = packed.N;
        const int K_blocks = packed.blocks_per_row;
        const int N_chunks = (N + 63) / 64;

        int num_threads = omp_get_max_threads();
        NativeVNNITileConfig cfg = computeTileConfig(N, K, M, packed.payload_bytes, num_threads);
        int n_block_chunks = cfg.n_block_chunks;
        int total_n_blocks = (N_chunks + n_block_chunks - 1) / n_block_chunks;
        int m_unroll = cfg.m_unroll;

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
        __m512i decode_lut = packed.is_nibble_lut
                                 ? build_decode_lut(packed.codebook_id)
                                 : _mm512_setzero_si512();
#endif

        auto do_gemm = [&]()
        {
            std::vector<Q8_1Block> A_q8_local(K_blocks);

#pragma omp for collapse(2) schedule(static)
            for (int m_start = 0; m_start < M; m_start += m_unroll)
            {
                for (int block_idx = 0; block_idx < total_n_blocks; ++block_idx)
                {
                    int chunk_start = block_idx * n_block_chunks;
                    int chunk_count = std::min(n_block_chunks, N_chunks - chunk_start);
                    int m_end = std::min(m_start + m_unroll, M);

                    for (int m = m_start; m < m_end; ++m)
                    {
                        // Quantize this row's activations
                        const float *row_a = A_fp32 + m * K;
                        for (int kb = 0; kb < K_blocks; ++kb)
                        {
                            int block_start = kb * 32;
                            int block_len = std::min(32, K - block_start);
                            simd::quantize_single_block(row_a + block_start, A_q8_local[kb], block_len);
                        }

                        float *row_c = C + m * ldc;

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)
                        gemv_native_vnni_avx512_block(packed, A_q8_local.data(), row_c,
                                                      chunk_start, chunk_count, K_blocks, N,
                                                      decode_lut);
#else
                        for (int ci = 0; ci < chunk_count; ++ci)
                        {
                            int chunk = chunk_start + ci;
                            int n_start = chunk * 64;
                            int n_cols = std::min(64, N - n_start);
                            gemv_native_vnni_scalar(packed, A_q8_local.data(), row_c + n_start, n_cols, K_blocks);
                        }
#endif
                    }
                }
            }
        };

        OMP_WORKSHARE_REGION(do_gemm);
    }

} // namespace llaminar2::cpu::native_vnni