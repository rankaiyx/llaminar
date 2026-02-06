/**
 * @file EmbedQ8Block.h
 * @brief Unified int8 embedding block for GPU embedding lookup
 *
 * All quantized embedding tables (Q4_0, Q8_0, Q6_K, IQ4_NL, etc.) are repacked
 * into this common format on the CPU before uploading to GPU. This allows a single
 * GPU kernel to handle every quantization format via the IINT8Unpackable interface.
 *
 * Dequantization: value[i] = qs[i] * half_to_float(d) + half_to_float(m)
 *
 * Memory savings vs FP32 upload path:
 *   FP32: 128 bytes / 32 elements = 4.0 B/elem
 *   EmbedQ8: 36 bytes / 32 elements = 1.125 B/elem  (3.56× smaller)
 *
 *   Qwen2.5-0.5B (151936 × 896):
 *     FP32:    519 MB per device
 *     EmbedQ8: 146 MB per device
 *
 * This file is safe to include from CUDA (.cu) and HIP (.hip) files.
 * No host-side dependencies (no STL, no SIMD, no TensorClasses).
 *
 * @author David Sanftenberg
 */

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define EMBED_Q8_HOST_DEVICE __host__ __device__
#elif defined(__HIPCC__)
#define EMBED_Q8_HOST_DEVICE __host__ __device__
#else
#define EMBED_Q8_HOST_DEVICE
#endif

namespace llaminar2
{

    /**
     * @brief Unified int8 embedding block for GPU lookup
     *
     * 36 bytes per 32 elements. Supports both symmetric (m=0) and
     * asymmetric (m≠0) quantization formats via the IINT8Unpackable repack path.
     *
     * Symmetric formats (Q4_0, Q8_0, Q6_K, IQ4_NL, ...):
     *   value[i] = qs[i] * scale        (m is 0x0000)
     *
     * Asymmetric formats (Q4_1, Q5_1, Q4_K, Q5_K, ...):
     *   value[i] = qs[i] * scale + min
     */
    struct EmbedQ8Block
    {
        uint16_t d;     ///< FP16 scale factor (from IINT8Unpackable::get_block_scale)
        uint16_t m;     ///< FP16 min/offset  (from IINT8Unpackable::get_block_min, 0 for symmetric)
        int8_t qs[32];  ///< 32 int8 values in native range (from IINT8Unpackable::unpack_block_to_int8)

        static constexpr int BLOCK_SIZE = 32;
    };

    static_assert(sizeof(EmbedQ8Block) == 36, "EmbedQ8Block must be 36 bytes");

} // namespace llaminar2

