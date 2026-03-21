/**
 * @file CPUNativeVNNIWeightPacker.h
 * @brief Repacks tensor weights from native quantized formats into a CPU-cache-friendly
 *        layout for NativeVNNI GEMV/GEMM with inline decode + vpdpbusd accumulation.
 *
 * ## Design
 *
 * Unlike the existing CPUQuantisedGemmKernel path (which pre-decodes ALL formats to INT8),
 * this packer preserves the native weight bytes (Q4_0 nibbles, IQ4_NL nibbles, etc.) and
 * reorganizes them for efficient streaming during GEMV/GEMM.
 *
 * The kernel decodes blocks inline at compute time, trading a small decode cost for
 * 2-4× less memory traffic on memory-bound GEMV (M=1).
 *
 * ## Packed Layout
 *
 * Weights are arranged in N-blocks of 64 columns for AVX-512 SIMD:
 *
 * ```
 * payload: [N_chunks][blocks_per_row][64][payload_bytes]
 *   - N_chunks = ceil(N/64)
 *   - blocks_per_row = ceil(K/32)
 *   - payload_bytes = format-specific (Q4_0=16, IQ4_NL=16, Q6_K=24, etc.)
 *
 * scales: [N_chunks][blocks_per_row][64] × FP32
 *   - One scale per 32-element block per column
 *
 * mins: [N_chunks][blocks_per_row][64] × FP32  (only for asymmetric formats)
 *   - One min offset per block per column
 * ```
 *
 * This layout ensures:
 * - Sequential K-block streaming within an N-chunk → good prefetch/cache line use
 * - 64 contiguous scales per K-block → aligned ZMM loads
 * - Native payload preserved → 2× less memory for Q4_0 vs INT8
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "CPUNativeVNNIDecode.h"
#include "tensors/AlignedVector.h"
#include "tensors/FP16Utils.h"
#include "tensors/TensorClasses.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "utils/Logger.h"

namespace llaminar2::cpu::native_vnni
{

    /**
     * @brief CPU-packed native VNNI weights.
     *
     * All buffers are allocated and filled by packWeightsCPUNativeVNNI().
     *
     * ## Layout
     *
     * Weights are stored in two complementary layouts:
     *
     * 1. **payload**: Raw native bytes in [N_chunks][bpr][64][payload_bytes] layout
     *    Used by the scalar reference path.
     *
     * 2. **native_interleaved**: VNNI-interleaved bytes with inline metadata:
     *    `[N_chunks][bpr][ groups×4 ZMMs×64B | 128B comp | 128B scales | 128B mins ]`
     *    Each 64-byte ZMM holds 16 columns × 4 consecutive native payload bytes.
     *    Metadata (comp, scales, mins) is embedded at the end of each K-block
     *    so the entire working set is a single sequential memory stream per thread.
     *    This improves L3 cache utilization for large-K shapes like FFN_Down.
     */
    struct CPUNativeVNNIPackedWeights
    {
        /// Raw native payload bytes in [N_chunks][blocks_per_row][64][payload_bytes] layout
        /// Only populated for nibble-LUT formats (is_nibble_lut == true).
        std::vector<uint8_t> payload;

        /// VNNI-interleaved weight data with inline metadata (64-byte aligned).
        ///
        /// Each K-block contains group data followed by comp, scales, and optionally mins:
        ///   [groups × 4 ZMMs × 64B | 128B comp_int16 | 128B scales_fp16 | 128B mins_fp16]
        ///
        /// For nibble-LUT formats: 4 groups (1024B data) + 256B metadata = 1280B (symmetric)
        /// For INT8 pre-decoded: 8 groups (2048B data) + 256B metadata = 2304B (symmetric)
        /// Asymmetric formats add 128B for mins.
        AlignedVector<uint8_t> native_interleaved;

        /// Flat INT8 buffer used as intermediate during packing (INT8 pre-decoded formats only).
        /// Layout: [N_chunks][blocks_per_row][64][32] int8_t
        /// Freed after interleaving; only retained when keepDecodedBuffer is true (scalar tests).
        std::vector<int8_t> int8_flat;

        /// Original dimensions
        int N = 0;
        int K = 0;

        /// Padded N (rounded up to multiple of 64)
        int N_padded = 0;

        /// K-blocks per row (ceil(K/32))
        int blocks_per_row = 0;

        /// Bytes per native payload block
        int payload_bytes = 0;

        /// Format codebook ID for kernel dispatch
        uint8_t codebook_id = 0;

        /// Whether format has non-zero min offsets
        bool is_asymmetric = false;

        /// Whether format uses 256-element superblocks
        bool is_superblock = false;

        /// Whether this format uses vpshufb nibble LUT decode in the GEMV inner loop.
        /// True for: Q4_0, IQ4_NL, Q4_1, IQ4_XS (4-bit nibble formats).
        /// False for: Q5_0, Q5_1, Q6_K, Q3_K, Q2_K, IQ2/3/1* (pre-decoded INT8).
        bool is_nibble_lut = false;

        /// Bytes of pure group data per K-block (before metadata).
        /// 1024 for nibble-LUT (4 groups × 4 ZMMs × 64 bytes).
        /// 2048 for INT8 pre-decoded (8 groups × 4 ZMMs × 64 bytes).
        int data_stride = 1024;

        /// Total bytes per K-block including inline metadata.
        /// = data_stride + 128 (comp) + 128 (scales) [+ 128 (mins) if asymmetric]
        int interleaved_block_stride = 1280;

        // -------------------------------------------------------------------
        // Accessors for the kernel inner loop
        // -------------------------------------------------------------------

        /// Payload pointer for N-chunk c, K-block kb, column n_local (within chunk)
        inline const uint8_t *blockPayload(int c, int kb, int n_local) const
        {
            size_t offset = ((size_t)c * blocks_per_row * 64 + (size_t)kb * 64 + n_local) * payload_bytes;
            return payload.data() + offset;
        }

        /// Scale for N-chunk c, K-block kb, column n_local (returns FP32, converts from FP16)
        inline float blockScale(int c, int kb, int n_local) const
        {
            return fp16_to_fp32(chunkScales(c, kb)[n_local]);
        }

        /// Min for N-chunk c, K-block kb, column n_local (returns FP32, converts from FP16)
        inline float blockMin(int c, int kb, int n_local) const
        {
            return fp16_to_fp32(chunkMins(c, kb)[n_local]);
        }

        /// Scales pointer for N-chunk c, K-block kb (contiguous 64 FP16 values, inline in native_interleaved)
        inline const uint16_t *chunkScales(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return reinterpret_cast<const uint16_t *>(
                native_interleaved.data() + block_offset + data_stride + 128);
        }

        /// Mins pointer for N-chunk c, K-block kb (contiguous 64 FP16 values, inline in native_interleaved)
        inline const uint16_t *chunkMins(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return reinterpret_cast<const uint16_t *>(
                native_interleaved.data() + block_offset + data_stride + 256);
        }

        /// Payload pointer for N-chunk c, K-block kb (contiguous 64 × payload_bytes)
        inline const uint8_t *chunkPayload(int c, int kb) const
        {
            size_t offset = ((size_t)c * blocks_per_row + kb) * 64 * payload_bytes;
            return payload.data() + offset;
        }

        /// Native-interleaved B data for N-chunk c, K-block kb, group g, ZMM z (0..3)
        /// Returns pointer to 64 bytes: 16 columns × 4 values each.
        /// For nibble-LUT: group ∈ [0,3], each holds 4 native payload bytes.
        /// For INT8 pre-decoded: group ∈ [0,7], each holds 4 INT8 values.
        inline const uint8_t *interleavedB(int c, int kb, int group, int z) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return native_interleaved.data() + block_offset + group * 256 + z * 64;
        }

        /// Pre-computed compensation for N-chunk c, K-block kb (contiguous 64 INT16, inline in native_interleaved)
        inline const int16_t *chunkComp(int c, int kb) const
        {
            size_t block_offset = ((size_t)c * blocks_per_row + kb) * interleaved_block_stride;
            return reinterpret_cast<const int16_t *>(
                native_interleaved.data() + block_offset + data_stride);
        }

        /// Flat INT8 values for N-chunk c, K-block kb, column n_local (32 INT8 values)
        /// Only valid for INT8 pre-decoded formats (is_nibble_lut == false).
        /// Only available if keepDecodedBuffer was true during packing.
        inline const int8_t *blockInt8(int c, int kb, int n_local) const
        {
            size_t offset = ((size_t)c * blocks_per_row * 64 + (size_t)kb * 64 + n_local) * 32;
            return int8_flat.data() + offset;
        }

        /// Release the intermediate INT8 decode buffer to save memory.
        /// The AVX-512 GEMV hot path only uses native_interleaved, not int8_flat.
        /// After calling this, blockInt8() is invalid.
        void releaseDecodedBuffer()
        {
            int8_flat.clear();
            int8_flat.shrink_to_fit();
        }
    };

    /**
     * @brief Pack tensor weights into CPUNativeVNNIPackedWeights.
     *
     * Uses the IINT8Unpackable interface to extract native block data
     * from any supported quantized tensor format.
     *
     * @param weights     Source tensor (must implement IINT8Unpackable + vnniFormatInfo)
     * @param out         Output packed weights
     * @param row_start   First row to pack (for TP slicing), default 0
     * @param row_end     One-past-last row, default -1 (all rows)
     * @return true on success
     */
    inline bool packWeightsCPUNativeVNNI(const TensorBase *weights,
                                         CPUNativeVNNIPackedWeights &out,
                                         int row_start = 0, int row_end = -1)
    {
        // Validate
        int full_N = weights->shape()[0];
        int K = weights->shape()[1];
        if (row_end < 0)
            row_end = full_N;
        if (row_start < 0)
            row_start = 0;
        if (row_end > full_N)
            row_end = full_N;
        if (row_start >= row_end)
        {
            LOG_ERROR("[CPUNativeVNNI] Invalid row range: [" << row_start << ", " << row_end << ")");
            return false;
        }

        // Get IINT8Unpackable interface
        const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
        if (!unpackable)
        {
            LOG_ERROR("[CPUNativeVNNI] Tensor does not implement IINT8Unpackable");
            return false;
        }

        // Get native format info
        const NativeVnniFormatInfo *fmt = unpackable->vnniFormatInfo();
        if (!fmt)
        {
            LOG_ERROR("[CPUNativeVNNI] Tensor does not provide vnniFormatInfo (8-bit formats not supported)");
            return false;
        }

        int N = row_end - row_start;
        int N_padded = (N + 63) / 64 * 64;
        int blocks_per_row = (K + 31) / 32;
        int N_chunks = N_padded / 64;

        out.N = N;
        out.K = K;
        out.N_padded = N_padded;
        out.blocks_per_row = blocks_per_row;
        out.payload_bytes = fmt->payload_bytes;
        out.codebook_id = fmt->codebook_id;
        out.is_asymmetric = fmt->is_asymmetric;
        out.is_superblock = fmt->is_superblock;
        out.is_nibble_lut = is_nibble_lut_format(fmt->codebook_id);
        out.data_stride = out.is_nibble_lut ? 1024 : 2048;
        out.interleaved_block_stride = out.data_stride + 256 + (out.is_asymmetric ? 128 : 0);

        // Temporary per-column metadata arrays used during packing.
        // These get copied inline into native_interleaved during the interleaving pass.
        size_t total_blocks = (size_t)N_chunks * blocks_per_row * 64;
        std::vector<uint16_t> temp_scales(total_blocks, 0);
        std::vector<uint16_t> temp_mins(total_blocks, 0);
        std::vector<int16_t> temp_comp(total_blocks, 0);

        bool use_superblock = (unpackable->superblock_size() == 256);

        if (out.is_nibble_lut)
        {
            // =================================================================
            // NIBBLE-LUT PATH (Q4_0, IQ4_NL, Q4_1, IQ4_XS)
            //
            // Store raw native payload bytes + VNNI-interleaved native bytes.
            // The GEMV kernel decodes nibbles at runtime via vpshufb LUT.
            // Memory: 0.5 byte/element (half of INT8).
            // =================================================================

            out.payload.resize(total_blocks * fmt->payload_bytes, 0);

// Pass 1: Extract scales/mins from superblocks
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                int src_row = row_start + n;
                int chunk = n / 64;
                int n_local = n % 64;
                int kb = 0;

                if (use_superblock)
                {
                    int K_superblocks = blocks_per_row / 8;
                    for (int sb = 0; sb < K_superblocks; ++sb)
                    {
                        int8_t sb_vals[256];
                        float sb_scales[8];
                        float sb_mins[8];
                        unpackable->unpack_superblock_to_int8(src_row, sb, sb_vals, sb_scales, sb_mins);

                        for (int i = 0; i < 8; ++i)
                        {
                            size_t idx = (size_t)chunk * blocks_per_row * 64 + (size_t)(kb + i) * 64 + n_local;
                            temp_scales[idx] = fp32_to_fp16(sb_scales[i]);
                            temp_mins[idx] = fp32_to_fp16(sb_mins[i]);
                        }
                        kb += 8;
                    }
                }
                for (; kb < blocks_per_row; ++kb)
                {
                    size_t idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64 + n_local;
                    temp_scales[idx] = fp32_to_fp16(unpackable->get_block_scale(src_row, kb));
                    temp_mins[idx] = fp32_to_fp16(unpackable->get_block_min(src_row, kb));
                }
            }

            // Pass 2: Pack native payload via packVnniBlock into temp GPU-interleaved buffer
            std::vector<uint8_t> temp_payload((size_t)blocks_per_row * N_padded * fmt->payload_bytes, 0);
            std::vector<uint16_t> temp_scales2((size_t)blocks_per_row * N_padded, 0);
            std::vector<uint16_t> temp_mins2((size_t)blocks_per_row * N_padded, 0);

            VnniPackContext gpu_ctx;
            gpu_ctx.raw_bytes = nullptr;
            gpu_ctx.N = N_padded;
            gpu_ctx.K = K;
            gpu_ctx.blocks_per_row = blocks_per_row;
            gpu_ctx.payload_bytes = fmt->payload_bytes;
            gpu_ctx.payload_array = temp_payload.data();
            gpu_ctx.scales_array = temp_scales2.data();
            gpu_ctx.mins_array = fmt->is_asymmetric ? temp_mins2.data() : nullptr;
            gpu_ctx.emins_array = nullptr;

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    unpackable->packVnniBlock(gpu_ctx, n, kb);
                }
            }

            // Pass 3: Reorganize from GPU interleaved to CPU blocked layout
#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    for (int n_local = 0; n_local < 64; ++n_local)
                    {
                        int n = chunk * 64 + n_local;
                        if (n >= N) continue;
                        size_t gpu_idx = (size_t)kb * N_padded + n;
                        size_t cpu_idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64 + n_local;
                        std::memcpy(out.payload.data() + cpu_idx * fmt->payload_bytes,
                                    temp_payload.data() + gpu_idx * fmt->payload_bytes,
                                    fmt->payload_bytes);
                    }
                }
            }

            // Pass 4: Build VNNI-interleaved B buffer (4 groups) + inline comp/scales/mins
            size_t interleaved_total = (size_t)N_chunks * blocks_per_row * out.interleaved_block_stride;
            out.native_interleaved.resize(interleaved_total, 0);

#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    const uint8_t *chunk_payload = out.chunkPayload(chunk, kb);
                    int n_cols = std::min(64, N - chunk * 64);

                    size_t block_offset = ((size_t)chunk * blocks_per_row + kb) * out.interleaved_block_stride;
                    uint8_t *interleaved_dst = out.native_interleaved.data() + block_offset;

                    for (int group = 0; group < 4; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint8_t *zmm_dst = interleaved_dst + group * 256 + z * 64;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                int col = z * 16 + lane;
                                if (col < n_cols)
                                {
                                    const uint8_t *src = chunk_payload + (size_t)col * fmt->payload_bytes;
                                    zmm_dst[lane * 4 + 0] = src[group * 4 + 0];
                                    zmm_dst[lane * 4 + 1] = src[group * 4 + 1];
                                    zmm_dst[lane * 4 + 2] = src[group * 4 + 2];
                                    zmm_dst[lane * 4 + 3] = src[group * 4 + 3];
                                }
                                else
                                {
                                    zmm_dst[lane * 4 + 0] = 0;
                                    zmm_dst[lane * 4 + 1] = 0;
                                    zmm_dst[lane * 4 + 2] = 0;
                                    zmm_dst[lane * 4 + 3] = 0;
                                }
                            }
                        }
                    }

                    // Write comp inline after group data
                    size_t meta_idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64;
                    int16_t *inline_comp = reinterpret_cast<int16_t *>(interleaved_dst + 1024);
                    for (int col = 0; col < 64; ++col)
                    {
                        if (col < n_cols)
                        {
                            int8_t decoded[32];
                            decode_native_block(out.codebook_id,
                                                chunk_payload + (size_t)col * fmt->payload_bytes,
                                                decoded);
                            int32_t sum = 0;
                            for (int i = 0; i < 32; ++i)
                                sum += decoded[i];
                            inline_comp[col] = static_cast<int16_t>(sum);
                        }
                        else
                        {
                            inline_comp[col] = 0;
                        }
                    }

                    // Write scales inline after comp
                    uint16_t *inline_scales = reinterpret_cast<uint16_t *>(interleaved_dst + 1024 + 128);
                    std::memcpy(inline_scales, &temp_scales[meta_idx], 64 * sizeof(uint16_t));

                    // Write mins inline after scales (if asymmetric)
                    if (out.is_asymmetric)
                    {
                        uint16_t *inline_mins = reinterpret_cast<uint16_t *>(interleaved_dst + 1024 + 256);
                        std::memcpy(inline_mins, &temp_mins[meta_idx], 64 * sizeof(uint16_t));
                    }
                }
            }
        }
        else
        {
            // =================================================================
            // INT8 PRE-DECODED PATH (Q5_0, Q5_1, Q6_K, Q3_K, Q2_K, IQ2/3/1*)
            //
            // Decode to INT8 at pack time, store in VNNI-interleaved layout
            // with 8 groups. The GEMV kernel loads pre-decoded INT8 directly.
            // Memory: 1.0 byte/element.
            // =================================================================

            size_t int8_total = (size_t)N_chunks * blocks_per_row * 64 * 32;
            out.int8_flat.resize(int8_total, 0);

// Single pass: extract scales/mins AND decoded INT8 values
#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                int src_row = row_start + n;
                int chunk = n / 64;
                int n_local = n % 64;
                int kb = 0;

                if (use_superblock)
                {
                    int K_superblocks = blocks_per_row / 8;
                    for (int sb = 0; sb < K_superblocks; ++sb)
                    {
                        int8_t sb_vals[256];
                        float sb_scales[8];
                        float sb_mins[8];
                        unpackable->unpack_superblock_to_int8(src_row, sb, sb_vals, sb_scales, sb_mins);

                        for (int i = 0; i < 8; ++i)
                        {
                            size_t idx = (size_t)chunk * blocks_per_row * 64 + (size_t)(kb + i) * 64 + n_local;
                            temp_scales[idx] = fp32_to_fp16(sb_scales[i]);
                            temp_mins[idx] = fp32_to_fp16(sb_mins[i]);

                            // Store decoded INT8 values
                            size_t flat_offset = idx * 32;
                            std::memcpy(out.int8_flat.data() + flat_offset, sb_vals + i * 32, 32);
                        }
                        kb += 8;
                    }
                }

                // Process remaining blocks individually
                for (; kb < blocks_per_row; ++kb)
                {
                    size_t idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64 + n_local;
                    temp_scales[idx] = fp32_to_fp16(unpackable->get_block_scale(src_row, kb));
                    temp_mins[idx] = fp32_to_fp16(unpackable->get_block_min(src_row, kb));

                    // Decode to INT8
                    size_t flat_offset = idx * 32;
                    unpackable->unpack_block_to_int8(src_row, kb, out.int8_flat.data() + flat_offset);
                }
            }

            // Build VNNI-interleaved INT8 buffer (8 groups) + inline comp/scales/mins
            size_t interleaved_total = (size_t)N_chunks * blocks_per_row * out.interleaved_block_stride;
            out.native_interleaved.resize(interleaved_total, 0);

#pragma omp parallel for schedule(static) collapse(2)
            for (int chunk = 0; chunk < N_chunks; ++chunk)
            {
                for (int kb = 0; kb < blocks_per_row; ++kb)
                {
                    int n_cols = std::min(64, N - chunk * 64);

                    size_t block_offset = ((size_t)chunk * blocks_per_row + kb) * out.interleaved_block_stride;
                    uint8_t *interleaved_dst = out.native_interleaved.data() + block_offset;

                    // Interleave: [8 groups][4 ZMMs][64 bytes]
                    // Group g covers INT8[g*4..g*4+3] for each column
                    for (int group = 0; group < 8; ++group)
                    {
                        for (int z = 0; z < 4; ++z)
                        {
                            uint8_t *zmm_dst = interleaved_dst + group * 256 + z * 64;
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                int col = z * 16 + lane;
                                if (col < n_cols)
                                {
                                    size_t flat_idx = ((size_t)chunk * blocks_per_row * 64 +
                                                       (size_t)kb * 64 + col) * 32;
                                    const int8_t *vals = out.int8_flat.data() + flat_idx;
                                    // Cast to uint8_t for storage; vpdpbusd interprets B as signed
                                    zmm_dst[lane * 4 + 0] = static_cast<uint8_t>(vals[group * 4 + 0]);
                                    zmm_dst[lane * 4 + 1] = static_cast<uint8_t>(vals[group * 4 + 1]);
                                    zmm_dst[lane * 4 + 2] = static_cast<uint8_t>(vals[group * 4 + 2]);
                                    zmm_dst[lane * 4 + 3] = static_cast<uint8_t>(vals[group * 4 + 3]);
                                }
                                else
                                {
                                    zmm_dst[lane * 4 + 0] = 0;
                                    zmm_dst[lane * 4 + 1] = 0;
                                    zmm_dst[lane * 4 + 2] = 0;
                                    zmm_dst[lane * 4 + 3] = 0;
                                }
                            }
                        }
                    }

                    // Write comp inline after group data
                    size_t meta_idx = (size_t)chunk * blocks_per_row * 64 + (size_t)kb * 64;
                    int16_t *inline_comp = reinterpret_cast<int16_t *>(interleaved_dst + 2048);
                    for (int col = 0; col < 64; ++col)
                    {
                        if (col < n_cols)
                        {
                            size_t flat_idx = ((size_t)chunk * blocks_per_row * 64 +
                                               (size_t)kb * 64 + col) * 32;
                            const int8_t *vals = out.int8_flat.data() + flat_idx;
                            int32_t sum = 0;
                            for (int i = 0; i < 32; ++i)
                                sum += vals[i];
                            inline_comp[col] = static_cast<int16_t>(sum);
                        }
                        else
                        {
                            inline_comp[col] = 0;
                        }
                    }

                    // Write scales inline after comp
                    uint16_t *inline_scales = reinterpret_cast<uint16_t *>(interleaved_dst + 2048 + 128);
                    std::memcpy(inline_scales, &temp_scales[meta_idx], 64 * sizeof(uint16_t));

                    // Write mins inline after scales (if asymmetric)
                    if (out.is_asymmetric)
                    {
                        uint16_t *inline_mins = reinterpret_cast<uint16_t *>(interleaved_dst + 2048 + 256);
                        std::memcpy(inline_mins, &temp_mins[meta_idx], 64 * sizeof(uint16_t));
                    }
                }
            }
        }

        // Release the intermediate INT8 decode buffer.
        // The AVX-512 GEMV uses only native_interleaved; int8_flat is dead weight.
        // For FFN_Down (7B): frees ~68 MB, reducing TLB and virtual memory pressure.
        out.releaseDecodedBuffer();

        return true;
    }

} // namespace llaminar2::cpu::native_vnni
