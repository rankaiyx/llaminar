/**
 * @file Q8_1GemmKernel.h
 * @brief Q8_1 × Q8_0 → FP32 GEMM kernel using IQ8_1Decodable interface
 * @author David Sanftenberg
 *
 * Computes C_fp32 = A × B where A is any IActivationTensor (FP32/FP16/BF16/Q8_1)
 * and B is Q8_0 quantized tensor (weights).
 *
 * Key Innovation: Uses IQ8_1Decodable interface to convert activations on-the-fly
 * - FP32/FP16/BF16: Quantize to Q8_1 during GEMM (thread-local blocks)
 * - Q8_1Tensor: Direct pointer access (zero-copy)
 *
 * Q8_1 Format (Activations):
 * - Blocks of 32 int8 values
 * - Per-block FP16 scale (uint16_t d)
 * - Per-block FP16 pre-computed sum (uint16_t s = d × Σ(qs[i]))
 * - Block size: 36 bytes (2 + 2 + 32)
 *
 * Q8_0 Format (Weights):
 * - Blocks of 32 int8 values
 * - Per-block FP16 scale (uint16_t d)
 * - Block size: 34 bytes (2 + 32)
 *
 * Formula: C = (A_scales ⊗ B_scales) ⊙ (A_quants × B_quants - A_sums × 128)
 * Where A_sums are PRE-COMPUTED (eliminates 448 horizontal reductions per layer!)
 *
 * Microkernel: MR×NR (manageable register pressure, 2048 FLOPs per call)
 */

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <omp.h>

#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "utils/DebugEnv.h"
#include "utils/CPUFeatures.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar2
{

    // Forward declarations
    class IActivationTensor;

    /**
     * @class Q8_1GemmKernel
     * @brief High-performance Q8_1 × Q8_0 GEMM using IQ8_1Decodable interface
     *
     * Key Design Decisions:
     * 1. Generic activation input - accepts any IActivationTensor via IQ8_1Decodable
     * 2. Pre-computed sums - Q8_1 blocks include s = d × Σ(qs[i]) (eliminates 448 reductions!)
     * 3. On-the-fly quantization - FP32/FP16/BF16 quantized to Q8_1 during GEMM
     * 4. Zero-copy for Q8_1 - Q8_1Tensor returns direct pointer (no allocation)
     * 5. Parameterized microkernel (MR×NR template) - tunable for different architectures
     * 6. Scale packing - B scales packed alongside quantized values
     * 7. No A-packing - Q8_1 blocks accessed via IQ8_1Decodable (zero overhead for Q8_1Tensor)
     * 8. B packing - transpose + s8→u8 for VNNI layout + scale extraction
     * 9. NC/KC blocking - cache-friendly blocking for large matrices
     *
     * Autotuner Configuration Space:
     * ==============================
     * The template parameters form a configuration space that can be explored by autotuners
     * to find optimal settings for different problem sizes (M, N, K) and hardware platforms.
     *
     * Key Insight: Different optimizations perform better at different problem sizes!
     * - JR_UNROLL=4: +5.5% at M=4096, but -20% at M=512 (register pressure hurts small M)
     * - JR_UNROLL=2: Balanced performance across all M sizes (current default)
     *
     * Autotuner should select configuration based on problem characteristics:
     * - Small M (512-1024): Prefer low register pressure → JR_UNROLL=2, smaller MR/NR
     * - Medium M (2048-4096): Higher ILP beneficial → JR_UNROLL=4 can win
     * - Large M (8192+): Memory bandwidth dominates → JR_UNROLL has minimal impact
     *
     * @tparam MR_PARAM M register blocking (rows per microkernel)
     *   Range: [8, 16, 32] - Sweep found 32 optimal for most cases
     *   Trade-off: Larger = better A reuse, but higher register pressure
     *
     * @tparam NR_PARAM N register blocking (columns per microkernel)
     *   Range: [32, 64, 128] - Sweep found 128 optimal (+5.3% over 64)
     *   Trade-off: Larger = better B panel reuse, but more B data per microkernel
     *
     * @tparam PREFETCH_A_PARAM A block prefetch distance (0-5, default 1)
     *   Range: [0, 1, 2, 4] - Prefetches A blocks kb+PREFETCH_A iterations ahead
     *   Implementation: Uses _mm_prefetch with _MM_HINT_T0 (L1 cache)
     *   Trade-off: Too low = cache misses, too high = pollutes cache
     *   Note: PREFETCH_A=0 disables prefetching (relies on hardware prefetcher)
     *
     * @tparam NC_PARAM N cache blocking (0=auto, default 896 fits L2)
     *   Range: [0, 448, 896, 1792] - 0 means auto-select based on N
     *   Trade-off: Fits working set in L2 cache (B panel + C accumulator)
     *
     * @tparam KC_PARAM K cache blocking in K-blocks (0=auto, default 128 max storage)
     *   Range: [0, 28, 56, 128] - 0 means use all K_blocks (no blocking)
     *   Trade-off: Streaming GEMM removed accum_vec, so KC can now be larger
     *
     * @tparam JR_UNROLL_PARAM Column unrolling factor for dpbusd ILP (1, 2, 4, 6, or 8, default 2)
     *   Range: [1, 2, 4, 6, 8] - Controls inner loop unrolling
     *   Trade-off: Higher = more ILP but higher register pressure (hurts small M)
     *   Performance impact measured (Nov 2025):
     *     M=512:  JR=2: 242 GFLOPS, JR=4: 207 GFLOPS (-14%)
     *     M=4096: JR=2: 424 GFLOPS, JR=4: 470 GFLOPS (+11%)
     *     M=8192: JR=2: 452 GFLOPS, JR=4: 491 GFLOPS (+9%)
     *   Implementation: Template helper function with compile-time unrolling
     *     (eliminates code duplication - all unroll factors use same logic)
     *
     * @tparam JR_BATCH_PARAM Batch size for column reduction (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, or 20, default 18)
     *   Range: [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20] - Controls batched reduction parallelism
     *   Trade-off: Larger = more vectorization in reduction, but higher register pressure
     *   Performance impact:
     *     JR_BATCH=8: Baseline (128 reductions instead of 1024)
     *     JR_BATCH=16: +vectorization but +register pressure
     *     JR_BATCH=4: -vectorization but better for small NR
     *
     * @tparam STREAMING_PARAM Enable streaming accumulation (true=FP32 accumulation in K-loop, false=INT32+post-processing)
     *   Default: false - Original 3-pass algorithm has best overall balance
     *   Note: Streaming reduces L1 misses but causes 20% perf regression (serial dependencies)
     *   Working set: FP32 result[MR][NR] = 16 KB (fits L1!) vs INT32 accum[MR][NR][K_blocks] = 458 KB (14× L1)
     *   Performance: Streaming has 4.5% L1 miss rate but only 360 GFLOPS (original: 45% miss, 456 GFLOPS)
     */
    template <int MR_PARAM = 32, int NR_PARAM = 128, int PREFETCH_A_PARAM = 1,
              int NC_PARAM = 0, int KC_PARAM = 0, int JR_UNROLL_PARAM = 2, int JR_BATCH_PARAM = 18>
    class Q8_1GemmKernelTemplate
    {
    public:
        static constexpr int MR = MR_PARAM;                 // M register blocking
        static constexpr int NR = NR_PARAM;                 // N register blocking
        static constexpr int BLOCK_SIZE = 32;               // Q8_0Block::BLOCK_SIZE
        static constexpr int PREFETCH_A = PREFETCH_A_PARAM; // A prefetch distance
        static constexpr int VECTOR_WIDTH = 16;             // K-blocks per iteration (MUST be 16 for __m512)
        static constexpr int NC = NC_PARAM;                 // N cache blocking (see compute_nc_blocking)
        static constexpr int KC = KC_PARAM;                 // K cache blocking (see compute_kc_blocking)
        static constexpr int JR_UNROLL = JR_UNROLL_PARAM;   // Column unrolling factor (1, 2, 4, 6, or 8)
        static constexpr int JR_BATCH = JR_BATCH_PARAM;     // Batched reduction size (1, 2, 4, 6, ..., 18, 20)

        /**
         * NC_PARAM semantics:
         *   > 0: Explicit NC value (in columns)
         *   < 0: -TARGET_B_SIZE_KB (e.g., -512 = 512KB target, -1024 = 1MB target)
         *   = 0: Auto (default 512KB target)
         *
         * KC_PARAM semantics (POST-STREAMING, no accum_vec constraint):
         *   > 0: Explicit KC value (in K-blocks)
         *   < 0: -MAX_METADATA_KB (e.g., -256 = 256KB metadata limit)
         *   = 0: Auto adaptive (128→256→512 based on K_blocks)
         *
         * Streaming metadata: ~640 bytes/K-block (was 16KB with accum_vec!)
         *   - KC=128:  82KB (L1-fit)
         *   - KC=256: 164KB (L2-fit)
         *   - KC=512: 328KB (L2-friendly)
         */

        /**
         * @brief Packed B panel structure (optimized for 64-byte ZMM loads)
         *
         * Layout for one panel (NR columns):
         * - quants: [kb][jr_pair][64] where jr_pair = jr/2 (2 columns per ZMM)
         * - scales[K_blocks][NR]: FP16 scales for each column/block combination
         *
         * Key optimization: Pack 2 columns per ZMM (no padding waste!)
         * - Columns 0,1 → ZMM[0]: [32 bytes col0][32 bytes col1]
         * - Columns 2,3 → ZMM[1]: [32 bytes col2][32 bytes col3]
         * - Columns 4,5 → ZMM[2]: [32 bytes col4][32 bytes col5]
         * - Columns 6,7 → ZMM[3]: [32 bytes col6][32 bytes col7]
         */
        struct PackedBPanel
        {
            std::vector<uint8_t> quants;  // [K_blocks][NR/2][64] (2 cols per ZMM)
            std::vector<uint16_t> scales; // [K_blocks*NR]
            int K_blocks;

            PackedBPanel(int k_blocks) : K_blocks(k_blocks)
            {
                // Layout: [kb][jr_pair][64 bytes] where each 64 bytes = 2 columns
                // Total: K_blocks × (NR/2) × 64 bytes
                size_t bytes_per_zmm = 64; // 2 columns × 32 bytes (no padding!)
                size_t total_bytes = k_blocks * (NR / 2) * bytes_per_zmm;
                quants.resize(total_bytes, 128); // Fill with zero-value (128 in u8)
                scales.resize(k_blocks * NR);
            }
        };

        // ====================================================================
        // UNIT-TESTABLE HELPER FUNCTIONS
        // ====================================================================

        /**
         * @brief Compute Q8_1 pre-computed sum from block
         * @param a_sum FP16 pre-computed sum (s = d × Σ(qs[i]))
         * @param a_scale FP16 scale factor
         * @return Σ(qs[i]) = s / d
         */
        static inline float compute_sum_qs(uint16_t a_sum_fp16, uint16_t a_scale_fp16)
        {
            float sum_a = fp16_to_fp32(a_sum_fp16);
            float a_scale = fp16_to_fp32(a_scale_fp16);
            // Add epsilon to avoid division by zero for zero blocks
            return sum_a / std::max(a_scale, 1e-10f);
        }

        /**
         * @brief Apply Q8_0 compensation formula
         *
         * Q8_0 uses SYMMETRIC quantization (range [-127, 127]) with NO zero-point offset.
         * Unlike asymmetric quantization schemes, Q8_0 does NOT require 128 compensation.
         *
         * The original formula `accum - sum_qs*128` was incorrect and caused sign errors
         * (208% error on test data). Correct formula is simply: accum (no compensation).
         *
         * @param accum Raw dpbusd accumulator (Σ(a_qs[i] * b_qs[i]))
         * @param sum_qs Sum of A quantized values (Σ(a_qs[i])) - unused for Q8_0
         * @return Unmodified accumulator (Q8_0 symmetric quantization needs no compensation)
         */
        static inline int32_t apply_compensation(int32_t accum, float sum_qs)
        {
            // Q8_0 is symmetric quantization - no compensation needed!
            (void)sum_qs; // Unused for Q8_0, kept for API compatibility
            return accum;
        }

        /**
         * @brief Scale compensated value to FP32
         * @param compensated Compensated int32 value
         * @param a_scale A block scale (FP16 converted to FP32)
         * @param b_scale B block scale (FP16 converted to FP32)
         * @return FP32 result (compensated * a_scale * b_scale)
         */
        static inline float scale_to_fp32(int32_t compensated, float a_scale, float b_scale)
        {
            return static_cast<float>(compensated) * a_scale * b_scale;
        }

        /**
         * @brief Complete Q8_1×Q8_0 block computation (single block pair)
         * @param accum Raw dpbusd result
         * @param a_sum_fp16 Q8_1 pre-computed sum (FP16)
         * @param a_scale_fp16 A block scale (FP16)
         * @param b_scale_fp16 B block scale (FP16)
         * @return FP32 result for this block pair
         */
        static inline float compute_block_result(int32_t accum,
                                                 uint16_t a_sum_fp16,
                                                 uint16_t a_scale_fp16,
                                                 uint16_t b_scale_fp16)
        {
            float sum_qs = compute_sum_qs(a_sum_fp16, a_scale_fp16);
            int32_t compensated = apply_compensation(accum, sum_qs);
            float a_scale = fp16_to_fp32(a_scale_fp16);
            float b_scale = fp16_to_fp32(b_scale_fp16);
            return scale_to_fp32(compensated, a_scale, b_scale);
        }

        /**
         * @brief Compute block index in column-major layout
         * @param row Row index in B (column in original K×N matrix)
         * @param k_block K-block index
         * @param K_blocks Total number of K-blocks
         * @return Linear block index
         */
        static inline size_t block_index_col_major(size_t row, size_t k_block, size_t K_blocks)
        {
            return row * K_blocks + k_block;
        }

        /**
         * @brief Unpack single B column value from packed layout (SCALAR VERSION)
         *
         * Packed layout: [kb][jr_pair][64] where 64 bytes = [col0_bytes0..31][col1_bytes0..31]
         *
         * @param B_quants Packed quantized values
         * @param kb K-block index
         * @param jr Column index (0..NR-1)
         * @param k_in Element within block (0..31)
         * @return Unpacked int8 value (converted from uint8 storage)
         */
        static inline int8_t unpack_B_value(const uint8_t *B_quants, int kb, int jr, int k_in)
        {
            // Determine which jr_pair contains column jr
            int jr_pair = jr / 2;
            int col_offset = (jr % 2) * 32; // 0 for even jr, 32 for odd jr

            // Calculate base address for this jr_pair's 64-byte ZMM
            const uint8_t *zmm_base = B_quants + (kb * (NR / 2) + jr_pair) * 64;

            // Read value and convert from uint8 (0-255) to int8 (-128 to 127)
            uint8_t b_val_u8 = zmm_base[col_offset + k_in];
            return static_cast<int8_t>(b_val_u8 - 128);
        }

        /**
         * @brief Verify B packing/unpacking round-trip correctness
         *
         * Tests that pack_B_panel() followed by unpack_B_value() returns original data.
         *
         * @param original_blocks Original Q8_0 blocks [N][K_blocks] (column-major)
         * @param packed Packed B panel
         * @param N Number of columns
         * @param K_blocks Number of K-blocks
         * @return true if all values match, false otherwise
         */
        static inline bool verify_B_packing(const Q8_0Block *original_blocks,
                                            const PackedBPanel &packed,
                                            int N, int K_blocks)
        {
            for (int jr = 0; jr < N; ++jr)
            {
                for (int kb = 0; kb < K_blocks; ++kb)
                {
                    const Q8_0Block &orig_block = original_blocks[jr * K_blocks + kb];

                    // Verify each quantized value
                    for (int k_in = 0; k_in < BLOCK_SIZE; ++k_in)
                    {
                        int8_t original = orig_block.qs[k_in];
                        int8_t unpacked = unpack_B_value(packed.quants.data(), kb, jr, k_in);

                        if (original != unpacked)
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        /**
         * @brief Compute per-block contribution (SCALAR VERSION for testing vectorized path)
         *
         * This is what the vectorized path should compute for each K-block:
         * contribution = accum[kb] * a_scale[kb] * b_scale[kb]
         *
         * @param accum_kb Accumulator for this K-block (dpbusd result)
         * @param a_scale_fp16 A scale for this K-block (FP16)
         * @param b_scale_fp16 B scale for this K-block (FP16)
         * @return FP32 contribution for this K-block
         */
        static inline float compute_per_block_contribution(int32_t accum_kb,
                                                           uint16_t a_scale_fp16,
                                                           uint16_t b_scale_fp16)
        {
            // Q8_0 symmetric quantization: NO compensation needed!
            int32_t compensated = accum_kb;

            // Scale to FP32
            float a_scale = fp16_to_fp32(a_scale_fp16);
            float b_scale = fp16_to_fp32(b_scale_fp16);

            return static_cast<float>(compensated) * a_scale * b_scale;
        }

        /**
         * @brief Compute GEMM result for single output element (sum across K-blocks)
         *
         * This is the reference implementation that the vectorized path should match.
         *
         * @param accum_values Array of accumulator values [K_blocks]
         * @param a_scales Array of A scales [K_blocks] (FP16)
         * @param b_scales Array of B scales [K_blocks] (FP16)
         * @param K_blocks Number of K-blocks
         * @return FP32 result (sum of per-block contributions)
         */
        static inline float compute_gemm_result_scalar(const int32_t *accum_values,
                                                       const uint16_t *a_scales,
                                                       const uint16_t *b_scales,
                                                       int K_blocks)
        {
            float result = 0.0f;
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                result += compute_per_block_contribution(accum_values[kb], a_scales[kb], b_scales[kb]);
            }
            return result;
        }

        /**
         * @brief Verify vectorized post-processing against scalar reference
         *
         * Tests that the vectorized reduction produces the same result as scalar accumulation.
         * This isolates potential bugs in the SIMD horizontal reduction logic.
         *
         * @param accum_values Accumulator values for one output element [K_blocks]
         * @param a_scales A scales for one row [K_blocks] (FP16)
         * @param b_scales B scales for one column [K_blocks] (FP16)
         * @param K_blocks Number of K-blocks
         * @return true if vectorized and scalar results match within tolerance
         */
        static inline bool verify_vectorized_reduction(const int32_t *accum_values,
                                                       const uint16_t *a_scales,
                                                       const uint16_t *b_scales,
                                                       int K_blocks)
        {
            // Compute scalar reference
            float scalar_result = compute_gemm_result_scalar(accum_values, a_scales, b_scales, K_blocks);

            // Compute vectorized result (simulate what the kernel does)
            float vectorized_result = 0.0f;

            // Process in chunks of 16 (VECTOR_WIDTH for AVX-512)
            constexpr int VECTOR_WIDTH = 16;
            int kb = 0;

            for (; kb + VECTOR_WIDTH <= K_blocks; kb += VECTOR_WIDTH)
            {
                // Load 16 accumulators and convert to FP32
                __m512i accum_i32[1];
                accum_i32[0] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(&accum_values[kb]));
                __m512 accum_f32 = _mm512_cvtepi32_ps(accum_i32[0]);

                // Load 16 A scales (FP16) and convert to FP32
                __m256i a_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a_scales[kb]));
                __m512 a_scales_f32 = _mm512_cvtph_ps(a_scales_fp16);

                // Load 16 B scales (FP16) and convert to FP32
                __m256i b_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b_scales[kb]));
                __m512 b_scales_f32 = _mm512_cvtph_ps(b_scales_fp16);

                // Compute: accum * a_scale * b_scale
                __m512 scaled = _mm512_mul_ps(accum_f32, a_scales_f32);
                scaled = _mm512_mul_ps(scaled, b_scales_f32);

                // Horizontal reduction
                vectorized_result += _mm512_reduce_add_ps(scaled);
            }

            // Scalar tail
            for (; kb < K_blocks; ++kb)
            {
                vectorized_result += compute_per_block_contribution(accum_values[kb], a_scales[kb], b_scales[kb]);
            }

            // Compare with tolerance (FP32 rounding differences expected)
            float abs_diff = std::abs(scalar_result - vectorized_result);
            float rel_diff = abs_diff / std::max(std::abs(scalar_result), 1e-6f);

            return (rel_diff < 1e-5f); // 0.001% tolerance for FP32 rounding
        }

        /**
         * @brief Compute dot product using dpbusd (for testing)
         *
         * Simulates the dpbusd instruction to verify it computes what we expect.
         * dpbusd: dst[i] = src[i] + a.byte[4*i+0]*b.byte[4*i+0] + ... + a.byte[4*i+3]*b.byte[4*i+3]
         * where a is unsigned bytes, b is signed bytes
         */
        static inline int32_t compute_dpbusd_dot_product(const uint8_t *B_unsigned,
                                                         const int8_t *A_signed, int length)
        {
            int32_t sum = 0;
            for (int i = 0; i < length; ++i)
            {
                sum += static_cast<int32_t>(B_unsigned[i]) * static_cast<int32_t>(A_signed[i]);
            }
            return sum;
        }

        /**
         * @brief Compute reference dot product (signed × signed, no dpbusd)
         *
         * This is what the edge case microkernel computes.
         */
        static inline int32_t compute_reference_dot_product(const int8_t *B_signed,
                                                            const int8_t *A_signed, int length)
        {
            int32_t sum = 0;
            for (int i = 0; i < length; ++i)
            {
                sum += static_cast<int32_t>(B_signed[i]) * static_cast<int32_t>(A_signed[i]);
            }
            return sum;
        }

        /**
         * @brief Apply compensation to dpbusd result
         *
         * dpbusd computes Σ((B+128) × A), we want Σ(B × A)
         * So: Σ(B × A) = dpbusd_result - 128 × Σ(A)
         *
         * @param dpbusd_result Raw dpbusd accumulator value
         * @param sum_qs Sum of A quantized values: Σ(qs)
         * @return Compensated result (still in quantized space, needs scaling)
         */
        static inline int32_t apply_dpbusd_compensation(int32_t dpbusd_result, float sum_qs)
        {
            // CRITICAL: sum_qs is ALREADY Σ(A_quantized), not FP32!
            // Do NOT divide by scale here - that's done later
            float compensation = 128.0f * sum_qs;
            return static_cast<int32_t>(static_cast<float>(dpbusd_result) - compensation);
        }

        /**
         * @brief Verify dpbusd + compensation equals reference
         *
         * Tests that: dpbusd((B+128), A) - 128*Σ(A) = Σ(B × A)
         *
         * @param B_signed B values in signed int8 space
         * @param A_signed A values in signed int8 space
         * @param sum_a Q8_1 precomputed sum: s = d × Σ(A_qs) (FP32 space!)
         * @param a_scale Q8_1 scale factor: d
         * @param length Number of elements
         *
         * CRITICAL: In Q8_1 format, sum_a is the FP32-space precomputed sum (s),
         *           NOT the quantized-space sum (Σ(qs)). To get Σ(qs), divide by d.
         */
        static inline bool verify_dpbusd_compensation(const int8_t *B_signed, const int8_t *A_signed,
                                                      float sum_a, float a_scale, int length)
        {
            // Convert B to unsigned for dpbusd
            std::vector<uint8_t> B_unsigned(length);
            for (int i = 0; i < length; ++i)
            {
                B_unsigned[i] = static_cast<uint8_t>(B_signed[i] + 128);
            }

            // Compute dpbusd result
            int32_t dpbusd_result = compute_dpbusd_dot_product(B_unsigned.data(), A_signed, length);

            // CRITICAL: sum_a is FP32-space precomputed sum (s = d × Σ(qs))
            // To get quantized-space sum, divide by scale: Σ(qs) = s / d
            float sum_qs = sum_a / std::max(a_scale, 1e-10f);

            // Apply compensation
            int32_t compensated = apply_dpbusd_compensation(dpbusd_result, sum_qs);

            // Compute reference (what edge case computes)
            int32_t reference = compute_reference_dot_product(B_signed, A_signed, length);

            // They should match exactly (integer arithmetic)
            return compensated == reference;
        }

        /**
         * @brief Main GEMM entry point: C = A × B (IActivationTensor × Q8_0 → FP32)
         *
         * @param M Number of rows in A and C
         * @param N Number of columns in B and C
         * @param K Number of columns in A / rows in B (must be multiple of 32)
         * @param A Activation tensor (any type implementing IActivationTensor/IQ8_1Decodable)
         * @param B Q8_0 quantized weight matrix (K × N)
         * @param C Output FP32 matrix (M × N)
         * @param ldc Leading dimension of C
         *
         * @note K must be a multiple of BLOCK_SIZE (32)
         * @note C must be pre-allocated (M × N floats)
         * @note A is accessed via IQ8_1Decodable interface (on-the-fly quantization for FP32/FP16/BF16)
         */
        static void gemm(int M, int N, int K,
                         const IActivationTensor &A,
                         const Q8_0Tensor &B,
                         float *C, int ldc)
        {
            // Compile-time parameter validation
            static_assert(MR > 0, "MR must be positive");
            static_assert(NR > 0, "NR must be positive");
            static_assert(MR % 8 == 0, "MR must be a multiple of 8 for vectorization");
            static_assert(NR % 2 == 0, "NR must be even (packed in pairs)");
            static_assert(NR % 8 == 0, "NR must be a multiple of 8 for batched reduction");
            static_assert(BLOCK_SIZE == 32, "BLOCK_SIZE must be 32 (Q8_0/Q8_1 format)");
            static_assert(PREFETCH_A >= 0, "PREFETCH_A must be non-negative");
            static_assert(VECTOR_WIDTH == 16, "VECTOR_WIDTH must be 16 (AVX-512)");
            static_assert(JR_UNROLL >= 1 && JR_UNROLL <= 8, "JR_UNROLL must be in range [1, 8]");
            static_assert(JR_UNROLL == 1 || JR_UNROLL == 2 || JR_UNROLL == 4 ||
                              JR_UNROLL == 6 || JR_UNROLL == 8,
                          "JR_UNROLL must be 1, 2, 4, 6, or 8");
            static_assert(NR % JR_UNROLL == 0, "NR must be divisible by JR_UNROLL");
            static_assert(JR_BATCH >= 1 && JR_BATCH <= 20, "JR_BATCH must be in range [1, 20]");
            static_assert(JR_BATCH == 1 || JR_BATCH == 2 || JR_BATCH == 4 || JR_BATCH == 6 ||
                              JR_BATCH == 8 || JR_BATCH == 10 || JR_BATCH == 12 || JR_BATCH == 14 ||
                              JR_BATCH == 16 || JR_BATCH == 18 || JR_BATCH == 20,
                          "JR_BATCH must be 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, or 20");
            static_assert(JR_BATCH <= NR, "JR_BATCH must be less than or equal to NR");
            // Note: NR no longer required to be divisible by JR_BATCH (tail handling added)

            // Validate dimensions
            if (K % BLOCK_SIZE != 0)
            {
                throw std::runtime_error("Q8_1 GEMM: K must be multiple of 32");
            }

            // Diagnostic: log thread count on first call
            static bool first_call = true;
            if (first_call)
            {
                int max_threads = omp_get_max_threads();
#pragma omp parallel
                {
#pragma omp single
                    {
                        int num_threads = omp_get_num_threads();
                        // std::cout << "[Q8_1 GEMM] Using " << num_threads << " threads (max=" << max_threads << ")" << std::endl;
                    }
                }
                first_call = false;
            }

            // Cast A to IQ8_1Decodable interface (required for block access)
            const IQ8_1Decodable *A_decodable = dynamic_cast<const IQ8_1Decodable *>(&A);
            if (!A_decodable)
            {
                throw std::runtime_error("Q8_1 GEMM: A must implement IQ8_1Decodable interface");
            }

            // Profiling: measure phase timings (enabled via env var, evaluated ONCE at startup)
            static const bool enable_profiling = (std::getenv("Q8_0_PROFILE") != nullptr);
            static const bool enable_detailed_profiling = (std::getenv("Q8_0_PROFILE_DETAILED") != nullptr);
            std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_pack_start, t_pack_end, t_gemm_start, t_gemm_end;
            if (enable_profiling)
            {
                t_start = std::chrono::high_resolution_clock::now();
            }

            const int K_blocks = K / BLOCK_SIZE;

            // Determine NC and KC blocking (auto-tune based on cache sizes)
            // NC: L2 cache blocking for N dimension (targets ~512KB of B data in L2)
            // KC: K-dimension blocking limited by storage (MAX_K_BLOCKS=128)
            const int nc_blocks = (NC == 0) ? compute_nc_blocking(N, K_blocks) : NC;
            const int kc_blocks = (KC == 0) ? compute_kc_blocking(K_blocks) : KC;

            // Get raw B block data (weights remain Q8_0)
            // B is structured as: [rows][K_blocks] with Q8_0Block structs
            const Q8_0Block *B_blocks = reinterpret_cast<const Q8_0Block *>(
                B.get_raw_block_at(0, 0));

            // NOTE: A blocks are accessed via IQ8_1Decodable interface (on-the-fly quantization)
            // No direct pointer access needed - we call A_decodable->decode_to_q8_1(row, k_block) per block

            // KC blocking loop (outermost): Process K in chunks for cache efficiency
            for (int kc_start = 0; kc_start < K_blocks; kc_start += kc_blocks)
            {
                const int kc_size = std::min(kc_blocks, K_blocks - kc_start);

                // NC blocking loop: Process N in chunks for L3 cache reuse
                for (int nc_start = 0; nc_start < N; nc_start += nc_blocks)
                {
                    const int nc_size = std::min(nc_blocks, N - nc_start);
                    const int nc_panels = (nc_size + NR - 1) / NR;

                    // Pack B into panels (one per NR columns) - SERIAL
                    // This is only ~2.6% of total work (802K ops vs 30M total)
                    // Serial execution avoids false sharing and is simpler
                    std::vector<PackedBPanel> B_panels;
                    B_panels.reserve(nc_panels);

                    if (enable_profiling && kc_start == 0 && nc_start == 0)
                    {
                        t_pack_start = std::chrono::high_resolution_clock::now();
                    }
                    for (int panel = 0; panel < nc_panels; ++panel)
                    {
                        const int j_start = nc_start + panel * NR;
                        const int panel_width = std::min(NR, N - j_start);

                        B_panels.emplace_back(kc_size);
                        pack_B_panel(panel_width, kc_size,
                                     B_blocks + j_start * K_blocks + kc_start,
                                     B_panels.back());
                    }
                    if (enable_profiling && kc_start == K_blocks - kc_size && nc_start == N - nc_size)
                    {
                        t_pack_end = std::chrono::high_resolution_clock::now();
                    }

                    // M-loop parallelization - this is where 97%+ of the work is
                    if (enable_profiling && kc_start == 0 && nc_start == 0)
                    {
                        t_gemm_start = std::chrono::high_resolution_clock::now();
                    }
#pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < M; i += MR)
                    {
                        const int m_block = std::min(MR, M - i);

                        // N-loop (panels of NR columns)
                        for (int panel = 0; panel < nc_panels; ++panel)
                        {
                            const int j_start = nc_start + panel * NR;
                            const int n_block = std::min(NR, N - j_start);

                            float *C_block = C + i * ldc + j_start;

                            // Call microkernel
                            if (m_block == MR && n_block == NR)
                            {
                                // Fast path: full MR×NR tile
                                // Pass i (row index), kc_start (K-block offset), A_decodable interface
                                microkernel_full(kc_size, i, kc_start, A_decodable,
                                                 B_panels[panel], C_block, ldc);
                            }
                            else
                            {
                                // Edge case: partial tile
                                microkernel_edge(m_block, n_block, kc_size, i, kc_start, A_decodable,
                                                 B_panels[panel], C_block, ldc);
                            }
                        }
                    }
                    if (enable_profiling && kc_start == K_blocks - kc_size && nc_start == N - nc_size)
                    {
                        t_gemm_end = std::chrono::high_resolution_clock::now();
                    }
                } // NC loop
            } // KC loop

            // Print profiling results
            if (enable_profiling)
            {
                auto t_end = std::chrono::high_resolution_clock::now();
                double t_total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                double t_pack_ms = std::chrono::duration<double, std::milli>(t_pack_end - t_pack_start).count();
                double t_gemm_ms = std::chrono::duration<double, std::milli>(t_gemm_end - t_gemm_start).count();

#pragma omp critical
                {
                    std::cout << "[Q8_1 PROFILE] M=" << M << " N=" << N << " K=" << K << std::endl;
                    std::cout << "  B packing:  " << std::fixed << std::setprecision(2)
                              << t_pack_ms << " ms (" << std::setprecision(1)
                              << (100.0 * t_pack_ms / t_total_ms) << "%)" << std::endl;
                    std::cout << "  GEMM:       " << std::fixed << std::setprecision(2)
                              << t_gemm_ms << " ms (" << std::setprecision(1)
                              << (100.0 * t_gemm_ms / t_total_ms) << "%)" << std::endl;
                    std::cout << "  Total:      " << std::fixed << std::setprecision(2)
                              << t_total_ms << " ms" << std::endl;
                    double gflops = (2.0 * M * N * K) / (t_total_ms * 1e6);
                    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                              << gflops << " GFLOPS" << std::endl;
                }
            }
        }

    private:
        /**
         * @brief Compute optimal NC blocking based on L2/L3 cache size
         *
         * Target: Keep NC × KC block of B in L2 cache for maximum reuse
         *
         * CRITICAL OPTIMIZATION (Nov 2025):
         * - OLD: Hardcoded 512KB target (underutilizes cache on server CPUs)
         * - NEW: Auto-detect L2 cache size via CPUFeatures.h
         * - Xeon Gold 6238R: 28MB total L2 (1MB per core × 28 cores)
         *
         * Strategy:
         * - Use 50% of total L2 per socket for B data (rest for A, C, metadata)
         * - Each column × K_blocks = K_blocks × 34 bytes/block (Q8_0Block)
         * - NC = (L2_total / 2) / (K_blocks × 34)
         *
         * Example (Xeon Gold 6238R, K=4096 blocks):
         * - L2_total = 28MB
         * - Target B size = 14MB (50% of L2)
         * - NC = 14MB / (4096 × 34) = 14MB / 139KB = 103 blocks → 3328 columns
         * - OLD limit: 512KB / 139KB = 3.7 blocks → 448 columns (7.4× smaller!)
         *
         * Template parameter NC_PARAM can override:
         * - NC_PARAM > 0: Use explicit value (in columns, not bytes)
         * - NC_PARAM < 0: Use as -TARGET_B_SIZE_KB (e.g., -512 = 512KB target)
         * - NC_PARAM = 0: Use auto-detected L2 cache size (default)
         *
         * @param N Total N dimension
         * @param K_blocks Total K dimension in blocks
         * @return NC blocking size (multiple of NR)
         */
        static int compute_nc_blocking(int N, int K_blocks)
        {
            // Determine target B size based on NC_PARAM
            int target_b_size;
            if constexpr (NC_PARAM < 0)
            {
                // Negative NC_PARAM = -TARGET_B_SIZE_KB (e.g., -512 = 512KB)
                target_b_size = (-NC_PARAM) * 1024;
            }
            else
            {
                // Auto-detect L2 cache and use 50% for B data
                uint32_t l2_total = cpu_l2_cache_total();
                target_b_size = l2_total / 2; // Conservative: leave room for A, C, metadata
            }

            constexpr int BYTES_PER_BLOCK = 34; // Q8_0Block size

            int nc_target = target_b_size / (K_blocks * BYTES_PER_BLOCK);

            // Round down to multiple of NR
            nc_target = (nc_target / NR) * NR;

            // Ensure at least NR, at most N
            nc_target = std::max(NR, std::min(nc_target, N));

            return nc_target;
        }

        /**
         * @brief Compute optimal KC blocking parameter
         *
         * EMPIRICAL FINDINGS (Nov 2024, Xeon Gold 6238R, Qwen 2.5 0.5B Q8_0):
         *   - KC ∈ {128, 256, 512, 896} all deliver ~500 GFLOPS (variance <1%)
         *   - Working set: 616 KB (1.9% of 28MB L2 cache)
         *   - No cache thrashing observed even for KC=K_blocks
         *
         * CONCLUSION: KC choice is NOT cache-limited for typical inference sizes.
         * Strategy below uses adaptive cache analysis to select KC, but empirical
         * testing shows performance is dominated by memory bandwidth and VNNI
         * throughput, not cache capacity.
         *
         * STRATEGY: Use working set size relative to L2 cache capacity
         *
         * Working set components:
         * 1. Microkernel metadata (per MR×NR tile):
         *    - accum_vec: MR × NR × KC × 4 bytes (INT32)
         *    - sum_qs: KC × MR × 2 bytes (INT16, transposed)
         *    - a_scales: MR × KC × 2 bytes (FP16)
         *    - b_scales_f32: NR × KC × 4 bytes (FP32)
         *
         * 2. Packed B panel (NC columns):
         *    - quants: KC × (NC/2) × 64 bytes (packed pairs)
         *    - scales: KC × NC × 2 bytes (FP16)
         *
         * 3. A blocks (MR rows):
         *    - Q8_1Block: MR × KC × 36 bytes (32 qs + 2 d + 2 s)
         *
         * L2 CACHE BUDGET:
         * - Total L2: Detected via CPUFeatures (e.g., 28MB on Xeon Gold 6238R)
         * - Target: Working set ≤ 40% of L2 (conservative)
         *
         * Set LLAMINAR_KC_VERBOSE=1 for detailed cache analysis logging.
         *
         * @see KC_BLOCKING_EMPIRICAL_ANALYSIS.md for full benchmark results
         * - Reports optimal KC for current problem size
         * - Set LLAMINAR_KC_VERBOSE=1 for detailed logging
         *
         * Template parameter KC_PARAM can override:
         *   - KC_PARAM > 0: Use explicit value (in blocks)
         *   - KC_PARAM < 0: Use as -MAX_METADATA_KB (e.g., -256 = 256KB metadata limit)
         *   - KC_PARAM = 0: Use adaptive cache-aware strategy (default)
         *
         * @param K_blocks Total K dimension in blocks
         * @return KC blocking size (optimized for cache)
         */
        static int compute_kc_blocking(int K_blocks)
        {
            if constexpr (KC_PARAM > 0)
            {
                // Explicit KC value provided
                return KC_PARAM;
            }

            if constexpr (KC_PARAM < 0)
            {
                // Negative KC_PARAM = -MAX_METADATA_KB
                // Legacy mode: only count microkernel metadata
                int max_metadata_kb = -KC_PARAM;
                size_t metadata_per_block = MR * sizeof(int16_t) + MR * sizeof(uint16_t) + NR * sizeof(float);
                int kc_limit = (max_metadata_kb * 1024) / metadata_per_block;
                return std::min(kc_limit, K_blocks);
            }

            // ADAPTIVE STRATEGY: Base KC on working set vs L2 cache capacity
            static const bool verbose = (std::getenv("LLAMINAR_KC_VERBOSE") != nullptr);

            // Get L2 cache size
            uint32_t l2_total = cpu_l2_cache_total();
            size_t l2_budget = static_cast<size_t>(l2_total * 0.4); // 40% of L2 for working set

            // Assume NC will use default (512KB B target)
            // For empirical tuning, we'll measure with actual NC later
            constexpr int NC_estimate = 128; // Conservative estimate

            // Find largest KC where working set fits in L2 budget
            int candidate_kc_values[] = {128, 256, 512, 1024, 2048, 4096, 8192};
            int selected_kc = 256; // Conservative default

            for (int kc_candidate : candidate_kc_values)
            {
                if (kc_candidate > K_blocks)
                    break;

                size_t working_set = compute_working_set_size(kc_candidate, NC_estimate);

                if (working_set <= l2_budget)
                {
                    selected_kc = kc_candidate;

                    if (verbose)
                    {
                        static bool first_log = true;
                        if (first_log)
                        {
                            std::cout << "[KC_BLOCKING] K_blocks=" << K_blocks
                                      << ", L2=" << (l2_total / 1024 / 1024) << "MB"
                                      << ", Budget=" << (l2_budget / 1024) << "KB" << std::endl;
                            std::cout << "[KC_BLOCKING] KC=" << kc_candidate
                                      << ", Working set=" << (working_set / 1024) << "KB"
                                      << " (" << (100.0 * working_set / l2_total) << "% of L2)" << std::endl;
                            first_log = false;
                        }
                    }
                }
                else
                {
                    // Working set exceeds budget, use previous KC
                    break;
                }
            }

            return selected_kc;
        }

    public:
        /**
         * @brief Compute actual working set size for given KC value (PUBLIC for benchmarking)
         *
         * Calculates the total memory footprint during microkernel execution:
         * - Microkernel metadata (sum_qs, a_scales, b_scales_f32, accum)
         * - Packed B panel data
         * - A block footprint (approximate, depends on tensor type)
         *
         * @param KC_val KC blocking value
         * @param NC_val NC blocking value
         * @return Total working set size in bytes
         */
        static size_t compute_working_set_size(int KC_val, int NC_val)
        {
            // Microkernel metadata per tile (MR × NR)
            size_t sum_qs_size = MR * KC_val * sizeof(int16_t);     // INT16 transposed
            size_t a_scales_size = MR * KC_val * sizeof(uint16_t);  // FP16
            size_t b_scales_f32_size = NR * KC_val * sizeof(float); // FP32 converted
            size_t accum_size = MR * NR * KC_val * sizeof(int32_t); // INT32 3D array

            size_t microkernel_metadata = sum_qs_size + a_scales_size + b_scales_f32_size + accum_size;

            // Packed B panel (NC columns × KC blocks)
            size_t b_panel_quants = KC_val * (NC_val / 2) * 64;         // Packed layout
            size_t b_panel_scales = KC_val * NC_val * sizeof(uint16_t); // FP16 scales
            size_t b_panel_total = b_panel_quants + b_panel_scales;

            // A block footprint (approximate - MR rows × KC blocks)
            // Q8_1Block = 32 bytes qs + 2 bytes d + 2 bytes s = 36 bytes
            size_t a_blocks_size = MR * KC_val * 36;

            return microkernel_metadata + b_panel_total + a_blocks_size;
        }

        // Make pack_B_panel public for unit testing
        /**
         * @brief Pack B panel into column-major layout with padding
         *
         * Input:  B_blocks[panel_width][K_blocks] (Q8_0Block structs)
         * Output: packed.quants[K_blocks][NR][64] (each column = 32 data + 32 padding)
         *         packed.scales[NR][K_blocks] (FP16, TRANSPOSED for sequential access)
         *
         * Layout enables direct ZMM loads: quants[kb * NR * 64 + jr * 64] loads column jr of block kb
         *
         * OPTIMIZATION: B scales stored as [jr][kb] instead of [kb][jr] for stride-1 vectorization
         */
        static void pack_B_panel(int panel_width, int K_blocks,
                                 const Q8_0Block *B_blocks,
                                 PackedBPanel &packed)
        {

            uint8_t *quants_base = packed.quants.data();
            uint16_t *scales_base = packed.scales.data();

            // Process each K-block
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                // Extract scales for this block - TRANSPOSED layout [jr][kb]
                for (int jr = 0; jr < panel_width; ++jr)
                {
                    const Q8_0Block &block = B_blocks[jr * K_blocks + kb];
                    scales_base[jr * K_blocks + kb] = block.d; // Transposed!
                }
                // Zero-fill scales for partial panels
                for (int jr = panel_width; jr < NR; ++jr)
                {
                    scales_base[jr * K_blocks + kb] = 0; // Zero scale
                }

                // Pack quantized values: [kb][jr_pair][64] where 64 bytes = 2 columns
                // Layout: [col0_bytes0..31][col1_bytes0..31] (no padding waste!)
                for (int jr_pair = 0; jr_pair < NR / 2; ++jr_pair)
                {
                    int jr0 = jr_pair * 2;     // First column in pair
                    int jr1 = jr_pair * 2 + 1; // Second column in pair

                    // Get ZMM base for this pair
                    uint8_t *zmm_base = quants_base + (kb * (NR / 2) + jr_pair) * 64;

                    // Pack first column (bytes 0..31)
                    if (jr0 < panel_width)
                    {
                        const Q8_0Block &block = B_blocks[jr0 * K_blocks + kb];
                        for (int k_in = 0; k_in < BLOCK_SIZE; ++k_in)
                        {
                            zmm_base[k_in] = static_cast<uint8_t>(block.qs[k_in] + 128);
                        }
                    }
                    // else: already initialized to 128 (zero)

                    // Pack second column (bytes 32..63)
                    if (jr1 < panel_width)
                    {
                        const Q8_0Block &block = B_blocks[jr1 * K_blocks + kb];
                        for (int k_in = 0; k_in < BLOCK_SIZE; ++k_in)
                        {
                            zmm_base[32 + k_in] = static_cast<uint8_t>(block.qs[k_in] + 128);
                        }
                    }
                    // else: already initialized to 128 (zero)
                }
            }
        }

        /**
         * @brief Full MR×NR microkernel with sum_qs-based compensation (ORIGINAL - 3-pass)
         *
         * This is the original production implementation using sum_qs = sA/dA reconstruction.
         * K-loop computes sum_qs as INT16, post-processing applies compensation in quantized space.
         *
         * PERFORMANCE NOTE: This implementation suffers from L1 cache thrashing (45% miss rate)
         * due to the large accum_vec buffer (458 KB for K=896, 14× larger than L1 cache).
         * Use microkernel_streaming instead for 2-3× better performance.
         *
         * @param K_blocks Number of K blocks to process
         * @param i_base Row offset in A matrix
         * @param kc_start K-block offset
         * @param A_decodable Pointer to IQ8_1Decodable interface for A matrix
         * @param B_packed Packed B panel
         * @param C Output matrix pointer
         * @param ldc Leading dimension of C
         */
        __attribute__((always_inline)) static void microkernel_full_sumqs(
            int K_blocks,
            int i_base,   // Row offset in A matrix
            int kc_start, // K-block offset
            const IQ8_1Decodable *A_decodable,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            // Thread-local profiling accumulators (only allocated if detailed profiling enabled)
            thread_local static struct
            {
                double t_buffer_init_ms = 0.0;
                double t_k_loop_load_a_ms = 0.0;
                double t_k_loop_load_b_ms = 0.0;
                double t_k_loop_compute_ms = 0.0;
                double t_postprocess_ms = 0.0;
                int64_t call_count = 0;
            } perf_stats;

            static const bool enable_detailed_profiling = (std::getenv("Q8_0_PROFILE_DETAILED") != nullptr);

            auto t_microkernel_start = std::chrono::high_resolution_clock::now();

            // CORRECTED: Store per-block accumulations (not summed across blocks)
            // accum[ir][jr][kb] = dot product for row ir, col jr, K-block kb
            // sum_qs[kb][ir] = Σ(qs[i]) as INT16 for row ir, K-block kb (extracted from Q8_1 FP16 sum)
            //                  OPTIMIZATION: Convert FP16→INT16 in K-loop (eliminates post-processing overhead)
            //                  LAYOUT: TRANSPOSED to [K_blocks][MR] for contiguous vector stores (eliminates scatter!)
            //                  STORAGE: INT16 (range [-4096, 4096]) - halves memory bandwidth vs INT32

            auto t_buffer_init_start = std::chrono::high_resolution_clock::now();

            // OPTIMIZATION: Dynamic heap allocation (was thread-local static, but caused stack overflow)
            // With large microkernels (48×48), thread-local static arrays exceeded thread stack limits
            // Heap allocation adds ~1μs overhead but prevents crashes
            // For small microkernels (8×8-16×16), this is still L1-cacheable

            // Allocate buffers dynamically based on actual K_blocks needed
            std::vector<int32_t> accum_vec(MR * NR * K_blocks, 0);
            // CRITICAL FIX: Add padding for gather instruction overread
            // The int32 gather reads 4 bytes (2 int16 values), so when ir=MR-1 it reads sum_qs(kb, MR-1) and sum_qs(kb, MR)
            // We need space for K_blocks*MR elements PLUS MR extra elements for safe overread
            std::vector<int16_t> sum_qs_vec(K_blocks * MR + MR, 0); // INT16: TRANSPOSED layout [K_blocks][MR], halves bandwidth
            std::vector<uint16_t> a_scales_vec(MR * K_blocks);

            int32_t *accum_storage = accum_vec.data();
            int16_t *sum_qs_storage = sum_qs_vec.data();
            uint16_t *a_scales_storage = a_scales_vec.data();

            auto t_buffer_init_end = std::chrono::high_resolution_clock::now();

            if (enable_detailed_profiling)
            {
                perf_stats.t_buffer_init_ms += std::chrono::duration<double, std::milli>(t_buffer_init_end - t_buffer_init_start).count();
            }

            // OPTIMIZATION: Allocate fp32 B scale storage for fused conversion
            // Convert B scales from fp16→fp32 ONCE (eliminates post-processing conversions)
            // Note: NOT converting A scales - doing so in K-loop adds 60% overhead to Load A phase
            // CRITICAL FIX: Add padding for AVX-512 vectorized load overread
            // The _mm512_loadu_ps loads 16 floats, so when jr=NR-1 and kb>0,
            // it reads b_scales_f32(jr, kb) through b_scales_f32(jr, kb+15),
            // which crosses row boundaries: jr*K_blocks+kb+15 can exceed NR*K_blocks-1
            // Example: NR=128, K_blocks=16, jr=127, kb=1 → index 127*16+1+15=2048 (out of bounds for size 2048)
            // We need space for NR*K_blocks elements PLUS VECTOR_WIDTH-1 extra elements for safe overread
            std::vector<float> b_scales_f32_vec(NR * K_blocks + VECTOR_WIDTH);
            float *b_scales_f32_storage = b_scales_f32_vec.data();

            // Helper lambdas for 3D indexing
            auto accum = [&](int ir, int jr, int kb) -> int32_t &
            {
                return accum_storage[ir * NR * K_blocks + jr * K_blocks + kb];
            };
            auto sum_qs = [&](int kb, int ir) -> int16_t &
            {
                // TRANSPOSED layout: [K_blocks][MR] for contiguous vector stores
                // INT16 storage: range [-4096, 4096] for 32 signed int8 values
                return sum_qs_storage[kb * MR + ir];
            };
            auto a_scales = [&](int ir, int kb) -> uint16_t &
            {
                return a_scales_storage[ir * K_blocks + kb];
            };
            auto b_scales_f32 = [&](int jr, int kb) -> float &
            {
                return b_scales_f32_storage[jr * K_blocks + kb];
            };

            const __m512i ones_u8 = _mm512_set1_epi8(1);
            const uint8_t *B_quants = B_packed.quants.data();
            const uint16_t *B_scales = B_packed.scales.data();

            // PHASE 2 REGISTER TILING: Keep VNNI accumulators in registers across K-loop
            // Declare register tile: MR rows × NR columns of 512-bit accumulators
            // This eliminates per-kb horizontal reductions (reduces from MR×NR×K_blocks to MR×NR total)
            __m512i tile_accum_regs[MR][NR];
            for (int ir = 0; ir < MR; ++ir)
            {
                for (int jr = 0; jr < NR; ++jr)
                {
                    tile_accum_regs[ir][jr] = _mm512_setzero_si512();
                }
            }

            // Timing accumulators for K-loop phases
            double t_load_a_accum = 0.0;
            double t_load_b_accum = 0.0;
            double t_compute_accum = 0.0;

            // K-loop over blocks
            // (each block has different scales, can't accumulate across blocks in registers)
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                auto t_load_a_start = std::chrono::high_resolution_clock::now();

                // OPTIMIZATION: Prefetch A blocks for next iteration
                // Prefetch blocks PREFETCH_A iterations ahead to hide memory latency
                // _MM_HINT_T0 = L1 cache (temporal locality expected)
                // Only prefetch if we have future blocks to prefetch
                if constexpr (PREFETCH_A > 0)
                {
                    const int kb_prefetch = kb + PREFETCH_A;
                    if (kb_prefetch < K_blocks)
                    {
                        // Prefetch all MR rows for the future block
                        // Strategy: Prefetch the Q8_1Block structure which contains:
                        //   - 32 bytes of qs (int8 quantized values)
                        //   - 2 bytes d (FP16 scale)
                        //   - 2 bytes s (FP16 sum)
                        // Modern CPUs fetch 64-byte cache lines, so one prefetch per block suffices
                        for (int ir = 0; ir < MR; ++ir)
                        {
                            const Q8_1Block *prefetch_ptr = A_decodable->decode_to_q8_1(i_base + ir, kc_start + kb_prefetch);
                            _mm_prefetch(reinterpret_cast<const char *>(prefetch_ptr), _MM_HINT_T0);
                        }
                    }
                }

                // Load A blocks via IQ8_1Decodable interface (supports all activation types)
                __m512i a_vec[MR];

                // PHASE 1: Load all MR blocks and extract metadata
                // (Can't vectorize decode_to_q8_1 calls - may do on-the-fly quantization)
                //
                // CRITICAL OPTIMIZATION (Nov 2024): sum_qs is now precomputed as INT16 during quantization!
                // We no longer compute sum_a / a_scale here - just load block.sum_qs directly.
                // This eliminates FP16→FP32 conversion, division, rounding, and packing from K-loop.

                // OPTIMIZATION: Unroll by 2 to issue parallel 256-bit loads (exploit ILP)
                // Modern CPUs have 2-3 load units and can execute multiple loads simultaneously
                {
                    int ir = 0;
                    for (; ir + 1 < MR; ir += 2)
                    {
                        // Decode both blocks
                        const Q8_1Block *a_block_ptr0 = A_decodable->decode_to_q8_1(i_base + ir, kc_start + kb);
                        const Q8_1Block *a_block_ptr1 = A_decodable->decode_to_q8_1(i_base + ir + 1, kc_start + kb);
                        const Q8_1Block &a_block0 = *a_block_ptr0;
                        const Q8_1Block &a_block1 = *a_block_ptr1;

                        // Extract precomputed INT16 sum_qs (zero FP overhead!)
                        sum_qs(kb, ir) = a_block0.sum_qs;
                        sum_qs(kb, ir + 1) = a_block1.sum_qs;

                        // Extract FP16 scales
                        a_scales(ir, kb) = a_block0.d;
                        a_scales(ir + 1, kb) = a_block1.d;

                        // Load both qs arrays in parallel (2 independent 256-bit loads)
                        // CPU can issue these simultaneously to different load units
                        __m256i a_ymm0 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(a_block0.qs));
                        __m256i a_ymm1 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(a_block1.qs));

                        // Zero-extend to 512 bits (zero-latency casts)
                        a_vec[ir] = _mm512_castsi256_si512(a_ymm0);
                        a_vec[ir + 1] = _mm512_castsi256_si512(a_ymm1);
                    }

                    // Scalar tail (handle odd MR)
                    for (; ir < MR; ++ir)
                    {
                        // Decode to Q8_1 format (zero-copy for Q8_1Tensor, on-the-fly for FP32/FP16/BF16)
                        const Q8_1Block *a_block_ptr = A_decodable->decode_to_q8_1(i_base + ir, kc_start + kb);
                        const Q8_1Block &a_block = *a_block_ptr;

                        // Extract precomputed INT16 sum_qs and FP16 scale
                        sum_qs(kb, ir) = a_block.sum_qs;
                        a_scales(ir, kb) = a_block.d;

                        // Load 32 bytes (256 bits) and zero-extend to 512 bits
                        __m256i a_ymm = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(a_block.qs));
                        // Zero-extend to 512 bits (upper 256 bits = 0)
                        a_vec[ir] = _mm512_castsi256_si512(a_ymm);
                    }
                }

                // PHASE 2 ELIMINATED: No more sum_qs computation!
                // The entire 200+ line vectorized sum_qs extraction is gone.
                // sum_qs is now populated by simple loads above (INT16, no FP math).

                auto t_load_a_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_load_a_accum += std::chrono::duration<double, std::milli>(t_load_a_end - t_load_a_start).count();
                }

                auto t_load_b_start = std::chrono::high_resolution_clock::now();

                // NOTE: B prefetch disabled - sequential access triggers hardware prefetcher naturally
                // Manual prefetch increased L1 miss rate from 35.65% to 36.08% (cache pollution)

                // Load B blocks for NR columns - OPTIMIZED: Parallel 2× 256-bit loads (exploit dual load ports!)
                // Layout: [kb][jr_pair][64] where 64 bytes = [col0_32bytes][col1_32bytes]
                // Instead of loading 512 bits then extracting, issue 2 parallel 256-bit loads
                __m512i b_vec[NR];
                const uint8_t *B_block_base = B_quants + kb * (NR / 2) * 64;

                for (int jr_pair = 0; jr_pair < NR / 2; ++jr_pair)
                {
                    // Base pointer for this column pair
                    const uint8_t *pair_base = B_block_base + jr_pair * 64;

                    // PARALLEL LOADS: CPU issues both simultaneously to AGU0/AGU1!
                    // Load first column (bytes 0-31, independent address)
                    __m256i col0_ymm = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(pair_base));

                    // Load second column (bytes 32-63, independent address)
                    __m256i col1_ymm = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(pair_base + 32));

                    // Zero-extend to 512 bits (zero-latency casts)
                    b_vec[jr_pair * 2] = _mm512_castsi256_si512(col0_ymm);
                    b_vec[jr_pair * 2 + 1] = _mm512_castsi256_si512(col1_ymm);
                }

                auto t_load_b_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_load_b_accum += std::chrono::duration<double, std::milli>(t_load_b_end - t_load_b_start).count();
                }

                auto t_compute_start = std::chrono::high_resolution_clock::now();

                // Accumulate this kb's contribution to the tile accumulators
                // NOTE: sum_A NO LONGER COMPUTED - use pre-computed Q8_1 sums instead!
                // Q8_1 format: s = d × Σ(qs[i]), so Σ(qs[i]) = s / d
                for (int ir = 0; ir < MR; ++ir)
                {
                    for (int jr = 0; jr < NR; ++jr)
                    {
                        // Accumulate this kb's VNNI result to register accumulator
                        tile_accum_regs[ir][jr] = _mm512_dpbusd_epi32(tile_accum_regs[ir][jr], b_vec[jr], a_vec[ir]);
                    }
                }

                auto t_compute_end = std::chrono::high_resolution_clock::now();
                if (enable_detailed_profiling)
                {
                    t_compute_accum += std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
                }
            }

            // PHASE 2 COMPLETE: Reduce register accumulators to accum array (deferred from K-loop)
            // Horizontal reduction happens ONCE per (ir, jr) instead of K_blocks times!
            // This dramatically reduces reduction overhead (from MR×NR×K_blocks to MR×NR)
            auto t_reduction_start = std::chrono::high_resolution_clock::now();
            for (int ir = 0; ir < MR; ++ir)
            {
                for (int jr = 0; jr < NR; ++jr)
                {
                    int32_t total_dot = _mm512_reduce_add_epi32(tile_accum_regs[ir][jr]);
                    // Store to accum array (post-processing expects this layout)
                    // NOTE: We store sum across ALL kb to index 0, rest are unused
                    accum(ir, jr, 0) = total_dot;
                }
            }
            auto t_reduction_end = std::chrono::high_resolution_clock::now();
            // TODO: Add t_k_loop_reduction_ms to perf_stats struct if detailed profiling needed
            // if (enable_detailed_profiling) {
            //     perf_stats.t_k_loop_reduction_ms += std::chrono::duration<double, std::milli>(t_reduction_end - t_reduction_start).count();
            // }

            // Accumulate K-loop phase timings
            if (enable_detailed_profiling)
            {
                perf_stats.t_k_loop_load_a_ms += t_load_a_accum;
                perf_stats.t_k_loop_load_b_ms += t_load_b_accum;
                perf_stats.t_k_loop_compute_ms += t_compute_accum;
            }

            // FUSED CONVERSION: Convert B scales fp16→fp32 (vectorized, eliminates post-processing conversions)
            // B scales layout: [jr][kb] (transposed for post-processing)
            // B_scales already declared above, use B_K_blocks for correct dimension
            const int B_K_blocks = B_packed.K_blocks;

            for (int jr = 0; jr < NR; ++jr)
            {
                int kb = 0;
                // AVX-512 16-wide vectorized conversion
                for (; kb + 16 <= K_blocks; kb += 16)
                {
                    __m256i scales_fp16 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m512 scales_fp32 = _mm512_cvtph_ps(scales_fp16);
                    _mm512_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // AVX2 8-wide vectorized tail
                for (; kb + 8 <= K_blocks; kb += 8)
                {
                    __m128i scales_fp16 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m256 scales_fp32 = _mm256_cvtph_ps(scales_fp16);
                    _mm256_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // AVX2 4-wide vectorized tail
                for (; kb + 4 <= K_blocks; kb += 4)
                {
                    __m128i scales_fp16 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i *>(&B_scales[jr * B_K_blocks + kb]));
                    __m128 scales_fp32 = _mm_cvtph_ps(scales_fp16);
                    _mm_storeu_ps(&b_scales_f32(jr, kb), scales_fp32);
                }

                // Scalar tail (only 0-3 blocks remaining)
                for (; kb < K_blocks; ++kb)
                {
                    b_scales_f32(jr, kb) = fp16_to_fp32(B_scales[jr * B_K_blocks + kb]);
                }
            }

            auto t_postprocess_start = std::chrono::high_resolution_clock::now();

            // Post-processing: OPTIMIZED with batched reductions and B scale pre-conversion
            // Key optimizations:
            // 1. FUSED B CONVERSIONS: B scales already converted to fp32 once (eliminates NR cvtph_ps calls)
            // 2. VECTORIZED A CONVERSIONS: Convert VECTOR_WIDTH A scales per iteration (amortized cost)
            // 3. BATCHED REDUCTIONS: Process jr in chunks of 8, reducing 1024→128 horizontal sums
            //
            // Formula: C[i,j] = Σ_kb ((accum[kb] - 128*sum_qs[kb]) * a_scale[kb] * b_scale[kb])
            // NOTE: dpbusd computes Σ((B+128)*A), so we must subtract 128*Σ(A) to get Σ(B*A)
            //       where Σ(A) = sum_qs = sum_a / a_scale (from Q8_1 pre-computed sum)

            // Process ir rows sequentially (cache-friendly for accum access)
            for (int ir = 0; ir < MR; ++ir)
            {
                // BATCHED REDUCTION: Process jr columns in batches (parameterized via JR_BATCH template)
                // Note: When NR % JR_BATCH != 0, last batch may be partial (handled by actual_batch_size)

                for (int jr_base = 0; jr_base < NR; jr_base += JR_BATCH)
                {
                    // CRITICAL: Calculate actual batch size to avoid accessing jr >= NR
                    // When jr_base + JR_BATCH > NR, we must limit iterations to prevent buffer overruns
                    const int actual_batch_size = std::min(JR_BATCH, NR - jr_base);

                    // Accumulate results for actual_batch_size columns simultaneously
                    __m512 result_vecs[JR_BATCH];
                    for (int jj = 0; jj < actual_batch_size; ++jj)
                    {
                        result_vecs[jj] = _mm512_setzero_ps();
                    }

                    // Process K_blocks in chunks of VECTOR_WIDTH
                    int kb = 0;
                    for (; kb + VECTOR_WIDTH <= K_blocks; kb += VECTOR_WIDTH)
                    {
                        // Load VECTOR_WIDTH sum_qs values (INT16) and convert to INT32, then FP32
                        // OPTIMIZATION: sum_qs stored as INT16 in K-loop (halves memory bandwidth!)
                        // NOTE: With transposed layout [K_blocks][MR], we need to gather across K-blocks for same row
                        // AVX-512 has no native int16 gather, so we use two int32 gathers with scale=2
                        __m256i gather_indices_256 = _mm256_setr_epi32(
                            0 * MR, 1 * MR, 2 * MR, 3 * MR, 4 * MR, 5 * MR, 6 * MR, 7 * MR);

                        // Gather first 8 int32 values (each contains 2 int16s, we want lower 16 bits)
                        __m256i sum_qs_i32_lo_raw = _mm256_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb, ir)), gather_indices_256, 2);
                        __m256i sum_qs_i32_lo = _mm256_srai_epi32(_mm256_slli_epi32(sum_qs_i32_lo_raw, 16), 16);

                        // Gather second 8 int32 values
                        __m256i sum_qs_i32_hi_raw = _mm256_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb + 8, ir)), gather_indices_256, 2);
                        __m256i sum_qs_i32_hi = _mm256_srai_epi32(_mm256_slli_epi32(sum_qs_i32_hi_raw, 16), 16);

                        // Combine into 512-bit (16 int32 values)
                        __m512i sum_qs_i32 = _mm512_inserti64x4(_mm512_castsi256_si512(sum_qs_i32_lo), sum_qs_i32_hi, 1);
                        __m512 sum_qs_f32 = _mm512_cvtepi32_ps(sum_qs_i32);

                        // Convert VECTOR_WIDTH A scales fp16→fp32 (vectorized, amortized across JR_BATCH columns)
                        __m256i a_scales_fp16 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i *>(&a_scales(ir, kb)));
                        __m512 a_scales_vec = _mm512_cvtph_ps(a_scales_fp16);

                        // Precompute compensation constant
                        const __m512 compensation_const = _mm512_set1_ps(128.0f);

                        // Process all columns in the batch (respects actual_batch_size)
                        for (int jj = 0; jj < actual_batch_size; ++jj)
                        {
                            int jr = jr_base + jj;

                            // Load VECTOR_WIDTH int32 accumulators and convert to float
                            __m512i accum_i32 = _mm512_loadu_si512(&accum(ir, jr, kb));
                            __m512 accum_f32 = _mm512_cvtepi32_ps(accum_i32);

                            // Load VECTOR_WIDTH fp32 b_scales (already converted!)
                            __m512 b_scales_vec = _mm512_loadu_ps(&b_scales_f32(jr, kb));

                            // COMPENSATION: compensated = accum - sum_qs * 128 (FUSED FMA!)
                            // Now matches Q8_0 exactly (no FP16 conversion, no division, uses fused FMA)
                            __m512 compensated = _mm512_fnmadd_ps(sum_qs_f32, compensation_const, accum_f32); // 4 cycles

                            // Compute: compensated * a_scale * b_scale
                            __m512 scaled = _mm512_mul_ps(compensated, a_scales_vec);
                            scaled = _mm512_mul_ps(scaled, b_scales_vec);

                            // Accumulate to result
                            result_vecs[jj] = _mm512_add_ps(result_vecs[jj], scaled);
                        }
                    }

                    // Horizontal reductions for the batch (reduced count vs non-batched)
                    float results[JR_BATCH];
                    for (int jj = 0; jj < actual_batch_size; ++jj)
                    {
                        results[jj] = _mm512_reduce_add_ps(result_vecs[jj]);
                    }

                    // VECTORIZED TAIL: Process remaining K_blocks in chunks of 8
                    constexpr int TAIL_VEC_WIDTH = 8;
                    for (; kb + TAIL_VEC_WIDTH <= K_blocks; kb += TAIL_VEC_WIDTH)
                    {
                        // Load 8 sum_qs values (INT16) and convert to INT32, then FP32
                        // NOTE: With transposed layout, use AVX2 gather with scale=2 for int16
                        __m256i gather_indices = _mm256_setr_epi32(
                            0 * MR, 1 * MR, 2 * MR, 3 * MR, 4 * MR, 5 * MR, 6 * MR, 7 * MR);
                        // Gather 8 int32 values (each contains 2 int16s, we want lower 16 bits)
                        __m256i sum_qs_i32_raw = _mm256_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb, ir)), gather_indices, 2);
                        // Extract lower 16 bits and sign-extend to 32-bit
                        __m256i sum_qs_i32 = _mm256_srai_epi32(_mm256_slli_epi32(sum_qs_i32_raw, 16), 16);
                        __m256 sum_qs_f32 = _mm256_cvtepi32_ps(sum_qs_i32);

                        // Load 8 A scales (FP16) and convert to FP32
                        __m128i a_scales_fp16 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i *>(&a_scales(ir, kb)));
                        __m256 a_scales_vec = _mm256_cvtph_ps(a_scales_fp16);

                        // Precompute compensation constant
                        const __m256 compensation_const = _mm256_set1_ps(128.0f);

                        // Process all columns in the batch (respects actual_batch_size)
                        for (int jj = 0; jj < actual_batch_size; ++jj)
                        {
                            int jr = jr_base + jj;

                            // Load 8 int32 accumulators and convert to float
                            __m256i accum_i32 = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i *>(&accum(ir, jr, kb)));
                            __m256 accum_f32 = _mm256_cvtepi32_ps(accum_i32);

                            // Load 8 fp32 b_scales (already converted!)
                            __m256 b_scales_vec = _mm256_loadu_ps(&b_scales_f32(jr, kb));

                            // COMPENSATION: compensated = accum - sum_qs * 128 (FUSED FMA!)
                            __m256 compensated = _mm256_fnmadd_ps(sum_qs_f32, compensation_const, accum_f32);

                            // Compute: compensated * a_scale * b_scale
                            __m256 scaled = _mm256_mul_ps(compensated, a_scales_vec);
                            scaled = _mm256_mul_ps(scaled, b_scales_vec);

                            // Horizontal reduction (AVX2-compatible)
                            // Sum 8 floats: [a b c d e f g h] → a+b+c+d+e+f+g+h
                            __m128 lo = _mm256_castps256_ps128(scaled);
                            __m128 hi = _mm256_extractf128_ps(scaled, 1);
                            __m128 sum4 = _mm_add_ps(lo, hi);     // [a+e, b+f, c+g, d+h]
                            __m128 shuf = _mm_movehdup_ps(sum4);  // [b+f, b+f, d+h, d+h]
                            __m128 sum2 = _mm_add_ps(sum4, shuf); // [a+b+e+f, *, c+d+g+h, *]
                            shuf = _mm_movehl_ps(shuf, sum2);     // [c+d+g+h, *, *, *]
                            __m128 sum1 = _mm_add_ss(sum2, shuf); // [a+b+c+d+e+f+g+h, *, *, *]
                            results[jj] += _mm_cvtss_f32(sum1);
                        }
                    }

                    // AVX2 4-wide vectorized tail (process 4 K_blocks at once)
                    constexpr int TAIL_VEC_WIDTH_4 = 4;

                    for (; kb + TAIL_VEC_WIDTH_4 <= K_blocks; kb += TAIL_VEC_WIDTH_4)
                    {
                        // Load 4 sum_qs (INT16) and convert to FP32
                        // NOTE: With transposed layout, use AVX gather intrinsic
                        // Indices: [0*MR, 1*MR, 2*MR, 3*MR] in int16 (multiply by 2 for byte offset)
                        __m128i gather_indices = _mm_setr_epi32(0 * MR, 1 * MR, 2 * MR, 3 * MR);
                        // AVX gather for 4 int32 values (each contains 2 int16s, we want lower 16 bits)
                        __m128i sum_qs_i32_raw = _mm_i32gather_epi32(
                            reinterpret_cast<const int *>(&sum_qs(kb, ir)), gather_indices, 2);
                        // Extract lower 16 bits and sign-extend to 32-bit
                        __m128i sum_qs_i32 = _mm_srai_epi32(_mm_slli_epi32(sum_qs_i32_raw, 16), 16);
                        __m128 sum_qs_f32 = _mm_cvtepi32_ps(sum_qs_i32);

                        // Load 4 A scales (FP16) and convert to FP32
                        // Note: Only loading 64 bits (4×FP16), so use _mm_loadl_epi64
                        __m128i a_scales_fp16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&a_scales(ir, kb)));
                        __m128 a_scales_vec = _mm_cvtph_ps(a_scales_fp16);

                        for (int jj = 0; jj < actual_batch_size; ++jj)
                        {
                            int jr = jr_base + jj;

                            // Load 4 accumulators (INT32) and convert to FP32
                            __m128i accum_i32 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&accum(ir, jr, kb)));
                            __m128 accum_f32 = _mm_cvtepi32_ps(accum_i32);

                            // Load 4 B scales (FP32, already converted!)
                            __m128 b_scales_vec = _mm_loadu_ps(&b_scales_f32(jr, kb));

                            // Compensation: compensated = accum - sum_qs * 128
                            __m128 compensation_const = _mm_set1_ps(128.0f);
                            __m128 compensated = _mm_fnmadd_ps(sum_qs_f32, compensation_const, accum_f32);

                            // Scale: result = compensated * a_scale * b_scale
                            __m128 scaled = _mm_mul_ps(compensated, a_scales_vec);
                            scaled = _mm_mul_ps(scaled, b_scales_vec);

                            // Horizontal reduction: 4→1
                            __m128 shuf = _mm_movehdup_ps(scaled);  // [b, b, d, d]
                            __m128 sum2 = _mm_add_ps(scaled, shuf); // [a+b, *, c+d, *]
                            shuf = _mm_movehl_ps(shuf, sum2);       // [c+d, *, *, *]
                            __m128 sum1 = _mm_add_ss(sum2, shuf);   // [a+b+c+d, *, *, *]
                            results[jj] += _mm_cvtss_f32(sum1);
                        }
                    }

                    // Scalar tail for remaining K_blocks (< 4)
                    for (; kb < K_blocks; ++kb)
                    {
                        // Load pre-extracted sum_qs (INT32) and A scale (FP16)
                        // OPTIMIZATION: sum_qs pre-extracted as INT32 in K-loop (no FP16 conversion, no division!)
                        // Load sum_qs (INT16), convert to INT32 for computation
                        int16_t sum_qs_val_i16 = sum_qs(kb, ir);
                        int32_t sum_qs_val = static_cast<int32_t>(sum_qs_val_i16);
                        float a_scale = fp16_to_fp32(a_scales(ir, kb));

                        for (int jj = 0; jj < actual_batch_size; ++jj)
                        {
                            int jr = jr_base + jj;
                            float b_scale = b_scales_f32(jr, kb); // Already fp32!

                            // COMPENSATION: compensated = accum - sum_qs * 128 (now matches Q8_0!)
                            // dpbusd gives Σ((B+128)*A), we want Σ(B*A) = Σ((B+128)*A) - 128*Σ(A)
                            int32_t accum_val = accum(ir, jr, kb);
                            float compensated = static_cast<float>(accum_val - sum_qs_val * 128);

                            results[jj] += compensated * a_scale * b_scale;
                        }
                    }

                    // Write results to C (only for columns actually processed)
                    for (int jj = 0; jj < actual_batch_size; ++jj)
                    {
                        int jr = jr_base + jj;
                        C[ir * ldc + jr] = results[jj];
                    }
                }

                // NOTE: Tail handling for NR % JR_BATCH != 0 is now handled automatically
                // by actual_batch_size calculation in the main loop above.
                // No separate tail loop needed!
            }

            auto t_postprocess_end = std::chrono::high_resolution_clock::now();

            // Update profiling statistics
            if (enable_detailed_profiling)
            {
                perf_stats.t_postprocess_ms += std::chrono::duration<double, std::milli>(t_postprocess_end - t_postprocess_start).count();
                perf_stats.call_count++;

                // Print statistics every 1000 calls (per thread)
                if (perf_stats.call_count % 1000 == 0)
                {
                    int thread_id = omp_get_thread_num();
                    double total_ms = perf_stats.t_buffer_init_ms + perf_stats.t_k_loop_load_a_ms +
                                      perf_stats.t_k_loop_load_b_ms + perf_stats.t_k_loop_compute_ms +
                                      perf_stats.t_postprocess_ms;

#pragma omp critical
                    {
                        std::cout << "\n[Q8_1 DETAILED PROFILE - Thread " << thread_id << "]" << std::endl;
                        std::cout << "  Calls:              " << perf_stats.call_count << std::endl;
                        std::cout << "  Total time:         " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
                        std::cout << "  Avg per call:       " << std::fixed << std::setprecision(4) << (total_ms / perf_stats.call_count) << " ms" << std::endl;
                        std::cout << "\n  Phase Breakdown:" << std::endl;
                        std::cout << "    Buffer init:      " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_buffer_init_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_buffer_init_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    K-loop Load A:    " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_k_loop_load_a_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_k_loop_load_a_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    K-loop Load B:    " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_k_loop_load_b_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_k_loop_load_b_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    K-loop Compute:   " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_k_loop_compute_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_k_loop_compute_ms / total_ms) << "%)" << std::endl;
                        std::cout << "    Post-process:     " << std::fixed << std::setprecision(2)
                                  << perf_stats.t_postprocess_ms << " ms ("
                                  << std::setprecision(1) << (100.0 * perf_stats.t_postprocess_ms / total_ms) << "%)" << std::endl;

                        // Per-call averages for each phase
                        std::cout << "\n  Per-Call Averages:" << std::endl;
                        std::cout << "    Buffer init:      " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_buffer_init_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    K-loop Load A:    " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_k_loop_load_a_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    K-loop Load B:    " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_k_loop_load_b_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    K-loop Compute:   " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_k_loop_compute_ms / perf_stats.call_count) << " ms/call" << std::endl;
                        std::cout << "    Post-process:     " << std::scientific << std::setprecision(2)
                                  << (perf_stats.t_postprocess_ms / perf_stats.call_count) << " ms/call" << std::endl;
                    }
                }
            }
        } // End microkernel_full_sumqs

        /**
         * @brief Helper: Process N_COLS columns with dpbusd + compensation (compile-time unrolled)
         *
         * This helper eliminates code duplication by template-generating the unrolled loops.
         * All unroll factors (1, 2, 4, 6, 8) use the same implementation with compile-time bounds.
         *
         * @tparam N_COLS Number of columns to process (1, 2, 4, 6, or 8)
         * @tparam ScaleAccessor Type of the lambda for accessing B scales (deduced)
         */
        template <int N_COLS, typename ScaleAccessor>
        __attribute__((always_inline)) static void process_n_columns(
            int jr, int ir, int kb, int ldc,
            float sA, float dA,
            const __m512i *a_vec,
            const __m512i *b_vec,
            const ScaleAccessor &b_scales_f32,
            float *C)
        {
            // Allocate accumulators (compiler will optimize unused ones away)
            __m512i acc_vec[8];
            int32_t dp[8];
            float dB[8];
            float main[8];
            float comp[8];

// Issue all dpbusd operations first (maximize ILP)
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                acc_vec[i] = _mm512_setzero_si512();
                acc_vec[i] = _mm512_dpbusd_epi32(acc_vec[i], b_vec[jr + i], a_vec[ir]);
            }

// Horizontal reductions (can execute in parallel)
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                dp[i] = _mm512_reduce_add_epi32(acc_vec[i]);
            }

// Load B scales and compute compensation
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                dB[i] = b_scales_f32(jr + i, kb);
                main[i] = static_cast<float>(dp[i]) * dA * dB[i];
                comp[i] = 128.0f * sA * dB[i];
            }

// Write results to C
#pragma GCC unroll 8
            for (int i = 0; i < N_COLS; ++i)
            {
                C[ir * ldc + jr + i] += main[i] - comp[i];
            }
        }

        /**
         * @brief Dispatcher: Full MR×NR microkernel (selects implementation based on debugEnv settings)
         *
         * @param K_blocks Number of K blocks to process
         * @param i_base Row offset in A matrix
         * @param kc_start K-block offset
         * @param A_decodable Pointer to IQ8_1Decodable interface for A matrix
         * @param B_packed Packed B panel
         * @param C Output matrix pointer
         * @param ldc Leading dimension of C
         */
        static void microkernel_full(
            int K_blocks,
            int i_base,
            int kc_start,
            const IQ8_1Decodable *A_decodable,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            if (false)
            {
                // Experimental microkernels would go here and dispatch after a check
                // to debugEnv for their feature flag being set
            }
            else
            {
                // DEFAULT: The microkernel with sum_qs extraction
                microkernel_full_sumqs(K_blocks, i_base, kc_start, A_decodable, B_packed, C, ldc);
            }
        }

        /**
         * @brief Edge-case microkernel (partial tiles)
         *
         * Scalar fallback for non-multiple-of-8 dimensions
         */
        /**
         * @brief Edge microkernel: Handles cases where M < MR or N < NR
         *
         * @param m_block Actual number of rows to process (≤ MR)
         * @param n_block Actual number of columns to process (≤ NR)
         * @param K_blocks Number of K blocks (32-element blocks)
         * @param i_base Row offset in A matrix
         * @param kc_start K-block offset
         * @param A_decodable Pointer to IQ8_1Decodable interface for A matrix
         * @param B_packed Packed B panel
         * @param C Output matrix pointer
         * @param ldc Leading dimension of C
         */
        static void microkernel_edge(
            int m_block, int n_block, int K_blocks, int i_base, int kc_start,
            const IQ8_1Decodable *A_decodable,
            const PackedBPanel &B_packed,
            float *C, int ldc)
        {
            const uint8_t *B_quants = B_packed.quants.data();
            const uint16_t *B_scales = B_packed.scales.data();
            const int B_K_blocks = B_packed.K_blocks; // Use packed panel's K dimension

            // Pre-allocate storage for vectorized K-block reduction
            // (amortize allocation cost across all m_block×n_block elements)
            std::vector<int32_t> block_dots_storage(K_blocks);
            std::vector<int32_t> sum_qs_storage(K_blocks); // Q8_1 compensation: sum_qs per block
            std::vector<uint16_t> a_scales_storage(K_blocks);
            std::vector<uint16_t> b_scales_storage(K_blocks);

            for (int i = 0; i < m_block; ++i)
            {
                for (int j = 0; j < n_block; ++j)
                {
                    // PHASE 1: Compute all K-block dot products using dpbusd (VECTORIZED!)
                    // Determine which jr_pair contains column j (needed for B access)
                    int jr_pair = j / 2;
                    int col_offset = (j % 2) * 32; // 0 for even j, 32 for odd j

                    for (int kb = 0; kb < K_blocks; ++kb)
                    {
                        // Decode A block via IQ8_1Decodable interface
                        const Q8_1Block *a_block_ptr = A_decodable->decode_to_q8_1(i_base + i, kc_start + kb);
                        const Q8_1Block &a_block = *a_block_ptr;

                        // Store scales for vectorized reduction
                        a_scales_storage[kb] = a_block.d;
                        b_scales_storage[kb] = B_scales[j * B_K_blocks + kb];

                        // Q8_1 COMPENSATION: Extract precomputed sum_qs (INT16) from Q8_1 block
                        // Nov 2024: sum_qs is now stored as INT16 during quantization, no FP conversion needed!
                        sum_qs_storage[kb] = static_cast<int32_t>(a_block.sum_qs);

                        // OPTIMIZATION: Use dpbusd for 32-element dot product (32 scalar ops → 1 SIMD instruction!)
                        // Calculate base address for this jr_pair's 64-byte ZMM
                        const uint8_t *zmm_base = B_quants + (kb * (NR / 2) + jr_pair) * 64;

                        // Load A vector (32×INT8) and zero-extend to 512 bits
                        __m256i a_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a_block.qs));
                        __m512i a_vec = _mm512_castsi256_si512(a_ymm);

                        // Load B vector (32×UINT8) and zero-extend to 512 bits
                        __m256i b_ymm = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&zmm_base[col_offset]));
                        __m512i b_vec = _mm512_castsi256_si512(b_ymm);

                        // Vectorized dot product using dpbusd
                        __m512i acc = _mm512_setzero_si512();
                        acc = _mm512_dpbusd_epi32(acc, b_vec, a_vec);

                        // Horizontal reduction to scalar
                        block_dots_storage[kb] = _mm512_reduce_add_epi32(acc);
                    }

                    // PHASE 2: Vectorized K-block reduction (16-wide, 8-wide, 4-wide, scalar tail)
                    // Q8_1 Formula: result = Σ_kb ((block_dot[kb] - 128*sum_qs[kb]) * a_scale[kb] * b_scale[kb])
                    // NOTE: dpbusd computes Σ((B+128)*A), so we must subtract 128*Σ(A) to get Σ(B*A)
                    float result = 0.0f;
                    int kb = 0;

                    // 16-wide vectorization (AVX-512)
                    for (; kb + 16 <= K_blocks; kb += 16)
                    {
                        // Load 16 block_dots (INT32) and convert to FP32
                        __m512i dots_i32 = _mm512_loadu_si512(&block_dots_storage[kb]);
                        __m512 dots_f32 = _mm512_cvtepi32_ps(dots_i32);

                        // Load 16 sum_qs (INT32) and convert to FP32
                        __m512i sum_qs_i32 = _mm512_loadu_si512(&sum_qs_storage[kb]);
                        __m512 sum_qs_f32 = _mm512_cvtepi32_ps(sum_qs_i32);

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        const __m512 compensation_const = _mm512_set1_ps(128.0f);
                        __m512 compensated = _mm512_fnmadd_ps(sum_qs_f32, compensation_const, dots_f32);

                        // Load 16 A scales (FP16) and convert to FP32
                        __m256i a_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&a_scales_storage[kb]));
                        __m512 a_scales_f32 = _mm512_cvtph_ps(a_scales_fp16);

                        // Load 16 B scales (FP16) and convert to FP32
                        __m256i b_scales_fp16 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&b_scales_storage[kb]));
                        __m512 b_scales_f32 = _mm512_cvtph_ps(b_scales_fp16);

                        // Compute: compensated * a_scale * b_scale
                        __m512 scaled = _mm512_mul_ps(compensated, a_scales_f32);
                        scaled = _mm512_mul_ps(scaled, b_scales_f32);

                        // Accumulate (horizontal reduction)
                        result += _mm512_reduce_add_ps(scaled);
                    }

                    // 8-wide vectorization (AVX2)
                    for (; kb + 8 <= K_blocks; kb += 8)
                    {
                        // Load 8 block_dots (INT32) and convert to FP32
                        __m256i dots_i32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&block_dots_storage[kb]));
                        __m256 dots_f32 = _mm256_cvtepi32_ps(dots_i32);

                        // Load 8 sum_qs (INT32) and convert to FP32
                        __m256i sum_qs_i32 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&sum_qs_storage[kb]));
                        __m256 sum_qs_f32 = _mm256_cvtepi32_ps(sum_qs_i32);

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        const __m256 compensation_const = _mm256_set1_ps(128.0f);
                        __m256 compensated = _mm256_fnmadd_ps(sum_qs_f32, compensation_const, dots_f32);

                        // Load 8 A scales (FP16) and convert to FP32
                        __m128i a_scales_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&a_scales_storage[kb]));
                        __m256 a_scales_f32 = _mm256_cvtph_ps(a_scales_fp16);

                        // Load 8 B scales (FP16) and convert to FP32
                        __m128i b_scales_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&b_scales_storage[kb]));
                        __m256 b_scales_f32 = _mm256_cvtph_ps(b_scales_fp16);

                        // Compute: compensated * a_scale * b_scale
                        __m256 scaled = _mm256_mul_ps(compensated, a_scales_f32);
                        scaled = _mm256_mul_ps(scaled, b_scales_f32);

                        // Horizontal reduction (AVX2-compatible)
                        __m128 lo = _mm256_castps256_ps128(scaled);
                        __m128 hi = _mm256_extractf128_ps(scaled, 1);
                        __m128 sum4 = _mm_add_ps(lo, hi);
                        __m128 shuf = _mm_movehdup_ps(sum4);
                        __m128 sum2 = _mm_add_ps(sum4, shuf);
                        shuf = _mm_movehl_ps(shuf, sum2);
                        __m128 sum1 = _mm_add_ss(sum2, shuf);
                        result += _mm_cvtss_f32(sum1);
                    }

                    // 4-wide vectorization (AVX2)
                    for (; kb + 4 <= K_blocks; kb += 4)
                    {
                        // Load 4 block_dots (INT32) and convert to FP32
                        __m128i dots_i32 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&block_dots_storage[kb]));
                        __m128 dots_f32 = _mm_cvtepi32_ps(dots_i32);

                        // Load 4 sum_qs (INT32) and convert to FP32
                        __m128i sum_qs_i32 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&sum_qs_storage[kb]));
                        __m128 sum_qs_f32 = _mm_cvtepi32_ps(sum_qs_i32);

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        const __m128 compensation_const = _mm_set1_ps(128.0f);
                        __m128 compensated = _mm_fnmadd_ps(sum_qs_f32, compensation_const, dots_f32);

                        // Load 4 A scales (FP16) and convert to FP32
                        __m128i a_scales_fp16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&a_scales_storage[kb]));
                        __m128 a_scales_f32 = _mm_cvtph_ps(a_scales_fp16);

                        // Load 4 B scales (FP16) and convert to FP32
                        __m128i b_scales_fp16 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&b_scales_storage[kb]));
                        __m128 b_scales_f32 = _mm_cvtph_ps(b_scales_fp16);

                        // Compute: compensated * a_scale * b_scale
                        __m128 scaled = _mm_mul_ps(compensated, a_scales_f32);
                        scaled = _mm_mul_ps(scaled, b_scales_f32);

                        // Horizontal reduction: 4→1
                        __m128 shuf = _mm_movehdup_ps(scaled);
                        __m128 sum2 = _mm_add_ps(scaled, shuf);
                        shuf = _mm_movehl_ps(shuf, sum2);
                        __m128 sum1 = _mm_add_ss(sum2, shuf);
                        result += _mm_cvtss_f32(sum1);
                    }

                    // Scalar tail (0-3 blocks remaining)
                    for (; kb < K_blocks; ++kb)
                    {
                        int32_t block_dot = block_dots_storage[kb];
                        int32_t sum_qs_val = sum_qs_storage[kb];

                        // Apply Q8_1 compensation: compensated = block_dot - 128*sum_qs
                        float compensated = static_cast<float>(block_dot - sum_qs_val * 128);

                        float a_scale = fp16_to_fp32(a_scales_storage[kb]);
                        float b_scale = fp16_to_fp32(b_scales_storage[kb]);
                        result += compensated * a_scale * b_scale;
                    }

                    C[i * ldc + j] = result; // FIXED: Use assignment (=) not accumulation (+=)
                }
            }
        }
    };

    // Production kernel configuration
    // 32×128 microkernel with prefetch=1, optimized via comprehensive parameter sweep (21,060 configurations tested)
    // Achieves 500+ GFLOPS on Xeon Gold 6238R (Nov 2025)
    using Q8_1GemmKernel = Q8_1GemmKernelTemplate<32, 128, 1, 0, 0, 2, 18>;

} // namespace llaminar2
