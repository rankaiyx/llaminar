/**
 * @file JitVWeightedAccum.h
 * @brief JIT Microkernel μK3: Weighted V accumulation
 * @author David Sanftenberg
 * @date December 2025
 *
 * JIT-generated weighted V accumulation for attention context computation.
 * Mirrors the reference VWeightedAccum.h microkernel.
 *
 * Algorithm:
 *   For each block b in [0, num_blocks):
 *     d_v = fp16_to_fp32(V[b].d)
 *     v_dequant[0:31] = V[b].qs[0:31] * d_v
 *     context[b*32 : (b+1)*32] += weight * v_dequant
 *
 * Register conventions:
 * - Input: V block pointer in GP reg, weight in zmm_weight()
 * - Output: Context accumulators in zmm_accum(0..N) or stack
 * - Scratch: zmm8-9 for V dequantization, zmm15 for scale
 */

#pragma once

#include "JitMicrokernelBase.h"
#include "../../../jit/RegisterGuard.h"

namespace llaminar::v2::kernels::jit
{

    /**
     * @brief JIT code emitter for weighted V accumulation
     *
     * Emits code for: context += weight * dequant(V)
     * Called once per K/V position in the attention inner loop.
     */
    class JitVWeightedAccumEmitter
    {
    public:
        /**
         * @brief Emit weighted V accumulation for one V row
         *
         * For each block, dequantizes V values and accumulates weighted sum
         * into context registers/stack.
         *
         * Prerequisites:
         * - zmm_weight() contains softmax weight (broadcast)
         *
         * Context storage:
         * - Blocks 0-1: zmm_accum(0-3) for 64 floats
         * - Blocks 2+: Spilled to stack
         *
         * Clobbers:
         * - zmm8, zmm9 (V dequantization)
         * - zmm15 (V scale)
         * - zmm_scratch(0-1) for spilled block load/store
         *
         * @param gen Code generator
         * @param reg_V_ptr GP register pointing to V blocks for this position
         * @param num_blocks Number of Q8_1 blocks (head_dim / 32)
         * @param spill_base_offset Stack offset for spilled context accumulators
         */
        void emit_weighted_accum(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_V_ptr,
            int num_blocks,
            int spill_base_offset)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_v_weighted_accum (" + std::to_string(num_blocks) + " blocks)");

            // Register aliases
            Zmm zmm_v_lo = gen.zmm_input(0);  // zmm8: V values 0-15 (dequantized)
            Zmm zmm_v_hi = gen.zmm_input(1);  // zmm9: V values 16-31 (dequantized)
            Zmm zmm_d_v = gen.zmm_scratch(5); // zmm25: V scale

            for (int b = 0; b < num_blocks; ++b)
            {
                gen.debug_emit("  V block " + std::to_string(b));

                int v_offset = b * 36; // Q8_1Block layout: 36 bytes per block

                // Load V scale: d_V (FP16 at offset 0)
                gen.vpbroadcastw(Xmm(zmm_d_v.getIdx()), gen.ptr[reg_V_ptr + v_offset]);
                gen.vcvtph2ps(Xmm(zmm_d_v.getIdx()), Xmm(zmm_d_v.getIdx()));
                gen.vbroadcastss(zmm_d_v, Xmm(zmm_d_v.getIdx()));

                // Load V data (32 int8 values at offset 4)
                // Sign-extend to int32, convert to float
                gen.vpmovsxbd(zmm_v_lo, gen.ptr[reg_V_ptr + v_offset + 4]);      // 16 int8 -> int32
                gen.vpmovsxbd(zmm_v_hi, gen.ptr[reg_V_ptr + v_offset + 4 + 16]); // Next 16

                gen.vcvtdq2ps(zmm_v_lo, zmm_v_lo); // int32 -> float
                gen.vcvtdq2ps(zmm_v_hi, zmm_v_hi);

                // Scale by d_V
                gen.vmulps(zmm_v_lo, zmm_v_lo, zmm_d_v);
                gen.vmulps(zmm_v_hi, zmm_v_hi, zmm_d_v);

                // Accumulate: context += weight * v_dequant
                // Use FMA: context = context + weight * v
                if (b == 0)
                {
                    // Block 0: context floats 0-31 in zmm_accum(0), zmm_accum(1)
                    gen.vfmadd231ps(gen.zmm_accum(0), zmm_v_lo, gen.zmm_weight());
                    gen.vfmadd231ps(gen.zmm_accum(1), zmm_v_hi, gen.zmm_weight());
                }
                else if (b == 1)
                {
                    // Block 1: context floats 32-63 in zmm_accum(2), zmm_accum(3)
                    gen.vfmadd231ps(gen.zmm_accum(2), zmm_v_lo, gen.zmm_weight());
                    gen.vfmadd231ps(gen.zmm_accum(3), zmm_v_hi, gen.zmm_weight());
                }
                else
                {
                    // Spilled blocks: load from stack, FMA, store back
                    int spill_lo = spill_base_offset + (b - 2) * 128;
                    int spill_hi = spill_lo + 64;

                    // Use Scratch1 (zmm21) for temp, NOT Scratch0 (zmm20) which holds weight!
                    Zmm zmm_tmp = gen.zmm_scratch(1);

                    // Low half
                    gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_lo]);
                    gen.vfmadd231ps(zmm_tmp, zmm_v_lo, gen.zmm_weight());
                    gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_tmp);

                    // High half
                    gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_hi]);
                    gen.vfmadd231ps(zmm_tmp, zmm_v_hi, gen.zmm_weight());
                    gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_tmp);
                }
            }
        }

        /**
         * @brief Emit context rescaling by correction factor
         *
         * When softmax max increases, all previous context values must be
         * rescaled by exp(old_max - new_max).
         *
         * @param gen Code generator
         * @param num_blocks Number of blocks
         * @param spill_base_offset Stack offset for spilled accumulators
         */
        void emit_rescale_context(
            JitMicrokernelBase &gen,
            int num_blocks,
            int spill_base_offset)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_rescale_context");

            // Rescale register-resident blocks
            gen.vmulps(gen.zmm_accum(0), gen.zmm_accum(0), gen.zmm_corr());
            gen.vmulps(gen.zmm_accum(1), gen.zmm_accum(1), gen.zmm_corr());

            if (num_blocks >= 2)
            {
                gen.vmulps(gen.zmm_accum(2), gen.zmm_accum(2), gen.zmm_corr());
                gen.vmulps(gen.zmm_accum(3), gen.zmm_accum(3), gen.zmm_corr());
            }

            // Rescale spilled blocks
            if (num_blocks > 2)
            {
                // CRITICAL: Use Scratch5 (zmm25) for temp, NOT Scratch0-3 (zmm20-23)!
                // During FA2 tile processing, Score0-3 (xmm20-23) are still live and
                // contain the attention scores. Using zmm20-23 would clobber them
                // before they're used for weight computation.
                //
                // Scratch4 (zmm24) holds tile_max, so use Scratch5 (zmm25).
                Zmm zmm_tmp = gen.zmm_scratch(5); // zmm25 - safe during FA2

                for (int b = 2; b < num_blocks; ++b)
                {
                    int spill_lo = spill_base_offset + (b - 2) * 128;
                    int spill_hi = spill_lo + 64;

                    gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_lo]);
                    gen.vmulps(zmm_tmp, zmm_tmp, gen.zmm_corr());
                    gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_tmp);

                    gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_hi]);
                    gen.vmulps(zmm_tmp, zmm_tmp, gen.zmm_corr());
                    gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_tmp);
                }
            }
        }

        /**
         * @brief Emit context initialization (zero all accumulators)
         *
         * @param gen Code generator
         * @param num_blocks Number of blocks
         * @param spill_base_offset Stack offset for spilled accumulators
         */
        void emit_init_context(
            JitMicrokernelBase &gen,
            int num_blocks,
            int spill_base_offset)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_init_context");

            // Zero register-resident blocks
            gen.vxorps(gen.zmm_accum(0), gen.zmm_accum(0), gen.zmm_accum(0));
            gen.vxorps(gen.zmm_accum(1), gen.zmm_accum(1), gen.zmm_accum(1));

            if (num_blocks >= 2)
            {
                gen.vxorps(gen.zmm_accum(2), gen.zmm_accum(2), gen.zmm_accum(2));
                gen.vxorps(gen.zmm_accum(3), gen.zmm_accum(3), gen.zmm_accum(3));
            }

            // Zero spilled blocks
            if (num_blocks > 2)
            {
                Zmm zmm_zero = gen.zmm_scratch(0);
                gen.vxorps(zmm_zero, zmm_zero, zmm_zero);

                for (int b = 2; b < num_blocks; ++b)
                {
                    int spill_lo = spill_base_offset + (b - 2) * 128;
                    int spill_hi = spill_lo + 64;

                    gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_zero);
                    gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_zero);
                }
            }
        }

        /**
         * @brief Emit context normalization by 1/sum
         *
         * @param gen Code generator
         * @param inv_sum_zmm ZMM containing 1/sum (broadcast)
         * @param num_blocks Number of blocks
         * @param spill_base_offset Stack offset for spilled accumulators
         */
        void emit_normalize_context(
            JitMicrokernelBase &gen,
            const Xbyak::Zmm &inv_sum_zmm,
            int num_blocks,
            int spill_base_offset)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_normalize_context");

            // Normalize register-resident blocks
            gen.vmulps(gen.zmm_accum(0), gen.zmm_accum(0), inv_sum_zmm);
            gen.vmulps(gen.zmm_accum(1), gen.zmm_accum(1), inv_sum_zmm);

            if (num_blocks >= 2)
            {
                gen.vmulps(gen.zmm_accum(2), gen.zmm_accum(2), inv_sum_zmm);
                gen.vmulps(gen.zmm_accum(3), gen.zmm_accum(3), inv_sum_zmm);
            }

            // Normalize spilled blocks
            if (num_blocks > 2)
            {
                Zmm zmm_tmp = gen.zmm_scratch(0);

                for (int b = 2; b < num_blocks; ++b)
                {
                    int spill_lo = spill_base_offset + (b - 2) * 128;
                    int spill_hi = spill_lo + 64;

                    gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_lo]);
                    gen.vmulps(zmm_tmp, zmm_tmp, inv_sum_zmm);
                    gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_tmp);

                    gen.vmovups(zmm_tmp, gen.ptr[gen.rsp + spill_hi]);
                    gen.vmulps(zmm_tmp, zmm_tmp, inv_sum_zmm);
                    gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_tmp);
                }
            }
        }

        /**
         * @brief Emit weighted V accumulation from cached V blocks (prefill mode)
         *
         * Similar to emit_weighted_accum but reads V from stack cache instead
         * of a memory pointer. Used in prefill mode where V blocks are cached
         * on stack for reuse across multiple Q positions.
         *
         * Prerequisites:
         * - zmm_weight() contains softmax weight (broadcast)
         * - V blocks are cached on stack with 64-byte padding per block
         *
         * Context storage:
         * - For prefill, context is in zmm_accum(0-3) loaded from memory
         * - No spilling needed since we reload from context buffer each Q iteration
         *
         * @param gen Code generator
         * @param reg_v_cache_ptr GP register pointing to V cache on stack
         * @param num_blocks Number of Q8_1 blocks (head_dim / 32)
         */
        void emit_weighted_accum_from_cache(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_v_cache_ptr,
            int num_blocks)
        {
            using namespace Xbyak;

            gen.debug_emit("emit_v_weighted_accum_from_cache (" + std::to_string(num_blocks) + " blocks)");

            // Register aliases
            Zmm zmm_v_lo = gen.zmm_input(0);  // zmm8: V values 0-15 (dequantized)
            Zmm zmm_v_hi = gen.zmm_input(1);  // zmm9: V values 16-31 (dequantized)
            Zmm zmm_d_v = gen.zmm_scratch(5); // zmm25: V scale

            for (int b = 0; b < num_blocks; ++b)
            {
                // V blocks are padded to 64 bytes on stack
                int v_cache_offset = b * 64;

                // Load V scale: d_V (FP16 at offset 0 of padded block)
                gen.vpbroadcastw(Xmm(zmm_d_v.getIdx()), gen.ptr[reg_v_cache_ptr + v_cache_offset]);
                gen.vcvtph2ps(Xmm(zmm_d_v.getIdx()), Xmm(zmm_d_v.getIdx()));
                gen.vbroadcastss(zmm_d_v, Xmm(zmm_d_v.getIdx()));

                // Load V data (32 int8 values at offset 4 of padded block)
                gen.vpmovsxbd(zmm_v_lo, gen.ptr[reg_v_cache_ptr + v_cache_offset + 4]);
                gen.vpmovsxbd(zmm_v_hi, gen.ptr[reg_v_cache_ptr + v_cache_offset + 4 + 16]);

                gen.vcvtdq2ps(zmm_v_lo, zmm_v_lo);
                gen.vcvtdq2ps(zmm_v_hi, zmm_v_hi);

                // Scale by d_V
                gen.vmulps(zmm_v_lo, zmm_v_lo, zmm_d_v);
                gen.vmulps(zmm_v_hi, zmm_v_hi, zmm_d_v);

                // Accumulate: context += weight * v_dequant
                // In prefill mode, context is in zmm_accum(0-3) for blocks 0-1
                // For head_dim > 64 (blocks > 2), we'd need more registers or spilling
                // Current implementation supports head_dim up to 64 (num_blocks <= 2)
                if (b == 0)
                {
                    gen.vfmadd231ps(gen.zmm_accum(0), zmm_v_lo, gen.zmm_weight());
                    gen.vfmadd231ps(gen.zmm_accum(1), zmm_v_hi, gen.zmm_weight());
                }
                else if (b == 1)
                {
                    gen.vfmadd231ps(gen.zmm_accum(2), zmm_v_lo, gen.zmm_weight());
                    gen.vfmadd231ps(gen.zmm_accum(3), zmm_v_hi, gen.zmm_weight());
                }
                else if (b == 2)
                {
                    // For head_dim=128 (4 blocks), use additional scratch registers
                    // Avoid zmm20 (Scratch0) which holds weight!
                    // Use zmm21-22 (Scratch1-2) for block 2
                    gen.vfmadd231ps(gen.zmm_scratch(1), zmm_v_lo, gen.zmm_weight());
                    gen.vfmadd231ps(gen.zmm_scratch(2), zmm_v_hi, gen.zmm_weight());
                }
                else if (b == 3)
                {
                    // Use zmm23-24 (Scratch3-4) for block 3
                    gen.vfmadd231ps(gen.zmm_scratch(3), zmm_v_lo, gen.zmm_weight());
                    gen.vfmadd231ps(gen.zmm_scratch(4), zmm_v_hi, gen.zmm_weight());
                }
                // For larger head_dim, we'd need to spill to memory
            }
        }

        /**
         * @brief FA2 Interleaved V Accumulation for 4 KV positions
         *
         * Processes 4 V rows simultaneously with interleaved loads and FMAs
         * to hide memory latency. This is the FA2 optimization that pipelines
         * V data loading with compute.
         *
         * Pattern:
         *   Load V[0] block
         *   Load V[1] block | FMA V[0]
         *   Load V[2] block | FMA V[1]
         *   Load V[3] block | FMA V[2]
         *                   | FMA V[3]
         *
         * This overlaps memory access with compute, reducing stalls.
         *
         * @param gen Code generator
         * @param reg_V_ptr Pointer to first V row (V[kv_start])
         * @param v_stride Bytes between consecutive V rows
         * @param num_blocks Number of Q8_1 blocks per head
         * @param weight_xmm0..3 Scalar weights for V[0..3]
         * @param spill_base_offset Stack offset for spilled accumulators
         */
        void emit_interleaved_v_accum_4x(
            JitMicrokernelBase &gen,
            const Xbyak::Reg64 &reg_V_ptr,
            int64_t v_stride,
            int num_blocks,
            const Xbyak::Xmm &weight_xmm0,
            const Xbyak::Xmm &weight_xmm1,
            const Xbyak::Xmm &weight_xmm2,
            const Xbyak::Xmm &weight_xmm3,
            int spill_base_offset)
        {
            using namespace Xbyak;
            using namespace llaminar2::jit;

            gen.debug_emit("emit_interleaved_v_accum_4x (" + std::to_string(num_blocks) + " blocks)");

            // === REGISTER GUARDS ===
            // Borrow registers used internally by this function.
            // NOTE: weight_xmm0-3 are passed in by caller, so caller owns them.
            //       We don't borrow them here - caller is responsible for their lifetime.

            // V data loading registers:
            // - Use Input0-1 (zmm8-9) for V[0] data (safe, doesn't conflict with Q)
            // - Use Accum4-5 (zmm4-5) for V[1] data (CHANGED from Input2-3 to avoid Q conflict!)
            //   Q data is in zmm10-13, so we CANNOT use zmm10-11 for V loading!
            auto guard_v_lo_0 = gen.borrow<Input0>(); // zmm8
            auto guard_v_hi_0 = gen.borrow<Input1>(); // zmm9
            auto guard_v_lo_1 = gen.borrow<Accum4>(); // zmm4 - CHANGED from Input2 to avoid Q conflict!
            auto guard_v_hi_1 = gen.borrow<Accum5>(); // zmm5 - CHANGED from Input3 to avoid Q conflict!

            // V scale register (Scratch5 = zmm25)
            auto guard_d_v = gen.borrow<Scratch5>();

            // Use the passed-in weight XMMs directly for broadcast targets
            // Broadcast into the SAME registers they came from
            Zmm zmm_w0 = Zmm(weight_xmm0.getIdx());
            Zmm zmm_w1 = Zmm(weight_xmm1.getIdx());
            Zmm zmm_w2 = Zmm(weight_xmm2.getIdx());
            Zmm zmm_w3 = Zmm(weight_xmm3.getIdx());

            gen.vbroadcastss(zmm_w0, weight_xmm0);
            gen.vbroadcastss(zmm_w1, weight_xmm1);
            gen.vbroadcastss(zmm_w2, weight_xmm2);
            gen.vbroadcastss(zmm_w3, weight_xmm3);

            // V data registers for interleaving
            Zmm zmm_v_lo_0 = guard_v_lo_0.zmm();
            Zmm zmm_v_hi_0 = guard_v_hi_0.zmm();
            Zmm zmm_v_lo_1 = guard_v_lo_1.zmm();
            Zmm zmm_v_hi_1 = guard_v_hi_1.zmm();
            Zmm zmm_d_v = guard_d_v.zmm();

            for (int b = 0; b < num_blocks; ++b)
            {
                gen.debug_emit("  Block " + std::to_string(b) + " (4x interleaved)");

                int v_block_offset = b * 36;

                // Determine which accumulators to use for this block
                Zmm zmm_ctx_lo, zmm_ctx_hi;
                bool use_spill = false;
                int spill_lo = 0, spill_hi = 0;

                if (b == 0)
                {
                    zmm_ctx_lo = gen.zmm_accum(0);
                    zmm_ctx_hi = gen.zmm_accum(1);
                }
                else if (b == 1)
                {
                    zmm_ctx_lo = gen.zmm_accum(2);
                    zmm_ctx_hi = gen.zmm_accum(3);
                }
                else
                {
                    // Spilled blocks
                    use_spill = true;
                    spill_lo = spill_base_offset + (b - 2) * 128;
                    spill_hi = spill_lo + 64;
                    // zmm20-23 hold weights! Cannot use them.
                    // zmm10-13 (Input2-5) are free (Q released).
                    zmm_ctx_lo = gen.zmm_input(2); // zmm10
                    zmm_ctx_hi = gen.zmm_input(3); // zmm11
                }

                if (use_spill)
                {
                    // Load spilled context
                    gen.vmovups(zmm_ctx_lo, gen.ptr[gen.rsp + spill_lo]);
                    gen.vmovups(zmm_ctx_hi, gen.ptr[gen.rsp + spill_hi]);
                }

                // ================================================================
                // Interleaved load/compute pattern for 4 V rows
                // ================================================================

                // --- V[0]: Load and dequantize ---
                gen.vpbroadcastw(Xmm(zmm_d_v.getIdx()), gen.ptr[reg_V_ptr + v_block_offset]);
                gen.vcvtph2ps(Xmm(zmm_d_v.getIdx()), Xmm(zmm_d_v.getIdx()));
                gen.vbroadcastss(zmm_d_v, Xmm(zmm_d_v.getIdx()));

                gen.vpmovsxbd(zmm_v_lo_0, gen.ptr[reg_V_ptr + v_block_offset + 4]);
                gen.vpmovsxbd(zmm_v_hi_0, gen.ptr[reg_V_ptr + v_block_offset + 4 + 16]);
                gen.vcvtdq2ps(zmm_v_lo_0, zmm_v_lo_0);
                gen.vcvtdq2ps(zmm_v_hi_0, zmm_v_hi_0);
                gen.vmulps(zmm_v_lo_0, zmm_v_lo_0, zmm_d_v);
                gen.vmulps(zmm_v_hi_0, zmm_v_hi_0, zmm_d_v);

                // --- V[1]: Load scale while V[0] is ready for FMA ---
                int64_t v1_offset = v_stride + v_block_offset;
                gen.vpbroadcastw(Xmm(zmm_d_v.getIdx()), gen.ptr[reg_V_ptr + v1_offset]);
                gen.vcvtph2ps(Xmm(zmm_d_v.getIdx()), Xmm(zmm_d_v.getIdx()));
                gen.vbroadcastss(zmm_d_v, Xmm(zmm_d_v.getIdx()));

                // FMA V[0] while loading V[1] data
                gen.vfmadd231ps(zmm_ctx_lo, zmm_v_lo_0, zmm_w0);
                gen.vpmovsxbd(zmm_v_lo_1, gen.ptr[reg_V_ptr + v1_offset + 4]);

                gen.vfmadd231ps(zmm_ctx_hi, zmm_v_hi_0, zmm_w0);
                gen.vpmovsxbd(zmm_v_hi_1, gen.ptr[reg_V_ptr + v1_offset + 4 + 16]);

                gen.vcvtdq2ps(zmm_v_lo_1, zmm_v_lo_1);
                gen.vcvtdq2ps(zmm_v_hi_1, zmm_v_hi_1);
                gen.vmulps(zmm_v_lo_1, zmm_v_lo_1, zmm_d_v);
                gen.vmulps(zmm_v_hi_1, zmm_v_hi_1, zmm_d_v);

                // --- V[2]: Load scale while V[1] is ready for FMA ---
                int64_t v2_offset = 2 * v_stride + v_block_offset;
                gen.vpbroadcastw(Xmm(zmm_d_v.getIdx()), gen.ptr[reg_V_ptr + v2_offset]);
                gen.vcvtph2ps(Xmm(zmm_d_v.getIdx()), Xmm(zmm_d_v.getIdx()));
                gen.vbroadcastss(zmm_d_v, Xmm(zmm_d_v.getIdx()));

                // FMA V[1] while loading V[2] data
                gen.vfmadd231ps(zmm_ctx_lo, zmm_v_lo_1, zmm_w1);
                gen.vpmovsxbd(zmm_v_lo_0, gen.ptr[reg_V_ptr + v2_offset + 4]); // Reuse zmm_v_lo_0

                gen.vfmadd231ps(zmm_ctx_hi, zmm_v_hi_1, zmm_w1);
                gen.vpmovsxbd(zmm_v_hi_0, gen.ptr[reg_V_ptr + v2_offset + 4 + 16]);

                gen.vcvtdq2ps(zmm_v_lo_0, zmm_v_lo_0);
                gen.vcvtdq2ps(zmm_v_hi_0, zmm_v_hi_0);
                gen.vmulps(zmm_v_lo_0, zmm_v_lo_0, zmm_d_v);
                gen.vmulps(zmm_v_hi_0, zmm_v_hi_0, zmm_d_v);

                // --- V[3]: Load scale while V[2] is ready for FMA ---
                int64_t v3_offset = 3 * v_stride + v_block_offset;
                gen.vpbroadcastw(Xmm(zmm_d_v.getIdx()), gen.ptr[reg_V_ptr + v3_offset]);
                gen.vcvtph2ps(Xmm(zmm_d_v.getIdx()), Xmm(zmm_d_v.getIdx()));
                gen.vbroadcastss(zmm_d_v, Xmm(zmm_d_v.getIdx()));

                // FMA V[2] while loading V[3] data
                gen.vfmadd231ps(zmm_ctx_lo, zmm_v_lo_0, zmm_w2);
                gen.vpmovsxbd(zmm_v_lo_1, gen.ptr[reg_V_ptr + v3_offset + 4]); // Reuse zmm_v_lo_1

                gen.vfmadd231ps(zmm_ctx_hi, zmm_v_hi_0, zmm_w2);
                gen.vpmovsxbd(zmm_v_hi_1, gen.ptr[reg_V_ptr + v3_offset + 4 + 16]);

                gen.vcvtdq2ps(zmm_v_lo_1, zmm_v_lo_1);
                gen.vcvtdq2ps(zmm_v_hi_1, zmm_v_hi_1);
                gen.vmulps(zmm_v_lo_1, zmm_v_lo_1, zmm_d_v);
                gen.vmulps(zmm_v_hi_1, zmm_v_hi_1, zmm_d_v);

                // --- FMA V[3] (final) ---
                gen.vfmadd231ps(zmm_ctx_lo, zmm_v_lo_1, zmm_w3);
                gen.vfmadd231ps(zmm_ctx_hi, zmm_v_hi_1, zmm_w3);

                if (use_spill)
                {
                    // Store spilled context back
                    gen.vmovups(gen.ptr[gen.rsp + spill_lo], zmm_ctx_lo);
                    gen.vmovups(gen.ptr[gen.rsp + spill_hi], zmm_ctx_hi);
                }
            }
        }
    };

} // namespace llaminar::v2::kernels::jit
