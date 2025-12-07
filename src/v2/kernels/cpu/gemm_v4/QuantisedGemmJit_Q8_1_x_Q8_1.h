/**
 * @file QuantisedGemmJit_Q8_1_x_Q8_1.h
 * @brief JIT-compiled AVX-512 VNNI GEMM kernel for Q8_1 x Q8_1 matrix multiplication.
 * @author David Sanftenberg
 */

#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "../../../tensors/BlockStructures.h"
#include <cstdint>

namespace llaminar2
{
    namespace gemm_v4
    {

        struct QuantisedGemmQ8_1Params
        {
            const void *A_row;   // Pointer to Q8_1 blocks for A row
            const void *B_start; // Pointer to Q8_1 blocks for B (start of first row)
            float *C_row;        // Pointer to output float row
            int N;               // Number of B rows to process
            int K_blocks;        // Number of blocks in K dimension
            int B_stride_bytes;  // Stride between B rows in bytes
            float alpha;         // Alpha scale
            float beta;          // Beta scale
        };

        /**
         * @class QuantisedGemmJit_Q8_1_x_Q8_1
         * @brief JIT kernel for Q8_1 vector * Q8_1 matrix (A_row * B^T)
         *
         * Computes dot product of one A row against N B rows.
         * Uses AVX-512 VNNI.
         *
         * Algorithm:
         * - Loop over N in chunks of 4.
         * - For each chunk, accumulate 4 dot products in parallel.
         * - Inner loop over K blocks.
         * - Accumulate partial sums in YMM registers (8 floats per B-row).
         * - Accumulate correction terms separately.
         * - At end of K loop, reduce partial sums, apply correction, scale, and store to C.
         */
        class QuantisedGemmJit_Q8_1_x_Q8_1 : public Xbyak::CodeGenerator
        {
        public:
            QuantisedGemmJit_Q8_1_x_Q8_1(int prefetch_dist = 0, int unroll_n = 4, int unroll_k = 1)
                : Xbyak::CodeGenerator(4096 * 8), prefetch_dist_(prefetch_dist), unroll_n_(unroll_n), unroll_k_(unroll_k)
            {
                if (unroll_n_ != 4 && unroll_n_ != 8)
                    unroll_n_ = 4;
                if (unroll_k_ < 1 || unroll_k_ > 4)
                    unroll_k_ = 1;
                generate();
            }

            using kernel_func_t = void (*)(const QuantisedGemmQ8_1Params *params);

            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
            }

        private:
            int prefetch_dist_;
            int unroll_n_;
            int unroll_k_;

            void generate()
            {
                using namespace Xbyak;

                // Registers
                const Reg64 &reg_params = rdi;
                const Reg64 &reg_A = rsi;
                const Reg64 &reg_B_base = rdx;
                const Reg64 &reg_C = rcx;
                const Reg64 &reg_N = r8;
                const Reg64 &reg_K_blocks = r9;

                const Reg64 &reg_B_stride = r10;
                const Reg64 &reg_n = r11;
                const Reg64 &reg_k = r12;
                const Reg64 &reg_A_cursor = r13;
                const Reg64 &reg_B_cursor = r14;
                const Reg64 &reg_tmp = r15;
                const Reg64 &reg_tmp2 = rax;

                // ZMM/YMM
                // We process 4 rows of B at a time.
                // Accumulators for partial sums (8 floats each)
                const Zmm &zmm_acc0 = zmm0;
                const Zmm &zmm_acc1 = zmm1;
                const Zmm &zmm_acc2 = zmm2;
                const Zmm &zmm_acc3 = zmm3;

                // Accumulators for correction (scalar float, but we use XMM/ZMM)
                const Zmm &zmm_corr = zmm4; // Stores 4 correction values

                const Zmm &zmm_A_qs = zmm5; // YMM actually
                const Zmm &zmm_A_scale = zmm6;
                const Zmm &zmm_128 = zmm7;

                const Zmm &zmm_B_qs = zmm8;
                const Zmm &zmm_res = zmm9;
                const Zmm &zmm_scale_prod = zmm10;

                const Zmm &zmm_alpha = zmm11;
                const Zmm &zmm_beta = zmm12;

                // Prologue
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // Load params
                mov(reg_A, ptr[reg_params + 0]);
                mov(reg_B_base, ptr[reg_params + 8]);
                mov(reg_C, ptr[reg_params + 16]);
                mov(reg_N.cvt32(), ptr[reg_params + 24]);
                mov(reg_K_blocks.cvt32(), ptr[reg_params + 28]);
                mov(reg_B_stride.cvt32(), ptr[reg_params + 32]);

                vbroadcastss(Ymm(zmm_alpha.getIdx()), ptr[reg_params + 36]);
                vbroadcastss(Ymm(zmm_beta.getIdx()), ptr[reg_params + 40]);

                // Constants
                mov(reg_tmp2.cvt32(), 0x80808080);
                vpbroadcastd(Ymm(zmm_128.getIdx()), reg_tmp2.cvt32());

                // Loop over N
                xor_(reg_n, reg_n);

                Label loop_N_label;
                L(loop_N_label);
                {
                    cmp(reg_n, reg_N);
                    jge("end_kernel", T_NEAR);

                    // Check if we have at least unroll_n_ rows left
                    mov(reg_tmp, reg_N);
                    sub(reg_tmp, reg_n);
                    cmp(reg_tmp, unroll_n_);
                    jl("tail_N", T_NEAR);

                    // Process unroll_n_ rows
                    // Init accumulators
                    for (int i = 0; i < unroll_n_; ++i)
                    {
                        vxorps(Ymm(i), Ymm(i), Ymm(i));                // zmm_acc[i]
                        vxorps(Ymm(20 + i), Ymm(20 + i), Ymm(20 + i)); // zmm_corr[i]
                    }

                    mov(reg_A_cursor, reg_A);

                    // Setup B cursors
                    mov(reg_B_cursor, reg_B_base);
                    mov(reg_tmp, reg_n);
                    imul(reg_tmp, reg_B_stride);
                    add(reg_B_cursor, reg_tmp); // B_cursor points to B[n]

                    // K loop
                    xor_(reg_k, reg_k);
                    Label loop_K_label;
                    L(loop_K_label);
                    {
                        // Load A block
                        vmovdqu8(Ymm(zmm_A_qs.getIdx()), ptr[reg_A_cursor + 4]);
                        // XOR 128 to make unsigned
                        vpxord(Ymm(zmm_A_qs.getIdx()), Ymm(zmm_A_qs.getIdx()), Ymm(zmm_128.getIdx()));

                        // Load A scale (FP16 -> FP32)
                        vpbroadcastw(Ymm(zmm_A_scale.getIdx()), ptr[reg_A_cursor]);
                        vcvtph2ps(Ymm(zmm_A_scale.getIdx()), Ymm(zmm_A_scale.getIdx()));

                        // Process unroll_n_ B rows
                        for (int j = 0; j < unroll_n_; ++j)
                        {
                            mov(reg_tmp, reg_B_stride);
                            imul(reg_tmp, reg_tmp, j);
                            add(reg_tmp, reg_B_cursor); // B[n+j] current block base

                            if (prefetch_dist_ > 0)
                            {
                                prefetcht0(ptr[reg_tmp + prefetch_dist_ * 36]);
                            }

                            vmovdqu8(Ymm(zmm_B_qs.getIdx()), ptr[reg_tmp + 4]);

                            // Dot product
                            vxorps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx())); // Clear temp int acc
                            vpdpbusd(Ymm(zmm_res.getIdx()), Ymm(zmm_A_qs.getIdx()), Ymm(zmm_B_qs.getIdx()));

                            // Convert to float
                            vcvtdq2ps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()));

                            // Load B scale
                            vpbroadcastw(Ymm(zmm_scale_prod.getIdx()), ptr[reg_tmp]);
                            vcvtph2ps(Ymm(zmm_scale_prod.getIdx()), Ymm(zmm_scale_prod.getIdx()));

                            // Multiply scales: A_scale * B_scale
                            vmulps(Ymm(zmm_scale_prod.getIdx()), Ymm(zmm_scale_prod.getIdx()), Ymm(zmm_A_scale.getIdx()));

                            // Apply scale to dot product
                            vmulps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()), Ymm(zmm_scale_prod.getIdx()));

                            // Accumulate to partial sums
                            vaddps(Ymm(j), Ymm(j), Ymm(zmm_res.getIdx()));

                            // Correction: 128 * B_sum * Scale
                            vpbroadcastw(Ymm(zmm_res.getIdx()), ptr[reg_tmp + 2]);
                            vpmovsxwd(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx())); // int16 -> int32
                            vcvtdq2ps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx())); // int32 -> float

                            vmulps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()), Ymm(zmm_scale_prod.getIdx())); // sum * scale

                            vaddps(Ymm(20 + j), Ymm(20 + j), Ymm(zmm_res.getIdx()));
                        }

                        // Advance cursors
                        add(reg_A_cursor, 36);
                        add(reg_B_cursor, 36);

                        inc(reg_k);
                        cmp(reg_k, reg_K_blocks);
                        jl(loop_K_label, T_NEAR);
                    }

                    // Reduction and Store
                    auto reduce_and_store = [&](const Zmm &acc, const Zmm &corr, int offset_idx)
                    {
                        Ymm ymm_acc = Ymm(acc.getIdx());
                        vextractf128(xmm30, ymm_acc, 1);
                        vaddps(xmm30, xmm30, Xmm(acc.getIdx()));
                        vpermilps(xmm31, xmm30, 0b01001110);
                        vaddps(xmm30, xmm30, xmm31);
                        vpermilps(xmm31, xmm30, 0b10110001);
                        vaddps(xmm30, xmm30, xmm31);
                        // xmm30[0] is sum.

                        // Correction
                        mov(reg_tmp2.cvt32(), 0x43000000); // 128.0f
                        vmovd(xmm31, reg_tmp2.cvt32());
                        vmulss(Xmm(corr.getIdx()), Xmm(corr.getIdx()), xmm31);

                        // Subtract correction
                        vsubss(xmm30, xmm30, Xmm(corr.getIdx()));

                        // Apply alpha/beta
                        mov(reg_tmp, reg_n);
                        add(reg_tmp, offset_idx);
                        shl(reg_tmp, 2);
                        add(reg_tmp, reg_C);

                        vmovss(xmm31, ptr[reg_tmp]);                   // Load C
                        vmulss(xmm31, xmm31, Xmm(zmm_beta.getIdx()));  // beta * C
                        vmulss(xmm30, xmm30, Xmm(zmm_alpha.getIdx())); // alpha * dot
                        vaddss(xmm30, xmm30, xmm31);

                        vmovss(ptr[reg_tmp], xmm30); // Store C
                    };

                    for (int i = 0; i < unroll_n_; ++i)
                    {
                        reduce_and_store(Zmm(i), Zmm(20 + i), i);
                    }

                    add(reg_n, unroll_n_);
                    jmp(loop_N_label, T_NEAR);
                }

                L("tail_N");
                {
                    // Handle remaining rows 1 by 1
                    Label loop_tail;
                    L(loop_tail);
                    cmp(reg_n, reg_N);
                    jge("end_kernel", T_NEAR);

                    // Process 1 row
                    vxorps(Ymm(zmm_acc0.getIdx()), Ymm(zmm_acc0.getIdx()), Ymm(zmm_acc0.getIdx()));
                    vxorps(Ymm(zmm20.getIdx()), Ymm(zmm20.getIdx()), Ymm(zmm20.getIdx())); // Correction

                    mov(reg_A_cursor, reg_A);

                    // B cursor
                    mov(reg_tmp, reg_n);
                    imul(reg_tmp, reg_B_stride);
                    mov(reg_B_cursor, reg_B_base);
                    add(reg_B_cursor, reg_tmp);

                    xor_(reg_k, reg_k);
                    Label loop_K_tail;
                    L(loop_K_tail);
                    {
                        // Load A
                        vmovdqu8(Ymm(zmm_A_qs.getIdx()), ptr[reg_A_cursor + 4]);
                        vpxord(Ymm(zmm_A_qs.getIdx()), Ymm(zmm_A_qs.getIdx()), Ymm(zmm_128.getIdx()));
                        vpbroadcastw(Ymm(zmm_A_scale.getIdx()), ptr[reg_A_cursor]);
                        vcvtph2ps(Ymm(zmm_A_scale.getIdx()), Ymm(zmm_A_scale.getIdx()));

                        // Load B
                        vmovdqu8(Ymm(zmm_B_qs.getIdx()), ptr[reg_B_cursor + 4]);

                        // Dot
                        vxorps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()));
                        vpdpbusd(Ymm(zmm_res.getIdx()), Ymm(zmm_A_qs.getIdx()), Ymm(zmm_B_qs.getIdx()));
                        vcvtdq2ps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()));

                        // Scale
                        vpbroadcastw(Ymm(zmm_scale_prod.getIdx()), ptr[reg_B_cursor]);
                        vcvtph2ps(Ymm(zmm_scale_prod.getIdx()), Ymm(zmm_scale_prod.getIdx()));
                        vmulps(Ymm(zmm_scale_prod.getIdx()), Ymm(zmm_scale_prod.getIdx()), Ymm(zmm_A_scale.getIdx()));
                        vmulps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()), Ymm(zmm_scale_prod.getIdx()));
                        vaddps(Ymm(zmm_acc0.getIdx()), Ymm(zmm_acc0.getIdx()), Ymm(zmm_res.getIdx()));

                        // Correction
                        vpbroadcastw(Ymm(zmm_res.getIdx()), ptr[reg_B_cursor + 2]);
                        vpmovsxwd(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()));
                        vcvtdq2ps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()));
                        vmulps(Ymm(zmm_res.getIdx()), Ymm(zmm_res.getIdx()), Ymm(zmm_scale_prod.getIdx()));
                        vaddps(Ymm(zmm20.getIdx()), Ymm(zmm20.getIdx()), Ymm(zmm_res.getIdx()));

                        add(reg_A_cursor, 36);
                        add(reg_B_cursor, 36);
                        inc(reg_k);
                        cmp(reg_k, reg_K_blocks);
                        jl(loop_K_tail, T_NEAR);
                    }

                    // Reduce and store
                    Ymm ymm_acc = Ymm(zmm_acc0.getIdx());
                    vextractf128(xmm30, ymm_acc, 1);
                    vaddps(xmm30, xmm30, Xmm(zmm_acc0.getIdx()));
                    vpermilps(xmm31, xmm30, 0b01001110);
                    vaddps(xmm30, xmm30, xmm31);
                    vpermilps(xmm31, xmm30, 0b10110001);
                    vaddps(xmm30, xmm30, xmm31);

                    mov(reg_tmp2.cvt32(), 0x43000000); // 128.0f
                    vmovd(xmm31, reg_tmp2.cvt32());
                    vmulss(Xmm(zmm20.getIdx()), Xmm(zmm20.getIdx()), xmm31);
                    vsubss(xmm30, xmm30, Xmm(zmm20.getIdx()));

                    mov(reg_tmp, reg_n);
                    shl(reg_tmp, 2);
                    add(reg_tmp, reg_C);
                    vmovss(xmm31, ptr[reg_tmp]);
                    vmulss(xmm31, xmm31, Xmm(zmm_beta.getIdx()));
                    vmulss(xmm30, xmm30, Xmm(zmm_alpha.getIdx()));
                    vaddss(xmm30, xmm30, xmm31);
                    vmovss(ptr[reg_tmp], xmm30);

                    inc(reg_n);
                    jmp(loop_tail, T_NEAR);
                }

                L("end_kernel");
                pop(r15);
                pop(r14);
                pop(r13);
                pop(r12);
                pop(rbp);
                pop(rbx);
                ret();
            }
        };

    }
}
