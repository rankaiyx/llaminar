/**
 * @file QuantisedGemmJit_FP32_x_Q8_1.h
 * @brief JIT-compiled AVX-512 GEMM kernel for FP32 x Q8_1 matrix multiplication.
 * @author David Sanftenberg
 *
 * This kernel is designed for the attention context computation: scores @ V
 * where scores are FP32 softmax probabilities and V is Q8_1 quantized.
 *
 * Operation: C[m,n] = A[m,k] @ B[k,n]^T (or A @ B depending on layout)
 * - A: FP32 attention weights [seq_len, kv_len] (row-major, strided)
 * - B: Q8_1 V tensor [kv_len, head_dim] (row-major, strided)
 * - C: FP32 output [seq_len, head_dim] (row-major, strided)
 *
 * For attention: C = scores @ V where:
 * - scores[q, kv] is the attention weight for query q attending to key/value kv
 * - V[kv, d] is the value vector at position kv, dimension d
 * - C[q, d] = Σ_kv scores[q, kv] * V[kv, d]
 */

#pragma once

#include "../../../../../external/onednn/third_party/xbyak/xbyak.h"
#include "../../../tensors/BlockStructures.h"
#include <cstdint>

namespace llaminar2
{
    namespace gemm_v4
    {

        /**
         * @brief Parameters for FP32 x Q8_1 strided GEMM
         *
         * Computes: C[m, n] = Σ_k A[m, k] * dequant(B[k, n])
         * where dequant(B) = B.qs[i] * B.d
         */
        struct FP32xQ8_1Params
        {
            const float *A;     // FP32 scores [M, K] with stride A_stride
            const void *B;      // Q8_1 blocks for V [K, N] with stride B_stride_bytes
            float *C;           // FP32 output [M, N] with stride C_stride
            int M;              // Number of output rows (seq_len for attention)
            int N;              // Number of output cols (head_dim for attention)
            int K;              // Reduction dimension (kv_len for attention)
            int A_stride;       // Stride of A in floats (typically K or kv_len)
            int B_stride_bytes; // Stride of B in bytes between rows
            int C_stride;       // Stride of C in floats (typically N or head_dim * n_heads)
            float alpha;        // Scale factor (typically 1.0)
            float beta;         // Accumulation factor (typically 0.0)
        };

        /**
         * @class QuantisedGemmJit_FP32_x_Q8_1
         * @brief JIT kernel for FP32 weights × Q8_1 values (scores @ V)
         *
         * Algorithm:
         * For each output element C[m, n]:
         *   sum = 0
         *   for k in 0..K:
         *     # Get Q8_1 block containing V[k, n]
         *     block_idx = n / 32
         *     elem_idx = n % 32
         *     V_dequant = B[k].blocks[block_idx].qs[elem_idx] * B[k].blocks[block_idx].d
         *     sum += A[m, k] * V_dequant
         *   C[m, n] = alpha * sum + beta * C[m, n]
         *
         * Optimization: Process output columns in blocks of 32 (one Q8_1 block)
         * to amortize scale factor loading.
         */
        class QuantisedGemmJit_FP32_x_Q8_1 : public Xbyak::CodeGenerator
        {
        public:
            QuantisedGemmJit_FP32_x_Q8_1(int m_blocking = 1)
                : Xbyak::CodeGenerator(4096 * 16), m_blocking_(m_blocking)
            {
                generate();
            }

            using kernel_func_t = void (*)(const FP32xQ8_1Params *params);

            kernel_func_t get_kernel()
            {
                return getCode<kernel_func_t>();
            }

        private:
            int m_blocking_;

            void generate()
            {
                using namespace Xbyak;

                // System V AMD64 ABI: rdi = first arg (params pointer)
                const Reg64 &reg_params = rdi;
                const Reg64 &reg_A_base = rsi;
                const Reg64 &reg_B_base = rdx;
                const Reg64 &reg_C_base = rcx;
                const Reg64 &reg_M = r8;
                const Reg64 &reg_N = r9;
                const Reg64 &reg_K = r10;

                const Reg64 &reg_A_stride = r11;
                const Reg64 &reg_B_stride = r12;
                const Reg64 &reg_C_stride = r13;

                const Reg64 &reg_m = r14;
                const Reg64 &reg_n = r15;
                const Reg64 &reg_k = rax;
                const Reg64 &reg_tmp = rbx;
                const Reg64 &reg_tmp2 = rbp;

                // Accumulators: Zmm0-3 for m_blocking_ output rows (16 floats each)
                // Zmm4-7: Scratch for loading/conversion
                // Zmm8: Loaded FP32 weights (broadcasted per k)
                // Zmm9: Dequantized Q8_1 values (16 floats)
                // Zmm10: Scale broadcast
                // Zmm11: Alpha
                // Zmm12: Beta

                // Prologue
                push(rbx);
                push(rbp);
                push(r12);
                push(r13);
                push(r14);
                push(r15);

                // Load params
                mov(reg_A_base, ptr[reg_params + offsetof(FP32xQ8_1Params, A)]);
                mov(reg_B_base, ptr[reg_params + offsetof(FP32xQ8_1Params, B)]);
                mov(reg_C_base, ptr[reg_params + offsetof(FP32xQ8_1Params, C)]);
                mov(reg_M.cvt32(), ptr[reg_params + offsetof(FP32xQ8_1Params, M)]);
                mov(reg_N.cvt32(), ptr[reg_params + offsetof(FP32xQ8_1Params, N)]);
                mov(reg_K.cvt32(), ptr[reg_params + offsetof(FP32xQ8_1Params, K)]);
                mov(reg_A_stride.cvt32(), ptr[reg_params + offsetof(FP32xQ8_1Params, A_stride)]);
                mov(reg_B_stride.cvt32(), ptr[reg_params + offsetof(FP32xQ8_1Params, B_stride_bytes)]);
                mov(reg_C_stride.cvt32(), ptr[reg_params + offsetof(FP32xQ8_1Params, C_stride)]);

                vbroadcastss(zmm11, ptr[reg_params + offsetof(FP32xQ8_1Params, alpha)]);
                vbroadcastss(zmm12, ptr[reg_params + offsetof(FP32xQ8_1Params, beta)]);

                // Convert strides to bytes where needed
                shl(reg_A_stride, 2); // A_stride * sizeof(float)
                shl(reg_C_stride, 2); // C_stride * sizeof(float)

                // Outer loop over M (output rows)
                xor_(reg_m, reg_m);

                Label loop_M;
                L(loop_M);
                {
                    cmp(reg_m, reg_M);
                    jge("end_kernel", T_NEAR);

                    // Inner loop over N (output columns) in chunks of 16 (half ZMM)
                    // Q8_1 blocks are 32 elements, but we process 16 at a time for ZMM alignment
                    xor_(reg_n, reg_n);

                    Label loop_N;
                    L(loop_N);
                    {
                        cmp(reg_n, reg_N);
                        jge("end_row", T_NEAR);

                        // Check if we have at least 16 elements left
                        mov(reg_tmp, reg_N);
                        sub(reg_tmp, reg_n);
                        cmp(reg_tmp, 16);
                        jl("tail_N", T_NEAR);

                        // Initialize accumulator to zero
                        vxorps(zmm0, zmm0, zmm0);

                        // Calculate which Q8_1 block this column range falls into
                        // block_idx = n / 32, elem_offset = n % 32
                        mov(reg_tmp, reg_n);
                        shr(reg_tmp, 5); // block_idx = n / 32
                        mov(reg_tmp2, reg_n);
                        and_(reg_tmp2, 31); // elem_offset = n % 32

                        // K loop: accumulate weighted V values
                        xor_(reg_k, reg_k);

                        Label loop_K;
                        L(loop_K);
                        {
                            cmp(reg_k, reg_K);
                            jge("end_K", T_NEAR);

                            // Load A[m, k] and broadcast
                            // A_ptr = A_base + m * A_stride + k * sizeof(float)
                            mov(rax, reg_m);
                            imul(rax, reg_A_stride);
                            add(rax, reg_A_base);
                            mov(rbx, reg_k);
                            shl(rbx, 2); // k * sizeof(float)
                            add(rax, rbx);
                            vbroadcastss(zmm8, ptr[rax]);

                            // Load Q8_1 block from B[k, block_idx]
                            // B_ptr = B_base + k * B_stride + block_idx * sizeof(Q8_1Block)
                            mov(rax, reg_k);
                            imul(rax, reg_B_stride);
                            add(rax, reg_B_base);
                            mov(rbx, reg_tmp);  // block_idx
                            imul(rbx, rbx, 36); // * sizeof(Q8_1Block)
                            add(rax, rbx);

                            // rax now points to Q8_1Block
                            // Load scale (FP16 at offset 0)
                            vpbroadcastw(xmm10, ptr[rax]);
                            vcvtph2ps(zmm10, ymm10);

                            // Load 16 int8 values starting at elem_offset
                            // Q8_1Block layout: d(2), sum_qs(2), qs[32](32)
                            // qs starts at offset 4
                            lea(rbx, ptr[rax + 4]); // Point to qs[0]
                            add(rbx, reg_tmp2);     // Add elem_offset

                            // Load 16 int8 values
                            vmovdqu8(xmm9, ptr[rbx]);

                            // Sign-extend int8 -> int32 (16 elements)
                            vpmovsxbd(zmm9, xmm9);

                            // Convert int32 -> float
                            vcvtdq2ps(zmm9, zmm9);

                            // Apply scale: V_dequant = qs * d
                            vmulps(zmm9, zmm9, zmm10);

                            // Accumulate: sum += A[m,k] * V_dequant
                            vfmadd231ps(zmm0, zmm8, zmm9);

                            inc(reg_k);
                            jmp(loop_K, T_NEAR);
                        }

                        L("end_K");

                        // Apply alpha and beta, store result
                        // C[m, n:n+16] = alpha * sum + beta * C[m, n:n+16]

                        // Calculate C pointer: C_base + m * C_stride + n * sizeof(float)
                        mov(rax, reg_m);
                        imul(rax, reg_C_stride);
                        add(rax, reg_C_base);
                        mov(rbx, reg_n);
                        shl(rbx, 2);
                        add(rax, rbx);

                        // Apply alpha
                        vmulps(zmm0, zmm0, zmm11);

                        // Check if beta != 0
                        vxorps(zmm4, zmm4, zmm4);
                        vucomiss(xmm12, xmm4);
                        je("skip_beta", T_NEAR);

                        // Load existing C and accumulate
                        vmovups(zmm4, ptr[rax]);
                        vfmadd231ps(zmm0, zmm4, zmm12);

                        L("skip_beta");
                        vmovups(ptr[rax], zmm0);

                        add(reg_n, 16);
                        jmp(loop_N, T_NEAR);
                    }

                    // Tail handling for N < 16
                    L("tail_N");
                    {
                        cmp(reg_n, reg_N);
                        jge("end_row", T_NEAR);

                        // Process remaining elements one at a time
                        // Initialize scalar accumulator
                        vxorps(xmm0, xmm0, xmm0);

                        // Calculate block_idx and elem_offset for this column
                        mov(reg_tmp, reg_n);
                        shr(reg_tmp, 5); // block_idx = n / 32
                        mov(reg_tmp2, reg_n);
                        and_(reg_tmp2, 31); // elem_offset = n % 32

                        xor_(reg_k, reg_k);

                        Label loop_K_tail;
                        L(loop_K_tail);
                        {
                            cmp(reg_k, reg_K);
                            jge("end_K_tail", T_NEAR);

                            // Load A[m, k]
                            mov(rax, reg_m);
                            imul(rax, reg_A_stride);
                            add(rax, reg_A_base);
                            mov(rbx, reg_k);
                            shl(rbx, 2);
                            add(rax, rbx);
                            vmovss(xmm8, ptr[rax]);

                            // Load Q8_1 block
                            mov(rax, reg_k);
                            imul(rax, reg_B_stride);
                            add(rax, reg_B_base);
                            mov(rbx, reg_tmp);
                            imul(rbx, rbx, 36);
                            add(rax, rbx);

                            // Load scale
                            vpbroadcastw(xmm10, ptr[rax]);
                            vcvtph2ps(xmm10, xmm10);

                            // Load single int8 value
                            lea(rbx, ptr[rax + 4]);
                            add(rbx, reg_tmp2);
                            movsx(eax, byte[rbx]);
                            vcvtsi2ss(xmm9, xmm9, eax);

                            // Dequantize
                            vmulss(xmm9, xmm9, xmm10);

                            // Accumulate
                            vfmadd231ss(xmm0, xmm8, xmm9);

                            inc(reg_k);
                            jmp(loop_K_tail, T_NEAR);
                        }

                        L("end_K_tail");

                        // Store single result
                        mov(rax, reg_m);
                        imul(rax, reg_C_stride);
                        add(rax, reg_C_base);
                        mov(rbx, reg_n);
                        shl(rbx, 2);
                        add(rax, rbx);

                        vmulss(xmm0, xmm0, xmm11);

                        vxorps(xmm4, xmm4, xmm4);
                        vucomiss(xmm12, xmm4);
                        je("skip_beta_tail", T_NEAR);

                        vmovss(xmm4, ptr[rax]);
                        vfmadd231ss(xmm0, xmm4, xmm12);

                        L("skip_beta_tail");
                        vmovss(ptr[rax], xmm0);

                        inc(reg_n);
                        jmp("tail_N", T_NEAR);
                    }

                    L("end_row");
                    inc(reg_m);
                    jmp(loop_M, T_NEAR);
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

    } // namespace gemm_v4
} // namespace llaminar2
