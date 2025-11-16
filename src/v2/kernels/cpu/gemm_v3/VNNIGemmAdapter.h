/**
 * @file VNNIGemmAdapter.h
 * @brief Adapter layer between Tensor API and low-level VNNI GEMM kernel
 * @author David Sanftenberg
 *
 * This adapter handles:
 * - Tensor format conversion (IActivationTensor/Q8_0Tensor → raw int8_t*)
 * - B-matrix packing to VNNI layout
 * - Scale extraction from quantized tensors
 * - Bias handling
 */

#pragma once

#include "VNNIGemm.h"
#include "ActivationPackingAdapters.h"
#include "WeightPackingAdapters.h"
#include "tensors/Tensors.h"
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace llaminar2
{

    /**
     * @brief Adapter function template for VNNI GEMM kernels
     *
     * Converts high-level Tensor API to low-level kernel call with proper packing.
     * First-cut implementation assumes:
     * - FP32 activations (A)
     * - Q8_0 quantized weights (B)
     * - Symmetric per-row activation quantization
     * - Symmetric per-column weight quantization
     * - No zero-points (symmetric quantization only)
     * - M, N, K must be multiples of M_R, N_R, K_BLK (edge tiles not yet implemented)
     *
     * @tparam M_R Micro-kernel M dimension
     * @tparam N_R Micro-kernel N dimension
     * @tparam K_BLK K block size
     * @tparam UNROLL_K K-loop unroll factor
     * @tparam PREFETCH_B_L1 L1 prefetch distance
     * @param M Number of rows in A and C
     * @param N Number of columns in B and C
     * @param K Number of columns in A / rows in B
     * @param A Activation tensor (FP32)
     * @param B Q8_0 weight tensor
     * @param C Output FP32 matrix (row-major, M x N)
     * @param ldc Leading dimension of C (should be >= N)
     * @param bias Optional bias vector [N], nullptr if not used
     */
    template <int M_R, int N_R, int K_BLK, int UNROLL_K, int PREFETCH_B_L1>
    void vnni_gemm_adapter(
        int M, int N, int K,
        const IActivationTensor &A,
        const Q8_0Tensor &B,
        float *C, int ldc,
        const float *bias = nullptr)
    {
        static_assert(M_R % 4 == 0, "M_R must be multiple of 4");
        static_assert(N_R % 16 == 0, "N_R must be multiple of 16");
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        if (M % M_R != 0 || N % N_R != 0 || K % K_BLK != 0)
        {
            for (int i = 0; i < M; ++i)
            {
                std::memset(C + i * ldc, 0, N * sizeof(float));
            }
            return;
        }

        const int T = K / K_BLK;

        std::vector<int8_t> B_packed_storage;
        PackedB Bp;
        std::vector<float> wgt_scales;
        pack_q8_0_weights_to_vnni_format<K_BLK>(B, K, N, B_packed_storage, Bp, wgt_scales);

        std::vector<float> bias_vec;
        const float *bias_ptr = bias;
        if (!bias)
        {
            bias_vec.resize(N, 0.0f);
            bias_ptr = bias_vec.data();
        }

        std::vector<float> A_fp32(M * K);
        const TensorBase *A_base = dynamic_cast<const TensorBase *>(&A);
        if (!A_base)
        {
            for (int i = 0; i < M; ++i)
            {
                std::memset(C + i * ldc, 0, N * sizeof(float));
            }
            return;
        }
        A_base->to_fp32_span(0, M * K, A_fp32.data());

        std::vector<int8_t> A_int8(M * K, 0);
        std::vector<float> act_block_scales(T, 1.0f);

        for (int t = 0; t < T; ++t)
        {
            const int k0 = t * K_BLK;
            float max_abs = 0.0f;

            for (int m = 0; m < M; ++m)
            {
                const float *row = A_fp32.data() + m * K + k0;
                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    max_abs = std::max(max_abs, std::abs(row[kk]));
                }
            }

            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            act_block_scales[t] = scale;
            const float inv_scale = 1.0f / scale;

            for (int m = 0; m < M; ++m)
            {
                const float *src_row = A_fp32.data() + m * K + k0;
                int8_t *dst_row = A_int8.data() + m * K + k0;
                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    float val = src_row[kk] * inv_scale;
                    val = std::max(-128.0f, std::min(127.0f, std::round(val)));
                    dst_row[kk] = static_cast<int8_t>(val);
                }
            }
        }

        gemm_int8_vnni_kernel<
            M_R, N_R, K_BLK, UNROLL_K, PREFETCH_B_L1,
            64,
            false,
            false,
            true>(
            A_int8.data(),
            Bp,
            C,
            bias_ptr,
            act_block_scales.data(),
            wgt_scales.data(),
            M, N, K);
    }

} // namespace llaminar2
