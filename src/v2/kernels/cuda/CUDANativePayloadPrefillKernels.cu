/**
 * @file CUDANativePayloadPrefillKernels.cu
 * @brief Q4_0 native-payload tensor-core prefill kernels.
 *
 * Weights stay in native payload form in VRAM. Each CTA loads compact Q4_0
 * payload blocks, decodes them into a transient shared-memory INT8 tile, then
 * reuses the existing mma.sync.m16n8k32 fragment path for compute.
 */

#include <cuda_runtime.h>

#include "CUDANativePayloadDecodeCommon.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace
{
    using llaminar2::cuda_native_payload::fp16_bits_to_float;

    constexpr int BK = 32;
    constexpr int Q40_PAYLOAD_BYTES = llaminar2::cuda_native_payload::CodebookTraits<0>::payload_bytes;
    constexpr int SMEM_PAD = 16;
    constexpr int SMEM_STRIDE = BK + SMEM_PAD;
    constexpr int STAGES = 2;
    constexpr int MIN_GRID_BLOCKS = 64;

    enum class Q40PrefillShape
    {
        TpNarrow,
        Balanced,
        Wide,
    };

    static thread_local Q40PrefillShape g_last_q40_prefill_shape = Q40PrefillShape::Balanced;
    static std::atomic<uint64_t> g_q40_prefill_shape_counts[3] = {};

    __device__ __forceinline__ int frag_row(int lane_id, int elem)
    {
        return (elem >> 1) * 8 + (lane_id >> 2);
    }

    __device__ __forceinline__ int frag_col(int lane_id, int elem)
    {
        return (lane_id & 3) * 2 + (elem & 1);
    }

    __device__ __forceinline__ void cp_async_cg_16_zfill_128(
        void *smem_dst, const void *gmem_src, int src_size)
    {
        const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gmem_src), "r"(src_size) : "memory");
    }

    __device__ __forceinline__ void cp_async_commit()
    {
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    template <int N>
    __device__ __forceinline__ void cp_async_wait()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
    }

    __device__ __forceinline__ void load_ldmatrix_a_m16n8k32(
        uint32_t frag[4],
        const int *smem_base,
        int stride_words,
        int lane_id)
    {
#if __CUDA_ARCH__ >= 800
        const int *xs = smem_base + (lane_id % 16) * stride_words + (lane_id / 16) * 4;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
                     : "l"(xs));
#else
        (void)frag;
        (void)smem_base;
        (void)stride_words;
        (void)lane_id;
#endif
    }

    __device__ __forceinline__ void load_ldmatrix_b_m16n8k32(
        uint32_t frag[2],
        const int *smem_base,
        int stride_words,
        int lane_id)
    {
#if __CUDA_ARCH__ >= 800
        const int *xs = smem_base + (lane_id % 8) * stride_words + ((lane_id / 8) * 4) % 8;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                     : "=r"(frag[0]), "=r"(frag[1])
                     : "l"(xs));
#else
        (void)frag;
        (void)smem_base;
        (void)stride_words;
        (void)lane_id;
#endif
    }

    __device__ __forceinline__ void mma_m16n8k32_s8(
        int32_t D[4],
        const uint32_t A[4],
        const uint32_t B[2])
    {
#if __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%0, %1, %2, %3};\n"
            : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
              "r"(B[0]), "r"(B[1]));
#else
        (void)D;
        (void)A;
        (void)B;
#endif
    }

    template <int BN, int BLOCK_SIZE>
    __device__ __forceinline__ void load_q40_tile_to_smem(
        const uint8_t *raw_payload,
        const uint16_t *scales_B,
        int8_t *decoded_tile,
        uint16_t *decoded_scales,
        int kb,
        int block_n,
        int N)
    {
        for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
        {
            int32_t *dst_words = reinterpret_cast<int32_t *>(decoded_tile + col * SMEM_STRIDE);
            const int gcol = block_n + col;
            if (gcol < N)
            {
                int32_t packed_groups[8];
                const size_t linear = static_cast<size_t>(kb) * N + gcol;
                llaminar2::cuda_native_payload::decode_groups_vec<0>(
                    raw_payload + static_cast<size_t>(col) * Q40_PAYLOAD_BYTES,
                    packed_groups);
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = packed_groups[g];
                decoded_scales[col] = scales_B[linear];
            }
            else
            {
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = 0;
                decoded_scales[col] = uint16_t{0};
            }
        }
    }

    template <int BN, int BLOCK_SIZE>
    __device__ __forceinline__ void decode_q40_raw_tile_to_smem(
        const uint8_t *raw_payload,
        int8_t *decoded_tile,
        int valid_cols)
    {
        for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
        {
            int32_t *dst_words = reinterpret_cast<int32_t *>(decoded_tile + col * SMEM_STRIDE);
            if (col < valid_cols)
            {
                int32_t packed_groups[8];
                llaminar2::cuda_native_payload::decode_groups_vec<0>(
                    raw_payload + static_cast<size_t>(col) * Q40_PAYLOAD_BYTES,
                    packed_groups);
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = packed_groups[g];
            }
            else
            {
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    dst_words[g] = 0;
            }
        }
    }

    template <int BM, int BN, int WARPS_M, int WARPS_N, int SPLIT_K = 1, bool SINGLE_PASS_MATERIALIZE = false>
    __global__ void q40NativePayloadTensorCoreKernel(
        const int8_t *__restrict__ A,
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ scales_B,
        float *__restrict__ C,
        const float *__restrict__ scales_A,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int M,
        int N,
        int K,
        float alpha,
        float beta)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        constexpr int WARP_M = BM / WARPS_M;
        constexpr int WARP_N = BN / WARPS_N;
        constexpr int WM = WARP_M / 16;
        constexpr int WN = WARP_N / 8;
        constexpr int A_VEC_LOADS = BM * BK / 16;

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0);
        static_assert(WARP_N % 8 == 0);

        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N;
        const int wc = warp_id % WARPS_N;
        const int gid = lane_id >> 2;

        const int block_m = blockIdx.x * BM;
        const int block_n = blockIdx.y * BN;

        const int num_k_blocks_total = K / BK;
        int kb_begin = 0;
        int kb_end = num_k_blocks_total;
        if constexpr (SPLIT_K > 1)
        {
            const int blocks_per_part = (num_k_blocks_total + SPLIT_K - 1) / SPLIT_K;
            kb_begin = static_cast<int>(blockIdx.z) * blocks_per_part;
            kb_end = min(kb_begin + blocks_per_part, num_k_blocks_total);
        }
        const int num_k_iters = kb_end - kb_begin;
        if (num_k_iters <= 0)
            return;

        __shared__ int8_t smem_A[STAGES][BM * SMEM_STRIDE];
        __shared__ int8_t smem_B[STAGES][BN * SMEM_STRIDE];
        __shared__ uint8_t smem_B_raw[STAGES][BN * Q40_PAYLOAD_BYTES];
        __shared__ uint16_t smem_scales_B[STAGES][BN];

        float acc[WM][WN][4];
#pragma unroll
        for (int i = 0; i < WM; ++i)
#pragma unroll
            for (int j = 0; j < WN; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    acc[i][j][e] = 0.0f;

        auto load_A_tile = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll 4
            for (int idx = threadIdx.x; idx < A_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4;
                const int row = linear >> 5;
                const int col = linear & 31;
                const int grow = block_m + row;
                void *dst = &smem_A[stage][row * SMEM_STRIDE + col];
                const bool valid = grow < M;
                const void *src = valid
                                      ? static_cast<const void *>(&A[static_cast<size_t>(grow) * K + kb * BK + col])
                                      : static_cast<const void *>(A);
                cp_async_cg_16_zfill_128(dst, src, valid ? 16 : 0);
            }
        };

        auto load_B_payload_tile = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll 2
            for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
            {
                const int gcol = block_n + col;
                void *dst = &smem_B_raw[stage][col * Q40_PAYLOAD_BYTES];
                const bool valid = gcol < N;
                const void *src = valid
                                      ? static_cast<const void *>(&payload[(static_cast<size_t>(kb) * N + gcol) * Q40_PAYLOAD_BYTES])
                                      : static_cast<const void *>(payload);
                cp_async_cg_16_zfill_128(dst, src, valid ? Q40_PAYLOAD_BYTES : 0);
            }
        };

        auto materialize_B_stage = [&](int stage, int kb) __attribute__((always_inline))
        {
            if constexpr (SINGLE_PASS_MATERIALIZE)
            {
                load_q40_tile_to_smem<BN, BLOCK_SIZE>(
                    smem_B_raw[stage],
                    scales_B,
                    smem_B[stage],
                    smem_scales_B[stage],
                    kb,
                    block_n,
                    N);
            }
            else
            {
                const int valid_cols = max(0, min(BN, N - block_n));
                for (int col = threadIdx.x; col < BN; col += BLOCK_SIZE)
                {
                    const int gcol = block_n + col;
                    smem_scales_B[stage][col] = (gcol < N)
                                                    ? scales_B[static_cast<size_t>(kb) * N + gcol]
                                                    : uint16_t{0};
                }
                __syncthreads();
                decode_q40_raw_tile_to_smem<BN, BLOCK_SIZE>(smem_B_raw[stage], smem_B[stage], valid_cols);
            }
        };

        auto compute_k_block = [&](int stage, int kb) __attribute__((always_inline))
        {
#pragma unroll
            for (int wi = 0; wi < WM; ++wi)
            {
                const int a_row_base = wr * WARP_M + wi * 16;

                uint32_t A_frag[4];
                load_ldmatrix_a_m16n8k32(
                    A_frag,
                    reinterpret_cast<const int *>(&smem_A[stage][a_row_base * SMEM_STRIDE]),
                    SMEM_STRIDE / 4,
                    lane_id);

                const int grow0 = block_m + a_row_base + gid;
                const int grow1 = grow0 + 8;
                const float sa0 = (grow0 < M) ? scales_A[grow0 * num_k_blocks_total + kb] : 0.0f;
                const float sa1 = (grow1 < M) ? scales_A[grow1 * num_k_blocks_total + kb] : 0.0f;

#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int b_col_base = wc * WARP_N + wj * 8;
                    const int gc0 = block_n + b_col_base + frag_col(lane_id, 0);
                    const int gc1 = block_n + b_col_base + frag_col(lane_id, 1);
                    const float sb0 = (gc0 < N) ? fp16_bits_to_float(smem_scales_B[stage][gc0 - block_n]) : 0.0f;
                    const float sb1 = (gc1 < N) ? fp16_bits_to_float(smem_scales_B[stage][gc1 - block_n]) : 0.0f;

                    uint32_t B_frag[2];
                    load_ldmatrix_b_m16n8k32(
                        B_frag,
                        reinterpret_cast<const int *>(&smem_B[stage][b_col_base * SMEM_STRIDE]),
                        SMEM_STRIDE / 4,
                        lane_id);

                    int32_t D_frag[4] = {0, 0, 0, 0};
                    mma_m16n8k32_s8(D_frag, A_frag, B_frag);

#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        const float sa = (e < 2) ? sa0 : sa1;
                        const float sb = (e & 1) ? sb1 : sb0;
                        acc[wi][wj][e] += static_cast<float>(D_frag[e]) * sa * sb;
                    }
                }
            }
        };

        load_A_tile(0, kb_begin);
        load_B_payload_tile(0, kb_begin);
        cp_async_commit();
        cp_async_wait<0>();
        materialize_B_stage(0, kb_begin);
        if constexpr (SINGLE_PASS_MATERIALIZE)
            __syncthreads();
        else
            __syncthreads();

        for (int ki = 0; ki < num_k_iters; ++ki)
        {
            const int stage = ki & 1;
            const int kb = kb_begin + ki;

            if (ki + 1 < num_k_iters)
            {
                load_A_tile(stage ^ 1, kb + 1);
                load_B_payload_tile(stage ^ 1, kb + 1);
                cp_async_commit();
            }

            compute_k_block(stage, kb);

            if (ki + 1 < num_k_iters)
            {
                cp_async_wait<0>();
                materialize_B_stage(stage ^ 1, kb + 1);
                if constexpr (SINGLE_PASS_MATERIALIZE)
                    __syncthreads();
                else
                    __syncthreads();
            }
        }

        const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

#pragma unroll
        for (int wj = 0; wj < WN; ++wj)
        {
            const int tile_n = block_n + wc * WARP_N + wj * 8;
            const int gc0 = tile_n + frag_col(lane_id, 0);
            const int gc1 = tile_n + frag_col(lane_id, 1);
            const bool gc0_valid = gc0 < N;
            const bool gc1_valid = gc1 < N;
            const float bias0 = (bias && gc0_valid) ? bias[gc0] : 0.0f;
            const float bias1 = (bias && gc1_valid) ? bias[gc1] : 0.0f;

#pragma unroll
            for (int wi = 0; wi < WM; ++wi)
            {
                const int tile_m = block_m + wr * WARP_M + wi * 16;
                const bool interior = (tile_m + 15 < M) && (tile_n + 7 < N);

                if (interior && simple_epilogue)
                {
                    const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                    const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                    const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                    const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;

                    if constexpr (SPLIT_K > 1)
                    {
                        atomicAdd(&C[out_idx0], acc[wi][wj][0] * alpha);
                        atomicAdd(&C[out_idx1], acc[wi][wj][1] * alpha);
                        atomicAdd(&C[out_idx2], acc[wi][wj][2] * alpha);
                        atomicAdd(&C[out_idx3], acc[wi][wj][3] * alpha);
                    }
                    else
                    {
                        C[out_idx0] = acc[wi][wj][0] * alpha;
                        C[out_idx1] = acc[wi][wj][1] * alpha;
                        C[out_idx2] = acc[wi][wj][2] * alpha;
                        C[out_idx3] = acc[wi][wj][3] * alpha;
                    }
                    continue;
                }

#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const int gr = tile_m + frag_row(lane_id, e);
                    const int gc = (e & 1) ? gc1 : gc0;

                    if (gr < M && gc < N)
                    {
                        const int out_idx = gr * N + gc;
                        float val = acc[wi][wj][e] * alpha;

                        if constexpr (SPLIT_K > 1)
                        {
                            if (blockIdx.z == 0)
                            {
                                if (beta != 0.0f && C_existing)
                                    val += beta * C_existing[out_idx];
                                if (bias)
                                    val += (e & 1) ? bias1 : bias0;
                            }
                            atomicAdd(&C[out_idx], val);
                        }
                        else
                        {
                            if (beta != 0.0f && C_existing)
                                val += beta * C_existing[out_idx];
                            if (bias)
                                val += (e & 1) ? bias1 : bias0;
                            C[out_idx] = val;
                        }
                    }
                }
            }
        }
#else
        (void)A;
        (void)payload;
        (void)scales_B;
        (void)C;
        (void)scales_A;
        (void)C_existing;
        (void)bias;
        (void)M;
        (void)N;
        (void)K;
        (void)alpha;
        (void)beta;
#endif
    }

    constexpr int q40PrefillShapeIndex(Q40PrefillShape shape)
    {
        switch (shape)
        {
        case Q40PrefillShape::TpNarrow:
            return 0;
        case Q40PrefillShape::Balanced:
            return 1;
        case Q40PrefillShape::Wide:
            return 2;
        }
        return 1;
    }

    const char *q40PrefillShapeName(Q40PrefillShape shape)
    {
        switch (shape)
        {
        case Q40PrefillShape::TpNarrow:
            return "native_q40_tc_tp_narrow";
        case Q40PrefillShape::Balanced:
            return "native_q40_tc_balanced";
        case Q40PrefillShape::Wide:
            return "native_q40_tc_wide";
        }
        return "native_q40_tc_balanced";
    }

    void recordQ40PrefillShape(Q40PrefillShape shape)
    {
        g_last_q40_prefill_shape = shape;
        g_q40_prefill_shape_counts[q40PrefillShapeIndex(shape)].fetch_add(1, std::memory_order_relaxed);
    }

    int roundUpPow2Clamped(int value, int max_value)
    {
        int out = 1;
        while (out < value && out < max_value)
            out <<= 1;
        return out;
    }

    int chooseSplitK(Q40PrefillShape shape, int M, int N, int K, int bm, int bn)
    {
        const int grid_blocks = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
        const int num_k_blocks = K / BK;
        const bool k_rich = (K >= 2 * N);
        const bool recover_underfill =
            grid_blocks < MIN_GRID_BLOCKS &&
            num_k_blocks >= 32;

        if (!k_rich && !recover_underfill)
            return 1;
        if (!recover_underfill && grid_blocks >= MIN_GRID_BLOCKS)
            return 1;
        if (num_k_blocks < 4)
            return 1;

        const int needed = (MIN_GRID_BLOCKS + grid_blocks - 1) / grid_blocks;
        int split_k = roundUpPow2Clamped(needed, 8);
        const int min_k_blocks_per_partition = k_rich ? 2 : 8;
        while (split_k > 1 && (num_k_blocks / split_k) < min_k_blocks_per_partition)
            split_k >>= 1;
        return std::max(1, split_k);
    }

    Q40PrefillShape classifyQ40PrefillShape(int M, int N, int K)
    {
        (void)M;
        if (N <= 1536)
            return Q40PrefillShape::TpNarrow;
        if (N <= 3072 && (2 * N) <= K)
            return Q40PrefillShape::TpNarrow;
        if (N >= 8192)
            return Q40PrefillShape::Wide;
        if ((4 * N) >= (9 * K))
            return Q40PrefillShape::Wide;
        return Q40PrefillShape::Balanced;
    }

    bool isAmperePlus(int device_id)
    {
        static int cached_device = -1;
        static bool cached_result = false;
        if (cached_device == device_id)
            return cached_result;

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
            return false;

        cached_result = (prop.major >= 8);
        cached_device = device_id;
        return cached_result;
    }

    template <int BM, int BN, int WM, int WN, int SPLIT_K = 1, bool SINGLE_PASS_MATERIALIZE = false>
    bool launchQ40TensorCoreVariant(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream)
    {
        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        if constexpr (SPLIT_K > 1)
            cudaMemsetAsync(d_C_fp32, 0, static_cast<size_t>(M) * N * sizeof(float), cuda_stream);

        q40NativePayloadTensorCoreKernel<BM, BN, WM, WN, SPLIT_K, SINGLE_PASS_MATERIALIZE><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_C_fp32,
            d_scales_A_block,
            d_C_existing,
            d_bias,
            M,
            N,
            K,
            alpha,
            beta);
        return cudaGetLastError() == cudaSuccess;
    }
}

extern "C"
{
    bool cudaNativePayloadPrefillQ40_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % BK) != 0)
            return false;
        if (!isAmperePlus(cuda_device_id))
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        const Q40PrefillShape shape = classifyQ40PrefillShape(M, N, K);
        recordQ40PrefillShape(shape);

        switch (shape)
        {
        case Q40PrefillShape::TpNarrow:
        {
            const int split_k = chooseSplitK(shape, M, N, K, 128, 64);
            switch (split_k)
            {
            case 8:
                return launchQ40TensorCoreVariant<128, 64, 2, 1, 8, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 4:
                return launchQ40TensorCoreVariant<128, 64, 2, 1, 4, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 2:
                return launchQ40TensorCoreVariant<128, 64, 2, 1, 2, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            default:
                return launchQ40TensorCoreVariant<128, 64, 2, 1, 1, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            }
        }

        case Q40PrefillShape::Wide:
        {
            const int split_k = chooseSplitK(shape, M, N, K, 64, 128);
            switch (split_k)
            {
            case 8:
                return launchQ40TensorCoreVariant<64, 128, 2, 2, 8, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 4:
                return launchQ40TensorCoreVariant<64, 128, 2, 2, 4, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 2:
                return launchQ40TensorCoreVariant<64, 128, 2, 2, 2, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            default:
                return launchQ40TensorCoreVariant<64, 128, 2, 2, 1, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            }
        }

        case Q40PrefillShape::Balanced:
        default:
        {
            const int split_k = chooseSplitK(shape, M, N, K, 128, 128);
            switch (split_k)
            {
            case 8:
                return launchQ40TensorCoreVariant<128, 128, 2, 2, 8, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 4:
                return launchQ40TensorCoreVariant<128, 128, 2, 2, 4, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            case 2:
                return launchQ40TensorCoreVariant<128, 128, 2, 2, 2, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            default:
                return launchQ40TensorCoreVariant<128, 128, 2, 2, 1, true>(
                    d_A_int8, d_payload, d_scales, d_C_fp32, d_scales_A_block,
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream);
            }
        }
        }
    }

    const char *cudaNativePayloadPrefillQ40_lastSelectedFamily()
    {
        return q40PrefillShapeName(g_last_q40_prefill_shape);
    }

    unsigned long long cudaNativePayloadPrefillQ40_getFamilyCount(int family_index)
    {
        if (family_index < 0 || family_index >= 3)
            return 0;
        return g_q40_prefill_shape_counts[family_index].load(std::memory_order_relaxed);
    }

    void cudaNativePayloadPrefillQ40_resetFamilyCounts()
    {
        for (auto &count : g_q40_prefill_shape_counts)
            count.store(0, std::memory_order_relaxed);
    }
}
