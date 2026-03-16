/**
 * @file CUDABatchGemmOps.cu
 * @brief CUDA helper kernels for batched GEMM: NativeVNNI weight interleaving
 *        and row-major output splitting.
 *
 * Used to concatenate gate+up (or Q+K+V) NativeVNNI weights into a single
 * weight matrix so the prefill GEMM can run as one kernel launch instead
 * of 2-3 sequential launches.
 *
 * Weight interleaving runs ONCE at init time (before graph capture).
 * Output splitting runs per-inference (inside the GEMM dispatch).
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace
{
    // ─── Weight interleave kernel ────────────────────────────────────
    // NativeVNNI layout: element (b, n) at offset (b * N + n) * element_bytes
    // where b = K-block index [0, blocks_per_row), n = column [0, N)
    //
    // Interleave two weight arrays along the N dimension:
    //   combined(b, n) = src1(b, n)          for n < N1
    //   combined(b, n) = src2(b, n - N1)     for n >= N1
    //
    // For uint4 (16-byte) payload blocks, each thread handles one cell.
    // For uint16_t scales, each thread handles one scale value.

    __global__ void interleave_uint4_kernel(
        const uint4 *__restrict__ src1,
        const uint4 *__restrict__ src2,
        uint4 *__restrict__ dst,
        int N1, int N2, int N_total,
        int blocks_per_row)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_cells = blocks_per_row * N_total;
        if (idx >= total_cells)
            return;

        const int b = idx / N_total;
        const int n = idx % N_total;

        dst[idx] = (n < N1)
                       ? src1[b * N1 + n]
                       : src2[b * N2 + (n - N1)];
    }

    __global__ void interleave_u16_kernel(
        const uint16_t *__restrict__ src1,
        const uint16_t *__restrict__ src2,
        uint16_t *__restrict__ dst,
        int N1, int N2, int N_total,
        int blocks_per_row)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_cells = blocks_per_row * N_total;
        if (idx >= total_cells)
            return;

        const int b = idx / N_total;
        const int n = idx % N_total;

        dst[idx] = (n < N1)
                       ? src1[b * N1 + n]
                       : src2[b * N2 + (n - N1)];
    }

    __global__ void interleave_u32_kernel(
        const uint32_t *__restrict__ src1,
        const uint32_t *__restrict__ src2,
        uint32_t *__restrict__ dst,
        int N1, int N2, int N_total,
        int blocks_per_row)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_cells = blocks_per_row * N_total;
        if (idx >= total_cells)
            return;

        const int b = idx / N_total;
        const int n = idx % N_total;

        dst[idx] = (n < N1)
                       ? src1[b * N1 + n]
                       : src2[b * N2 + (n - N1)];
    }

    // ─── Output split kernel ─────────────────────────────────────────
    // Splits [M, N_total] row-major FP32 into [M, N1] + [M, N2]
    // using vectorized float4 loads/stores where possible.
    //
    // 2D grid: blockIdx.x covers columns, blockIdx.y covers rows.
    // This gives perfect coalescing for each output.

    __global__ void copy_columns_kernel(
        const float *__restrict__ src, // [M, N_total]
        float *__restrict__ dst,       // [M, N_dst]
        int M, int N_dst, int N_total, int col_offset)
    {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y;
        if (col >= N_dst || row >= M)
            return;

        dst[row * N_dst + col] = src[row * N_total + col_offset + col];
    }

} // anonymous namespace

extern "C"
{
    // ─── 2-way NativeVNNI payload interleave ─────────────────────────
    // Interleaves two NativeVNNI payload arrays (uint8_t) into a combined
    // buffer. payload_bytes must be 16 (Q4_0, IQ4_NL) for uint4 path.
    bool cudaBatchGemm_interleavePayloads2(
        const uint8_t *d_src1, int N1,
        const uint8_t *d_src2, int N2,
        uint8_t *d_dst,
        int blocks_per_row,
        int payload_bytes,
        int cuda_device_id,
        void *stream)
    {
        if (!d_src1 || !d_src2 || !d_dst)
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        const int N_total = N1 + N2;
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        if (payload_bytes == 16)
        {
            // uint4 path (16-byte aligned, one thread per payload block)
            const int total_cells = blocks_per_row * N_total;
            const int threads = 256;
            const int blocks = (total_cells + threads - 1) / threads;

            interleave_uint4_kernel<<<blocks, threads, 0, s>>>(
                reinterpret_cast<const uint4 *>(d_src1),
                reinterpret_cast<const uint4 *>(d_src2),
                reinterpret_cast<uint4 *>(d_dst),
                N1, N2, N_total, blocks_per_row);
        }
        else
        {
            // Generic byte-level copy for non-standard payload sizes
            // Use u32 path for 4-byte multiples, otherwise fall back to byte copy
            // For now, only support payload_bytes that are multiples of 16
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    // ─── 2-way NativeVNNI scale interleave ───────────────────────────
    bool cudaBatchGemm_interleaveScales2(
        const uint16_t *d_src1, int N1,
        const uint16_t *d_src2, int N2,
        uint16_t *d_dst,
        int blocks_per_row,
        int cuda_device_id,
        void *stream)
    {
        if (!d_src1 || !d_src2 || !d_dst)
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        const int N_total = N1 + N2;
        const int total_cells = blocks_per_row * N_total;
        const int threads = 256;
        const int blocks = (total_cells + threads - 1) / threads;
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        interleave_u16_kernel<<<blocks, threads, 0, s>>>(
            d_src1, d_src2, d_dst,
            N1, N2, N_total, blocks_per_row);

        return cudaGetLastError() == cudaSuccess;
    }

    // ─── 2-way NativeVNNI mins interleave (asymmetric codebooks) ─────
    bool cudaBatchGemm_interleaveMins2(
        const uint16_t *d_src1, int N1,
        const uint16_t *d_src2, int N2,
        uint16_t *d_dst,
        int blocks_per_row,
        int cuda_device_id,
        void *stream)
    {
        // Same layout as scales
        return cudaBatchGemm_interleaveScales2(
            d_src1, N1, d_src2, N2, d_dst, blocks_per_row, cuda_device_id, stream);
    }

    // ─── 2-way NativeVNNI emins interleave (IQ formats) ─────────────
    bool cudaBatchGemm_interleaveEmins2(
        const uint32_t *d_src1, int N1,
        const uint32_t *d_src2, int N2,
        uint32_t *d_dst,
        int blocks_per_row,
        int cuda_device_id,
        void *stream)
    {
        if (!d_src1 || !d_src2 || !d_dst)
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        const int N_total = N1 + N2;
        const int total_cells = blocks_per_row * N_total;
        const int threads = 256;
        const int blocks = (total_cells + threads - 1) / threads;
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        interleave_u32_kernel<<<blocks, threads, 0, s>>>(
            d_src1, d_src2, d_dst,
            N1, N2, N_total, blocks_per_row);

        return cudaGetLastError() == cudaSuccess;
    }

    // ─── Split [M, N_total] row-major → [M, N1] + [M, N2] ──────────
    bool cudaBatchGemm_splitOutput2(
        const float *d_combined, // [M, N_total] row-major
        float *d_out1, int N1,   // [M, N1]
        float *d_out2, int N2,   // [M, N2]
        int M,
        int cuda_device_id,
        void *stream)
    {
        if (!d_combined || !d_out1 || !d_out2)
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        const int N_total = N1 + N2;
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        const int threads = 256;

        // Copy columns [0, N1) → out1
        {
            dim3 grid((N1 + threads - 1) / threads, M);
            copy_columns_kernel<<<grid, threads, 0, s>>>(
                d_combined, d_out1, M, N1, N_total, 0);
        }

        // Copy columns [N1, N_total) → out2
        {
            dim3 grid((N2 + threads - 1) / threads, M);
            copy_columns_kernel<<<grid, threads, 0, s>>>(
                d_combined, d_out2, M, N2, N_total, N1);
        }

        return cudaGetLastError() == cudaSuccess;
    }

    // ─── Split [M, N_total] row-major → [M, N1] + [M, N2] + [M, N3] ─
    bool cudaBatchGemm_splitOutput3(
        const float *d_combined, // [M, N_total] row-major
        float *d_out1, int N1,
        float *d_out2, int N2,
        float *d_out3, int N3,
        int M,
        int cuda_device_id,
        void *stream)
    {
        if (!d_combined || !d_out1 || !d_out2 || !d_out3)
            return false;
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        const int N_total = N1 + N2 + N3;
        cudaStream_t s = static_cast<cudaStream_t>(stream);

        const int threads = 256;

        // Copy columns [0, N1) → out1
        {
            dim3 grid((N1 + threads - 1) / threads, M);
            copy_columns_kernel<<<grid, threads, 0, s>>>(
                d_combined, d_out1, M, N1, N_total, 0);
        }

        // Copy columns [N1, N1+N2) → out2
        {
            dim3 grid((N2 + threads - 1) / threads, M);
            copy_columns_kernel<<<grid, threads, 0, s>>>(
                d_combined, d_out2, M, N2, N_total, N1);
        }

        // Copy columns [N1+N2, N_total) → out3
        {
            dim3 grid((N3 + threads - 1) / threads, M);
            copy_columns_kernel<<<grid, threads, 0, s>>>(
                d_combined, d_out3, M, N3, N_total, N1 + N2);
        }

        return cudaGetLastError() == cudaSuccess;
    }

    // ─── Device memory helpers ───────────────────────────────────────
    bool cudaBatchGemm_allocDevice(void **d_ptr, size_t bytes, int cuda_device_id)
    {
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;
        return cudaMalloc(d_ptr, bytes) == cudaSuccess;
    }

    void cudaBatchGemm_freeDevice(void *d_ptr)
    {
        if (d_ptr)
        {
            cudaError_t err = cudaFree(d_ptr);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaBatchGemm_freeDevice failed: %s ptr=%p\n",
                        cudaGetErrorString(err), d_ptr);
            }
        }
    }

} // extern "C"
