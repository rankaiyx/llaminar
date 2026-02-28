/**
 * @file ROCmArgmaxKernels.hip
 * @brief ROCm/HIP Argmax kernel for greedy sampling
 *
 * Provides a GPU-side argmax over FP32 logits, returning the index and value
 * of the maximum element. Used for greedy decode sampling to avoid D2H of
 * the entire logits tensor (~600 KB for 152K vocab).
 *
 * Performance: ~5µs for 76K elements on MI50 vs ~170µs for CPU std::max_element
 */

#include <hip/hip_runtime.h>
#include <cfloat>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"

// ============================================================================
// Argmax Reduction Kernel — Single-block shared memory reduction
// ============================================================================
// For vocab_local sizes up to ~150K, a single block with 1024 threads
// is sufficient: each thread scans ~150 elements, then shared mem reduction.
// This avoids the complexity of a multi-block reduction + second kernel pass.

__global__ void rocm_argmax_f32_kernel(
    const float* __restrict__ data,
    int n,
    float* __restrict__ out_value,
    int* __restrict__ out_index)
{
    extern __shared__ char shared_mem[];
    float* smax = reinterpret_cast<float*>(shared_mem);
    int* sidx = reinterpret_cast<int*>(shared_mem + blockDim.x * sizeof(float));

    const int tid = threadIdx.x;

    // Phase 1: Each thread finds the max in its strided portion
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < n; i += blockDim.x)
    {
        float val = data[i];
        if (val > local_max)
        {
            local_max = val;
            local_idx = i;
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    // Phase 2: Tree reduction in shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (smax[tid + stride] > smax[tid])
            {
                smax[tid] = smax[tid + stride];
                sidx[tid] = sidx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Phase 3: Thread 0 writes result
    if (tid == 0)
    {
        *out_value = smax[0];
        *out_index = sidx[0];
    }
}

// ============================================================================
// Extern "C" Wrapper
// ============================================================================

extern "C"
{

/**
 * @brief Launch argmax kernel on device
 *
 * Finds the index and value of the maximum element in a float array.
 * Uses a single block with shared memory reduction.
 *
 * @param data    Device pointer to FP32 data
 * @param n       Number of elements
 * @param out_value Device pointer to store max value (1 float)
 * @param out_index Device pointer to store max index (1 int)
 * @param device_idx GPU device index
 * @param stream  HIP stream (nullptr for default)
 * @return true on success
 */
bool rocmOps_argmax_f32(
    const float* data,
    int n,
    float* out_value,
    int* out_index,
    int device_idx,
    void* stream)
{
    if (n <= 0 || !data || !out_value || !out_index)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_idx);

    // Single block, 1024 threads — good for up to ~1M elements
    // (each thread handles n/1024 elements in the scan phase)
    const int threads = 1024;
    const size_t smem_size = threads * (sizeof(float) + sizeof(int));

    hipLaunchKernelGGL(
        rocm_argmax_f32_kernel,
        dim3(1), dim3(threads), smem_size, static_cast<hipStream_t>(stream),
        data, n, out_value, out_index);

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Argmax FP32 kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
