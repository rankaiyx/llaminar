/**
 * @file cuda_hip_context_probe.cu
 * @brief Diagnostic tool to identify exactly what HIP corrupts in CUDA state
 *
 * This program systematically tests CUDA functionality before and after HIP
 * initialization to identify the precise state that gets corrupted.
 *
 * Build with:
 *   nvcc -o cuda_hip_context_probe cuda_hip_context_probe.cu -lcuda -I/opt/rocm/include -L/opt/rocm/lib -lamdhip64
 *
 * Run with:
 *   ./cuda_hip_context_probe
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <dlfcn.h>

// Forward declare HIP types to avoid header conflicts
typedef void *hipStream_t;
typedef int hipError_t;
#define hipSuccess 0

// HIP function pointers (loaded dynamically to avoid link-time conflicts)
typedef hipError_t (*hipInit_t)(unsigned int);
typedef hipError_t (*hipSetDevice_t)(int);
typedef hipError_t (*hipGetDeviceCount_t)(int *);
typedef hipError_t (*hipMalloc_t)(void **, size_t);
typedef hipError_t (*hipFree_t)(void *);
typedef hipError_t (*hipDeviceSynchronize_t)();
typedef hipError_t (*hipMemset_t)(void *, int, size_t);
typedef const char *(*hipGetErrorString_t)(hipError_t);

static hipInit_t hip_init = nullptr;
static hipSetDevice_t hip_set_device = nullptr;
static hipGetDeviceCount_t hip_get_device_count = nullptr;
static hipMalloc_t hip_malloc = nullptr;
static hipFree_t hip_free = nullptr;
static hipDeviceSynchronize_t hip_device_synchronize = nullptr;
static hipMemset_t hip_memset = nullptr;
static hipGetErrorString_t hip_get_error_string = nullptr;

// Simple CUDA kernel for testing
__global__ void simpleAddKernel(float *out, const float *in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        out[idx] = in[idx] + 1.0f;
    }
}

// Structure to capture CUDA context state
struct CUDAContextState
{
    int current_device;
    CUcontext driver_context;
    CUdevice driver_device;
    unsigned int primary_ctx_flags;
    int primary_ctx_active;
    size_t free_mem;
    size_t total_mem;
    cudaStream_t default_stream_status;

    void capture(int device = 0)
    {
        cudaGetDevice(&current_device);
        cuCtxGetCurrent(&driver_context);
        cuDeviceGet(&driver_device, device);
        cuDevicePrimaryCtxGetState(driver_device, &primary_ctx_flags, &primary_ctx_active);
        cudaMemGetInfo(&free_mem, &total_mem);
    }

    void print(const char *label) const
    {
        printf("\n=== %s ===\n", label);
        printf("  Runtime current device: %d\n", current_device);
        printf("  Driver context: %p\n", (void *)driver_context);
        printf("  Driver device: %d\n", driver_device);
        printf("  Primary ctx flags: %u\n", primary_ctx_flags);
        printf("  Primary ctx active: %d\n", primary_ctx_active);
        printf("  Memory: %zu MB free / %zu MB total\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));
    }

    bool compare(const CUDAContextState &other, const char *label) const
    {
        printf("\n=== Comparing %s ===\n", label);
        bool changed = false;

        if (current_device != other.current_device)
        {
            printf("  [CHANGED] current_device: %d -> %d\n", current_device, other.current_device);
            changed = true;
        }
        if (driver_context != other.driver_context)
        {
            printf("  [CHANGED] driver_context: %p -> %p\n", (void *)driver_context, (void *)other.driver_context);
            changed = true;
        }
        if (driver_device != other.driver_device)
        {
            printf("  [CHANGED] driver_device: %d -> %d\n", driver_device, other.driver_device);
            changed = true;
        }
        if (primary_ctx_flags != other.primary_ctx_flags)
        {
            printf("  [CHANGED] primary_ctx_flags: %u -> %u\n", primary_ctx_flags, other.primary_ctx_flags);
            changed = true;
        }
        if (primary_ctx_active != other.primary_ctx_active)
        {
            printf("  [CHANGED] primary_ctx_active: %d -> %d\n", primary_ctx_active, other.primary_ctx_active);
            changed = true;
        }

        if (!changed)
        {
            printf("  No observable state changes\n");
        }
        return changed;
    }
};

bool loadHIPRuntime()
{
    void *hip_lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
    if (!hip_lib)
    {
        printf("Failed to load libamdhip64.so: %s\n", dlerror());
        return false;
    }

    hip_init = (hipInit_t)dlsym(hip_lib, "hipInit");
    hip_set_device = (hipSetDevice_t)dlsym(hip_lib, "hipSetDevice");
    hip_get_device_count = (hipGetDeviceCount_t)dlsym(hip_lib, "hipGetDeviceCount");
    hip_malloc = (hipMalloc_t)dlsym(hip_lib, "hipMalloc");
    hip_free = (hipFree_t)dlsym(hip_lib, "hipFree");
    hip_device_synchronize = (hipDeviceSynchronize_t)dlsym(hip_lib, "hipDeviceSynchronize");
    hip_memset = (hipMemset_t)dlsym(hip_lib, "hipMemset");
    hip_get_error_string = (hipGetErrorString_t)dlsym(hip_lib, "hipGetErrorString");

    if (!hip_init || !hip_set_device || !hip_get_device_count ||
        !hip_malloc || !hip_free || !hip_device_synchronize)
    {
        printf("Failed to load HIP symbols\n");
        return false;
    }

    return true;
}

bool testCUDAKernelLaunch(const char *phase, int device = 0)
{
    printf("\n--- Testing CUDA kernel launch (%s) ---\n", phase);

    cudaError_t err;

    // Set device
    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        printf("  cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("  cudaSetDevice(%d): OK\n", device);

    // Allocate memory
    float *d_in = nullptr, *d_out = nullptr;
    const int N = 256;

    err = cudaMalloc(&d_in, N * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("  cudaMalloc(d_in) failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    printf("  cudaMalloc(d_in): OK (%p)\n", d_in);

    err = cudaMalloc(&d_out, N * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("  cudaMalloc(d_out) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        return false;
    }
    printf("  cudaMalloc(d_out): OK (%p)\n", d_out);

    // Initialize input
    err = cudaMemset(d_in, 0, N * sizeof(float));
    if (err != cudaSuccess)
    {
        printf("  cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }
    printf("  cudaMemset: OK\n");

    // Test stream creation
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
    {
        printf("  cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }
    printf("  cudaStreamCreate: OK\n");

    // Launch kernel
    printf("  Launching kernel...\n");
    simpleAddKernel<<<1, N, 0, stream>>>(d_out, d_in, N);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("  KERNEL LAUNCH FAILED: %s (code %d)\n", cudaGetErrorString(err), (int)err);
        cudaStreamDestroy(stream);
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }
    printf("  Kernel launch: OK\n");

    // Sync
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        printf("  cudaStreamSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(stream);
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }
    printf("  cudaStreamSynchronize: OK\n");

    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("  CUDA kernel test PASSED\n");
    return true;
}

bool testCUDAKernelOnNewThread(const char *phase)
{
    printf("\n--- Testing CUDA kernel on NEW thread (%s) ---\n", phase);

    bool result = false;
    std::thread t([&result, phase]()
                  { result = testCUDAKernelLaunch(phase, 0); });
    t.join();

    return result;
}

bool initializeHIP()
{
    printf("\n=== Initializing HIP Runtime ===\n");

    hipError_t err = hip_init(0);
    if (err != hipSuccess)
    {
        printf("  hipInit failed: %s\n", hip_get_error_string(err));
        return false;
    }
    printf("  hipInit: OK\n");

    int device_count = 0;
    err = hip_get_device_count(&device_count);
    if (err != hipSuccess)
    {
        printf("  hipGetDeviceCount failed: %s\n", hip_get_error_string(err));
        return false;
    }
    printf("  HIP device count: %d\n", device_count);

    if (device_count > 0)
    {
        err = hip_set_device(0);
        if (err != hipSuccess)
        {
            printf("  hipSetDevice failed: %s\n", hip_get_error_string(err));
            return false;
        }
        printf("  hipSetDevice(0): OK\n");
    }

    return true;
}

bool doHIPAllocation()
{
    printf("\n=== Doing HIP Memory Allocation ===\n");

    void *d_ptr = nullptr;
    hipError_t err = hip_malloc(&d_ptr, 1024);
    if (err != hipSuccess)
    {
        printf("  hipMalloc failed: %s\n", hip_get_error_string(err));
        return false;
    }
    printf("  hipMalloc: OK (%p)\n", d_ptr);

    err = hip_memset(d_ptr, 0, 1024);
    if (err != hipSuccess)
    {
        printf("  hipMemset failed: %s\n", hip_get_error_string(err));
        hip_free(d_ptr);
        return false;
    }
    printf("  hipMemset: OK\n");

    err = hip_device_synchronize();
    if (err != hipSuccess)
    {
        printf("  hipDeviceSynchronize failed: %s\n", hip_get_error_string(err));
        hip_free(d_ptr);
        return false;
    }
    printf("  hipDeviceSynchronize: OK\n");

    hip_free(d_ptr);
    printf("  HIP allocation test PASSED\n");
    return true;
}

bool tryContextRecovery()
{
    printf("\n=== Attempting Context Recovery ===\n");

    // Method 1: Re-retain primary context
    printf("\n[Recovery 1] Re-retain primary context\n");
    CUdevice device;
    CUresult cu_err = cuDeviceGet(&device, 0);
    if (cu_err != CUDA_SUCCESS)
    {
        printf("  cuDeviceGet failed: %d\n", cu_err);
    }
    else
    {
        CUcontext ctx;
        cu_err = cuDevicePrimaryCtxRetain(&ctx, device);
        if (cu_err != CUDA_SUCCESS)
        {
            printf("  cuDevicePrimaryCtxRetain failed: %d\n", cu_err);
        }
        else
        {
            printf("  cuDevicePrimaryCtxRetain: OK (ctx=%p)\n", ctx);
            cu_err = cuCtxSetCurrent(ctx);
            if (cu_err != CUDA_SUCCESS)
            {
                printf("  cuCtxSetCurrent failed: %d\n", cu_err);
            }
            else
            {
                printf("  cuCtxSetCurrent: OK\n");
            }
        }
    }

    if (testCUDAKernelLaunch("after recovery 1"))
    {
        printf("  Recovery 1 SUCCEEDED!\n");
        return true;
    }

    // Method 2: Retain and push a fresh primary context
    printf("\n[Recovery 2] Use primary context with explicit push\n");
    CUcontext new_ctx;
    CUdevice cu_device;
    cuDeviceGet(&cu_device, 0);
    cu_err = cuDevicePrimaryCtxRetain(&new_ctx, cu_device);
    if (cu_err != CUDA_SUCCESS)
    {
        printf("  cuDevicePrimaryCtxRetain failed: %d\n", cu_err);
    }
    else
    {
        printf("  cuDevicePrimaryCtxRetain: OK (ctx=%p)\n", new_ctx);
        cuCtxPushCurrent(new_ctx);
        if (testCUDAKernelLaunch("after recovery 2"))
        {
            printf("  Recovery 2 SUCCEEDED!\n");
            cuCtxPopCurrent(nullptr);
            cuDevicePrimaryCtxRelease(cu_device);
            return true;
        }
        cuCtxPopCurrent(nullptr);
        cuDevicePrimaryCtxRelease(cu_device);
    }

    // Method 3: Reset device
    printf("\n[Recovery 3] cudaDeviceReset\n");
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        printf("  cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("  cudaDeviceReset: OK\n");
        if (testCUDAKernelLaunch("after recovery 3"))
        {
            printf("  Recovery 3 SUCCEEDED!\n");
            return true;
        }
    }

    printf("\n  All recovery methods FAILED\n");
    return false;
}

void testThreadLocalState()
{
    printf("\n=== Testing Thread-Local State Isolation ===\n");

    // Test 1: Does HIP on separate thread affect main thread CUDA?
    printf("\n[Test] HIP on separate thread\n");

    // First verify CUDA works on main thread
    if (!testCUDAKernelLaunch("main thread before HIP"))
    {
        printf("  CUDA already broken before HIP!\n");
        return;
    }

    // Capture state before
    CUDAContextState before;
    before.capture();
    before.print("Before HIP thread");

    // Do HIP on separate thread
    std::thread hip_thread([]()
                           {
        printf("\n  [HIP Thread] Starting\n");
        if (!loadHIPRuntime()) {
            printf("  [HIP Thread] Failed to load HIP\n");
            return;
        }
        if (!initializeHIP()) {
            printf("  [HIP Thread] Failed to init HIP\n");
            return;
        }
        if (!doHIPAllocation()) {
            printf("  [HIP Thread] Failed HIP allocation\n");
            return;
        }
        printf("  [HIP Thread] Done\n"); });
    hip_thread.join();

    // Capture state after
    CUDAContextState after;
    after.capture();
    after.print("After HIP thread");

    before.compare(after, "HIP thread impact");

    // Test if CUDA still works
    if (testCUDAKernelLaunch("main thread after HIP thread"))
    {
        printf("\n  HIP on separate thread DID NOT break CUDA\n");
    }
    else
    {
        printf("\n  HIP on separate thread DID break CUDA\n");
    }
}

int main()
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║       CUDA/HIP Context Corruption Diagnostic Tool           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    // Initialize CUDA driver API
    CUresult cu_err = cuInit(0);
    if (cu_err != CUDA_SUCCESS)
    {
        printf("cuInit failed: %d\n", cu_err);
        return 1;
    }
    printf("cuInit: OK\n");

    int cuda_device_count = 0;
    cudaGetDeviceCount(&cuda_device_count);
    printf("CUDA device count: %d\n", cuda_device_count);

    if (cuda_device_count == 0)
    {
        printf("No CUDA devices found!\n");
        return 1;
    }

    // ========================================
    // Phase 1: Baseline CUDA functionality
    // ========================================
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 1: Baseline CUDA Functionality              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    CUDAContextState state_phase1;
    state_phase1.capture();
    state_phase1.print("Phase 1 - Initial state");

    if (!testCUDAKernelLaunch("Phase 1 - baseline"))
    {
        printf("\nFATAL: CUDA kernel doesn't work even before HIP!\n");
        return 1;
    }

    // ========================================
    // Phase 2: Load HIP (but don't init)
    // ========================================
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 2: Load HIP Library (no init)               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    if (!loadHIPRuntime())
    {
        printf("Failed to load HIP runtime\n");
        return 1;
    }

    CUDAContextState state_phase2;
    state_phase2.capture();
    state_phase1.compare(state_phase2, "After HIP library load");

    if (!testCUDAKernelLaunch("Phase 2 - after HIP load"))
    {
        printf("\n*** CUDA BROKEN after loading HIP library ***\n");
    }

    // ========================================
    // Phase 3: Initialize HIP
    // ========================================
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 3: Initialize HIP Runtime                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    if (!initializeHIP())
    {
        printf("Failed to initialize HIP\n");
        return 1;
    }

    CUDAContextState state_phase3;
    state_phase3.capture();
    state_phase2.compare(state_phase3, "After HIP init");

    if (!testCUDAKernelLaunch("Phase 3 - after HIP init"))
    {
        printf("\n*** CUDA BROKEN after HIP init ***\n");
    }

    // ========================================
    // Phase 4: Do HIP memory operations
    // ========================================
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 4: HIP Memory Operations                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    if (!doHIPAllocation())
    {
        printf("HIP allocation failed\n");
    }

    CUDAContextState state_phase4;
    state_phase4.capture();
    state_phase3.compare(state_phase4, "After HIP allocation");

    if (!testCUDAKernelLaunch("Phase 4 - after HIP allocation"))
    {
        printf("\n*** CUDA BROKEN after HIP memory operations ***\n");
    }

    // ========================================
    // Phase 5: Test on new thread
    // ========================================
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 5: CUDA on New Thread                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    if (!testCUDAKernelOnNewThread("Phase 5 - new thread"))
    {
        printf("\n*** CUDA BROKEN even on new thread ***\n");
    }
    else
    {
        printf("\n*** CUDA WORKS on new thread! ***\n");
    }

    // ========================================
    // Phase 6: Recovery attempts
    // ========================================
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           PHASE 6: Context Recovery Attempts                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    if (!testCUDAKernelLaunch("Phase 6 - before recovery"))
    {
        tryContextRecovery();
    }

    // ========================================
    // Summary
    // ========================================
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                        SUMMARY                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    printf("\nFinal context state:\n");
    CUDAContextState final_state;
    final_state.capture();
    final_state.print("Final");

    printf("\nChanges from initial state:\n");
    state_phase1.compare(final_state, "Initial -> Final");

    return 0;
}
