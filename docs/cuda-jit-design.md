# CUDA JIT Compilation Design

## Architecture: NVRTC-based JIT Compilation

### Current Problem
- **37,380 precompiled variants** = 25-minute build time
- Only use 1-10 configs in practice (autotuner selects best)
- 250 .cu files × ~4MB each = **~1GB of .o files**

### Proposed Solution: NVRTC JIT

Compile kernels on-demand at runtime, cache compiled binaries to disk.

---

## Implementation Design

### 1. Kernel Source Template (String)

```cpp
// src/v2/kernels/cuda/CudaGemmKernelTemplate.cuh
const char* GEMM_KERNEL_TEMPLATE = R"(
#include <cuda_runtime.h>

template <typename Decoder, int TILE_M, int TILE_N, int TILE_K,
          int THREADS_M, int THREADS_N, int WORK_M, int WORK_N,
          int PREFETCH_STAGES, bool TRANSPOSE_SMEM, int VECTORIZE_LOAD>
__global__ void quantized_gemm_kernel_variant(
    const float* __restrict__ A, float* __restrict__ C,
    int m, int n, int k, Decoder decoder)
{
    // ... kernel implementation (copy from CudaGemmVariantsBaseline.cu)
}

// Explicit instantiation with template parameters
template __global__ void quantized_gemm_kernel_variant<
    IQ4_NL_Decoder<IQ4_NLBlock>, 
    ${TILE_M}, ${TILE_N}, ${TILE_K},
    ${THREADS_M}, ${THREADS_N}, ${WORK_M}, ${WORK_N},
    ${PREFETCH_STAGES}, ${TRANSPOSE_SMEM}, ${VECTORIZE_LOAD}>(
    const float*, float*, int, int, int, IQ4_NL_Decoder<IQ4_NLBlock>);
)";
```

### 2. JIT Compilation Manager

```cpp
// src/v2/kernels/cuda/CudaGemmJIT.h
class CudaGemmJIT {
public:
    static CudaGemmJIT& instance();
    
    // Compile kernel for specific config (with caching)
    CUfunction getKernel(const CudaGemmConfig& config);
    
private:
    // In-memory cache (this run)
    std::map<CudaGemmConfig, CUfunction> kernel_cache_;
    
    // Disk cache (persistent across runs)
    std::string cache_dir_ = "~/.cache/llaminar/cuda_kernels/";
    
    // Compile single config
    CUfunction compileKernel(const CudaGemmConfig& config);
    
    // Load from disk cache
    std::optional<CUfunction> loadFromCache(const CudaGemmConfig& config);
    
    // Save to disk cache
    void saveToCache(const CudaGemmConfig& config, const void* cubin, size_t size);
};
```

### 3. NVRTC Compilation Pipeline

```cpp
CUfunction CudaGemmJIT::compileKernel(const CudaGemmConfig& config) {
    // 1. Check disk cache first
    if (auto cached = loadFromCache(config)) {
        return *cached;
    }
    
    // 2. Generate specialized source code
    std::string source = generateKernelSource(config);
    
    // 3. Compile with NVRTC
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, source.c_str(), "gemm_kernel.cu", 0, nullptr, nullptr);
    
    // Compile options (match your build flags)
    const char* opts[] = {
        "--gpu-architecture=compute_75",  // Match your GPU
        "-std=c++17",
        "-use_fast_math",
        "-O3"
    };
    nvrtcCompileProgram(prog, 4, opts);
    
    // 4. Get PTX/CUBIN
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = new char[ptx_size];
    nvrtcGetPTX(prog, ptx);
    
    // 5. Load module and function
    CUmodule module;
    cuModuleLoadDataEx(&module, ptx, 0, nullptr, nullptr);
    
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, "quantized_gemm_kernel_variant");
    
    // 6. Save to disk cache
    saveToCache(config, ptx, ptx_size);
    
    delete[] ptx;
    nvrtcDestroyProgram(&prog);
    
    return kernel;
}
```

### 4. Disk Cache Format

**Cache directory structure:**
```
~/.cache/llaminar/cuda_kernels/
├── sm_75/  (GPU architecture)
│   ├── gemm_64_64_32_16_16_4_4_1_0_2.cubin  (config fingerprint)
│   ├── gemm_64_64_32_16_16_4_4_1_0_2.meta   (metadata: compile time, CUDA version)
│   └── ...
└── sm_80/
    └── ...
```

**Cache key** (config fingerprint):
```cpp
std::string getCacheKey(const CudaGemmConfig& config) {
    return fmt::format("gemm_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.cubin",
        config.tile_m, config.tile_n, config.tile_k,
        config.threads_m, config.threads_n,
        config.work_m, config.work_n,
        config.prefetch, config.transpose ? 1 : 0, config.vectorize);
}
```

### 5. Launch Integration

**Before (Precompiled Registry)**:
```cpp
auto launcher = CudaGemmKernelRegistry::instance().get_launcher(config);
if (launcher != nullptr) {
    return launcher(A, B_blocks, C, m, n, k, gridDim, blockDim, stream);
}
```

**After (JIT)**:
```cpp
// Get JIT-compiled kernel (compiles on first use, cached afterward)
CUfunction kernel = CudaGemmJIT::instance().getKernel(config);

// Launch with CUDA Driver API
IQ4_NL_Decoder decoder(B_blocks, n, k/32);
void* args[] = {&A, &C, &m, &n, &k, &decoder};
cuLaunchKernel(kernel, 
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    0, stream, args, nullptr);
```

---

## Performance Analysis

### Compile Time Comparison

| Approach | First Run | Subsequent Runs | Binary Size |
|----------|-----------|-----------------|-------------|
| **Precompile All** | 25 minutes | 0ms | ~1GB |
| **JIT (no cache)** | 500ms/config | 500ms/config | ~10MB |
| **JIT (cached)** | 500ms × N configs | 5-10ms (load cache) | ~10MB + cache |

**Typical usage** (autotuner finds best 5 configs):
- First run: 500ms × 5 = **2.5 seconds** compile time
- Subsequent runs: **50ms** (load 5 configs from cache)

### Memory Usage

| Approach | Disk Space | RAM (loaded modules) |
|----------|------------|----------------------|
| **Precompile All** | 1GB (.o files) | 500MB (all kernels linked) |
| **JIT (5 active)** | 10MB (source) + 2MB (cache) | 50MB (5 modules) |

---

## Migration Path

### Phase 1: Hybrid (JIT + Fallback)

Keep precompiled registry as fallback, add JIT for new configs:

```cpp
CudaError_t launchGemmKernel(..., const CudaGemmConfig& config) {
    // Try JIT first (preferred)
    if (auto kernel = CudaGemmJIT::instance().tryGetKernel(config)) {
        return launchJIT(kernel, ...);
    }
    
    // Fallback to precompiled registry
    if (auto launcher = CudaGemmKernelRegistry::instance().get_launcher(config)) {
        return launcher(...);
    }
    
    return cudaErrorInvalidConfiguration;
}
```

### Phase 2: JIT-Only

Remove precompiled variants entirely:
- Delete `generate_cuda_gemm_variants.py`
- Delete 250 generated .cu files
- Remove `CudaGemmKernelRegistry`
- Use JIT for all configs

**Build time reduction**: 25 minutes → **30 seconds** (just core files)

---

## Alternative: Template Header-Only (Simpler JIT)

**Even simpler approach**: Put kernel template in header, let compiler JIT-compile when called.

```cpp
// CudaGemmKernelTemplate.cuh (header-only)
template <typename Decoder, int TILE_M, int TILE_N, int TILE_K, ...>
__global__ void quantized_gemm_kernel_variant(...) {
    // Full implementation in header
}

// Usage (implicit JIT by nvcc)
quantized_gemm_kernel_variant<IQ4_NL_Decoder, 64, 64, 32, ...>
    <<<grid, block, 0, stream>>>(...);
```

**Pros:**
✅ **Simplest approach** - no NVRTC needed
✅ Compiler JIT-compiles on first use
✅ Works with CUDA Runtime API (not Driver API)

**Cons:**
❌ Still slow first compile (all configs in one file)
❌ Can't cache across runs (recompiles every launch)

---

## Recommendation

### For Production: **NVRTC JIT with Disk Cache**

**Why:**
1. **25-minute build → 2.5-second first run** (1000× faster for typical usage)
2. **Subsequent runs: 50ms** (cache load)
3. **1GB binary → 10MB** (100× smaller)
4. **Only compile what you use** (5 configs vs 37,380)

**Implementation effort:** ~500 lines of code
- CudaGemmJIT class (~200 lines)
- Kernel template string (~200 lines - copy existing kernel)
- Cache management (~100 lines)

**Timeline:** 1-2 days

### Quick Test: Header-Only Template

**For testing if JIT works well:**

Just put the kernel in a `.cuh` header and include it. Let the compiler implicitly JIT-compile. This tests the concept with zero new infrastructure.

**Implementation effort:** ~30 minutes (just move kernel to header)

---

## Next Steps

1. **Prototype header-only JIT** (30 minutes)
   - Move kernel to `CudaGemmKernelTemplate.cuh`
   - Test compile time and performance
   
2. **If successful, implement NVRTC JIT** (1-2 days)
   - Add disk cache for persistence
   - Benchmark first-run vs cached performance
   
3. **Remove precompiled variants** (30 minutes)
   - Delete generated files
   - Update build system
   - **Final result: 25-minute build → 30-second build**

---

## Open Questions

1. **Cache invalidation**: When to recompile?
   - Kernel source changes
   - CUDA version changes
   - Compiler flags change
   
2. **Multi-GPU support**: Cache per architecture?
   - Compile once per SM version (sm_75, sm_80, sm_86, etc.)
   
3. **Error handling**: What if JIT compile fails?
   - Fallback to CPU?
   - Fallback to slower precompiled config?

4. **Autotuner integration**: 
   - Should autotuner trigger JIT compilation upfront?
   - Or lazy-compile during autotuning?
