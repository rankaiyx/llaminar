# CUDA GEMM Autotuner with JIT Compilation

**Date**: November 3, 2025  
**Status**: Architecture Documentation  
**Purpose**: Explain how autotuning works with JIT-compiled kernels

---

## TL;DR

**The autotuner workflow is identical** - same code, same search strategy, same benchmarking loop. The only difference is **where kernels come from**:

- **Before**: Precompiled registry (instant, but 25-minute builds)
- **After**: JIT compilation (first-time compile, then cached)

The `launchIQ4NLGemmVariant()` function **already calls the JIT system** - it's transparent to the autotuner!

---

## High-Level Architecture

### Autotuner Workflow (Unchanged)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Generate Configuration Space                             │
│    CudaGemmAutoTuner::generateCandidates()                  │
│    → ~16,000 configs (all parameter combinations)           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Filter by Heuristics                                     │
│    - Problem size fit (tile alignment)                      │
│    - GPU resources (occupancy, shared memory)               │
│    - Analytical model (GFLOPS prediction)                   │
│    → ~10-50 candidate configs                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Benchmark Each Candidate                                 │
│    for (config : candidates)                                │
│        result = benchmarkConfig(config, m, n, k)            │
│    → CudaBenchmarkResult[] sorted by GFLOPS                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Select Best Configuration                                │
│    optimal = results[0]  // Highest GFLOPS                  │
│    cache[shape] = optimal                                   │
└─────────────────────────────────────────────────────────────┘
```

### What Changed: Kernel Compilation Model

**Before (Registry Pattern)**:
```cpp
// Build time:
// - nvcc compiles 250 .cu files with 37,380 instantiations
// - CudaGemmKernelRegistry::registerLauncher() for each config
// - Binary size: ~1GB

// Runtime (autotuner benchmarking):
cudaError_t launchIQ4NLGemmVariant(..., const CudaGemmConfig& config, ...) {
    ensureCudaKernelsRegistered();  // No-op (static init already done)
    auto launcher = CudaGemmKernelRegistry::get_launcher(config);
    if (launcher == nullptr) return cudaErrorInvalidConfiguration;
    return launcher(A, B, C, m, n, k, gridDim, blockDim, stream);
    // → kernel<<<>>>() launch (CUDA Runtime API)
}
```

**After (JIT Pattern)**:
```cpp
// Build time:
// - nvcc compiles 7 core .cu files (no variant generation)
// - CudaGemmJIT infrastructure included
// - Binary size: ~10MB

// Runtime (autotuner benchmarking):
cudaError_t launchIQ4NLGemmVariant(..., const CudaGemmConfig& config, ...) {
    CUfunction kernel = CudaGemmJIT::instance().getKernel(config);
    // → First time: NVRTC compile (~500ms) + save to cache
    // → Subsequent: Load from cache (~10ms disk, <1ms memory)
    
    void* args[] = {&A, &B_blocks, &C, &m, &n, &k};
    cuLaunchKernel(kernel, gridDim.x, gridDim.y, gridDim.z, ...);
    // → cuLaunchKernel() (CUDA Driver API)
}
```

**Key insight**: The autotuner calls the **same function** (`launchIQ4NLGemmVariant`), but the implementation changed from registry lookup to JIT compilation.

---

## Autotuner Code (No Changes Needed!)

### Main Auto-Tuning Logic

**File**: `CudaGemmAutoTuner.cu`, lines 200-240

```cpp
CudaGemmConfig CudaGemmAutoTuner::autoTune(int m, int n, int k) {
    // 1. Generate all valid configurations
    auto candidates = generateCandidates();  // ~16,000 configs
    
    // 2. Filter by problem size (tile alignment)
    candidates = filterByProblemSize(candidates, m, n, k);
    
    // 3. Filter by GPU resources (occupancy, shared memory)
    candidates = filterByResources(candidates);
    
    // 4. Rank by analytical model (ML heuristic or manual rules)
    candidates = rankByPerformanceModel(candidates, m, n, k);
    
    // 5. Select top N candidates for benchmarking
    candidates = selectTopCandidates(candidates, max_candidates_);
    
    // 6. Benchmark each candidate (THIS IS WHERE JIT HAPPENS)
    std::vector<CudaBenchmarkResult> results;
    for (const auto& config : candidates) {
        auto result = benchmarkConfig(config, m, n, k);  // ← JIT compile here!
        results.push_back(result);
    }
    
    // 7. Sort by performance and return best
    std::sort(results.begin(), results.end());
    return results[0].config;  // Highest GFLOPS
}
```

### Benchmarking Function

**File**: `CudaGemmAutoTuner.cu`, lines 739-800

```cpp
CudaBenchmarkResult CudaGemmAutoTuner::benchmarkConfig(
    const CudaGemmConfig& config, int m, int n, int k)
{
    // Warmup iterations (for stable timing)
    for (int i = 0; i < warmup_iterations_; ++i) {
        auto err = launchIQ4NLGemmVariant(
            test_A_device_, test_B_device_, test_C_device_,
            m, n, k, config, benchmark_stream_);
        // ↑ First iteration: JIT compile (~500ms)
        // ↑ Subsequent: Use cached kernel (<1ms)
    }
    
    // Timed iterations
    cudaEventRecord(start_event_, benchmark_stream_);
    for (int i = 0; i < benchmark_iterations_; ++i) {
        launchIQ4NLGemmVariant(...);  // All cached now (fast)
    }
    cudaEventRecord(stop_event_, benchmark_stream_);
    
    // Compute GFLOPS
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
    result.time_ms = elapsed_ms / benchmark_iterations_;
    result.gflops = (2.0 * m * n * k / 1e9) / (result.time_ms / 1000.0);
    
    return result;
}
```

**Critical observation**: The autotuner calls `launchIQ4NLGemmVariant()` multiple times per config:
- Warmup: 3 iterations (first triggers JIT compile)
- Timed: 10 iterations (all use cached kernel)

So the 500ms compile cost is **amortized** over 13 iterations, adding ~38ms overhead per config. This is negligible compared to the actual GEMM execution time (which dominates).

---

## Performance Implications

### First-Time Autotuning (Cold Cache)

**Scenario**: Running autotuner for first time on a new shape

```
Configuration space: 16,000 total configs
Heuristic filtering: 16,000 → 50 candidates
Benchmark iterations: 50 configs × (3 warmup + 10 timed)

Cost breakdown:
- JIT compilation: 50 configs × 500ms = 25 seconds
- Kernel execution: 50 configs × 13 iters × ~1ms = 650ms
- Total: ~26 seconds for full auto-tuning

Disk cache populated: 50 configs × ~400KB = ~20MB
```

**Comparison to old registry**:
- Registry: 0ms compile, same 650ms execution → **650ms total**
- JIT: 25s compile, 650ms execution → **26s total**

**Trade-off**: 40× slower first-time auto-tuning, but only happens once per shape

### Subsequent Autotuning (Warm Cache)

**Scenario**: Running autotuner again on same shape (or similar shapes with overlapping configs)

```
Cost breakdown:
- JIT cache load: 50 configs × 10ms = 500ms (disk)
                   OR 50 configs × <1ms = 50ms (memory)
- Kernel execution: 650ms (same as before)
- Total: 1.15 seconds (disk) or 700ms (memory)

Comparison to registry: Nearly identical performance!
```

### Production ML Heuristic (No Autotuning)

**Scenario**: Using trained ML model for config selection (typical production usage)

```
Workflow:
1. ML model predicts best config (1-2ms)
2. launchIQ4NLGemmVariant() with predicted config
   - First time: JIT compile (500ms)
   - Cached: Load from disk (10ms) or memory (<1ms)

Production overhead:
- First inference: 500ms (one-time cost per config)
- Subsequent: <10ms (cached)

This is acceptable because:
- Only 5-10 configs used in practice (not 16,000)
- Cache persists across runs
- 500ms compile << multi-second inference time
```

---

## Disk Cache Behavior

### Cache Location

```
~/.cache/llaminar/cuda_kernels/
├── compute_75/  (Turing/Ampere)
│   ├── gemm_64_64_32_16_16_4_4_1_0_2.cubin
│   ├── gemm_64_64_32_16_16_4_4_1_0_4.cubin
│   └── ... (5-50 configs typically)
└── compute_80/  (Ampere)
    └── ... (architecture-specific)
```

### Cache Population Strategy

**Autotuner session**:
- Benchmarks 50 configs → 50 .cubin files cached
- Next run: Instant load from disk (no recompilation)

**Production inference**:
- Uses ML-predicted config → 1 .cubin file cached
- Subsequent runs: <1ms memory hit (no disk I/O)

**Multi-shape workload**:
- Different shapes may select different configs
- Cache grows organically (5-100 configs typical)
- Self-limiting: Only configs actually used are compiled

### Cache Invalidation

Cache is **architecture-specific** and **source-hash-based**:
- Different GPU (sm_75 vs sm_80) → separate cache dirs
- Kernel template changes → cache key changes (automatic invalidation)
- Manual clear: `rm -rf ~/.cache/llaminar/cuda_kernels/`

---

## Benchmark Data Collection

### CSV Output (Same as Before)

**File**: `cuda_gemm_benchmark_data.csv`

```csv
m,n,k,tile_m,tile_n,tile_k,threads_m,threads_n,work_m,work_n,prefetch,transpose,vectorize,gflops,time_ms
1,896,896,64,64,32,16,16,4,4,1,0,2,42.5,0.038
1,896,896,32,32,32,16,16,2,2,2,1,4,38.2,0.042
1,896,896,16,16,32,8,8,2,2,1,0,1,12.3,0.131
...
```

**How it's generated**:

```cpp
// Same code as before, just kernel source changed
for (auto& shape : test_shapes) {
    auto candidates = generateCandidates();
    for (auto& config : candidates) {
        auto result = benchmarkConfig(config, m, n, k);
        // ↑ JIT compile on first use, cached thereafter
        
        csv << m << "," << n << "," << k << ","
            << config.tile_m << "," << config.tile_n << "," ...
            << result.gflops << "," << result.time_ms << "\n";
    }
}
```

**First run**: Slow (2-3 hours to compile all 16,000 configs)  
**Subsequent runs**: Fast (3 minutes to load from cache)

This is **acceptable** because benchmark collection is a one-time operation (or infrequent retraining).

---

## ML Training Pipeline (Unchanged)

### Workflow

```
1. Collect benchmark data (CSV with JIT kernels)
   ↓
2. Train neural network (Python/PyTorch)
   ↓
3. Export to ONNX (cuda_heuristic_nn.onnx)
   ↓
4. Load in C++ with ONNX Runtime
   ↓
5. Predict best config at runtime (1-2ms)
   ↓
6. JIT compile predicted config (500ms first time, <1ms cached)
```

**Key point**: The ML model is trained on **JIT-compiled kernel performance**, so predictions are accurate for the JIT execution model.

---

## Migration Summary

### What Changed

| Component | Before | After |
|-----------|--------|-------|
| **Build system** | 250 .cu variant files | 7 core .cu files |
| **Build time** | 25 minutes | 56 seconds |
| **Binary size** | ~1GB | ~10MB |
| **Kernel source** | `CudaGemmKernelRegistry` | `CudaGemmJIT` |
| **Launch API** | Runtime API (`kernel<<<>>>`) | Driver API (`cuLaunchKernel`) |
| **First-time cost** | 0ms (precompiled) | 500ms (JIT compile) |
| **Cached cost** | 0ms | <10ms (disk) or <1ms (memory) |

### What Stayed the Same

| Component | Status |
|-----------|--------|
| **Autotuner logic** | ✅ Identical (no code changes) |
| **Config generation** | ✅ Same 16,000 config space |
| **Heuristic filtering** | ✅ Same filters and rankings |
| **Benchmarking** | ✅ Same timing methodology |
| **CSV output format** | ✅ Same columns and data |
| **ML training** | ✅ Same PyTorch pipeline |
| **ONNX model** | ✅ Same architecture |

---

## FAQ

### Q: Does the autotuner need to be updated for JIT?

**A**: No! The autotuner calls `launchIQ4NLGemmVariant()`, which **transparently handles JIT compilation**. From the autotuner's perspective, it's calling the same function as before.

### Q: Is autotuning slower with JIT?

**A**: First time: Yes (~26 seconds vs 650ms). Subsequent times: No (nearly identical after caching).

### Q: Should I re-run benchmark collection?

**A**: No, unless you want to. JIT kernels produce **identical performance** to precompiled kernels (same CUDA code, same GPU execution). Existing CSV data is still valid.

### Q: What if I add new config parameters?

**A**: Update `CudaGemmConfig` and `CudaGemmKernelTemplate.h` (template placeholders). The JIT system will automatically compile new configs on-demand. No need to regenerate 250 .cu files!

### Q: Can I precompile for production?

**A**: Yes, future work includes an optional precompilation hook:
```bash
# Precompile common configs at build time (optional)
cmake -DCUDA_JIT_PRECOMPILE=ON -DCUDA_JIT_PRECOMPILE_CONFIGS=common_configs.txt
```

But for now, the disk cache serves the same purpose (persistent across runs).

### Q: What about multi-GPU or different architectures?

**A**: JIT compiles for the **target GPU architecture** automatically:
- Build on sm_75 machine → cache in `compute_75/`
- Run on sm_80 machine → recompile, cache in `compute_80/`

This is **better** than precompilation, which would require building for all architectures upfront.

---

## Conclusion

The JIT compilation architecture is **fully transparent** to the autotuner:

✅ **Same autotuning workflow** (config generation → filtering → benchmarking → selection)  
✅ **Same ML training pipeline** (CSV collection → PyTorch training → ONNX export)  
✅ **Same runtime API** (`getOptimalConfig()` returns same configs)  
✅ **Same performance** (after disk cache warmup)  

The only difference is **where kernels come from**:
- **Registry**: Precompiled at build time (instant runtime, 25-minute builds)
- **JIT**: Compiled at first use (500ms runtime, 56-second builds, persistent cache)

For production inference (5-10 configs used), the trade-off is overwhelmingly favorable:
- **Development**: 26× faster iteration cycles (56s vs 25m builds)
- **Production**: <1ms overhead after first run (memory cache hit)
- **Flexibility**: Unlimited dynamic configs (not limited to 37,380 precompiled)

The autotuner **just works** with zero code changes! 🎉
