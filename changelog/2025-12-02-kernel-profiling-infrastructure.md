# Kernel Profiling Infrastructure

**Date**: 2025-12-02

## Summary

Added per-operation kernel profiling to Llaminar V2 for performance analysis. This allows identifying bottlenecks by breaking down inference time into individual kernel categories.

## Changes

### New Features

1. **ProfileConfig in DebugEnv** (`src/v2/utils/DebugEnv.h`)
   - Added `ProfileConfig` struct with `enabled`, `per_layer`, `per_iteration`, `print_interval` settings
   - Controlled via `LLAMINAR_PROFILE_KERNELS=1` environment variable

2. **KernelProfiler Updated** (`src/v2/utils/KernelProfiler.h`)
   - Modified `isEnabled()` to use `debugEnv().profile.enabled` instead of direct `getenv`
   - Existing infrastructure (KernelType enum, atomic stats, RAII timing) leveraged

3. **Pipeline Instrumentation** (`src/v2/pipelines/PipelineBase.cpp`, `Qwen2Pipeline.cpp`)
   - Added `KERNEL_PROFILE_SCOPE` to key operations:
     - `rms_norm()` → KernelType::RMS_NORM
     - `project()` → KernelType::GEMM_Q8/LM_HEAD/FFN_* (based on snapshot_key)
     - `swiglu()` → KernelType::SWIGLU
     - `apply_rope()` → KernelType::ROPE
     - `add_residual()` → KernelType::RESIDUAL_ADD
     - `compute_attention()` → KernelType::ATTENTION
     - `embedding_batch()` → KernelType::EMBEDDING

4. **FusedGEMM Instrumentation** (`src/v2/kernels/cpu/gemm_v4/FusedGEMM.cpp`)
   - Separate profiling for QUANTIZE_Q8 and GEMM_Q8 phases

5. **BenchmarkRunner Integration** (`src/v2/utils/BenchmarkRunner.cpp`)
   - Auto-prints kernel profiling summary after benchmark results
   - Resets profiling after warmup (only measures benchmark iterations)

### Usage

```bash
# Run benchmark with kernel profiling
LLAMINAR_PROFILE_KERNELS=1 ./run_llaminar.sh -- --benchmark -m model.gguf -n 50
```

### Sample Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         KERNEL PROFILING SUMMARY                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Kernel Type      │  Calls   │  Total (ms)  │  Avg (µs)  │  Min/Max (µs)     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GEMM_Q8         │    7128  │     8947.85  │   1255.31  │   33.47/32651.11 ║
║  ATTENTION       │    1188  │     5915.90  │   4979.71  │  181.00/86178.09 ║
║  FFN_DOWN        │    1188  │     3891.00  │   3275.25  │  429.99/34938.14 ║
║  ...             │          │              │            │                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TOTAL KERNEL TIME:   19993.79 ms   ( 30.31 kernel tok/s)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Key Findings from Profiling

### Model Size vs Profile Shape

| Model Size | GEMM % | Attention % | FFN % | Character |
|------------|--------|-------------|-------|-----------|
| 0.5B | ~20% | ~57% | ~11% | Attention-bound |
| 3B | ~45% | ~30% | ~20% | Compute-bound |

**Insight**: Smaller models are attention-bound (OpenMP overhead dominates); larger models are compute-bound (JIT GEMM dominates).

## Design Notes

- **Zero overhead when disabled**: `isEnabled()` check at start of RAII timer
- **Lock-free accumulation**: Uses atomic operations for thread-safe stats
- **Centralized config**: Via `debugEnv().profile.enabled`, not scattered `getenv()` calls
- **Resets after warmup**: Only benchmark iterations are measured

## Files Modified

- `src/v2/utils/DebugEnv.h` - Added ProfileConfig
- `src/v2/utils/KernelProfiler.h` - Use DebugEnv for isEnabled()
- `src/v2/utils/BenchmarkRunner.cpp` - Print summary, reset after warmup
- `src/v2/pipelines/PipelineBase.cpp` - Instrument core operations
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Instrument embedding
- `src/v2/kernels/cpu/gemm_v4/FusedGEMM.cpp` - Instrument quant + GEMM
- `.github/copilot-instructions.md` - Documentation
