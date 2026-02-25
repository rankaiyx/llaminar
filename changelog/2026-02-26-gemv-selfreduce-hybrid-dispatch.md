# GEMV Self-Reducing Scatter + Hybrid Dispatch

**Date**: 2026-02-26  
**Category**: Performance Optimization  
**Impact**: +3% over scatter+reduce baseline, +30% cumulative over original 3-kernel pipeline

## Summary

Added a self-reducing scatter kernel variant that fuses the reduce phase into the scatter
kernel using a "last-to-arrive" atomic counter pattern. Combined with a hybrid dispatch
strategy that selects the optimal kernel based on problem geometry, this eliminates the
reduce kernel launch overhead for small-N shapes (K/V projections) where the reduce kernel
was catastrophically slow (~27μs for N=512 on 60 CUs).

## Motivation

Profiled the scatter+reduce pipeline with `rocprofv3 --kernel-trace` and found:
- **Reduce kernel for N=512**: 27μs (4 blocks on 60 CUs = 93% idle CUs)
- **Reduce kernel for N=3584**: 7.5μs (acceptable)
- **Reduce kernel for N=18944**: 4μs (fast, CUs well-utilized)

The reduce kernel is a separate kernel launch that reads KB partials and sums them.
For small N, the grid is tiny and most CUs sit idle.

## ISA Analysis (gfx906)

Extracted and analyzed scatter + reduce kernel ISA via `rocprofv3`:

| Feature | Status | Notes |
|---------|--------|-------|
| Weight loads | ✅ Vectorized | `global_load_dwordx2` for int2 (CPT=2) |
| LDS reads (dot loop) | ✅ Vectorized | `ds_read_b128` (4 K-groups at once) |
| v_dot4_i32_i8 | ✅ 10 instances | 8 unrolled + 2 remainder |
| Partial stores | ✅ Vectorized | `global_store_dwordx2` (2×float) |
| INT8 quantize writes | ❌ Single byte | `ds_write_b8` (not vectorized) |
| Load pipelining | ❌ Serial | `s_waitcnt vmcnt(0)` between loads |
| Register count | 32 VGPRs, 48 SGPRs | 8 waves/SIMD occupancy |

## Design

### Self-Reducing Scatter Kernel

After writing FP32 partials (same as original scatter), each block:
1. Issues `__threadfence()` to make partial writes globally visible
2. Atomically increments `d_counter[blockIdx.x]` via `atomicAdd`
3. If this block is the last to arrive (`old + 1 == valid_kb`):
   - Reads all `valid_kb` partials for its output columns
   - Sums them, applies `scale_B * alpha + beta * C_existing + bias`
   - Writes the final output directly

This eliminates the separate reduce kernel entirely for the shapes that use it.

### Hybrid Dispatch Strategy

The self-reduce kernel adds `__threadfence()` + `atomicAdd` overhead (~2μs aggregate),
which causes a slight regression for large-N shapes where the reduce kernel is already fast.

The hybrid dispatch selects the optimal path:
- **`grid_n < 16` (N < 2048)**: Self-reduce (reduce kernel would be catastrophically slow)
- **`grid_n >= 16` (N >= 2048)**: 2-kernel scatter+reduce (reduce kernel is fast, avoid atomic overhead)

Counter buffer: 16-int static per-device allocation (allocated once, reused across calls).

## Performance Results

### Self-Reduce vs Scatter+Reduce (Qwen 7B decode, M=1)

| Shape | N | K | Scatter(ms) | SelfReduce(ms) | Speedup |
|-------|---|---|-------------|----------------|---------|
| Q proj | 3584 | 3584 | 0.045 | 0.044 | 1.00x (2-kernel path) |
| K proj | 512 | 3584 | 0.043 | 0.035 | **1.23x** (self-reduce) |
| V proj | 512 | 3584 | 0.043 | 0.035 | **1.23x** (self-reduce) |
| Wo proj | 3584 | 3584 | 0.044 | 0.044 | 1.00x (2-kernel path) |
| FFN Gate | 18944 | 3584 | 0.119 | 0.119 | 1.00x (2-kernel path) |
| FFN Up | 18944 | 3584 | 0.108 | 0.108 | 1.00x (2-kernel path) |
| FFN Down | 3584 | 18944 | 0.137 | 0.137 | 1.00x (2-kernel path) |
| LM Head | 152064 | 3584 | 0.636 | 0.634 | 1.00x (KB=1 path) |

Per-layer: 0.542 → 0.523 ms  
All 28 + LM: 15.78 → 15.28 ms (**1.03x**)  
Throughput: 63.4 → 65.4 tok/s

### Cumulative vs Original 3-Kernel Pipeline

| Pipeline | Per-layer(ms) | All 28+LM(ms) | tok/s | vs 3-Kernel |
|----------|--------------|----------------|-------|-------------|
| 3-Kernel (quant+GEMV+scale) | 0.685 | 19.84 | 50.4 | baseline |
| Scatter+Reduce (2-kernel) | 0.540 | 15.77 | 63.4 | **1.26x** |
| Hybrid Self-Reduce | 0.523 | 15.28 | 65.4 | **1.30x** |

### Correctness

32/32 shapes pass (both Qwen 7B and Qwen 0.5B), cosine similarity ≥ 0.999992.

## Code Changes

- `src/v2/kernels/rocm/ROCmGemvKernel.hip`:
  - Added `gemv_fused_scatter_selfreduce_int8_vnni_kernel_t` kernel (~180 lines)
  - Added `rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni` dispatch function
  - Updated production `rocmGemv_fused_scatter_fp32_int8_vnni` with hybrid dispatch
  - Static per-device counter buffer allocation (16 ints per device)
- `tests/v2/performance/kernels/rocm/Perf__ROCmGemvKernel.cpp`:
  - Added `testCorrectnessSelfReduce` method
  - Added `benchmarkSelfReduce` method
  - Added `Correctness_SelfReduce` test (16/16 shapes)
  - Added `Benchmark_SelfReduceVsScatter` test
  - Added extern declaration for self-reduce dispatch

## Build

Full release build: 1449/1449 targets pass.
