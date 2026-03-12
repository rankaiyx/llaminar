# ROCm GEMV Dispatch Threshold Fix & KB Heuristic Tuning

**Date**: 2025-07-15
**Category**: Performance / ROCm GEMV Kernel

## Summary

Two dispatch heuristic improvements in the ROCm INT8 VNNI GEMV kernel that eliminate a catastrophic fallback path for TP-sharded small-N shapes and improve KB (K-dimension block splitting) selection.

## Problem

When tensor-parallel sharding reduces K/V projection output dimensions below N=128 (e.g., Qwen 3B at TP=4: N=64, or Qwen 0.5B at TP=2: N=64), these shapes fell through the dispatch chain to a **fallback kernel** (`gemv_int8_int8_vnni_blockwise_scaled_kernel`) that:
- Uses TILE_N=256 (4 wavefronts per block), wasting 75%+ of threads for N=64
- Has NO KB splitting, producing a single CU-starved launch
- Was 2-4x slower than the optimized LDS-kred path

Additionally, the KB heuristic had two minor inefficiencies:
1. `kb_max` used floor division (`act_blocks / wk`), preventing shapes with fractional remainder from reaching the optimal split count
2. The small-K regime didn't cap splits based on `grid_n`, causing excessive atomicAdd contention at moderate grid sizes

## Changes

### 1. Lower LDS-kred Dispatch Threshold (`ROCmGemvKernel_INT8_VNNI.hip`)

**Before**: `use_lds_kreduce_blockwise = !use_wide_vec4 && use_vec4 && (N >= 128)`
**After**: `use_lds_kreduce_blockwise = !use_wide_vec4 && use_vec4`

The LDS k-reduce kernel already has N bounds checking (`n_base < N` guard on all reads/writes), so it handles N < TN=128 correctly — grid_n=1 with excess threads simply skipping. The occupancy selector adapts KB to fill CUs at any grid_n.

### 2. KB Heuristic: Ceiling Division for `kb_max`

**Before**: `kb_max = max(1, act_blocks / wk)` (floor division)
**After**: `kb_max = max(1, (act_blocks + wk - 1) / wk)` (ceiling division)

For Wo proj at TP=4 (act_blocks=28, wk=8): cap changes from 3 → 4, matching the sweep-optimal KB=4 (7.3% improvement per sweep data).

### 3. KB Heuristic: Grid-N Occupancy Cap for Small-K Regime

**Before**: `ideal_kb = acts_per_wave_single_kb` (no grid_n awareness)
**After**: `ideal_kb = min(acts_per_wave_single_kb, max(1, 2*NUM_CUS / grid_n))`

Prevents over-splitting when `grid_n × ideal_kb` would exceed ~2×CUs total blocks, reducing atomicAdd contention. For Q proj at TP=2 (grid_n=14, acts_per_wave=14): KB changes from 14 → 8, closer to the sweep-optimal KB=7.

### 4. Test Classifier Update (`Perf__ROCmTPScaling.cpp`)

Updated `classifyGemvDispatch()` to match the new dispatch logic — removes the N≥128 gate so small-N shapes are correctly labeled "LDS-kred".

## Results

### GEMV TP Scaling Efficiency (layer total, all projections)

| Model | TP | Before | After | Change |
|-------|-----|--------|-------|--------|
| 0.5B | TP=2 | 51.1% | **58.2%** | **+7.1pp** |
| 0.5B | TP=4 | 27.4% | **32.0%** | **+4.6pp** |
| 3B | TP=2 | 68.7% | 69.6% | +0.9pp |
| 3B | TP=4 | 29.3% | **41.1%** | **+11.8pp** |
| 7B | TP=2 | 70.9% | 72.0% | +1.1pp |
| 7B | TP=4 | 49.1% | 49.6% | +0.5pp |
| 14B | TP=2 | 72.8% | 72.3% | -0.5pp (noise) |
| 14B | TP=4 | 51.7% | 53.1% | +1.4pp |
| 32B | TP=2 | 96.9% | 97.8% | +0.9pp |
| 32B | TP=4 | 74.6% | 75.2% | +0.6pp |

### Key Per-Projection Improvements

| Shape | Before | After | Speedup |
|-------|--------|-------|---------|
| 3B K/V proj TP=4 (N=64, K=2048) | 41 µs | 15.5 µs | **2.7x** |
| 0.5B K/V proj TP=2 (N=64, K=896) | 22.9 µs | 14.1 µs | **1.6x** |
| 0.5B K/V proj TP=4 (N=32, K=896) | 22.7 µs | 13.9 µs | **1.6x** |
| 7B Q proj TP=2 (N=1792, KB 14→8) | 26.4 µs | 25.4 µs | 1.04x |

### No Regressions

7B/14B/32B models show ≤1.5pp efficiency change (within run-to-run noise), confirming no regression for shapes already using LDS-kred.

## Files Changed

- `src/v2/kernels/rocm/gemm/ROCmGemvKernel_INT8_VNNI.hip` — Dispatch threshold and KB heuristic
- `tests/v2/performance/kernels/rocm/Perf__ROCmTPScaling.cpp` — Test dispatch classifier
