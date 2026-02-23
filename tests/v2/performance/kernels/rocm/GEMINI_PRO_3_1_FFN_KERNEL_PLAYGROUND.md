# Gemini Pro 3.1 FFN Kernel Playground Kit

This kit is a **fast micro-harness** for iterating on ROCm FFN Up/Gate kernel changes without running the full production benchmark suite each time.

## Files
- Harness: [Microbench__GeminiFFNKernelPlayground.hip](Microbench__GeminiFFNKernelPlayground.hip)
- Build target: `v2_perf_rocm_gemini_ffn_kernel_playground`

## What it reproduces
It locks to the exact tuning shapes you care about:
- `Qwen2.5-0.5B_FFN_Up`  → `(M,N,K)=(128,4864,896)`
- `Qwen2.5-0.5B_FFN_Gate`→ `(128,4864,896)`
- `Qwen2.5-3B_FFN_Up`    → `(128,11008,2048)`
- `Qwen2.5-3B_FFN_Gate`  → `(128,11008,2048)`

And it uses the same split-K mode you’ve been tuning (`split_k=4` by default).

## Fast iteration loop
## 1) Build once
```bash
cmake --build build_v2_release --target v2_perf_rocm_gemini_ffn_kernel_playground --parallel
```

## 2) Run quick check
```bash
./build_v2_release/tests/v2/v2_perf_rocm_gemini_ffn_kernel_playground --warmup 1 --iters 4
```

## 3) Run stability pass
```bash
./build_v2_release/tests/v2/v2_perf_rocm_gemini_ffn_kernel_playground --warmup 2 --iters 12
```

## 4) Edit only the candidate zone
Inside the harness, modify code only between:
- `GEMINI TUNING ZONE START`
- `GEMINI TUNING ZONE END`

The baseline path is intentionally duplicated and must stay unchanged for apples-to-apples comparison.

## Output interpretation
For each shape, you get:
- `baseline ms`
- `candidate ms`
- `speedup (baseline/candidate)`
- `max_abs_diff(INT32) baseline_vs_cpu_ref`
- `max_abs_diff(INT32) candidate_vs_cpu_ref`
- `max_abs_diff(INT32) baseline_vs_candidate`

Correctness uses an OpenMP-parallelized CPU reference (`int8 x int8 -> int32`), and the target is linked with OpenMP (`-fopenmp` + `OpenMP::OpenMP_CXX`).
Set `OMP_NUM_THREADS` to control CPU-check parallelism.

### Promotion gate (recommended)
Treat a variant as viable only if:
- `baseline_vs_cpu_ref == 0` and `candidate_vs_cpu_ref == 0` on all 4 shapes
- Mean speedup improves across repeats
- 3B minima do not regress

## Agent prompt template for Gemini Pro 3.1
Use this exact structure when delegating:

```text
You are tuning only the candidate branch in
Microbench__GeminiFFNKernelPlayground.hip for the AMD Mi50 (gfx906 architecture).

Constraints:
1) Do not change baseline branch.
2) Keep launch geometry unchanged unless explicitly requested.
3) Maintain exact INT32 parity (max_abs_diff must remain 0).
4) Optimize for 3B FFN Up/Gate first, while avoiding 0.5B regressions.

Workflow:
- Propose one micro-optimization.
- Apply change in candidate zone only.
- Rebuild target v2_perf_rocm_gemini_ffn_kernel_playground.
- Run with --warmup 1 --iters 4, then --warmup 2 --iters 12.
- Report per-shape baseline/candidate/speedup and parity.
- If regressive, revert and try next variant.
```

## Suggested first variant classes
- Address generation reduction (pointer increments / typed loads)
- Control-flow simplification in tail path
- Lightweight unroll experiments only where register pressure stays stable
- Atomic-path micro-tuning without extra launches

## Notes
- This harness uses random but deterministic int8 inputs (`--seed` available).
- It is intentionally synthetic and focused on kernel throughput + parity.
- After finding a promising variant here, validate in production path with:
  - `v2_perf_rocm_prefill_dispatch_comparison`

---

## Tuning Results (2025-02-23, gfx906 MI50)

### Winning Variant: V7 — Software-Pipelined Loop

The candidate zone currently contains **V7**, the best variant found after 7 iterations.
V7 pre-loads the first k-group, then in the main loop issues next-iteration loads
*before* computing the current iteration's `sdot4` operations. This increases the
load-to-use distance, giving the memory subsystem more time to return data before
the ALU needs it. The dead tail path (`N%128` remainder) is also removed since all
target shapes have `N` divisible by 128.

**ISA characteristics (gfx906)**:
- 17 VGPRs, 16 SGPRs → 10 waves/SIMD (maximum occupancy)
- Inner loop order: `global_load_dwordx4` (next) → `v_dot4_i32_i8` (current) → pointer advance
- Baseline inner loop order: `global_load_dwordx4` → `s_waitcnt` → `v_dot4_i32_i8` (load-then-stall)

### Variants Tried

| Variant | Technique | VGPRs | Result |
|---------|-----------|-------|--------|
| V1 | `int4` vectorized B load + `#pragma unroll 4` | 29 | **Regressive** (2-8% slower), VGPR inflation killed occupancy |
| V2 | Manual 2× unroll with interleaved B accesses | 29 | **Regressive** (~12% slower) |
| V3 | `#pragma clang loop unroll_count(2)` + `__restrict__` | 29 | **Regressive** (17-19% slower on 3B) |
| V4 | Remove dead tail path (N%128==0 for targets) | 17 | **Promising** (~2% on quick check, ~9% stability) |
| V5 | LDS B tile caching with `__syncthreads` | — | **Regressive** (15% slower on 3B) |
| V6 | Production-style index addressing (recompute `d_A+m*K+kg*4`) | 17 | Identical ISA to V4 |
| **V7** | **Software-pipelined loop (pre-load → compute → rotate)** | **17** | **Winner: ~10% 3B speedup** |

**Key insight**: On gfx906, any unrolling that pushes VGPRs past 24 drops occupancy from 10→8 waves/SIMD, which is immediately regressive. The compiler already auto-generates `global_load_dwordx4` for 4 consecutive int32 reads — explicit vectorized loads don't help. Only structural changes (dead code removal, load/compute reordering) produce different ISA without inflating registers.

### Playground Benchmark (warmup=2, iters=12)

```
[Shape] Qwen2.5-0.5B_FFN_Up (M=128 N=4864 K=896)
  baseline:    0.1867 ms    5.977 TOPS
  candidate:   0.1858 ms    6.004 TOPS
  speedup: 1.005x

[Shape] Qwen2.5-0.5B_FFN_Gate (M=128 N=4864 K=896)
  baseline:    0.1865 ms    5.984 TOPS
  candidate:   0.1879 ms    5.938 TOPS
  speedup: 0.992x

[Shape] Qwen2.5-3B_FFN_Up (M=128 N=11008 K=2048)
  baseline:    0.7496 ms    7.700 TOPS
  candidate:   0.6820 ms    8.462 TOPS
  speedup: 1.099x  ← target shape

[Shape] Qwen2.5-3B_FFN_Gate (M=128 N=11008 K=2048)
  baseline:    0.7525 ms    7.670 TOPS
  candidate:   0.6819 ms    8.463 TOPS
  speedup: 1.103x  ← target shape
```

**Summary**: ~10% throughput improvement on 3B FFN shapes (7.7 → 8.5 TOPS), 0.5B shapes within noise. Perfect INT32 parity (max_abs_diff=0) on all shapes across all runs.

### Production Validation (`v2_perf_rocm_prefill_dispatch_comparison`)

The production dispatch comparison (which uses the CK library for FFN, not this playground kernel) shows that CK legacy is still faster for FFN Up/Gate shapes. This playground kernel is a candidate for replacing the CK path once the ~6% FFN gap is closed:

For direct production-path benchmarking of the grid-kpar kernel body, use:

```bash
# baseline loop body
LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=0 \
./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison

# V7 software-pipelined loop body
LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=1 \
./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison
```

`LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY` values:
- `0`: baseline production loop body
- `1`: software-pipelined loop body (V7-style)

| Shape Class | CK Legacy (ms) | New Native (ms) | Speedup | Status |
|-------------|----------------|-----------------|---------|--------|
| Attention | 0.865 / 1.804 | 0.696 / 1.784 | 1.24x / 1.01x | **New wins** |
| FFN_Down | 2.267 / 5.075 | 2.013 / 4.844 | 1.13x / 1.05x | **New wins** |
| FFN_Up | 1.919 / 4.420 | 2.039 / 4.902 | 0.94x / 0.90x | CK still ahead |
| FFN_Gate | 1.937 / 4.438 | 2.045 / 4.910 | 0.95x / 0.90x | CK still ahead |
| LM_Head | 45.41 / 47.46 | 56.73 / 72.31 | 0.80x / 0.66x | CK still ahead |
