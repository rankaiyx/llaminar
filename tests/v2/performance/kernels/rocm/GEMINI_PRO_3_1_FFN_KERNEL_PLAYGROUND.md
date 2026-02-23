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

## Tuning Results (2026-02-23, gfx906 MI50)

### Current Winner: V10 — LDS B-Tile Caching + Grid Swizzle

The candidate cooperatively loads B data into LDS (shared memory) to eliminate
redundant global memory B loads across the 8 m-rows per thread block. Combined
with the V8 grid swizzle for L2 locality and software-pipelined A loads.

**Three orthogonal optimizations stacked**:
1. **LDS B-tile caching** (V10): 256 threads cooperatively load 128×8 B tile (4KB)
   into `__shared__` memory. Eliminates 4× redundant B loads across wavefronts.
2. **Grid swizzle** (V8): `blockIdx.x` = M-tiles first, concentrating L2 footprint.
3. **A load software pipelining**: Pre-fetches next k-group's A value during compute.

**Why this is a massive win**: In the original kernel, all 4 wavefronts in a block
independently load the same B data from global memory. Even with L1 caching, each
wavefront issues 128 B-dwordx4 VMEM_RD instructions. With LDS, B is loaded once
cooperatively (4 VMEM_RD per wave per tile × 16 tiles = 64 total) and served from
LDS (~4 cycle latency) for all subsequent accesses. The A loads, which the compiler
vectorizes into dwordx4 (8 consecutive k-groups), add only 32 VMEM_RD per wave.
Total: **96 VMEM_RD/wave** (vs 256 baseline) — a **62.5% reduction**.

**Register/occupancy trade-off**: The LDS approach requires 32 VGPRs (vs 20 for
baseline), reducing occupancy from 10 to 8 waves/SIMD. Despite this 20% occupancy
loss, the 62.5% VMEM_RD reduction delivers a net **1.58× speedup**.

### V10 Implementation Details

All changes are in [Microbench__GeminiFFNKernelPlayground.hip](Microbench__GeminiFFNKernelPlayground.hip),
inside the `if constexpr (IsCandidate)` tuning zone.

**Change 1 — LDS B-tile cooperative loading** (KT=8):

256 threads in the block cooperatively load a 128×8 B tile (4KB) into `__shared__`
memory. Each thread loads 4 int32 values across 4 iterations, with the mapping
designed for perfect coalescing: consecutive threads load consecutive N-columns,
so each sub-wave (16 threads) loads exactly one 64B cache line.

```cpp
constexpr int KT = 8;
__shared__ int32_t b_lds[KT * 128]; // 8 × 128 = 4KB

#pragma unroll
for (int i = 0; i < 4; ++i) {
    const int flat = i * 256 + tid;
    const int kg_l = flat >> 7;   // / 128
    const int n_l  = flat & 127;  // % 128
    b_lds[kg_l * 128 + n_l] =
        b_global_base[(kt + kg_l) * N + n_l];
}
__syncthreads();
```

**Why KT=8**: Divides evenly for both 0.5B (56/8=7 tiles) and 3B (128/8=16 tiles),
avoiding partial tile handling. 4KB per block → up to 16 blocks/CU (64KB LDS total),
more than enough for the 8 waves/SIMD occupancy.

**Change 2 — Compute from LDS with A software pipelining**:

The inner loop reads B from LDS (4-cycle latency) and A from global memory. A loads
are software-pipelined: the next k-group's A value is prefetched while computing
the current. The compiler vectorizes the A loads into dwordx4 because 8 consecutive
k-groups are addressed sequentially.

```cpp
int32_t a_packed = a_row[kt];
#pragma unroll
for (int kg_l = 0; kg_l < KT - 1; ++kg_l) {
    const int32_t a_next = a_row[kt + kg_l + 1]; // prefetch
    const int off = kg_l * 128 + n_local;
    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_lds[off+0], acc[0], false);
    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_lds[off+1], acc[1], false);
    acc[2] = __builtin_amdgcn_sdot4(a_packed, b_lds[off+2], acc[2], false);
    acc[3] = __builtin_amdgcn_sdot4(a_packed, b_lds[off+3], acc[3], false);
    a_packed = a_next;
}
// Last iteration (no prefetch) ...
```

**Change 3 — Grid swizzle** (retained from V8):

The grid launch uses `x=M-tiles, y=N-tiles` so consecutive blocks share the same
B N-tile columns for the cooperative LDS load, improving L2 hit rate during the
load phase.

**How the three optimizations interact**:
- LDS eliminates **intra-block B redundancy** (4× wavefront duplication → 1 load)
- Grid swizzle improves **inter-block L2 reuse** (consecutive CUs share B N-tiles)
- A pipelining hides **global A load latency** via load-to-use distance overlap
- The unrolled KT=8 inner loop + consecutive addressing enables **A dwordx4 vectorization**,
  further reducing VMEM_RD from 128 to 32 per wave (bonus)

### L2 Cache Profiling — V10 vs Baseline (rocprof v1, `TCC_HIT_sum` / `TCC_MISS_sum`)

| Shape | Baseline L2 Hit% | V10 L2 Hit% | Baseline L2 Miss/wave | V10 L2 Miss/wave |
|-------|------------------|-------------|----------------------|------------------|
| 0.5B  | 89.8% | 91.0% | 20.7 | 17.2 |
| **3B** | **72.7%** | **90.8%** | **98** | **32.5** |

The 3B L2 miss count per wave dropped **67%** (98 → 32.5). Total HBM traffic
dropped from ~141 MB to ~46 MB — a **3× reduction**. For 0.5B the improvement
is modest since B already fits in L2.

### Variants Tried

| Variant | Technique | VGPRs | Result |
|---------|-----------|-------|--------|
| V1 | `int4` vectorized B load + `#pragma unroll 4` | 29 | **Regressive** (2-8% slower), VGPR inflation killed occupancy |
| V2 | Manual 2× unroll with interleaved B accesses | 29 | **Regressive** (~12% slower) |
| V3 | `#pragma clang loop unroll_count(2)` + `__restrict__` | 29 | **Regressive** (17-19% slower on 3B) |
| V4 | Remove dead tail path (N%128==0 for targets) | 17 | **Promising** (~2% on quick check, ~9% stability) |
| V5 | LDS B tile caching with `__syncthreads` | — | **Regressive** (15% slower on 3B) — no swizzle, no pipelining |
| V6 | Production-style index addressing (recompute `d_A+m*K+kg*4`) | 17 | Identical ISA to V4 |
| V7 | Software-pipelined loop (pre-load → compute → rotate) | 17 | ~10% 3B speedup (inner loop only) |
| V8 | V7 + swizzled grid (x=M-tiles, y=N-tiles) | 17 | ~14.5% 3B, 7.7→8.8 TOPS |
| V9a | Two-pass split-K (staging buffer + reduction kernel) | 20 | **Regressive** (3-15% slower) — reduction kernel overhead (~30μs) exceeded atomic savings |
| V9b | CPT=8 (double N-work per thread, 256-element tiles) | 24 | **Regressive** (35-38% slower) — strided B access 2×'d L2 txns despite 95% hit rate |
| **V10** | **LDS B-tile (KT=8) + grid swizzle + A pipelining** | **32** | **Winner: 1.58× on 3B (0.456ms), 12.7 TOPS** |

**Key insights**:
- On gfx906, any unrolling that pushes VGPRs past 24 drops occupancy from 10→8 waves/SIMD.
  However, V10 proves that **reduced occupancy can be offset by dramatically fewer VMEM instructions**.
- The compiler auto-generates `global_load_dwordx4` for 4 consecutive int32 reads — explicit vectorized loads don't help.
- For 3B shapes, the B working set per z-slice (5.5 MB) exceeds L2 (4 MB). The grid swizzle reduces the concurrent footprint to ~2.4 MB by ensuring blocks sharing an N-tile are scheduled together.
- 0.5B B data (1.06 MB) fits in L2 regardless, so the swizzle is neutral there.
- CPT=8 (V9b) was catastrophically regressive despite maintaining 10 waves/SIMD: the strided B access pattern (32-byte thread spacing vs 16-byte) doubled L2 cache line transactions per VMEM_RD instruction.
- Two-pass split-K (V9a) proved atomicAdd is NOT the bottleneck: the reduction kernel overhead (~30μs) exceeded any savings from eliminating atomics.
- V5 (early LDS attempt) failed because it lacked the grid swizzle and A-load pipelining that V10 combines. The cooperative loading pattern with per-sub-wave coalescing is also critical.
- The total VMEM_RD reduction from LDS (62.5%) far outweighs the occupancy cost of 32 VGPRs (20% fewer waves/SIMD).

### Playground Benchmark (V10, warmup=4, iters=16)

```
[Shape] Qwen2.5-0.5B_FFN_Up (M=128 N=4864 K=896)
  baseline:    0.1841 ms    6.059 TOPS
  candidate:   0.1149 ms    9.707 TOPS
  speedup:     1.60x

[Shape] Qwen2.5-0.5B_FFN_Gate (M=128 N=4864 K=896)
  baseline:    0.1819 ms    6.134 TOPS
  candidate:   0.1154 ms    9.669 TOPS
  speedup:     1.58x

[Shape] Qwen2.5-3B_FFN_Up (M=128 N=11008 K=2048)
  baseline:    0.7183 ms    8.034 TOPS
  candidate:   0.4560 ms   12.658 TOPS
  speedup:     1.58x

[Shape] Qwen2.5-3B_FFN_Gate (M=128 N=11008 K=2048)
  baseline:    0.7224 ms    7.990 TOPS
  candidate:   0.4555 ms   12.670 TOPS
  speedup:     1.59x
```

**Summary**: **~58% throughput improvement** on all FFN shapes (6.1→9.7 TOPS on 0.5B,
8.0→12.7 TOPS on 3B). **12.7 TOPS INT8 on 3B shapes** — up from 8.8 TOPS (V8).
Perfect INT32 parity (max_abs_diff=0) on all shapes.

### V10 Profiling (rocprof v1, gfx906)

**Instruction Mix (3B, per wave)**:

| Metric | Baseline | V10 | Change |
|--------|----------|-----|--------|
| VGPR | 20 | 32 | +12 (8 waves/SIMD) |
| VMEM_RD/wave | 256 | 96 | **-62.5%** |
| VALU/wave | 1067 | 876 | -18% |
| VMEM_WR/wave | 4 | 4 | same |

**L2 Cache (3B)**:

| Metric | Baseline | V10 | Change |
|--------|----------|-----|--------|
| TCC_HIT/wave | 258 | 320 | +24% |
| TCC_MISS/wave | 98 | 32.5 | **-67%** |
| L2 Hit Rate | 72.7% | 90.8% | +18pp |
| HBM Traffic | 141 MB | 46 MB | **3× reduction** |

**Why VMEM_RD = 96 (not 128)**:
- A loads: 128 k-groups of consecutive int32 → compiler vectorizes to 2× dwordx4
  per KT=8 tile = 2 × 16 tiles = 32 VMEM_RD
- B cooperative loads: 4 per wave per tile × 16 tiles = 64 VMEM_RD
- Total: 32 + 64 = 96 ✓

### Production Validation (`v2_perf_rocm_prefill_dispatch_comparison`)

The production dispatch comparison (which uses the CK library for FFN, not this playground kernel) shows that CK legacy is still faster for FFN Up/Gate shapes. This playground kernel is a candidate for replacing the CK path once the ~6% FFN gap is closed:

For direct production-path benchmarking of the grid-kpar kernel body, use:

> Note: V8-style grid swizzle is now default-on in production. Use
> `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_SWIZZLE=0` to force unswizzled
> behavior for A/B checks.

```bash
# baseline loop body
LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=0 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_SWIZZLE=0 \
./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison

# V7 software-pipelined loop body
LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_SWIZZLE=1 \
./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison

# V10 LDS B-tile + swizzle + pipelined loop body
LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=2 \
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_SWIZZLE=1 \
./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison
```

`LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY` values:
- `0`: baseline production loop body
- `1`: software-pipelined loop body (V7-style)
- `2`: LDS B-tile + software-pipelined loop body (V10-style)

| Shape Class | CK Legacy (ms) | New Native (ms) | Speedup | Status |
|-------------|----------------|-----------------|---------|--------|
| Attention | 0.865 / 1.804 | 0.696 / 1.784 | 1.24x / 1.01x | **New wins** |
| FFN_Down | 2.267 / 5.075 | 2.013 / 4.844 | 1.13x / 1.05x | **New wins** |
| FFN_Up | 1.919 / 4.420 | 2.039 / 4.902 | 0.94x / 0.90x | CK still ahead |
| FFN_Gate | 1.937 / 4.438 | 2.045 / 4.910 | 0.95x / 0.90x | CK still ahead |
| LM_Head | 45.41 / 47.46 | 56.73 / 72.31 | 0.80x / 0.66x | CK still ahead |
