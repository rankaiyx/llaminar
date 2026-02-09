# Kernel Tuning & Optimisation Project

**Started**: 2026-02-09  
**Model**: Qwen2.5-7B-Instruct IQ4_NL (28 layers, 3584 hidden dim, 28 heads)  
**Benchmark**: 596 prefill tokens + 128 decode tokens, 3 iterations averaged after warmup  

---

## Baseline Numbers (2026-02-09)

### Throughput

| Metric | ROCm (MI50 gfx906) | CUDA (RTX 3090 SM 8.6) | CUDA / ROCm |
|---|---|---|---|
| **Prefill** | 172.8 tok/s | 1,493.7 tok/s | 8.6x |
| **Decode** | 14.2 tok/s | 25.7 tok/s | 1.8x |
| **Wall-clock total** | 12,487 ms | 5,385 ms | 2.3x |

### GPU Kernel Time Breakdown — Decode Phase

| Kernel | ROCm (ms) | ROCm % | CUDA (ms) | CUDA % | ROCm / CUDA |
|---|---|---|---|---|---|
| **GEMM** (CK / CUTLASS) | 10,847 | 46% | 5,710 | 49% | 1.9x |
| **Flash Attn Decode** | 12,003 | 51% | 5,634 | 48% | 2.1x |
| RMS Norm | 414 | 1.7% | 153 | 1.3% | 2.7x |Le
| Residual Add | 156 | 0.7% | 93 | 0.8% | 1.7x |
| RoPE | 126 | 0.5% | 60 | 0.5% | 2.1x |
| SwiGLU | 81 | 0.3% | 47 | 0.4% | 1.7x |
| Embedding | 6 | 0.0% | 3 | 0.0% | 2.0x |
| **Decode GPU total** | **23,632** | | **11,700** | | **2.0x** |

### GPU Kernel Time Breakdown — Prefill Phase

| Kernel | ROCm (ms) | CUDA (ms) | ROCm / CUDA |
|---|---|---|---|
| **Flash Attn Prefill** | 7,840 | 861 | 9.1x |
| **GEMM** (CK / CUTLASS) | 2,306 | 111 | 20.8x |
| RoPE | 5 | 39 | 0.1x (CUDA slower?) |
| SwiGLU | 16 | 14 | 1.2x |
| RMS Norm | 10 | 5 | 2.0x |
| Residual Add | 8 | 6 | 1.4x |
| **Prefill GPU total** | **10,185** | **1,035** | **9.8x** |

### Per-Kernel Averages — Decode

| Kernel | ROCm avg (µs) | CUDA avg (µs) |
|---|---|---|
| GEMM per call | 300.8 | 76.4 |
| Flash Attn Decode per call | 1,116 | 524 |
| RMS Norm per call | 19.2 | 7.1 |
| RoPE per call | 12.1 | 9.2 |
| SwiGLU per call | 9.0 | 5.6 |
| Residual Add per call | 7.6 | 4.5 |

---

## Observations

1. **Decode is 97% GEMM + Attention** on both GPUs. Everything else is noise.
2. **Prefill on ROCm is catastrophically slow** — 9.8x slower than CUDA overall.
   - GEMM prefill is **20.8x slower** — likely dominated by VNNI→CK repack overhead.
   - Flash Attn prefill is **9.1x slower** — MI50 FA2 kernel vs optimised cuDNN/FA2 on Ampere.
3. **Decode GEMM on ROCm is only 1.9x slower** — reasonable given MI50 INT8 vs RTX 3090 INT8 hardware.
4. ROCm GEMM call count (43.7K) vs CUDA (76.2K) — different dispatch granularity.

---

## Tuning Roadmap

### Phase 1: ROCm Flash Attention Decode (COMPLETED)
- [x] **v2: float4 + parallel output** — 1,116 → 382 µs/call (2.92x, +40% decode tok/s)
- [x] **v3: wavefront-cooperative KV** — 382 → 57.6 µs/call (6.6x, +71% decode tok/s from baseline)
- [x] **exp2f hardware intrinsic** — 57.6 → 56.2 µs/call (2.4% kernel improvement, negligible end-to-end)
- [x] **GQA-aware blocks** — ATTEMPTED & REVERTED: 5.1× regression due to CU starvation (32 vs 224 blocks on 60 CUs)

### Phase 2: ROCm GEMM Decode (NEXT — 89% of decode time)
- [ ] **GEMM decode** — CK INT8 GEMV tuning, tile sizes, occupancy (303.5 µs/call avg, 10,954 ms total)
- [ ] **GEMM prefill repack overhead** — VNNI→row-major repack runs PER CK GEMM call (~65-200µs each)
- [ ] Ops kernels (RMS Norm, RoPE, SwiGLU) — low priority, <3% of time

### Phase 2: CUDA Kernel Tuning  
- [ ] CUTLASS GEMM tile configuration
- [ ] Flash Attention decode optimisation
- [ ] Fused kernel opportunities

---

## Architecture Notes

### ROCm GEMM Kernel Dispatch

```
multiply_tensor() / multiply_fused_tensor()
  ├── M == 1  →  GEMV fast path (reads VNNI layout directly)
  │     ├── rocmQuantGemm_quantizeActivations()   FP32 → INT8 + scale
  │     ├── rocmGemv_int8_int8_int32_vnni()       INT8×INT8 GEMV
  │     └── rocmQuantGemm_applyScaling()          INT32 → FP32
  │
  └── M > 1  →  CK GEMM path (needs row-major)
        ├── rocmQuantGemm_quantizeActivations()   FP32 → INT8 + scale
        ├── rocmGemv_repackVNNI_to_rowmajor()     ⚠️ 65-200µs PER CALL
        ├── rocmQuantGemm_executeNoScale()         CK INT8×INT8→INT32
        └── rocmQuantGemm_applyScaling()           INT32 → FP32
```

### Weight Storage (Option B — VNNI-only on device)

- **VNNI layout** `[K/4 × N × 4]` — sole persistent device copy, optimal for GEMV decode
- **Row-major scratch** `d_B_rowmajor_scratch` — workspace buffer, repacked on-demand for CK
- Dual-layout would cost ~2× VRAM for weights but eliminate ~15-25ms repack per prefill

### Repack Cost Analysis (Qwen 7B, 28 layers)

Per layer in prefill:
- QKV fused (3 proj): `3584×3584` × 3 = 3 repacks
- Wo projection: `3584×3584` × 1 = 1 repack  
- GateUp fused (2 proj): `18944×3584` × 2 = 2 repacks (expensive!)
- Down projection: `3584×18944` × 1 = 1 repack (expensive!)

Per prefill: **28 × ~7 = ~196 repacks**  
At ~65-200µs each: **~15-35ms pure repack overhead**  
(Actual total prefill GEMM time on ROCm: 2,306ms — repack is ~1-1.5%, not dominant)

### Call Count Discrepancy

ROCm GEMM: 43.7K calls — profiled at `multiply_tensor()` entry  
CUDA GEMM: 76.2K calls — profiled per-CUTLASS-call inside projection loops  
Ratio: 1.74× matches average ~1.7 sub-projections per fused call (QKV=3, GateUp=2)

---

## Tuning Log

### Entry 5: exp2f Hardware Intrinsic (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Changes**:
1. **`fast_expf()` inline helper** — `exp2f(x * 1.4426950408889634f)` replaces all `expf()` calls
2. On GCN ISA, `exp2f` maps to `v_exp_f32` (single-cycle transcendental ALU instruction), vs `expf` which requires a multi-step polynomial (~50+ cycles)
3. Applied to: main loop softmax (2 calls/KV pos), inter-wavefront reduction (2 calls/wavefront pair)
4. Accuracy: ~2 ULP vs ~1 ULP for `expf`, acceptable for attention softmax

**Results**:

| Metric | v3 (expf) | v3+exp2f | Change |
|---|---|---|---|
| **Flash Attn Decode avg** | 57.6 µs | **56.2 µs** | **-2.4%** |
| **Flash Attn Decode total** | 619 ms | **604 ms** | -15 ms |
| **Decode throughput** | ~24.3 tok/s | ~24.3 tok/s | within noise |

**Conclusion**: Measurable 2.4% per-call improvement to the kernel. End-to-end impact negligible since flash attention is only ~2% of decode time. Kept as a zero-risk improvement.

### Entry 4: GQA-Aware Blocks — ATTEMPTED & REVERTED (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Architecture change**: Grid indexed by KV heads instead of Q heads. Each block processes all 7 Q heads that share a KV head, loading K/V once.

**Changes attempted**:
1. Grid changed from `(n_heads=28, num_splits=8, batch=1)` = 224 blocks to `(n_kv_heads=4, num_splits=8, batch=1)` = 32 blocks
2. All 7 Q vectors loaded into shared memory (3.5KB vs 512B)
3. K/V cached in lane-local registers, reused across 7 Q heads
4. 7× wavefrontReduceSum per KV position (dot product for each Q head)
5. Inter-wavefront reduction processed one Q head at a time (3 syncs × 7 heads)

**Results**: **5.1× regression — REVERTED**

| Metric | v3 | GQA (reverted) | Change |
|---|---|---|---|
| **Flash Attn Decode avg** | 57.6 µs | **296.1 µs** | **5.1× slower** |
| **Decode throughput** | ~24.3 tok/s | **20.29 tok/s** | **-16.5%** |

**Root cause**: CU starvation. MI50 has 60 CUs. With 32 blocks, only 53% of CUs are active (vs 224/60 = 3.7 blocks/CU in v3). The 7× K/V bandwidth savings cannot compensate for the massive parallelism loss. Additionally, `FD_MAX_GQA_RATIO=16` array sizes (64+16+16 = 96 VGPRs for arrays alone) likely caused register spilling, further degrading performance.

**Lesson**: For short kernels on wide GPUs (60 CUs), grid occupancy dominates bandwidth savings. GQA-aware blocking only wins when: (a) the kernel is bandwidth-bound AND (b) there are enough blocks to fill the GPU. With 4 KV heads × 8 splits = 32 blocks, MI50 is fundamentally under-utilised. Would need num_splits ≈ 56 to maintain 224 blocks, but at kv_len=600 that's only ~11 KV positions per split — too small for efficient wavefront-cooperative processing.

### Entry 3: Flash Attention Decode v3 — Wavefront-Cooperative (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Architecture change**: Complete repartitioning — lanes own output dims, not KV positions.

**Changes**:
1. **Wavefront-cooperative KV processing** — each wavefront cooperates on one KV position at a time (all 64 lanes load contiguous K/V elements). Was: each thread independently loaded from separate KV positions (uncoalesced)
2. **Lanes own output dims** — each lane holds only 2 floats of O (for head_dim=128), down from 128 floats per thread. Drops VGPR usage from ~140 to ~15
3. **Zero O reduction within wavefront** — since all lanes process the same KV positions cooperatively, no cross-lane O shuffling is needed. Was: 128 × wavefrontReduceSum = 768 `__shfl_xor` per lane
4. **Dot product via wavefront reduce** — only 6 shuffles per KV position (one `wavefrontReduceSum` for the partial dot)
5. **Coalesced memory access** — adjacent lanes load adjacent floats from K and V. Perfect coalescing vs the old strided-KV-position pattern
6. **`__launch_bounds__(256, 4)`** — higher occupancy hint (was 2) since register pressure dropped dramatically
7. **Correct `__syncthreads` placement** — barriers moved outside `if` blocks (old v2 had UB with sync inside `if (wavefront_id == 0)`)

**Results**:

| Metric | Baseline (v1) | v2 | v3 | v1→v3 |
|---|---|---|---|---|
| **Decode throughput** | 14.2 tok/s | 19.86 tok/s | **~24.3 tok/s** | **+71%** |
| **Flash Attn Decode total** | 12,003 ms | 4,107 ms | **619 ms** | **-94.8%** |
| **Flash Attn Decode avg** | 1,116 µs | 382 µs | **57.6 µs** | **19.4x** |
| **Flash Attn % of decode** | 51% | 15% | **2.7%** | Eliminated as bottleneck |
| **Decode wall-clock** | ~9,014 ms | ~6,446 ms | **~5,273 ms** | **-41.5%** |

**Post-v3 decode breakdown**:

| Kernel | Time (ms) | % of decode |
|---|---|---|
| GEMM_CK | 10,954 | 89% |
| Flash Attn Decode | 619 | 5% |
| RMS Norm | 412 | 3% |
| Residual Add | 158 | 1% |
| RoPE | 124 | 1% |
| SwiGLU | 80 | <1% |

**Conclusion**: Flash attention is no longer a meaningful optimisation target. GEMM is 89% of decode. Further attention work (GQA-aware blocks, exp2f) deprioritised — combined they'd yield <6% decode improvement.

### Entry 2: Flash Attention Decode Optimisation — ROCm (2026-02-09)

**Target**: `flash_decoding_mi50_kernel` in `ROCmFlashAttentionKernels.hip`

**Changes**:
1. **float4 vectorized K/V loads** — dot product and V accumulation now use `float4`, reducing memory transactions by 4×
2. **Deferred O_local rescale** — online softmax `O_local[128]` rescale fused with V accumulation (single `scale_old * o + p * v` pass)
3. **Efficient wavefront reduction** — new `wavefrontReduceML()` reduces ONLY m and l across 64 lanes (12 `__shfl_xor` calls instead of 780). Each thread then rescales its own O_local to the converged m
4. **Parallel output write** — all lanes in wavefront 0 write their portion of head_dim (was: thread 0 serial loop over 128 elements)
5. **`__launch_bounds__(256, 2)`** on decode kernel — guides compiler register allocation for better occupancy
6. **`__launch_bounds__(256, 4)`** on reduce kernel — reduce kernel is register-light, can support higher occupancy
7. **Hoisted KV pointer arithmetic** — `kv_head_offset` and `kv_stride` computed once outside loop

**Results**:

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| **Decode throughput** | 14.2 tok/s | 19.86 tok/s | **+40%** |
| **Flash Attn Decode total** | 12,003 ms | 4,107 ms | **-65.8%** |
| **Flash Attn Decode avg** | 1,116 µs | 382 µs | **2.92x faster** |
| **Decode GPU time share** | 51% | 15% | GEMM now dominant |
| **GEMM decode** | 10,847 ms | 10,904 ms | unchanged |
| **Decode wall-clock** | ~9,014 ms | ~6,446 ms | **-28.5%** |

**Post-optimisation decode breakdown**:

| Kernel | Time (ms) | % of decode |
|---|---|---|
| GEMM_CK | 10,904 | 69% |
| Flash Attn Decode | 4,107 | 26% |
| RMS Norm | 416 | 2.6% |
| Residual Add | 160 | 1.0% |
| RoPE | 124 | 0.8% |
| SwiGLU | 81 | 0.5% |

**Next target**: GEMM decode is now the clear bottleneck (69% of decode GPU time).

### Entry 1: Baseline (2026-02-09)
- Established baseline profiling with phase-separated GPU event timing
- Fixed ROCm kernel profiling (was using CPU chrono instead of HIP events)
- Added prefill/decode phase breakdown to profiling output
