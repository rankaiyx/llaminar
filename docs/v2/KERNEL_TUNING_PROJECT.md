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

### Phase 1: ROCm Kernel Tuning
- [ ] **GEMM prefill repack overhead** — VNNI→row-major repack runs PER CK GEMM call (~65-200µs each, ~168 per prefill). Store dual layout on device to eliminate.
- [x] **Flash Attention decode** — ~~1,116 µs/call avg~~ → **382 µs/call** (2.92x speedup, 40% decode throughput gain)
- [ ] **GEMM decode** — CK INT8 GEMV tuning, tile sizes, occupancy (300 µs/call avg)
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
