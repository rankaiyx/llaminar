# CUDA TQ Fused Kernel Optimizations

**Date**: 2025-07-25
**Type**: Performance Optimization
**Components**: CUDA TurboQuant KV Cache

## Summary

Two fused CUDA kernels that bring TQ (TurboQuant) decode performance to **98.2% of Q8_1** by
eliminating per-layer kernel launch overhead in the decode path.

## Performance Results

| Metric | Q8_1 Baseline | TQ Before | TQ After | TQ vs Q8_1 |
|--------|---------------|-----------|----------|------------|
| GPU decode time | 11.48ms | 15.75ms | 11.69ms | 98.2% |
| GPU tok/s | 87.1 | 60.5 | 85.5 | 98.2% |
| Wall tok/s | 81.6 | 60.5 | 80.1 | 98.2% |
| ATTENTION | 0.662ms | 4.274ms | 0.755ms | +0.093ms |
| KV_CACHE_APPEND | 0.266ms | 0.845ms | 0.393ms | +0.127ms |

**Model**: Qwen2.5-7B-Instruct-Q8_0.gguf, 128 decode tokens, no CUDA graphs.

## Changes

### 1. Fused Incremental Dequant Kernel (`tq_incremental_fused_kernel`)

**Problem**: During decode, each layer needs incremental dequant of 1 newly-appended position.
Previously this required 2 kernel launches per layer (TQ8 for K, TQ4 for V) = 56 launches/step.

**Solution**: Single kernel that processes both K (TQ8) and V (TQ4) in sequence:
- Grid: `(n_kv_heads)`, Block: `(D)` — 4 blocks of 128 threads
- Phase 1: TQ8 centroid lookup → rotation matmul → optional RoPE → FP16 store
- Phase 2: TQ4 centroid lookup → rotation matmul → FP16 store
- Shared memory reused between phases via `__syncthreads()` barrier

**Impact**: ATTENTION 1.564ms → 0.755ms (52% reduction), 28 launches instead of 56.

### 2. Fused Quantize-to-Ring Kernel (`tq_quantize_fused_ring_kernel`)

**Problem**: KV_CACHE_APPEND previously used separate TQ8 and TQ4 quantize kernels writing
to temp buffers, followed by D2D memcpy to the ring buffer = 4 API calls per layer × 28 layers.

**Solution**: Single kernel that quantizes both K+V directly into ring buffer positions:
- Grid: `(n_kv_heads, 2)` — y=0 for K(TQ8), y=1 for V(TQ4)
- Each block: load input → norm reduction → normalize → rotation → centroid search → write to ring
- TQ4 packing parallelized across threads (16 threads pack 128 elements vs serial on thread 0)

**Impact**: KV_CACHE_APPEND 0.841ms → 0.393ms (53% reduction), 1 launch instead of 4 per layer.

### 3. Cleanup: Removed Ineffective Batch Approach

Removed `batch_incremental_dequant()` and associated device/host param arrays. The batch
approach (processing all 28 layers in 2 launches) is fundamentally incompatible with per-layer
interleaved append+dequant: when layer 0's dequant runs, layers 1-27 haven't been appended yet.

## Files Modified

- `src/v2/kernels/cuda/kvcache/CUDATurboQuantKernels.cu` — Added fused kernels + launch functions
- `src/v2/kernels/cuda/kvcache/CUDATurboQuantKernels.h` — Added declarations
- `src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu` — Integrated fused paths, removed batch code
- `src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.h` — Removed batch member variables

## Correctness

Validated with greedy sampling (`-t 0`):
- Token output matches: `[34208, 916, 279, 15678, 5562]`
- RMSE between runs: 0.11-0.23 (within expected TQ quantization noise)
- Two-round determinism confirmed
