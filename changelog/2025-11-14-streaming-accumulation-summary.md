# Streaming Accumulation Experiment - Executive Summary

**Date**: November 14, 2025  
**Duration**: ~2 hours  
**Result**: ❌ Experiment failed - Reverted to original algorithm

---

## Quick Facts

| Metric | Original | Streaming | Winner |
|--------|----------|-----------|--------|
| **Performance** | 456.8 GFLOPS | 360.3 GFLOPS | **Original** ✅ |
| **L1 miss rate** | 45.43% | 4.57% | Streaming ✅ |
| **IPC** | 2.08 | 1.87 | **Original** ✅ |
| **Overall** | **Well-balanced** | Cache-optimized but serial | **Original** ✅ |

---

## What We Tried

**Hypothesis**: The 45% L1 cache miss rate is killing performance. If we eliminate the 458 KB `accum_vec` buffer by streaming directly to FP32, we'll get 2-3× speedup.

**Implementation**:
- Added `microkernel_streaming()` that accumulates to 16 KB FP32 buffer (fits L1)
- Eliminated 458 KB INT32 accumulator buffer
- Single-pass algorithm (compute + accumulate simultaneously)

---

## What We Found

**L1 cache improvement was HUGE**: 45% → 4.5% miss rate (10× better!)

**But performance got WORSE**: 456 → 360 GFLOPS (20% slower!)

### Why?

The streaming version introduced worse bottlenecks:

1. **114,688 horizontal reductions** (`_mm512_reduce_add_epi32` @ ~10 cycles each)
2. **896 FP32 divisions per microkernel** (scalar division @ ~10-15 cycles)
3. **+27% more instructions** (136B → 173B total)
4. **Lower IPC** (2.08 → 1.87 due to serial dependencies)

**The original algorithm is smarter**:
- Batches reductions (few serial bottlenecks)
- Uses SIMD division (better throughput)
- Tolerates 45% L1 miss rate via hardware prefetching
- High IPC (2.08) from vectorized operations

---

## Key Insight

**Cache misses alone don't determine performance.**

The performance equation is:
```
Performance = f(cache_misses, instruction_latency, IPC, instruction_count)
```

Optimizing only cache_misses can make other factors worse.

The original 3-pass algorithm has the best **overall balance**.

---

## What Remains

**Streaming accumulation code** is still in the codebase:
- `microkernel_streaming()` function (lines 997-1142)
- `STREAMING_PARAM` template parameter
- Disabled by default (`STREAMING=false`)
- Available as `Q8_1GemmKernelStreaming` alias

**Why keep it?**
- Documents the experiment
- Useful for future research
- Shows cache behavior can be improved (for other bottlenecks)

---

## Next Steps

Based on this experiment, **DO NOT** pursue streaming accumulation further.

Instead, focus on:

1. **Profile instruction mix** with `perf record -e cycles:pp`
2. **Investigate register blocking** (tile MR×NR to 8×32)
3. **Analyze memory access patterns** to improve hardware prefetch effectiveness
4. **Consider SIMD optimizations** for existing algorithm

The 456 GFLOPS baseline is actually good - don't break it trying to "fix" L1 cache.

---

## Lessons Learned

1. **Hardware prefetchers are powerful** - Even 45% L1 miss rate can be tolerable
2. **IPC matters** - Serial operations (divisions, reductions) kill performance
3. **Instruction count matters** - More instructions = more cycles, even with better cache
4. **Profiling is essential** - Our initial analysis was incomplete
5. **Don't optimize in isolation** - Always measure overall performance

---

## Files Changed

**Modified** (kept streaming code, disabled by default):
- `src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`
  - Added `microkernel_streaming()` function
  - Added `STREAMING_PARAM` template parameter
  - Updated dispatcher to choose between streaming and original
  - Default: `STREAMING=false` (original algorithm)

**Documentation**:
- `changelog/2025-11-14-q8-1-streaming-accumulation-analysis.md` - Full analysis
- `perf_analysis_q8_1_gemm.md` - Updated with streaming results
- `changelog/2025-11-14-streaming-accumulation-summary.md` - This file

**No functional changes** - Original performance restored (456.8 GFLOPS).

---

## Bottom Line

**Streaming accumulation was a good idea that didn't work.**

The problem wasn't as simple as "L1 cache thrashing". The original algorithm is actually well-designed and balanced. Future optimizations should focus on instruction mix and memory access patterns, not cache miss rate alone.

**Production status**: Original 3-pass algorithm @ 456.8 GFLOPS ✅
