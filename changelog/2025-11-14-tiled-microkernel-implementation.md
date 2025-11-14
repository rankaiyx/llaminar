# Tiled MR×NR Microkernel Implementation (Nov 14, 2025)

## Summary

Implemented **tiled MR×NR processing** for Q8_1 GEMM kernel to reduce L1 cache footprint while preserving high-IPC algorithm structure. This addresses the 45% L1 D-cache miss rate discovered in the original implementation without repeating the performance regression seen with streaming accumulation (20% slower).

## Motivation

**Original Algorithm Performance** (MR=32, NR=128):
- **Performance**: 456.8 GFLOPS (excellent)
- **L1 D-cache miss rate**: 45.43% (poor)
- **Working set**: ~458 KB accum_vec buffer
- **IPC**: 2.08 (high - good parallelism)

**Streaming Accumulation Experiment** (FAILED):
- **Performance**: 360.3 GFLOPS (-20% regression)
- **L1 D-cache miss rate**: 4.57% (10× better!)
- **Working set**: ~16 KB (fits L1)
- **IPC**: 1.87 (degraded due to serial dependencies)
- **Root cause**: 114,688 horizontal reductions + 896 FP32 divisions per microkernel killed instruction-level parallelism

**Lesson Learned**: L1 cache miss rate alone doesn't determine performance. Instruction mix and IPC are equally critical.

## Strategy: Tiled MR×NR Processing

**Design Goals**:
1. ✅ Reduce L1 footprint to <32 KB (L1 cache size)
2. ✅ Preserve original algorithm structure (vectorized dpbusd, batched reductions)
3. ✅ Maintain high IPC (~2.08)
4. ✅ Target ≥420 GFLOPS (≤10% regression acceptable)

**Tile Parameters**:
```
TILE_MR = 8
TILE_NR = 32
Per-tile footprint = 8 × 32 × 28 × 4 = 28.7 KB (fits L1!)
Tiles per microkernel = (32 / 8) × (128 / 32) = 4 × 4 = 16 tiles
```

**Algorithm Structure**:
```cpp
for (int i_tile = 0; i_tile < MR; i_tile += TILE_MR) {           // 4 iterations
    for (int j_tile = 0; j_tile < NR; j_tile += TILE_NR) {       // 4 iterations
        // Allocate tile-local buffers (~29 KB total)
        std::vector<int32_t> accum_tile(TILE_MR * TILE_NR * K_blocks, 0);
        std::vector<int16_t> sum_qs_tile(TILE_MR * K_blocks);
        std::vector<uint16_t> a_scales_tile(TILE_MR * K_blocks);
        
        // K-loop: Same vectorized dpbusd as original
        for (int kb = 0; kb < K_blocks; ++kb) {
            for (int ir = 0; ir < TILE_MR; ++ir) {
                // Load A block, extract sum_qs, scale
                // Inner loop: 32 vectorized dpbusd operations (same as original)
            }
        }
        
        // Post-processing: Same batched SIMD reductions as original
        for (int ir = 0; ir < TILE_MR; ++ir) {
            for (int jr_batch = 0; jr_batch < TILE_NR / 16; ++jr_batch) {
                // Load B scales, convert accum to FP32, apply compensations
                // Write 16 results to C (vectorized store)
            }
        }
    }
}
```

## Implementation Details

### Files Modified

**src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h** (2759 lines):

1. **Lines 129-135**: Added `USE_TILING_PARAM` template parameter
   ```cpp
   template <int MR_PARAM = 32, int NR_PARAM = 128, int PREFETCH_A_PARAM = 1,
             int NC_PARAM = 0, int KC_PARAM = 0, int JR_UNROLL_PARAM = 2, int JR_BATCH_PARAM = 18,
             bool STREAMING_PARAM = false, bool USE_TILING_PARAM = false>
   class Q8_1GemmKernelTemplate
   ```

2. **Line 146**: Added static constexpr for tiling flag
   ```cpp
   static constexpr bool USE_TILING = USE_TILING_PARAM; // Tiled MR×NR processing
   ```

3. **Lines 1135-1439**: **NEW `microkernel_tiled()` function (305 lines)**
   - Outer tile loops (4×4 = 16 tiles)
   - Per-tile buffers: accum_tile (28.7 KB), sum_qs_tile, a_scales_tile
   - Same K-loop logic: Load A, extract sum_qs, vectorized dpbusd
   - Same post-processing: Batched SIMD reductions, compensations
   - Write results to C with global indices `(i_tile + ir) * ldc + (j_tile + jr)`

4. **Lines 2470-2495**: Updated dispatcher `microkernel_full()`
   ```cpp
   if constexpr (USE_TILING) {
       // EXPERIMENTAL: Tiled MR×NR processing (8×32 tiles, ~29 KB per tile, L1-friendly)
       microkernel_tiled(K_blocks, i_base, kc_start, A_decodable, B_packed, C, ldc);
   } else if constexpr (STREAMING) {
       // EXPERIMENTAL: Streaming accumulation (disabled - 20% slower)
       microkernel_streaming(...);
   } else {
       // Default: Original 3-pass algorithm
       microkernel_full_sumqs(...);
   }
   ```

5. **Lines 2717-2720**: Added kernel alias for testing
   ```cpp
   using Q8_1GemmKernelTiled = Q8_1GemmKernelTemplate<32, 128, 1, 0, 0, 2, 18, false, true>;
   ```

### Key Implementation Choices

**Why 8×32 tiles?**
- **Footprint**: 8×32×28×4 = 28.7 KB < 32 KB (L1 cache)
- **Tile count**: 16 tiles (4×4) = reasonable overhead
- **Balance**: Not too small (excessive looping) or too large (spills to L2)

**Alternatives considered**:
- 8×16: 14.3 KB (32 tiles) - too many tiles, more loop overhead
- 4×32: 14.3 KB (32 tiles) - same issue
- 16×64: 57.3 KB - doesn't fit L1

**Why preserve original algorithm?**
- Streaming accumulation failed due to serial dependencies (horizontal reductions, FP32 divisions)
- Original algorithm has excellent IPC (2.08) and GFLOPS (456)
- Only problem is L1 footprint - tiling fixes that without changing instruction mix

## Testing Plan

### Step 1: Baseline Performance (Default Kernel)

**Command**:
```bash
cd build_v2_release
./tests/v2/v2_perf_q8_1_gemm --gtest_filter='Q8_1_GEMM_Performance.LargeBatchedPrefill'
```

**Expected**: 450-460 GFLOPS (from comprehensive parameter sweep)

### Step 2: Enable Tiling and Retest

**Modify** `Q8_1GemmKernel.h` line 2718:
```cpp
// Change default from:
using Q8_1GemmKernel = Q8_1GemmKernelTemplate<32, 128, 1, 0, 0, 2, 18, false, false>;

// To:
using Q8_1GemmKernel = Q8_1GemmKernelTemplate<32, 128, 1, 0, 0, 2, 18, false, true>; // USE_TILING=true
```

**Rebuild**:
```bash
cmake --build build_v2_release --target v2_perf_q8_1_gemm --parallel
```

**Run**:
```bash
./tests/v2/v2_perf_q8_1_gemm --gtest_filter='Q8_1_GEMM_Performance.LargeBatchedPrefill'
```

**Success Criteria**: ≥420 GFLOPS (≤10% regression acceptable)

### Step 3: Profile Cache Behavior

**Command**:
```bash
perf stat -e \
  cycles,instructions,\
  L1-dcache-loads,L1-dcache-load-misses,\
  LLC-loads,LLC-load-misses \
  ./tests/v2/v2_perf_q8_1_gemm --gtest_filter='Q8_1_GEMM_Performance.LargeBatchedPrefill'
```

**Expected**:
- **L1 D-cache miss rate**: <10% (down from 45%)
- **IPC**: ~2.08 (similar to original)
- **Instructions**: ~136B (similar to original)

**Success Criteria**: L1 miss rate <15%

## Expected Outcomes

### Scenario A: Success (Enable by Default)

**Conditions**:
- ✅ GFLOPS ≥420 (≤10% regression)
- ✅ L1 miss rate <15% (significant improvement from 45%)
- ✅ IPC ≥2.0 (maintains instruction-level parallelism)

**Action**: Update default kernel to USE_TILING=true

**Rationale**: Better cache efficiency with acceptable performance tradeoff

### Scenario B: Partial Success (Keep as Option)

**Conditions**:
- ⚠️ GFLOPS 380-419 (10-20% regression)
- ✅ L1 miss rate <10% (excellent cache behavior)

**Action**: Keep tiling as optional (Q8_1GemmKernelTiled), default remains USE_TILING=false

**Rationale**: Users can trade performance for cache efficiency if needed (e.g., cache-constrained environments)

### Scenario C: Failure (Disable Tiling)

**Conditions**:
- ❌ GFLOPS <380 (>20% regression)
- OR ❌ L1 miss rate >20% (no significant improvement)

**Action**: Document failure, disable tiling, investigate other options

**Rationale**: Tiling overhead outweighs benefits

## Comparison: Streaming vs Tiling

| Metric | Original | Streaming (FAILED) | Tiling (Target) |
|--------|----------|-------------------|-----------------|
| **GFLOPS** | 456.8 | 360.3 (-20%) | ≥420 (-≤10%) |
| **L1 miss rate** | 45.43% | 4.57% | <10% |
| **IPC** | 2.08 | 1.87 | ~2.08 |
| **Instructions** | 136.5B | 173.2B (+27%) | ~136B |
| **Working set** | 458 KB | 16 KB | 29 KB per tile |
| **Algorithm** | 3-pass vectorized | Direct FP32 | 3-pass vectorized (tiled) |
| **Status** | Default | Disabled | Testing |

**Key Difference**: Streaming changed the algorithm (serial dependencies), tiling preserves it (just smaller chunks).

## Alternative Approaches (If Tiling Fails)

1. **Prefetch tuning**: Aggressive prefetch of B matrix blocks
2. **Software pipelining**: Overlap K-loop iteration N+1 loads with N compute
3. **Cache-aware blocking**: NC/KC parameter tuning for L2/L3 cache
4. **Register blocking**: Smaller MR×NR with more aggressive reuse
5. **Hybrid approach**: Tiling + software pipelining
6. **Hardware-specific**: Target Ice Lake+ cache hierarchy
7. **Accept 45% L1 miss rate**: If tiling costs too much, original may be optimal

## Next Steps

1. ✅ Implementation complete (305 lines, dispatcher wired)
2. 🔄 Build in progress (background)
3. ⏳ Run baseline test (original kernel)
4. ⏳ Enable tiling, rebuild, retest
5. ⏳ Profile with perf stat
6. ⏳ Compare results and decide on default

## Implementation Status

- **Code complete**: ✅ 305-line microkernel_tiled function
- **Dispatcher wired**: ✅ if constexpr (USE_TILING) branch
- **Kernel alias**: ✅ Q8_1GemmKernelTiled for testing
- **Build status**: 🔄 In progress (64 instantiation files compiling)
- **Testing**: ⏳ Pending build completion

## References

- **Streaming accumulation failure**: `changelog/2025-11-14-streaming-accumulation-failed.md`
- **Streaming root cause**: `changelog/2025-11-14-streaming-failure-root-cause.md`
- **Parameter sweep**: `changelog/2025-11-14-q8_1-gemm-parameter-sweep-results.md`
- **Prefetch implementation**: `changelog/2025-11-14-q8_1-gemm-prefetch-implementation.md`

---

**Author**: GitHub Copilot  
**Date**: November 14, 2025  
**Session**: Q8_1 GEMM cache optimization continuation  
