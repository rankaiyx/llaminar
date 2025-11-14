# Dense DPBUSD Microkernel Bug Fix

**Date**: 2025-01-XX  
**Component**: Q8_1 GEMM Kernel - Dense DPBUSD Microkernel  
**Status**: ✅ **FIXED** - All 115 tests passing

## Summary

Fixed critical correctness bug in dense DPBUSD microkernel that caused 221% relative L2 error for matrices with K_blocks ≥ 16. The bug was in the vectorized post-processing path that incorrectly assumed `sum_qs` values were contiguous in memory.

## Problem

### Symptom
- **Small matrices (K_blocks ≤ 8)**: ✅ Correct (Rel L2: 0.5%)
- **Large matrices (K_blocks = 16)**: ❌ Wrong (Rel L2: 221%, Max Abs: 53.4)
- **Pattern**: Bug appeared only when vectorized path was used (K_blocks = 16)

### Root Cause

The `sum_qs` array is stored in memory as:
```cpp
auto sum_qs = [&](int kb, int ir) -> int16_t & {
    return sum_qs_storage[kb * MR + ir];  // Row-major by kb
};
```

For a given `ir`, consecutive `kb` values are **32 elements apart** (MR=32):
- `sum_qs(0, ir)` at `sum_qs_storage[0 * 32 + ir]`
- `sum_qs(1, ir)` at `sum_qs_storage[1 * 32 + ir]`  (32 elements away)
- `sum_qs(2, ir)` at `sum_qs_storage[2 * 32 + ir]`  (64 elements away)
- etc.

However, the vectorized path (line 1726) incorrectly assumed they were contiguous:
```cpp
// WRONG: Loads 16 consecutive int16 values
__m512i sum_qs_vec_i16 = _mm512_cvtepi16_epi32(
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&sum_qs(kb_base, ir)))
);
```

This loaded:
- `sum_qs(kb_base, ir+0)`, `sum_qs(kb_base, ir+1)`, ..., `sum_qs(kb_base, ir+15)`  ❌ WRONG

Instead of:
- `sum_qs(kb_base+0, ir)`, `sum_qs(kb_base+1, ir)`, ..., `sum_qs(kb_base+15, ir)`  ✅ CORRECT

### Why Small K Worked

The bug only manifested when `K_blocks == VECTOR_WIDTH (16)`:

| K_blocks | Vectorized? | Result |
|----------|-------------|--------|
| 1 | No (tail path) | ✅ PASS |
| 8 | No (tail path) | ✅ PASS |
| 16 | Yes (vectorized) | ❌ FAIL (221% error) |

The scalar tail path (lines 1733-1737) correctly gathered non-contiguous values:
```cpp
// CORRECT: Iterates over kb dimension
for (int k = 0; k < actual_kb_count; ++k) {
    sum_qs_tmp[k] = static_cast<float>(sum_qs(kb_base + k, ir));
}
```

## Solution

Changed the vectorized path to **always use scalar gathering** (same logic as tail path):

```diff
- // Load sum_qs (handle partial vector at end)
- __m512 sum_qs_vec_f32;
- if (actual_kb_count == VECTOR_WIDTH)
- {
-     __m512i sum_qs_vec_i16 = _mm512_cvtepi16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(&sum_qs(kb_base, ir))));
-     sum_qs_vec_f32 = _mm512_cvtepi32_ps(sum_qs_vec_i16);
- }
- else
- {
-     // Scalar path for tail
-     alignas(64) float sum_qs_tmp[VECTOR_WIDTH] = {0};
-     for (int k = 0; k < actual_kb_count; ++k)
-     {
-         sum_qs_tmp[k] = static_cast<float>(sum_qs(kb_base + k, ir));
-     }
-     sum_qs_vec_f32 = _mm512_load_ps(sum_qs_tmp);
- }
+ // Load sum_qs (handle partial vector at end)
+ // NOTE: sum_qs is stored as [kb * MR + ir], so values for different kb
+ // are NOT consecutive in memory. Must use scalar path to gather them.
+ __m512 sum_qs_vec_f32;
+ {
+     // Scalar path (always, since sum_qs layout is not contiguous in kb)
+     alignas(64) float sum_qs_tmp[VECTOR_WIDTH] = {0};
+     for (int k = 0; k < actual_kb_count; ++k)
+     {
+         sum_qs_tmp[k] = static_cast<float>(sum_qs(kb_base + k, ir));
+     }
+     sum_qs_vec_f32 = _mm512_load_ps(sum_qs_tmp);
+ }
```

**File**: `/workspaces/llaminar/src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`  
**Lines**: 1721-1738

## Validation

### Test Results

**Before Fix**:
```
[Large 512×512×512] Rel L2: 2.216e+00, Max Abs: 5.344e+01  ❌ FAILED
Expected: (rel_l2) < (0.01), actual: 2.2164676874713529
```

**After Fix**:
```
[Large 512×512×512] Rel L2: 5.321e-03, Max Abs: 2.066e-01  ✅ PASSED
[  PASSED  ] 115 tests (including config space sweep)
```

### Full Test Suite

```bash
LLAMINAR_USE_DENSE_DPBUSD=1 ./build_v2_release/tests/v2/v2_test_q8_1_gemm_kernel
```

**Results**:
- ✅ **115/115 tests passed** (100%)
- ✅ **7,020 kernel configurations tested** (config space sweep)
- ✅ **0 failures**
- ⏱️ **223 seconds** total runtime

### Tested Configurations

| Test | K_blocks | Status | Error |
|------|----------|--------|-------|
| SmallMatrix_8x8x32 | 1 | ✅ PASS | Rel L2: 0.52% |
| SingleKBlock_32x32x32 | 1 | ✅ PASS | Rel L2: 0.54% |
| Q8_1_times_Q8_0_BasicCorrectness | 8 | ✅ PASS | Rel L2: 0.53% |
| LargeMatrix_512x512x512 | 16 | ✅ PASS | Rel L2: 0.53% |

## Performance Impact

The scalar gathering path has minimal performance impact:
- **Overhead**: ~16 scalar loads + array stores per vector block
- **Frequency**: Once per `(ir, kb_base)` pair (amortized over all `jr`)
- **Compared to**: 32×128 dpbusd operations per microkernel

**Expected**: <1% performance difference (dominated by dpbusd, not sum_qs loading)

## Alternative Solutions Considered

### 1. Change Storage Layout ❌
```cpp
// Store sum_qs contiguously in kb dimension
auto sum_qs = [&](int kb, int ir) -> int16_t & {
    return sum_qs_storage[ir * K_blocks + kb];  // Column-major
};
```
- **Pros**: Enables vectorized load
- **Cons**: Changes indexing throughout code, harder to verify correctness

### 2. Use AVX-512 Gather Instructions ❌
```cpp
__m512i indices = _mm512_setr_epi32(0*32, 1*32, 2*32, ..., 15*32);
__m512i sum_qs_vec = _mm512_i32gather_epi32(indices, &sum_qs(kb_base, ir), 2);
```
- **Pros**: Vectorized solution
- **Cons**: Complex, gather instructions are slow (similar to scalar loop)

### 3. Scalar Gathering (Selected) ✅
- **Pros**: Simple, proven correct (same as working tail path), easy to verify
- **Cons**: Not vectorized (but negligible performance impact)

## Code Files Changed

1. **`src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`** (lines 1721-1738):
   - Removed vectorized `sum_qs` load path
   - Always use scalar gathering (same as tail path)
   - Added explanatory comment about memory layout

2. **`src/v2/utils/DebugEnv.h`** (lines 85-107):
   - Added `use_dense_dpbusd` flag to `GemmConfig`
   - Environment variable: `LLAMINAR_USE_DENSE_DPBUSD=1`

3. **`src/v2/kernels/cpu/gemm_v2/Q8_1GemmKernel.h`** (lines 2790-2823):
   - Modified dispatcher to check runtime flag
   - Enables testing both modes from same binary

## Runtime Configuration

**Enable Dense Mode**:
```bash
export LLAMINAR_USE_DENSE_DPBUSD=1
./llaminar2 [args...]
```

**Disable (Default)**:
```bash
unset LLAMINAR_USE_DENSE_DPBUSD
./llaminar2 [args...]
```

## Lessons Learned

1. **Memory Layout Assumptions Are Dangerous**:
   - Always verify vectorized loads match storage layout
   - Don't assume row-major vs column-major without checking lambda definition

2. **Scalar Tail Paths Are Reference Implementations**:
   - If scalar tail works but vectorized fails, check memory layout assumptions
   - Tail paths often handle edge cases correctly that vectorized paths miss

3. **K-Dimension Dependence Revealed Layout Bugs**:
   - Bug scaled with `K_blocks`, not `M` or `N`
   - Indicated issue with K-dimension indexing (sum_qs indexed by kb)

4. **Test Coverage Gaps**:
   - Tests covered K_blocks=1,8 (tail path) but not K_blocks=16 (vectorized)
   - Need tests that specifically exercise vectorized vs tail paths

## Future Work

1. **Performance Profiling**:
   - Measure actual performance difference between scalar/vectorized sum_qs loads
   - If significant, consider changing storage layout or using gather instructions

2. **Storage Layout Optimization**:
   - Consider SOA (Structure of Arrays) layout for better vectorization:
     ```cpp
     // All kb=0 values, then all kb=1 values, etc.
     sum_qs_storage[0 ... MR-1]     // kb=0, all ir
     sum_qs_storage[MR ... 2*MR-1]  // kb=1, all ir
     ```

3. **Add Test Coverage**:
   - Explicit tests for K_blocks=16,32,64 (vectorized path)
   - Test with K_blocks not divisible by VECTOR_WIDTH

## References

- **Bug Report**: Session 2025-01-XX
- **Fix Commit**: [Hash TBD]
- **Test Suite**: `tests/v2/v2_test_q8_1_gemm_kernel`
- **Architecture Docs**: `.github/copilot-instructions.md` (V2 Kernel Development)
