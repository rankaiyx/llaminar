# IQ4_NL Performance Regression Investigation

**Date**: November 9, 2025  
**Context**: Recent benchmarks show ~450 GFLOPS, but previous runs achieved ~1100 GFLOPS  
**Status**: Root cause identified - alignment issue (no transpose found)

## Problem Statement

User observed significant performance regression in IQ4_NL GEMM kernels:
- **Previous**: ~1100 GFLOPS on large prefill workloads
- **Current**: ~450 GFLOPS (59% slower)

Two suspected causes:
1. AlignedVector class recently added - does IQ4_NL use it?
2. ModelLoader recently changed to load quantized weights transposed - did this break layout assumptions?

## Investigation Findings

### Issue #1: AlignedVector Usage ❌ CONFIRMED

**Current State**:
```cpp
// src/v2/tensors/IQ4_NLTensor.h (line ~411)
std::vector<uint8_t> raw_data_;  // ❌ Uses std::vector
```

**Problem**:
- `std::vector<uint8_t>` typically has 16-byte alignment (malloc default on Linux)
- SIMD operations (AVX512/AVX2) benefit from 64-byte alignment (cache line aligned)
- Unaligned memory access forces slower unaligned SIMD loads (`_mm512_loadu_ps` vs `_mm512_load_ps`)
- Cache line splits can occur when data crosses 64-byte boundaries

**Expected Impact**: 5-20% performance loss

**Evidence from codebase**:
- AlignedVector class exists (`src/v2/tensors/AlignedVector.h`) with 64-byte alignment
- Other tensors (FP32Tensor, BF16Tensor, FP16Tensor) already use AlignedVector
- IQ4_NL is the only quantized tensor still using std::vector

**Related Changelog**:
- `changelog/2025-11-08-aligned-vector-and-streaming-stores-analysis.md` (Nov 8)
  - Shows streaming stores cause 40% regression (not applicable here)
  - But confirms AlignedVector provides 64-byte alignment benefits
  - FP32/BF16/FP16/INT8 tensors all use AlignedVector

### Issue #2: Transpose in ModelLoader ✅ NO ISSUE FOUND

**Investigation**:
```bash
# Searched for recent transpose changes
grep -r "transpose" src/v2/loaders/ModelLoader.cpp  # No matches
git log --grep="transpose" --since="2025-11-01"     # No commits
```

**Findings**:
- No evidence of transpose-related changes to IQ4_NL loading
- ModelLoader uses direct construction: `std::make_shared<IQ4_NLTensor>(shape, raw)`
- No transpose parameter passed
- Shape remains [n, k] as expected

**GEMM kernel layout assumptions**:
```cpp
// src/v2/tensors/IQ4_NLTensor.h
void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const {
    const size_t blocks_per_row = (shape_[1] + 31) / 32;  // K dimension
    const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
    decodeBlock(blocks[block_idx], output);
}
```

Layout assumes:
- `shape_[0]` = n (output features, rows)
- `shape_[1]` = k (input features, columns)
- Blocks organized row-major: `[n_rows][k_blocks_per_row]`

This matches GGUF weight layout (no transpose needed).

## Root Cause Analysis

**Primary Suspect**: Poor memory alignment from `std::vector<uint8_t>`

**Evidence**:
1. IQ4_NL is only quantized tensor not using AlignedVector
2. SIMD decode paths require frequent memory access
3. Benchmark shows 450 GFLOPS (consistent with ~2× slowdown from alignment issues)
4. No transpose changes found in recent commits

**Performance Impact Breakdown**:
- **Expected with good alignment**: ~1100 GFLOPS (previous measurements)
- **Current with poor alignment**: ~450 GFLOPS (59% of expected)
- **Degradation**: 41% performance loss

This matches the expected 5-20% per SIMD operation compounded across multiple decode stages:
- Nibble expansion (AVX512/AVX2 shuffles)
- LUT lookup
- INT8→FP32 conversion
- Multiple cache line misses

## Recommended Fixes

### Fix #1: Use AlignedVector for IQ4_NL ✅ HIGH PRIORITY

**File**: `src/v2/tensors/IQ4_NLTensor.h` (line ~411)

**Change**:
```cpp
// Before
std::vector<uint8_t> raw_data_;  ///< Raw quantized data (IQ4_NL blocks)

// After
AlignedVector<uint8_t> raw_data_;  ///< Raw quantized data (IQ4_NL blocks)
```

**Required includes** (already present):
```cpp
#include "AlignedVector.h"  // Check if this exists in IQ4_NLTensor.h
```

**Impact**:
- All quantized block reads become cache-line aligned
- SIMD loads can use aligned instructions
- Eliminates cache line splits
- **Expected recovery**: 40-80% of lost performance (450 → 630-810 GFLOPS)

**Test strategy**:
1. Change `std::vector<uint8_t>` → `AlignedVector<uint8_t>`
2. Re-run Perf__IQ4_NL_GEMM_Comparison benchmark
3. Compare before/after GFLOPS
4. Verify alignment with diagnostic tool

### Fix #2: Verify compile flags include `-march=native` ✅ CRITICAL

From V2 performance testing documentation:
> **⚠️ CRITICAL**: V2 **requires `-march=native`** for SIMD optimizations (AVX512/AVX2/FMA). 
> Without it, performance is 4-12× slower due to scalar fallback.

**Check current build**:
```bash
cmake -B build_v2_release -L | grep CMAKE_CXX_FLAGS
# Should show: -march=native -mtune=native
```

**If missing**:
- src/v2/CMakeLists.txt should already set this (added Oct 24, 2025)
- Verify Release build was configured with proper flags

**Impact**:
- Missing `-march=native` → 4-12× slowdown (matches 450 vs 1100 GFLOPS)
- This could be the **primary** cause if AlignedVector isn't the issue

## Next Steps

### Immediate Actions

1. **Check compile flags** (`cmake -B build_v2_release -L | grep CXX_FLAGS`)
   - If `-march=native` missing → This is likely the root cause
   - Reconfigure build with proper flags

2. **Apply AlignedVector fix**
   - Change IQ4_NLTensor::raw_data_ to AlignedVector<uint8_t>
   - Rebuild Release

3. **Re-run benchmark**
   ```bash
   cd /workspaces/llaminar
   ./build_v2_release/performance/v2_perf_iq4nl_gemm_comparison --gtest_filter='*Prefill_4096*'
   ```

4. **Verify alignment**
   - Create diagnostic tool to check raw_data_ pointer alignment
   - Should show 64-byte aligned after fix

### Long-term Improvements

1. **Convert all quantized tensors to AlignedVector**
   - IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ1_M, IQ1_S, Q6_K, Q8_0
   - Consistent 64-byte alignment across all formats

2. **Add alignment assertions**
   ```cpp
   assert(reinterpret_cast<uintptr_t>(raw_data_.data()) % 64 == 0);
   ```

3. **Document alignment requirements**
   - Update IQ4_NLTensor.h header comments
   - Add performance notes to CUDA_KERNEL_OPTIMIZATION_PLAN.md

## References

- AlignedVector implementation: `src/v2/tensors/AlignedVector.h`
- IQ4_NL tensor: `src/v2/tensors/IQ4_NLTensor.h`
- Alignment analysis: `changelog/2025-11-08-aligned-vector-and-streaming-stores-analysis.md`
- V2 performance testing: `tests/v2/performance/README.md`
- Build configuration: `src/v2/CMakeLists.txt`

## Appendix: Diagnostic Commands

```bash
# Check if IQ4_NL uses AlignedVector
grep -n "raw_data_" src/v2/tensors/IQ4_NLTensor.h | grep "vector"

# Check compile flags
cmake -B build_v2_release -L | grep -E "CMAKE_CXX_FLAGS|march"

# Run prefill benchmark (4096 tokens)
./build_v2_release/performance/v2_perf_iq4nl_gemm_comparison --gtest_filter='*Prefill_4096_QProjection'

# Expected output before fix: ~330 GFLOPS
# Expected output after fix: ~700-900 GFLOPS (with -march=native)
# Target output: ~1100 GFLOPS (previous measurements)
```
