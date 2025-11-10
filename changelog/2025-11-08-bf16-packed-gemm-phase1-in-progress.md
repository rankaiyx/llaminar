# BF16 Packed GEMM Implementation - Session Summary

**Date**: November 8, 2025  
**Status**: In Progress - Phase 1 Implementation  
**Branch**: master

## Objectives

Implement BF16×BF16→FP32 GEMM using the auto-tuned micro-kernel infrastructure:
- **Phase 1**: BF16→FP32 conversion during packing + FP32 micro-kernels (in progress)
- **Phase 2** (future): Native BF16 SIMD micro-kernels with AVX512_BF16 instructions
- **Phase 3** (future): Hybrid SimdTraits approach

## Work Completed

### 1. Implementation Files Created

**src/v2/kernels/cpu/BF16PackedGemm.h** (62 lines)
- Factory function declaration: `createBF16PackedGemm(A_tensor, B_tensor)`
- Documentation of Phase 1 strategy
- Usage example

**src/v2/kernels/cpu/BF16PackedGemm.cpp** (467 lines) - ⚠️ NEEDS REVISION
- `BF16MicroKernelAdapter` class (wraps micro-kernels with BF16 packing)
- `AutoTunedBF16PackedGemm` wrapper (delegates to auto-tuner)
- `registerBF16MicroKernelVariants()` factory (creates 1,225 adapters)
- **Issue**: Use-after-free bug due to storing raw B_tensor pointer in globally registered variants

**tests/v2/unit/Test__BF16PackedGemm.cpp** (535 lines)
- Comprehensive test suite with 10 test cases:
  - BasicCorrectness, LargeMatrices, NonSquareMatrices
  - SimdAlignment, SingleToken, MediumBatch
  - NumericalStability, AlphaBetaScaling, InvalidInputs
  - BenchmarkVsReference
- Uses reference FP32 GEMM for validation
- Includes performance benchmarking

### 2. Build System Integration

**src/v2/CMakeLists.txt**:
- Added `kernels/cpu/BF16PackedGemm.cpp` to llaminar2_core sources

**tests/v2/CMakeLists.txt**:
- Added `v2_test_bf16_packed_gemm` executable with proper labels
- Labels: "V2;Unit;Kernels;GEMM;BF16;AutoTuning;MicroKernels;CacheBlocking;CPU;AVX2;AVX512"
- MPI_PROCS: 1 (CPU-only kernel, single rank)

### 3. Build Results

✅ **Successful compilation** of llaminar2_core library  
✅ **Successful linking** of test executable  
❌ **Tests fail with two critical issues**:

## Issues Discovered

### Issue 1: Use-After-Free Bug 🔴 CRITICAL

**Error**:
```
==393064==ERROR: AddressSanitizer: heap-use-after-free on address 0x50e00015f57c
READ of size 1 at 0x50e00015f57c thread T0
#0 llaminar2::BF16Tensor::bf16_data() const
#1 llaminar2::kernels::gemm::BF16MicroKernelAdapter::multiply()
```

**Root Cause**:
- `BF16MicroKernelAdapter` stores raw `const BF16Tensor* B_tensor_` pointer
- Variants are registered globally once in static initialization
- Test creates `std::unique_ptr<BF16Tensor>` locally, then destroys it
- Auto-tuner tries to use globally registered variant → accesses freed memory

**Why This Happened**:
- Misunderstood the auto-tuner architecture
- Tried to use global variant registration with per-call tensor state
- Quantized GEMM uses `IBlockDecoder*` which is a reference to persistent tensor internals
- `BF16Tensor*` is a raw pointer to an object with independent lifetime

**Architecture Mismatch**:
```cpp
// What I tried (WRONG):
registerBF16MicroKernelVariants(B_tensor);  // Stores B_tensor* globally
// ... later in different test ...
B_tensor goes out of scope → destroyed
auto-tuner calls multiply() → use-after-free!

// What quantized GEMM does (CORRECT):
IBlockDecoder* decoder = tensor->createDecoder();  // Decoder persists
registerMicroKernelVariants(decoder);  // Safe to store globally
```

### Issue 2: Incorrect Results 🔴 CRITICAL

**Error**:
```
[Test__BF16PackedGemm.cpp:116] max_abs_diff=8.19244, max_rel_diff=4465.83, mismatches=1001/1024
Expected: (max_rel_diff) <= (rel_tol), actual: 4465.82568 vs 0.05
```

**Analysis**:
- Relative error of 4465× is catastrophically high
- 1001 out of 1024 elements (98%) mismatch
- Indicates fundamental logic bugs, not just floating-point precision

**Likely Causes** (to investigate after fixing use-after-free):
1. Incorrect BF16→FP32 conversion indexing
2. Wrong dimension handling in `packAPanel_BF16toFP32` / `packBPanel_BF16toFP32`
3. Transpose logic errors
4. Row-major vs column-major layout confusion

## Revised Implementation Plan

### Simplified Approach (Phase 1 Standalone)

**Problem**: Auto-tuner integration is too complex for initial proof-of-concept

**Solution**: Create standalone BF16 packed GEMM with hardcoded micro-kernel

**Architecture**:
```cpp
class StandaloneBF16PackedGemm : public ITensorGemm {
    const BF16Tensor* B_tensor_;  // Stored as member (lives as long as kernel)
    
    // Hardcoded good micro-kernel (e.g., 8×6 AVX512)
    static constexpr int MR = 8;
    static constexpr int NR = 6;
    
    bool multiply(...) {
        // 1. Get micro-kernel from registry
        auto bundle = MicroKernelRegistry::instance().get_kernel("simd::AVX512Tag", 8, 6, 4, 2);
        
        // 2. Cache blocking loop (MC × NC tiles)
        #pragma omp parallel
        for (int jc = 0; jc < n; jc += NC) {
            // Pack B panel with BF16→FP32 conversion
            packBPanel_BF16toFP32(...);
            
            for (int ic = 0; ic < m; ic += MC) {
                // Pack A panel with BF16→FP32 conversion
                packAPanel_BF16toFP32(...);
                
                // Call FP32 micro-kernel
                bundle.micro_kernel(...);
            }
        }
    }
};
```

**Benefits**:
- ✅ No global registration complexity
- ✅ B_tensor stored as member (no use-after-free)
- ✅ Simpler to debug packing logic bugs
- ✅ Still uses proven micro-kernel infrastructure
- ✅ Can add auto-tuning later (Phase 1.5)

**Drawbacks**:
- ❌ Not auto-tuned (but 8×6 AVX512 is a good general-purpose choice)
- ❌ Doesn't adapt to different problem sizes
- ✅ Acceptable for Phase 1 proof-of-concept

## Next Steps

1. **Revise BF16PackedGemm.cpp** to standalone implementation:
   - Remove `AutoTunedBF16PackedGemm`, `BF16MicroKernelAdapter`, `registerBF16MicroKernelVariants()`
   - Create `StandaloneBF16PackedGemm` with hardcoded 8×6 AVX512 micro-kernel
   - Store `B_tensor_` as member variable
   - Implement packing functions inline

2. **Debug packing logic**:
   - Verify BF16→FP32 conversion is correct
   - Check dimension indexing (row-major layout)
   - Validate transpose_B handling
   - Add logging for packed buffer inspection

3. **Run tests**:
   - Fix use-after-free (should be eliminated by standalone approach)
   - Fix incorrect results (debug packing logic)
   - Validate all 10 test cases pass

4. **(Optional) Add auto-tuning**:
   - Once standalone version works, can add auto-tuner integration
   - Use per-kernel auto-tuner instance (not global registration)
   - Benchmark different micro-kernel variants on first call

## Performance Baseline

**Expected Performance** (from Phase 1 analysis):
- Prefill (512 tokens): ~30-40 GFLOPS (should match or exceed existing BF16GemmKernel)
- Decode (single token): ~5-10 GFLOPS
- Memory bandwidth: 2× savings vs FP32 (BF16 is half the size)

**Test System**:
- CPU: 2 sockets, 56 physical cores (112 with HT)
- ISA: AVX-512 capable

## Related Files

**Core Implementation**:
- `src/v2/kernels/cpu/BF16PackedGemm.{h,cpp}` (needs revision)
- `src/v2/kernels/cpu/GemmMicroKernelRegistry.h` (unchanged)
- `src/v2/tensors/SIMDHelpers.h` (unchanged - provides BF16↔FP32 conversion)

**Testing**:
- `tests/v2/unit/Test__BF16PackedGemm.cpp` (comprehensive, should work once bugs fixed)

**Reference**:
- `src/v2/kernels/cpu/BF16GemmKernel.cpp` (existing OneDNN/OpenBLAS approach)
- `src/v2/kernels/cpu/QuantizedGemm.cpp` (auto-tuner integration pattern)
- `src/v2/kernels/cpu/TiledGemmSoftmax.h` (successful micro-kernel usage example)

## Lessons Learned

1. **Don't prematurely optimize**: Auto-tuner integration is complex - get standalone version working first
2. **Understand object lifetimes**: Raw pointers to stack objects cause use-after-free
3. **Follow existing patterns**: TiledGemmSoftmax directly uses micro-kernels without global registration
4. **Test incrementally**: Should have tested standalone version before adding auto-tuner layer
5. **SIMD padding is critical**: Learned from TiledGemmSoftmax fix - always add padding for buffer overruns

## Architectural Insights

**MicroKernelTemplate Infrastructure** (deeply understood now):
- 1,225 pre-compiled variants: 2 ISAs × 7 MR × 8 NR × 5 UNROLL_K × 5 PREFETCH_DIST
- Micro-kernels are **FP32-generic** (not quantized-specific)
- Can be used for:
  - ✅ Quantized GEMM (IQ4_NL, Q6_K) - via IBlockDecoder adapter
  - ✅ Attention scores (Q @ K^T) - via TiledGemmSoftmax
  - 🔄 BF16 GEMM (BF16×BF16→FP32) - via BF16 packing adapter (in progress)
  - ✅ FP32 GEMM - directly (FP32GemmKernel)

**Auto-Tuner Architecture**:
- Global variant registration works well for stateless kernels
- Doesn't work well for per-tensor state (like BF16Tensor pointers)
- Quantized GEMM works because IBlockDecoder is a persistent reference, not a temporary object
- Solution: Use per-kernel auto-tuner instance OR don't use auto-tuner for Phase 1

## Code Statistics

- **Lines Added**: ~1,064 lines (467 impl + 535 tests + 62 header)
- **Files Created**: 3 (header, impl, tests)
- **Files Modified**: 2 (CMakeLists)
- **Build Time**: ~10 seconds (incremental)
- **Test Status**: 0/10 passing (use-after-free + logic bugs)

## Session Duration

**Total Time**: ~2.5 hours
- Architecture analysis: ~30 min
- Implementation: ~1 hour
- Testing/debugging: ~1 hour

## Continuation Point

Ready to implement simplified standalone version. Next session should:
1. Replace auto-tuner-based implementation with standalone approach (~200 lines simpler)
2. Debug BF16 packing logic bugs
3. Validate all tests pass
4. Benchmark performance vs existing BF16GemmKernel
5. Optionally add auto-tuning if time permits

**Files to modify**:
- `src/v2/kernels/cpu/BF16PackedGemm.cpp` (complete rewrite - simpler)
- `tests/v2/unit/Test__BF16PackedGemm.cpp` (minimal changes - may need relaxed tolerances)
