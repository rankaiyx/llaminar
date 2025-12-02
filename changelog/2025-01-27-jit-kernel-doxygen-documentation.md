# JIT Kernel Comprehensive Doxygen Documentation

**Date**: 2025-01-27  
**Author**: David Sanftenberg  
**Type**: Documentation Enhancement

## Summary

Added comprehensive Doxygen documentation to all JIT GEMM kernel files to make the complex AVX-512 VNNI assembly code understandable for junior developers.

## Files Modified

### 1. `src/v2/kernels/cpu/gemm_v4/QuantisedGemmJit_M1.h` (1194 lines)

**Single-row (M=1) JIT kernel for autoregressive decoding**

Documentation added:
- File-level header with algorithm overview, memory layout, and register allocation table
- `QuantisedPackedWeights` struct with detailed field explanations
- `QuantisedGemmParams` struct with memory offset table (A at 0, B at 8, ..., do_swiglu at 112)
- Class documentation with code buffer size explanation (128KB)
- `generate()` method with comprehensive comments including:
  - Prologue and register preservation
  - N-loop structure and cursor calculations
  - K-loop inner computation with VNNI instructions
  - Asymmetric quantization correction for IQ4_NL
  - Scale application (INT32 → FP32)
  - Bias and mask addition
  - Softmax first pass with polynomial exp() approximation
  - SwiGLU activation with sigmoid() computation
  - Epilogue and store operations

### 2. `src/v2/kernels/cpu/gemm_v4/QuantisedGemmJit_M2.h` (1249 lines)

**Two-row (M=2) JIT kernel for improved throughput**

Documentation added:
- File-level header explaining two-row design benefits
- Class documentation with code buffer size (256KB)
- ZMM register allocation table for dual-row processing:
  - zmm0-3: Row 0 accumulators
  - zmm4-7: Row 1 accumulators
  - zmm8-11: Row 0 temporaries
  - zmm12-15: Row 1 temporaries
  - zmm16-19: B block values
  - zmm20-21: A values (Row 0 and Row 1)
  - zmm22-31: Constants, scales, softmax/SwiGLU
- Prologue documentation including zero buffer allocation
- N-loop setup with mins pointer fallback explanation
- K-loop documentation with:
  - C accumulator initialization from existing values
  - Asymmetric correction for both rows
  - VNNI inner loop with dual-row processing
  - Scale application for both rows
- Post-processing sections:
  - Bias addition (same bias for both rows)
  - Attention mask (separate mask per row)
  - Softmax with detailed exp() polynomial explanation
  - SwiGLU with swish(x) = x / (1 + exp(-x)) derivation
- Epilogue with store operations and N-loop advancement

### 3. `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` (1564 lines)

**High-level kernel orchestration**

Documentation added:
- Comprehensive file-level header with:
  - Architecture overview diagram
  - Quantization format descriptions (Q8_1, packed weights)
  - Kernel selection table (M=1 vs M=2)
  - Cache-aware blocking strategy
  - Parallelization strategy
  - Fused operations list
  - Usage example code
  - Performance considerations
- Class-level documentation with key features and thread safety notes
- Constructor documentation (generic and legacy type-specific)
- `pack_single_block()` method with packed layout explanation
- `pack_weights_generic()` method with:
  - Packing process description
  - Superblock optimization explanation
  - Thread parallelization notes
- `supports_device()` method documentation
- `multiply()` method documentation
- `multiply_fused()` method with comprehensive documentation:
  - All parameter descriptions
  - Computation pipeline (C init → A quant → GEMM → post-ops)
  - Softmax notes (first pass only)
  - SwiGLU notes (swish gate application)
  - Phase-by-phase inline comments

## Key Documentation Patterns

### Register Allocation Tables
```cpp
/**
 * | Register | Purpose |
 * |----------|---------|
 * | zmm0-3   | Row 0 C accumulators |
 * | zmm4-7   | Row 1 C accumulators |
 * | ...      | ... |
 */
```

### Algorithm Pseudocode
```cpp
/**
 * exp(x) ≈ 2^n * P(f)
 *   where n = floor(x/ln2), f = frac(x/ln2)
 *   P(f) = c1 + f*(c2 + f*(c3 + f*(c4 + f*c5)))
 */
```

### Memory Offset Tables
```cpp
/**
 * | Offset | Field | Type |
 * |--------|-------|------|
 * | 0      | A     | ptr  |
 * | 8      | B     | ptr  |
 * | ...    | ...   | ...  |
 */
```

### IEEE754 Bit Manipulation
```cpp
/**
 * 2^n via IEEE754 exponent manipulation:
 *   float(n) → int → add 127 → shift to exponent bits
 *   2^n = (n + 127) << 23
 */
```

## Verification

- ✅ Build succeeds: `cmake --build build_v2 --target llaminar2_core`
- ✅ All 109 unit tests pass: `ctest --test-dir build_v2 -R "^V2_Unit_"`

## Documentation Statistics

| File | Lines Before | Lines After | Added |
|------|-------------|-------------|-------|
| QuantisedGemmJit_M1.h | ~700 | 1194 | ~494 |
| QuantisedGemmJit_M2.h | ~700 | 1249 | ~549 |
| QuantisedGemmKernel.h | ~1100 | 1564 | ~464 |
| **Total** | ~2500 | 4007 | **~1507** |

Over 1500 lines of documentation added to help junior developers understand:
1. AVX-512 VNNI instruction semantics
2. Quantization formats (Q8_1, IQ4_NL)
3. Cache-aware blocking strategies
4. Polynomial approximations for exp() and sigmoid()
5. IEEE754 bit manipulation for 2^n computation
6. OpenMP parallelization patterns
