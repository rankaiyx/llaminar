# INT8 GEMM Migration Blocked - Architecture Mismatch

**Date**: 2025-11-09  
**Status**: ❌ **Blocked** - Reverted changes, INT8GemmKernel restored needed  
**Impact**: INT8Tensor cannot use auto-tuner pattern like other tensors

## Problem Statement

Attempted to migrate INT8Tensor to use auto-tuned GEMM (INT8PackedGemm) following the pattern established for FP32/FP16/BF16 tensors. However, INT8PackedGemm is **architecturally incompatible** with the current INT8Tensor implementation.

## Root Causes

### 1. Different Data Layouts
- **INT8GemmKernel** (removed): Works with per-channel quantized INT8 + FP32 scales
- **INT8PackedGemm**: Expects packed INT8 data with different layout assumptions
- **Mismatch**: INT8Tensor's `int8_data()` format doesn't match INT8PackedGemm's expectations

### 2. OneDNN vs Auto-Tuner Philosophy
- **INT8GemmKernel**: Wraps OneDNN's highly optimized s8s8s32 matmul primitives
- **INT8PackedGemm**: Hand-rolled AVX512-VNNI micro-kernels for auto-tuning
- **Reality**: OneDNN INT8 GEMM is **production-grade** and likely faster than custom kernels

### 3. Quantization Integration
- **INT8GemmKernel**: Handles FP32→INT8 quantization internally (per-row for activations)
- **INT8PackedGemm**: Expects pre-quantized data, no built-in quantization
- **Workflow mismatch**: Tests/production code expect `multiply(float *A, ...)` not `multiply(int8_t *A, ...)`

## What Was Attempted

###  Changes Made (Reverted)
1. ✅ Removed INT8GemmKernel.{h,cpp} (2 files, ~600 lines)
2. ✅ Updated INT8Tensor::createGemm() to use `createINT8PackedGemm()`
3. ✅ Modified INT8PackedGemm to accept weight tensor reference
4. ✅ Updated all tests to use `createGemm()` pattern
5. ❌ **Result**: 100% error rate - completely wrong results

### Test Failure
```
Expected: (rel_error) < (0.02f), actual: 1 vs 0.02
Relative error should be <2% for INT8 quantization
[Test__INT8GemmKernel] BasicMatrixMultiply: relative error = 100%
```

## Why INT8 is Different

### Other Tensors (FP32/FP16/BF16)
- **Direct computation**: FP32×FP32→FP32, BF16×BF16→FP32
- **Auto-tuner works**: Select optimal tile/unroll for throughput
- **Simple interface**: `multiply(const float *A, float *C, ...)`

### INT8 Tensor
- **Quantization required**: FP32→INT8 (lossy), then INT8×INT8→INT32→FP32
- **OneDNN optimized**: Hardware-accelerated VNNI instructions via library
- **Complex interface**: Per-channel scales, multiple output formats (INT32/FP32)
- **Production path**: OneDNN s8s8s32 is industry-standard, battle-tested

## Recommended Solution

**Keep INT8GemmKernel as a specialized implementation**

### Rationale
1. **OneDNN is optimal**: No point in hand-rolling INT8 kernels when OneDNN exists
2. **Different workflow**: INT8 inference has unique quantization requirements
3. **Proven correctness**: INT8GemmKernel has comprehensive tests and works
4. **Not worth complexity**: Forcing INT8 into auto-tuner adds no value

### Alternative: Make INT8PackedGemm Production-Ready
If auto-tuner pattern is mandatory:
1. Implement proper INT8Tensor→packed format conversion
2. Add FP32→INT8 quantization to INT8PackedGemm
3. Benchmark against OneDNN (likely slower)
4. **Estimated effort**: 2-3 days of development + testing

## Conclusion

**INT8 should remain a special case** with its own kernel implementation (INT8GemmKernel). The auto-tuner pattern works well for FP32/FP16/BF16 where we control the full pipeline, but INT8 benefits more from leveraging OneDNN's mature, hardware-optimized primitives.

**Recommendation**: Restore INT8GemmKernel and document it as the canonical INT8 GEMM approach.
