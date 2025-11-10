# CPUAttentionT Critical Bug Fixes - Quick Reference

**Date**: November 8, 2025  
**Impact**: 🔴 **CRITICAL** - Segfaults, heap corruption, buffer overflows  
**Status**: ✅ **FIXED** - All tests passing (17/17)

## Bug Summary Table

| Bug # | Type | Severity | Impact | Fixed Line |
|-------|------|----------|--------|------------|
| 1 | Output pointer type mismatch | 🔴 CRITICAL | Segfault, buffer overflow | `CPUAttentionT.h:110` |
| 2 | KV broadcast buffer size | 🔴 CRITICAL | Heap corruption, GQA crash | `CPUAttentionT.h:217` |
| 3 | memset size calculation | 🟡 MEDIUM | Uninitialized memory | `CPUAttentionT.h:292` |

## Bug #1: Output Pointer Type Mismatch

**Root Cause**: GEMM outputs FP32, but template cast to ElementType*

### The Problem

```cpp
// WRONG CODE (before fix):
bool compute(float *output, ...) {
    ElementType *output_typed = reinterpret_cast<ElementType *>(output);  // ❌
    //                                                          ^
    // For BF16: float* (4 bytes/element) → uint16_t* (2 bytes/element)
    
    return compute_typed(..., output_typed, ...);
}

bool compute_typed(..., ElementType *output, ...) {
    ElementType *output_h = output + h * head_dim;  // ❌ WRONG ARITHMETIC
    //                               ^
    // For BF16: offset = h * head_dim * 2 bytes (WRONG!)
    // Should be: offset = h * head_dim * 4 bytes (float)
    
    gemm->multiply_activations_strided(
        ...,
        reinterpret_cast<float*>(output_h),  // Converts back, but TOO LATE
        ...);                                 // Already pointing to wrong memory!
}
```

### The Fix

```cpp
// CORRECT CODE (after fix):
bool compute(float *output, ...) {
    // DO NOT CAST OUTPUT! Keep as float*
    return compute_typed(..., output, ...);  // ✅ Pass float* directly
}

bool compute_typed(..., float *output, ...) {  // ✅ Changed signature
    float *output_h = output + h * head_dim;  // ✅ CORRECT ARITHMETIC
    //                         ^
    // Offset = h * head_dim * 4 bytes (correct for float)
    
    gemm->multiply_activations_strided(
        ...,
        output_h,  // ✅ Already float*, no cast needed
        ...);
}
```

### Why This Matters

**GEMM kernel signature** (the truth):
```cpp
bool multiply_activations_strided(
    const float *A,  // Input (BF16 data interpreted as float)
    const float *B,  // Input (BF16 data interpreted as float)
    float *C,        // 🔴 OUTPUT IS ALWAYS FLOAT*, NOT ELEMENTTYPE*!
    ...);
```

**Example calculation** (h=1, head_dim=4):
- **Wrong**: `uint16_t* output_h = output + 1*4 = output + 8 bytes`
- **Correct**: `float* output_h = output + 1*4 = output + 16 bytes`
- **Result**: 8-byte misalignment → **buffer overflow!**

---

## Bug #2: KV Broadcast Buffer Size Mismatch

**Root Cause**: `broadcast_kv_heads()` writes FP32, but buffer allocated as ElementType

### The Problem

```cpp
// WRONG CODE (before fix):
std::vector<ElementType> K_broadcast;  // ❌ uint16_t for BF16
K_broadcast.resize(seq_len * n_heads * head_dim);  // 2 bytes/element
//                                                  ^
// Example: seq_len=2, n_heads=4, head_dim=4
// Allocated: 2 * 4 * 4 * 2 = 64 bytes

broadcast_kv(K, K_broadcast.data(), ...);
// ↓
attention_utils::broadcast_kv_heads(
    reinterpret_cast<const float*>(K),
    reinterpret_cast<float*>(K_broadcast.data()),  // ❌ uint16_t* → float*
    ...);
// Writes: 2 * 4 * 4 * 4 = 128 bytes  ← OVERFLOW!
```

### The Fix

```cpp
// CORRECT CODE (after fix):
std::vector<float> K_broadcast;  // ✅ Always float
K_broadcast.resize(seq_len * n_heads * head_dim);  // 4 bytes/element
//                                                  ^
// Allocated: 2 * 4 * 4 * 4 = 128 bytes (correct!)

attention_utils::broadcast_kv_heads(
    reinterpret_cast<const float*>(K),
    K_broadcast.data(),  // ✅ float* → float*, no size mismatch
    ...);
// Writes: 2 * 4 * 4 * 4 = 128 bytes  ← FITS PERFECTLY
```

### Impact

**Only triggers in GQA/MQA** (when `n_heads != n_kv_heads`):
- MHA (`n_heads == n_kv_heads`): ✅ No broadcast, no bug
- GQA (`n_heads > n_kv_heads`): ❌ **Heap corruption!**
- MQA (`n_kv_heads == 1`): ❌ **Heap corruption!**

**Error message**: `corrupted double-linked list` (glibc heap check)

---

## Bug #3: Output memset Size Calculation

**Root Cause**: Used `sizeof(ElementType)` instead of `sizeof(float)` for output buffer

### The Problem

```cpp
// WRONG CODE (before fix):
std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(ElementType));
//                                                      ^^^^^^^^^^^^^^^^^
// For BF16: sizeof(uint16_t) = 2 bytes
// Clears: seq_len * n_heads * head_dim * 2 bytes (ONLY HALF THE BUFFER!)
```

### The Fix

```cpp
// CORRECT CODE (after fix):
std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));
//                                                      ^^^^^^^^^^^^^
// Always: sizeof(float) = 4 bytes
// Clears: seq_len * n_heads * head_dim * 4 bytes (ENTIRE BUFFER)
```

### Impact

**Less severe than Bugs #1-2**, but still bad:
- ✅ **No crash** (writes stay within allocated buffer)
- ❌ **Uninitialized memory** (second half of buffer not cleared)
- ❌ **Non-deterministic output** (depends on previous memory contents)
- ❌ **Flaky tests** (works sometimes, fails sometimes)

---

## Architectural Principle: FP32 Output Everywhere

### The Golden Rule

> **GEMM kernels ALWAYS output FP32, regardless of input precision.**

### Why?

1. **Numerical stability**: Attention softmax needs high precision
2. **Logit precision**: Final logits for sampling need FP32 accuracy
3. **Intermediate workspaces**: Scores, weights, context all FP32
4. **Simplicity**: Single output type (no conversion needed)

### Where Precision Matters

| Component | Precision | Reason |
|-----------|-----------|--------|
| **Q/K/V inputs** | BF16/FP16 | 🟢 Memory savings (2x reduction) |
| **GEMM computation** | FP32 | 🟡 Internal (hardware accelerated) |
| **Scores workspace** | FP32 | 🔴 Required (softmax needs precision) |
| **Attention weights** | FP32 | 🔴 Required (softmax output) |
| **Context output** | FP32 | 🔴 Required (GEMM output) |
| **Final output** | FP32 | 🔴 Required (logits for sampling) |

### Memory Layout Truth Table

| Buffer | Declared Type | Actual Type | Element Size |
|--------|---------------|-------------|--------------|
| Input Q/K/V | `const ElementType*` | BF16/FP16/FP32 | 2/2/4 bytes |
| Scores workspace | `float*` | FP32 | 4 bytes |
| Weights workspace | `float*` | FP32 | 4 bytes |
| KV broadcast buffer | `float*` | FP32 | 4 bytes |
| Output buffer | `float*` | FP32 | 4 bytes |

## Testing Strategy: How We Found These Bugs

### Bug Detection Matrix

| Test Case | Bug #1 | Bug #2 | Bug #3 |
|-----------|--------|--------|--------|
| FP32.BasicAttention | ✅ Pass | ✅ Pass | ✅ Pass |
| FP32.MultiHeadAttention | ✅ Pass | ✅ Pass | ✅ Pass |
| FP32.GroupedQueryAttention | ✅ Pass | ✅ Pass | ✅ Pass |
| BF16.BasicAttention | ❌ Crash | ✅ Pass | 🟡 Flaky |
| BF16.MultiHeadAttention | ❌ Crash | ✅ Pass | 🟡 Flaky |
| **BF16.GroupedQueryAttention** | ❌ **Crash** | ❌ **Crash** | 🟡 **Flaky** |

**Critical finding**: Only `GroupedQueryAttention` test with BF16 exposed all bugs!

### Why GQA Test is Critical

```cpp
// GroupedQueryAttention test dimensions:
const int seq_len = 2;
const int n_heads = 4;      // Query heads
const int n_kv_heads = 2;   // KV heads (TRIGGERS BROADCAST)
const int head_dim = 4;

// This triggers:
// 1. Bug #1: Output buffer writes (all BF16 tests)
// 2. Bug #2: KV broadcast (only when n_heads != n_kv_heads) ← UNIQUE!
// 3. Bug #3: memset size (all BF16 tests)
```

**Lesson**: Always test edge cases (GQA, MQA, large batches)!

## Quick Fix Checklist

When adding new precision types to CPUAttentionT:

- [ ] ✅ **Output** stays `float*` (never cast to ElementType*)
- [ ] ✅ **Broadcast buffers** are `std::vector<float>` (not ElementType)
- [ ] ✅ **memset** uses `sizeof(float)` (not sizeof(ElementType))
- [ ] ✅ **Workspaces** are `FP32Tensor` (not TensorType)
- [ ] ✅ **GEMM output pointers** are `float*` (no reinterpret_cast needed)
- [ ] ✅ **Test GQA** (`n_heads != n_kv_heads`) for broadcast bugs

## Related Files

**Fixed Files**:
- `src/v2/kernels/cpu/CPUAttentionT.h` - Template implementation
- `tests/v2/unit/Test__CPUAttentionT.cpp` - BF16/FP16/INT32 tests

**Documentation**:
- `changelog/2025-11-08-phase3b-bf16-testing-complete.md` - Full session log
- `changelog/2025-11-08-cpuattentiont-bug-fixes.md` - This file

## Verification Commands

```bash
# Build tests
cmake --build build_v2 --target v2_test_cpu_attention_t --parallel

# Run all tests (should show 17/17 passing)
./build_v2/tests/v2/v2_test_cpu_attention_t

# Run just BF16 GQA test (critical for Bug #2)
./build_v2/tests/v2/v2_test_cpu_attention_t --gtest_filter="CPUAttentionT_BF16.GroupedQueryAttention"

# Expected output:
# [==========] Running 17 tests from 4 test suites.
# [  PASSED  ] 17 tests.
```

## References

- **Issue Tracker**: N/A (discovered during development)
- **Original Implementation**: Phase 3 (CPUAttentionT template)
- **Bug Discovery Session**: Phase 3b (BF16 testing)
- **Commits**: See git log for 2025-11-08
