# GEMM Requantization Support - Design Document

**Date**: November 8, 2025  
**Status**: 🔧 **PROPOSED** - Design phase  
**Goal**: Add optional BF16/FP16 output requantization to GEMM kernels for 50% memory savings

## Problem Statement

**Current Architecture**: GEMM kernels **always output FP32**, requiring FP32 workspaces:
- Scores workspace: `seq_len × n_heads × seq_len × 4 bytes`
- Weights workspace: `seq_len × n_heads × seq_len × 4 bytes`
- Output buffer: `seq_len × n_heads × head_dim × 4 bytes`

**Example (Qwen 2.5 0.5B, 512 tokens)**:
- Per-layer overhead: ~30 MB FP32 workspaces
- 24 layers: **720 MB total**
- With BF16 requantization: **360 MB** (50% reduction!)

**Critical Question**: Can we requantize GEMM output to BF16/FP16 **without breaking numerical stability**?

## Numerical Stability Analysis

### Where Precision Matters

| Operation | Current Precision | Can Requantize? | Reason |
|-----------|-------------------|-----------------|--------|
| **Q @ K^T scores** | FP32 | ⚠️ **Maybe** | Input to softmax (needs analysis) |
| **Softmax(scores)** | FP32 | ❌ **NO** | Requires exp() precision |
| **Attention weights** | FP32 | ✅ **YES** | Already normalized [0,1] |
| **weights @ V output** | FP32 | ✅ **YES** | Final activations (tolerant) |

### Critical Path: Softmax Precision

**Current flow**:
```
Q @ K^T → FP32 scores → softmax(FP32) → FP32 weights → weights @ V → FP32 output
```

**With requantization (Option A - Conservative)**:
```
Q @ K^T → FP32 scores → softmax(FP32) → BF16 weights → weights @ V → BF16 output
         ↑ Keep FP32      ↑ Must be FP32   ↑ Can requantize  ↑ Can requantize
```

**With requantization (Option B - Aggressive)**:
```
Q @ K^T → BF16 scores → convert to FP32 → softmax(FP32) → BF16 weights → weights @ V → BF16 output
         ↑ Requantize   ↑ On-the-fly      ↑ Must be FP32   ↑ Requantize  ↑ Requantize
```

**Recommendation**: Start with **Option A** (keep scores FP32, requantize weights/output)

## Proposed API Design

### Option 1: Output Precision Parameter (RECOMMENDED)

Add optional `output_dtype` parameter to GEMM interface:

```cpp
enum class GemmOutputPrecision {
    FP32,    // Default (current behavior)
    BF16,    // Requantize to BF16 after computation
    FP16,    // Requantize to FP16 after computation
    MATCH_INPUT  // Match input precision (BF16→BF16, FP16→FP16, FP32→FP32)
};

class ITensorGemm : public ITensorKernel {
public:
    virtual bool multiply_activations_strided(
        const float *A, const float *B, float *C,  // C buffer size depends on output_dtype!
        int m, int n, int k,
        int lda, int ldb, int ldc,
        bool transpose_B = true,
        float alpha = 1.0f, float beta = 0.0f,
        const MPIContext *mpi_ctx = nullptr,
        int device_idx = -1,
        GemmOutputPrecision output_dtype = GemmOutputPrecision::FP32) = 0;
        //                                  ^^ NEW PARAMETER
};
```

**Buffer size implications**:
```cpp
// Caller MUST allocate C buffer with correct size:
size_t C_size_bytes;
switch (output_dtype) {
    case GemmOutputPrecision::FP32: C_size_bytes = m * ldc * sizeof(float); break;
    case GemmOutputPrecision::BF16: C_size_bytes = m * ldc * sizeof(uint16_t); break;
    case GemmOutputPrecision::FP16: C_size_bytes = m * ldc * sizeof(uint16_t); break;
}
```

### Option 2: Separate Requantization Method (Alternative)

Keep GEMM output FP32, add explicit requantization step:

```cpp
class ITensorGemm : public ITensorKernel {
public:
    // Existing GEMM (always outputs FP32)
    virtual bool multiply_activations_strided(..., float *C, ...) = 0;
    
    // NEW: Fused GEMM + requantization
    virtual bool multiply_activations_strided_requantized(
        const float *A, const float *B, uint16_t *C_bf16,  // Output is uint16_t*
        int m, int n, int k,
        int lda, int ldb, int ldc,
        bool transpose_B = true,
        float alpha = 1.0f, float beta = 0.0f,
        GemmOutputPrecision output_dtype = GemmOutputPrecision::BF16,
        ...) = 0;
};
```

**Pros**: Explicit, no ambiguity about buffer size  
**Cons**: API duplication, more complex

**Recommendation**: Use **Option 1** (single method with precision parameter)

## Implementation Plan

### Phase 1: Add GemmOutputPrecision Enum and Parameter

**Files to modify**:
1. `src/v2/tensors/TensorKernels.h` - Add enum and update interface
2. `src/v2/kernels/cpu/BF16GemmKernel.h` - Update signatures
3. `src/v2/kernels/cpu/FP32GemmKernel.h` - Update signatures
4. `src/v2/kernels/cpu/FP16GemmKernel.h` - Update signatures (when implemented)

**Changes**:
```cpp
// TensorKernels.h (add before ITensorGemm):
enum class GemmOutputPrecision {
    FP32,         ///< Output FP32 (current default)
    BF16,         ///< Requantize output to BF16
    FP16,         ///< Requantize output to FP16
    MATCH_INPUT   ///< Match input tensor precision
};

// ITensorGemm interface:
virtual bool multiply_activations_strided(
    const float *A, const float *B, float *C,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    bool transpose_B = true,
    float alpha = 1.0f, float beta = 0.0f,
    const MPIContext *mpi_ctx = nullptr,
    int device_idx = -1,
    GemmOutputPrecision output_dtype = GemmOutputPrecision::FP32) = 0;
```

**Estimated time**: 1 hour

---

### Phase 2: Implement BF16 Requantization in BF16GemmKernel

**File**: `src/v2/kernels/cpu/BF16GemmKernel.cpp`

**Algorithm**:
```cpp
bool BF16GemmKernel::multiply_activations_strided(
    const float *A, const float *B, float *C,
    int m, int n, int k, int lda, int ldb, int ldc,
    bool transpose_B, float alpha, float beta,
    const MPIContext *mpi_ctx, int device_idx,
    GemmOutputPrecision output_dtype)
{
    if (output_dtype == GemmOutputPrecision::FP32) {
        // Current path: output FP32 directly
        return multiply_activations_strided_fp32_output(...);
    }
    
    // NEW PATH: Compute in FP32, then requantize
    
    // 1. Allocate temporary FP32 buffer (or use thread-local pool)
    std::vector<float> C_fp32(m * ldc);
    
    // 2. Perform GEMM in FP32
    bool success = multiply_activations_strided_fp32_output(
        A, B, C_fp32.data(), m, n, k, lda, ldb, ldc,
        transpose_B, alpha, beta, mpi_ctx, device_idx);
    if (!success) return false;
    
    // 3. Requantize FP32 → BF16/FP16
    uint16_t *C_quantized = reinterpret_cast<uint16_t*>(C);
    
    if (output_dtype == GemmOutputPrecision::BF16) {
        llaminar2::simd::fp32_to_bf16(C_fp32.data(), C_quantized, m * ldc);
    } else if (output_dtype == GemmOutputPrecision::FP16) {
        llaminar2::simd::fp32_to_fp16(C_fp32.data(), C_quantized, m * ldc);
    }
    
    return true;
}
```

**Optimization opportunities**:
1. **Thread-local buffer pool** (avoid allocations)
2. **Fused requantization** (during GEMM output write, not separate pass)
3. **SIMD vectorization** (AVX2/AVX512 for FP32→BF16/FP16)

**Estimated time**: 2-3 hours

---

### Phase 3: Update CpuAttentionKernelT to Use Requantization

**File**: `src/v2/kernels/cpu/CpuAttentionKernelT.h`

**Changes**:

1. **Template parameter for workspace precision**:
```cpp
template <typename TensorType, 
          GemmOutputPrecision WorkspacePrecision = GemmOutputPrecision::FP32>
class CpuAttentionKernelT : public ITensorAttention {
    // ...
};
```

2. **Workspace allocation based on precision**:
```cpp
// Instead of always FP32Tensor:
if constexpr (WorkspacePrecision == GemmOutputPrecision::FP32) {
    owned_scores = std::make_shared<FP32Tensor>(...);
} else if constexpr (WorkspacePrecision == GemmOutputPrecision::BF16) {
    owned_scores = std::make_shared<BF16Tensor>(...);
}
```

3. **GEMM calls with requantization**:
```cpp
// Q @ K^T with optional requantization
gemm->multiply_activations_strided(
    reinterpret_cast<const float*>(Q_h), K_h, scores_h,
    seq_len, seq_len, head_dim,
    lda, ldb, ldc,
    true, scale, 0.0f,
    nullptr, -1,
    WorkspacePrecision);  // ← Use template parameter
```

4. **Softmax handling**:
```cpp
// CRITICAL: Softmax MUST operate on FP32!
if constexpr (WorkspacePrecision != GemmOutputPrecision::FP32) {
    // Convert BF16/FP16 → FP32 for softmax
    std::vector<float> scores_fp32(seq_len * seq_len);
    convert_to_fp32(scores_h, scores_fp32.data(), seq_len * seq_len);
    
    primitives::softmax_row_major_fp32(scores_fp32.data(), seq_len, seq_len, causal, 1.0f, true);
    
    // Convert back to BF16/FP16
    convert_from_fp32(scores_fp32.data(), scores_h, seq_len * seq_len);
} else {
    // Direct softmax on FP32 workspace
    primitives::softmax_row_major_fp32(scores_h, seq_len, seq_len, causal, 1.0f, true);
}
```

**Estimated time**: 3-4 hours

---

### Phase 4: Add Tests for Requantization Path

**Files**:
1. `tests/v2/unit/Test__BF16GemmKernel.cpp` - Test requantization correctness
2. `tests/v2/unit/Test__CpuAttentionKernelT.cpp` - Test attention with BF16 workspaces

**Test cases**:
```cpp
TEST(BF16GemmKernel, RequantizeToBF16) {
    // Test: GEMM with output_dtype=BF16 produces same results as
    // GEMM(FP32) + manual fp32_to_bf16(), within BF16 tolerance
}

TEST(CpuAttentionKernelT_BF16, BF16WorkspacesBasicAttention) {
    // Test: CpuAttentionKernelT with BF16 workspaces produces similar results
    // to FP32 workspaces, within BF16 tolerance (5e-3)
}
```

**Estimated time**: 2-3 hours

---

### Phase 5: Performance Benchmarking

**Metrics to measure**:
1. **Memory usage**: FP32 vs BF16 workspaces (expect 50% reduction)
2. **Throughput**: Tokens/sec (expect small overhead from requantization)
3. **Latency**: Prefill time (expect 5-10% overhead)
4. **Numerical accuracy**: Compare outputs vs FP32 baseline

**Benchmark script**:
```bash
# Compare FP32 vs BF16 workspace performance
./benchmark_attention_workspaces.sh \
  --model qwen2.5-0.5b-instruct-q8_0.gguf \
  --seq-lens 128,256,512,1024,2048 \
  --workspace-precisions fp32,bf16,fp16
```

**Expected results**:
- **Memory**: 50% reduction (BF16 vs FP32)
- **Speed**: 5-10% slower (conversion overhead)
- **Accuracy**: <1% relative difference in outputs

**Estimated time**: 2-3 hours

---

## Total Effort Estimate

| Phase | Time | Status |
|-------|------|--------|
| 1. API design (enum + signatures) | 1 hour | Not started |
| 2. BF16GemmKernel requantization | 2-3 hours | Not started |
| 3. CpuAttentionKernelT integration | 3-4 hours | Not started |
| 4. Testing | 2-3 hours | Not started |
| 5. Benchmarking | 2-3 hours | Not started |
| **Total** | **10-14 hours** | |

## Risks and Mitigation

### Risk 1: Softmax Numerical Instability

**Risk**: BF16 scores may cause softmax overflow/underflow  
**Mitigation**: 
- Keep scores in FP32 (Option A)
- If using BF16 scores, convert to FP32 before softmax
- Add validation tests comparing against FP32 baseline

### Risk 2: Performance Regression

**Risk**: FP32→BF16 conversion overhead may negate memory savings  
**Mitigation**:
- Use SIMD-optimized conversion (AVX2/AVX512)
- Benchmark before/after to quantify overhead
- Make requantization **optional** (default to FP32 for safety)

### Risk 3: API Complexity

**Risk**: Buffer size ambiguity (FP32 vs BF16) may cause bugs  
**Mitigation**:
- Clear documentation of buffer size requirements
- Runtime assertions for buffer size validation
- Comprehensive tests covering both paths

## Alternative Approach: In-Place Requantization

**Idea**: Requantize output **during GEMM write**, not as separate pass

**Benefit**: No temporary FP32 buffer needed, better cache locality

**Implementation**:
```cpp
// In GEMM kernel, instead of:
C[i] = result_fp32;

// Do:
if (output_dtype == GemmOutputPrecision::BF16) {
    uint16_t *C_bf16 = reinterpret_cast<uint16_t*>(C);
    C_bf16[i] = fp32_to_bf16_scalar(result_fp32);
} else {
    C[i] = result_fp32;
}
```

**Complexity**: Requires changes to low-level GEMM loops (high risk)  
**Recommendation**: Start with **separate requantization pass** (Phase 2), optimize later

## Recommended Approach

### Short-term (Next 4-6 hours):

1. ✅ **Implement Option A (Conservative)**:
   - Keep Q@K^T scores in **FP32** (critical for softmax)
   - Requantize attention weights to **BF16** (after softmax)
   - Requantize output to **BF16**
   
2. ✅ **API**: Add `GemmOutputPrecision` enum + parameter to GEMM interface

3. ✅ **Implementation**: BF16GemmKernel with separate requantization pass

4. ✅ **Testing**: Validate numerical accuracy within BF16 tolerance

### Long-term (Future optimization):

1. ⏸ **Option B (Aggressive)**: Requantize Q@K^T scores to BF16 if numerical tests pass

2. ⏸ **Fused requantization**: Integrate into GEMM output write for zero-copy

3. ⏸ **GPU support**: Extend to CUDAGemmKernel/ROCmGemmKernel

## Decision: Proceed?

**Questions for user**:

1. **Should we pause Phase 3b/4 and implement requantization now?**
   - Pro: 50% memory savings is significant
   - Con: Delays template interface work (Phase 4-7)

2. **Which option to implement first?**
   - Option A (Conservative): Keep scores FP32, requantize weights/output
   - Option B (Aggressive): Requantize all intermediate tensors

3. **Target precision for workspaces?**
   - BF16 only (simpler, 50% savings)
   - BF16 + FP16 (more complex, broader support)

4. **Priority: Memory vs Speed?**
   - Memory-first: Accept 5-10% slowdown for 50% memory reduction
   - Speed-first: Keep FP32 workspaces, optimize later

**My recommendation**: 
- ✅ Implement **Option A** with **BF16 requantization**
- ✅ Start with **separate requantization pass** (simpler, lower risk)
- ✅ Make it **optional** (default to FP32 for backward compatibility)
- ✅ Complete in **~6 hours** before returning to Phase 4

Let me know your preference and I'll proceed!
