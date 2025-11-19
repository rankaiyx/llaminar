# V2 Attention Architecture Finalized

**Date:** November 7, 2025  
**Session:** V2 Architecture Development - Attention Refactoring  
**Status:** Design Complete, Ready for Implementation

---

## Executive Summary

Finalized architectural design for V2 Attention refactoring with **result-tensor pattern** and **native precision support**. All critical decisions made, ready to proceed with implementation.

### Key Architectural Decisions

1. ✅ **Native Softmax Implementation** (FP32/BF16/FP16/INT32 with SIMD)
2. ✅ **Tensor Class Disambiguation** (FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor)
3. ✅ **Full INT32 Attention Support** (for INT8 quantized pipelines)
4. ✅ **Result-Tensor Pattern** (result.computeOp(inputs...) instead of factories)
5. ✅ **Template-Based Implementation** (CpuAttentionKernelT<TensorType> eliminates code duplication)

---

## Design Evolution

### Problem Identification

**Issue 1:** Dummy tensor creation pattern in CPUAttention.cpp (lines 125-133, 282-284, 362-364)
```cpp
FP32Tensor dummy({1, 1});  // ❌ Heap allocation just to get kernel!
auto gemm = dummy.createGemm();
```

**Issue 2:** Hardcoded FP32 assumptions throughout
- float pointers everywhere
- FP32Tensor workspace allocation
- `sizeof(float)` in memset calls
- No support for BF16/FP16/INT32

**Issue 3:** Proposed precision-specific methods would duplicate 95% of logic
```cpp
bool compute_fp32(...);  // ❌ 95% identical
bool compute_bf16(...);  // ❌ 95% identical
bool compute_fp16(...);
```

### Solution Evolution

**Iteration 1:** Precision-specific methods
- ❌ Code duplication (rejected)

**Iteration 2:** Template class + kernel factories
- ✅ Template eliminates duplication
- ❌ New factory layer inconsistent with V2 design
- Partially accepted

**Iteration 3 (FINAL):** Template class + result-tensor pattern
- ✅ Template eliminates duplication
- ✅ Result-tensor pattern (no factories needed)
- ✅ Consistent with existing V2 tensor-centric design
- ✅ **APPROVED**

---

## Final Architecture

### 1. Result-Tensor Pattern

**Principle:** ALL operations (unary, binary, ternary) are tensor-centric. Result tensor owns the computation.

```cpp
// Unary ops: Modify tensor in-place
tensor.applyRMSNorm(gamma, ...);        // tensor modified
Q.applyRoPE(K, positions, ...);         // Q and K modified

// Binary ops: Result computes itself from inputs
result.computeGemm(A, B, ...);          // result = A × B (writes to self)
scores.computeGemm(Q, K, ...);          // scores = Q × K^T (writes to self)

// Ternary ops: Result computes itself from inputs  
output.computeAttention(Q, K, V, ...);  // output = attention(Q, K, V) (writes to self)
```

**Advantages:**
- ✅ No dummy tensors
- ✅ No explicit factory calls in pipeline code
- ✅ Clean mental model: `result.computeOp(inputs...)`
- ✅ Result tensor owns kernel creation internally (via createGemm/createAttention)
- ✅ Precision handled via virtual dispatch (no branching)

### 2. Template-Based Kernels

```cpp
/**
 * @brief CPU Attention kernel (templated by tensor type)
 * 
 * @tparam TensorType Tensor class (FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor)
 */
template<typename TensorType>
class CpuAttentionKernelT : public ITensorAttention
{
public:
    using ElementType = typename TensorType::value_type;  // float, uint16_t, int32_t
    
    bool compute(
        const ElementType *Q, const ElementType *K, const ElementType *V,
        ElementType *output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal = false, ...);
    
private:
    // All helper methods templated (broadcast_kv, compute_scores, compute_context)
    void broadcast_kv(const ElementType *K_in, ElementType *K_out, ...);
    void compute_scores(const ElementType *Q, const ElementType *K, ...);
    void compute_context(const ElementType *scores, const ElementType *V, ...);
};

// Explicit instantiations (4 types)
template class CpuAttentionKernelT<FP32Tensor>;   // ElementType = float
template class CpuAttentionKernelT<BF16Tensor>;   // ElementType = uint16_t
template class CpuAttentionKernelT<FP16Tensor>;   // ElementType = uint16_t (disambiguated!)
template class CpuAttentionKernelT<INT32Tensor>;  // ElementType = int32_t
```

### 3. ActivationTraits Pattern

**Purpose:** Provide tensor-type-specific implementations for precision-dependent operations.

```cpp
template<typename TensorType>
struct ActivationTraits;

template<>
struct ActivationTraits<FP32Tensor>
{
    using ElementType = float;
    using TensorClass = FP32Tensor;
    
    // Native FP32 softmax (AVX512/AVX2)
    static void apply_softmax(float *scores, int rows, int cols, ...);
    
    // FP32 GEMM kernel
    static std::unique_ptr<ITensorGemm> create_activation_gemm();
    
    // FP32 workspace allocation
    static std::shared_ptr<TensorBase> allocate_workspace(const std::vector<size_t> &shape);
};

// Similar specializations for BF16Tensor, FP16Tensor, INT32Tensor
```

---

## Critical Decisions Finalized

### Decision 1: Native Softmax (No Conversion)

**Question:** Should BF16/FP16 softmax convert to FP32 internally or use native implementation?

**Decision:** ✅ **Native implementation** for all precisions

**Rationale:**
- Performance-first approach (V2 design philosophy)
- Eliminates conversion overhead (2× memory bandwidth)
- Matches RMSNorm/RoPE pattern (native precision primitives)
- Enables SIMD optimizations per precision

**Implementation:**
- `softmax_row_fp32_avx512/avx2/scalar` - FP32 with AVX512/AVX2
- `softmax_row_bf16_avx512/avx2/scalar` - **NEW** - Native BF16 with AVX512_BF16
- `softmax_row_fp16_avx2/scalar` - **NEW** - Native FP16 with F16C
- `softmax_row_int32` - INT32→FP32 conversion (no native INT32 softmax)

**Status:** Softmax primitives are **BLOCKING dependency** for Phase 3

---

### Decision 2: Tensor Class Disambiguation (Not Primitive Types)

**Question:** How to distinguish FP16 from BF16 (both uint16_t)?

**Decision:** ✅ **Use tensor classes** (FP32Tensor, BF16Tensor, FP16Tensor, INT32Tensor)

**Rationale:**
- Clear disambiguation (no uint16_t ambiguity)
- Type-safe API at tensor level
- Matches V2 design pattern (tensor-centric)
- Simplifies template instantiation

**Before (ambiguous):**
```cpp
template<typename ActType>
struct ActivationTraits;

template<> struct ActivationTraits<uint16_t> {  // ❌ BF16 or FP16?
    // ...
};
```

**After (clear):**
```cpp
template<typename TensorType>
struct ActivationTraits;

template<> struct ActivationTraits<FP32Tensor> { ... };   // ✅ FP32
template<> struct ActivationTraits<BF16Tensor> { ... };   // ✅ BF16
template<> struct ActivationTraits<FP16Tensor> { ... };   // ✅ FP16
template<> struct ActivationTraits<INT32Tensor> { ... };  // ✅ INT32
```

---

### Decision 3: Full INT32 Attention Support

**Question:** Should INT32 attention be supported or explicitly unsupported?

**Decision:** ✅ **Full support** for INT8 quantized pipelines

**Rationale:**
- INT8 pipelines use INT32 accumulation for matrix multiplication
- Attention operates on INT32 activations (Q, K, V)
- Critical for low-precision inference (mobile, edge devices)
- Conversion approach is acceptable (softmax not compute-bound)

**Implementation:**
```cpp
template<>
struct ActivationTraits<INT32Tensor>
{
    static void apply_softmax(int32_t *scores, int rows, int cols, ...)
    {
        constexpr float INT32_TO_FP32_SCALE = 1.0f / 65536.0f;
        constexpr float FP32_TO_INT32_SCALE = 65536.0f;
        
        std::vector<float> fp32_row(cols);
        
        for (int i = 0; i < rows; ++i) {
            // Convert INT32→FP32
            for (int j = 0; j < cols; ++j) {
                fp32_row[j] = static_cast<float>(scores[i * cols + j]) * INT32_TO_FP32_SCALE;
            }
            
            // Apply FP32 softmax
            softmax_row_fp32_avx512(fp32_row.data(), cols, ...);
            
            // Convert FP32→INT32
            for (int j = 0; j < cols; ++j) {
                scores[i * cols + j] = static_cast<int32_t>(fp32_row[j] * FP32_TO_INT32_SCALE);
            }
        }
    }
};
```

---

## IActivationTensor Interface Updates

### New Methods (Result-Tensor Pattern)

```cpp
class IActivationTensor {
    // ===== Binary Operations (result computes itself from inputs) =====
    
    /**
     * @brief Compute self = A × B (activation × activation GEMM)
     */
    virtual bool computeGemm(
        const float *A,
        const float *B,
        int m, int n, int k,
        bool transpose_A = false,
        bool transpose_B = false,
        float alpha = 1.0f,
        float beta = 0.0f,
        const MPIContext *mpi_ctx = nullptr,
        int device_idx = -1) = 0;
    
    /**
     * @brief Compute self = quantized_weight × activation (quantized GEMM)
     */
    virtual bool computeGemmQuantized(
        const IBlockDecoder *weight_decoder,
        const float *activation,
        int m, int n, int k,
        bool transpose_weight = false,
        float alpha = 1.0f,
        float beta = 0.0f,
        const MPIContext *mpi_ctx = nullptr,
        int device_idx = -1) = 0;
    
    // ===== Ternary Operations (result computes itself from inputs) =====
    
    /**
     * @brief Compute self = attention(Q, K, V)
     */
    virtual bool computeAttention(
        const float *Q,
        const float *K,
        const float *V,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal = false,
        int window_size = -1,
        TensorBase *workspace_scores = nullptr,
        TensorBase *workspace_buffer = nullptr,
        TensorBase *workspace_context = nullptr,
        TensorBase *workspace_mask = nullptr,
        const MPIContext *mpi_ctx = nullptr,
        int device_idx = -1) = 0;
    
    // ===== Kernel Creation (used internally by compute methods) =====
    
    /**
     * @brief Create GEMM kernel matching this tensor's precision
     * 
     * Used internally by computeGemm() and computeGemmQuantized().
     * Should be protected/private in final implementation.
     */
    virtual std::unique_ptr<ITensorGemm> createGemm() = 0;
    
    /**
     * @brief Create Attention kernel matching this tensor's precision
     * 
     * Used internally by computeAttention().
     * Should be protected/private in final implementation.
     */
    virtual std::unique_ptr<ITensorAttention> createAttention() = 0;
};
```

### Existing Methods (Unchanged)

```cpp
class IActivationTensor {
    // ===== Unary Operations (modify self in-place) =====
    
    virtual bool applyRMSNorm(...) = 0;  // ✅ Keep
    virtual bool applyRoPE(...) = 0;     // ✅ Keep
    virtual bool applySoftmax(...) = 0;  // ✅ Keep
    virtual bool applySwiGLU(...) = 0;   // ✅ Keep
};
```

---

## Pipeline Usage Example

**Before (dummy tensor pattern):**
```cpp
// ❌ OLD BROKEN PATTERN
FP32Tensor dummy({1, 1});  // Heap allocation!
auto attention_kernel = dummy.createAttention();
attention_kernel->compute(Q.data(), K.data(), V.data(), output.data(), ...);
```

**After (result-tensor pattern):**
```cpp
// ✅ NEW CLEAN PATTERN
bool Qwen2Pipeline::attention_block(int layer, TensorBase *input, TensorBase *output) {
    // Allocate result tensors (same precision as input)
    auto Q = allocateActivation({seq_len, n_heads * head_dim});
    auto K = allocateActivation({seq_len, n_kv_heads * head_dim});
    auto V = allocateActivation({seq_len, n_kv_heads * head_dim});
    auto attn_out = allocateActivation({seq_len, n_heads * head_dim});
    
    // Unary op: modify input in-place
    input->applyRMSNorm(weights_.norm_gamma[layer].data(), seq_len, d_model);
    
    // Binary ops: result tensors compute themselves
    Q->computeGemmQuantized(weights_.wq[layer]->as_decoder(), input->data(), ...);
    K->computeGemmQuantized(weights_.wk[layer]->as_decoder(), input->data(), ...);
    V->computeGemmQuantized(weights_.wv[layer]->as_decoder(), input->data(), ...);
    
    // Unary op: modify Q and K in-place
    Q->applyRoPE(K->mutable_data(), position_ids, ...);
    
    // Ternary op: attn_out computes itself from Q, K, V
    attn_out->computeAttention(Q->data(), K->data(), V->data(), ...);
    
    // Binary op: output computes itself
    output->computeGemmQuantized(weights_.wo[layer]->as_decoder(), attn_out->data(), ...);
    
    return true;
}
```

**Key improvements:**
- ✅ No dummy tensors
- ✅ No factories exposed to pipeline code
- ✅ Clean mental model: `result.computeOp(inputs...)`
- ✅ No precision branching (virtual dispatch handles it)

---

## Implementation Roadmap

### Phase 1: Native Softmax Primitives (BLOCKING)

**Files to create/modify:**
- `src/v2/kernels/cpu/primitives/SoftmaxPrimitives.h`

**Functions to implement:**
- `softmax_row_fp32_avx512/avx2/scalar` (verify existing)
- `softmax_row_bf16_avx512/avx2/scalar` (**NEW**)
- `softmax_row_fp16_avx2/scalar` (**NEW**)

**INT32 softmax:** Handled in ActivationTraits (FP32 conversion)

**Priority:** **CRITICAL** - Blocking for all subsequent phases

---

### Phase 2: ActivationTraits Implementation

**Files to create:**
- `src/v2/kernels/ActivationTraits.h`

**Specializations:**
- `ActivationTraits<FP32Tensor>`
- `ActivationTraits<BF16Tensor>`
- `ActivationTraits<FP16Tensor>`
- `ActivationTraits<INT32Tensor>`

**Each provides:**
- `ElementType` typedef
- `apply_softmax()` static method
- `create_activation_gemm()` static method
- `allocate_workspace()` static method

---

### Phase 3: Refactor CPUAttention → CpuAttentionKernelT<TensorType>

**Files to modify:**
- `src/v2/kernels/cpu/CPUAttention.h`
- `src/v2/kernels/cpu/CPUAttention.cpp`

**Changes:**
1. Rename `CPUAttention` → `CpuAttentionKernelT<TensorType>`
2. Replace `float *` with `ElementType *`
3. Use `ActivationTraits<TensorType>` for precision-specific ops
4. Template all helper methods
5. Remove ALL dummy tensor creation
6. Add explicit instantiations:
   ```cpp
   template class CpuAttentionKernelT<FP32Tensor>;
   template class CpuAttentionKernelT<BF16Tensor>;
   template class CpuAttentionKernelT<FP16Tensor>;
   template class CpuAttentionKernelT<INT32Tensor>;
   ```

---

### Phase 4: Update IActivationTensor Interface

**Files to modify:**
- `src/v2/tensors/Tensors.h`

**Changes:**
1. Add `computeGemm(A, B, ...)`
2. Add `computeGemmQuantized(weight, activation, ...)`
3. Add `computeAttention(Q, K, V, ...)`
4. Keep `createGemm()` and `createAttention()` (mark as protected in future)
5. Keep existing unary operations

---

### Phase 5: Implement Compute Methods in Tensor Classes

**Files to modify:**
- `src/v2/tensors/FP32Tensor.cpp`
- `src/v2/tensors/BF16Tensor.cpp`
- `src/v2/tensors/FP16Tensor.cpp`
- `src/v2/tensors/INT32Tensor.cpp`

**Pattern (example for FP32Tensor::computeAttention):**
```cpp
bool FP32Tensor::computeAttention(
    const float *Q, const float *K, const float *V,
    int seq_len, int n_heads, int n_kv_heads, int head_dim,
    bool causal, ...)
{
    // Create kernel matching this tensor's precision
    auto kernel = this->createAttention();  // Returns CpuAttentionKernelT<FP32Tensor>
    
    // Execute: writes to this->mutable_data()
    return kernel->compute(
        Q, K, V,
        this->mutable_data(),  // ← writes to SELF (FP32 output)
        seq_len, n_heads, n_kv_heads, head_dim,
        causal, ...);
}
```

---

### Phase 6: Update Pipeline Code

**Files to modify:**
- `src/v2/pipelines/Qwen2Pipeline.cpp`

**Changes:**
1. Replace dummy tensor pattern with result-tensor pattern
2. Use `result.computeGemm(A, B, ...)` instead of factory calls
3. Use `output.computeAttention(Q, K, V, ...)` for attention
4. Eliminate all dummy tensor allocations
5. Verify no precision branching needed

---

### Phase 7: Precision Parity Testing

**Files to create:**
- `tests/v2/unit/Test__CPUAttention.cpp`

**Test coverage:**
- FP32 vs BF16 vs FP16 vs INT32 numerical accuracy
- Relative L2 error tolerances:
  - FP32: 1e-5
  - BF16: 1e-3
  - FP16: 1e-2
  - INT32: 1e-1
- GQA/MQA correctness (n_heads ≠ n_kv_heads)
- Causal masking
- Native softmax correctness
- Performance benchmarks per precision

---

## Success Criteria

### Functional Requirements

- ✅ All 4 precisions (FP32/BF16/FP16/INT32) pass parity tests
- ✅ No dummy tensor creation anywhere
- ✅ No precision branching in pipeline code
- ✅ GQA/MQA works correctly
- ✅ Causal masking works correctly
- ✅ Native softmax for FP32/BF16/FP16
- ✅ INT32 softmax via FP32 conversion

### Code Quality Requirements

- ✅ Zero code duplication (single template implementation)
- ✅ Clean result-tensor pattern throughout
- ✅ Type-safe API (tensor classes eliminate ambiguity)
- ✅ Comprehensive test coverage (4 precisions × multiple scenarios)
- ✅ Performance validated per precision

### Performance Requirements

- ✅ FP32: Match existing performance (baseline)
- ✅ BF16: Within 10% of FP32 (AVX512_BF16 optimized)
- ✅ FP16: Within 15% of FP32 (F16C optimized)
- ✅ INT32: Within 50% of FP32 (conversion overhead acceptable)

---

## Risk Mitigation

### Risk 1: Softmax Primitive Implementation Complexity

**Mitigation:**
- Start with scalar implementations (correctness first)
- Add AVX2 optimizations second
- Add AVX512 optimizations last
- Comprehensive unit tests per variant

### Risk 2: Binary Size Increase (4× template instantiations)

**Mitigation:**
- Accept as acceptable tradeoff for zero runtime overhead
- Use link-time optimization (LTO) if available
- Profile binary size after implementation
- Optimize only if becomes problematic

### Risk 3: INT32 Softmax Conversion Overhead

**Mitigation:**
- Profile to confirm softmax is not compute-bound
- Conversion overhead likely acceptable (<5% total time)
- Future optimization: Native INT32 softmax if needed

---

## Documentation

**Design documents:**
- `V2_UNARY_BINARY_OPS_DESIGN.md` (1600+ lines) - Complete architectural design
- This changelog (current document) - Finalized decisions summary

**Next session deliverable:**
- `changelog/YYYY-MM-DD-v2-attention-phase1-softmax-primitives.md` - Softmax implementation
- `changelog/YYYY-MM-DD-v2-attention-phase3-template-implementation.md` - CpuAttentionKernelT refactor
- `changelog/YYYY-MM-DD-v2-attention-complete.md` - Final session summary

---

## Approvals

**Architecture approved by:** User (dbsanfte)  
**Date:** November 7, 2025  
**Status:** Ready for implementation  
**Next step:** Phase 1 - Native softmax primitives (BLOCKING)

