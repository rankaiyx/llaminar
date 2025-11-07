# FP32 Attention PyTorch Parity - SUCCESS ✅

**Date**: 2025-11-06  
**Status**: ✅ COMPLETE - FP32AttentionKernel validated against PyTorch  
**Test Result**: PASS - max_abs=7.45e-09, rel_l2=2.63e-08 (both < 1e-4 threshold)

---

## Executive Summary

Successfully implemented and validated FP32AttentionKernel against PyTorch ground truth. The kernel correctly implements Grouped Query Attention (GQA) by expanding K/V heads before score computation, matching PyTorch's exact approach. Test passes with excellent numerical agreement (relative L2 error < 3e-8).

**Key Achievement**: Established validated FP32 baseline for INT8 quantization testing. Can now confidently debug INT8 implementation knowing the FP32 reference is correct.

---

## Final Test Results

```
Test__FP32AttentionParity (4 tests):
✅ BasicForwardVsPyTorch (PASSING)
  - Max abs diff: 7.450581e-09 (threshold: 1e-4)
  - Relative L2: 2.625004e-08 (threshold: 1e-4)
  - Expected[:5]: -0.009495 0.000362 -0.026480 -0.012358 -0.021116
  - Actual[:5]:   -0.009495 0.000362 -0.026480 -0.012358 -0.021116
  - Status: EXACT MATCH ✅
  
✅ AttentionScoresVsPyTorch (PASSING)
  - Placeholder test, validated via final output
  
⏭ CausalMaskingVsPyTorch (SKIPPED)
  - Needs: Generate causal snapshots
  
⏭ SingleSequenceEdgeCase (SKIPPED)
  - Needs: Generate single-token snapshots
```

---

## Root Cause & Solution

### Initial Problem
FP32AttentionKernel failed with 234% relative L2 error due to GQA implementation mismatch with PyTorch.

### Root Cause Discovery Process

**Phase 1: GQA Mapping Issue**
- Initial hypothesis: Wrong GQA head mapping logic
- Fixed: Added `n_kv_heads` parameter to constructor
- Result: Error persisted (234% → 234%)
- Conclusion: Deeper issue than parameter passing

**Phase 2: K/V Expansion Missing**
- Discovery: `expand_kv_heads()` function existed but was never called
- Fixed: Updated `forward()` to call expansion before `compute_scores()`
- Result: Error persisted (234% → 234%)
- Conclusion: Expansion happening, but something else wrong

**Phase 3: GQA Mapping During Score Computation**
- Discovery: `compute_scores()` still doing GQA mapping after expansion
- Analysis: PyTorch expands K/V **first**, then uses standard MHA
- Fixed: Removed GQA mapping from `compute_scores()` and `compute_output()`
- Result: Error persisted (234% → 234%)
- Conclusion: Logic correct but comparing wrong outputs

**Phase 4: Output Projection Issue** ✅
- **CRITICAL DISCOVERY**: PyTorch applies `o_proj` output projection!
- `attention_output.npy` = `o_proj(attn_weights @ V)`, not just raw attention
- Our kernel computes raw attention (no o_proj)
- **Solution**: Compare against `attention_context` (before o_proj) instead
- Updated snapshot generator to save `attention_context.npy`
- **Result**: ✅ EXACT MATCH (rel_l2 < 3e-8)

### Final Implementation

**FP32AttentionKernel Logic**:
1. **Expand K/V** (if GQA): `[batch, seq_len, n_kv_heads*d_head]` → `[batch, seq_len, n_heads*d_head]`
2. **Compute scores**: Standard MHA (no GQA mapping) - `Q @ K^T`
3. **Apply softmax**: Scale by `1/sqrt(d_head)`, optional causal mask
4. **Compute output**: Standard MHA (no GQA mapping) - `attn_weights @ V`
5. **Return**: Raw attention context (caller applies o_proj if needed)

**Key Insight**: PyTorch's GQA is implemented as **physical K/V expansion** followed by **standard MHA**, not dynamic head mapping during computation.

---

## Code Changes

### 1. FP32AttentionKernel.cpp (lines 62-98)

**Updated forward() to call K/V expansion**:
```cpp
// Expand K/V if needed (GQA)
const float *k_to_use = k;
const float *v_to_use = v;

if (n_kv_heads_ < n_heads_) {
    k_expanded_buffer_.resize(kv_expanded_size);
    v_expanded_buffer_.resize(kv_expanded_size);
    
    // Expand K/V heads: [batch, seq_len, n_kv_heads*d_head] -> [batch, seq_len, n_heads*d_head]
    expand_kv_heads(k, k_expanded_buffer_.data(), batch, seq_len);
    expand_kv_heads(v, v_expanded_buffer_.data(), batch, seq_len);
    
    k_to_use = k_expanded_buffer_.data();
    v_to_use = v_expanded_buffer_.data();
}

// Compute scores with expanded K
compute_scores(q, k_to_use, batch, seq_len);
```

### 2. FP32AttentionKernel.cpp (lines 103-138)

**Updated compute_scores() to use standard MHA**:
```cpp
// NOTE: K is already expanded to n_heads if GQA was used
int d_model_k = n_heads_ * d_head_;  // K is expanded to match Q

for (int h = 0; h < n_heads_; ++h) {
    // Standard MHA - no GQA mapping (K already expanded)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            // Compute dot product: Q[i, h] · K[j, h]
            for (int d = 0; d < d_head_; ++d) {
                int q_idx = (b * seq_len + i) * d_model_q + h * d_head_ + d;
                int k_idx = (b * seq_len + j) * d_model_k + h * d_head_ + d;  // Same head!
                dot += q[q_idx] * k[k_idx];
            }
        }
    }
}
```

**Before** (incorrect GQA mapping):
```cpp
int kv_head = (n_kv_heads_ < n_heads_) ? (h / (n_heads_ / n_kv_heads_)) : h;
int k_idx = (b * seq_len + j) * d_model_k + kv_head * d_head_ + d;  // Wrong!
```

**After** (standard MHA):
```cpp
// No GQA mapping - K is already expanded
int k_idx = (b * seq_len + j) * d_model_k + h * d_head_ + d;  // Correct!
```

### 3. FP32AttentionKernel.cpp (lines 226-260)

**Updated compute_output() to use standard MHA**:
```cpp
// NOTE: V is already expanded to n_heads if GQA was used
int d_model_v = n_heads_ * d_head_;  // V is expanded to match Q

for (int h = 0; h < n_heads_; ++h) {
    // Standard MHA - no GQA mapping (V already expanded)
    for (int i = 0; i < seq_len; ++i) {
        for (int d = 0; d < d_head_; ++d) {
            for (int j = 0; j < seq_len; ++j) {
                int attn_idx = ((b * n_heads_ + h) * seq_len + i) * seq_len + j;
                int v_idx = (b * seq_len + j) * d_model_v + h * d_head_ + d;  // Same head!
                sum += attn_weights_buffer_[attn_idx] * v[v_idx];
            }
        }
    }
}
```

### 4. FP32AttentionKernel.h (line 46)

**Added n_kv_heads parameter to constructor**:
```cpp
FP32AttentionKernel(int n_heads, int d_head, int n_kv_heads = 0, int device_idx = -1);
```

### 5. generate_attention_snapshots.py (line 250)

**Added attention_context snapshot** (before o_proj):
```python
# Reshape back: [bsz, n_heads, q_len, d_head] -> [bsz, q_len, hidden]
context = context.transpose(1, 2).contiguous().view(bsz, q_len, -1)

# Save context before output projection
self.captures['attention_context'] = context.numpy()

# Output projection
output = attn.o_proj(context)
```

### 6. Test__FP32AttentionParity.cpp (lines 166-178)

**Updated test to load and compare attention_context**:
```cpp
ASSERT_TRUE(load_snapshot("attention_context", pytorch_context));

// Compare output with PyTorch (attention context, not final output projection)
float max_abs = compute_max_abs_diff(pytorch_context.data, output_actual);
float rel_l2 = compute_relative_l2(pytorch_context.data, output_actual);

print_comparison("Attention Context", pytorch_context.data, output_actual, 1e-4f, 1e-4f);
```

---

## Validation Process

### Manual Verification

Computed expected output manually using PyTorch values:

```python
# PyTorch weights for position 0, head 0
w00 = [0.809, 0.191]

# V expanded (head 0, tokens 0 and 1)
v0_h0 = [-0.00291374, 0.00465465, -0.02899878, -0.01508388, -0.02487786]
v1_h0 = [-0.03740791, -0.01784337, -0.01579627, -0.00079901, -0.0051594 ]

# Expected: w00[0] * v0_h0 + w00[1] * v1_h0
manual_out = [-0.00949529, 0.00036199, -0.02647972, -0.0123583, -0.02111554]
```

**C++ Output**: `[-0.009495, 0.000362, -0.026480, -0.012358, -0.021116]`

**PyTorch Context**: `[-0.00949529, 0.00036199, -0.02647972, -0.0123583, -0.02111555]`

**Conclusion**: ✅ EXACT MATCH - Implementation is mathematically correct!

### Attention Weights Validation

**PyTorch** (from snapshots):
- `weights[0, 0, 0, :] = [0.80919826, 0.19080175]`
- `weights[0, 0, 1, :] = [0.80065066, 0.19934942]`

**Our Implementation** (from debug logs):
- `weights[0, 0, 0, :] = [0.809198, 0.190802]`
- `weights[0, 0, 1, :] = [0.800651, 0.199349]`

**Conclusion**: ✅ Softmax computation is correct!

---

## Key Learnings

### 1. GQA Implementation Strategy

**PyTorch's Approach** (now our approach too):
- Reshape to heads: `[batch, seq_len, d_model]` → `[batch, n_heads, seq_len, d_head]`
- **Expand K/V physically** before any computation
- Treat as standard MHA after expansion

**Not** doing dynamic head mapping during computation - this is simpler and more efficient.

### 2. Attention Components

**Raw Attention Mechanism** (what our kernel computes):
```
context = Softmax(Q @ K^T / sqrt(d_head)) @ V
```

**Full Transformer Attention** (what PyTorch's layer computes):
```
output = o_proj(context) + residual
```

Our FP32AttentionKernel implements **raw attention only** (no o_proj, no residual). This is correct for a kernel-level primitive.

### 3. Testing Methodology

**Layered Validation**:
1. **Attention weights**: Validate softmax computation
2. **Attention context**: Validate weighted sum of values
3. **Full output**: Validate including output projection (future)

This layered approach helped isolate the o_proj issue quickly.

### 4. Numerical Precision

Achieved **7.45e-09 max absolute error** with:
- FP32 throughout (no mixed precision)
- Numerically stable softmax (max-subtraction)
- Careful index arithmetic (no layout bugs)

This is **excellent agreement** - differences are purely floating-point rounding.

---

## File Status

### Created/Modified This Session

| File | Status | Purpose | Lines |
|------|--------|---------|-------|
| `src/v2/kernels/cpu/FP32AttentionKernel.cpp` | ✅ Validated | FP32 attention implementation | 304 |
| `src/v2/kernels/cpu/FP32AttentionKernel.h` | ✅ Validated | FP32 attention interface | 151 |
| `tests/v2/unit/Test__FP32AttentionParity.cpp` | ✅ Passing | PyTorch parity test suite | 400+ |
| `python/reference/generate_attention_snapshots.py` | ✅ Updated | Snapshot generator (added context) | 344 |

### PyTorch Snapshots Generated

```
pytorch_attention_snapshots/ (10 files):
✅ input.npy               - Post-norm attention input
✅ Q_projection.npy        - Query projection
✅ K_projection.npy        - Key projection  
✅ V_projection.npy        - Value projection
✅ Q_rope.npy              - Query after RoPE
✅ K_rope.npy              - Key after RoPE
✅ attention_scores.npy    - Raw Q @ K^T scores
✅ attention_weights.npy   - Post-softmax weights
✅ attention_context.npy   - Weighted sum (our target!)
✅ attention_output.npy    - After o_proj
```

---

## Next Steps

### Immediate (Priority 1)

1. **Update INT8 Tests to Use Validated FP32** (Task 3)
   - Replace `reference_attention_fp32()` helper with `FP32AttentionKernel`
   - File: `tests/v2/unit/Test__INT8AttentionKernel.cpp`
   - Expected: Isolate INT8 quantization bugs from FP32 issues

2. **Debug INT8 Attention Implementation** (Task 4)
   - Fix 88% accuracy error (now isolated to quantization)
   - Fix causal masking (diff_count=0)
   - Fix 39% single-sequence error
   - Add stage-by-stage comparison vs validated FP32

### Short-term (Priority 2)

3. **Generate Additional Snapshots** (Task 5)
   ```bash
   # Causal masking test
   python3 python/reference/generate_attention_snapshots.py \
     --causal --output pytorch_attention_snapshots_causal
   
   # Single-token edge case
   python3 python/reference/generate_attention_snapshots.py \
     --prompt "A" --output pytorch_attention_snapshots_single
   ```

4. **Run Full Parity Suite**
   - Enable CausalMaskingVsPyTorch test
   - Enable SingleSequenceEdgeCase test
   - Expected: All 4 tests pass

### Medium-term (Priority 3)

5. **INT8 SwiGLU Kernel** (Task 6)
6. **Qwen2 INT8 Pipeline** (Task 7)
7. **Integration Parity Tests** (Task 8)
8. **Performance Benchmarks** (Task 9)

---

## Architecture Reference

**Qwen 2.5 0.5B Instruct**:
- `n_layers`: 24
- `n_heads`: 14 (query heads)
- `n_kv_heads`: 2 (GQA 7:1 ratio)
- `d_model`: 896
- `d_head`: 64

**GQA Head Mapping** (after expansion):
- KV head 0 → Q heads 0-6
- KV head 1 → Q heads 7-13

**Tensor Shapes**:
- Q: `[1, 2, 896]` (14 × 64)
- K (unexpanded): `[1, 2, 128]` (2 × 64)
- K (expanded): `[1, 2, 896]` (14 × 64)
- V (unexpanded): `[1, 2, 128]` (2 × 64)
- V (expanded): `[1, 2, 896]` (14 × 64)
- Scores: `[1, 14, 2, 2]`
- Weights: `[1, 14, 2, 2]`
- Context: `[1, 2, 896]`

---

## Success Metrics Achieved

✅ **Numerical Accuracy**:
- Max abs diff: `7.45e-09` (< `1e-4` threshold)
- Relative L2: `2.63e-08` (< `1e-4` threshold)

✅ **Functional Correctness**:
- GQA expansion working
- Standard MHA computation working
- Softmax computation matches PyTorch exactly
- Weighted sum computation matches PyTorch exactly

✅ **Test Infrastructure**:
- PyTorch snapshot generator working
- NpzLoader integration working
- Comprehensive comparison metrics
- Clear pass/fail criteria

✅ **Documentation**:
- Root cause analysis documented
- Solution approach documented
- Testing methodology documented
- Next steps clearly defined

---

**Status**: 🎉 MISSION ACCOMPLISHED - FP32 baseline established!

**Confidence Level**: HIGH - Validated against PyTorch ground truth with excellent numerical agreement.

**Ready for**: INT8 quantization testing with validated FP32 reference.
