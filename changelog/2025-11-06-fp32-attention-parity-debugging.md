# FP32 Attention PyTorch Parity - Debugging Session

**Date**: 2025-11-06  
**Status**: 🔄 IN PROGRESS - Debugging GQA implementation mismatch  
**Test Result**: FAIL - 234% relative L2 error (expected < 0.01%)

---

## Executive Summary

Created comprehensive FP32 attention parity test suite against PyTorch ground truth. Test infrastructure is complete and working, but reveals fundamental GQA (Grouped Query Attention) implementation mismatch between our FP32AttentionKernel and PyTorch's approach.

**Key Finding**: PyTorch **expands K/V heads first**, then computes standard MHA scores. Our implementation tries to do GQA mapping **during score computation**, which produces incorrect results.

---

## Test Infrastructure Status

### ✅ Completed Components

1. **Python Snapshot Generator** (`python/reference/generate_attention_snapshots.py`)
   - 400+ lines
   - Generates 9 PyTorch reference snapshots
   - Tested working: Captured Qwen 2.5 0.5B attention intermediate states
   - Output: `pytorch_attention_snapshots/` directory

2. **FP32AttentionKernel** (`src/v2/kernels/cpu/FP32AttentionKernel.{h,cpp}`)
   - 430 lines total
   - Compiles successfully
   - **But has GQA bug** - produces 234% error vs PyTorch

3. **Parity Test Suite** (`tests/v2/unit/Test__FP32AttentionParity.cpp`)
   - 400+ lines
   - 4 test cases (1 failing, 1 passing, 2 skipped)
   - Integrated into build system
   - Uses NpzLoader to load PyTorch snapshots
   - Comprehensive comparison metrics (max abs diff, relative L2)

4. **NpzLoader Infrastructure** (`tests/v2/pytorch_parity/NpzLoader.h`)
   - Existing from V1
   - Loads .npy files into NpyArray struct
   - Verified working

### Test Results Summary

```
Test__FP32AttentionParity (4 tests):
✗ BasicForwardVsPyTorch (FAILING)
  - Max abs diff: 0.196 (expected < 1e-4)
  - Relative L2: 2.34 (234% error, expected < 0.01%)
  - Expected[:5]: -0.008978 0.005964 -0.002661 0.003692 0.006627
  - Actual[:5]:   -0.009495 0.000362 -0.026480 -0.012358 -0.021116
  - Diagnosis: GQA implementation mismatch
  
✓ AttentionScoresVsPyTorch (PASSING)
  - Placeholder test, no actual validation
  
⏭ CausalMaskingVsPyTorch (SKIPPED)
  - Needs: Generate causal snapshots with --causal flag
  
⏭ SingleSequenceEdgeCase (SKIPPED)
  - Needs: Generate single-token snapshots
```

---

## Root Cause Analysis: GQA Implementation Mismatch

### PyTorch's Approach (Correct)

```python
# 1. Project to Q/K/V
Q = attn.q_proj(hidden_states)  # [bsz, q_len, 896] (14 heads × 64)
K = attn.k_proj(hidden_states)  # [bsz, q_len, 128] (2 kv_heads × 64)
V = attn.v_proj(hidden_states)  # [bsz, q_len, 128]

# 2. Reshape to heads
Q_heads = Q.view(bsz, q_len, 14, 64).transpose(1, 2)  # [bsz, 14, q_len, 64]
K_heads = K.view(bsz, q_len, 2, 64).transpose(1, 2)   # [bsz, 2, q_len, 64]
V_heads = V.view(bsz, q_len, 2, 64).transpose(1, 2)   # [bsz, 2, q_len, 64]

# 3. Apply RoPE (to head-reshaped tensors)
Q_rope, K_rope = apply_rope(Q_heads, K_heads, ...)

# 4. **CRITICAL: Expand K/V heads BEFORE score computation**
num_key_value_groups = 14 // 2 = 7
K_rope = repeat_kv(K_rope, 7)  # [bsz, 2, q_len, 64] -> [bsz, 14, q_len, 64]
V_heads = repeat_kv(V_heads, 7)  # [bsz, 2, q_len, 64] -> [bsz, 14, q_len, 64]

# 5. Compute scores (now standard MHA - no GQA mapping needed)
scores = torch.matmul(Q_rope, K_rope.transpose(2, 3)) / sqrt(d_head)
# scores: [bsz, 14, q_len, q_len] - all 14 heads computed independently

# 6. Softmax
attn_weights = F.softmax(scores, dim=-1)

# 7. Output
context = torch.matmul(attn_weights, V_heads)
```

**Key Insight**: After K/V expansion, PyTorch treats it as **standard MHA** with 14 heads. No special GQA logic in score computation.

### Our Current Approach (Incorrect)

```cpp
// compute_scores() implementation
for (int h = 0; h < n_heads_; ++h) {  // h = 0..13
    int kv_head = h / (n_heads_ / n_kv_heads_);  // GQA mapping: 0..6 -> 0, 7..13 -> 1
    
    // Try to compute Q[h] @ K[kv_head]^T directly
    // This produces WRONG results because:
    // 1. K/V are still in [bsz, q_len, n_kv_heads*d_head] shape
    // 2. We're trying to do head mapping in a flat layout
    // 3. Indexing doesn't match PyTorch's head-reshaped approach
}
```

**Problem**: We're trying to do GQA mapping **during** score computation with flat K/V tensors. PyTorch expands K/V heads **before** score computation and works with head-reshaped tensors.

---

## Solution: Match PyTorch's Approach

### Required Changes to FP32AttentionKernel

**Step 1: Expand K/V Heads**

Implement `expand_kv_heads()` to replicate K/V from `[bsz, q_len, n_kv_heads*d_head]` to `[bsz, q_len, n_heads*d_head]`:

```cpp
void FP32AttentionKernel::expand_kv_heads(
    const float *kv,          // [bsz, seq_len, n_kv_heads * d_head]
    float *kv_expanded,       // [bsz, seq_len, n_heads * d_head] OUT
    int batch, int seq_len)
{
    int num_groups = n_heads_ / n_kv_heads_;  // 14 / 2 = 7
    
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int kv_h = 0; kv_h < n_kv_heads_; ++kv_h) {
                // Replicate this kv_head to 'num_groups' output heads
                for (int g = 0; g < num_groups; ++g) {
                    int out_head = kv_h * num_groups + g;  // 0->0..6, 1->7..13
                    
                    for (int d = 0; d < d_head_; ++d) {
                        int in_idx = (b * seq_len + i) * (n_kv_heads_ * d_head_) 
                                   + kv_h * d_head_ + d;
                        int out_idx = (b * seq_len + i) * (n_heads_ * d_head_) 
                                    + out_head * d_head_ + d;
                        kv_expanded[out_idx] = kv[in_idx];
                    }
                }
            }
        }
    }
}
```

**Step 2: Update forward() Pipeline**

```cpp
bool FP32AttentionKernel::forward(...) {
    // Allocate buffers
    scores_buffer_.resize(batch * n_heads_ * seq_len * seq_len);
    attn_weights_buffer_.resize(batch * n_heads_ * seq_len * seq_len);
    
    // Expand K/V if GQA (BEFORE score computation!)
    const float *k_to_use = k;
    const float *v_to_use = v;
    
    if (n_kv_heads_ < n_heads_) {
        size_t expanded_size = batch * seq_len * n_heads_ * d_head_;
        k_expanded_buffer_.resize(expanded_size);
        v_expanded_buffer_.resize(expanded_size);
        
        expand_kv_heads(k, k_expanded_buffer_.data(), batch, seq_len);
        expand_kv_heads(v, v_expanded_buffer_.data(), batch, seq_len);
        
        k_to_use = k_expanded_buffer_.data();
        v_to_use = v_expanded_buffer_.data();
    }
    
    // Now compute standard MHA scores (no GQA mapping)
    compute_scores(q, k_to_use, batch, seq_len);
    apply_softmax(batch, seq_len, use_causal_mask, eps);
    compute_output(v_to_use, output, batch, seq_len);
}
```

**Step 3: Simplify compute_scores()**

```cpp
void FP32AttentionKernel::compute_scores(
    const float *q,           // [batch, seq_len, n_heads * d_head]
    const float *k_expanded,  // [batch, seq_len, n_heads * d_head] (after expansion!)
    int batch, int seq_len)
{
    int d_model_q = n_heads_ * d_head_;
    int d_model_k = n_heads_ * d_head_;  // Now same as Q after expansion
    
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n_heads_; ++h) {  // Standard MHA - no GQA mapping!
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < d_head_; ++d) {
                        int q_idx = (b * seq_len + i) * d_model_q + h * d_head_ + d;
                        int k_idx = (b * seq_len + j) * d_model_k + h * d_head_ + d;
                        dot += q[q_idx] * k_expanded[k_idx];
                    }
                    
                    int score_idx = ((b * n_heads_ + h) * seq_len + i) * seq_len + j;
                    scores_buffer_[score_idx] = dot;  // Raw scores (no scaling yet)
                }
            }
        }
    }
}
```

Similarly update `compute_output()` to use `v_expanded` with standard MHA indexing.

---

## Implementation Plan

### Phase 1: Fix GQA Implementation (CURRENT)

1. ✅ Identified root cause (GQA mapping vs K/V expansion mismatch)
2. ⏳ Implement `expand_kv_heads()` (stub already exists, needs proper implementation)
3. ⏳ Update `forward()` to expand K/V before score computation
4. ⏳ Simplify `compute_scores()` to standard MHA (remove GQA mapping)
5. ⏳ Simplify `compute_output()` to standard MHA
6. ⏳ Rebuild and test

**Expected Result**: `BasicForwardVsPyTorch` should pass with rel_l2 < 1e-4

### Phase 2: Additional Test Coverage

7. Generate causal snapshots: `python3 python/reference/generate_attention_snapshots.py --causal --output pytorch_attention_snapshots_causal`
8. Generate single-token snapshots: `python3 python/reference/generate_attention_snapshots.py --prompt 'A' --output pytorch_attention_snapshots_single`
9. Run all 4 parity tests

**Expected Result**: All parity tests pass

### Phase 3: Validate INT8 Against Fixed FP32

10. Update `Test__INT8AttentionKernel.cpp` to use `FP32AttentionKernel` as reference
11. Remove buggy `reference_attention_fp32()` helper
12. Relax tolerances to rel_l2 < 0.05 (5% for quantization)
13. Rebuild and run INT8 tests

**Expected Result**: INT8 tests now use validated baseline (isolate quantization bugs)

---

## File Status

### Created This Session

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `tests/v2/unit/Test__FP32AttentionParity.cpp` | 400+ | ✅ Working | PyTorch parity test suite |
| `src/v2/kernels/cpu/FP32AttentionKernel.h` | 150 | ⚠️ Has bug | FP32 attention interface |
| `src/v2/kernels/cpu/FP32AttentionKernel.cpp` | 280 | ⚠️ Has bug | FP32 attention implementation |
| `python/reference/generate_attention_snapshots.py` | 400+ | ✅ Tested | Snapshot generator |

### Modified This Session

| File | Change | Reason |
|------|--------|--------|
| `src/v2/CMakeLists.txt` | Added FP32AttentionKernel.cpp | Build integration |
| `tests/v2/CMakeLists.txt` | Added v2_test_fp32_attention_parity | Test integration |

---

## Debugging Notes

### Evolution of Errors

1. **First run** (before n_kv_heads fix):
   - Max abs diff: `3.76913718e+31`
   - Diagnosis: Uninitialized memory (n_kv_heads=0 caused wrong buffer sizes)

2. **After n_kv_heads fix**:
   - Max abs diff: `0.196`
   - Relative L2: `2.34` (234%)
   - Diagnosis: GQA implementation mismatch (current state)

### Key Insights

1. **GQA is not "just mapping heads"**: PyTorch's approach is to **physically expand** K/V tensors, then treat it as standard MHA. This is simpler and matches how attention is conceptually described.

2. **Shape assumptions**: Our implementation assumed we could do GQA mapping in flat tensors, but this doesn't match how PyTorch reshapes to `[bsz, n_heads, seq_len, d_head]` before any attention computation.

3. **Testing methodology works**: Even though FP32 kernel has a bug, the parity test infrastructure correctly identified it with clear error metrics. This validates the PyTorch parity approach.

---

## Next Steps

**Immediate** (Priority 1):
- Fix `expand_kv_heads()` implementation
- Update `forward()` to call K/V expansion before scores
- Simplify `compute_scores()` and `compute_output()` to standard MHA
- Test and verify parity

**Short-term** (Priority 2):
- Generate additional snapshots (causal, single-token)
- Validate all parity tests pass
- Update INT8 tests to use validated FP32 baseline

**Medium-term** (Priority 3):
- Debug INT8 implementation (now isolated from FP32 issues)
- Implement INT8 SwiGLU kernel
- Build Qwen2 INT8 pipeline

---

## Architecture Reference

**Qwen 2.5 0.5B Instruct**:
- `n_layers`: 24
- `n_heads`: 14 (query heads)
- `n_kv_heads`: 2 (GQA 7:1 ratio)
- `d_model`: 896
- `d_head`: 64

**GQA Head Mapping** (after expansion):
- KV head 0 → Q heads 0, 1, 2, 3, 4, 5, 6
- KV head 1 → Q heads 7, 8, 9, 10, 11, 12, 13

**Tensor Shapes**:
- Q: `[batch, seq_len, 896]` (14 × 64)
- K: `[batch, seq_len, 128]` (2 × 64)
- V: `[batch, seq_len, 128]` (2 × 64)
- K_expanded: `[batch, seq_len, 896]` (after expansion)
- V_expanded: `[batch, seq_len, 896]` (after expansion)
- Scores: `[batch, 14, seq_len, seq_len]`
- Output: `[batch, seq_len, 896]`

---

## Success Criteria

**FP32 Parity Test**:
- ✅ Max abs diff < 1e-4
- ✅ Relative L2 < 1e-4 (0.01%)
- ✅ All 4 test cases pass (basic, scores, causal, single-token)

**INT8 Validation**:
- ✅ Uses validated FP32 as reference
- ✅ Relative L2 < 0.05 (5% quantization tolerance)
- ✅ Causal masking works
- ✅ Single-sequence edge case handles correctly

---

**Status**: 🔄 Ready to implement GQA fix. Clear path forward identified.
