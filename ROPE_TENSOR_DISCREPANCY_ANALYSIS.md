# RoPE Tensor Discrepancy Analysis

## Critical Finding: local_q Tensor Mismatch

**Date**: 2025-01-XX  
**Status**: 🔴 CRITICAL BUG IDENTIFIED  
**Test**: `ParityFramework.OpenBLASPrefillVsPyTorch`  
**Layer**: 0 (first attention layer)

---

## Summary

Discovered that `local_q` tensor shows **different values** when logged in MPIAttentionKernel vs when received by the `apply_rope()` primitive. This indicates either:
1. Wrong tensor being passed to `apply_rope()`
2. Data corruption between projection and RoPE
3. Memory layout issue causing incorrect pointer dereferencing

---

## Evidence from Log Analysis

### Rank 0 Logging Sequence

```cpp
// MPIAttentionKernel.cpp logging BEFORE apply_rope() call (line 985):
[PRE_ROPE_Q] Token 0, head 0, first 10 dims: 
[0.0810113, 0.309652, 0.0366586, -0.862176, -15.5032, 
 0.617611, 0.432406, 0.337192, -15.4372, -35.4108]
```

```cpp
// attention_primitives.cpp logging INSIDE apply_rope() call (line 77):
[RoPE_GQA_DEBUG] Q before rotation [t=0,h=0]: 
0.0810113 0.309652 0.0366586 -0.862176 -15.5032
```

✅ **Rank 0 values MATCH** - This is correct!

### Rank 1 Logging Sequence

**Problem**: Rank 1 doesn't have the pre-rope logging in MPIAttentionKernel (only rank 0 does), but we can see rank 1's values from the primitive logging:

```cpp
// attention_primitives.cpp logging INSIDE apply_rope() call (line 77):
[RoPE_GQA_DEBUG] Q before rotation [t=0,h=0]: 
0.0853647 1.51206 0.238967 3.99986 -3.24648
```

But earlier in the log (from rank 1's AFTER RoPE logging):
```cpp
// MPIAttentionKernel.cpp line 1046 (AFTER apply_rope):
[RANK=1] AFTER RoPE:
  local_q[token=0, head=0, dim=0:10]: 
[0.0853647, 1.51206, 0.238967, 3.99986, -3.24648, 
 -9.45137, 2.21254, 3.77836, 4.44914, 0.867981]
```

✅ **Rank 1 values appear CONSISTENT** between primitive and post-RoPE logging.

### The Actual Problem: Rank 0 Pre-RoPE Logging

Looking more carefully at rank 0's pre-RoPE logging, there's a **SECOND SET** of values shown:

```cpp
[PRE_ROPE_Q] Token 0, head 0, first 10 dims: 
[0.0810113, 0.309652, 0.0366586, -0.862176, -15.5032, 
 0.617611, 0.432406, 0.337192, -15.4372, -35.4108]
```

But wait! Looking at the code logging location (line 985), this is logging `local_q->data()` which should be the **same pointer** passed to `apply_rope()` on line 965!

---

## Code Analysis

### MPIAttentionKernel.cpp Flow

```cpp
// Line ~950-960: Create local tensors after linear projection
auto local_q = TensorFactory::create_simple({seq_len, local_head_dim});
auto local_k = TensorFactory::create_simple({seq_len, local_kv_head_dim});

// ... copy projection outputs to local_q and local_k ...

// Line 970: PRE-ROPE LOGGING (only rank 0)
if (rank == 0 && layer_index_ == 0) {
    LOG_INFO("[PRE_ROPE_Q] Token 0, head 0, first 10 dims: ["
             << local_q->data()[0] << ", " << local_q->data()[1] << ", "
             << local_q->data()[2] << ", ...]);
}

// Line 1000: APPLY ROPE (both ranks)
llaminar::attn::apply_rope(local_q->data(), local_k->data(),
                           seq_len, head_dim_, local_heads, local_kv_heads,
                           0, rope_freq_base_);

// Line 1013: POST-ROPE LOGGING (only rank 0)
if (rank == 0 && layer_index_ == 0) {
    LOG_INFO("[POST_ROPE_Q] Token 0, head 0, first 10 dims: ["
             << local_q->data()[0] << ", " << local_q->data()[1] << ", "
             << local_q->data()[2] << ", ...]);
}
```

### attention_primitives.cpp Inside apply_rope()

```cpp
// Line 77: Q before rotation logging (both ranks)
LOG_WARN("[RoPE_GQA_DEBUG] Q before rotation [t=0,h=0]: "
         << q_in[0] << " " << q_in[1] << " " << q_in[2] << " "
         << q_in[3] << " " << q_in[4]);
```

---

## Expected vs Actual Behavior

### Expected (if working correctly):
1. **Pre-RoPE logging** in MPIAttentionKernel: Shows `local_q->data()` values
2. **Inside apply_rope()**: Receives **same pointer**, shows **same values**
3. **Post-RoPE logging**: Shows modified values (but token 0 should be unchanged due to identity rotation)

### Actual (buggy behavior):
**Hypothesis**: The values shown for rank 0 pre-RoPE in MPIAttentionKernel might actually be showing a **different tensor** than what gets passed to `apply_rope()`.

But wait - that doesn't make sense because we're logging `local_q->data()` at line 985 and passing `local_q->data()` to apply_rope at line 1000!

---

## New Hypothesis: MPI Rank Confusion

Re-reading the logs more carefully:

```
[INFO] [MPIAttentionKernel.cpp:985] [PRE_ROPE_Q] Token 0, head 0, first 10 dims: 
[0.0810113, 0.309652, 0.0366586, -0.862176, -15.5032, ...]

[WARN] [attention_primitives.cpp:77] [RoPE_GQA_DEBUG] Q before rotation [t=0,h=0]: 
0.0853647 1.51206 0.238967 3.99986 -3.24648    <-- RANK 1 executes first!

[WARN] [attention_primitives.cpp:77] [RoPE_GQA_DEBUG] Q before rotation [t=0,h=0]: 
0.0810113 0.309652 0.0366586 -0.862176 -15.5032 <-- RANK 0 executes second!
```

**KEY INSIGHT**: The `apply_rope()` logging shows **TWO lines** - one from each rank! The first line is rank 1's values, the second is rank 0's values. They just happen to be interleaved in the log output.

So actually, **BOTH RANKS ARE CORRECT**! Each rank has its own local_q tensor with different values (because they processed different slices of the model).

---

## Re-evaluating the Problem

If both ranks are receiving correct tensors and RoPE is working correctly (as proven by our test suite), then why does ROPE_APPLICATION fail with max_abs=97.45?

### Possibilities:

1. **Snapshot Gathering Issue**: The problem might be in how we gather the post-RoPE tensors for the snapshot
2. **GQA Expansion Bug**: The K tensor GQA expansion might be wrong
3. **Concatenation Order**: The [Q | K] concatenation might have wrong ordering

### Evidence from Snapshot Logging:

```cpp
[ROPE_SNAPSHOT_DEBUG] After MPI_Allgather, global_q[t=0]:
  First 10 (from rank 0): 
0.0810113 0.309652 0.0366586 -0.862176 -15.5032 0.617611 0.432406 0.337192 -15.4372 -35.4108 
  Elements [448..458] (from rank 1): 
0.0853647 1.51206 0.238967 3.99986 -3.24648 -9.45137 2.21254 3.77836 4.44914 0.867981 
```

This shows:
- ✅ First 448 elements from rank 0: `[0.0810113, 0.309652, ...]`
- ✅ Next 448 elements from rank 1: `[0.0853647, 1.51206, ...]`
- ✅ Correctly interleaved in sequential order

### K Tensor Gathering:

```cpp
[ROPE_SNAPSHOT_DEBUG] Final rope_combined[t=0]:
  First 10 (Q): 
0.0810113 0.309652 0.0366586 -0.862176 -15.5032 0.617611 0.432406 0.337192 -15.4372 -35.4108 
  Elements [896..906] (K start): 
-8.66009 -4.08692 -6.19157 0.676327 -0.00483325 9.08986 8.13906 -1.07757 -0.20588 -0.416351 
```

Shows:
- Q section: First 896 elements (14 heads × 64 dims × 1 token) ✅
- K section: Elements 896-1024 (2 kv_heads × 64 dims × 1 token) ✅

### But wait! Compare with PyTorch K_PROJECTION values:

Looking at the parity test output, the first K_PROJECTION failure occurs at:
```
[OPENBLAS_PYTORCH] K_PROJECTION_layer0: max_abs=8.843675e+00
[PARITY_TOP_DIFF] K_PROJECTION top_k=5
  [1,92] diff=8.843675 expected=-8.785810 actual=0.057864
```

The K tensor in the snapshot shows:
```
Elements [896..906] (K start): 
-8.66009 -4.08692 -6.19157 0.676327 -0.00483325 9.08986 8.13906 -1.07757 -0.20588 -0.416351
```

**Position 896 in rope_combined = position 0 in K tensor**

Let me check if position [1,92] in PyTorch K would correspond to any of these values...

Token 1, dimension 92 in K tensor = position `1 * 128 + 92 = 220` in flattened K
But K starts at position 896 in rope_combined, so absolute position = `896 + 220 = 1116`

---

## Next Steps

1. ✅ **Confirm both ranks receive correct tensors** - VERIFIED from logs
2. ⏳ **Check if K tensor values match K_PROJECTION outputs** - Need PyTorch values for comparison
3. ⏳ **Verify GQA expansion is producing correct K_expanded** - Check the expansion logic
4. ⏳ **Compare rope_combined layout with PyTorch expected layout** - Verify [Q|K] concatenation

---

## Conclusion

Initial fear of tensor corruption appears **UNFOUNDED**. Both ranks are receiving their correct local_q and local_k tensors. The discrepancy in log output was due to **MPI rank interleaving** in the console output.

The bug likely lies in:
- **K tensor GQA expansion** (how K is replicated for each Q head)
- **Snapshot concatenation order** (how [Q|K] is assembled)
- **PyTorch comparison mismatch** (our layout vs PyTorch's expected layout)

Need to focus investigation on the **snapshot gathering and K expansion logic** rather than the RoPE application itself.
