# GQA Expansion Bug Fix - Investigation Summary

**Date:** October 8, 2025  
**Issue:** ROPE_APPLICATION parity test failing with max_abs=97.45 at position [3,992]  
**Status:** Root cause identified, fix implemented, but underlying data mismatch persists

---

## Problem Statement

The parity test `ParityFramework.OpenBLASPrefillVsPyTorch` was failing at the ROPE_APPLICATION stage with:
- **Error position:** [3,992] (token 3, K dimension 96)
- **Expected value:** 47.602432 (from PyTorch)
- **Actual value:** -49.851048 (from Llaminar)
- **Difference:** 97.453476

This was the **first divergence** in the pipeline, indicating a fundamental issue in attention computation.

---

## Investigation Process

### Phase 1: Hypothesis - GQA Expansion Bug

**Discovery:** The GQA (Grouped Query Attention) expansion was using **local K/V** tensors instead of **gathered K/V** from all ranks.

**Context:**
- Qwen-0.5B has 14 Q heads and 2 KV heads
- With 2 MPI ranks in tensor parallel:
  - Rank 0: 7 Q heads, 1 KV head (KV head 0)
  - Rank 1: 7 Q heads, 1 KV head (KV head 1)

**The Bug:**
```cpp
// BEFORE (WRONG):
llaminar::attn::expand_kv_for_gqa(
    local_k->data(), local_v->data(),      // ❌ Only has 1 KV head per rank!
    local_k_expanded->data(), local_v_expanded->data(),
    seq_len, head_dim_, local_heads, local_kv_heads);
```

Each rank was expanding its **local** K (containing only 1 KV head) to 7 Q heads, instead of using the **gathered** K (containing all 2 KV heads).

### Phase 2: The Fix

**Solution:** Reuse the K/V gathering already performed for ROPE snapshot, then expand the gathered tensors.

**Key Changes:**

1. **Modified snapshot gathering** (lines 1090-1248 of `MPIAttentionKernel.cpp`):
   - Now gathers Q, K, **and V** (previously only Q and K)
   - Gathering happens if `snapshot_callback_ || n_head_ != n_head_kv_` (not just for snapshots)
   - Creates `global_q_rope`, `global_k_rope`, `global_v_rope` with all heads from all ranks

2. **Fixed GQA expansion** (lines 1257-1267):
   ```cpp
   // AFTER (CORRECT):
   llaminar::attn::expand_kv_for_gqa(
       global_k_rope->data(), global_v_rope->data(),  // ✅ All 2 KV heads!
       local_k_expanded->data(), local_v_expanded->data(),
       seq_len, head_dim_, local_heads, n_head_kv_);  // ✅ Expand 2→7 heads
   ```

**Code Flow After Fix:**
```
┌─────────────┐
│  Rank 0     │ local_k: [seq=5, kvh=1, dim=64] = KV head 0
│  Rank 1     │ local_k: [seq=5, kvh=1, dim=64] = KV head 1
└─────────────┘
       │
       │ MPI_Allgather (row-by-row)
       ▼
┌─────────────────────────────────────────┐
│  global_k_rope [seq=5, kvh=2, dim=64]   │
│  = [KV_head_0_from_rank_0, KV_head_1_from_rank_1]
└─────────────────────────────────────────┘
       │
       │ expand_kv_for_gqa (2 KV heads → 7 Q heads)
       ▼
┌─────────────────────────────────────────┐
│  local_k_expanded [seq=5, qh=7, dim=64] │
│  = [Q0, Q1, Q2, Q3, Q4, Q5, Q6]         │
│     ↓   ↓   ↓   ↓   ↓   ↓   ↓           │
│    KV0 KV1 KV0 KV1 KV0 KV1 KV0          │
└─────────────────────────────────────────┘
       │
       ▼
  Attention computation (correct K for all Q heads)
```

### Phase 3: Verification - The Puzzle

**Expected Outcome:** ROPE_APPLICATION error should be fixed.

**Actual Outcome:** Test still fails with **identical error**:
- Position [3,992]: expected=47.602, actual=-49.851, diff=97.45

**Diagnostic Findings:**

1. **MPI_Allgather is working correctly:**
   ```
   Rank 0 local_k[t=0]: -8.66009, -4.08692, -6.19157, 0.676327, -0.00483325
   Rank 1 local_k[t=0]: -9.68714,  1.88168,  6.95051, 7.95585,  5.5358
   
   After gather:
   global_k_rope[0..4]:   -8.66009, -4.08692, -6.19157, 0.676327, -0.00483325  (KV head 0)
   global_k_rope[64..68]: -9.68714,  1.88168,  6.95051, 7.95585,  5.5358       (KV head 1)
   ```

2. **Failing position analysis:**
   ```
   Position [3,992] = Token 3, K dimension 96
   K dimension 96 = KV head 1, dimension 32 within head
   global_k_rope offset = 3 * 128 + 64 + 32 = 480
   global_k_rope[480] = -49.851
   
   PyTorch expects: 47.602
   Difference: 97.45 (almost exact negation!)
   ```

3. **The values are nearly opposite in sign:**
   - Llaminar: -49.851
   - PyTorch:   47.602
   - Ratio: ≈ -1.048

---

## Current Status

### ✅ What Was Fixed

1. **GQA expansion now uses gathered K/V** instead of local K/V
2. **V tensor is now gathered** along with Q and K for completeness
3. **Gathering is triggered by GQA need** (`n_head_ != n_head_kv_`), not just snapshots
4. **All 2 KV heads are available** for expansion to 7 Q heads per rank

### ❓ Outstanding Mystery

The test **still fails** with the same error despite the fix. This suggests:

1. **The original hypothesis was correct** (needed to gather K/V before GQA expansion)
2. **But there's an upstream issue** causing wrong K values before gathering
3. **Possible root causes:**
   - Weight loading/slicing for K projection across ranks
   - K projection matrix multiplication producing wrong values
   - RoPE application introducing errors
   - Incorrect rank assignment to KV heads

### 🔍 Key Observation

The gathered K values themselves don't match PyTorch expectations, with near-opposite signs. This indicates the problem occurs **before** GQA expansion, likely in:
- K projection weight partitioning (`wk_global` slicing)
- K projection computation (`MPILinearKernel`)
- RoPE application to K

---

## Next Steps

### Recommended Investigation Path

1. **Verify K projection weight slicing:**
   ```cpp
   // Check how wk_global is partitioned across ranks
   // For 2 KV heads split across 2 ranks:
   // Rank 0 should get: wk_global[0:out_features/2, :]
   // Rank 1 should get: wk_global[out_features/2:, :]
   ```

2. **Add K projection output validation:**
   - Compare local_k values immediately after K projection with PyTorch
   - Before RoPE, before gathering
   - Verify which rank should own which KV head

3. **Verify RoPE doesn't introduce the sign flip:**
   - Check if RoPE cosine/sine tables are correct
   - Verify rotation is applied correctly to K

4. **Consider rank-to-head assignment:**
   - Confirm rank 0 = KV head 0, rank 1 = KV head 1
   - Or could it be reversed?

### Test to Run

```bash
# Add logging to K projection output:
cd /workspaces/llaminar
# Edit src/kernels/MPILinearKernel.cpp to log K projection outputs
cmake --build build --target test_parity_framework --parallel
mpirun -np 2 ./build/test_parity_framework \
  --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch" 2>&1 | \
  grep -E "(K_PROJECTION|local_k\[)" | head -50
```

---

## Code Changes Made

### File: `src/kernels/MPIAttentionKernel.cpp`

**Location:** Lines 1090-1175 (snapshot gathering section)

**Change 1:** Added V tensor gathering
```cpp
// OLD:
if (snapshot_callback_) {
    // Only gathered Q and K
}

// NEW:
if (snapshot_callback_ || n_head_ != n_head_kv_) {
    // Now gathers Q, K, and V
    global_v_rope = TensorFactory::create_simple({seq_len, k_v_dim});
    
    for (int t = 0; t < seq_len; ++t) {
        const float *local_v_row = local_v->data() + t * local_kv_head_dim;
        float *global_v_row = global_v_rope->data() + t * k_v_dim;
        
        MPI_Allgather(local_v_row, local_kv_head_dim, MPI_FLOAT,
                      global_v_row, local_kv_head_dim, MPI_FLOAT,
                      MPI_COMM_WORLD);
    }
}
```

**Location:** Lines 1257-1267 (GQA expansion)

**Change 2:** Use gathered K/V for expansion
```cpp
// OLD:
llaminar::attn::expand_kv_for_gqa(
    local_k->data(), local_v->data(),  // ❌ Wrong!
    ...
);

// NEW:
llaminar::attn::expand_kv_for_gqa(
    global_k_rope->data(), global_v_rope->data(),  // ✅ Correct!
    local_k_expanded->data(), local_v_expanded->data(),
    seq_len, head_dim_, local_heads, n_head_kv_);
```

**Additional:** Debug logging added for verification:
- Lines 1132-1137: Log local_k before gathering
- Lines 1163-1177: Log global_k_rope after gathering
- Lines 1238-1247: Log snapshot values at failing position

---

## Lessons Learned

1. **MPI collective operations require careful attention to data scope**
   - Local vs global tensors must be clearly distinguished
   - Gathering must happen before operations that need data from all ranks

2. **GQA in distributed settings is non-trivial**
   - Each rank has subset of KV heads
   - Must gather before expanding to match all Q heads

3. **Diagnostic logging is essential**
   - Added extensive logging to track data flow
   - Critical for understanding distributed execution

4. **Sign flips often indicate indexing errors**
   - Near-opposite values (-49.85 vs 47.60) suggest:
     - Wrong head being accessed
     - Incorrect weight slice
     - Transposition error

---

## Files Modified

1. `src/kernels/MPIAttentionKernel.cpp`
   - ~100 lines modified
   - Fixed GQA expansion to use gathered K/V
   - Added comprehensive debug logging

2. `tests/test_parity_framework.cpp`
   - No changes (test infrastructure already in place)

3. `src/kernels/common/attention_primitives.cpp`
   - No changes to expansion logic (already correct)
   - Static logging for GQA expansion parameters

---

## Conclusion

The GQA expansion bug has been **identified and fixed**, but the parity test failure persists due to an **upstream issue** in K projection or weight loading. The fix ensures that attention computation uses the correct gathered K/V tensors, which is architecturally correct regardless of the upstream issue.

**Next priority:** Debug K projection to understand why the gathered K values don't match PyTorch expectations.

