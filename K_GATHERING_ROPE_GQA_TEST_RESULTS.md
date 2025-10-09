# K Gathering and RoPE GQA Test Results

**Date:** October 8, 2025  
**Test File:** `tests/test_k_gathering_and_rope_gqa.cpp`  
**Purpose:** Validate two hypotheses for ROPE_APPLICATION failure (max_abs=97.45)

## Executive Summary

✅ **Both hypotheses VALIDATED - gathering and RoPE GQA are CORRECT**
- MPI_Allgather produces sequential head order [head0 | head1] ✓
- RoPE uses correct KV-head indices for GQA ✓
- K RoPE results are independent of Q head count ✓
- Multi-rank distributed RoPE matches sequential reference ✓

❌ **Bug is NOT in primitives - must be in MPIAttentionKernel integration**
- Isolation tests prove the building blocks work correctly
- Real ROPE_APPLICATION still fails with max_abs=97.45
- Bug likely in how MPIAttentionKernel calls/integrates these primitives

## Test Suite Overview

### Test 1: AllgatherProducesSequentialHeadOrder ✅ PASS

**Hypothesis:** MPI_Allgather concatenates K heads in wrong order

**Test Setup:**
- 2 MPI ranks, each with 1 KV head (64 dims)
- Rank 0 fills local K with pattern `1.0 + 0.01*dim`
- Rank 1 fills local K with pattern `2.0 + 0.01*dim`
- Perform MPI_Allgather

**Expected Result:** `[head0_all_dims | head1_all_dims]` = `[1.0...1.63 | 2.0...2.63]`

**Actual Result:**
```
Gathered K[0, 0:70]: 1, 1.01, ..., 1.63 | 2, 2.01, ..., 2.63
✓ PASS: K tensor has sequential head order [head0 | head1]
```

**Conclusion:** MPI_Allgather IS correctly producing sequential head order.

---

### Test 2: RopeUsesCorrectKVHeadIndices ✅ PASS

**Hypothesis:** RoPE applies wrong head indices for GQA (q_heads ≠ k_heads)

**Test Setup:**
- Q: 4 heads × 8 dims = 32 total
- K: 2 KV heads × 8 dims = 16 total (GQA)
- Single rank (tests RoPE logic without MPI)
- Initialize with known patterns, apply RoPE

**Verification Method:**
- Manually compute expected RoPE for each K head using theta_i = freq_base^(-2i/head_dim)
- CRITICAL: Uses KV-head index (0, 1), NOT Q-head indices (0-3)
- Compare actual vs expected for all positions and dimension pairs

**Result:**
```
[ROPE_GQA_TEST] Checking KV-head 0
[ROPE_GQA_TEST] Checking KV-head 1
✓ PASS: RoPE uses correct KV-head indices for GQA
```

**Conclusion:** RoPE correctly uses KV-head indices 0, 1 for theta calculation, independent of Q head count.

---

### Test 3: RopeKIndependentOfQHeadCount ✅ PASS

**Hypothesis:** K tensor RoPE depends on Q head count (which would be a bug)

**Test Setup:**
- Fixed K configuration: 2 KV heads × 8 dims
- Variable Q configurations: 2, 4, 8, 14 heads (including real Qwen-0.5B)
- Same initial K values for all runs
- Apply RoPE with different q_heads

**Verification:**
- Compare K tensor results across all q_heads variants
- All should be bitwise identical

**Result:**
```
[ROPE_INDEPENDENCE_TEST] Comparing K results across q_heads variants
✓ PASS: K RoPE is independent of q_heads count
```

**Conclusion:** K tensor RoPE calculation is completely independent of Q head count, as required for correct GQA.

---

### Test 4: MultiRankKGatheringAfterRope ✅ PASS

**Hypothesis:** Distributed RoPE + gathering differs from sequential RoPE

**Test Setup:**
- 2 MPI ranks, 2 KV heads total (1 per rank)
- Each rank: 64 dims, 5 sequence positions
- Rank 0 initializes head 0: `[0, 1, 2, ..., 63]`
- Rank 1 initializes head 1: `[100, 101, 102, ..., 163]`

**Sequential Reference (Rank 0):**
1. Create full K tensor `[head0 | head1]`
2. Apply RoPE to complete tensor
3. Store as reference

**Distributed Approach (Both Ranks):**
1. Each rank applies RoPE to local head
2. MPI_Allgather to reconstruct full tensor
3. Compare with reference

**Result:**
```
[MULTI_RANK_ROPE_TEST] Comparing gathered K with reference
Max difference: 0 at index 0
✓ PASS: Multi-rank K gathering after RoPE is correct
```

**Conclusion:** Distributed RoPE computation followed by MPI_Allgather produces EXACT same result as sequential RoPE (zero difference!).

## Analysis and Implications

### What We Proved

1. **MPI_Allgather Order is Correct**
   - Produces sequential `[head0 | head1]` layout
   - Matches PyTorch's expected head ordering
   - No rank-based concatenation bug

2. **RoPE GQA Logic is Correct**
   - Uses proper KV-head indices (not Q-head indices)
   - Theta calculation correct for each K head
   - Independent of Q head count
   - Matches mathematical RoPE specification

3. **Distributed vs Sequential Equivalence**
   - Local RoPE + gathering = global RoPE
   - Zero numerical error (exact match)
   - MPI communication preserves values

### What This Means for the Bug

Since **both hypotheses are validated as CORRECT**, the ROPE_APPLICATION failure (max_abs=97.45) must be caused by:

#### Most Likely: Integration Bug in MPIAttentionKernel

1. **Incorrect tensor slicing before RoPE**
   - Maybe Q/K projections aren't properly shaped before passing to RoPE
   - Head dimensions might be wrong

2. **Wrong parameters passed to apply_rope()**
   - Incorrect `seq_len`, `head_dim`, `q_heads`, or `k_heads`
   - Wrong `n_past` value (should be 0 for prefill)

3. **Data corruption between projection and RoPE**
   - Memory alignment issues
   - Buffer overruns
   - Incorrect tensor views

4. **Snapshot gathering bug**
   - The primitives work, but snapshot code doesn't call them correctly
   - Maybe gathering happens before RoPE instead of after
   - Or concatenation order is wrong in snapshot code specifically

#### Less Likely: Weight Loading Issues

5. **Q/K projection weights are wrong**
   - But Q_PROJECTION and K_PROJECTION pass with small errors (0.09-0.11)
   - Unlikely to cause 1000x amplification

## Next Steps

### High Priority: Debug MPIAttentionKernel Integration

1. **Add detailed logging to ROPE_APPLICATION snapshot capture:**
   ```cpp
   // Before RoPE
   LOG_INFO("Pre-RoPE Q[0,0:10]: " << q_values);
   LOG_INFO("Pre-RoPE K[0,0:10]: " << k_values);
   
   // Apply RoPE
   llaminar::attn::apply_rope(q, k, seq_len, head_dim, q_heads, k_heads, n_past, freq_base);
   
   // After RoPE
   LOG_INFO("Post-RoPE Q[0,0:10]: " << q_values);
   LOG_INFO("Post-RoPE K[0,0:10]: " << k_values);
   
   // After gathering
   LOG_INFO("Gathered Q shape: " << gathered_q_shape);
   LOG_INFO("Gathered K shape: " << gathered_k_shape);
   ```

2. **Verify parameters passed to apply_rope():**
   - Print `seq_len`, `head_dim`, `q_heads`, `k_heads`, `n_past`, `freq_base`
   - Compare with expected values for Qwen-0.5B

3. **Check tensor shapes at each step:**
   - Q projection output: `[seq_len, 14*64]`
   - K projection output: `[seq_len, 2*64]`
   - After RoPE: same shapes
   - After gather: Q `[seq_len, 14*64]`, K `[seq_len, 2*64]`
   - Concatenated: `[seq_len, 14*64 + 2*64] = [seq_len, 1024]`

4. **Compare with PyTorch step-by-step:**
   - Pre-RoPE Q/K values should match (within 0.1 error from projections)
   - Post-RoPE divergence indicates parameter mismatch or corruption
   - Gathering divergence indicates MPI communication issue

### Validation Criteria

After finding and fixing the bug:
- Q_PROJECTION: ✓ Already passes (max_abs=0.11)
- K_PROJECTION: ✓ Already passes (max_abs=0.095)
- **ROPE_APPLICATION: Should pass with max_abs <0.15** (allowing for small projection errors + RoPE numerical precision)
- ATTENTION_SCORES: Should improve significantly

## Test Execution

### Single-Rank Tests (RoPE Logic)
```bash
./build/test_k_gathering_and_rope_gqa --gtest_filter="*RopeUsesCorrectKVHeadIndices*:*RopeKIndependentOfQHeadCount*"
```
Result: ✅ 2/2 tests passed (3ms)

### Multi-Rank Tests (MPI Gathering)
```bash
mpirun -np 2 ./build/test_k_gathering_and_rope_gqa --gtest_filter="*AllgatherProducesSequentialHeadOrder*:*MultiRankKGatheringAfterRope*"
```
Result: ✅ 2/2 tests passed (0ms)

### Full Test Suite
```bash
ctest --test-dir build -R KGatheringAndRopeGQA
```
Result: ✅ All tests passed

## Code Location

- Test file: `/workspaces/llaminar/tests/test_k_gathering_and_rope_gqa.cpp`
- CMake registration: `/workspaces/llaminar/CMakeLists.txt` lines ~953-966
- RoPE primitive: `/workspaces/llaminar/src/kernels/common/attention_primitives.cpp`
- Integration code: `/workspaces/llaminar/src/kernels/MPIAttentionKernel.cpp`

## Conclusion

The comprehensive test suite demonstrates that our low-level primitives (RoPE rotation and MPI gathering) are mathematically and distributionally correct. The ROPE_APPLICATION failure in the full pipeline must therefore stem from incorrect integration of these correct primitives within MPIAttentionKernel. The next investigation phase should focus on parameter validation, tensor shape verification, and step-by-step value comparison with PyTorch at the integration level.
