# Token Divergence Root Cause Analysis

**Date**: 2025-01-27  
**Status**: Investigation in Progress  
**Author**: David Sanftenberg

## Current Status

After implementing KVCacheProvider, we've made significant progress but still have token divergence starting at the first decode after prefill.

## Token Sequence Comparison

| Stage | PyTorch | Llaminar | Match? |
|-------|---------|----------|--------|
| Prefill result | 6 | 6 | ✅ YES |
| Decode step 0 (after token 6) | 25010 | 62162 | ❌ NO |
| Decode step 1 (after token 25010/62162) | 10 | 11 | ❌ NO |

## Key Findings

### 1. Attention Score Magnitudes are Correct
- **PyTorch** scores: `[90.346, 88.616, 87.401, 85.745, 85.104, 85.685]`
- **Llaminar** scores: `[90.909, 90.966, 91.520, 90.605, 90.042, 89.530]`
- Both in the same range (85-92), both use sqrt(head_dim) scaling ✅

### 2. One-Hot Softmax is Expected
- **PyTorch** softmax: `[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]` (one-hot)
- **Llaminar** softmax: `[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]` (one-hot)
- When scores have range ~5, exp(max)/exp(max-5) ≈ 148, naturally one-hot ✅

### 3. Both Attend to Same Position
- Both attend to position 0 (first token from prefill)
- Argmax of softmax matches ✅

### 4. BUT Score Magnitudes Differ
- PyTorch has more variation: 90.3 → 85.1 (range: 5.2)
- Llaminar has less variation: 91.5 → 89.5 (range: 2.0)
- This suggests Q or K vectors are different ❌

### 5. Accumulating Errors Through Layers

**Decode Step 0 (First Decode After Prefill)**:
- `Q_PROJECTION_layer0`: ✅ PASS (`max_abs=9.5e-07`, perfect!)
- `ATTENTION_OUTPUT_layer0`: ⚠️ Small error (`rel_l2=0.00635`)
- `Q_PROJECTION_layer3`: ❌ FAIL (`rel_l2=1.09`)
- `ATTENTION_NORM_layer5`: ❌ FAIL (`rel_l2=1.59`)
- `FINAL_NORM`: ❌ HUGE error (`rel_l2=10.366`, `max_abs=99.7`)
- `LM_HEAD`: ❌ HUGE error (`rel_l2=3.88`, `max_abs=18.9`)

Despite huge final errors, argmax still gives correct token 6 for prefill!

## Hypothesis: KV Cache Corruption During Prefill

The pattern suggests:
1. **Prefill works correctly** → generates token 6 ✅
2. **KV cache from prefill is corrupted** → used in first decode
3. **First decode starts with good Q** (`layer0 Q projection perfect`)
4. **But attention uses corrupted K cache** → produces slightly wrong attention output
5. **Errors accumulate** → by final layers, huge divergence
6. **Wrong token generated**: 62162 instead of 25010

## Evidence for Cache Corruption

1. **Q projection layer 0 is perfect** in first decode
   - This means the input (token 6 embedding) is correct
   - So the prefill OUTPUT is fine

2. **Attention scores differ** from PyTorch
   - PyTorch: `[90.346, 88.616, 87.401, 85.745, 85.104, 85.685]`
   - Llaminar: `[90.909, 90.966, 91.520, 90.605, 90.042, 89.530]`
   - Different dot products → Q or K must differ
   - Since Q is perfect, **K cache must be wrong**

3. **Errors accumulate layer by layer**
   - Layer 0: tiny error
   - Layer 3: moderate error
   - Layer 5: large error
   - Final: huge error
   - Classic error propagation through residual connections

## Possible Root Causes

### A. Cache Update During Prefill is Wrong
The `MPIAttentionKernel` might be updating the KV cache incorrectly during prefill. We need to verify:
- Does prefill populate ALL cache positions?
- Are cache positions correctly indexed?
- Is RoPE applied correctly to K before caching?

### B. Cache Retrieval During Decode is Wrong
The decode path might be reading cache incorrectly:
- Wrong indexing into cache?
- Wrong shape assumptions?
- Off-by-one errors in n_past?

### C. Cache Sharding Across MPI Ranks
Each rank owns a subset of KV heads. Possible issues:
- Wrong head indexing?
- Missing synchronization?
- Incomplete cache gathering?

## Next Steps

1. **Compare K cache values directly**
   - Capture K cache at end of prefill in both systems
   - Compare element-by-element
   - Find where they diverge

2. **Verify cache indexing**
   - Log cache write positions during prefill
   - Log cache read positions during decode
   - Check for off-by-one errors

3. **Check RoPE application**
   - Verify RoPE angles match PyTorch
   - Ensure RoPE applied before caching
   - Check position indices are correct

4. **Validate cache shapes**
   - Verify `[seq_len, local_kv_head_dim]` is correct
   - Check that all ranks have consistent shapes
   - Ensure no dimension mismatches

## Test Commands

```bash
# Run parity test with cache debugging
timeout 180 ./build/test_parity_framework \
  --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch" 2>&1 | \
  grep "K_VECTOR\|CACHE"

# Check K cache values
timeout 180 ./build/test_parity_framework \
  --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch" 2>&1 | \
  grep "local_k_cache first 10"

# Compare attention scores
timeout 180 ./build/test_parity_framework \
  --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch" 2>&1 | \
  grep "SCORE_COMPUTE"
```

## Summary

The KV cache architecture fix successfully transfers cache from prefill to decode, but the cached K values appear to be incorrect. This causes attention scores to differ from PyTorch, leading to accumulating errors and token divergence starting at the first decode after prefill.

The issue is NOT:
- ❌ Missing sqrt(head_dim) scaling
- ❌ Wrong attention position selection  
- ❌ Softmax numerical instability
- ❌ Cache transfer mechanism

The issue IS likely:
- ✅ K cache values populated incorrectly during prefill
- ✅ Cache indexing or RoPE application bug
- ✅ MPI sharding issue with KV heads
