# Token Divergence Root Cause - Deep Dive

**Date**: 2025-01-27  
**Status**: Active Investigation - K Cache Corruption Suspected  
**Author**: David Sanftenberg

## Executive Summary

After implementing KVCacheProvider successfully, we have:
- ✅ First generated token (6) matches PyTorch
- ✅ K/V cache architecture working (cache transfers from prefill to decode)
- ❌ **Massive logit divergence** starting at first decode after prefill

**Key Finding**: Attention output in layer 0 of first decode has **119% relative L2 error**, indicating fundamental corruption in K or V cache from prefill.

## Detailed Analysis

### Verified Components ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| Q Projection (new token) | **PERFECT** | max_abs < 1e-6 |
| K Projection (new token) | **PERFECT** | max_abs < 1e-6 |
| V Projection (new token) | **PERFECT** | max_abs < 1e-6 |
| RoPE Implementation | Correct | Matches HuggingFace pattern |
| Cache Transfer Mechanism | Working | Cache preserved across decode steps |
| Cache Append Logic | Working | Size grows correctly 5→6→7→8 |

### Divergence Magnitude 💥

**PyTorch vs Llaminar Logits (Decode Step 0 - First Decode After Prefill)**:

| Metric | PyTorch | Llaminar | Notes |
|--------|---------|----------|-------|
| **Top predicted token** | 25010 | 62162 | Completely different! |
| **Top logit value** | 16.36 | 10.73 | Different magnitudes |
| **Expected token logit** | 16.36 (token 25010) | 0.59 (token 25010) | **-15.77 difference!** |

### Layer 0 Attention Output Comparison

```
PyTorch ATTENTION_CONTEXT_0:
  Shape: (1, 1, 896)
  First 10: [-0.0147, -0.0158, 0.0185, 0.0133, -0.0009, 0.0072, 0.0113, 0.0192, -0.0376, 0.0296]
  Stats: mean=0.0007, std=0.0219, range=[-0.0718, 0.1636]

Llaminar ATTENTION_OUTPUT_layer0:
  Shape: (1, 896)
  First 10: [0.0043, -0.0010, 0.0143, 0.0064, -0.0010, -0.0101, -0.0025, 0.0152, -0.0015, -0.0028]
  Stats: mean=0.0007, std=0.0143, range=[-0.0318, 0.3219]

ERROR METRICS:
  Relative L2: 1.19 (119%!) ❌
  Max absolute: 0.332
  Close (rtol=1e-2)? FALSE
```

**This is CATASTROPHIC!** Layer 0 should be nearly perfect, but instead we have >100% error.

## K Cache Investigation

### Cache Values After Prefill

```
[CACHE_DEBUG] Prefill K cache (first 10 of first 3 rows):
  Row 0: -8.65756 -4.09954 -6.18955 0.677311 -0.00199093 9.08512 8.13929 -1.07392 -0.208544 -0.420254
  Row 1: -9.976 -8.44164 -7.02504 2.74511 0.0201734 9.52927 8.11122 -1.5855 -0.17186 -0.426768
```

### Cache Values During First Decode (Before Attention)

```
[RANK=0] After GQA expansion (using cache):
  attn_seq_len=5 (n_past + seq_len)
  K_expanded shape: [5, 896]
  K_expanded[0,0:5]: -8.65756 -4.09954 -6.18955 0.677311 -0.00199093
  K_expanded range: [-130.172, 121.13]
```

**OBSERVATION**: The cache values are consistent between prefill and decode! Row 0 starts with the same values.

### K Projection for New Token (Token 6)

```
PyTorch K_PROJECTION_0 (decode_step_0):
  Shape: (1, 1, 128)
  First 10: [-8.563963, -4.202896, -6.243303, 0.7566312, -0.05496807, 9.0412, ...]

Llaminar K_PROJECTION_layer0:
  Shape: (1, 128)
  First 10: [-8.563963, -4.202896, -6.243303, 0.75663126, -0.05496807, 9.041201, ...]

Difference: < 1e-6 ✅ PERFECT MATCH
```

## Hypothesis: PyTorch Cache Values Different from Llaminar

Since all NEW computations (Q/K/V for token 6) are perfect, the ONLY remaining possibility is that the **cached K and V values from prefill** are different between PyTorch and Llaminar.

### Theory 1: RoPE Angle Mismatch
- **Evidence**: RoPE uses position-dependent angles
- **Problem**: Maybe Llaminar uses wrong positions during prefill?
- **Test**: Compare actual RoPE angles used

### Theory 2: MPI Gather Layout Issue
- **Evidence**: Cache is gathered from multiple ranks via `MPI_Allgatherv`
- **Problem**: Maybe gather produces wrong layout?
- **Test**: Run single-rank test (no MPI) to isolate

### Theory 3: PyTorch Doesn't Cache Post-RoPE
- **Evidence**: PyTorch captures "K_PROJECTION" (pre-RoPE) in snapshots
- **Problem**: Maybe PyTorch caches pre-RoPE and re-applies RoPE during decode?
- **Test**: Check if PyTorch applies RoPE to cache during decode

## Next Investigation Steps

### Immediate Priority: Compare Actual Cache Values

We need to answer: **What K values does PyTorch use for attention during first decode?**

PyTorch doesn't save the K cache directly, but we can infer it:

1. **During prefill**: PyTorch computes K_PROJECTION for tokens 0-4
2. **Applies RoPE**: Creates rotated K values
3. **Caches**: Stores post-RoPE K values
4. **During decode**: Uses cached K values WITHOUT re-applying RoPE

To get PyTorch's cached K:
- Load `pytorch_snapshots_mapped/prefill/K_PROJECTION_0.npy` (if exists)
- OR capture new PyTorch snapshot with explicit cache values
- OR modify PyTorch script to save cache after prefill

### Test 1: Single-Rank Execution

```bash
# Run with 1 MPI rank to eliminate gather issues
mpirun -np 1 ./build/test_parity_framework \
  --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch"
```

If single-rank still has errors → MPI gather is NOT the issue.

### Test 2: Check RoPE Angles

Add logging to capture actual RoPE angles used:
```cpp
// In attention_primitives.cpp, apply_rope_to_head()
if (rank == 0 && position < 3) {
    LOG_INFO("[RoPE Debug] position=" << position 
             << " i=" << i 
             << " angle=" << angle 
             << " cos=" << cos_angle 
             << " sin=" << sin_angle);
}
```

Compare angles with PyTorch's expected values.

### Test 3: Modify PyTorch to Save Cache

Update `generate_test_snapshots.py` to explicitly save K/V cache after prefill:

```python
# After prefill, before first decode
if past_key_values is not None:
    for layer_idx, (k_cache, v_cache) in enumerate(past_key_values):
        np.save(f"pytorch_snapshots_mapped/prefill_cache/K_CACHE_layer{layer_idx}.npy", 
                k_cache.detach().cpu().numpy())
        np.save(f"pytorch_snapshots_mapped/prefill_cache/V_CACHE_layer{layer_idx}.npy", 
                v_cache.detach().cpu().numpy())
```

Then directly compare with Llaminar's cache.

### Test 4: Attention Primitive Unit Test

Create minimal test:
```cpp
// Use known Q, K, V values
// Compute attention output manually
// Compare with primitives::compute_qk_scores()
```

This isolates whether the attention computation itself is wrong.

## Critical Question

**Does PyTorch cache pre-RoPE or post-RoPE K/V values?**

From HuggingFace Transformers code pattern:
```python
query_states, key_states = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
# ... then cache key_states and value_states
```

So PyTorch caches POST-RoPE values. This matches Llaminar's approach.

## Smoking Gun Test

The simplest test:
1. Extract Llaminar's K cache after prefill (already have this in logs)
2. Extract PyTorch's K cache after prefill (need to modify script)
3. Compare element-by-element
4. Find FIRST divergent element
5. Trace back to root cause

## Conclusion

We've narrowed the problem to K/V cache corruption during prefill. All evidence points to the cached values being wrong, not the cache transfer mechanism.

**Most Likely Root Cause**: RoPE position indexing during prefill. Llaminar might be using wrong position indices (e.g., starting from 1 instead of 0, or using MPI rank offsets incorrectly).

**Recommended Next Step**: Add detailed RoPE angle logging during prefill to verify positions match PyTorch's expectations.
