# Decode Path Divergence Analysis

## Summary

After fixing the EMBEDDING shape mismatch, we now have clear evidence about where the PyTorch and Llaminar systems diverge during incremental decode.

## Key Findings

### 1. Embedding Layer: PERFECT MATCH ✅

```
token_0: EMBEDDING.npy (max_abs=0, rel_l2=0)
```

When both systems embed the **same token** (token 6), they produce **identical** embeddings. This conclusively proves:
- ✅ Embedding weights load correctly in both systems
- ✅ Embedding layer computation is identical
- ✅ No precision issues in embedding lookup

### 2. Layer 0 Inputs: ESSENTIALLY PERFECT ✅

```
Q_PROJECTION_layer0.npy (max_abs=9.5e-07, rel_l2=9.9e-08)
K_PROJECTION_layer0.npy (max_abs=7.6e-06, rel_l2=6.8e-07)
V_PROJECTION_layer0.npy (max_abs=1.9e-08, rel_l2=5.0e-09)
ATTENTION_NORM_layer0.npy (max_abs=5.9e-08, rel_l2=5.2e-09)
```

The QKV projections for layer 0 match to machine precision. This proves:
- ✅ Weight matrices for QKV projections load correctly
- ✅ Matrix multiplication produces identical results
- ✅ RMSNorm computation is identical

### 3. Layer 0 Output: FIRST DIVERGENCE ⚠️

```
ATTENTION_OUTPUT_layer0.npy FAILED (max_abs=0.314, rel_l2=0.0135)
```

Despite perfect inputs, the attention output diverges by ~1.3%. This is the **root cause** of the downstream divergence.

### 4. Cascading Divergence 🌊

After layer 0's small error, subsequent layers amplify the divergence:

```
Layer 0:  rel_l2 = 0.0135 (1.3%)
Layer 1:  rel_l2 = 0.106  (10.6%)  
Layer 2:  rel_l2 = 0.081  (8.1%)
...
Final:    rel_l2 = 10.1   (1010%)  [FINAL_NORM]
LM_HEAD:  rel_l2 = 6.44   (644%)
```

By the time we reach the final layer, the accumulated error is so large that the systems sample completely different tokens:
- PyTorch: `6 → 25010 → 10`
- Llaminar: `6 → 13956 → 99822`

## Root Cause Hypothesis

The divergence starts in **Layer 0's attention mechanism**, specifically between these perfectly matching inputs:
- Q, K, V projections (✅ match)
- Attention norm (✅ match)

And this output:
- Attention output (❌ 1.3% error)

Possible causes:
1. **Attention score computation** - Different implementations of Q @ K^T
2. **Softmax precision** - Numerical stability differences
3. **KV cache access** - Incremental decode uses cached K/V from prefill
4. **Scaling factors** - sqrt(head_dim) or temperature scaling
5. **GQA implementation** - Grouped query attention key/value expansion

## Next Steps

To isolate the exact divergence point, we need to capture intermediate attention stages:
1. **Attention scores** (before softmax)
2. **Attention weights** (after softmax)
3. **Attention context** (before output projection)

This will pinpoint whether the issue is in:
- Score computation
- Softmax numerical precision
- Context aggregation
- Output projection (though weights should match)

## Prefill Parity Status

User's claim: "prefill parity matches" - **needs verification**

If prefill truly matches, then:
- ✅ All weights load correctly (confirmed by layer 0 projections)
- ✅ Batch processing works correctly
- ❌ Incremental decode has different behavior

The divergence could be in:
- KV cache management during decode
- Positional encodings for incremental positions
- Attention mask handling (causal mask with cache)

## Recommendations

1. **Add attention score/softmax snapshots** to both systems
2. **Verify prefill parity** with explicit test
3. **Check KV cache** contents between systems
4. **Compare positional encoding** for decode steps
5. **Review attention mask** construction for incremental decode

---

*Generated: After EMBEDDING shape fix and layer-by-layer divergence analysis*
