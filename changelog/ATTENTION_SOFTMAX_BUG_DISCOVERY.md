# Critical Discovery: Llaminar Attention Softmax Bug

## Date
2025-10-11

## Summary
Found the root cause of token divergence between PyTorch and Llaminar during incremental decode.

## The Bug

**Llaminar's attention softmax is producing one-hot vectors instead of probability distributions.**

### Evidence

For token_0 (first decode step after prefill), layer 0, head 0:

**PyTorch (CORRECT)**:
```
ATTENTION_SCORES: Not captured in incremental mode (inside HuggingFace model)
ATTENTION_SOFTMAX: [0.188, 0.199, 0.347, 0.139, 0.079, 0.047]
Sum: 1.000 ✓ (proper probability distribution)
```

**Llaminar (BUG)**:
```
ATTENTION_SCORES: [0.0, 0.0, 0.0, 0.0, 0.0, 89.53]
ATTENTION_SOFTMAX: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
Sum: 1.000 (technically valid, but wrong distribution!)
```

The softmax is mathematically correct (exp([0,0,0,0,0,89.5]) → [0,0,0,0,0,1]), but the INPUT scores are wrong.

## Root Cause Analysis

The attention scores before softmax show only the LAST position has a non-zero value:
- Positions 0-4: score = 0.0
- Position 5: score = 89.53

This suggests **Q·K^T is only being computed for one position**, likely due to:

1. **KV Cache Bug**: The cached K vectors from prefill might not be accessible during decode
2. **Attention Computation Bug**: The Q·K^T might only compute against the most recent K
3. **Masking Bug**: Overly aggressive causal masking might be zeroing out valid positions

## Impact

When attention only considers the last token:
- Context from earlier tokens is ignored
- The model essentially becomes "memoryless" during decode
- Different hidden states → different outputs → token divergence

Layer 0 attention output differs by 1.3% (rel_l2=0.0135), which then cascades through 23 more layers, eventually causing completely different token predictions:
- PyTorch: `6 → 25010 → 10`  
- Llaminar: `6 → 13956 → 99822`

## Investigation Steps

### Confirmed Working
- ✅ Embedding layer: perfect match when embedding same token
- ✅ Layer 0 Q/K/V projections: match to machine precision  
- ✅ Layer 0 attention norm: match to machine precision
- ✅ Softmax implementation: correctly computes exp/normalize
- ✅ PyTorch attention: proper probability distribution

### Confirmed Broken
- ❌ Llaminar attention scores: only last position non-zero
- ❌ Llaminar attention softmax: one-hot vectors
- ❌ Llaminar attention output: 1.3% error that cascades

## Next Steps

1. **Check KV cache during incremental decode**
   - Verify cache is populated from prefill
   - Verify cache is accessible during decode
   - Check cache indexing/addressing

2. **Trace Q·K^T computation**
   - Check if compute_qk_scores uses full cache
   - Verify seq_len vs attn_seq_len parameters
   - Check if GQA K expansion is correct

3. **Review causal masking**
   - Ensure mask doesn't zero valid positions
   - Check mask shape during incremental decode

## Code Locations

- Llaminar softmax: `src/kernels/MPIAttentionKernel.cpp:1780-1830`
- Score computation: `llaminar::attn::compute_qk_scores()`
- Snapshot capture: `MPIAttentionKernel.cpp:1747, 1827`

---

**Status**: Bug identified, root cause under investigation
