# RoPE Investigation Summary - 2025-01-27

## Bottom Line

**RoPE implementation is CORRECT**. The token divergence is caused by a **K_PROJECTION mismatch**, not RoPE.

## Evidence

### 1. RoPE IS Working

Test output shows RoPE correctly modifying values in layers 1+:
```
Layer 1 PRE_ROPE_Q:  [0.134055, 0.363254, 0.015141, ...]
Layer 1 POST_ROPE_Q: [0.242447, -0.317964, -0.91603, ...]  ← DIFFERENT
```

### 2. Position 0 Identity is Correct

Layer 0 shows identical PRE/POST values at position 0, which is mathematically correct:
- At position 0: angle = 0 * inv_freq = 0
- cos(0) = 1, sin(0) = 0
- Rotation becomes identity: x' = x * 1 - y * 0 = x

```
Layer 0 PRE_ROPE_K:  [-8.65756, -4.09954, -6.18955, 0.677311, ...]
Layer 0 POST_ROPE_K: [-8.65756, -4.09954, -6.18955, 0.677311, ...]  ← SAME (correct!)
```

### 3. K_PROJECTION Mismatch Identified

Comparing Llaminar's PRE_ROPE_K (which equals POST_ROPE_K at pos=0) with PyTorch's K_CACHE (which also equals K_PROJECTION at pos=0):

```
PyTorch  K_CACHE[pos=0]:  [-8.637, -4.111, -6.199,  0.705,  0.004, ...]
Llaminar PRE_ROPE_K[pos=0]: [-8.658, -4.100, -6.190,  0.677, -0.002, ...]
Differences:                [ 0.021,  0.011,  0.010,  0.028,  0.006, ...]
```

**These ~0.02 differences mean the K_PROJECTION itself is producing different values!**

## Root Cause Chain

1. K_PROJECTION outputs differ by ~0.02 between PyTorch and Llaminar
2. These values pass through RoPE (which works correctly)
3. Wrong K values get cached
4. During decode, wrong cached K → wrong attention → wrong logits → wrong tokens

## Investigation Points

The K_PROJECTION differences could be caused by:

### A. Quantization
- Model uses Q4_0 quantization
- Possible dequant differences between PyTorch and Llaminar
- Expected magnitude: ~0.01-0.03 ✓ (matches observed)

### B. Weight Loading/Sharding
- MPI distributes weights across ranks
- Possible transpose or stride issues
- Would affect all positions uniformly ✓ (matches pattern)

### C. Matrix Multiplication Backend
- COSMA vs OpenBLAS numerical differences
- Different accumulation order
- Unlikely to cause systematic 0.02 offset

### D. Input Mismatch
- Hidden states feeding into K_PROJECTION might differ
- Would need to verify prefill input matches PyTorch

## Next Debugging Steps

### Option 1: Verify Input to K Projection
```bash
# Compare hidden states (input) between PyTorch and Llaminar
# If inputs match, issue is in projection itself
# If inputs differ, backtrack to previous layer
```

### Option 2: Check K Weight Quantization
```bash
# Compare K weight matrix values after dequant
# PyTorch dequant vs Llaminar dequant
# Look for systematic bias in dequant output
```

### Option 3: Test with F32 Weights
```bash
# Convert model to FP32 to eliminate quantization variable
# If problem persists → weight loading issue
# If problem disappears → quantization issue
```

### Option 4: Add K_PROJECTION Debug Logging
```cpp
# Log first few elements of:
# - Input hidden states
# - K weight matrix (after any dequant)
# - K projection output
# Compare element-by-element with PyTorch
```

## Why printf Debugging Failed

Multiple attempts to add printf/fprintf to `attention_primitives.cpp` produced no output:
- MPI stdout/stderr redirection
- OpenMP parallel regions
- Build/linking issues

**Recommendation**: Use existing LOG_DEBUG infrastructure or add logging BEFORE OpenMP regions.

## Recommendation

**Focus next on K weight dequantization** since:
1. Model is quantized (Q4_0)
2. Observed differences (~0.02) match typical quant errors
3. Would affect all positions uniformly (matches pattern)
4. Can be tested by comparing weight values directly

Check `LLAMINAR_DEQUANT_STATS` and `LLAMINAR_DEQUANT_ANOMALIES` environment flags to see if any warnings were logged during weight loading.

