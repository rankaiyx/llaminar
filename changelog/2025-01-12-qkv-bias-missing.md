# Q/K/V Bias Parameters Missing - Root Cause of Parity Failure

**Date**: January 12, 2025  
**Issue**: OpenBLAS vs PyTorch prefill parity test fails immediately at Q_PROJECTION_layer0  
**Status**: ROOT CAUSE IDENTIFIED  

## Problem Summary

The parity test shows massive divergence starting at Q_PROJECTION_layer0:
- **max_abs**: 71.9062 (extremely high)
- **rel_l2**: 0.976415 (complete mismatch)

All subsequent stages fail because this error compounds through the network.

## Investigation Timeline

### Initial Hypothesis (WRONG)
Suspected GGUF weights were stored transposed relative to PyTorch layout.

**Evidence Against**:
- Weight verification passes perfectly for all 24 layers
- Changing weight contracts broke verification (rel_l2=1.3)
- Data layout matches exactly between GGUF and PyTorch

### Data Layout Verification (CORRECT)
Verified that weight data is correctly loaded:
```
Rank 0 Q weight [0,:5]: [-0.00227356, -0.00500488, 0.0187988, 0.0124512, 0.00408936]
PyTorch expected:       [-0.00227356, -0.00500488, 0.01879883, 0.01245117, 0.00408936]
✓ MATCH
```

### Matrix Multiplication Analysis
```cpp
// Current Llaminar code:
Q = input @ W_q^T  // No bias

// PyTorch Linear layer:
Q = input @ W_q^T + b_q  // WITH BIAS
```

### Root Cause Discovery

**GGUF Model HAS Bias Parameters**:
```
blk.0.attn_q.bias: shape=[896]
blk.0.attn_k.bias: shape=[128]  
blk.0.attn_v.bias: shape=[128]
... (all 24 layers)
```

**PyTorch Uses Biases**:
```python
q_proj.bias[:5] = [-0.0150, 0.0255, -0.1035, -0.1357, -14.4375]
```

**Verification**:
```python
# WITHOUT bias:
manual_q = input @ W_q.T
# Result: [0.09756, 0.28886, 0.14250, ...]
# Doesn't match PyTorch

# WITH bias:
manual_q = input @ W_q.T + b_q
# Result: [0.08260687, 0.31437594, 0.03898931, -0.8742539, -15.520945]
# Matches PyTorch: max_diff=0.002, rel_l2=1.4e-5 ✓ PERFECT
```

## Impact

**Missing Biases Affect**:
1. ✗ Q projection (n_head*head_dim = 896 elements)
2. ✗ K projection (n_kv_head*head_dim = 128 elements)
3. ✗ V projection (n_kv_head*head_dim = 128 elements)
4. Possibly O projection (needs verification)
5. Possibly MLP layers (needs verification)

**Cascading Failures**:
- Q/K mismatch → Attention scores wrong
- Attention scores wrong → Softmax wrong
- Softmax wrong → Context vectors wrong
- All subsequent layers accumulate error
- Final output completely diverges

## Required Fixes

### 1. Add Bias Weight Contracts

File: `src/weight_contracts.h`

Add bias contracts for Q/K/V projections:
```cpp
// After existing Q/K/V weight contracts (~line 785):
WeightShapeContract("blk.{layer}.attn_q.bias", {"n_head*head_dim"},
                   WeightRole::BIAS, WeightSlicing::ROW_SLICED),
WeightShapeContract("blk.{layer}.attn_k.bias", {"n_kv_head*head_dim"},
                   WeightRole::BIAS, WeightSlicing::ROW_SLICED),
WeightShapeContract("blk.{layer}.attn_v.bias", {"n_kv_head*head_dim"},
                   WeightRole::BIAS, WeightSlicing::ROW_SLICED),
```

### 2. Update MPIAttentionKernel

File: `src/kernels/MPIAttentionKernel.cpp`

Current code uses `matmul_with_bias()` but passes `nullptr` for bias:
```cpp
// Line ~1000:
matmul_with_bias(input->data(), local_wq->data(), local_q->data(),
                local_bq ? local_bq->data() : nullptr,  // Currently nullptr!
                seq_len, local_head_dim, d_model, "Q_projection");
```

Need to:
1. Load bias tensors from model
2. Slice them appropriately for MPI ranks
3. Pass actual bias data to `matmul_with_bias()`

### 3. Verify Other Projections

Check if these also need bias:
- `attn_output` (O projection)
- `ffn_up` projection
- `ffn_down` projection  
- `ffn_gate` projection

### 4. Update Test Infrastructure

Weight verifier needs to know about biases to properly validate slicing.

## Verification Plan

1. ✓ Confirm GGUF has biases (DONE)
2. ✓ Verify PyTorch uses biases (DONE)
3. ✓ Prove bias fixes the math (DONE - manual computation matches)
4. Add bias contracts
5. Update ModelLoader to load biases
6. Update MPIAttentionKernel to use biases
7. Rerun parity test - expect Q_PROJECTION_layer0 to PASS
8. Check subsequent stages for additional issues

## Timeline Estimate

- Bias contract implementation: 30 minutes
- ModelLoader updates: 30 minutes
- Kernel updates: 1 hour
- Testing and debugging: 1-2 hours

**Total**: 3-4 hours to full parity (optimistic)

## Notes

- This explains why weight verification passed but matmul failed
- The weights themselves are correct
- The computation is missing a critical component
- Classic case of "the code does what I wrote, not what I meant"
- Should have checked HuggingFace model structure more carefully upfront

## References

- PyTorch Linear layer: `torch.nn.Linear(in_features, out_features, bias=True)`
- GGUF spec includes bias tensors
- Qwen2.5 architecture uses biases throughout (unlike LLaMA which often omits them)
