# Bias Parameter Implementation - Partial Progress

**Date**: January 12, 2025 (Evening Session)  
**Status**: IN PROGRESS - Biases loaded but not applied correctly  
**Test Result**: Q_PROJECTION_layer0 still fails (max_abs=71.9, rel_l2=0.976)

## Summary

Successfully identified that Q/K/V projection biases were missing as the root cause of parity failure. Implemented bias loading infrastructure, but biases are not yet being applied during matrix multiplication.

## What Works

1. ✅ **Weight Contracts** - Removed from system (biases use direct loading)
2. ✅ **Direct Bias Loading** - `loader.loadTensor("blk.{layer}.attn_{q,k,v}.bias")` works
3. ✅ **Bias Storage** - `weights.bq/bk/bv` vectors populated for all 24 layers
4. ✅ **Bias Values** - Loaded correctly (verified: layer 0 Q bias first=-0.0149536 matches PyTorch)
5. ✅ **Kernel Infrastructure** - MPIAttentionKernel already has full bias support (slicing, validation, application)

## What Doesn't Work

❌ **Bias Application** - Despite biases being loaded and passed to kernel, Q projection output still matches `input @ W^T` without the `+ bias` term.

## Investigation Findings

### Bias Loading Chain
```
loader.loadTensor("blk.0.attn_q.bias")  // ✅ Returns [896] tensor with correct values
  ↓
weights.bq.push_back(bq)                // ✅ Stored in weights struct
  ↓
attn_inputs = {..., weights.bq[0], ...} // ✅ Passed to kernel (position 5)
  ↓
auto bq_global = inputs[5]              // ✅ Extracted in kernel
  ↓
local_bq = ... (slicing logic)          // ❓ UNCERTAIN - may be nullptr
  ↓
matmul_with_bias(..., local_bq->data()) // ❓ UNCERTAIN - bias may not be applied
```

### Possible Issues

1. **`weights_are_sharded` flag** - Kernel may be taking wrong code path for bias handling
2. **Bias slicing** - Global bias may not be properly distributed to ranks
3. **Null check** - `local_bq` may be nullptr due to validation failure
4. **MPI rank mismatch** - Bias loading may fail on one rank but not the other

## Code Changes Made

### src/weight_contracts.h
- Initially added ROW_SLICED contracts for biases (FAILED - 2D slicing on 1D tensors)
- Changed to REPLICATED contracts (FAILED - validator still rejected)
- **FINAL**: Removed bias contracts entirely (direct loading doesn't need them)

### src/qwen_pipeline.cpp
- Kept direct `loader.loadTensor()` for biases (lines 1121-1126)
- Did NOT change kernel input structure (biases already being passed)

### src/kernels/MPIAttentionKernel.cpp
- NO CHANGES NEEDED - kernel already fully supports biases
- Bias extraction: line 437
- Bias validation: lines 526-559
- Bias slicing: lines 835-856
- Bias application: lines 1011-1013

## Next Steps (Priority Order)

1. **Add bias presence logging** - Verify `local_bq` is not nullptr
   ```cpp
   LOG_INFO("[BIAS_DEBUG] local_bq=" << (local_bq ? "PRESENT" : "nullptr") 
            << " size=" << (local_bq ? local_bq->size() : 0));
   ```

2. **Check `weights_are_sharded` path** - Determine which code branch is taken
   ```cpp
   LOG_INFO("[WEIGHT_PATH] weights_are_sharded=" << weights_are_sharded);
   ```

3. **Verify bias values in kernel** - Print first 5 bias values in kernel
   ```cpp
   if (local_bq) {
       LOG_INFO("[BIAS_VALUES] first 5: [" << local_bq->data()[0] << ", " 
                << local_bq->data()[1] << ", " << local_bq->data()[2] << "]");
   }
   ```

4. **Check matmul_with_bias implementation** - Verify bias is actually added
   ```cpp
   // In matmul_with_bias function (line 254):
   if (bias) {
       LOG_INFO("[MATMUL] Applying bias to output");
   } else {
       LOG_INFO("[MATMUL] NO BIAS - bias pointer is null");
   }
   ```

5. **Compare with/without bias** - Temporarily disable bias to confirm impact
   ```cpp
   // Force nullptr to see if output changes
   matmul_with_bias(..., nullptr /* local_bq->data() */, ...);
   ```

## Root Cause Hypothesis

Most likely: `weights_are_sharded=true` but biases are NOT pre-sharded (they're full [896] tensors), causing kernel to skip bias slicing logic and leaving `local_bq=nullptr`.

**Solution**: Either:
- A) Pre-slice biases before passing to kernel
- B) Force `weights_are_sharded=false` when biases are present
- C) Add special handling for REPLICATED biases in sharded weight path

## Time Spent

- Investigation: ~2 hours
- Implementation attempts: ~1.5 hours
- **Total**: ~3.5 hours

## Estimated Remaining Work

- Debug logging: 15 minutes
- Fix identification: 30 minutes
- Implementation: 30 minutes
- Testing: 30 minutes
- **Total**: ~2 hours to completion
