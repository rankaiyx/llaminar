# KV Cache Architecture Issue Discovery

**Date:** 2025-01-27  
**Author:** Claude (via David Sanftenberg)  
**Issue:** Token divergence during incremental decode due to empty KV cache

## Executive Summary

Discovered a critical architectural issue where the KV cache populated during prefill is never passed to the incremental decode path, resulting in all-zero attention scores during decode and complete token divergence.

## Root Cause Analysis

### The Problem
During incremental decode, the attention mechanism produces one-hot softmax outputs `[1, 0, 0, 0, 0, 0]` instead of proper distributions like `[0.188, 0.199, 0.347, 0.139, 0.079, 0.047]` because the KV cache is all zeros.

### Investigation Trail
1. **Symptom**: Attention softmax produces one-hot vectors during decode
2. **Discovery 1**: Attention scores before softmax are `[0, 0, 0, 0, 0, 89.5]` - only last position non-zero
3. **Discovery 2**: Causal masking had coordinate system bug (comparing relative vs absolute positions) - **FIXED**
4. **Discovery 3**: Even with correct masking, scores remain zero for past positions
5. **Discovery 4**: K cache values during decode are all zeros!
   ```
   K_VECTOR h=0, j=0: 0 0 0 0 0 0 0 0 0 0
   K_VECTOR h=0, j=1: 0 0 0 0 0 0 0 0 0 0
   ...
   K_VECTOR h=0, j=5: 3.54468 5.04721 0.409897 6.8323...  (current token only!)
   ```

### Architectural Split

The codebase has **TWO SEPARATE EXECUTION PATHS**:

#### Prefill Path
- Uses `PrefillProviderFactory` → `PrefillProvider`
- Located in `src/prefill_provider*.{h,cpp}`
- Returns logits via provider
- **Populates KV cache** but through a different mechanism

#### Decode Path  
- Uses `QwenPipeline::execute()` → `MPIAttentionKernel`
- Standard transformer block processing
- Expects pre-populated KV cache in `k_cache_[layer_idx]` and `v_cache_[layer_idx]`
- **Never receives the cache from prefill!**

### The Disconnect

```
PREFILL:
  PrefillProvider::execute() 
    ↓ (populates cache internally)
    ↓ (cache NOT returned to QwenPipeline)
  QwenPipeline::k_cache_[0...N] = ???  ← **NEVER UPDATED!**

DECODE:
  QwenPipeline::execute()
    ↓ (reads k_cache_[0...N])
    ↓ (finds all zeros!)
  MPIAttentionKernel receives empty cache
    ↓
  Attention scores = [0,0,0,0,0,X]  ← **BUG!**
```

## Fixes Applied

### 1. Causal Mask Coordinate System Bug ✅
**File:** `src/kernels/common/attention_primitives.cpp:274-283`

**Before:**
```cpp
// Incorrect - comparing relative query position with absolute cache position
if (causal && j > i)
```

**After:**
```cpp
// Calculate absolute query position for KV cache scenarios
int abs_q_pos = (k_seq_len - q_seq_len) + i;
// Correct - compare absolute positions
if (causal && j > abs_q_pos)
```

**Impact:** Fixes causal masking for decode scenarios where `k_seq_len != q_seq_len`.

### 2. COSMA Path KV Cache Return ✅
**File:** `src/qwen_pipeline.cpp:2312`

**Before:**
```cpp
std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out};  // Only 1 output!
```

**After:**
```cpp
std::vector<std::shared_ptr<TensorBase>> attn_outputs = {attn_out, nullptr, nullptr};

// Update KV cache with kernel outputs (same as OpenBLAS path!)
if (use_kv_cache_ && attn_outputs.size() >= 3)
{
    if (attn_outputs[1] && attn_outputs[2])
    {
        k_cache_[layer_idx] = attn_outputs[1];
        v_cache_[layer_idx] = attn_outputs[2];
    }
}
```

**Impact:** COSMA prefill path now properly returns and stores KV cache (though currently unused since COSMA is disabled in test).

## Critical Remaining Issue ⚠️

### **PrefillProvider Does Not Populate QwenPipeline's KV Cache**

The `PrefillProvider` system manages its own internal cache state but never updates `QwenPipeline::k_cache_` and `QwenPipeline::v_cache_` vectors that the decode path depends on.

**Required Fix:**
The `PrefillProvider::execute()` method must:
1. Populate the KV cache during prefill
2. **Return or transfer** the cache to `QwenPipeline::k_cache_[0...N]` and `v_cache_[0...N]`

**Suggested Approach:**
```cpp
// In QwenPipeline::prefill() after provider->execute():
bool success = provider->execute(tokens, weights_iface, output, ctx, metrics);

// ADD THIS:
if (success && use_kv_cache_)
{
    // Get cache from provider and store in pipeline
    provider->getKVCache(k_cache_, v_cache_);  // NEW METHOD NEEDED!
}
```

Or alternatively, have PrefillProvider directly populate the pipeline's cache vectors.

## Test Results

**Before Fixes:**
- ✗ Token divergence on all decode tokens
- ✗ 580/585 stages failed
- ✗ Generated tokens: 6 → 13956 → 99822 (vs PyTorch: 6 → 3290 → 30337)

**After Causal Mask + COSMA Cache Fixes:**
- ✗ Still failing (PrefillProvider doesn't populate cache)
- ✗ 580/585 stages failed
- ✗ Same token divergence

**Root Issue:** The architectural split between prefill and decode paths means cache is never transferred.

## Next Steps

1. **URGENT:** Implement KV cache transfer from PrefillProvider to QwenPipeline
   - Add `PrefillProvider::getKVCache()` method
   - Or have provider directly write to pipeline's cache vectors
   - Update all provider implementations (OpenBLAS, COSMA)

2. **Test:** Verify cache has values during decode
   - Add logging in `MPIAttentionKernel` to confirm K cache non-zero
   - Check that attention scores distribute properly

3. **Validate:** Run `ParityFramework.TrueIncrementalDecodeVsPyTorch` test
   - Should pass with proper KV cache population

4. **Clean up:** Remove excessive debug logging added during investigation
   - `CACHE_DEBUG`, `CACHE_UPDATE_DEBUG`, `Q_VECTOR`, `K_VECTOR`, etc.

## Files Modified

### Core Fixes
- `src/kernels/common/attention_primitives.cpp` - Causal mask coordinate system
- `src/qwen_pipeline.cpp` - COSMA path cache return

### Debug Logging (to be cleaned up)
- `src/kernels/MPIAttentionKernel.cpp` - Extensive cache and vector logging
- `src/qwen_pipeline.cpp` - Cache update condition logging

## Lessons Learned

1. **Follow the data flow end-to-end** - The prefill→decode split was non-obvious
2. **Check ALL code paths** - PrefillProvider vs execute() were completely separate
3. **Trust the evidence** - Cache was zero because it was never populated, not corruption
4. **Architecture matters** - Dual execution paths created integration gap

## References

- Issue tracked in conversation with 100K+ tokens of investigation
- Related: COSMA integration, prefill optimization, incremental decode
- Test: `tests/test_parity_framework.cpp::TrueIncrementalDecodeVsPyTorch`
