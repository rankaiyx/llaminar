# Next Steps for K Cache Debugging

## Summary of Evidence

1. ✅ **First token correct (6)**: Prefill output is correct
2. ❌ **Second token wrong (62162 vs 25010)**: First decode after prefill fails
3. ✅ **New Q/K/V perfect**: Token 6's Q/K/V projections match PyTorch exactly
4. ❌ **Attention output catastrophic (119% error)**: Layer 0 attention completely wrong
5. ✅ **Single-rank same error**: Not an MPI gather issue
6. ✅ **Cache size correct**: Cache grows 5→6→7→8 as expected

## Critical Realization

**The prefill LOGITS are correct (generate token 6), but the K/V CACHE from prefill is wrong!**

This means:
- During prefill, the forward pass computes correct attention → correct final output → correct token
- But the K/V values that get CACHED during prefill are somehow corrupted
- When decode tries to use that corrupted cache → wrong attention → wrong output

## Hypothesis: Cache Population Bug

During prefill, two things happen:
1. **Attention computation** (uses local K/V, computes scores, produces attention output)
2. **Cache population** (saves K/V for future use)

These might be using **different** K/V values!

### Evidence

Looking at `MPIAttentionKernel.cpp` prefill mode:
```cpp
// PREFILL MODE: Initialize cache with current K/V
attn_seq_len = seq_len;
local_k_cache = local_k; // Share the tensor (no copy needed)
local_v_cache = local_v;
```

Then later:
```cpp
// Copy to output tensor
memcpy(outputs[1]->data(), local_k_cache->data(), local_k_cache->size() * sizeof(float));
memcpy(outputs[2]->data(), local_v_cache->data(), local_v_cache->size() * sizeof(float));
```

So we're returning `local_k_cache` which is just a shared pointer to `local_k`.

But wait - **when is RoPE applied?**

Looking at the code flow:
1. Compute K projection → `local_k` (pre-RoPE)
2. Apply RoPE → modifies `local_k` **in-place**
3. Set cache → `local_k_cache = local_k` (post-RoPE)
4. Compute attention using `local_k` (post-RoPE) ✅
5. Return `local_k_cache` (post-RoPE) ✅

So that SHOULD be correct...

### Wait - Check Gather Order!

Actually, looking more carefully at the code:

```cpp
// STEP 4: Gather Q/K/V for snapshotting (BEFORE RoPE!)
// CRITICAL: PyTorch captures Q_PROJECTION/K_PROJECTION BEFORE RoPE is applied
```

So we gather for snapshots BEFORE RoPE. That's correct for matching PyTorch.

```cpp
// STEP 5: Apply RoPE to Q and K (AFTER snapshotting but BEFORE attention!)
```

Then we apply RoPE to `local_q` and `local_k`.

```cpp
// STEP 5.1: UPDATE KV CACHE (after RoPE, before attention)
if (is_decode_mode) {
    // Append new K/V...
} else {
    // PREFILL MODE
    local_k_cache = local_k; // Share the tensor
}
```

So in prefill, we set `local_k_cache = local_k` AFTER RoPE. That should be correct.

But wait - is `local_k` the LOCAL portion (per MPI rank) or the GLOBAL (gathered) version?

Looking at variable definitions:
- `local_k` is created at the start: `TensorFactory::create_simple({seq_len, local_kv_head_dim})`
- Where `local_kv_head_dim = (n_head_kv_ / world_size) * head_dim_`

So `local_k` contains ONLY this rank's KV heads, not all heads!

## THE BUG!

During prefill:
1. Each rank computes its local K projection (subset of KV heads)
2. Each rank applies RoPE to its local K
3. Each rank sets `local_k_cache = local_k` (LOCAL, not global!)
4. Each rank returns `local_k_cache` which contains ONLY local heads

During decode:
1. Each rank receives its own `local_k_cache` from previous step
2. Kernel gathers all ranks' caches via `MPI_Allgatherv` → creates `global_k_cache`
3. Attention uses `global_k_cache` (all KV heads)

**But during PREFILL**, attention also needs all KV heads! Let me check what happens:

Looking at prefill attention code... searching for how K is used in prefill...

Actually, let me check if there's GQA expansion during prefill too:

```cpp
if (n_head_ != n_head_kv_)
{
    // GQA: replicate K/V heads from CACHE to match Q head count
    // Use cache (which contains all past tokens + current token)
    local_k_expanded = TensorFactory::create_simple({attn_seq_len, local_head_dim});
    
    if (world_size > 1)
    {
        // Need to gather full cache from all ranks for GQA expansion
        global_k_cache = TensorFactory::create_simple({attn_seq_len, k_v_dim});
        ...
        MPI_Allgatherv(local_k_cache->data(), ...)
    }
}
```

So even in prefill, if we have GQA (n_head != n_head_kv), we need to gather!

And Qwen DOES use GQA! From config: `n_head=14, n_head_kv=2`.

So during prefill:
1. Each rank has `local_k` (1 KV head out of 2 total, since world_size=2)
2. We need to GATHER to get all 2 KV heads
3. Then expand to 14 Q heads
4. Then compute attention

**Is this gather happening during prefill?**

Let me check... the GQA code path should trigger in both prefill and decode...

Actually yes! The code doesn't distinguish between prefill/decode for GQA expansion. It always gathers if `world_size > 1` and `n_head_ != n_head_kv_`.

So that should be fine...

## Alternative Theory: Position Indexing

Maybe the issue is that RoPE positions are wrong?

During prefill, tokens are at positions 0,1,2,3,4.
During decode, new token is at position 5.

But maybe we're using wrong positions? Let me check what `n_past_` is during prefill:

```cpp
const bool is_decode_mode = (n_past_ > 0);
```

So during prefill, `n_past_ = 0`. That's correct.

Then when computing RoPE:
```cpp
apply_rope(local_q->data(), local_k->data(), seq_len, head_dim_, 
           local_heads, local_kv_heads, n_past_, rope_freq_base_);
```

So `n_past=0` for prefill. Inside `apply_rope`, positions would be `0, 1, 2, 3, 4` for the 5 tokens. That's correct!

## I'm Stuck

I can't find the bug just by code review. We need empirical debugging:

1. **Add K cache value logging**: Log the actual K cache values being saved during prefill
2. **Compare with PyTorch**: Modify PyTorch script to save K cache after prefill
3. **Element-by-element comparison**: Find first divergent value
4. **Trace backwards**: From divergent value, figure out which step introduced the error

## Action Plan

1. Modify `generate_incremental_decode_snapshots.py` to save prefill cache
2. Run PyTorch to get reference K cache
3. Run Llaminar with detailed cache logging
4. Compare and identify divergence point
