# MPI_Allgather Optimization Analysis - MPIAttentionKernel

**Date**: October 13, 2025  
**Author**: David Sanftenberg  
**Type**: Performance Analysis  
**Scope**: MPI communication overhead in attention computation

## Executive Summary

The MPIAttentionKernel performs **11 distinct MPI_Allgather/Allgatherv operations** per attention layer execution. This analysis identifies opportunities to reduce MPI overhead through caching, batching, and architectural changes.

## Current MPI_Allgather Inventory

### 1. Q/K/V Projection Gathering (Lines 1316-1380)
**Location**: After projection, before RoPE  
**Purpose**: Snapshot validation (Q_PROJECTION, K_PROJECTION, V_PROJECTION stages)  
**Operations**: 
- 3 × `seq_len` MPI_Allgather calls (one per timestep for Q, K, V)
- **Total**: `3 × seq_len` allgathers

```cpp
for (int t = 0; t < seq_len; ++t) {
    MPI_Allgather(local_q_row, local_head_dim, MPI_FLOAT, global_q_row, ...);  // Per token
}
for (int t = 0; t < seq_len; ++t) {
    MPI_Allgather(local_k_row, local_kv_head_dim, MPI_FLOAT, global_k_row, ...); // Per token
}
for (int t = 0; t < seq_len; ++t) {
    MPI_Allgather(local_v_row, local_kv_head_dim, MPI_FLOAT, global_v_row, ...); // Per token
}
```

**Impact**: For prefill with `seq_len=5`: **15 allgathers**  
**Impact**: For decode with `seq_len=1`: **3 allgathers**

---

### 2. Post-RoPE Q/K/V Gathering (Lines 1722-1742)
**Location**: After RoPE application  
**Purpose**: Snapshot validation (ROPE_APPLICATION stage)  
**Operations**:
- 3 × `seq_len` MPI_Allgather calls (one per timestep for Q, K, V after RoPE)
- **Total**: `3 × seq_len` allgathers

```cpp
for (int t = 0; t < seq_len; ++t) {
    MPI_Allgather(local_q_row, local_head_dim, MPI_FLOAT, global_q_row, ...);
    MPI_Allgather(local_k_row, local_kv_head_dim, MPI_FLOAT, global_k_row, ...);
    MPI_Allgather(local_v_row, local_kv_head_dim, MPI_FLOAT, global_v_row, ...);
}
```

**Impact**: For prefill with `seq_len=5`: **15 allgathers**  
**Impact**: For decode with `seq_len=1`: **3 allgathers**

---

### 3. KV Cache Gathering for GQA Expansion (Lines 1893-1911)
**Location**: Before GQA expansion  
**Purpose**: Gather full KV cache from all ranks for attention computation  
**Operations**:
- 1 × MPI_Allgather (metadata: sendcount exchange)
- 2 × MPI_Allgatherv (K and V cache data)
- **Total**: 3 allgathers

```cpp
MPI_Allgather(&sendcount_k, 1, MPI_INT, recvcounts_k.data(), ...);  // Metadata
MPI_Allgatherv(local_k_cache->data(), sendcount_k, MPI_FLOAT, global_k_cache->data(), ...);
MPI_Allgatherv(local_v_cache->data(), sendcount_k, MPI_FLOAT, global_v_cache->data(), ...);
```

**Impact**: **CRITICAL PATH** - Always executes for GQA attention  
**Volume**: `attn_seq_len × n_head_kv × head_dim` floats per gather (grows with context!)

---

### 4. Unmasked Attention Scores Gathering (Lines 2065-2077)
**Location**: After Q@K computation, before softmax  
**Purpose**: Snapshot validation (ATTENTION_SCORES stage)  
**Operations**:
- 1 × MPI_Allgather (metadata)
- 1 × MPI_Allgatherv (unmasked scores)
- **Total**: 2 allgathers

```cpp
MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), ...);
MPI_Allgatherv(unmasked_scores.data(), sendcount, MPI_FLOAT, global_scores.data(), ...);
```

**Impact**: For prefill with `seq_len=5`: Gathers `local_heads × 5 × 5 = local_heads × 25` floats  
**Impact**: For decode with `seq_len=1`: Gathers `local_heads × 1 × attn_seq_len` floats (grows with context!)

---

### 5. Softmax Scores Gathering (Lines 2209-2220)
**Location**: After softmax application  
**Purpose**: Snapshot validation (ATTENTION_SOFTMAX stage)  
**Operations**:
- 1 × MPI_Allgather (metadata)
- 1 × MPI_Allgatherv (softmax scores)
- **Total**: 2 allgathers

```cpp
MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), ...);
MPI_Allgatherv(scores.data(), sendcount, MPI_FLOAT, global_softmax.data(), ...);
```

**Impact**: Same volume as unmasked scores gather (duplicated data!)

---

### 6. Attended Output Gathering (Lines 2320-2332)
**Location**: After scores @ V computation  
**Purpose**: Snapshot validation (ATTENTION_CONTEXT stage)  
**Operations**:
- `seq_len` × MPI_Allgather calls (one per timestep)
- **Total**: `seq_len` allgathers

```cpp
for (int t = 0; t < seq_len; ++t) {
    MPI_Allgather(local_attended_row, local_head_dim, MPI_FLOAT, global_attended_row, ...);
}
```

**Impact**: For prefill with `seq_len=5`: **5 allgathers**  
**Impact**: For decode with `seq_len=1`: **1 allgather**

---

## Total MPI_Allgather Count Per Layer

### Prefill (seq_len=5)
```
Q/K/V projection gathering:      3 × 5 = 15 allgathers
Post-RoPE Q/K/V gathering:       3 × 5 = 15 allgathers
KV cache gathering:              3 allgathers
Unmasked scores gathering:       2 allgathers
Softmax scores gathering:        2 allgathers
Attended output gathering:       5 allgathers
────────────────────────────────────────────────
TOTAL PER LAYER:                 42 allgathers
```

**For 24-layer model**: **1,008 allgathers per prefill** 😱

### Decode (seq_len=1)
```
Q/K/V projection gathering:      3 × 1 = 3 allgathers
Post-RoPE Q/K/V gathering:       3 × 1 = 3 allgathers
KV cache gathering:              3 allgathers
Unmasked scores gathering:       2 allgathers
Softmax scores gathering:        2 allgathers
Attended output gathering:       1 allgather
────────────────────────────────────────────────
TOTAL PER LAYER:                 14 allgathers
```

**For 24-layer model**: **336 allgathers per token decode** 😱

---

## Optimization Opportunities

### 🔥 CRITICAL - Eliminate Snapshot-Only Gathers (80%+ reduction!)

**Problem**: Most allgathers exist ONLY for snapshot validation, not for correctness.

**Solution**: Add `snapshot_callback_` guard to SKIP gathers when not testing:

```cpp
// BEFORE (always gathers)
if (world_size > 1) {
    for (int t = 0; t < seq_len; ++t) {
        MPI_Allgather(local_q_row, ...);  // Expensive!
    }
    if (rank == 0) {
        snapshot_callback_(...);  // Only rank 0 uses result
    }
}

// AFTER (only gather when needed)
if (world_size > 1 && snapshot_callback_) {  // ← Add guard!
    for (int t = 0; t < seq_len; ++t) {
        MPI_Allgather(local_q_row, ...);
    }
    if (rank == 0) {
        snapshot_callback_(...);
    }
}
```

**Impact**:
- **Prefill**: 42 → 8 allgathers per layer (81% reduction)
- **Decode**: 14 → 4 allgathers per layer (71% reduction)

**Affected Operations**:
1. ✅ Q/K/V projection gathering (Lines 1310-1400) - Snapshot only
2. ✅ Post-RoPE Q/K/V gathering (Lines 1720-1790) - Snapshot only
3. ✅ Unmasked scores gathering (Lines 2057-2100) - Snapshot only
4. ✅ Softmax scores gathering (Lines 2200-2230) - Snapshot only
5. ✅ Attended output gathering (Lines 2315-2340) - Snapshot only

**Keep**:
- KV cache gathering (Lines 1893-1911) - **REQUIRED** for GQA expansion

---

### 🚀 MEDIUM - Batch Row-by-Row Gathers

**Problem**: Per-timestep loops create `seq_len` separate MPI calls for Q/K/V:

```cpp
for (int t = 0; t < seq_len; ++t) {
    MPI_Allgather(local_q_row, local_head_dim, ...);  // seq_len calls
}
```

**Solution**: Use single MPI_Allgatherv with custom datatype or manual layout:

```cpp
// Single gather for entire tensor
MPI_Allgatherv(local_q->data(), seq_len * local_head_dim, MPI_FLOAT,
               global_q->data(), recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
```

**Challenge**: Need to ensure correct interleaving of ranks within each timestep.  
**Benefit**: 3 allgathers instead of `3 × seq_len` for Q/K/V snapshots  
**Impact**: Marginal if snapshot guard is applied (already skipped in production)

---

### 💡 LOW - Cache KV Across Decode Steps

**Problem**: KV cache gathering happens **every decode step**, even though most of the cache is unchanged.

**Current**:
```cpp
// Every decode: gather entire cache (seq_len=n_past+1)
MPI_Allgatherv(local_k_cache->data(), attn_seq_len * local_kv_dim, ...);
```

**Idea**: Incremental gather only for new token:
```cpp
// First time: gather full cache
if (first_decode_step) {
    MPI_Allgatherv(local_k_cache->data(), attn_seq_len * local_kv_dim, ...);
    cached_k = global_k_cache;  // Save for reuse
}
// Subsequent: only gather new token, append to cached
else {
    MPI_Allgatherv(local_k_cache->data() + n_past * local_kv_dim, local_kv_dim, ...);
    append_to_cache(cached_k, new_k);
}
```

**Challenge**: 
- Need to track cache state across invocations
- Invalidation strategy for cache_capacity overflow
- Complexity vs benefit tradeoff

**Benefit**: Reduces gather volume linearly with decode length  
**Impact**: Moderate for long decode sequences (>50 tokens)

---

### 🎯 EASY WIN - Eliminate Duplicate Score Gathers

**Problem**: We gather attention scores **twice**:
1. Unmasked scores (line 2074) - ATTENTION_SCORES snapshot
2. Softmax scores (line 2218) - ATTENTION_SOFTMAX snapshot

**Solution**: Only gather softmax scores (more useful for debugging):

```cpp
// REMOVE unmasked gather entirely (lines 2057-2110)
// KEEP only softmax gather (lines 2200-2230)
```

**Impact**: -2 allgathers per layer  
**Tradeoff**: Lose unmasked scores snapshot (rarely used)

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (Immediate - 1 hour)
1. ✅ Add `snapshot_callback_` guard to all snapshot-only gathers
2. ✅ Remove duplicate unmasked scores gather

**Expected Reduction**:
- Prefill: 42 → 6 allgathers/layer (86% reduction)
- Decode: 14 → 3 allgathers/layer (79% reduction)

### Phase 2: Medium Effort (Optional - 2-4 hours)
3. Batch row-by-row gathers into single MPI_Allgatherv
4. Profile to measure actual performance impact

### Phase 3: Advanced (Future research)
5. Investigate incremental KV cache gathering
6. Explore MPI one-sided operations (MPI_Get) for asynchronous cache access

---

## Performance Estimates

### Current Baseline (24-layer model)
- **Prefill (seq_len=5)**: 1,008 allgathers
- **Decode per token**: 336 allgathers

### After Phase 1 (Production - no snapshots)
- **Prefill (seq_len=5)**: 144 allgathers (86% reduction)
- **Decode per token**: 72 allgathers (79% reduction)

### Only Critical Path Remains
- **KV cache gather**: 2 allgatherv per layer (required for GQA)
- **Metadata gather**: 1 allgather per layer (sendcount exchange)

**Total**: 3 allgathers per layer × 24 layers = **72 allgathers** (minimum achievable)

---

## Code Impact Estimate

### Files to Modify
- `src/kernels/MPIAttentionKernel.cpp`: Add guards around 6 gather sites

### Lines of Code
- **Add**: ~6 lines (wrap existing blocks with `if (snapshot_callback_)`)
- **Remove**: ~50 lines (duplicate unmasked scores gather)

### Risk
- **Low**: Snapshot-only gathers have no effect on correctness
- **Testing**: Verify parity tests still pass (they should, as they enable snapshots)

---

## Conclusion

**Immediate action**: Apply Phase 1 guards to eliminate 80%+ of MPI overhead.  
**Production impact**: Massive reduction in communication for inference workloads.  
**Testing impact**: Zero - snapshots still work when callback is registered.

This is a **high-impact, low-risk optimization** that should be implemented immediately.
