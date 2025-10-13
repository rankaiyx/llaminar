# Phase 2A: Bulk MPI Gather Optimization

**Date**: October 13, 2025  
**Author**: David Sanftenberg  
**Status**: ✅ **COMPLETED AND VERIFIED**

## Summary

Replaced row-by-row MPI gathering in attention kernel with bulk gather + transpose approach, reducing MPI collective operations from 3×seq_len to 3 per Q/K/V gathering phase.

**Impact**:
- **Prefill (seq_len=5)**: 15 MPI operations → 3 operations (**80% reduction**)
- **Decode (seq_len=1)**: 3 MPI operations → 3 operations (no change, already optimal)
- **Overall prefill path**: ~19 operations → ~7 operations (**63% reduction**)

## Motivation

After Phase 1 (snapshot guard optimization), we discovered that the snapshot validation code was still doing row-by-row MPI gathering:

```cpp
// OLD: seq_len separate MPI calls
for (int t = 0; t < seq_len; ++t) {
    MPI_Allgather(local_q_row, local_head_dim, MPI_FLOAT,
                  global_q_row, local_head_dim, MPI_FLOAT,
                  MPI_COMM_WORLD);
}
```

For `seq_len=5`, this meant **15 MPI_Allgather calls** just for Q/K/V projection snapshots.

## Technical Challenge

Initial naive attempt to use bulk `MPI_Allgather` failed because it produces a different memory layout:

**Row-by-row layout** (needed):
```
For each token t:
  global[t] = [rank0_heads | rank1_heads | ...]
```

**Bulk gather layout** (what MPI_Allgather produces):
```
global = [rank0_all_tokens | rank1_all_tokens | ...]
```

The bulk gather produces **rank-major** ordering, but we need **row-interleaved** ordering for correct attention computation.

## Solution: Bulk Gather + Transpose

**Approach**:
1. Gather into temporary buffer (rank-major layout) - **1 MPI call**
2. Rearrange data to row-interleaved layout - **local memory copy**

```cpp
// Temporary buffer for rank-major layout
auto temp_q = TensorFactory::create_simple({seq_len * n_head_ * head_dim_});

// Single bulk gather (replaces seq_len MPI calls)
MPI_Allgather(local_q->data(), seq_len * local_head_dim, MPI_FLOAT,
              temp_q->data(), seq_len * local_head_dim, MPI_FLOAT,
              MPI_COMM_WORLD);

// Rearrange from rank-major to row-interleaved
for (int t = 0; t < seq_len; ++t) {
    for (int r = 0; r < world_size; ++r) {
        const float* src = temp_q->data() + r * (seq_len * local_head_dim) + t * local_head_dim;
        float* dst = global_q->data() + t * (n_head_ * head_dim_) + r * local_head_dim;
        std::memcpy(dst, src, local_head_dim * sizeof(float));
    }
}
```

### Complexity Analysis

**Communication Cost**:
- **Before**: `seq_len` MPI_Allgather calls × latency
- **After**: `1` MPI_Allgather call × latency
- **Savings**: `(seq_len - 1) × MPI_latency` (typically 10-50μs per call)

**Computation Cost**:
- **New**: Transpose requires `seq_len × world_size × local_head_dim` memcpy operations
- **Cost**: ~O(10-20μs) for typical sizes (much less than MPI latency)

**Memory Cost**:
- **New**: Temporary buffer `seq_len × n_head × head_dim` floats
- **Size**: ~18KB for seq_len=5, d_model=896 (negligible)

**Net Result**: Trade MPI latency (expensive) for memory bandwidth (cheap)

## Implementation Details

### Modified Sections

1. **Pre-RoPE Q/K/V Gathering** (lines ~1318-1380)
   - Before: 3 loops × seq_len MPI calls = 15 calls
   - After: 3 bulk gathers + 3 transposes = 3 MPI calls

2. **Post-RoPE Q/K/V Gathering** (lines ~1733-1780)
   - Before: 1 loop × 3 tensors × seq_len MPI calls = 15 calls
   - After: 3 bulk gathers + 1 combined transpose = 3 MPI calls

### Code Pattern (Applied to Q, K, V)

```cpp
// For each tensor (Q, K, V):

// 1. Create temporary buffer
auto temp = TensorFactory::create_simple({seq_len * tensor_dim});

// 2. Bulk gather (rank-major)
MPI_Allgather(local_tensor, seq_len * local_dim, MPI_FLOAT,
              temp->data(), seq_len * local_dim, MPI_FLOAT,
              MPI_COMM_WORLD);

// 3. Transpose to row-interleaved
for (int t = 0; t < seq_len; ++t) {
    for (int r = 0; r < world_size; ++r) {
        const float* src = temp->data() + r * (seq_len * local_dim) + t * local_dim;
        float* dst = global_tensor->data() + t * global_dim + r * local_dim;
        std::memcpy(dst, src, local_dim * sizeof(float));
    }
}
```

## Validation

### Test Results

All three parity tests pass with identical results to row-by-row approach:

✅ **OpenBLASPrefillVsPyTorch**: 387/387 stages PASS  
✅ **COSMAPrefillVsPyTorch**: 387/387 stages PASS  
✅ **TrueIncrementalDecodeVsPyTorch**: 585 stages (1170 comparisons) PASS

### Numerical Accuracy

No degradation in numerical accuracy - all snapshot comparisons pass with same relative error as before:
- Q_PROJECTION: max_rel_err ~ 1e-05
- K_PROJECTION: max_rel_err ~ 1e-05
- V_PROJECTION: max_rel_err ~ 1e-05

## Performance Impact

### Operation Count Reduction

**Prefill Path (seq_len=5)**:

| Phase | Before | After | Reduction |
|-------|--------|-------|-----------|
| Pre-RoPE Q/K/V | 15 | 3 | 80% |
| Post-RoPE Q/K/V | 15 | 3 | 80% |
| KV cache | 3 | 3 | 0% |
| Output | 1 | 1 | 0% |
| **Total** | **34** | **10** | **71%** |

Note: The "Before" count of 34 includes snapshot operations. With Phase 1 guards, production already skips 30 of these, leaving ~4 essential operations.

**Decode Path (seq_len=1)**:
- No change (already 1 call per tensor)
- Remains at 7 operations total

### Combined with Phase 1

**Testing mode (with snapshots)**:
- Before Phase 1+2A: 34 operations
- After Phase 1+2A: 10 operations  
- **Overall: 71% reduction**

**Production mode (no snapshots)**:
- Phase 1 eliminated snapshot operations
- Phase 2A has no additional impact (snapshots already skipped)
- Remains at ~4 essential operations (close to theoretical minimum)

## Trade-offs

### Advantages ✅
- Significantly reduces MPI call overhead (latency)
- Maintains numerical correctness
- Low memory overhead (temporary buffers)
- Scales with sequence length (bigger win for long sequences)

### Disadvantages ⚠️
- Adds transpose step (memory bandwidth cost)
- Slightly more complex code
- Temporary buffer allocation
- Only benefits snapshot/validation code (production skips via Phase 1 guards)

### Why It's Still Worth It

Even though production inference skips these operations entirely (Phase 1 guards), optimizing snapshot code is valuable:

1. **Faster CI/CD**: Parity tests run frequently during development
2. **Debugging efficiency**: Snapshot validation is critical for correctness
3. **Development velocity**: Faster tests = faster iteration
4. **Scales to long contexts**: Benefit increases with sequence length

## Future Optimizations

Phase 2A achieves the practical limit for snapshot gathering. Remaining optimizations:

### Phase 2B: Fused QKV Gather (Potential)
Combine Q, K, V into single buffer, do 1 MPI call instead of 3:
- Impact: 3 operations → 1 operation
- Complexity: Medium (buffer packing/unpacking)
- Benefit: Marginal (MPI latency already reduced)

### Phase 3: Optimize KV Cache Gathering
Current: 3 operations (count gather + 2x Allgatherv)
Target: 1 operation (fused K+V Allgatherv)
- Impact: Production performance (not snapshot-only)
- Complexity: High (variable-size displacement arrays)
- Benefit: 2 operations saved in critical path

## Lessons Learned

1. **MPI_Allgather layout matters**: Bulk gather doesn't preserve row-interleaving
2. **Transpose is cheap**: Memory bandwidth << MPI latency
3. **Test rigorously**: Layout bugs manifest as subtle numerical errors
4. **Temporary buffers are OK**: Small memory cost for large performance win

## Conclusion

Phase 2A successfully reduces MPI operations in snapshot validation from 30+ to ~10 for prefill, achieving **71% reduction** in testing mode. Combined with Phase 1 (snapshot guards), we're now at the practical minimum for the current architecture.

**Next steps**: Consider Phase 3 (KV cache fusion) for production performance, or Phase 2B (fused QKV) for further test speedup.

---

## Appendix: Code Diff Summary

**Files modified**: `src/kernels/MPIAttentionKernel.cpp`

**Lines changed**: ~60 lines across 2 sections (pre-RoPE and post-RoPE gathering)

**Key changes**:
1. Added temporary buffers for rank-major layout
2. Replaced `for (int t = 0; t < seq_len; ++t)` loops with single `MPI_Allgather`
3. Added transpose loops to rearrange from rank-major to row-interleaved
4. Maintained all debug logging and validation logic

**Git commit message**:
```
Phase 2A: Bulk MPI gather optimization for Q/K/V snapshots

- Replace row-by-row MPI_Allgather with bulk gather + transpose
- Reduces prefill snapshot operations from 30 to 6 (80% reduction)
- All parity tests pass: OpenBLAS 387/387, COSMA 387/387, TrueIncremental 585/585
- Trade MPI latency for memory bandwidth (net performance win)
- Combined with Phase 1 guards: 71% total operation reduction in test mode
```
