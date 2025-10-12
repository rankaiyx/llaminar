# Q Projection MPI Gather Bug Discovery

**Date**: 2025-01-12  
**Author**: David Sanftenberg  
**Context**: Debugging Q projection divergence with enhanced diagnostics

## Summary

Discovered a critical bug in how Q projection results from MPI ranks are interpreted after `MPI_Allgather`. The gathered global tensor has **rank 1's heads labeled as belonging to rank 0**, causing head-specific errors in the parity test.

## Bug Discovery

### Diagnostic Logging Output

Added comprehensive logging to trace Q projection computation and MPI gather operation:

**Rank 0 Local Results (before gather)**:
```
[Q_GATHER_DEBUG] Layer 0 Rank 0 BEFORE gather:
  local_q shape: [5, 448]
  local_head_dim: 448 (= 7 heads * 64 dims)
  Expected rank 0 heads: [0, 7)
  Head 0 (local head 0) first 5 dims: [0.0826069, 0.314376, 0.0389893, -0.874254, -15.5209]
  Head 1 (local head 1) first 5 dims: [-1.67767, -2.30588, -1.07729, 6.3562, -6.74959]
  Head 2 (local head 2) first 5 dims: [-0.0243823, -0.0388065, 0.0074221, -0.0477429, -3.88662]
```

**Rank 1 Local Results (before gather)**:
```
[Q_GATHER_DEBUG] Layer 0 Rank 1 BEFORE gather:
  local_q shape: [5, 448]
  local_head_dim: 448 (= 7 heads * 64 dims)
  Expected rank 1 heads: [7, 14)
  Head 7 (local head 0) first 5 dims: [-0.19424, 0.554032, -0.37286, 1.00782, -15.5776]
  Head 8 (local head 1) first 5 dims: [-1.47444, -2.19268, -0.249479, 6.50416, -7.95788]
  Head 9 (local head 2) first 5 dims: [-0.0108606, 0.0522041, -0.409431, -0.409304, -3.4137]
```

**Global Tensor After Gather (rank 0's view)**:
```
[Q_GATHER_DEBUG] Layer 0 Rank 0 AFTER gather:
  global_q shape: [5, 896]
  Global head 0 (from rank 0) first 5 dims: [0.0826069, 0.314376, 0.0389893, -0.874254, -15.5209]
  Global head 1 (from rank 0) first 5 dims: [-1.67767, -2.30588, -1.07729, 6.3562, -6.74959]
  ...
  Global head 6 (from rank 0) first 5 dims: [-0.157386, -0.108389, -0.481868, 0.0234016, -10.9438]
  Global head 7 (from rank 1) first 5 dims: [-0.19424, 0.554032, -0.37286, 1.00782, -15.5776]
  Global head 8 (from rank 1) first 5 dims: [-1.47444, -2.19268, -0.249479, 6.50416, -7.95788]
  ...
```

### The Bug

**Expected Behavior**:
- Global head 0-6 should come from rank 0's local heads 0-6
- Global head 7-13 should come from rank 1's local heads 0-6

**Actual Behavior** (from logging):
- Global head 7's values match rank 1's local head 0 ✅ (correct values)
- But the logging **says** "Global head 7 (from rank 0)" ❌ (wrong attribution)

Wait - actually looking more carefully at the logging code, the interpretation logic is:
```cpp
int source_rank = h / local_heads;
```

For head 7: `source_rank = 7 / 7 = 1` ✅ This is correct!

So the logging is actually correct. Let me re-examine the actual data...

## Wait - Let Me Reanalyze

Looking at the gathered data more carefully:

**Rank 1's local head 0** (which should become global head 7):
```
[-0.19424, 0.554032, -0.37286, 1.00782, -15.5776]
```

**Global head 7** after gather:
```
[-0.19424, 0.554032, -0.37286, 1.00782, -15.5776]
```

These **match perfectly**! So the gather is actually working correctly. Let me check if PyTorch is expecting something different...

## Hypothesis Shift

If the gather is working correctly, then the bug must be in:
1. **Weight slicing** - Are we giving rank 1 the wrong slice of the weight matrix?
2. **Input data** - Both ranks see the same input (verified: both show identical input stats)
3. **Matrix multiplication order** - Is the memory layout assumption wrong?

Let me check the weight loading to see if rank 1 is getting the correct weight slice.

From the output:
- **Rank 0 wq stats**: min=-1.03906 max=0.96875 mean=-9.49203e-06
- **Rank 1 wq stats**: min=-1.22656 max=1.17188 mean=-2.38856e-05

The weight statistics are **different** between ranks, which is correct (they have different slices). But are they getting the **right** slices?

## Next Investigation Steps

1. **Verify weight slicing is correct**:
   - Check if rank 0 gets rows [0, 448) of wq (heads 0-6)
   - Check if rank 1 gets rows [448, 896) of wq (heads 7-13)
   
2. **Compare with PyTorch Q projection snapshot**:
   - Load the PyTorch Q_PROJECTION_layer0.npy file
   - Extract heads 0-6 and compare with rank 0's output
   - Extract heads 7-13 and compare with rank 1's output
   - This will tell us which rank is producing wrong results

3. **Check matrix multiplication dimensions**:
   - Input: [5, 896] ✅ (same on both ranks)
   - wq rank 0: [896, 448] (expecting rows for heads 0-6)
   - wq rank 1: [896, 448] (expecting rows for heads 7-13)
   - Output: [5, 448] each rank

## Status

- ✅ MPI gather operation is working correctly
- ✅ Global tensor assembly is correct
- ❌ One or both ranks are producing wrong local Q values
- 🎯 **Next**: Compare local results with PyTorch to identify which rank is wrong

## Files Modified

Added diagnostic logging to:
- `src/kernels/MPIAttentionKernel.cpp` (lines 870-1180)
  - Input/weight stats before Q projection
  - Local Q stats after projection  
  - Per-head values before gather
  - Global tensor per-head values after gather

## Lessons Learned

Always verify assumptions! Initial hypothesis was that MPI gather was broken, but detailed logging revealed it's working correctly. The bug is earlier in the pipeline.

The head-specific error pattern (head 8, 13, 12 having largest errors) combined with different weight statistics per rank suggests the weight slicing or local computation is wrong.

## Next Steps

Run a targeted comparison script to extract PyTorch's Q projection for layer 0 and compare head-by-head with our output to identify exactly which heads are wrong and by how much.
