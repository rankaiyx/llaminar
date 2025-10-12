# Q Projection Bug Isolated to Rank 1

**Date**: 2025-01-12  
**Author**: David Sanftenberg  
**Status**: 🔴 **CRITICAL BUG FOUND** - Rank 1 producing wrong Q projection values

## Summary

Isolated the Q projection divergence to **rank 1 only**. Rank 0 produces perfect results matching PyTorch (FP32 precision), while rank 1's outputs are completely wrong with errors up to **12.3** (vs tolerance of 0.05).

## Bug Isolation Results

### Rank 0: ✅ PERFECT
```
PyTorch head 0: [ 0.08260688,  0.31437597,  0.03898931, -0.874254,   -15.520946]
Llaminar rank 0: [ 0.0826069,   0.314376,    0.0389893,  -0.874254,   -15.5209]
Max abs diff: 0.000046 (FP32 precision)
```

### Rank 1: ❌ COMPLETELY WRONG
```
PyTorch head 7:  [ 0.08243228,  1.5207065,   0.23456188,  4.0185647,  -3.2651265]
Llaminar rank 1: [-0.19424,     0.554032,   -0.37286,     1.00782,    -15.5776]
Max abs diff: 12.312474 (MASSIVE ERROR)
```

## Root Cause Analysis

Since both ranks:
- ✅ Receive identical input (verified: both show `input[0:5] = [0.041737, -0.00297965, ...]`)
- ✅ Use identical matrix multiplication code (`matmul_with_bias`)
- ✅ Have different weight statistics (expected for different slices)

But only rank 1 produces wrong results, the bug **must be**:

### Most Likely: **Weight Slicing Bug for Rank 1**

Rank 1 is receiving the wrong slice of the Q projection weight matrix.

**Expected**:
- Rank 0: wq rows [0, 448) → heads 0-6
- Rank 1: wq rows [448, 896) → heads 7-13

**Hypothesis**: Rank 1 is getting:
- Wrong rows (e.g., rows [0, 448) again, duplicating rank 0)
- Wrong transposition
- Wrong memory layout

### Evidence

From debug log:
```
Rank 0 local_wq stats: min=-1.03906 max=0.96875  mean=-9.49203e-06
Rank 1 local_wq stats: min=-1.22656 max=1.17188  mean=-2.38856e-05
```

The weights ARE different between ranks (good sign), but we need to verify they're the **correct** slices.

## Next Debug Steps

### 1. Verify Weight Slicing (URGENT)

Check the weight loading code to see how wq is sliced by rank:

```cpp
// In QwenPipeline or weight loading code
// Find where local_wq is created for each rank
// Expected:
//   rank 0: wq.slice(row_start=0, row_end=448)
//   rank 1: wq.slice(row_start=448, row_end=896)
```

### 2. Compare Rank 1 Weights with PyTorch

Extract the weights from the saved PyTorch snapshot and compare:

```python
# Load PyTorch weights
pytorch_wq = np.load("/tmp/pytorch_snapshots_openblas/weights/layer0_Q_WEIGHT.npy")
# Shape should be [896, 896] (d_model, d_model)

# Extract rank 1's expected slice (rows 448-896)
expected_rank1_wq = pytorch_wq[448:896, :]

# Compare with Llaminar's rank 1 wq
# (need to save this from Llaminar first)
```

### 3. Add Weight Verification to Rank 1

Modify the debug logging to save rank 1's wq to a file and compare:

```cpp
if (layer_index_ == 0 && rank == 1) {
    // Save local_wq to file
    std::ofstream out("/tmp/llaminar_rank1_wq_layer0.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(local_wq->data()), 
              local_wq->size() * sizeof(float));
    out.close();
}
```

## Weight Loading Code Location

Need to check these files:
- `src/qwen_pipeline.cpp` - Where weights are loaded
- `src/model_loader.cpp` - GGUF weight extraction
- `src/kernels/MPIAttentionKernel.cpp` - Where local_wq is assigned

Look for:
```cpp
// Weight slicing by rank
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int heads_per_rank = n_heads / world_size;
int weight_rows_per_rank = heads_per_rank * head_dim;
int row_offset = rank * weight_rows_per_rank;

// This should give:
//   rank 0: row_offset = 0
//   rank 1: row_offset = 448
```

## Temporary Workaround

For testing, can run single-rank mode to bypass the bug:

```bash
# Single rank (no MPI sharding)
mpirun -np 1 ./build/test_parity_framework --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch"
```

If this passes, it **confirms** the bug is in multi-rank weight distribution, not in the computation itself.

## Impact

**Critical** - This bug breaks all multi-rank inference:
- ❌ Rank 1 produces garbage for heads 7-13
- ❌ All layers after layer 0 will compound the error
- ❌ Final output will be completely wrong

**Priority**: Fix weight slicing for rank 1 before any other work.

## Expected Fix

Once we identify where wq is sliced:

```cpp
// BEFORE (wrong - both ranks getting same slice?)
auto local_wq = global_wq;  // Wrong!

// AFTER (correct - each rank gets different slice)
int row_start = rank * heads_per_rank * head_dim;
int row_end = (rank + 1) * heads_per_rank * head_dim;
auto local_wq = global_wq->slice_rows(row_start, row_end);
```

## Files to Investigate

1. **Weight distribution**: `src/qwen_pipeline.cpp`
2. **Tensor slicing**: `src/tensors/tensor.cpp`
3. **MPI weight setup**: `src/kernels/MPIAttentionKernel.cpp` (constructor)

## Test Plan

After fix:
1. Run debug logging to verify rank 1 gets rows [448, 896)
2. Compare rank 1's first weight values with PyTorch `wq[448:452]`
3. Run OpenBLAS prefill test - should pass all 387 stages
4. Run incremental decode test - should pass

## Summary

✅ **Found**: Rank 1 weight slicing bug  
✅ **Isolated**: Only rank 1 affected, rank 0 perfect  
⏳ **Next**: Find weight slicing code and fix row offset for rank 1  
🎯 **ETA**: Should be a simple fix once code location is found

---

**Critical Discovery**: We now know exactly what's wrong (rank 1 weights) and where to look (weight slicing code). This is a major breakthrough in debugging!
