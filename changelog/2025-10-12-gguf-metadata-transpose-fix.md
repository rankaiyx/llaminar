# GGUF Metadata vs Data Layout Fix

**Date**: 2025-10-12  
**Author**: David Sanftenberg  
**Status**: ✅ **COMPLETE** - Weight verification passing  

## Summary

Fixed critical architectural bug in ModelLoader where GGUF file metadata dimensions are **backwards** from actual data layout, causing incorrect transpose operations and weight mismatches.

## Root Cause Analysis

### The GGUF Quirk

GGUF files have a metadata/data layout mismatch:

**Example: token_embd.weight**
- **GGUF metadata claims**: `[896, 151669]` = `[d_model, vocab_size]`
- **Actual data stored as**: `[151669, 896]` = `[vocab_size, d_model]` in row-major
- **Proof**: First 3 raw values match `PyTorch[0][:3]`, not `PyTorch[:3][0]`

This exists because llama.cpp's `ggml_mul_mat` implicitly transposes, so they store metadata backwards but data correctly.

### Previous (Broken) Behavior

1. **ModelLoader.parseTensorInfo()** swapped dimensions: `[896, 151669]` → `[151669, 896]` ✓
2. **ModelLoader.loadTensor()** then TRANSPOSED the data to match swapped metadata ✗
   - But data was ALREADY in correct `[151669, 896]` layout!
   - Transpose corrupted it: `embedding[0][1]` landed at `embedding[1][0]`
3. **Result**: All weights had massive errors (rel_l2 ~1.4)

### Impact Discovered

```
Embedding verification:
  PyTorch  [0,:5]: [-0.00982666, 0.0407715, 0.00964355, ...]
  Llaminar [0,:5]: [-0.00982666, -0.0151367, 0.0327148, ...]  ✗ WRONG
  Llaminar [1,:5]: [0.0407715, ...]  ← This is where [0][1] went!

Weight verification: 24/24 layers FAILED
  - Q/K/V/O weights: rel_l2 ~1.4 (massive mismatch)
  - All sliced weights incorrect
```

## Changes Made

### 1. ModelLoader.cpp - Removed Incorrect Transpose

**File**: `src/model_loader.cpp`

**Before** (lines 1095-1148):
```cpp
// WRONG: Transpose data to match swapped metadata
if (dims_were_swapped) {
    std::vector<float> transposed(data_f32.size());
    for (size_t i = 0; i < gguf_rows; ++i) {
        for (size_t j = 0; j < gguf_cols; ++j) {
            transposed[j * gguf_rows + i] = data_f32[i * gguf_cols + j];
        }
    }
    data_f32 = std::move(transposed);
}
```

**After**:
```cpp
// GGUF METADATA vs DATA LAYOUT - THE TRUTH:
// ==========================================
// GGUF metadata dimensions are BACKWARDS from actual data layout!
//
// After dimension swap, metadata matches data layout!
//   - Metadata now says: [151669, 896]
//   - Data is stored as: [151669, 896] in row-major
//   - Therefore: NO DATA TRANSPOSE NEEDED!
//
// Just log for verification:
if (dims_were_swapped) {
    LOG_DEBUG("[DIMENSION_FIX] " << tensor_name 
             << ": GGUF metadata was backwards, corrected to match actual data layout"
             << " - data already matches, no transpose needed");
}
```

### 2. ModelLoader.cpp - Fixed Row Shard Offset Calculation

**File**: `src/model_loader.cpp`, function `loadTensorRowShard()` (line ~2250)

**Before**:
```cpp
// WRONG: Use backwards GGUF metadata for offset calc
size_t rows = info->gguf_dimensions[0];  // 896 (backwards!)
size_t cols = info->gguf_dimensions[1];  // 151669 (backwards!)
const float *src = full->data.data() + row_offset * cols;  // Wrong stride!
```

**After**:
```cpp
// Use CORRECTED dimensions (after metadata swap)
// Actual data matches corrected dimensions
size_t rows = info->dimensions[0];  // 151669 (correct!)
size_t cols = info->dimensions[1];  // 896 (correct!)
const float *src = full->data.data() + row_offset * cols;  // Correct stride!
```

**Impact**: Row sharding for Q/K/V weights was using wrong stride, loading wrong data regions.

### 3. mpi_slicing_helper.cpp - Eliminated Spurious Transpose Logic

**File**: `src/mpi_slicing_helper.cpp`, function `calculate_slice()` (lines 100-135)

**Before**:
```cpp
// WRONG: Compare gguf_dims vs expected_shape to decide transpose
bool gguf_is_transposed = (gguf_dims[0] == params.expected_shape[1] &&
                           gguf_dims[1] == params.expected_shape[0]);

if (gguf_is_transposed) {
    params.is_column_slice = true;  // Do opposite slice
    params.needs_transpose = true;   // Then transpose
}
```

**After**:
```cpp
// CRITICAL FIX: After ModelLoader dimension correction, data layout ALWAYS matches
// PyTorch convention (expected_shape). Just slice directly!
//
// ROW_SLICED → slice rows (dim 0), no transpose
// COL_SLICED → slice columns (dim 1), no transpose

if (contract.slice_type == WeightSliceType::ROW_SLICED) {
    params.is_column_slice = false;
    params.needs_transpose = false;
}
```

**Impact**: Sliced Q/K/V/O weights were doing column-slice-then-transpose instead of direct row slice.

### 4. weight_contracts.h - Updated Contract Transpose Flags

**File**: `src/weight_contracts.h` (lines 760-782)

Updated all sliced weight contracts to set `needs_transpose_data=false`:

```cpp
// OLD: All had needs_transpose_data=true
WeightShapeContract("blk.{layer}.attn_q.weight", {...}, true)

// NEW: All set to false - data already correct after ModelLoader fix
WeightShapeContract("blk.{layer}.attn_q.weight", {...}, false)
```

**Affected weights**: Q, K, V, O projections (all sliced weights)

## Verification

### Test Results

```bash
# Before fix
WEIGHT_VERIFY: ✗ All 24 layers FAILED
  Layer 0 Q MISMATCH: rel_l2=1.483966
  Layer 1 Q MISMATCH: rel_l2=1.449604
  ...all layers failed...

# After fix
WEIGHT_VERIFY: ✓ All weights verified successfully!
WEIGHT_VERIFY: All 24 layers verified successfully
```

### Embedding Verification

```python
# PyTorch ground truth
PyTorch embedding[0,:5]: [-0.00982666, 0.04077148, 0.00964355, 0.00066376, -0.02709961]

# Llaminar after fix
DIMENSION_DEBUG: token_embd.weight first_3_values=[-0.00982666, 0.0407715, 0.00964355] ✓
```

### Slice Parameters

```
# Before fix
Query projection: offset=0, count=448, is_column=1, transpose=1  ✗

# After fix  
Query projection: offset=0, count=448, is_column=0, transpose=0  ✓
```

## Remaining Work

Weight loading is now CORRECT, but parity tests still fail due to **compute errors** (not weight errors):

```
First divergence: Q_PROJECTION_layer0 (max_abs=71.9062, rel_l2=0.976415)
```

This indicates kernel/attention assembly issues, NOT weight loading issues. Next debugging should focus on:

1. MPI attention kernel correctness
2. Q/K/V projection computation
3. Attention score assembly

The weight loading foundation is now solid.

## Files Changed

1. `src/model_loader.cpp` - Removed incorrect transpose, fixed row shard
2. `src/mpi_slicing_helper.cpp` - Fixed slice type logic
3. `src/weight_contracts.h` - Updated transpose flags
4. `CMakeLists.txt` - Added DEBUG logging to all CTests (separate task)

## Lessons Learned

1. **Always verify ground truth empirically** - Comments claimed GGUF stores `[d, vocab]` but data proves otherwise
2. **GGUF metadata is backwards** - This is a fundamental GGUF quirk, not a bug
3. **Transpose is expensive and error-prone** - Avoid when possible by understanding data layout
4. **Test with real data, not synthetic** - Square matrices `[896, 896]` hide transpose bugs!

## Breaking Changes

None - This fixes broken behavior that never worked correctly.

## Migration Guide

No migration needed - weights are now loaded correctly for the first time.
