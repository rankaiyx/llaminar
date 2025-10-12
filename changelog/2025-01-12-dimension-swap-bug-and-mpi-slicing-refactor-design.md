# Dimension Swap Bug Root Cause & MPI Slicing Refactor Design

**Date**: January 12, 2025  
**Author**: David Sanftenberg  
**Status**: Design Document (Implementation Pending)

## Problem Summary

We discovered a critical bug where rank > 0 produces wrong outputs in Q projection (12.3 max error vs PyTorch). Investigation revealed fundamental architectural issues:

### Bug Root Cause

1. **GGUF stores ALL 2D tensors in llama.cpp convention** (transposed from PyTorch):
   - Embeddings: `[d_model, vocab_size]` → need `[vocab_size, d_model]`
   - Weight matrices: `[in_features, out_features]` → need `[out_features, in_features]`

2. **ModelLoader swaps dimensions** (metadata only, not data):
   ```cpp
   if (n_dims == 2) {
       std::swap(tensor.dimensions[0], tensor.dimensions[1]);
   }
   ```

3. **loadTensorRowShard uses swapped dimensions for offset calculation**:
   ```cpp
   size_t rows = info->dimensions[0];  // After swap!
   size_t cols = info->dimensions[1];  // After swap!
   const float *src = full->data.data() + row_offset * cols;  // WRONG STRIDE!
   ```

4. **Data is NEVER transposed** - it stays in original GGUF layout
5. **Result**: Rank 0 gets correct bytes (offset 0), rank 1 gets wrong bytes (wrong stride)

### Example: K Weight (Qwen 0.5B)

```
GGUF on disk:     [896, 128]  (in_features, out_features)
After dim swap:   [128, 896]  (logical view for PyTorch)
Data in memory:   [896, 128]  (still original layout!)

Rank 1 wants logical rows [64:128] of [128, 896]
But calculates offset as: 64 * 896 = 57,344 floats
Correct offset should be: 64 * 128 = 8,192 floats (using GGUF layout)

Actually, it's even more complex - we want ROWS of the logical [128, 896],
which are COLUMNS of the physical [896, 128]!
```

### Attempted Fixes That Failed

1. ❌ **Remove dimension swap entirely**: Broke embedding validation (expected `[vocab, d]`, got `[d, vocab]`)
2. ❌ **Swap only embeddings**: Broke FFN weights (expected `[4864, 896]`, got `[896, 4864]`)
3. ❌ **Use gguf_dimensions for offset calc**: Wrong for transposed indexing (rows→columns)

## Architectural Issues Identified

### Issue 1: No Centralized Dimension Convention

Current state:
- ModelLoader does dimension swap
- loadTensorRowShard uses swapped dimensions
- Kernels expect PyTorch convention
- No single source of truth
- Scattered, confusing logic

### Issue 2: Duplicate MPI Slicing Logic

Current state (qwen_pipeline.cpp):
```cpp
// Q weights
const int q_row_offset = mpi_rank * q_heads_per_rank * head_dim;
const int q_row_count = q_heads_per_rank * head_dim;
loader.loadTensorRowShard(q_name, q_row_offset, q_row_count, dest);

// K weights  
const int kv_row_offset = mpi_rank * kv_heads_per_rank * head_dim;
const int kv_row_count = kv_heads_per_rank * head_dim;
loader.loadTensorRowShard(k_name, kv_row_offset, kv_row_count, dest);

// V weights
// ... exact same logic ...

// O weights (column slicing instead!)
const int o_col_offset = mpi_rank * q_heads_per_rank * head_dim;
const int o_col_count = q_heads_per_rank * head_dim;
loader.loadTensorColumnShards(o_name, o_col_offset, o_col_count, dest);

// FFN weights (replicated, but could be sliced)
// ... different logic ...
```

This logic is:
- Duplicated 5+ times per layer
- Error-prone (easy to use wrong slice type)
- Hard to maintain (changes need updates in many places)
- Not validated by weight contracts

### Issue 3: Weight Contracts Don't Load

Current state:
```cpp
// Weight contracts defined in weight_contracts.h
WeightShapeContract q_contract("attn_q.weight", {"n_head*head_dim", "d_model"}, 
                               "Q projection", false, ROW_SLICED, "n_head");

// But loading happens elsewhere (qwen_pipeline.cpp)
auto wq = loader.loadTensorRowShard(...);  // Manual slicing

// Then validation happens later
q_contract.validate_with_mpi(wq, config, rank, size);  // Just checks!
```

The contract knows:
- Expected dimensions
- Slicing strategy (ROW_SLICED)
- Slice parameter (n_head)

But doesn't:
- Actually perform the loading
- Handle dimension transformations
- Ensure correct slicing

## Proposed Solution

### Design: Contract-Driven Loading & MPI Slicing Helper

#### 1. Enhanced Weight Contract

```cpp
struct WeightShapeContract {
    // ... existing fields ...
    
    // NEW: Storage format information
    std::vector<std::string> gguf_dim_expressions;  // Dimensions in GGUF file
    bool needs_transpose_data;  // Whether to transpose actual data
    
    // NEW: Loading integration
    std::shared_ptr<TensorBase> load(ModelLoader& loader, 
                                     const TransformerLayerConfig& cfg,
                                     int mpi_rank, int mpi_size,
                                     int layer_index = -1) const;
                                     
    // Returns correctly-sliced tensor in expected logical layout
};
```

#### 2. MPI Slicing Helper

```cpp
namespace mpi_slicing {

/**
 * @brief Calculate slice parameters for MPI-aware tensor loading.
 * 
 * Handles all complexity of:
 * - Row vs column slicing
 * - Dimension conventions (GGUF vs PyTorch)
 * - Head distribution across ranks
 * - Offset and count calculation
 */
struct SliceParams {
    int offset;      // Starting index in GGUF layout
    int count;       // Number of elements to load in GGUF layout
    bool is_column;  // True for column slice, false for row slice
    
    // Dimension transformation info
    std::vector<int> expected_shape;  // Shape after loading (PyTorch convention)
    bool needs_transpose;             // Whether to transpose data after loading
};

/**
 * @brief Calculate MPI slice parameters from contract.
 */
SliceParams calculate_slice(const WeightShapeContract& contract,
                           const TransformerLayerConfig& cfg,
                           int mpi_rank, int mpi_size);

/**
 * @brief Load tensor with automatic MPI slicing based on contract.
 * 
 * This is the ONE FUNCTION that handles all weight loading complexity:
 * - Determines correct slicing strategy from contract
 * - Calculates correct offsets for GGUF layout
 * - Loads the right bytes
 * - Transposes if needed
 * - Returns tensor in expected PyTorch layout
 */
std::shared_ptr<TensorBase> load_with_contract(
    ModelLoader& loader,
    const WeightShapeContract& contract,
    const TransformerLayerConfig& cfg,
    int mpi_rank,
    int mpi_size,
    int layer_index = -1);

}  // namespace mpi_slicing
```

#### 3. Usage Example

```cpp
// Define contract (once, in weight_contracts.h)
WeightShapeContract Q_CONTRACT(
    "attn_q.weight",
    {"n_head*head_dim", "d_model"},           // Expected (PyTorch)
    {"d_model", "n_head*head_dim"},           // GGUF storage
    "Q projection weight",
    true,                                      // Needs data transpose
    WeightSliceType::ROW_SLICED,
    "n_head"
);

// Use contract (in qwen_pipeline.cpp)
weights.wq[layer] = mpi_slicing::load_with_contract(
    loader, Q_CONTRACT, config, mpi_rank, mpi_size, layer);
    
// Validation happens automatically during load!
// No manual offset calculation!
// No dimension swap confusion!
// No duplicated logic!
```

### Implementation Plan

#### Phase 1: Core Infrastructure (2-3 hours)

1. **Add mpi_slicing namespace** (`src/mpi_slicing_helper.h`)
   - `SliceParams` struct
   - `calculate_slice()` function
   - `load_with_contract()` function

2. **Enhance WeightShapeContract**
   - Add `gguf_dim_expressions`
   - Add `needs_transpose_data`
   - Add `load()` method that calls `mpi_slicing::load_with_contract()`

3. **Add ModelLoader::loadTensorWithTranspose()**
   - Loads data
   - Optionally transposes
   - Returns in expected layout

#### Phase 2: Migration (3-4 hours)

1. **Update all weight contracts** in `weight_contracts.h`
   - Add GGUF dimensions
   - Add transpose flags
   - Validate against current code

2. **Migrate qwen_pipeline.cpp** loading
   - Replace manual `loadTensorRowShard` calls with `contract.load()`
   - Remove manual offset calculations
   - Simplify loading loop

3. **Test and validate**
   - All tests pass
   - PyTorch parity perfect
   - MPI slicing correct

#### Phase 3: Cleanup (1-2 hours)

1. **Remove deprecated code**
   - Remove manual dimension swap in parseTensorInfo
   - Remove gguf_dimensions field (no longer needed)
   - Clean up comments

2. **Documentation**
   - Update WEIGHT_MATRIX_CONVENTIONS.md
   - Document contract-driven loading
   - Add examples

### Benefits

1. **Single Source of Truth**: Contracts define everything about a weight
2. **No Dimension Confusion**: Conversion handled transparently
3. **No Duplicate Logic**: One function does all loading
4. **Type Safety**: Contracts enforce correct shapes
5. **Easier Debugging**: Clear error messages from contracts
6. **Easier Maintenance**: Add new weight = add one contract
7. **MPI Correctness**: Slicing logic in one tested place

### Migration Safety

- Keep old `loadTensorRowShard` for backward compatibility during migration
- Migrate one contract at a time
- Test each migration
- Remove old code only after all tests pass

## Current Status

- **Bug Identified**: Dimension swap + offset calculation mismatch
- **Design Complete**: Contract-driven loading with MPI slicing helper
- **Implementation**: NOT STARTED (design doc only)
- **Next Steps**: Implement Phase 1 core infrastructure

## Notes for Implementation

- Start with Q weight (most straightforward)
- Then K/V weights (similar pattern)
- Then O weight (column slicing)
- Then FFN weights (currently replicated)
- Test at each step with PyTorch parity

## References

- `src/weight_contracts.h` - Current contract system
- `src/model_loader.{h,cpp}` - Tensor loading
- `src/qwen_pipeline.cpp` - Current manual slicing
- `docs/WEIGHT_MATRIX_CONVENTIONS.md` - Dimension conventions
