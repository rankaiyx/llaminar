# MPI Slicing Helper Implementation - Contract-Driven Weight Loading

**Date**: January 12, 2025  
**Author**: David Sanftenberg  
**Status**: Infrastructure Complete, Ready for Migration

## Summary

Implemented comprehensive contract-driven weight loading infrastructure to centralize and simplify MPI-aware tensor loading. This addresses the root cause of rank > 0 bugs where dimension swaps and manual offset calculations led to incorrect weight slices.

## Changes Implemented

### 1. New MPI Slicing Helper Module

**Files Created:**
- `src/mpi_slicing_helper.h` - Interface for contract-driven loading
- `src/mpi_slicing_helper.cpp` - Implementation with dimension/slicing logic
- Added to `CMakeLists.txt` as part of `llaminar_core`

**Key Components:**

```cpp
namespace mpi_slicing {

struct SliceParams {
    int offset;                      // Offset in GGUF layout
    int count;                       // Count in GGUF layout  
    bool is_column_slice;            // Row or column slice
    std::vector<int> expected_shape; // Final PyTorch shape
    bool needs_transpose;            // Whether to transpose data
};

SliceParams calculate_slice(const WeightShapeContract& contract, ...);
std::shared_ptr<TensorBase> load_with_contract(...);

}
```

### 2. Enhanced WeightShapeContract

**File Modified:** `src/weight_contracts.h`

**New Fields:**
```cpp
struct WeightShapeContract {
    // Existing fields...
    
    // NEW: GGUF storage metadata
    std::vector<std::string> gguf_dim_expressions;  // Dimensions in GGUF
    bool needs_transpose_data;                      // Transpose during load
    std::string name_pattern;                       // Pattern with {layer}
    std::string role_description;                   // For logging
    
    // NEW: Loading method
    std::shared_ptr<TensorBase> load(ModelLoader& loader,
                                     const TransformerLayerConfig& cfg,
                                     int mpi_rank, int mpi_size,
                                     int layer_index = -1) const;
};
```

**Updated Constructor:**
```cpp
WeightShapeContract(name, dims, desc, transpose, slice_type, slice_param,
                   gguf_dims, transpose_data)
```

### 3. Updated All Qwen Weight Contracts

**File Modified:** `src/weight_contracts.h` (Qwen contracts)

**Example - Q Weight:**
```cpp
// OLD (just validation)
WeightShapeContract("attn_q.weight", {"n_head*head_dim", "d_model"},
                   "Q projection", false, ROW_SLICED, "n_head")

// NEW (with loading metadata)
WeightShapeContract("attn_q.weight", {"n_head*head_dim", "d_model"},
                   "Q projection", false, ROW_SLICED, "n_head",
                   {"d_model", "n_head*head_dim"}, true)
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^
//  GGUF dimensions (transposed)    needs_transpose_data
```

**All Updated Contracts:**
- ✅ Global weights: token_embedding, output_norm, lm_head
- ✅ Attention weights: attn_q, attn_k, attn_v, attn_output
- ✅ FFN weights: w_gate, w_up, w_down
- ✅ Norms: attn_norm, ffn_norm

### 4. Added Column Shard Helper to ModelLoader

**File Modified:** `src/model_loader.{h,cpp}`

**New Method:**
```cpp
bool ModelLoader::loadTensorColumnShard(const std::string& tensor_name,
                                       int col_offset, int col_count,
                                       float* dest);
```

Wraps existing `loadTensorColumnShards` for single-shard convenience (mirrors `loadTensorRowShard`).

## How It Works

### The Dimension Problem We Solved

**GGUF Storage vs PyTorch Convention:**
```
GGUF stores:     [in_features, out_features]
PyTorch expects: [out_features, in_features]

Example - K weight (Qwen 0.5B):
GGUF:     [896, 128]   (d_model, n_head_kv*head_dim)
PyTorch:  [128, 896]   (n_head_kv*head_dim, d_model)
```

**The Trap:**
- Dimension swap is METADATA only - data stays in GGUF layout!
- MPI rank 1 wants rows [64:128] of PyTorch view
- But those are COLUMNS [64:128] of GGUF data!

### Contract-Driven Solution

1. **Contract Defines Both Dimensions:**
   ```cpp
   dim_expressions:      {"n_head_kv*head_dim", "d_model"}  // PyTorch
   gguf_dim_expressions: {"d_model", "n_head_kv*head_dim"}  // GGUF
   slice_type:           ROW_SLICED
   needs_transpose_data: true
   ```

2. **calculate_slice() Determines Strategy:**
   ```cpp
   // Detects: PyTorch rows = GGUF columns (transposed!)
   params.is_column_slice = true;
   params.offset = 64;
   params.count = 64;
   params.needs_transpose = true;
   ```

3. **load_with_contract() Executes:**
   ```cpp
   // Load COLUMNS [64:128] from GGUF
   loader.loadTensorColumnShard(name, 64, 64, dest);
   
   // Transpose to get PyTorch layout [64, 896]
   transpose(dest, [896, 64] → [64, 896]);
   ```

4. **Result: Correct Tensor:**
   ```cpp
   Shape: [64, 896]  // Matches PyTorch expectation
   Data: Correctly sliced K weight for rank 1
   ```

## Benefits

### Before (Manual Slicing)

**qwen_pipeline.cpp (~150 lines of duplicate logic):**
```cpp
// Q weights
const int q_row_offset = mpi_rank * q_heads_per_rank * head_dim;
const int q_row_count = q_heads_per_rank * head_dim;
auto wq = TensorFactory::create_simple({q_row_count, d_model});
loader.loadTensorRowShard(q_name, q_row_offset, q_row_count, wq->data());

// K weights (BROKEN - dimension swap causes wrong offset!)
const int kv_row_offset = mpi_rank * kv_heads_per_rank * head_dim;
const int kv_row_count = kv_heads_per_rank * head_dim;
auto wk = TensorFactory::create_simple({kv_row_count, d_model});
loader.loadTensorRowShard(k_name, kv_row_offset, kv_row_count, wk->data());
// ❌ BUG: K is [896, 128] in GGUF but code assumes [128, 896]!

// V weights (same bug)
// O weights (different slicing logic!)
// FFN weights (yet another pattern)
```

**Problems:**
- ❌ Dimension confusion scattered across codebase
- ❌ Manual offset calculation error-prone
- ❌ Duplicate slicing logic in every pipeline
- ❌ No validation that slicing matches contract
- ❌ Hard to debug dimension mismatches

### After (Contract-Driven)

**qwen_pipeline.cpp (1 line per weight!):**
```cpp
// Get weight contracts
auto contracts = llaminar::getQwenWeightContracts();

// Load ALL weights with correct slicing automatically
for (int layer = 0; layer < n_layers; ++layer) {
    weights.wq[layer] = contracts.layer_weights[Q_IDX].load(
        loader, config, mpi_rank, mpi_size, layer);
    weights.wk[layer] = contracts.layer_weights[K_IDX].load(
        loader, config, mpi_rank, mpi_size, layer);
    weights.wv[layer] = contracts.layer_weights[V_IDX].load(
        loader, config, mpi_rank, mpi_size, layer);
    // ... etc
}
// ✅ Dimension handling: automatic
// ✅ Offset calculation: automatic
// ✅ Transpose: automatic
// ✅ Validation: automatic
```

**Benefits:**
- ✅ Single source of truth for dimensions
- ✅ No manual offset calculations
- ✅ Automatic dimension handling
- ✅ Type-safe loading
- ✅ Clear error messages
- ✅ Easier to add new models

## Implementation Details

### Dimension Expression Evaluation

```cpp
static int evaluate_dim_expression(const std::string& expr, 
                                   const TransformerLayerConfig& cfg) {
    // Supports:
    // - Literals: "896", "128", "151669"
    // - Variables: "d_model", "n_head", "head_dim", "n_head_kv", "d_ff"
    // - Multiplication: "n_head*head_dim"
    
    if (expr.find('*') != npos) {
        auto [left, right] = split(expr, '*');
        return evaluate(left, cfg) * evaluate(right, cfg);
    }
    if (expr == "d_model") return cfg.d_model;
    // ... etc
}
```

### Transpose Logic

```cpp
// For K weight: GGUF [896, 64] → PyTorch [64, 896]
for (int i = 0; i < load_rows; ++i) {
    for (int j = 0; j < load_cols; ++j) {
        transposed[j * load_rows + i] = original[i * load_cols + j];
    }
}
// Result: [j, i] indexing instead of [i, j]
```

### Row vs Column Slice Detection

```cpp
bool gguf_is_transposed = (gguf_dims[0] == expected[1] && 
                          gguf_dims[1] == expected[0]);

if (contract.slice_type == ROW_SLICED) {
    if (gguf_is_transposed) {
        // PyTorch rows = GGUF columns → use column shard
        params.is_column_slice = true;
        params.needs_transpose = true;
    } else {
        // PyTorch rows = GGUF rows → use row shard
        params.is_column_slice = false;
        params.needs_transpose = false;
    }
}
```

## Testing Plan

### Phase 1: Infrastructure Validation (Current)
- ✅ All code compiles
- ✅ CMakeLists integration works
- ✅ Test binary links successfully
- ⏳ TODO: Simple unit test of `calculate_slice()`
- ⏳ TODO: Test transpose correctness

### Phase 2: Migration (Next)
- ⏳ TODO: Migrate qwen_pipeline.cpp to use `contract.load()`
- ⏳ TODO: Remove manual offset calculations
- ⏳ TODO: Run parity test - verify all ranks pass

### Phase 3: Cleanup (Final)
- ⏳ TODO: Remove gguf_dimensions field (no longer needed)
- ⏳ TODO: Remove dimension swap from ModelLoader
- ⏳ TODO: Update documentation

## Known Limitations

1. **Replicated Weights with Transpose**: Not yet implemented
   - All replicated weights (embeddings, norms) currently don't need transpose
   - Would throw error if contract specifies `needs_transpose_data=true` + `REPLICATED`

2. **Only 2D Tensors**: Current implementation handles 2D weight matrices only
   - 1D tensors (norms) work but don't need special handling
   - 3D+ would require extension

3. **No Tiled/Block Distribution**: Currently supports:
   - REPLICATED (full copy on all ranks)
   - ROW_SLICED (first dimension distributed)
   - COL_SLICED (second dimension distributed)
   - Does NOT support block/tiled distributions (future work)

## Migration Guide

To migrate a pipeline to contract-driven loading:

```cpp
// OLD CODE:
const int offset = mpi_rank * heads_per_rank * head_dim;
const int count = heads_per_rank * head_dim;
auto weight = TensorFactory::create_simple({count, d_model});
loader.loadTensorRowShard(name, offset, count, weight->data());

// NEW CODE:
auto weight = contract.load(loader, config, mpi_rank, mpi_size, layer);
// Done! All complexity handled internally.
```

## Files Modified

```
src/mpi_slicing_helper.h        (NEW - 118 lines)
src/mpi_slicing_helper.cpp      (NEW - 270 lines)
src/weight_contracts.h          (MODIFIED - added 6 fields, 1 method)
src/model_loader.h              (MODIFIED - added loadTensorColumnShard)
src/model_loader.cpp            (MODIFIED - added wrapper implementation)
CMakeLists.txt                  (MODIFIED - added mpi_slicing_helper.cpp)
```

## Next Steps

1. **Test the Infrastructure** - Write unit test for calculate_slice()
2. **Migrate qwen_pipeline.cpp** - Replace manual loading with contract.load()
3. **Run Parity Test** - Verify all weights pass on both ranks
4. **Cleanup** - Remove deprecated dimension swap code
5. **Document** - Update WEIGHT_MATRIX_CONVENTIONS.md

## Related Issues

- Original bug: Rank 1 Q projection 12.3 max error vs PyTorch
- Root cause: Dimension swap + manual offset calculation mismatch
- Solution: Contract-driven loading with automatic dimension handling

## References

- Design doc: `changelog/2025-01-12-dimension-swap-bug-and-mpi-slicing-refactor-design.md`
- Weight contracts: `src/weight_contracts.h`
- GGUF conventions: `docs/WEIGHT_MATRIX_CONVENTIONS.md`
