# Weight Slicing Contract System Implementation

**Date**: 2025-01-27  
**Author**: David Sanftenberg  
**Status**: ✅ Complete and Validated

## Summary

Implemented a comprehensive weight slicing contract system that validates MPI-distributed weight loading with explicit slicing metadata. The system provides fail-fast validation, clear error messages, and executable documentation of the weight distribution strategy.

## Background

### Problem
- Weight validation was disabled in MPI mode ("skip validation in MPI")
- No structured way to validate sliced weights had correct dimensions per rank
- Difficult to diagnose weight loading bugs across distributed ranks
- No executable documentation of which weights are sliced vs replicated

### Root Cause
- Existing `WeightShapeContract` system only supported full (non-sliced) dimensions
- Validation couldn't distinguish between row-sliced, column-sliced, and replicated weights
- No MPI context passed through validation chain

## Solution Architecture

### Core Extensions

#### 1. WeightSliceType Enum
```cpp
enum class WeightSliceType {
    REPLICATED,   // Full weight on every rank
    ROW_SLICED,   // First dimension sliced by heads (Q/K/V)
    COL_SLICED    // Second dimension sliced by heads (O)
};
```

#### 2. Extended WeightShapeContract
Added slicing metadata and MPI-aware validation:
```cpp
struct WeightShapeContract {
    WeightSliceType slice_type;
    std::string slice_parameter;  // "n_head", "n_head_kv", "d_ff"
    
    // MPI-aware validation
    void validate_with_mpi(tensor, cfg, mpi_rank, mpi_size, layer_idx);
    
    // Calculate expected sliced dimensions
    std::vector<int> evaluate_sliced(cfg, mpi_rank, mpi_size);
    
    // Enhanced error messages with MPI context
    std::string format_error_mpi(actual_shape, mpi_rank, mpi_size);
};
```

#### 3. ModelWeightContracts MPI Methods
```cpp
struct ModelWeightContracts {
    void validate_global_with_mpi(weights, cfg, mpi_rank, mpi_size);
    void validate_layer_with_mpi(weights, cfg, layer_idx, mpi_rank, mpi_size);
};
```

#### 4. Updated Qwen Contracts
Defined explicit slicing strategy for all weights:

**Attention Weights** (Tensor Parallel by Attention Heads):
- `blk.{i}.attn_q.weight`: ROW_SLICED by n_head → [448, 896] per rank
- `blk.{i}.attn_k.weight`: ROW_SLICED by n_head_kv → [64, 896] per rank (GQA)
- `blk.{i}.attn_v.weight`: ROW_SLICED by n_head_kv → [64, 896] per rank
- `blk.{i}.attn_output.weight`: COL_SLICED by n_head → [896, 448] per rank

**FFN Weights** (Replicated - ReplicatedDataParallel mode):
- `blk.{i}.ffn_gate.weight`: REPLICATED → [4864, 896] full on every rank
- `blk.{i}.ffn_up.weight`: REPLICATED → [4864, 896] full on every rank  
- `blk.{i}.ffn_down.weight`: REPLICATED → [896, 4864] full on every rank

**Global Weights** (Replicated):
- `token_embd.weight`: REPLICATED → [151669, 896]
- `blk.{i}.attn_norm.weight`: REPLICATED → [896]
- `blk.{i}.ffn_norm.weight`: REPLICATED → [896]
- `output_norm.weight`: REPLICATED → [896]
- `output.weight`: REPLICATED → [151669, 896]

**Rationale**: FFN weights are currently replicated per `docs/tensor_parallel_architecture.md`:
- Mode: ReplicatedDataParallel for models <32B params
- No memory pressure at Qwen 0.5B scale (~500MB per rank)
- Avoids AllReduce overhead, simpler for small/medium models
- Future: FFN slicing planned when model size exceeds per-socket memory

## Implementation Details

### Files Modified

**src/weight_contracts.h** (comprehensive extension):
- Lines 23-43: Added WeightSliceType enum and extended constructor
- Lines 60-109: Added validate_with_mpi() method
- Lines 150-205: Added evaluate_sliced() for dimension calculation
- Lines 207-230: Added get_slice_parameter_value() helper
- Lines 275-335: Added format_error_mpi() for enhanced errors
- Lines 420-475: Added validate_global/layer_with_mpi()
- Lines 580-650: Updated getQwenWeightContracts() with slicing metadata

**src/qwen_pipeline_adapter.h** (added MPI validation method):
- Lines 55-95: Added validate_with_mpi() alongside existing validate()

**src/qwen_pipeline_adapter.cpp** (enabled MPI validation):
- Lines 29-56: Removed "skip validation in MPI" workaround
- Now calls weights->validate_with_mpi(cfg, mpi_rank, mpi_size)
- Works for both single-rank and multi-rank configurations

### Validation Logic

```cpp
// In QwenPipelineAdapter::loadWeights()
int mpi_rank = 0, mpi_size = 1;
MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

// Validate with MPI context
weights->validate_with_mpi(cfg, mpi_rank, mpi_size);

LOG_INFO("✓ All weights validated with MPI slicing (rank " 
         << mpi_rank << "/" << mpi_size << ")");
```

**evaluate_sliced() logic**:
```cpp
std::vector<int> evaluate_sliced(cfg, mpi_rank, mpi_size) {
    if (slice_type == REPLICATED) {
        return base_shape;  // Full dimensions
    }
    
    int param_value = get_slice_parameter_value(slice_parameter, cfg);
    
    if (param_value % mpi_size != 0) {
        throw std::runtime_error("Cannot evenly divide " + slice_parameter 
                                 + "=" + std::to_string(param_value) 
                                 + " by " + std::to_string(mpi_size) + " ranks");
    }
    
    int local_value = param_value / mpi_size;
    
    if (slice_type == ROW_SLICED) {
        return {local_value * rows_per_unit, base_shape[1]};  // Q/K/V
    } else {  // COL_SLICED
        return {base_shape[0], local_value * cols_per_unit};  // O
    }
}
```

### Error Messages

Enhanced error messages with MPI context:

**Example 1: Row-sliced mismatch**
```
Weight 'blk.0.attn_q.weight' validation failed (rank 1/2):
  Expected shape: [448, 896] (row-sliced by n_head=14, 7 per rank)
  Actual shape: [896, 896]
  Slice type: ROW_SLICED
  Slice parameter: n_head (14 total, 7 per rank)
```

**Example 2: Column-sliced mismatch**
```
Weight 'blk.0.attn_output.weight' validation failed (rank 0/2):
  Expected shape: [896, 448] (column-sliced by n_head=14, 7 per rank)
  Actual shape: [896, 896]
  Slice type: COL_SLICED
  Slice parameter: n_head (14 total, 7 per rank)
```

**Example 3: Replicated weight OK**
```
Weight 'blk.0.ffn_gate.weight' validation passed (rank 1/2):
  Expected shape: [4864, 896] (replicated)
  Actual shape: [4864, 896]
  Slice type: REPLICATED
```

## Testing Results

### Build Status
```bash
cmake --build build --target llaminar_core test_parity_framework --parallel
```
**Result**: ✅ Build successful, no errors

### Validation Status
```bash
mpirun -np 2 ./build/test_parity_framework \
  --gtest_filter="*TrueIncrementalDecodeVsPyTorch"
```

**Result**: ✅ Validation passing on both ranks
```
[INFO] [WeightLoad] MPI-aware loading: world_size=2 q_heads=14 kv_heads=2 
       q_heads_per_rank=7 kv_heads_per_rank=1

[INFO] [WeightLoad] blk.0.attn_q.weight row_slice=[0:448] shape=[448,896]
[INFO] [WeightLoad] blk.0.attn_k.weight row_slice=[0:64] shape=[64,896]
[INFO] [WeightLoad] blk.0.attn_v.weight row_slice=[0:64] shape=[64,896]
[INFO] [WeightLoad] blk.0.attn_output.weight col_slice=[0:448] shape=[896,448]
[INFO] [WeightLoad] blk.0.ffn_gate.weight replicated shape=[4864,896]
[INFO] [WeightLoad] blk.0.ffn_up.weight replicated shape=[4864,896]
[INFO] [WeightLoad] blk.0.ffn_down.weight replicated shape=[896,4864]

[INFO] ✓ All weights validated with MPI slicing (rank 0/2)
[INFO] ✓ All weights validated with MPI slicing (rank 1/2)
```

### Validated Scenarios

1. **Single-rank validation**: ✅ Works (backward compatible with rank=0, size=1)
2. **Multi-rank validation**: ✅ Works (MPI-aware dimension checking)
3. **Attention weight slicing**: ✅ Correct (Q/K/V row-sliced, O column-sliced)
4. **FFN weight replication**: ✅ Correct (gate/up/down replicated per architecture)
5. **Error detection**: ✅ Works (clear messages with slicing context)

## Benefits

### 1. Fail-Fast Validation
- Detects weight loading bugs immediately at load time
- No silent failures that cause numerical divergence later
- Validates every weight on every rank

### 2. Executable Documentation
- Contracts serve as machine-readable specification of slicing strategy
- Single source of truth for weight distribution
- Prevents drift between code and documentation

### 3. Clear Diagnostics
- Enhanced error messages explain expected vs actual dimensions
- Shows slicing type (row/column/replicated) and parameters
- Includes MPI context (rank X/Y) in all messages

### 4. Maintainability
- Easy to add new models: define contracts once
- Easy to evolve architecture: update contracts, validation follows
- Backward compatible: single-rank still works

### 5. Foundation for Future
- Ready for FFN slicing when model size exceeds memory
- Supports adding new slice types (e.g., channel-wise, block-wise)
- Extensible to other parallelism strategies (pipeline, data parallel)

## Future Enhancements

### 1. FFN Slicing Support (when needed)
When model size exceeds per-socket memory (~40-50GB):
```cpp
// Future contracts
{
    "blk.{i}.ffn_gate.weight",
    WeightShapeContract({d_ff, d_model}, {}, 
                       WeightSliceType::ROW_SLICED, "d_ff")
}
```

### 2. Dynamic Validation Levels
- Environment variable `LLAMINAR_WEIGHT_VALIDATION_LEVEL`:
  - `strict`: Validate all weights, fail on mismatch (current default)
  - `warn`: Validate but only warn on mismatch
  - `off`: Skip validation for performance

### 3. Weight Distribution Profiling
- Add timing for sliced vs replicated loading
- Memory usage tracking per rank
- Communication overhead measurement

### 4. Contract Generation from Architecture
- Auto-generate contracts from model config
- Support for dynamic architectures (MoE, sparse models)

## Related Work

**Previous Fixes**:
- `changelog/2025-01-27_qkvo_weight_slicing_fix.md`: Fixed Q/K/V/O slicing bugs
- `docs/tensor_parallel_architecture.md`: Architecture specification

**Code Integration**:
- Weight loading: `src/qwen_pipeline.cpp` (lines 1027-1175)
- Model metadata: `src/model_weights_provider.cpp` (lines 155-195)
- Test verification: `tests/weight_verifier.cpp` (lines 172-260)

## Conclusion

The weight slicing contract system is **production-ready** and provides:
- ✅ Validated correctness on both MPI ranks
- ✅ Fail-fast detection of weight loading bugs
- ✅ Clear diagnostics with MPI-aware error messages
- ✅ Executable documentation of slicing strategy
- ✅ Foundation for future architectural evolution

The system successfully validates the current ReplicatedDataParallel architecture (attention sliced, FFN replicated) and is ready to support future enhancements like FFN slicing when needed.
