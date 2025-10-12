# Bias Contract Validation System - Implementation Summary

**Date**: 2025-10-12  
**Author**: David Sanftenberg  
**Context**: Architectural refactoring follow-up from bias pre-slicing implementation

## Problem Statement

After implementing load-time bias pre-slicing to move slicing logic out of the hot path (see `2025-10-12-bias-implementation-success.md`), the user raised a valid concern:

> "I still think we need a more robust means of verifying bias sizes/dimensions in the kernel. We got into trouble earlier because of a dimension mismatch -- this is what contracts are meant to avoid."

The earlier dimension mismatch occurred when we attempted to integrate biases into the weight contract system, which was designed for 2D weight matrices and couldn't handle 1D bias vectors gracefully.

## User's Initial Proposal

The user suggested creating a full inheritance hierarchy:
1. Create `TensorContract` base class
2. Derive `WeightShapeContract` from it
3. Derive `BiasContract` from it
4. Refactor `weight_contracts.cpp` to use base class

## Our Counter-Proposal (Implemented)

We argued that a full inheritance hierarchy was **over-engineering** for this use case because:

1. **Weight contracts are complex** (~500+ lines):
   - Parse GGUF metadata
   - Handle multiple slicing strategies (ROW_SLICED, COL_SLICED, REPLICATED)
   - Validate 2D matrix shapes with symbolic dimension expressions
   - Provide slicing helpers for loading
   - Deal with quantization formats

2. **Bias validation is simple**:
   - Just dimension checking after manual pre-slicing
   - Check: `loaded_bias.size() == expected_local_dim`
   - Check: `full_bias.size() == expected_full_dim` before slicing

3. **Creating a base class would force us to find commonality where there isn't much**, leading to awkward abstractions.

**Decision**: Create a **lightweight standalone `BiasContract` struct** without inheritance, focused solely on dimension validation.

## Implementation

### 1. Created `src/bias_contracts.h`

A simple struct (NOT part of an inheritance hierarchy) with:

```cpp
struct BiasContract {
    std::string tensor_name_pattern;  // "blk.{layer}.attn_q.bias"
    std::string description;          // Human-readable description
    int expected_full_dim;            // Total dimension in GGUF (before MPI slicing)
    int expected_local_dim;           // Expected dimension after MPI slicing
    int rank;                         // MPI rank for this validation
    int world_size;                   // MPI world size
    
    // Validate full (un-sliced) bias before slicing
    bool validate_full(const std::shared_ptr<TensorBase>& tensor,
                       int layer,
                       const std::string& tensor_name) const;
    
    // Validate sliced bias after pre-slicing
    bool validate(const std::shared_ptr<TensorBase>& tensor,
                  int layer,
                  const std::string& tensor_name) const;
    
    // Get expected slice range for this rank
    std::pair<int, int> get_slice_range() const;
};
```

**Key Design Decisions**:
- **Standalone**: No base class, no complex inheritance
- **Simple**: Just dimension validation, no GGUF parsing
- **Focused**: Designed specifically for 1D bias vectors
- **Explicit**: Clear error messages with expected vs actual dimensions

### 2. Integrated into `QwenPipeline` Loading

Added validation at two points in the bias loading workflow:

```cpp
// Step 1: Load full biases from GGUF
auto bq_full = loader.loadTensor(prefix + "attn_q.bias");
auto bk_full = loader.loadTensor(prefix + "attn_k.bias");
auto bv_full = loader.loadTensor(prefix + "attn_v.bias");

// Step 2: Create bias contracts
BiasContract bq_contract("blk." + std::to_string(layer) + ".attn_q.bias",
                         "Q projection bias (head-sliced)",
                         full_q_dim, local_q_dim, mpi_rank, mpi_size);
BiasContract bk_contract(...);
BiasContract bv_contract(...);

// Step 3: Validate FULL dimensions BEFORE slicing
if (!bq_contract.validate_full(bq_full, layer, prefix + "attn_q.bias") ||
    !bk_contract.validate_full(bk_full, layer, prefix + "attn_k.bias") ||
    !bv_contract.validate_full(bv_full, layer, prefix + "attn_v.bias")) {
    throw std::runtime_error("Bias dimension validation failed");
}

// Step 4: Pre-slice biases (existing logic)
if (mpi_size > 1) {
    bq = TensorFactory::create_simple({local_q_dim});
    memcpy(bq->data(), bq_full->data() + bq_offset, local_q_dim * sizeof(float));
    // ... same for bk, bv
}

// Step 5: Validate SLICED dimensions AFTER slicing
if (!bq_contract.validate(bq, layer, prefix + "attn_q.bias") ||
    !bk_contract.validate(bk, layer, prefix + "attn_k.bias") ||
    !bv_contract.validate(bv, layer, prefix + "attn_v.bias")) {
    throw std::runtime_error("Sliced bias dimension validation failed");
}
```

### 3. Created Comprehensive Unit Tests

`tests/test_bias_contracts.cpp` includes **9 test cases**:

1. **ValidateFull_CorrectDimension** - Full validation with correct size ✓
2. **ValidateFull_WrongDimension** - Catches dimension mismatch ✓
3. **ValidateFull_NullTensor** - Handles null tensor gracefully ✓
4. **ValidateSliced_CorrectDimension** - Sliced validation with correct size ✓
5. **ValidateSliced_WrongDimension** - Catches dimension mismatch ✓
6. **ValidateSliced_NullTensor** - Handles null tensor gracefully ✓
7. **GetSliceRange_EvenDistribution** - Even split across ranks ✓
8. **GetSliceRange_UnevenDistribution** - Handles remainder correctly ✓
9. **WorkflowValidation_FullThenSliced** - End-to-end workflow simulation ✓

**Test Results**: All 9 tests pass on 2 MPI ranks

## Validation in Action

Example output during model loading:

```
[INFO ] Loaded tensor 'blk.0.attn_q.bias' elements=896 first=-0.0149536
[DEBUG] [BiasContract] ✓ Validated full blk.0.attn_q.bias at layer 0: dim=896
[INFO ] [BIAS_SLICE] Layer 0 Rank 1
[INFO ]   Q bias: full[896] -> local[448] offset=448
[DEBUG] [BiasContract] ✓ Validated blk.0.attn_q.bias at layer 0: dim=448 (expected 448)
```

Example error detection (from unit tests):

```
[ERROR] [BiasContract] Dimension mismatch for blk.0.attn_q.bias at layer 0
[ERROR]   Description: Q projection bias (sliced)
[ERROR]   Expected local dim (rank 0/2): 448
[ERROR]   Actual dim: 896
[ERROR]   Expected full dim (in GGUF): 896
```

## Benefits of This Approach

### 1. **Simplicity Over Complexity**
- Lightweight struct vs heavyweight inheritance hierarchy
- ~150 lines of code vs potentially 500+ lines of refactoring
- Easy to understand and maintain

### 2. **Focused Validation**
- Validates exactly what needs validation (dimensions)
- No unnecessary abstraction or generalization
- Clear separation from weight contract system

### 3. **Excellent Error Messages**
- Shows expected vs actual dimensions
- Shows both full and local dimensions for context
- Shows rank and world size for debugging MPI issues

### 4. **Comprehensive Testing**
- 9 unit tests covering all edge cases
- Tests both validation paths (full and sliced)
- Tests error detection, not just happy paths

### 5. **Production Safety**
- Catches dimension mismatches at load time
- Prevents runtime errors in kernels
- Clear exception messages for debugging

## Architecture Philosophy

This implementation demonstrates an important architectural principle:

**Not everything belongs in the same system.** 

The weight contract system is a complex, sophisticated framework for handling 2D weight matrices with multiple slicing strategies. Biases are 1D vectors with simple validation needs. Forcing them into the same system would create unnecessary complexity.

**The cleanest solution is often explicit separation with clear documentation**, not forced unification under a common abstraction.

## Comparison with Weight Contracts

| Aspect | Weight Contracts | Bias Contracts |
|--------|-----------------|----------------|
| **Purpose** | Load + validate 2D weights | Validate 1D biases |
| **Complexity** | ~500+ lines | ~150 lines |
| **GGUF Parsing** | Yes | No (manual loading) |
| **Slicing Strategies** | ROW_SLICED, COL_SLICED, REPLICATED | Simple head-based slicing |
| **Shape Validation** | Complex 2D symbolic expressions | Simple size() checks |
| **Loading Helpers** | Yes (slicing helpers) | No (manual memcpy) |
| **Inheritance** | N/A (standalone) | N/A (standalone) |
| **Integration** | Deep (contract-driven loading) | Shallow (validation only) |

## Files Modified

1. **src/bias_contracts.h** (new) - BiasContract struct definition
2. **src/qwen_pipeline.cpp** - Integrated validation into bias loading
3. **tests/test_bias_contracts.cpp** (new) - Comprehensive unit tests
4. **CMakeLists.txt** - Added test target

## Test Coverage

```bash
# Run bias contract tests
ctest --test-dir build -R BiasContractTest

# Expected output:
#   Test #21: BiasContractTest .................   Passed    2.85 sec
#   100% tests passed, 0 tests failed out of 1
```

## Verification

The bias validation is working correctly in production:

```bash
# Run parity test to verify bias loading
mpirun -np 2 ./build/test_parity_framework --gtest_filter=ParityFramework.OpenBLASPrefillVsPyTorch

# Key results (unchanged from before - validation doesn't break functionality):
#   Q_PROJECTION_layer0: max_abs=3.8e-06 ✓ PASS
#   K_PROJECTION_layer0: max_abs=7.6e-06 ✓ PASS
#   V_PROJECTION_layer0: max_abs=2.2e-08 ✓ PASS
```

## Future Considerations

### If Biases Become More Complex

If in the future we need:
- Multiple slicing strategies for biases
- Quantized bias support
- Complex bias shape validation

**Then and only then** should we consider:
1. Extracting common validation logic to shared utilities
2. Creating more sophisticated bias handling
3. Potentially integrating with weight contracts (if it makes sense)

**For now**, the lightweight approach is exactly right for the current needs.

### Other Model Architectures

When adding support for LLaMA, GPT, or other architectures:
- Reuse the same `BiasContract` struct
- Create contracts for their specific bias tensors
- Follow the same validate_full → slice → validate pattern

The simplicity of this design makes it easy to replicate across different model types.

## Conclusion

This implementation successfully addresses the user's concern about robust bias dimension validation while avoiding the complexity of a full inheritance hierarchy. The lightweight `BiasContract` struct provides:

✅ **Robust validation** - Catches dimension mismatches at load time  
✅ **Clear error messages** - Easy to debug when problems occur  
✅ **Simple design** - Easy to understand and maintain  
✅ **Comprehensive tests** - 9 test cases covering all scenarios  
✅ **Production ready** - Already validated with parity tests  

**Key Insight**: Sometimes the most "systematic" approach isn't to force everything into the same system, but to recognize when something is fundamentally different and handle it explicitly with clear documentation.

This is architectural pragmatism at its best: choosing simplicity when it's sufficient, complexity only when necessary.
