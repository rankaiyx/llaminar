# SwiGLU Formula Bug Fix and Regression Tests

**Date**: November 7, 2025
**Author**: David Sanftenberg

## Critical Bug Fixed

**CPUSwiGLUKernel had the gate/up arguments reversed**, causing 82.6% of FFN_SWIGLU outputs to fail parity with PyTorch.

### The Bug

**WRONG** (original implementation):
```cpp
// CPUSwiGLUKernel.cpp (BEFORE)
float sigmoid = 1.0f / (1.0f + std::exp(-g));  // Applied to gate!
float silu = g * sigmoid;
output[i] = silu * u;  // silu(gate) * up
```

**CORRECT** (fixed implementation):
```cpp
// CPUSwiGLUKernel.cpp (AFTER)
float sigmoid_u = 1.0f / (1.0f + std::exp(-u));  // Applied to up!
float silu_u = u * sigmoid_u;
output[i] = g * silu_u;  // gate * silu(up)
```

### Impact

**Before Fix**:
```
=== layer0_FFN_SWIGLU ===
  Elements:       43776
  Max abs diff:   7.1378
  Mean abs diff:  0.0367072
  Rel L2 norm:    0.636058
  Mismatches:     36168 / 43776 (82.6206%)
  Status:         ✗ FAILED
```

**After Fix**:
```
=== layer0_FFN_SWIGLU ===
  Elements:       43776
  Max abs diff:   2.86102e-06
  Mean abs diff:  3.64344e-08
  Rel L2 norm:    3.76318e-07
  Mismatches:     0 / 43776 (0%)
  Status:         ✓ PASSED
```

**Improvement**: Error reduced by **1,690,000× !** (0.636 → 3.76e-07)

## Root Cause Analysis

### PyTorch Reference (Correct)

```python
# python/reference/generate_qwen2_pipeline_snapshots.py (line 440-443)
# 3. SwiGLU activation: gate * silu(up)
# SiLU(x) = x * sigmoid(x)
silu_up = F.silu(up_proj)
swiglu_output = gate_proj * silu_up  # Correct: gate * silu(up)
```

### Llaminar V2 (Was Incorrect)

The bug was introduced when implementing CPUSwiGLUKernel. The SiLU activation was mistakenly applied to the **gate** input instead of the **up** input.

### Why This Matters

SwiGLU (Swish-Gated Linear Unit) is defined as:
```
SwiGLU(gate, up) = gate ⊙ (SiLU(up))
                 = gate ⊙ (up ⊙ σ(up))
```

Where:
- `⊙` = element-wise multiplication
- `σ(x)` = sigmoid(x) = 1 / (1 + exp(-x))
- `SiLU(x)` = x · σ(x) (Sigmoid Linear Unit, aka Swish)

The gate and up have **different roles**:
- **gate**: Controls information flow (modulates output)
- **up**: Transformed by non-linearity (SiLU) before modulation

Reversing them fundamentally changes the activation function's behavior.

## Regression Test Suite

Created comprehensive test suite to prevent future regressions: `tests/v2/unit/Test__SwiGLUParity.cpp`

### 5 Tests (All Passing)

#### 1. `KernelMatchesReference`
**Purpose**: Validate CPUSwiGLUKernel computes `gate * silu(up)` correctly

**Test Data**:
```cpp
gate = {1.0f, 2.0f, -1.0f, 0.0f, 3.0f}
up   = {2.0f, 1.0f,  3.0f, 4.0f, -2.0f}
```

**Result**: ✅ Max abs diff = 0 (perfect agreement)

#### 2. `CorrectFormulaVsWrongFormula`
**Purpose**: Prove that `gate * silu(up)` and `silu(gate) * up` differ significantly

**Test Data**:
```cpp
gate = {1.0f, 2.0f, -1.0f, 0.0f}
up   = {2.0f, 1.0f,  3.0f, 4.0f}

Correct: gate * silu(up)       = [ 1.761594,  1.4621172, -2.8577225,  0.0]
Wrong:   silu(gate) * up       = [ 1.4621172, 1.761594,  -0.8068243,  0.0]
Difference:                      [ 0.2994769, -0.2994769, -2.0508980,  0.0]
```

**Result**: ✅ Max abs diff = 2.05 (formulas produce very different outputs)

#### 3. `RealisticNeuralNetworkValues`
**Purpose**: Validate correctness on realistic FFN intermediate activations

**Test Data**: 32 elements (4 sequences × 8 features) with values typical of neural networks (-2.0 to +2.5)

**Result**: ✅ Max abs diff = 0 (threshold: 1e-5)

#### 4. `EdgeCases`
**Purpose**: Test extreme values (zeros, large positives/negatives, tiny values)

**Test Data**:
```cpp
gate = {0.0f,  10.0f, -10.0f,  1e-5f, -1e-5f}
up   = {0.0f, -10.0f,  10.0f, -1e-5f,  1e-5f}
```

**Result**: ✅ Max abs diff = 0 (threshold: 1e-5)

#### 5. `LargeBatch`
**Purpose**: Validate performance on production-sized batches

**Test Data**: 1024 elements (32 sequences × 32 features)

**Result**: ✅ Max abs diff = 0 (threshold: 1e-5)

## Integration

### CMake Build System

Added to `tests/v2/CMakeLists.txt`:
```cmake
add_executable(v2_test_swiglu_parity unit/Test__SwiGLUParity.cpp)
target_link_libraries(v2_test_swiglu_parity 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_SwiGLUParity 
    COMMAND $<TARGET_FILE:v2_test_swiglu_parity>
    LABELS "V2;Unit;FFN;RegressionTests;SwiGLU;ParityTesting;CPU;Activation"
    MPI_PROCS 1  # Single rank for unit testing
)
```

### CTest Integration

```bash
$ ctest -R "SwiGLUParity" --output-on-failure
Test #9: V2_Unit_SwiGLUParity .............   Passed    0.58 sec

100% tests passed, 0 tests failed out of 1
```

**Labels Applied**:
- `V2`, `Unit`, `FFN`, `RegressionTests`
- `SwiGLU`, `ParityTesting`, `CPU`, `Activation`

## E2E Test Impact

### Before SwiGLU Fix

**E2E Parity Status**: 3/5 tests passing
- ✅ Layer0_AttentionBlock (9/9 stages)
- ✅ SnapshotLoadingInfrastructure
- ❌ Layer0_FFNBlock (3/6 stages - **FFN_SWIGLU failing**)
- ❌ FinalNormAndLogits (likely cascading failure from FFN)

### After SwiGLU Fix

**E2E Parity Status**: **5/5 tests passing (COMPLETE PARITY!)**
- ✅ Layer0_AttentionBlock (9/9 stages)
- ✅ Layer0_FFNBlock (6/6 stages - **FFN_SWIGLU now passing!**)
- ✅ FinalNormAndLogits (2/2 stages)
- ✅ SnapshotLoadingInfrastructure
- ✅ All tests green!

## Cascade Effect

Fixing FFN_SWIGLU also fixed the downstream stages:

**FFN Block (After Fix)**:
```
=== layer0_FFN_NORM ===      rel_l2=2.04e-07  ✓ PASSED
=== layer0_FFN_GATE ===      rel_l2=2.59e-07  ✓ PASSED
=== layer0_FFN_UP ===        rel_l2=2.58e-07  ✓ PASSED
=== layer0_FFN_SWIGLU ===    rel_l2=3.76e-07  ✓ PASSED (WAS FAILING)
=== layer0_FFN_DOWN ===      rel_l2=3.99e-07  ✓ PASSED (WAS FAILING)
=== layer0_FFN_RESIDUAL ===  rel_l2=3.98e-07  ✓ PASSED (WAS FAILING)
```

**Final Stages (After Fix)**:
```
=== FINAL_NORM ===           rel_l2=7.22e-06  ✓ PASSED
=== LM_HEAD ===              rel_l2=6.92e-06  ✓ PASSED
```

## Historical Context

This was the **third critical bug** discovered during Qwen2 E2E parity debugging:

1. **Bias Tensor Bug**: PyTorch loaded 72 bias tensors despite `use_qkv_bias=False`
2. **RoPE Theta Bug**: Hardcoded `freq_base=10000.0` instead of `1000000.0` for Qwen2.5
3. **SwiGLU Formula Bug**: Applied SiLU to gate instead of up

Each bug blocked an entire class of pipeline stages from passing parity:
- Bug #1 → Blocked Q/K/V projections
- Bug #2 → Blocked RoPE stages
- Bug #3 → Blocked FFN stages + cascading failures

## Files Modified

### Source Code
- **`src/v2/kernels/cpu/CPUSwiGLUKernel.cpp`**: Fixed formula (8 lines changed)

### Tests
- **`tests/v2/unit/Test__SwiGLUParity.cpp`**: Created 5 comprehensive regression tests (270 lines)
- **`tests/v2/CMakeLists.txt`**: Added v2_test_swiglu_parity target

## Summary

**Status**: ✅ All 5 SwiGLU parity tests passing, E2E parity **COMPLETE** (5/5)
**Execution Time**: 0.58 seconds
**Lines of Code**: ~320 lines (test + build integration)
**Coverage**: Correct formula, wrong formula comparison, realistic data, edge cases, large batches

This regression test suite ensures future changes to SwiGLU don't reintroduce the formula reversal bug that broke FFN parity.
