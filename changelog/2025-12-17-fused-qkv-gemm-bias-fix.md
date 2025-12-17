# Fix: Missing QKV Biases in FusedQKVGEMMStage (LayerExecutor Path)

**Date**: 2025-12-17
**Author**: GitHub Copilot (Claude Opus 4.5)
**Test**: `Test__Pipeline_vs_LayerExecutor_FP32_Parity.PrefillParity`

## Summary

Fixed a bug where `FusedQKVGEMMStage` was not passing attention biases (Q, K, V biases) to the GEMM kernel, causing different outputs between the legacy `Qwen2Pipeline::attention_block()` path and the `Qwen2LayerExecutor` path.

## Bug Symptoms

- Legacy pipeline and LayerExecutor produced different Q/K/V projection outputs
- All inputs appeared identical (Q8_1 activations, packed weights, comp/scales arrays)
- The JIT kernel itself was correct - it was receiving different `bias` pointers:
  - Legacy: `bias=0x1de62440` (valid pointer)
  - Executor: `bias=0` (null)

## Root Cause

The `FusedQKVGEMMStage::Params` struct had no bias fields, and the stage was creating `FusedProjectionDesc` objects with `nullptr` for bias:

```cpp
// BEFORE (bug)
std::vector<ITensorGemm::FusedProjectionDesc> projections = {
    {gemm_q, output_q_fp32, params_.n_q, nullptr, nullptr, false, "Q"},  // nullptr bias!
    ...
};
```

## Fix

1. **Added bias fields to `FusedQKVGEMMStage::Params`**:
```cpp
struct Params {
    // ... existing fields ...
    const float *bias_q = nullptr;  // NEW
    const float *bias_k = nullptr;  // NEW
    const float *bias_v = nullptr;  // NEW
};
```

2. **Pass biases to kernel**:
```cpp
// AFTER (fixed)
std::vector<ITensorGemm::FusedProjectionDesc> projections = {
    {gemm_q, output_q_fp32, params_.n_q, params_.bias_q, nullptr, false, "Q"},
    ...
};
```

3. **Added bias fields to `Qwen2LayerWeights`** (executor's weight struct)

4. **Updated weight mapping** in `Qwen2Pipeline::layer_executor_forward()` to include biases

5. **Updated `Qwen2LayerExecutor::buildComputeGraph()`** to extract and pass bias pointers

## Files Changed

| File | Change |
|------|--------|
| `src/v2/execution/ComputeStage.h` | Added `bias_q/k/v` to `FusedQKVGEMMStage::Params` |
| `src/v2/execution/ComputeStage.cpp` | Pass bias pointers through `FusedProjectionDesc` |
| `src/v2/pipelines/qwen/Qwen2LayerExecutor.h` | Added `q_bias/k_bias/v_bias` to `Qwen2LayerWeights` |
| `src/v2/pipelines/qwen/Qwen2LayerExecutor.cpp` | Extract bias pointers and set in QKV params |
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | Map bias weights when creating executor weights |
| `tests/v2/integration/Test__LayerExecutor_Q8_1_vs_FP32_Parity.cpp` | Load bias weights from model |

## Test Results

```
╔════════════════════════════════════════════════════════════════╗
║  Prefill Logits Comparison
╚════════════════════════════════════════════════════════════════╝
  Top-1 match: YES ✓ (Legacy=11, Executor=11)
  Top-5 overlap: 5/5 (100%)
  Cosine similarity: 1.000000
  Max abs diff: 0.0000e+00
  Mean abs diff: 0.0000e+00
```

## Debugging Journey

The investigation used extensive logging to narrow down the divergence:
1. Verified same kernel instance (same `this` pointer)
2. Verified same JIT code (same `kernel_ptr`)
3. Verified same Q8_1 activations (same checksum)
4. Verified same packed weights (same checksum + pointer)
5. Verified same comp/scales arrays (same data checksums)
6. **Found**: `bias` pointer was different (non-null vs null)

The key insight was logging ALL fields of `QuantisedGemmParams`, not just the obvious ones.

## Related Issues

- Decode parity tests (`IncrementalDecodeParity`, `MultiStepDecodeParity`) fail with a different bug (attention sequence length mismatch) - unrelated to this fix.

## Unit Test Added

A dedicated unit test for `FusedQKVGEMMStage` was added to prevent regression:

**File**: `tests/v2/unit/Test__FusedQKVGEMMStage.cpp`

**Key Test Cases**:

| Test | Description |
|------|-------------|
| `BiasIsAppliedWhenSet` | **Regression test**: Runs GEMM with and without bias, verifies `(output_with_bias - output_no_bias) == bias` |
| `PartialBiasOnly_Q` | Tests that only Q bias is applied when K/V biases are nullptr |
| `ExecuteWithoutBias` | Verifies basic GEMM works without any bias |
| `StageType` | Confirms `type() == GEMM_FUSED_QKV` |
| `EstimatedFlops` | Verifies FLOP estimation for 3 projections |
| `SupportsBackend` | Confirms support for CPU/GPU backends |
| `NullContextFails`, `NullInputFails`, `NullWeightFails`, `InvalidDimensionsFails` | Error handling tests |

**Run**:
```bash
ctest --test-dir build_v2 -R V2_Unit_FusedQKVGEMMStage --output-on-failure
```
