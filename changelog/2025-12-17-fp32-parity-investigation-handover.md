# FP32 Parity Investigation Handover

**Date**: December 17, 2025  
**Branch**: `feature/typed-residuals`  
**Status**: Active Investigation - Root cause not yet identified

## Problem Statement

The `Test__Pipeline_vs_LayerExecutor_FP32_Parity.PrefillParity` test is failing. Two code paths that should produce identical outputs are diverging:

1. **Legacy Path**: `Qwen2Pipeline::attention_block()` using `FusedGEMM::execute()`
2. **Executor Path**: `Qwen2LayerExecutor` using `FusedQKVGEMMStage` which calls `QuantisedGemmKernel::multiply_fused()`

Both paths receive **identical normalized input** for Layer 0, but produce **different Q projection outputs**.

## Test Command

```bash
cd /workspaces/llaminar
cmake --build build_v2 --target v2_integration_pipeline_vs_layer_executor_fp32 --parallel
export LLAMINAR_LOG_LEVEL=INFO
timeout 300 mpirun --allow-run-as-root -np 1 ./build_v2/tests/v2/v2_integration_pipeline_vs_layer_executor_fp32 --gtest_filter="*PrefillParity"
```

## Key Evidence

### Layer 0 Inputs Are Identical

Both paths receive the same normalized input for Layer 0:
```
normalized[0:4] = 0.2139718533, 0, -0.06211324409, -0.1548501253
```

### Layer 0 Q8_1 Blocks Are Identical

Both paths quantize to the same Q8_1 blocks:
```
d=5995 sum_qs=275 qs[0:4]=118,0,-34,-85
```

### Layer 0 Q Outputs Are DIFFERENT

Despite identical inputs and identical Q8_1 quantization:

- **Legacy Q[0:4]**: `-0.004438523203, 0.1020245552, -0.1222847253, -0.4016433954`
- **Executor Q[0:4]**: `0.01051509008, 0.07651185989, -0.01876909845, -0.2659012079`

### Final Test Results

```
Top-1 match: NO (Legacy=11, Executor=116991)
Top-5 overlap: 0/5 (0%)
Cosine similarity: -0.064689
Max abs diff: 19.34
```

## Files Under Investigation

### Primary Suspects

1. **`src/v2/kernels/cpu/gemm_v4/FusedGEMM.cpp`** (Lines 90-220)
   - Legacy path's GEMM execution
   - Calls `quantize_activations()` then `multiply_with_precomputed_q8_1()`
   - Uses kernels from `KernelFactory::getOrCreateGemm()`

2. **`src/v2/execution/ComputeStage.cpp`** (Lines 348-430)
   - `FusedQKVGEMMStage::execute()` - Executor path
   - Calls `gemm_q->multiply_fused()` which delegates to `multiply_fused_multi()`

3. **`src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h`** (Lines 628-760)
   - `multiply_fused_multi()` - static method for fused projections
   - `multiply_fused()` (Line 803) - instance method that delegates to above

4. **`src/v2/kernels/KernelFactory.cpp`** (Lines 1863-2060)
   - `getOrCreateGemm()` - kernel caching and creation
   - Packs weights into `tensor->cache_`

### Supporting Files

- **`src/v2/pipelines/qwen/Qwen2Pipeline.cpp`** - Legacy pipeline (attention_block at line 733)
- **`src/v2/pipelines/qwen/Qwen2LayerExecutor.cpp`** - Executor implementation
- **`tests/v2/integration/Test__Pipeline_vs_LayerExecutor_FP32_Parity.cpp`** - Test file

## What Has Been Ruled Out

### ✅ Kernel Cache Statefulness

Disabled the kernel cache entirely (lines 1886-1913 in KernelFactory.cpp wrapped in `#if 0`). Test still fails with identical results. **Re-enabled cache** - not the issue.

### ✅ Input Data Mismatch

Both paths log identical:
- `normalized[0:4]` values
- First Q8_1 block (`d`, `sum_qs`, `qs[0:4]`)

### ✅ SwiGLU Buffer Mismatch (Fixed Earlier)

Previously fixed buffer assignment in `Qwen2LayerExecutor.cpp`:
- Line 692: Changed `swiglu_params.output = buffers.ffn_output` → `buffers.up`
- Line 718: Changed down_proj input from `buffers.ffn_output` → `buffers.up`

### ✅ Static JIT Kernel State

The JIT kernels (`QuantisedGemmJit_M1`, `QuantisedGemmJit_M2`) only contain generated code, no mutable state. The code generation happens once at static initialization.

## Current Hypothesis

**Different packed weights are being used** despite the same input tensor pointer.

Evidence:
- Both paths get kernels via `KernelFactory::getOrCreateGemm(tensor)`
- The tensor pointer appears to be the same
- But the GEMM outputs differ

Possible causes:
1. **Weight tensor `cache_` corruption** - The packed weights stored in `tensor->cache_` might be getting overwritten or corrupted between uses
2. **Different packing** - Despite same tensor pointer, somehow different weight packing is occurring
3. **Kernel selection difference** - FusedGEMM creates kernels directly, FusedQKVGEMMStage uses interface method

## Code Path Difference

### Legacy (FusedGEMM::execute)

```cpp
// FusedGEMM.cpp line 145-200
size_t buffer_size = gemm_kernels_[0]->get_quantized_activation_buffer_size(m, k);
std::vector<uint8_t> q8_1_buffer(buffer_size);
success = gemm_kernels_[0]->quantize_activations(input, q8_1_buffer.data(), m, k);
// Then loops calling:
success = gemm_kernels_[i]->multiply_with_precomputed_q8_1(q8_1_buffer.data(), ...);
```

### Executor (FusedQKVGEMMStage::execute → multiply_fused → multiply_fused_multi)

```cpp
// QuantisedGemmKernel.h line 680-760
std::vector<uint8_t> q8_1_buffer(buffer_size);
success = projections[0].kernel->quantize_activations(input, q8_1_buffer.data(), m, k);
// Then loops calling:
success = proj.kernel->multiply_with_precomputed_q8_1(q8_1_buffer.data(), ...);
```

Both should be equivalent, but they're getting different results.

## Next Steps to Investigate

1. **Compare packed_weights pointers and data**
   - Add logging to compare `pw.packed_data.data()` between FusedGEMM and multiply_fused_multi
   - Verify the actual packed weight bytes are identical

2. **Check kernel instance identity**
   - Are Legacy and Executor getting the same kernel instances for the same weight tensors?
   - Log kernel pointer and tensor pointer in both paths

3. **Verify weight tensor identity**
   - Confirm `layer.wq.get()` in Legacy == `params_.wq` in Executor
   - Both should come from the same `model_ctx_->getWeight()` call

4. **Byte-level comparison of packed weights**
   - If kernel pointers differ, compare first N bytes of `packed_weights_.packed_data`
   - Different packed data = different output (expected)
   - Same packed data = bug in GEMM computation itself

## Debug Logging Added

Current logging in `QuantisedGemmKernel.h` line 688:
```cpp
LOG_INFO("[multiply_fused_multi] input[0:8]=" << ...);
```

Current logging in `FusedGEMM.cpp` line 109:
```cpp
LOG_INFO("[FusedGEMM] input ptr=" << ... << " input[0:4]=" << ...);
```

Both paths produce identical input logs but different output logs.

## Model Under Test

- **Model**: `models/qwen2.5-0.5b-instruct-q4_0.gguf`
- **Architecture**: Qwen2, 24 layers
- **Dimensions**: d_model=896, d_ff=4864, n_heads=14, n_kv_heads=2, head_dim=64
- **Test Prompt**: "The quick brown fox jumps over the lazy dog" (9 tokens)

## Useful Commands

```bash
# Build
cmake --build build_v2 --target v2_integration_pipeline_vs_layer_executor_fp32 --parallel

# Run with INFO level (recommended)
export LLAMINAR_LOG_LEVEL=INFO && timeout 300 mpirun --allow-run-as-root -np 1 ./build_v2/tests/v2/v2_integration_pipeline_vs_layer_executor_fp32 --gtest_filter="*PrefillParity"

# Run with DEBUG level (verbose)
export LLAMINAR_LOG_LEVEL=DEBUG && timeout 300 mpirun --allow-run-as-root -np 1 ./build_v2/tests/v2/v2_integration_pipeline_vs_layer_executor_fp32 --gtest_filter="*PrefillParity"

# Filter logs for key comparison
... 2>&1 | grep -E "FusedGEMM.*input\[|FusedQKVGEMMStage.*input\[|Q OUTPUT|EXEC_ATTN.*Q\[|LEGACY_ATTN.*Q\["
```

## Summary

The bug is in Layer 0 Q projection. Same input → same Q8_1 quantization → **different GEMM output**. The most likely cause is that the two paths are using different packed weight data despite appearing to use the same weight tensor. The next agent should focus on comparing the actual packed weight pointers and bytes between the two code paths.
