# Precision Disambiguation: WeightPrecision vs ActivationPrecision

**Date**: November 7, 2025  
**Author**: David Sanftenberg  
**Issue**: Ambiguous `ComputePrecision` enum conflated two independent concerns  
**Solution**: Split into `WeightPrecision` (loading strategy) and `ActivationPrecision` (compute precision)

---

## Problem Statement

The original `ComputePrecision` enum was ambiguous and conflated two orthogonal concepts:

1. **Weight Loading Strategy**: How weights are stored in memory after loading from GGUF
   - Keep in original quantized format (IQ4_NL, Q6_K, etc.)
   - Convert to FP32/BF16/FP16/INT8 at load time

2. **Activation/Accumulation Precision**: What precision is used for intermediate computations
   - Hidden states between layers
   - GEMM accumulation buffers
   - Attention scores, softmax, RMSNorm outputs

**Example of confusion**:
- `ComputePrecision::FP32` meant:
  - "Dequantize all weights to FP32" **AND**
  - "Run all activations in FP32"
- But you might want:
  - Native quantized weights (IQ4_NL) with FP32 activations
  - FP32 weights with BF16 activations
  - INT8 weights with INT8 activations
  - Any combination that makes sense

This ambiguity made it impossible to express mixed-precision configurations like:
- "Load INT8 weights, run FP32 activations" (highest accuracy INT8 inference)
- "Keep IQ4_NL weights native, run BF16 activations" (memory-efficient BF16)

---

## Solution: Two Separate Enums

### 1. WeightPrecision (How weights are loaded)

```cpp
enum class WeightPrecision
{
    NATIVE,          // Keep weights in original GGUF format (default)
    CONVERT_TO_FP32, // Dequantize all weights to FP32 at load
    CONVERT_TO_BF16, // Dequantize all weights to BF16 at load
    CONVERT_TO_FP16, // Dequantize all weights to FP16 at load
    CONVERT_TO_INT8  // Dequantize all weights to INT8 at load
};
```

**Examples**:
- `NATIVE`: IQ4_NL weights stay as IQ4_NL (dequantized on-the-fly in GEMM kernels)
- `CONVERT_TO_FP32`: IQ4_NL weights converted to FP32 at load (no runtime dequant overhead)
- `CONVERT_TO_INT8`: Q8_0 weights converted to INT8 with separate scale factors

**Trade-offs**:
- `NATIVE`: Lowest memory (compressed), runtime dequant overhead
- `CONVERT_TO_FP32`: Highest memory (4 bytes/element), no runtime dequant, best for parity testing
- `CONVERT_TO_INT8`: Low memory (1 byte/element), enables AVX512-VNNI/CUDA INT8 kernels

### 2. ActivationPrecision (How activations are computed)

```cpp
enum class ActivationPrecision
{
    FP32, // 32-bit float activations (default, highest accuracy)
    BF16, // bfloat16 activations (Intel AMX, reduced bandwidth)
    FP16, // 16-bit float activations (ARM/GPU optimization)
    INT8  // 8-bit integer activations (AVX512-VNNI, CUDA INT8)
};
```

**Determines precision for**:
- Hidden states (layer outputs)
- Attention scores, softmax, context vectors
- GEMM accumulation buffers
- RMSNorm, SwiGLU intermediate results

**Trade-offs**:
- `FP32`: Highest accuracy, 4 bytes/element, baseline for validation
- `BF16`: ~1.5-2× faster on Intel AMX, 2 bytes/element, slight accuracy loss
- `FP16`: Faster on ARM/GPU, requires numerical stability care
- `INT8`: 4-8× faster on AVX512-VNNI/CUDA, significant accuracy trade-off

---

## Migration Guide

### Old API → New API

**PipelineConfig**:
```cpp
// OLD (ambiguous):
PipelineConfig config;
config.precision = ComputePrecision::FP32;

// NEW (explicit):
PipelineConfig config;
config.weight_precision = WeightPrecision::CONVERT_TO_FP32;
config.activation_precision = ActivationPrecision::FP32;
```

**ModelLoader**:
```cpp
// OLD:
auto tensor = loader.loadTensor("blk.0.attn_q.weight", 0, ComputePrecision::FP32);

// NEW:
auto tensor = loader.loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::CONVERT_TO_FP32);
```

**WeightManager**:
```cpp
// OLD:
auto mgr = std::make_shared<WeightManager>(loader, mpi_ctx, placement_map, 
                                           WeightDistributionStrategy::REPLICATED,
                                           ComputePrecision::FP32);

// NEW:
auto mgr = std::make_shared<WeightManager>(loader, mpi_ctx, placement_map,
                                           WeightDistributionStrategy::REPLICATED,
                                           WeightPrecision::CONVERT_TO_FP32);
```

### ComputePrecision Mapping

| Old `ComputePrecision` | New `WeightPrecision` | New `ActivationPrecision` |
|------------------------|----------------------|---------------------------|
| `MIXED` | `NATIVE` | `FP32` |
| `FP32` | `CONVERT_TO_FP32` | `FP32` |
| `BF16` | `CONVERT_TO_BF16` | `BF16` |
| `FP16` | `CONVERT_TO_FP16` | `FP16` |
| `INT8` | `CONVERT_TO_INT8` | `INT8` |
| `AUTO` | (resolve to specific values) | (resolve to specific values) |

### Backward Compatibility

The old `ComputePrecision` enum is **deprecated but still supported**:

```cpp
// DEPRECATED (still works):
enum class ComputePrecision
{
    MIXED, // Use WeightPrecision::NATIVE + ActivationPrecision::FP32
    FP32,  // Use WeightPrecision::CONVERT_TO_FP32 + ActivationPrecision::FP32
    BF16,  // Use WeightPrecision::CONVERT_TO_BF16 + ActivationPrecision::BF16
    FP16,  // Use WeightPrecision::CONVERT_TO_FP16 + ActivationPrecision::FP16
    INT8,  // Use WeightPrecision::CONVERT_TO_INT8 + ActivationPrecision::INT8
    AUTO   // Resolve to specific pair based on hardware
};
```

**Helper functions** for conversion:
```cpp
WeightPrecision computePrecisionToWeightPrecision(ComputePrecision old);
ActivationPrecision computePrecisionToActivationPrecision(ComputePrecision old);
```

**Backward-compatible wrapper**:
```cpp
// ModelLoader provides legacy method:
std::shared_ptr<TensorBase> loadTensorLegacy(const std::string &tensor_name,
                                             int device_idx,
                                             ComputePrecision precision);
// Internally converts to new API
```

---

## Use Cases Enabled

### 1. FP32 Parity Testing (Current)
```cpp
config.weight_precision = WeightPrecision::CONVERT_TO_FP32;
config.activation_precision = ActivationPrecision::FP32;
// All weights FP32, all activations FP32 → matches PyTorch
```

### 2. Memory-Efficient BF16 (Future)
```cpp
config.weight_precision = WeightPrecision::NATIVE; // Keep IQ4_NL compressed
config.activation_precision = ActivationPrecision::BF16; // 2× faster compute
// Weights stay quantized, activations use BF16 for speed
```

### 3. INT8 Performance Mode (Future)
```cpp
config.weight_precision = WeightPrecision::CONVERT_TO_INT8; // Dequant to INT8 once
config.activation_precision = ActivationPrecision::INT8; // All ops in INT8
// 4-8× faster on AVX512-VNNI/CUDA, end-to-end INT8 pipeline
```

### 4. Mixed-Precision INT8 (Highest Accuracy INT8)
```cpp
config.weight_precision = WeightPrecision::CONVERT_TO_INT8; // INT8 weights
config.activation_precision = ActivationPrecision::FP32; // FP32 activations
// INT8 GEMM benefits, FP32 accuracy for attention/softmax/RMSNorm
```

---

## Implementation Details

### Files Modified

**Core Enums**:
- `src/v2/pipelines/PipelineConfig.h`: Added `WeightPrecision`, `ActivationPrecision`, deprecated `ComputePrecision`
  - Conversion helpers: `computePrecisionToWeightPrecision()`, `computePrecisionToActivationPrecision()`
  - Updated `PipelineConfig` struct with both fields
  - Preserved old `precision` field for backward compatibility

**ModelLoader**:
- `src/v2/loaders/ModelLoader.h`: Updated `loadTensor()` signature to use `WeightPrecision`
  - Added `loadTensorLegacy()` for backward compatibility
- `src/v2/loaders/ModelLoader.cpp`: Updated implementation
  - Changed parameter from `ComputePrecision precision` to `WeightPrecision weight_precision`
  - Updated comments: "PRECISION MODE" → "WEIGHT PRECISION HANDLING"
  - Updated variable: `should_dequantize` → `should_convert`
  - Added `loadTensorLegacy()` wrapper that converts old enum to new

**WeightManager**:
- `src/v2/loaders/WeightManager.h`: Updated constructor to use `WeightPrecision`
  - Changed member: `precision_` → `weight_precision_`
- `src/v2/loaders/WeightManager.cpp`: Updated implementation
  - Updated constructor logging to show "Weight precision" instead of "Precision mode"
  - Updated log messages for new enum values (NATIVE, CONVERT_TO_FP32, etc.)
  - Updated `getReplicatedWeight()` to pass `weight_precision_` to loader

### Compilation Status

✅ **All changes compile successfully**  
⚠️ **Tests need updating** to use new API (currently using deprecated `ComputePrecision`)

---

## Testing Strategy

### Phase 1: Backward Compatibility ✅
- All existing code continues working with deprecated `ComputePrecision`
- Conversion helpers automatically map to new enums
- No breaking changes for existing tests

### Phase 2: Incremental Migration
Update tests to use new API:
1. `tests/v2/unit/loaders/Test__ModelLoader.cpp`: Update `loadTensor()` calls
2. `tests/v2/unit/pipelines/Test__PipelineBase_PrecisionMode.cpp`: Update to test both fields
3. `tests/v2/e2e/Test__Qwen2FP32Parity.cpp`: Already using `CONVERT_TO_FP32` semantics (just needs API update)

### Phase 3: Activation Precision Implementation
- Implement BF16/FP16/INT8 activation kernels
- Add tests for mixed-precision configurations
- Validate numerical accuracy vs FP32 baseline

---

## Benefits

1. **Clarity**: Explicit separation of weight loading vs compute precision
2. **Flexibility**: Can mix any weight precision with any activation precision
3. **Correctness**: Eliminates ambiguity that could lead to bugs
4. **Documentation**: Code is self-documenting (names match intent)
5. **Future-Proof**: Enables advanced mixed-precision strategies

---

## Future Work

1. **Implement Missing Conversions**:
   - `dequantizeToBF16()` in ModelLoader
   - `dequantizeToFP16()` in ModelLoader

2. **Activation Precision Kernels**:
   - BF16 RMSNorm, Softmax, Attention (Intel AMX)
   - FP16 kernels for ARM/GPU
   - INT8 full-pipeline kernels (AVX512-VNNI, CUDA)

3. **PipelineConfig Validation**:
   - Warn if weight_precision and activation_precision mismatch causes inefficiency
   - Example: `CONVERT_TO_FP32` + `INT8` activations → wasting memory

4. **Hardware Auto-Selection**:
   - Update `selectOptimalPrecision()` to return both weight and activation precision
   - Example: Intel AMX → `NATIVE` weights + `BF16` activations

---

## Conclusion

This refactor **eliminates ambiguity** in Llaminar's precision handling by separating:
- **Weight loading strategy** (`WeightPrecision`): How weights are stored in memory
- **Compute precision** (`ActivationPrecision`): What precision is used for intermediate activations

The change is **backward compatible** (deprecated API still works) but **strongly encourages migration** to the clearer, more flexible API. This sets a solid foundation for future mixed-precision optimizations.
