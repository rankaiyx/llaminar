# Q8_1 vs FP32 Parity Test Discovery

**Date**: 2025-12-08
**Author**: David Sanftenberg

## Summary

Created comprehensive E2E parity test suite for Q8_1 vs FP32 activation precision paths. The tests discovered that **Q8_1 activation precision mode is not yet fully implemented** in the Qwen2 pipeline.

## Test Suite

Created `/workspaces/llaminar/tests/v2/integration/Test__Q8_1_vs_FP32_Parity.cpp` with three tests:

1. **PrefillParity** - Compares logits after prefill phase
2. **IncrementalDecodeParity** - Compares token predictions during autoregressive decode  
3. **GreedySamplingEquivalence** - Verifies greedy sampling produces same token sequences

### Comparison Metrics
- Top-5 token overlap (threshold: 60%)
- KL divergence (threshold: 0.5 for prefill, 1.0 for decode)
- Top-1 token match rate

## Discovery: Q8_1 Activation Path Incomplete

All three tests SKIP with the message:
```
Q8_1 activation precision not yet implemented: Q8_1Tensor::mutable_data: quantized tensors are immutable
```

### Root Cause

The issue is in `Q8_1Tensor::mutable_data()`:
```cpp
// src/v2/tensors/Q8_1Tensor.cpp
float* Q8_1Tensor::mutable_data() {
    throw std::runtime_error("Q8_1Tensor::mutable_data: quantized tensors are immutable");
}
```

When `PipelineConfig::activation_precision = ActivationPrecision::Q8_1` is set:
1. Pipeline creates Q8_1Tensor buffers for activations
2. Pipeline tries to write to these buffers via `mutable_data()`
3. Exception is thrown because Q8_1Tensor is designed for **read-only weights**, not mutable activations

### Architectural Gap

The Q8_1 quantization is currently only designed for:
- **Weights**: Immutable, loaded from GGUF, used in GEMM kernels
- **NOT Activations**: The pipeline needs to write intermediate results

To support Q8_1 activations, we would need:
1. A mutable Q8_1 activation buffer type (different from weight Q8_1Tensor)
2. Quantization kernels to convert FP32 → Q8_1 dynamically
3. Modified pipeline to handle quantization/dequantization at layer boundaries

## What Works vs What Doesn't

| Feature | Status | Notes |
|---------|--------|-------|
| Q8_1 weight tensors | ✅ WORKS | Immutable, for GEMM |
| Q8_1 attention kernel | ✅ WORKS | JIT fused kernel for Q8_1 inputs |
| Q8_1 activation buffers | ❌ NOT IMPLEMENTED | mutable_data() throws |
| `ActivationPrecision::Q8_1` | ❌ NOT IMPLEMENTED | Pipeline path incomplete |

## Test Integration

Added to CTest with labels:
```cmake
add_v2_integration_test(V2_Integration_Q8_1_vs_FP32_Parity
    COMMAND v2_integration_q8_1_vs_fp32_parity
    LABELS "V2;Integration;TensorOperations;Quantization;Q8_1;Parity"
)
```

## Running the Tests

```bash
# Build
cmake --build build_v2 --target v2_integration_q8_1_vs_fp32_parity -j$(nproc)

# Run (will SKIP until Q8_1 activation support is implemented)
mpirun -np 2 ./build_v2/tests/v2/v2_integration_q8_1_vs_fp32_parity
```

## Future Work

To enable Q8_1 activation precision:

1. **Create mutable Q8_1 activation type**: Either a separate class or add write support to existing Q8_1Tensor
2. **Add FP32→Q8_1 quantization kernel**: For on-the-fly activation quantization
3. **Modify buffer allocation**: Use hybrid approach (FP32 for some ops, Q8_1 for GEMM inputs)
4. **Validate accuracy**: Run parity tests to ensure acceptable quality loss

The test suite is ready to validate correctness once the feature is implemented.

## Files Changed

- `tests/v2/integration/Test__Q8_1_vs_FP32_Parity.cpp` - NEW (650+ lines)
- `tests/v2/CMakeLists.txt` - Added test target
- `changelog/2025-12-08-q8_1-parity-test-discovery.md` - This file

## Related

- `changelog/2025-12-08-head-dim-analysis-jit-kernel.md` - Head dimension analysis
- `src/v2/kernels/cpu/QuantisedAttentionJit_Q8_1_Fused.h` - Q8_1 JIT kernel
- `src/v2/tensors/Q8_1Tensor.cpp` - Q8_1 tensor implementation (immutable)
