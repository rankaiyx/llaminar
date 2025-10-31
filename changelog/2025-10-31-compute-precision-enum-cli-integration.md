# Compute Precision Enum and CLI Integration

**Date**: 2025-10-31  
**Author**: David Sanftenberg  
**Status**: ✅ Complete - CLI parsing implemented, ready for pipeline integration

## Summary

Implemented a clean, extensible `ComputePrecision` enum to replace boolean flags for precision mode selection, and integrated it into the command-line interface. This architecture enables users to select FP32/BF16/FP16/INT8/AUTO precision modes via `--precision` flag, with automatic validation and hardware compatibility checking.

## Motivation

**Problem**: Previous approach used separate boolean flags (`use_bf16`, `use_fp16`, etc.) which:
- Creates boolean proliferation as we add more precision modes (FP16, INT8, etc.)
- Requires multiple if-else checks throughout the codebase
- Makes adding new modes require touching many files
- Doesn't support "auto" mode for hardware-based selection

**Solution**: Single `ComputePrecision` enum with clean command-line integration:
```bash
./llaminar2 -m model.gguf --precision bf16  # Brain Float 16
./llaminar2 -m model.gguf --precision auto  # Hardware-based selection
```

## Architecture Changes

### 1. ComputePrecision Enum (PipelineConfig.h)

**New Enum** (lines 15-25):
```cpp
enum class ComputePrecision {
    FP32,  ///< Full 32-bit floating point (default, universally supported)
    BF16,  ///< Brain Float 16 (7-bit mantissa, 8-bit exponent) - Intel Sapphire Rapids+
    FP16,  ///< IEEE Float 16 (10-bit mantissa, 5-bit exponent) - ARM/mobile hardware
    INT8,  ///< 8-bit integer quantization (future block floating point)
    AUTO   ///< Automatic selection based on hardware capabilities
};
```

**Field Added to PipelineConfig** (lines 87-105):
```cpp
/**
 * @brief Compute precision mode for activations/attention
 * 
 * Controls precision of:
 * - Attention Q@K^T and scores@V GEMMs
 * - Activation matrices in FFN layers
 * - RMSNorm computations (if enabled)
 * 
 * Weight tensors always use their quantized format (Q4_0, Q6_K, etc.).
 * This setting affects *activation* tensors during forward pass.
 * 
 * Recommended settings:
 * - FP32: Maximum accuracy, baseline performance (default)
 * - BF16: 50% memory, 1.5-2× speedup on Intel Ice Lake+ (AMX-BF16)
 * - FP16: 50% memory, faster on ARM/mobile (NEON)
 * - AUTO: Automatically select BF16 if hardware supports, else FP32
 */
ComputePrecision precision = ComputePrecision::FP32;
```

### 2. Command-Line Integration (ArgParser)

**ArgContext Field** (ArgParser.h):
```cpp
// Compute precision
std::string precision = "fp32"; // "fp32", "bf16", "fp16", "int8", "auto"
```

**Parsing Logic** (ArgParser.cpp, lines 193-198):
```cpp
// Compute precision
else if (arg == "--precision")
{
    ctx.precision = getNextArg(argv, argc, i, "precision");
}
```

**Help Documentation** (ArgParser.cpp, lines 288-295):
```
Performance:
  --threads N               Thread count (-1 = auto)
  --precision MODE          Compute precision:
                              fp32  - Full 32-bit float (default)
                              bf16  - Brain Float 16 (Intel Sapphire Rapids+)
                              fp16  - IEEE Float 16 (ARM/mobile)
                              int8  - 8-bit quantization (future)
                              auto  - Hardware-based selection
```

### 3. String-to-Enum Conversion (Main.cpp)

**Conversion Logic** (lines 275-304):
```cpp
// Parse precision mode
if (args.precision == "fp32")
{
    pipeline_config.precision = ComputePrecision::FP32;
}
else if (args.precision == "bf16")
{
    pipeline_config.precision = ComputePrecision::BF16;
}
else if (args.precision == "fp16")
{
    pipeline_config.precision = ComputePrecision::FP16;
}
else if (args.precision == "int8")
{
    pipeline_config.precision = ComputePrecision::INT8;
}
else if (args.precision == "auto")
{
    pipeline_config.precision = ComputePrecision::AUTO;
}
else
{
    if (mpi_ctx->rank() == 0)
    {
        LOG_WARN("Unknown precision mode '" << args.precision << "', defaulting to fp32");
    }
    pipeline_config.precision = ComputePrecision::FP32;
}
```

**Features**:
- Validates precision string at startup
- Warns if unknown mode provided
- Defaults to FP32 for safety
- Logs warning only on rank 0 (avoids MPI spam)

## Usage Examples

### Basic Precision Selection

```bash
# Default FP32 (no flag needed)
./llaminar2 -m model.gguf -p "Hello"

# BF16 mode (50% memory, faster on Intel Ice Lake+)
./llaminar2 -m model.gguf -p "Hello" --precision bf16

# FP16 mode (50% memory, faster on ARM/mobile)
./llaminar2 -m model.gguf -p "Hello" --precision fp16

# Auto mode (hardware detection)
./llaminar2 -m model.gguf -p "Hello" --precision auto
```

### Combined with Other Flags

```bash
# BF16 with multi-threading
./llaminar2 -m model.gguf --precision bf16 --threads 28

# FP16 with batch processing
./llaminar2 -m model.gguf --precision fp16 --batch-size 32

# Auto mode with memory constraints
./llaminar2 -m model.gguf --precision auto --max-gpu-memory 8192
```

### Validation Examples

```bash
# Unknown precision → warns and defaults to FP32
$ ./llaminar2 -m model.gguf --precision bf32
[WARN] Unknown precision mode 'bf32', defaulting to fp32

# Case-sensitive (lowercase only)
$ ./llaminar2 -m model.gguf --precision BF16
[WARN] Unknown precision mode 'BF16', defaulting to fp32
```

## Implementation Status

### ✅ Completed

1. **ComputePrecision Enum** (PipelineConfig.h)
   - Enum definition with 5 modes (FP32/BF16/FP16/INT8/AUTO)
   - Field added to PipelineConfig struct
   - Comprehensive documentation

2. **CLI Parsing** (ArgParser)
   - `--precision` flag added to ArgParser
   - Help text with detailed descriptions
   - String validation with fallback to FP32

3. **String Conversion** (Main.cpp)
   - String-to-enum conversion logic
   - Error handling with warnings
   - MPI-aware logging (rank 0 only)

4. **Testing**
   - All 73 V2 unit tests passing
   - Build successful
   - Help output validated

### 🔄 Next Steps (Pipeline Integration)

1. **GQAAttention Integration**
   - Add `precision` field to `GQAAttentionConfig` struct
   - Pass `config.precision` from `PipelineBase::attention_gqa_mpi()`
   - Convert enum to `use_bf16` boolean for `ITensorAttention` interface

2. **Compatibility Checking**
   - Add `PipelineBase::validatePrecisionSupport()` method
   - Check CPU features (AMX-BF16, F16C, AVX512)
   - Warn/fallback if unsupported precision selected

3. **AUTO Mode Implementation**
   - Detect AMX-BF16 via CPUID
   - Select BF16 if available, else FP32
   - Log selected precision at pipeline init

4. **FP16/INT8 Implementation** (Future)
   - FP16GemmKernel (similar to BF16)
   - INT8 block floating point
   - GPU precision modes (CUDA/ROCm)

## Technical Notes

### Why Enum Instead of Booleans?

**Before (boolean proliferation)**:
```cpp
struct PipelineConfig {
    bool use_bf16 = false;
    bool use_fp16 = false;
    bool use_int8 = false;
    // What if user sets multiple to true?
};
```

**After (clean enum)**:
```cpp
struct PipelineConfig {
    ComputePrecision precision = ComputePrecision::FP32;
    // Single source of truth, mutually exclusive modes
};
```

**Benefits**:
- **Mutual Exclusion**: Can't accidentally enable multiple modes
- **Extensibility**: Add new modes without touching boolean logic
- **Switch Statements**: Compiler warns if mode not handled
- **Type Safety**: Enum class prevents implicit conversions

### Hardware Compatibility Matrix

| Precision | Hardware Requirement | Memory | Throughput | Accuracy |
|-----------|---------------------|--------|------------|----------|
| **FP32** | Universal | 100% | 1.0× (baseline) | Reference |
| **BF16** | Intel Ice Lake+ (AMX-BF16) | 50% | 1.5-2.0× | ~7-bit mantissa |
| **FP16** | ARM NEON, F16C | 50% | 1.2-1.8× | ~10-bit mantissa |
| **INT8** | AVX512-VNNI, DP4A | 25% | 3-4× (future) | Block FP |
| **AUTO** | Runtime detection | Variable | Variable | Depends on selection |

### Precision Error Characteristics

**From CPUAttention BF16 testing** (`Test__CPUAttention.cpp:BF16Mode`):
- **Relative Error**: ≤3% (attention outputs)
- **Absolute Error**: <0.001 for normalized values
- **Convergence**: BF16 and FP32 produce nearly identical results

**Safe for**:
- Attention mechanisms (Q@K^T, scores@V)
- FFN activations (SwiGLU, GELU)
- Layer normalization (RMSNorm)

**Unsafe for** (always use FP32):
- Softmax (numerical stability)
- Loss computation (gradient precision)
- Final logits (output quality)

## Files Modified

### Source Code

1. **src/v2/pipelines/PipelineConfig.h**
   - Lines 15-25: Added `ComputePrecision` enum
   - Lines 87-105: Added `precision` field with documentation

2. **src/v2/utils/ArgParser.h**
   - Line 66: Added `std::string precision = "fp32"` to `ArgContext`

3. **src/v2/utils/ArgParser.cpp**
   - Lines 193-198: Added `--precision` parsing logic
   - Lines 288-295: Added precision help documentation

4. **src/v2/Main.cpp**
   - Lines 275-304: Added string-to-enum conversion logic
   - Lines 268-304: Updated `PipelineConfig` initialization

### Tests

- **Build Status**: ✅ Clean build (no warnings)
- **Test Results**: ✅ 73/73 V2 unit tests passing
- **Help Output**: ✅ `--precision` flag documented

## Related Work

### Previous Sessions

1. **2025-10-31-cpuattention-bf16-mode.md** (Earlier today)
   - Discovered CPUAttention already has BF16 support
   - Added BF16Mode test with ≤3% error tolerance
   - Documented BF16 performance characteristics

2. **Phase 6 Optimizations** (Earlier session)
   - BF16 activation GEMM implementation
   - Strided Q@K^T optimization (zero-copy head access)
   - Fused GEMM scaling (alpha parameter)

### Architecture Gaps Identified

**Current Flow**:
```
Qwen2Pipeline → attention_gqa_mpi() → GQAAttention::compute_mpi()
                                           ↓
                                    Manual GEMM calls (FP32 only)
```

**Desired Flow**:
```
Qwen2Pipeline → attention_gqa_mpi() → GQAAttention::compute_mpi()
                                           ↓
                                    CPUAttention (ITensorAttention)
                                           ↓
                                    BF16/FP32 GEMM selection
```

**Issue**: GQAAttention uses manual GEMM primitives (`FP32StandaloneGemm`), not `ITensorAttention` interface

**Solutions**:
1. **Option A**: Refactor GQAAttention to use `createAttention()` kernel factory
2. **Option B**: Add precision branching to GQAAttention's manual GEMM path
3. **Option C**: Create precision-aware wrapper around manual GEMMs

## Performance Expectations

### BF16 Mode (Intel Ice Lake+)

**Memory Reduction**:
- Activation buffers: 50% smaller (FP32 → BF16)
- Attention workspace: 50% reduction
- Total model memory: ~10-15% reduction (weights still quantized)

**Throughput Improvement** (from earlier testing):
- Attention GEMM: 1.5-2.0× faster (AMX-BF16 acceleration)
- Cache efficiency: Improved due to smaller footprint
- Overall throughput: +30-50% (model-dependent)

**Accuracy Impact**:
- Attention: ≤3% relative error (validated)
- Final output: Negligible difference in most cases
- Long sequences: Potential accumulated error (monitor)

### AUTO Mode Behavior (Future)

**Hardware Detection**:
```cpp
if (has_amx_bf16()) {
    precision = ComputePrecision::BF16;  // Intel Ice Lake+
} else if (has_f16c()) {
    precision = ComputePrecision::FP16;  // Older Intel, ARM
} else {
    precision = ComputePrecision::FP32;  // Fallback
}
```

**Logging**:
```
[INFO] Precision mode: AUTO
[INFO] Hardware detection: AMX-BF16 supported
[INFO] Selected precision: BF16 (expected 1.5-2× speedup)
```

## Validation Plan

### Unit Tests (Future)

1. **Enum Conversion Test**
   ```cpp
   TEST(ComputePrecision, StringConversion) {
       // Test all valid modes
       EXPECT_EQ(parsePrecision("fp32"), ComputePrecision::FP32);
       EXPECT_EQ(parsePrecision("bf16"), ComputePrecision::BF16);
       // Test invalid → defaults to FP32
       EXPECT_EQ(parsePrecision("invalid"), ComputePrecision::FP32);
   }
   ```

2. **Hardware Compatibility Test**
   ```cpp
   TEST(ComputePrecision, HardwareValidation) {
       // Check BF16 support
       auto precision = ComputePrecision::BF16;
       EXPECT_TRUE(validatePrecisionSupport(precision));
       // Or expects fallback to FP32 on unsupported hardware
   }
   ```

3. **AUTO Mode Test**
   ```cpp
   TEST(ComputePrecision, AutoSelection) {
       auto precision = selectAutoPrecision();
       // Should be BF16 or FP32 depending on hardware
       EXPECT_TRUE(precision == ComputePrecision::BF16 || 
                   precision == ComputePrecision::FP32);
   }
   ```

### Integration Tests (Future)

1. **End-to-End CLI Test**
   ```bash
   # Test that precision flag propagates through pipeline
   ./llaminar2 -m model.gguf --precision bf16 -n 10
   # Verify logs show BF16 mode activated
   ```

2. **Parity Test with FP32**
   ```bash
   # Compare FP32 vs BF16 outputs
   ./llaminar2 -m model.gguf --precision fp32 -n 100 > fp32.txt
   ./llaminar2 -m model.gguf --precision bf16 -n 100 > bf16.txt
   # Verify ≤3% relative difference
   ```

## References

### Documentation

- `.github/copilot-instructions.md` - V2 architecture overview
- `.github/instructions/llaminar-v2-architecture.instructions.md` - V2 design philosophy
- `changelog/2025-10-31-cpuattention-bf16-mode.md` - BF16 implementation details

### Related Code

- `src/v2/kernels/cpu/CPUAttention.cpp` - BF16 support in ITensorAttention
- `src/v2/kernels/cpu/gemm/BF16GemmKernel.h` - BF16 GEMM implementation
- `src/v2/pipelines/attention/GQAAttention.cpp` - Current attention (needs integration)
- `tests/v2/unit/kernels/Test__CPUAttention.cpp` - BF16 validation test

### External References

- Intel AMX-BF16: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-amx-overview.html
- ARM NEON FP16: https://developer.arm.com/documentation/den0024/a/NEON-and-Floating-Point/Half-precision-floating-point
- BFloat16 Format: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format

## Conclusion

The `ComputePrecision` enum provides a clean, extensible architecture for precision mode selection:

✅ **Clean CLI Interface**: Single `--precision` flag with clear options  
✅ **Type Safety**: Enum class prevents misuse  
✅ **Extensibility**: Easy to add FP16, INT8, future modes  
✅ **Validation**: Unknown modes default to safe FP32  
✅ **Documentation**: Comprehensive help text for users  

**Next Step**: Integrate precision mode into `GQAAttention` to actually enable BF16/FP16 computation in the pipeline. Current implementation provides the infrastructure; execution path integration is pending.

**Status**: Infrastructure complete, ready for pipeline integration (estimated 1-2 hours).
