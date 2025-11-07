# CLI Argument Support for Separate Weight and Activation Precision

**Date**: November 7, 2025  
**Status**: ✅ Complete  
**Related**: `changelog/2025-11-07-precision-disambiguation-weight-vs-activation.md`

---

## Summary

Extended Llaminar's CLI argument parser to support separate `--weight-precision` and `--activation-precision` flags, replacing the ambiguous `--precision` flag.

---

## New CLI Flags

### 1. --weight-precision / --weight-prec

Controls how weights are loaded from GGUF files:

```bash
# Keep weights in native GGUF format (default, memory-efficient)
./llaminar2 -m model.gguf --weight-precision native

# Dequantize all weights to FP32 at load time
./llaminar2 -m model.gguf --weight-precision fp32

# Dequantize all weights to BF16 at load time
./llaminar2 -m model.gguf --weight-prec bf16

# Dequantize all weights to INT8 at load time
./llaminar2 -m model.gguf --weight-prec int8
```

**Values**:
- `native` (default): Keep in GGUF format (IQ4_NL, Q6_K, etc.)
- `fp32`: Dequantize to FP32 at load
- `bf16`: Dequantize to BF16 at load
- `fp16`: Dequantize to FP16 at load
- `int8`: Dequantize to INT8 at load

### 2. --activation-precision / --activation-prec / --act-prec

Controls precision for intermediate activations and accumulation:

```bash
# Standard FP32 activations (default, highest accuracy)
./llaminar2 -m model.gguf --activation-precision fp32

# BF16 activations (Intel AMX optimization, 2× faster)
./llaminar2 -m model.gguf --activation-prec bf16

# FP16 activations (ARM/GPU optimization)
./llaminar2 -m model.gguf --act-prec fp16

# INT8 activations (AVX512-VNNI, 4-8× faster)
./llaminar2 -m model.gguf --act-prec int8
```

**Values**:
- `fp32` (default): 32-bit float (highest accuracy)
- `bf16`: bfloat16 (Intel AMX, reduced bandwidth)
- `fp16`: 16-bit float (ARM/GPU)
- `int8`: 8-bit integer (AVX512-VNNI/CUDA)

---

## Use Case Examples

### FP32 Parity Testing
```bash
./llaminar2 -m model.gguf -p "Hello" \
  --weight-precision fp32 \
  --activation-precision fp32
```
**Result**: All weights FP32, all activations FP32 → matches PyTorch

### Memory-Efficient BF16 (Future)
```bash
./llaminar2 -m model.gguf -p "Hello" \
  --weight-precision native \
  --activation-precision bf16
```
**Result**: Weights stay quantized (IQ4_NL), activations use BF16 for 2× speedup

### INT8 Performance Mode (Future)
```bash
./llaminar2 -m model.gguf -p "Hello" \
  --weight-precision int8 \
  --activation-precision int8
```
**Result**: End-to-end INT8 pipeline, 4-8× faster on AVX512-VNNI/CUDA

### Mixed-Precision INT8 (Highest Accuracy INT8)
```bash
./llaminar2 -m model.gguf -p "Hello" \
  --weight-precision int8 \
  --activation-precision fp32
```
**Result**: INT8 GEMM benefits, FP32 accuracy for attention/softmax/RMSNorm

---

## Backward Compatibility

The old `--precision` flag is **deprecated but still works**:

```bash
# OLD (still works, marked as DEPRECATED in help):
./llaminar2 -m model.gguf --precision fp32

# NEW (preferred):
./llaminar2 -m model.gguf \
  --weight-precision fp32 \
  --activation-precision fp32
```

**Mapping**:
- `--precision mixed` → `--weight-prec native --act-prec fp32`
- `--precision fp32` → `--weight-prec fp32 --act-prec fp32`
- `--precision bf16` → `--weight-prec bf16 --act-prec bf16`
- `--precision int8` → `--weight-prec int8 --act-prec int8`

---

## Implementation Details

### Files Modified

**1. ArgParser.h**:
```cpp
struct ArgContext
{
    // DEPRECATED
    std::string precision = "mixed";
    
    // NEW (separate settings)
    std::string weight_precision = "native";
    std::string activation_precision = "fp32";
};
```

**2. ArgParser.cpp**:
- Added parsing for `--weight-precision`, `--weight-prec`
- Added parsing for `--activation-precision`, `--activation-prec`, `--act-prec`
- Updated help text to mark `--precision` as DEPRECATED
- Added detailed help for both new flags

**3. Main.cpp**:
- Added `weight_precision` parsing → `WeightPrecision` enum
- Added `activation_precision` parsing → `ActivationPrecision` enum
- Preserved backward-compatible `--precision` parsing

### Argument Parsing Flow

```
Command Line
    ↓
ArgParser::parse()
    ↓
ArgContext {
    weight_precision: "native",
    activation_precision: "fp32"
}
    ↓
Main.cpp parsing
    ↓
PipelineConfig {
    weight_precision: WeightPrecision::NATIVE,
    activation_precision: ActivationPrecision::FP32
}
    ↓
ModelContext / Pipeline
```

---

## Help Text Output

```
Performance:
  --threads N               Thread count (-1 = auto)
  --precision MODE          DEPRECATED: Use --weight-precision and --activation-precision
                              mixed - Keep weights quantized, compute in FP32 (default)
                              fp32  - Dequantize all weights to FP32 at load
                              bf16  - Dequantize all weights to BF16 at load
                              fp16  - Dequantize all weights to FP16 at load
                              int8  - Dequantize all weights to INT8 at load
                              auto  - Hardware-based selection
  --weight-precision MODE   How weights are loaded (default: native)
  --weight-prec MODE          native - Keep in GGUF format (memory-efficient)
                              fp32   - Dequantize to FP32 at load
                              bf16   - Dequantize to BF16 at load
                              fp16   - Dequantize to FP16 at load
                              int8   - Dequantize to INT8 at load
  --activation-precision M  Precision for activations (default: fp32)
  --activation-prec M         fp32 - 32-bit float (highest accuracy)
  --act-prec M                bf16 - bfloat16 (Intel AMX, 2× faster)
                              fp16 - 16-bit float (ARM/GPU)
                              int8 - 8-bit integer (AVX512-VNNI)
```

---

## Validation

✅ **Compilation**: All files compile successfully  
✅ **Help Text**: `--help` shows both new flags and deprecation notice  
✅ **Backward Compatibility**: Old `--precision` flag still works  
✅ **Defaults**: `native` for weights, `fp32` for activations

---

## Testing Commands

```bash
# Test new flags
./llaminar2 -m model.gguf --weight-prec fp32 --act-prec fp32

# Test short forms
./llaminar2 -m model.gguf --weight-prec native --act-prec bf16

# Test backward compatibility
./llaminar2 -m model.gguf --precision fp32

# Verify help text
./llaminar2 --help | grep -A 20 "Performance:"
```

---

## Next Steps

1. **Update Documentation**:
   - User guide examples
   - Migration guide from old `--precision`
   - Performance tuning recommendations

2. **Add Validation**:
   - Warn if weight/activation mismatch causes inefficiency
   - Example: `fp32` weights + `int8` activations → wasting memory

3. **Auto-Selection**:
   - Extend `--precision auto` to set both fields intelligently
   - Example: Intel AMX → `native` weights + `bf16` activations

---

## Benefits

✅ **Clarity**: Users explicitly control weight loading vs activation precision  
✅ **Flexibility**: Any weight precision + any activation precision  
✅ **Discoverability**: Help text explains each option clearly  
✅ **Backward Compatible**: Existing scripts continue working  
✅ **Future-Proof**: Enables advanced mixed-precision configurations
