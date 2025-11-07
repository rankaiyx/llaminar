# Removed Deprecated `--precision` CLI Flag

**Date**: November 7, 2025  
**Status**: ✅ Complete  
**Related**: `changelog/2025-11-07-precision-disambiguation-weight-vs-activation.md`, `changelog/2025-11-07-cli-separate-weight-activation-precision.md`

---

## Summary

Completely removed the deprecated `--precision` CLI flag and all associated parsing code. Users must now use the separate `--weight-precision` and `--activation-precision` flags introduced earlier today.

---

## Motivation

After introducing separate `--weight-precision` and `--activation-precision` flags to eliminate the ambiguity of the old `--precision` flag, we kept the deprecated flag for backward compatibility. However, this creates confusion and technical debt. Since this is V2 (development/experimental), we can make breaking changes without affecting V1 production users.

**Benefits of removal**:
- ✅ Eliminates confusion about which flag to use
- ✅ Forces users to think clearly about weight vs activation precision
- ✅ Reduces code complexity and maintenance burden
- ✅ Prevents accidental use of deprecated API

---

## Breaking Changes

### ❌ Removed CLI Flag

The `--precision` flag has been **completely removed**:

```bash
# ❌ NO LONGER WORKS:
./llaminar2 -m model.gguf --precision fp32
./llaminar2 -m model.gguf --precision int8

# ✅ USE INSTEAD:
./llaminar2 -m model.gguf --weight-prec fp32 --act-prec fp32
./llaminar2 -m model.gguf --weight-prec int8 --act-prec int8
```

### Migration Guide

| Old Command | New Command |
|------------|-------------|
| `--precision mixed` | `--weight-prec native --act-prec fp32` |
| `--precision fp32` | `--weight-prec fp32 --act-prec fp32` |
| `--precision bf16` | `--weight-prec bf16 --act-prec bf16` |
| `--precision fp16` | `--weight-prec fp16 --act-prec fp16` |
| `--precision int8` | `--weight-prec int8 --act-prec int8` |
| `--precision auto` | *(not yet implemented for new flags)* |

---

## Implementation Details

### Files Modified

**1. src/v2/utils/ArgParser.h**:
```cpp
// REMOVED:
std::string precision = "mixed"; // DEPRECATED: use weight_precision and activation_precision

// KEPT:
std::string weight_precision = "native";
std::string activation_precision = "fp32";
```

**2. src/v2/utils/ArgParser.cpp**:
- Removed `--precision` flag parsing (7 lines)
- Removed help text for `--precision` (9 lines)
- Simplified Performance section in help output

**3. src/v2/Main.cpp**:
- Removed old precision parsing logic (40 lines)
- Added new `legacy_precision` derivation from `weight_precision` (30 lines)
- Updated precision logging to show weight and activation separately
- ModelContext::create calls now use derived `legacy_precision`

**4. Comment Updates**:
- `src/v2/tensors/Tensors.h`: Updated INT8Tensor comment (`--precision int8` → `--weight-precision int8`)
- `src/v2/loaders/ModelLoader.h`: Updated header comment (`--precision int8` → `--weight-precision int8`)

### Code Size Impact

**Removed**:
- ~56 lines of deprecated parsing and help text
- 1 field in ArgContext struct

**Added**:
- ~35 lines for legacy_precision derivation (needed for ModelContext API)
- More verbose precision logging (2 switches instead of 1)

**Net**: ~20 lines removed, cleaner separation of concerns

### Legacy API Compatibility

**Important**: The internal `ComputePrecision` enum and `ModelContext::create(... ComputePrecision)` signature are **NOT** removed. They remain for:
1. Internal API compatibility (tests, other code using ModelContext)
2. Conversion functions (`computePrecisionToWeightPrecision`, `computePrecisionToActivationPrecision`)
3. Future deprecation path (will be removed in later refactor)

The CLI simply derives the legacy value from the new settings:

```cpp
// Derive legacy ComputePrecision for ModelContext API
ComputePrecision legacy_precision;
if (pipeline_config.weight_precision == WeightPrecision::NATIVE) {
    legacy_precision = ComputePrecision::MIXED;
} else if (pipeline_config.weight_precision == WeightPrecision::CONVERT_TO_FP32) {
    legacy_precision = ComputePrecision::FP32;
}
// ... etc
```

---

## Help Output Changes

### Before (with deprecated flag):

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
                              [...]
  --activation-precision M  Precision for activations (default: fp32)
  --activation-prec M         fp32 - 32-bit float (highest accuracy)
                              [...]
```

### After (clean):

```
Performance:
  --threads N               Thread count (-1 = auto)
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

## Runtime Logging Changes

### Before:

```
[INFO] Precision mode: FP32 (all weights dequantized to FP32)
```

### After:

```
[INFO] Weight precision: FP32 (dequantize to FP32 at load)
[INFO] Activation precision: FP32 (32-bit float)
```

**Benefit**: Users see exactly what each precision setting controls.

---

## Testing

### Build Verification

```bash
# Clean build
cmake --build build_v2 --target llaminar2_core --parallel
cmake --build build_v2 --target llaminar2 --parallel

# ✅ Result: All targets build successfully
```

### Help Output Verification

```bash
./build_v2/llaminar2 --help | grep -A 20 "Performance:"

# ✅ Result: No mention of --precision, only --weight-precision and --activation-precision
```

### Runtime Test

```bash
# Test new flags work correctly
./build_v2/llaminar2 --list-devices --weight-prec fp32 --act-prec bf16

# ✅ Result: Logs show:
#   [INFO] Weight precision: FP32 (dequantize to FP32 at load)
#   [INFO] Activation precision: BF16 (bfloat16)
```

---

## Future Work

1. **Remove ComputePrecision enum entirely** (larger refactor):
   - Update ModelContext::create to accept WeightPrecision
   - Update all internal APIs
   - Remove conversion functions

2. **Implement `--precision auto` equivalent**:
   - `--weight-prec auto` - auto-select weight format
   - `--act-prec auto` - auto-select activation format based on hardware

3. **Add validation warnings**:
   - Warn if `fp32` weights with `int8` activations (memory waste)
   - Warn if mismatched precisions that hurt performance

---

## Conclusion

The deprecated `--precision` flag has been completely removed from the CLI. This forces a clean break from the ambiguous old API and ensures users explicitly specify weight and activation precision separately. Internal APIs still use ComputePrecision for now, but the CLI derives it from the new settings.

**Impact**: Breaking change for V2 users (acceptable since V2 is experimental)  
**Benefit**: Cleaner API, less confusion, better separation of concerns  
**Next**: Remove ComputePrecision enum from internal APIs in future refactor
