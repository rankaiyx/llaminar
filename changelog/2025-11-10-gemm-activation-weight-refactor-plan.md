# GEMM Kernel Refactor: Activation Type & Weight Access Abstraction

**Date**: November 10, 2025  
**Status**: 🚧 In Progress (Phase 1: Interfaces Complete)  
**Goal**: Genericize GemmKernelTemplate over activation precision (FP32/BF16/FP16/INT8) and weight access mode (FP32 decode vs raw quantized)

---

## Motivation

**Current Limitation**: `GemmKernelTemplate` assumes:
- Activations are always `float` (hard-coded panel buffers)
- Weights decode to `float` via `ITensorGemmTileDataProvider`

This prevents:
- ❌ BF16/FP16 activation storage (Phase 5+ memory optimization)
- ❌ Integer GEMM (INT8×IQ4_NL VNNI path from `IntegerGemm.cpp`)
- ❌ Fused dequantization (avoid materializing FP32 weight buffers)

**Goal**: Make `GemmKernelTemplate` generic over:
1. **Activation precision**: FP32, BF16, FP16, INT8
2. **Weight access mode**: FP32 decode, raw quantized blocks

---

## Architecture Overview

### Before (Current)

```
GemmKernelTemplate<ISA, TILE_M, TILE_N, ...>
  ├─ float A_panel[TILE_M * k]           // Hard-coded float
  ├─ float B_panel[TILE_N * k]           // Hard-coded float
  └─ ITensorGemmTileDataProvider decoder // Always decodes to FP32
       └─ decode_block_at() → float*
```

### After (Refactored)

```
GemmKernelTemplate<ISA, TILE_M, TILE_N, ActivationTraits, WeightAccessor, ...>
  ├─ ActivationTraits::storage_type A_panel[TILE_M * k]  // float, uint16_t, or int8_t
  ├─ WeightAccessor accessor                             // FP32 or quantized path
  │   ├─ FP32WeightAccessor: decode_block() → float*
  │   └─ QuantizedWeightAccessor: get_raw_block() → void* + get_scale() → float
  └─ Activation panel packing via ActivationTraits::pack_panel()
```

---

## Implementation Plan

### Phase 1: Interfaces ✅ COMPLETE

**Files Created**:
- `src/v2/kernels/cpu/gemm/ActivationTraits.h` (220 lines)
  - `ActivationStorageTraits<T>` template
  - Specializations: `float`, `uint16_t` (BF16/FP16), `int8_t`
  - Methods: `load()`, `store()`, `pack_panel()`
  
- `src/v2/kernels/cpu/gemm/WeightAccessor.h` (150 lines)
  - `FP32WeightAccessor`: Wraps `ITensorGemmTileDataProvider`
  - `QuantizedWeightAccessor`: Wraps `IQuantizedTileAccessor`

**Files Modified**:
- `src/v2/tensors/TensorKernels.h`
  - Added `IQuantizedTileAccessor` interface (raw quantized block access)
  - Updated `ITensorGemmTileDataProvider` documentation

### Phase 2: Template Refactor (In Progress)

**Target**: `src/v2/kernels/cpu/gemm/GemmKernelTemplate.h`

**Changes**:
1. Add template parameters:
   ```cpp
   template <
       typename ISA,
       int TILE_M, int TILE_N,
       typename ActivationTraits,  // NEW
       typename WeightAccessor,     // NEW
       int UNROLL_FACTOR = 8,
       int PREFETCH_DISTANCE = 5
   >
   ```

2. Replace hard-coded `float` with `ActivationTraits::storage_type`:
   ```cpp
   // Before
   alignas(64) float A_panel[TILE_M * block_size];
   
   // After
   using ActStorage = typename ActivationTraits::storage_type;
   alignas(64) ActStorage A_panel[TILE_M * block_size];
   ```

3. Update panel packing:
   ```cpp
   // Before
   std::memcpy(A_panel + i * block_size, A_row + k_start, k_count * sizeof(float));
   
   // After
   ActivationTraits::pack_panel(A_row + k_start, A_panel + i * block_size, 
                                 1, k_count, 0);
   ```

4. Update weight access:
   ```cpp
   // Before
   decoder->decode_block_at(jj + jt, kb, B_panel + jt * block_size);
   
   // After (FP32 path)
   accessor.decode_block(jj + jt, kb, B_panel + jt * block_size);
   
   // After (quantized path)
   const void* raw_block = accessor.get_raw_block(jj + jt, kb);
   float scale = accessor.get_block_scale(jj + jt, kb);
   // ... decode in registers during accumulation ...
   ```

### Phase 3: Adapter Update

**Target**: `src/v2/kernels/cpu/gemm/GemmMicroKernelAdapter.h`

**Changes**:
1. Add template parameters matching `GemmKernelTemplate`
2. Detect activation tensor type at runtime:
   ```cpp
   auto* act_tensor = dynamic_cast<IActivationTensor*>(A_tensor);
   if (auto* fp32 = dynamic_cast<FP32Tensor*>(act_tensor)) {
       // Use ActivationStorageTraits<float>
   } else if (auto* bf16 = dynamic_cast<BF16Tensor*>(act_tensor)) {
       // Use ActivationStorageTraits<uint16_t>
   }
   ```

3. Update packing buffers to match activation type

### Phase 4: AutoTuner Integration

**Target**: `src/v2/kernels/cpu/gemm/GemmAutoTuner.h`

**Changes**:
1. Extend cache key:
   ```cpp
   // Before: (m, n, k) → variant
   // After:  (m, n, k, activation_precision, weight_format) → variant
   ```

2. Add precision to variant metadata:
   ```cpp
   struct GemmVariantMeta {
       int mr, nr, unroll_k, prefetch_dist;
       ActivationPrecision act_precision;  // NEW
       TensorType weight_format;            // NEW
   };
   ```

3. Update heuristic scoring:
   - AVX512-VNNI variants only eligible for INT8 activations
   - BF16 activations prefer larger tiles (better bandwidth)

### Phase 5: Integer GEMM Integration

**Target**: `src/v2/kernels/cpu/gemm/IntegerGemm.cpp`

**Implementation**:
1. Create `IQ4NLQuantizedAccessor` implementing `IQuantizedTileAccessor`
2. Register INT8×IQ4_NL variants with autotuner:
   ```cpp
   auto variant = std::make_unique<GemmKernelTemplate<
       simd::AVX512Tag,
       4, 2,
       ActivationStorageTraits<int8_t>,
       QuantizedWeightAccessor<IQ4NL>,
       8, 5
   >>();
   GemmAutoTuner::instance().registerVariant(std::move(variant));
   ```

3. Wire into `ITensorGemm::multiply()` dispatch

### Phase 6: Testing & Validation

**New Tests**:
1. `Test__ActivationTraits.cpp` - Unit tests for each precision
2. `Test__WeightAccessor.cpp` - FP32 vs quantized path parity
3. Update `Test__IntegerGemm.cpp` to use autotuner path
4. Performance regression tests (GFLOPS comparison)

**Coverage**:
- FP32×IQ4_NL (existing, ensure no regression)
- BF16×IQ4_NL (new, Phase 5 memory optimization)
- INT8×IQ4_NL (new, autotuned VNNI path)

---

## Benefits

### 1. Memory Optimization (Phase 5)
- BF16 activation storage: 50% memory reduction
- Enables larger batch sizes / longer contexts
- Negligible accuracy loss (<2.5e-5 relative error)

### 2. Integer GEMM Performance
- Autotuned AVX512-VNNI path (currently manual in `IntegerGemm.cpp`)
- Fused dequantization (avoid FP32 weight materialization)
- Expected 1.5-2× speedup vs FP32 decode path

### 3. Code Reduction
- Single `GemmKernelTemplate` for all precisions
- Eliminate duplicate packing/tiling code
- Cleaner abstraction boundaries

### 4. Extensibility
- Future INT4 activations (further memory reduction)
- FP8 support (H100+ hardware)
- Mixed-precision attention (FP16 Q/K, BF16 V)

---

## Risks & Mitigation

### Risk 1: Template Bloat
**Problem**: Combinatorial explosion of template instantiations  
**Mitigation**:
- Only instantiate tested combinations (FP32×FP32, BF16×IQ4_NL, INT8×IQ4_NL)
- Use extern template declarations to control instantiation
- Monitor binary size (target: <10% increase)

### Risk 2: Performance Regression
**Problem**: Abstraction overhead in hot path  
**Mitigation**:
- All traits methods marked `inline` or `always_inline`
- Zero-cost abstraction for FP32 (current path)
- Benchmark before/after (require <5% regression for FP32)

### Risk 3: Complexity
**Problem**: More template parameters, harder to debug  
**Mitigation**:
- Comprehensive unit tests for each trait
- Clear error messages via `static_assert`
- Document usage patterns in README.md

---

## Success Criteria

- ✅ Phase 1 interfaces compile without warnings
- ⏳ FP32×IQ4_NL performance within 5% of baseline
- ⏳ BF16×IQ4_NL achieves <2.5e-5 accuracy vs FP32
- ⏳ INT8×IQ4_NL matches or exceeds `IntegerGemm.cpp` performance
- ⏳ All existing tests pass (24/24 V2 tests)
- ⏳ Binary size increase <10%

---

## Timeline

- **Phase 1** (Nov 10): ✅ Interfaces - COMPLETE
- **Phase 2** (Nov 11): Template refactor
- **Phase 3** (Nov 12): Adapter update
- **Phase 4** (Nov 13): AutoTuner integration
- **Phase 5** (Nov 14): Integer GEMM wiring
- **Phase 6** (Nov 15): Testing & validation

**Total**: 6 days (1 week sprint)

---

## Next Steps

1. Refactor `GemmKernelTemplate.h` to use `ActivationTraits` (FP32 only first)
2. Update `GemmMicroKernelAdapter.h` to match template signature
3. Ensure existing FP32 path compiles and passes tests
4. Add BF16 specialization and test accuracy
5. Wire INT8 path and validate vs `IntegerGemm.cpp` baseline

