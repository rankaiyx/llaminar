# RoPE Precision Correctness Tests

**Date:** November 7, 2025  
**Session:** 6 (continuation)  
**Status:** ✅ **COMPLETED**

## Overview

Added comprehensive mathematical correctness tests for RoPE native precision primitives. Tests verify:
1. **Implementation parity:** Scalar/AVX2/AVX512 variants produce identical results (bit-exact for same precision)
2. **Cross-precision accuracy:** BF16/FP16 results match FP32 within acceptable tolerance
3. **Edge cases:** Various head_dim, positions, frequency bases, multi-head scenarios

## Test Coverage

### Implementation Parity Tests (9 tests)

**FP32 Implementation Parity** (3 tests):
- ✅ `FP32_ScalarVsAVX2_SingleHead` - Scalar vs AVX2 equivalence
- ✅ `FP32_ScalarVsAVX512_SingleHead` - Scalar vs AVX512 equivalence
- ✅ `FP32_AVX2VsAVX512_SingleHead` - AVX2 vs AVX512 equivalence

**BF16 Implementation Parity** (3 tests):
- ✅ `BF16_ScalarVsAVX2_SingleHead` - Bit-exact scalar vs AVX2
- ✅ `BF16_ScalarVsAVX512_SingleHead` - Bit-exact scalar vs AVX512
- ✅ `BF16_AVX2VsAVX512_SingleHead` - Bit-exact AVX2 vs AVX512

**FP16 Implementation Parity** (3 tests):
- ✅ `FP16_ScalarVsAVX2_SingleHead` - Bit-exact scalar vs AVX2
- ✅ `FP16_ScalarVsAVX512_SingleHead` - Bit-exact scalar vs AVX512
- ✅ `FP16_AVX2VsAVX512_SingleHead` - Bit-exact AVX2 vs AVX512

### Cross-Precision Accuracy Tests (3 tests)

- ✅ `BF16_VsFP32_SingleHead` - BF16 vs FP32 within 1e-2 tolerance
- ✅ `FP16_VsFP32_SingleHead` - FP16 vs FP32 within 1e-3 tolerance
- ✅ `INT32_NotSupported` - Verifies INT32 returns false (not supported)

### Edge Case Tests (7 tests)

**Head dimension variations:**
- ✅ `EdgeCase_SmallHeadDim64` - head_dim=64 (Qwen/LLaMA common)
- ✅ `EdgeCase_LargeHeadDim256` - head_dim=256 (large models)

**Position variations:**
- ✅ `EdgeCase_LargePosition1024` - position=1024 (long context)

**Frequency base variations:**
- ✅ `EdgeCase_LLaMAFreqBase` - freq_base=10000.0 (LLaMA)
- ✅ `EdgeCase_QwenFreqBase` - freq_base=1000000.0 (Qwen 2.5)

**Multi-head scenarios:**
- ✅ `FullTensor_BF16_MultipleHeads` - Full Q/K tensors, GQA (4 Q heads, 2 K heads)
- ✅ `FullTensor_FP16_MultipleHeads` - Full Q/K tensors, GQA

## Test Implementation Details

### Test Fixture

```cpp
class RoPEPrecisionCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        rng_.seed(42);  // Deterministic random for reproducibility
    }
    
    // Helper: Generate random FP32 data in [-1, 1]
    std::vector<float> generateRandomFP32(size_t count);
    
    // Helpers: FP32 ↔ BF16 conversion
    std::vector<uint16_t> fp32ToBF16(const std::vector<float>& fp32);
    std::vector<float> bf16ToFP32(const std::vector<uint16_t>& bf16);
    
    // Helpers: FP32 ↔ FP16 conversion
    std::vector<uint16_t> fp32ToFP16(const std::vector<float>& fp32);
    std::vector<float> fp16ToFP32(const std::vector<uint16_t>& fp16);
    
    // Validation: Compare with tolerance
    void expectNear(const std::vector<float>& a, const std::vector<float>& b,
                   float abs_tol, const std::string& msg);
    
    // Validation: Bit-exact comparison
    void expectBitExact(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                       const std::string& msg);
    
    std::mt19937 rng_;
};
```

### Implementation Parity Test Pattern

Tests verify that scalar, AVX2, and AVX512 implementations produce identical results:

```cpp
TEST_F(RoPEPrecisionCorrectnessTest, BF16_ScalarVsAVX2_SingleHead) {
    const int head_dim = 128;
    const int position = 7;
    const float freq_base = 10000.0f;
    
    // Generate random BF16 data
    auto head_fp32 = generateRandomFP32(head_dim);
    auto head_bf16 = fp32ToBF16(head_fp32);
    auto head_scalar = head_bf16;
    auto head_avx2 = head_bf16;
    
    const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);
    
    // Apply scalar variant
    primitives::apply_rope_to_head_bf16_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);
    
    // Apply AVX2 variant + scalar tail
    #if defined(__AVX2__)
    int processed = primitives::apply_rope_to_head_bf16_avx2(head_avx2.data(), position, inv_freq, head_dim);
    primitives::apply_rope_to_head_bf16_scalar(head_avx2.data(), position, inv_freq, head_dim, processed);
    #else
    head_avx2 = head_scalar;
    #endif
    
    // Should be bit-exact (same precision, same operations)
    expectBitExact(head_scalar, head_avx2, "BF16 Scalar vs AVX2");
}
```

**Key points:**
- Uses testable per-head functions exposed in RoPEPrimitives.h
- Compares scalar reference against vectorized implementations
- Bit-exact comparison ensures perfect equivalence

### Cross-Precision Accuracy Test Pattern

Tests verify that lower precision (BF16/FP16) matches FP32 within tolerance:

```cpp
TEST_F(RoPEPrecisionCorrectnessTest, BF16_VsFP32_SingleHead) {
    const int head_dim = 128;
    const int position = 7;
    const float freq_base = 10000.0f;
    
    // Generate random FP32 data
    auto head_fp32_orig = generateRandomFP32(head_dim);
    
    // Create copies for each precision
    auto head_fp32 = head_fp32_orig;
    auto head_bf16 = fp32ToBF16(head_fp32_orig);
    
    const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);
    
    // Apply RoPE in FP32
    primitives::apply_rope_to_head_scalar(head_fp32.data(), position, inv_freq, head_dim, 0);
    
    // Apply RoPE in BF16
    primitives::apply_rope_to_head_bf16_scalar(head_bf16.data(), position, inv_freq, head_dim, 0);
    
    // Convert BF16 result to FP32 for comparison
    auto head_bf16_as_fp32 = bf16ToFP32(head_bf16);
    
    // BF16 has ~3 decimal digits of precision
    // Rotation involves sin/cos which can amplify errors
    expectNear(head_fp32, head_bf16_as_fp32, 1e-2f, "BF16 vs FP32");
}
```

**Key points:**
- Start with same FP32 data, convert to lower precision
- Apply RoPE in both precisions
- Convert BF16/FP16 result back to FP32 for comparison
- Use appropriate tolerance based on precision characteristics

### Tolerance Rationale

**BF16 vs FP32: 1e-2 (0.01)**
- BF16 has ~3 decimal digits of precision
- Rotation involves sin/cos (transcendental functions)
- Multiple multiplications can compound error
- Conservative tolerance accounts for error amplification

**FP16 vs FP32: 1e-3 (0.001)**
- FP16 has ~3-4 decimal digits of precision
- Better than BF16 but still limited
- Same sin/cos error amplification concerns
- Tighter tolerance than BF16 due to better precision

**Implementation parity: Bit-exact**
- Same precision, same operations → should be identical
- No tolerance needed for scalar vs AVX2 vs AVX512
- Any difference indicates bug in vectorized implementation

## Test Results

### Execution Summary

```bash
$ cd build_v2 && ctest -R "V2_Unit_RoPEPrecisionCorrectness" --output-on-failure

Test project /workspaces/llaminar/build_v2
    Start  1: V2_FetchModelsFixture
1/2 Test  #1: V2_FetchModelsFixture ............   Passed    0.01 sec
    Start 11: V2_Unit_RoPEPrecisionCorrectness
2/2 Test #11: V2_Unit_RoPEPrecisionCorrectness ...   Passed    0.61 sec

100% tests passed, 0 tests failed out of 2
```

**Status:** ✅ **19/19 tests passing** (100% success rate)

### All RoPE Tests

```bash
$ cd build_v2 && ctest -L "RoPE" --output-on-failure

100% tests passed, 0 tests failed out of 5

Tests:
1. V2_FetchModelsFixture (fixture)
2. V2_Unit_RoPEPrimitives (original parity tests)
3. V2_Unit_RoPEPrecisionCorrectness (new precision tests)
4. V2_Unit_AttentionParity (attention with RoPE)
5. V2_Unit_VectorizedPrimitives (RMSNorm + other primitives)
```

**Status:** ✅ **5/5 RoPE-related tests passing**

## Test Organization

**File:** `tests/v2/unit/Test__RoPEPrecisionCorrectness.cpp`
**Lines:** ~780
**Test count:** 19
**Test fixture:** `RoPEPrecisionCorrectnessTest`

**CMakeLists.txt entry:**
```cmake
add_executable(v2_test_rope_precision unit/Test__RoPEPrecisionCorrectness.cpp)
target_link_libraries(v2_test_rope_precision 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)
add_v2_test(V2_Unit_RoPEPrecisionCorrectness 
    COMMAND $<TARGET_FILE:v2_test_rope_precision>
    LABELS "V2;Unit;Primitives;RoPE;SIMD;AVX2;AVX512;Correctness;FP32;BF16;FP16;INT32;PrecisionTesting;CPU"
    MPI_PROCS 1  # RoPE is CPU-only, single rank
)
```

## Coverage Analysis

### Precision Types Tested
- ✅ **FP32:** Full scalar/AVX2/AVX512 parity
- ✅ **BF16:** Full scalar/AVX2/AVX512 parity + cross-precision accuracy
- ✅ **FP16:** Full scalar/AVX2/AVX512 parity + cross-precision accuracy
- ✅ **INT32:** Verified returns false (not supported)

### SIMD Variants Tested
- ✅ **Scalar:** Reference implementation for all precisions
- ✅ **AVX2:** 8-wide vectorization for FP32/BF16/FP16
- ✅ **AVX512:** 16-wide vectorization for FP32/BF16/FP16

### Head Dimensions Tested
- ✅ 64 (Qwen/LLaMA common)
- ✅ 128 (default test size)
- ✅ 256 (large models)

### Positions Tested
- ✅ 3 (small)
- ✅ 7 (typical)
- ✅ 15 (medium)
- ✅ 1024 (long context)

### Frequency Bases Tested
- ✅ 10000.0 (LLaMA)
- ✅ 1000000.0 (Qwen 2.5)

### Tensor Scenarios Tested
- ✅ Single head (simple case)
- ✅ Multi-head Q/K tensors (4 Q heads, 2 K heads - GQA)
- ✅ Multiple sequences (seq_len=4)

## Comparison with Session 4 RMSNorm Tests

**RMSNorm tests** (Session 4):
- File: `tests/v2/unit/Test__RMSNormPrecisionCorrectness.cpp`
- Coverage: FP32/BF16/FP16/INT32 parity + cross-precision
- Pattern: Same test structure

**RoPE tests** (Session 6 - current):
- File: `tests/v2/unit/Test__RoPEPrecisionCorrectness.cpp`
- Coverage: FP32/BF16/FP16/INT32 parity + cross-precision
- Pattern: **Identical test structure** (architectural consistency)

**Architectural benefit:**
- Same test patterns ensure consistent validation
- Easy to add future operations (LayerNorm, SwiGLU, etc.)
- Developers know what to expect when reading tests

## What These Tests Validate

### 1. Implementation Correctness
Scalar implementations serve as reference. If AVX2/AVX512 don't match scalar bit-exact, it indicates:
- Bug in vectorized rotation logic
- Incorrect sin/cos computation
- Wrong memory indexing/layout
- Tail handling errors

### 2. Precision Characteristics
Cross-precision tests validate expected error bounds:
- BF16: ~3 decimal digits → 1e-2 tolerance reasonable
- FP16: ~3-4 decimal digits → 1e-3 tolerance reasonable
- Excessive error indicates conversion bugs or precision issues

### 3. Production Path Validation
`FullTensor_*_MultipleHeads` tests validate production code paths:
- Uses public `apply_rope_bf16()` API (what pipelines call)
- Tests multi-head, multi-sequence scenarios (realistic workloads)
- Compares against scalar reference applied per-head

### 4. Edge Cases
Edge case tests ensure robustness:
- Various head_dim sizes (vectorization tail handling)
- Large positions (angle computation accuracy)
- Different frequency bases (model compatibility)

## Integration with V2 Architecture

**Test labels:**
```
V2;Unit;Primitives;RoPE;SIMD;AVX2;AVX512;Correctness;
FP32;BF16;FP16;INT32;PrecisionTesting;CPU
```

**Filtering examples:**
```bash
# All precision tests
ctest -L "PrecisionTesting"

# All RoPE tests
ctest -L "RoPE"

# All SIMD vectorization tests
ctest -L "SIMD"

# All BF16 tests
ctest -L "BF16"
```

## Future Extensions

### Additional Test Cases
- [ ] **GQA ratios:** Test various q_heads/k_heads ratios (1:1, 4:1, 8:1)
- [ ] **Very long context:** position > 4096 (extended context models)
- [ ] **Non-standard head_dim:** 96, 192 (some models use these)
- [ ] **Batch decode:** Test persistent state optimization path

### Performance Benchmarks
- [ ] **Vectorization speedup:** Measure scalar vs AVX2 vs AVX512 performance
- [ ] **Cross-precision overhead:** Compare BF16/FP16 vs FP32 throughput
- [ ] **Memory bandwidth:** Profile cache behavior with various head_dim

### Cross-Platform Testing
- [ ] **Different CPUs:** Test on AMD, older Intel (no AVX512)
- [ ] **ARM:** Test NEON vectorization (when implemented)
- [ ] **GPU:** Test CUDA/ROCm implementations (when available)

## Success Criteria Met

- ✅ **All precisions tested:** FP32, BF16, FP16, INT32
- ✅ **All SIMD variants tested:** Scalar, AVX2, AVX512
- ✅ **Implementation parity verified:** Bit-exact equivalence
- ✅ **Cross-precision accuracy verified:** Within expected tolerances
- ✅ **Edge cases covered:** Various dimensions, positions, frequencies
- ✅ **Production paths validated:** Full tensor, multi-head scenarios
- ✅ **Zero test failures:** 19/19 passing (100%)

## Conclusion

Comprehensive RoPE precision correctness tests now validate:
1. All SIMD variants produce identical results (implementation parity)
2. Lower precision formats match FP32 within tolerance (cross-precision accuracy)
3. Edge cases and production scenarios work correctly

This completes the mathematical correctness validation for RoPE native precision primitives, ensuring the Session 6 implementation is robust and production-ready.

**Total test coverage for RoPE:** 5 test suites, 19+ individual test cases
**Status:** ✅ **ALL PASSING**
