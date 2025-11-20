# 2025-11-20 - Added Unit Tests for Typed OneDNN GEMM Interface

## Summary
Added comprehensive unit tests for the new `multiply_activations_typed` interface in `OneDNNGemmKernel`. These tests verify the correctness of the template-based dispatch mechanism for FP32 and BF16 precisions, as well as strided matrix support.

## Changes
- Created `tests/v2/unit/kernels/gemm/Test__OneDNNGemmKernel_Typed.cpp`
- Fixed corruption in `tests/v2/CMakeLists.txt` (removed duplicate blocks and fixed test name collisions)
- Registered new test executable `v2_test_onednn_gemm_kernel_typed`

## Test Coverage
- `FloatFloat_Typed_MatchesReference`: Validates FP32 x FP32 -> FP32 GEMM against a reference implementation.
- `BF16BF16_Typed_MatchesReference`: Validates BF16 x BF16 -> FP32 GEMM (using `uint16_t` representation) against a reference implementation with simulated precision loss.
- `Strided_FloatFloat_MatchesContiguous`: Validates that strided inputs are correctly packed and computed by comparing against a dense reference.
- `Unsupported_Int8Int8_ReturnsFalse`: Verifies that the currently unimplemented INT8 path returns false and logs an error.

## Results
All tests passed successfully.
```
[==========] 4 tests from 1 test suite ran. (46 ms total)
[  PASSED  ] 4 tests.
```
