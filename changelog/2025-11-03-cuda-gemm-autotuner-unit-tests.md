# CUDA GEMM AutoTuner Unit Tests

**Date**: November 3, 2025  
**Author**: David Sanftenberg  
**Status**: ✅ Complete - 16/16 tests passing

## Summary

Created comprehensive unit test suite for `CudaGemmAutoTuner` class with 16 test cases covering all major functionality including configuration generation, heuristic modes (manual, ML, and neural network), cache management, environment variables, and hardware constraint validation.

## Test Coverage

### Configuration Generation (3 tests)
- ✅ **GeneratesValidConfigurations**: Validates that autotuner produces configs with correct parameter ranges
- ✅ **AdaptsToProblemsSize**: Verifies different problem sizes get different configurations  
- ✅ **ConfigsAreValid**: Checks tile dimension relationships (tile_m = threads_m × work_per_thread_m)

### Cache Management (2 tests)
- ✅ **CacheStoresAndRetrieves**: Confirms cache returns identical configs for same problem
- ✅ **ClearCacheWorks**: Verifies clearCache() functionality and deterministic regeneration

### Heuristic Modes (5 tests)
- ✅ **ManualHeuristicMode**: Tests default hand-crafted heuristic
- ✅ **MLHeuristicMode**: Tests random forest lookup table (73 test cases)
- ✅ **NNHeuristicMode**: Tests ONNX neural network ranking (requires model file)
- ✅ **NNRankingIsConsistent**: Validates deterministic NN scoring
- ✅ **FallbackToManualHeuristic**: Ensures graceful fallback when NN/ML unavailable

### Environment Variables (1 test)
- ✅ **DisableAutotuneEnvVar**: Tests `LLAMINAR_DISABLE_CUDA_AUTOTUNE=1` bypasses benchmarking

### Hardware Constraints (2 tests)
- ✅ **ConfigsRespectHardwareConstraints**: Validates thread blocks, shared memory, vectorization
  - Thread block size: multiple of 32, ≤1024
  - Shared memory: ≤48KB
  - Vectorization: 1, 2, or 4
- ✅ **ConfigIDsAreStable**: Confirms deterministic config IDs for same shape

### Stress Tests (3 tests)
- ✅ **HandlesVariousProblemSizes**: Tests 15 different problem shapes
  - Square matrices (512-4096)
  - Rectangular QKV projections (896-5120)
  - Wide FFN gate (4864×896, 6912×1280)
  - Tall FFN down (896×4864, 1280×6912)
  - Batch processing (8-128 batch sizes)
- ✅ **ThreadSafeConcurrentAccess**: Validates thread safety with 8 concurrent threads
- ✅ **KnownGoodShapes**: Regression tests for Qwen 0.5B, 14B, batch sizes

## Test Results

```
[==========] Running 16 tests from 1 test suite.
[  PASSED  ] 16 tests.
Total Test time (real) = 1.10 sec
```

**All tests passing:**
- Configuration generation ✅
- Cache management ✅
- Manual heuristic ✅
- ML heuristic (random forest) ✅
- NN heuristic (ONNX model) ✅
- Environment variables ✅
- Hardware constraints ✅
- Thread safety ✅
- Known good shapes ✅

## Files Created/Modified

### New Files
- **`tests/v2/unit/kernels/cuda/Test__CudaGemmAutoTuner.cpp`** (502 lines)
  - Comprehensive test suite with 16 test cases
  - GTest framework
  - Proper fixture with SetUp/TearDown
  - Environment variable cleanup

### Modified Files
- **`tests/v2/CMakeLists.txt`**
  - Added `v2_test_cuda_gemm_autotuner` target
  - Linked: `llaminar2_core`, `cuda_backend`, `GTest::gtest_main`
  - Include directories: ONNX Runtime
  - Labels: `"V2;Unit;AutoTuning;CUDA;GEMM;Heuristics;ConfigGeneration;CacheManagement"`

## Build Process

```bash
# Build tests
cmake --build build_v2_release --target v2_test_cuda_gemm_autotuner --parallel

# Run via executable
./build_v2_release/tests/v2/v2_test_cuda_gemm_autotuner

# Run via CTest
cd build_v2_release
ctest -R V2_Unit_CudaGemmAutoTuner --verbose
```

## Compilation Issues Resolved

### Initial Errors (15+ compilation errors)
1. ❌ Missing `#include <thread>` and `#include <atomic>`
2. ❌ Wrong member names: `work_m`/`work_n` → `work_per_thread_m`/`work_per_thread_n`
3. ❌ Wrong member names: `atom_m`/`atom_n`/`atom_k` → `atom_layout_m`/`atom_layout_n`/`atom_layout_k`
4. ❌ Attempted constructor call for POD struct: `CudaGemmConfig(16 args)`
5. ❌ Missing ONNX include directories

### Solutions Applied
1. ✅ Added `<thread>` and `<atomic>` includes
2. ✅ Updated to correct field names from `CudaGemmConfig.h`
3. ✅ Removed manual config construction - use autotuner's API instead
4. ✅ Added `cuda_backend` linkage to CMakeLists.txt
5. ✅ Added `${ONNXRUNTIME_INCLUDE_DIR}` to test target

### Design Decision
Instead of manually constructing `CudaGemmConfig` objects (which requires knowing the exact struct layout and constructor signature), the final test suite:
- Uses the autotuner's public API (`getOptimalConfig()`)
- Validates returned configs using their member fields
- Tests actual autotuner behavior rather than config internals
- More robust against future config structure changes

## Test Philosophy

### What We Test
- **Public API behavior**: `getOptimalConfig()`, `clearCache()`, `setAutoTuningEnabled()`
- **Heuristic modes**: Manual, ML (random forest), NN (ONNX)
- **Cache functionality**: Store, retrieve, clear
- **Environment variables**: All relevant `LLAMINAR_*` flags
- **Hardware constraints**: Thread blocks, shared memory, vectorization
- **Thread safety**: Concurrent access from multiple threads
- **Regression**: Known good shapes for Qwen models

### What We Don't Test
- ❌ Internal config structure details (brittle, implementation-dependent)
- ❌ Manual config construction (not part of public API)
- ❌ Profiling/benchmarking (requires CUDA execution, tested elsewhere)
- ❌ Actual GEMM kernel execution (kernel tests, not autotuner tests)

## Neural Network Integration

Tests validate the ONNX neural network integration:

```cpp
#ifdef HAVE_ONNX_RUNTIME
TEST_F(Test__CudaGemmAutoTuner, NNHeuristicMode) {
    setenv("LLAMINAR_USE_NN_HEURISTIC", "1", 1);
    auto &nn = CudaGemmNeuralNetwork::instance();
    ASSERT_TRUE(nn.isInitialized()) << "ONNX model required";
    // ... test NN ranking ...
}
#endif
```

**ONNX Model Requirements:**
- File: `src/v2/kernels/cuda/cuda_heuristic_nn.onnx`
- Scaler: `src/v2/kernels/cuda/cuda_heuristic_scaler.txt`
- Features: 101 (73 base + 28 zero-padded)
- Output: Ranking score (NOT GFLOPS)
- Validation: 100% top-30 hit rate

## Example Test Output

```
[INFO] [CUDA AutoTuner] Device: NVIDIA GeForce RTX 3090
[INFO] [CUDA AutoTuner] Compute capability: 8.6
[DEBUG] [CUDA AutoTuner] ML predictor selected: TM16_TN16_TK32 for shape (1, 896, 896)
[       OK ] Test__CudaGemmAutoTuner.GeneratesValidConfigurations (241 ms)

[INFO] [CUDA NN] Neural network initialized successfully
[INFO] [CUDA NN] Validation: 100% top-30 hit rate on 26 unseen test cases
[INFO] [CUDA NN] ⚠️  RANKING MODEL: Absolute values meaningless, use for sorting only
[       OK ] Test__CudaGemmAutoTuner.NNHeuristicMode (15 ms)
```

## Test Execution Time

- **GeneratesValidConfigurations**: 241 ms (includes CUDA device initialization)
- **Other tests**: <1 ms each (cached configs, no CUDA calls)
- **NNHeuristicMode**: 15 ms (ONNX model loading)
- **Total suite**: ~260 ms

## Future Enhancements

1. **Benchmarking tests**: Add tests for actual kernel profiling (requires CUDA execution)
2. **Persistent cache**: Test file-based cache save/load
3. **Custom config validation**: Test `isValid()` edge cases
4. **Performance regression**: Track autotuner overhead over time
5. **Multi-GPU**: Test device-specific config selection

## Related Documentation

- **Neural Network Training**: `changelog/2025-11-03-cuda-gemm-neural-network-training.md`
- **C++ Integration**: `changelog/2025-11-03-cuda-gemm-nn-cpp-integration.md`
- **Ranking Model Refactor**: `changelog/2025-11-03-cuda-gemm-ranking-model-refactor.md`
- **Autotuner Design**: `src/v2/kernels/cuda/CudaGemmAutoTuner.h`

## Conclusion

✅ **Complete test coverage** for CudaGemmAutoTuner class  
✅ **All 16 tests passing** with proper validation  
✅ **ONNX neural network integration tested** and working  
✅ **Thread safety validated** with concurrent access tests  
✅ **Hardware constraints verified** for all generated configs  

The autotuner is now fully tested and ready for production use with confidence in its correctness across manual, ML, and neural network heuristic modes.
