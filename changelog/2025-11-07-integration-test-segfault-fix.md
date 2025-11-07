# Integration Test Segfault Fix - ODR Violation in ENABLE_PIPELINE_SNAPSHOTS

**Date**: November 7, 2025  
**Author**: David Sanftenberg  
**Commit**: TBD

## Problem

4/8 V2 integration tests were segfaulting in Release builds with cryptic crashes at address `0x667567672e8c` (containing bytes "gufg." - part of "gguf" string):

**Failing Tests**:
- `V2_Integration_MPITensorParallelCorrectness` 
- `V2_Integration_MPIBatchedAttention`
- `V2_Integration_BatchedAttention`
- `V2_Integration_DequantEquivalency` (different failure - missing tensor types)

**Symptoms**:
- Tests PASSED in Debug build (`build_v2`)
- Tests FAILED in Release build (`build_v2_release`) with SIGSEGV (11)
- Crash location: `PipelineBase::~PipelineBase()` during destructor cleanup
- Crash address contained "gufg." string suggesting model path corruption
- Tests PASSED when Release built with `-DENABLE_SNAPSHOTS=ON`

**Debugging Journey**:
1. ❌ Tried ASAN build → tests passed (not memory corruption)
2. ❌ Tried -O0 build → tests still failed (not optimization bug)
3. ❌ Tried fixing MPI context conflicts in ModelLoader/ModelContext
4. ❌ Tried fixing member destruction order
5. ✅ Discovered tests pass with snapshots enabled → ODR violation!

## Root Cause

**ODR (One Definition Rule) Violation** in `ENABLE_PIPELINE_SNAPSHOTS` macro usage.

**Before Fix**:
- `tests/v2/CMakeLists.txt` line 194: **Unconditionally** defined `ENABLE_PIPELINE_SNAPSHOTS`
- `src/v2/CMakeLists.txt`: Conditionally defined (Debug/E2ERelease only, NOT Release)

**Result**:
- Tests compiled WITH `ENABLE_PIPELINE_SNAPSHOTS` → PipelineBase includes `std::map<std::string, std::vector<float>> snapshots_` member
- Core library compiled WITHOUT → PipelineBase EXCLUDES that member
- **Different class layouts** → memory corruption during destruction

**Class Layout Mismatch**:

```cpp
// Core library (Release, NO snapshots):
class PipelineBase {
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    // ... other members ...
    // NO snapshots_ member!
};

// Test code (ALWAYS snapshots):
class PipelineBase {
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    // ... other members ...
    std::map<std::string, std::vector<float>> snapshots_; // EXTRA MEMBER!
};
```

When MockPipeline (compiled in tests) was destroyed, it tried to destruct `snapshots_` member that didn't exist in the library's version → undefined behavior → SIGSEGV.

## Solution

**Fixed `tests/v2/CMakeLists.txt` line 191-202** to match core library's conditional compilation:

```cmake
# Before (WRONG - unconditional):
add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)

# After (CORRECT - conditional):
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
elseif(CMAKE_BUILD_TYPE MATCHES "E2ERelease")
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
elseif(ENABLE_SNAPSHOTS)
    add_compile_definitions(ENABLE_PIPELINE_SNAPSHOTS)
endif()
```

**Additional Fixes** (defensive improvements, not root cause):
- `ModelContext.h`: Added `owned_test_factory_` member for `createForTesting()` to prevent ModelLoader from creating conflicting internal MPI context
- `ModelContext.h`: Reordered members to ensure `owned_test_factory_` destructs after `loader_`
- `MockPipeline.cpp`: Updated to pass `mpi_ctx` through `createForTesting()`

## Testing

**Before Fix** (Release build WITHOUT snapshots):
```bash
ctest --test-dir build_v2_release -R "^V2_Integration_"
# Result: 4/8 passed, 4/8 SEGFAULT
```

**After Fix** (Release build WITHOUT snapshots):
```bash
ctest --test-dir build_v2_release -R "^V2_Integration_"  
# Result: 7/8 passed, 1/8 failed (DequantEquivalency - different issue)
```

**Tests Now Passing**:
- ✅ V2_Integration_MPITensorParallelCorrectness
- ✅ V2_Integration_MPIBatchedAttention
- ✅ V2_Integration_BatchedAttention
- ✅ V2_Integration_MPIVectorizedAttention
- ✅ V2_Integration_MPIStaging
- ✅ V2_Integration_MicroKernelAutoTuner
- ✅ V2_FetchModelsFixture
- ❌ V2_Integration_DequantEquivalency (different issue - missing tensor type implementations)

## Lessons Learned

1. **ODR violations can cause subtle memory corruption** that only manifests in specific build configurations
2. **Conditional compilation of class members is dangerous** - must ensure ALL translation units see the same definition
3. **Test build flags MUST match library build flags** when they affect class layout
4. **ASAN/valgrind won't catch ODR violations** - they're compile-time issues, not runtime memory errors
5. **Build configuration differences between Debug/Release can hide bugs** - always test both!

## Impact

- **Pre-commit hook** can now correctly run integration tests from Release build
- **CI/CD pipeline** will catch regressions in optimized builds
- **Performance testing** can use Release builds without worrying about snapshots overhead
- **E2E parity testing** remains available in E2ERelease builds with snapshots enabled

## Files Changed

1. `tests/v2/CMakeLists.txt` (line 191-202): Made `ENABLE_PIPELINE_SNAPSHOTS` conditional
2. `src/v2/loaders/ModelContext.h`: Added `owned_test_factory_` member and MPI context pass-through
3. `src/v2/testing/MockPipeline.cpp`: Updated to pass MPI context through createForTesting()
4. `src/v2/loaders/ModelLoader.cpp`: Added `initializeTestModel()` method (defensive fix)
5. `src/v2/testing/MockPipeline.h`: Fixed `getAllWeightNames()` to return dummy weight (defensive fix)
6. `.githooks/pre-commit`: Updated to run integration tests from Release build (line 88)

## Next Steps

- ✅ Commit these fixes
- ✅ Update pre-commit hook
- ⏳ Fix DequantEquivalency test (missing tensor type implementations - separate issue)
- ⏳ Document ODR best practices for future development
