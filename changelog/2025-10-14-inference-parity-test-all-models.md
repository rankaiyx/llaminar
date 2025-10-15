# Inference Parity Test - All Models Support

**Date:** October 14, 2025  
**Author:** GitHub Copilot  
**Status:** ✅ Implemented

## Summary

Extended the inference parity test infrastructure to automatically test **all GGUF models** in the `models/` folder, ensuring 1:1 token-level comparison between Llaminar and PyTorch for each model.

## Changes Made

### 1. Test Infrastructure Update

**File:** `tests/test_inference_parity.cpp`

#### Before (Single Test for All Models)
- Single monolithic test that looped through models
- No individual test results per model
- Used `EXPECT` so all models tested even if one failed
- Less granular test reporting

#### After (Parametrized Tests)
- Google Test parametrized test suite `InferenceParityTest`
- Each model gets its own individual test case
- Clear test names derived from model filenames
- Uses `ASSERT` for fail-fast behavior per model
- Better integration with CTest reporting

### 2. Key Features

#### Automatic Model Discovery
```cpp
std::vector<std::string> find_all_models()
{
    // Scans models/ directory for all .gguf files
    // Filters for Qwen models (LLaMA adapter not fully implemented)
    // Returns sorted list of model paths
}
```

#### Parametrized Test Registration
```cpp
INSTANTIATE_TEST_SUITE_P(
    AllModels,
    InferenceParityTest,
    ::testing::ValuesIn(find_all_models()),
    [](const ::testing::TestParamInfo<std::string>& info) {
        // Generate sanitized test names from filenames
        // e.g., "qwen2_5_0_5b_instruct_q4_0"
    }
);
```

#### Test Workflow Per Model
1. **Generate PyTorch Reference**: Calls `generate_text_reference.py` with GGUF model
2. **Load Model in Llaminar**: Uses same GGUF file
3. **Generate Text**: Both systems use greedy sampling (temp=0.0)
4. **Compare Tokens**: Token-by-token comparison
5. **Report Results**: Clear pass/fail per model

### 3. Models Tested

Currently tests **10 Qwen models** from `models/` directory:

1. `Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf` (FP32)
2. `qwen2.5-0.5b-instruct-fp16.gguf` (FP16)
3. `qwen2.5-0.5b-instruct-q2_k.gguf` (Q2_K)
4. `qwen2.5-0.5b-instruct-q3_k_m.gguf` (Q3_K_M)
5. `qwen2.5-0.5b-instruct-q4_0.gguf` (Q4_0)
6. `qwen2.5-0.5b-instruct-q4_k_m.gguf` (Q4_K_M)
7. `qwen2.5-0.5b-instruct-q5_0.gguf` (Q5_0)
8. `qwen2.5-0.5b-instruct-q5_k_m.gguf` (Q5_K_M)
9. `qwen2.5-0.5b-instruct-q6_k.gguf` (Q6_K)
10. `qwen2.5-0.5b-instruct-q8_0.gguf` (Q8_0)

**Note:** LLaMA models excluded until adapter fully implemented.

### 4. Running the Tests

```bash
# List all test cases
mpirun -np 2 ./build/test_inference_parity --gtest_list_tests

# Run all models
mpirun -np 2 ./build/test_inference_parity

# Run specific model
mpirun -np 2 ./build/test_inference_parity \
  --gtest_filter="*qwen2_5_0_5b_instruct_q4_0"

# Run via CTest
ctest --test-dir build -R InferenceParity
```

### 5. Test Output Example

```
[==========] Running 10 tests from 1 test suite.
[----------] 10 tests from AllModels/InferenceParityTest
[ RUN      ] AllModels/InferenceParityTest.SimpleEnglishPrompt/qwen2_5_0_5b_instruct_q4_0

--------------------------------------------------------------------------------
Testing model: models/qwen2.5-0.5b-instruct-q4_0.gguf
--------------------------------------------------------------------------------

[PYTORCH_REF] Generating reference for: 'The capital of France is'
[PYTORCH_REF] ✓ Reference generated successfully

[PYTORCH] Generated tokens: 12095, 11, 323, 432, 374, 7148, 6722, 315, 4505, 13
[PYTORCH] Text: " Paris, and the capital of France is Paris."

[LLAMINAR] Generating with 7 prompt tokens, max 10 new tokens
[LLAMINAR] Generated 10 tokens
[LLAMINAR] First few tokens: 12095, 11, 279, 6722, 315, 279, 3639, 15072, 374, 7148
[LLAMINAR] Generated text: " Paris, the capital of the United Kingdom is London"

[COMPARISON] Comparing 10 PyTorch tokens vs 10 Llaminar tokens
[MISMATCH] Position 2: PyTorch=323, Llaminar=279
[MISMATCH] Position 3: PyTorch=432, Llaminar=6722
...
[COMPARISON] ✗ Token mismatch detected

[  FAILED  ] AllModels/InferenceParityTest.SimpleEnglishPrompt/qwen2_5_0_5b_instruct_q4_0 (38341 ms)
```

## Benefits

### 1. **Comprehensive Coverage**
- Tests all quantization levels (FP32, FP16, Q2_K through Q8_0)
- Validates both Llaminar and PyTorch use identical GGUF files
- Ensures apples-to-apples comparison

### 2. **Clear Test Results**
- Each model has individual test case
- Easy to identify which quantization levels work
- CTest integration for CI/CD pipelines

### 3. **Fail-Fast Behavior**
- `ASSERT` stops test immediately on first failure
- Saves time when debugging specific model issues
- Clear failure location per model

### 4. **Maintainable**
- Adding new models is automatic (just drop in `models/`)
- Test names generated from filenames
- No hardcoded model lists to maintain

### 5. **Debugging Support**
- Detailed token mismatch reporting
- Shows exact position and values of divergence
- Prints generated text from both systems

## Documentation Updates

Added comprehensive section to `parity-test-framework.instructions.md`:

**"Loading and Using GGUF Files for PyTorch Inference"**

Documents how PyTorch reference implementation:
- Loads quantized GGUF files directly
- Dequantizes tensors to FP32 using `dequantize.py`
- Maps weights to HuggingFace model
- Runs inference with identical weights as Llaminar

This ensures developers understand the apples-to-apples comparison guarantee.

## Current Status

### ✅ **Working**
- Test infrastructure compiles and runs
- All 10 models discovered and tested
- PyTorch reference generation works for all models
- Token comparison logic functional
- Clear failure reporting

### ⚠️ **Known Issues**
- Token mismatches detected (implementation bugs to investigate)
- Example: q4_0 model shows divergence starting at position 2
- Not a test infrastructure issue - reveals real parity problems

### 🔜 **Next Steps**
1. Investigate token divergence root causes
2. Compare intermediate activation stages (use parity framework)
3. Identify numerical precision vs logic bugs
4. Fix divergence issues per quantization level
5. Validate all models achieve perfect parity

## Technical Details

### Test Structure

```cpp
// Test fixture
class InferenceParityTest : public ::testing::TestWithParam<std::string> {};

// Test case (runs once per model)
TEST_P(InferenceParityTest, SimpleEnglishPrompt)
{
    const std::string model_path = GetParam();
    // ... test logic ...
}

// Registration (automatic discovery)
INSTANTIATE_TEST_SUITE_P(AllModels, InferenceParityTest,
    ::testing::ValuesIn(find_all_models()), /* name generator */);
```

### Helper Functions

| Function | Purpose |
|----------|---------|
| `find_all_models()` | Discover all GGUF files in `models/` |
| `test_model_inference_parity()` | Run parity test for single model |
| `generate_pytorch_reference()` | Call Python script to generate reference |
| `generate_llaminar_text()` | Run Llaminar inference |
| `PyTorchReference::load()` | Parse JSON reference output |

## Files Modified

- ✅ `tests/test_inference_parity.cpp` - Parametrized test implementation
- ✅ `.github/instructions/parity-test-framework.instructions.md` - GGUF loading documentation
- ✅ `CMakeLists.txt` - Test target already configured

## Validation

```bash
# Test compiles successfully
cmake --build build --target test_inference_parity --parallel

# Lists 10 parametrized test cases (1 per model)
mpirun -np 2 ./build/test_inference_parity --gtest_list_tests

# Runs all tests (currently failing due to implementation bugs)
mpirun -np 2 ./build/test_inference_parity
```

## Conclusion

The inference parity test now provides **comprehensive, automated validation** of Llaminar against PyTorch for all quantized models. While token mismatches are currently detected (expected - reveals real bugs), the test infrastructure successfully validates that both systems use identical GGUF files and provides clear, actionable failure reports for debugging.

This establishes a **robust foundation** for ensuring Llaminar maintains inference parity as the codebase evolves.
