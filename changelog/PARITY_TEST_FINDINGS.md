# True Incremental Decode Parity Test - Initial Results

**Date**: October 11, 2025  
**Test**: `ParityFramework.TrueIncrementalDecodeVsPyTorch`  
**Status**: ✗ **FAILED** - Critical divergence detected  
**Duration**: 27.5 seconds

## Executive Summary

The dual-level validation system successfully identified a **critical functional divergence** between PyTorch and Llaminar. The token sequence comparison immediately flagged that the two systems are generating completely different outputs, indicating a serious bug that needs investigation.

## Key Findings

### 1. Token Sequence Divergence ⚠️ **CRITICAL**

**PyTorch Output:**
```
[62162 → 997 → 5 → 5 → 6 → 25010 → 1784 → 93]
```

**Llaminar Output:**
```
[6 → 13956 → 99822]
```

**Analysis:**
- ✗ **Different sequence lengths**: PyTorch generated 8 tokens, Llaminar only 3
- ✗ **Completely different token IDs**: No overlap in generated tokens
- ✗ **Functional failure**: The systems produce entirely different text

### 2. Stage-Level Issues

**Shape Mismatches:**
- PyTorch snapshots have shape `1x1` (single token, single feature?)
- Llaminar snapshots have correct shapes (`1x896`, `1x128`, etc.)

**Missing Snapshots:**
- Multiple `MLP_NORM_layerN.npy` files missing from Llaminar output
- Multiple `MLP_OUTPUT_layerN.npy` files missing from Llaminar output

**Statistics:**
- Tokens compared: 3
- Tokens passed: 0/3
- Stages failed: 729
- Stages passed: 0

## Root Cause Analysis

### Immediate Suspects

1. **PyTorch Snapshot Generation Issue**
   - PyTorch snapshots all have shape `1x1` - this is clearly wrong
   - Should have full tensor shapes like `1x896`, `1x128`, etc.
   - Problem is likely in `python/reference/generate_incremental_decode_snapshots.py`

2. **Token Input Mismatch**
   - Test sends tokens `[1, 2, 3, 4, 5, 6, 7, 8]` to PyTorch
   - But PyTorch generates `[62162, 997, 5, 5, 6, 25010, 1784, 93]`
   - The token IDs `5` and `6` appear in PyTorch output, but in different positions
   - This suggests the input token sequence might not be interpreted correctly

3. **Missing Stage Names**
   - Llaminar uses `FFN_*` naming (FFN_NORM, FFN_SWIGLU, etc.)
   - PyTorch expects `MLP_*` naming (MLP_NORM, MLP_OUTPUT)
   - Naming mismatch prevents comparison

## Dual-Level Validation Success ✅

Despite the test failure, the **dual-level validation worked perfectly**:

### Token Comparison (Level 1 - Functional)
✅ **Immediately identified the problem**:
```
[TOKEN SEQUENCE VALIDATION]
  ✗ Token sequences DIVERGE
    Functional output differs between systems
```

### Stage Comparison (Level 2 - Numerical)
✅ **Provided detailed diagnostics**:
- Showed shape mismatches
- Identified missing snapshots
- Counted failures per token

### Value of Dual Validation

Without token comparison, we might have:
1. Spent hours debugging shape mismatches
2. Missed that the fundamental output is completely wrong
3. Focused on numerical precision instead of functional correctness

**With token comparison:**
- Instant recognition of critical failure
- Clear priority: fix token divergence first
- Shape issues are secondary (might be snapshot generation bugs)

## Recommended Investigation Steps

### Step 1: Fix PyTorch Snapshot Shape Issue
**File**: `python/reference/generate_incremental_decode_snapshots.py`

**Check:**
```python
# In incremental_decode_with_cache(), verify tensor capture
token_snapshots = capturer.capture_stages([token_id], past_key_values=past_key_values)

# Are tensors being squeezed incorrectly?
# Are they being saved with wrong shapes?
```

**Expected Fix:**
- Snapshots should have full shapes (1×hidden_dim, not 1×1)
- May need to check `PipelineStageCapture.capture_stages()` implementation

### Step 2: Verify Token Input Interpretation
**Question**: Does PyTorch script correctly use the input tokens `[1,2,3,4,5,6,7,8]`?

**Check:**
```python
# In generate_incremental_decode_snapshots.py
def incremental_decode_with_cache(model_path, token_sequence, verbose):
    # Is token_sequence = [1,2,3,4,5,6,7,8]?
    # Or is it being interpreted differently?
    print(f"Processing token sequence: {token_sequence}")
```

### Step 3: Align Stage Naming
**Issue**: Llaminar uses `FFN_*`, PyTorch expects `MLP_*`

**Options:**
1. Update PyTorch snapshot code to use `FFN_*` naming
2. Update Llaminar to use `MLP_*` naming
3. Add name mapping in comparison logic

### Step 4: Debug Llaminar Token Generation
Once PyTorch snapshots are fixed, if divergence remains:

**Check:**
- Is Llaminar reading the correct input tokens?
- Is the embedding layer working correctly?
- Is there an off-by-one error in token indexing?

## Test Infrastructure Status

### What Works ✅
- ✅ Python snapshot generation executes (even if output is wrong)
- ✅ Llaminar incremental decode executes
- ✅ Snapshot saving and loading
- ✅ Token comparison logic
- ✅ Stage comparison logic
- ✅ Dual-level validation reporting
- ✅ Test completes without crashes

### What Needs Fixing ❌
- ❌ PyTorch snapshot shape (critical)
- ❌ Token sequence alignment
- ❌ Stage name consistency
- ❌ Missing Llaminar snapshots (MLP_NORM, MLP_OUTPUT)

## Next Actions

### Priority 1: Fix PyTorch Snapshot Generation
1. Investigate shape issue (`1x1` instead of `1x896`)
2. Verify tensor capture in `PipelineStageCapture`
3. Ensure `.npy` files save full tensors

### Priority 2: Verify Token Input
1. Add debug logging to show input token sequence
2. Confirm PyTorch processes `[1,2,3,4,5,6,7,8]` correctly
3. Check if there's a tokenizer/detokenizer issue

### Priority 3: Stage Naming Alignment
1. Decide on standard naming convention
2. Update code to use consistent names
3. Or add mapping logic in comparison

### Priority 4: Re-run Test
Once fixes are in place:
```bash
./build/test_parity_framework --gtest_filter=ParityFramework.TrueIncrementalDecodeVsPyTorch
```

Expected outcome after fixes:
- ✅ Token sequences match
- ✅ Stage shapes match
- ✅ Numerical values within tolerance
- ✅ Test passes

## Conclusion

The parity test framework is working as designed:
1. **Token comparison** caught the critical bug immediately
2. **Stage comparison** provided detailed diagnostics
3. **Dual-level validation** shows both functional and numerical status

The test failure is **expected and valuable** - it revealed real issues that need fixing. This is exactly what parity testing is for: catching bugs early before they cause downstream problems.

**Status**: Infrastructure ✅ | Implementation ❌ | Framework Value: **High** 🎯
