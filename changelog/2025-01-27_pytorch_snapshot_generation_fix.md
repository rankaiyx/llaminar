# PyTorch Snapshot Generation Fix

**Date**: 2025-01-27 (October 11, 2025 actual)  
**Author**: David Sanftenberg  
**Status**: ✅ Complete

## Summary

Fixed the missing `save_incremental_snapshots()` implementation in `generate_incremental_decode_snapshots.py` that was causing the incremental parity test to fail with "Could not load PyTorch tokens for comparison".

## Problem

The test `ParityFramework.TrueIncrementalDecodeVsPyTorch` was failing with:
```
Failed to open: pytorch_incremental_snapshots/sampled_tokens.json
  ⚠ Warning: Could not load PyTorch tokens for comparison
```

### Root Cause

The `save_incremental_snapshots()` function in `python/reference/generate_incremental_decode_snapshots.py` was **not implemented** - it was just an empty stub with a docstring:

```python
def save_incremental_snapshots(
    snapshots: Dict[int, Dict[str, np.ndarray]],
    sampled_tokens: List[int],
    output_dir: Path,
    verbose: bool = False
) -> None:
    """
    Save incremental decode snapshots to disk.
    ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)  # Only this line existed
    # Function body was missing!


def save_model_weights(...):  # Next function started immediately
```

This meant that:
1. The script would run and claim success
2. But it would only create the output directory and weights subdirectory
3. No token snapshots (token_0/, token_1/, etc.) were saved
4. No `sampled_tokens.json` was created
5. The C++ test couldn't load PyTorch reference data for comparison

## Solution

Implemented the `save_incremental_snapshots()` function to:

1. **Save sampled_tokens.json**:
```json
{
  "sampled_tokens": [6, 25010, 10],
  "num_tokens": 3,
  "description": "Greedy-sampled tokens from incremental decode"
}
```

2. **Save snapshot directories**:
   - `token_0/` - First decode token snapshots (195 stages)
   - `token_1/` - Second decode token snapshots (195 stages)
   - `token_2/` - Third decode token snapshots (195 stages)

3. **Each token directory contains**:
   - All pipeline stage snapshots as `.npy` files
   - `EMBEDDING.npy`, `Q_PROJECTION_layer0.npy`, etc.
   - 195 stages per incremental decode token

### Implementation

```python
def save_incremental_snapshots(
    snapshots: Dict[int, Dict[str, np.ndarray]],
    sampled_tokens: List[int],
    output_dir: Path,
    verbose: bool = False
) -> None:
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sampled tokens JSON
    tokens_file = output_dir / "sampled_tokens.json"
    with open(tokens_file, 'w') as f:
        json.dump({
            "sampled_tokens": sampled_tokens,
            "num_tokens": len(sampled_tokens),
            "description": "Greedy-sampled tokens from incremental decode"
        }, f, indent=2)
    
    # Save snapshots for each token
    for token_idx, stage_dict in sorted(snapshots.items()):
        token_dir = output_dir / f"token_{token_idx}"
        token_dir.mkdir(parents=True, exist_ok=True)
        
        for stage_name, stage_data in stage_dict.items():
            stage_path = token_dir / f"{stage_name}.npy"
            np.save(stage_path, stage_data)
```

## Testing

### Before Fix
```bash
$ ls pytorch_incremental_snapshots/
total 20
drwxr-xr-x  3 vscode vscode  4096 Oct 11 21:22 .
drwxrwxr-x 27 vscode vscode 12288 Oct 11 21:22 ..
drwxr-xr-x  2 vscode vscode  4096 Oct 11 21:22 weights  # Only weights!
```

### After Fix
```bash
$ ls pytorch_incremental_snapshots/
total 60
drwxr-xr-x  6 vscode vscode  4096 Oct 11 21:29 .
drwxrwxr-x 27 vscode vscode 12288 Oct 11 21:28 ..
-rw-r--r--  1 vscode vscode   139 Oct 11 21:28 sampled_tokens.json  ✓
drwxr-xr-x  2 vscode vscode 12288 Oct 11 21:28 token_0              ✓
drwxr-xr-x  2 vscode vscode 12288 Oct 11 21:28 token_1              ✓
drwxr-xr-x  2 vscode vscode 12288 Oct 11 21:28 token_2              ✓
drwxr-xr-x  2 vscode vscode  4096 Oct 11 21:29 weights

$ cat pytorch_incremental_snapshots/sampled_tokens.json
{
  "sampled_tokens": [
    6,
    25010,
    10
  ],
  "num_tokens": 3,
  "description": "Greedy-sampled tokens from incremental decode"
}
```

### Test Results

**Before**: 
```
Failed to open: pytorch_incremental_snapshots/sampled_tokens.json
  ⚠ Warning: Could not load PyTorch tokens for comparison
```

**After**:
```
[TOKEN SEQUENCE VALIDATION]
  ✗ Token sequences DIVERGE
    Functional output differs between systems
  Generated tokens: 400 → 1 → 66 (Llaminar)
  Reference tokens: 6 → 25010 → 10 (PyTorch)
```

The test now **successfully loads** PyTorch reference data! The token divergence is a separate issue (numerical computation bug, not data generation bug).

## Files Modified

**python/reference/generate_incremental_decode_snapshots.py**:
- Lines 306-370: Implemented `save_incremental_snapshots()` function
- Added JSON serialization for sampled tokens
- Added directory creation and .npy file saving for each token's snapshots
- Added verbose logging for debugging

## Impact

- ✅ PyTorch reference data now generates correctly
- ✅ Test can load and compare token sequences
- ✅ Test infrastructure is working as designed
- ⚠️  Token divergence revealed (separate bug to investigate)

## Next Steps

The fix enables the parity test to properly compare PyTorch vs Llaminar outputs, revealing that there is a **token sequence divergence**:
- PyTorch: [6, 25010, 10]
- Llaminar: [400, 1, 66]

This is now the primary bug to investigate. Possible causes:
1. Weight slicing issues (though contracts validate ✓)
2. Attention computation differences
3. RoPE application differences
4. Softmax numerical precision
5. KV cache management issues

The divergence starts from the **first generated token**, suggesting the issue is in the core inference path, not accumulation over time.

## Related Work

- `changelog/2025-01-27_weight_slicing_contracts.md` - Weight validation system (passing ✓)
- `tests/test_parity_framework.cpp` - Test infrastructure
- `python/reference/generate_incremental_decode_snapshots.py` - Snapshot generation script

## Lessons Learned

1. **Always verify test infrastructure**: The test was failing not because of code bugs, but because reference data wasn't being generated.

2. **Function stubs are dangerous**: The function existed with a docstring, so it looked complete, but had no implementation body.

3. **Manual testing catches infrastructure bugs**: Running the Python script manually revealed the missing JSON file immediately.

4. **Incremental debugging**: Fixing the data generation bug revealed the actual numerical bug underneath.

