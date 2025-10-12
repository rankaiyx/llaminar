# Snapshot Generation Investigation
**Date**: 2025-01-27  
**Issue**: Token divergence traced to PyTorch snapshot generation

## Summary

Investigated why parity tests showed stale/incorrect PyTorch snapshots. Found that:

1. ✅ **Test framework regenerates snapshots correctly** - Always cleans up and regenerates on every run (lines 2508-2512)
2. ✅ **Snapshot file format is correct** - PyTorch generates proper (1, 1, 896) shaped arrays
3. ❌ **Snapshot VALUES are wrong** - PyTorch is generating completely different values than Llaminar

## Investigation Details

### Root Cause of Confusion

The test was printing misleading error messages:
```
✗ Shape mismatch in EMBEDDING.npy: PyTorch=1x1 vs Llaminar=1x896
```

This made it appear that PyTorch was generating 1x1 scalars, when in reality:
- PyTorch generates: `(1, 1, 896)` - batch, seq, features
- Llaminar generates: `(1, 896)` - seq, features
- The test was only printing first 2 dimensions!

### Fix Applied

Modified `test_parity_framework.cpp` (lines 2873-2912) to:
1. Print ALL shape dimensions, not just first two
2. Squeeze leading singleton dimensions before comparison
3. Show both original and squeezed shapes in error messages

### Actual Problem

After fixing the shape comparison, the real issue became clear:
```
[TRUE_INCR]   ✗ K_PROJECTION_layer2.npy FAILED (max_abs=94.6962, rel_l2=11.7875)
[TRUE_INCR]   ✗ Q_PROJECTION_layer8.npy FAILED (max_abs=17.5985, rel_l2=2.53481)
[TRUE_INCR]   ✗ V_PROJECTION_layer22.npy FAILED (max_abs=8.97918, rel_l2=3.03131)
```

**MASSIVE value divergence** (up to 94.7 absolute error!). PyTorch is generating completely different activation values.

## Answer to Original Question

**"Do we have stale snapshots because we were running scripts manually?"**

**NO**. The test framework is innocent. It:
- Cleans up old snapshots before EVERY run (`rm -rf pytorch_incremental_snapshots`)
- Generates fresh PyTorch snapshots using `generate_incremental_decode_snapshots.py`
- Compares them against fresh Llaminar snapshots

The "stale" snapshots we saw earlier were from manual script runs during debugging, not from the automated test.

## Next Steps

The real problem is **PyTorch snapshot generation produces wrong values**. Need to investigate:

1. **Model loading**: Does PyTorch load the GGUF weights correctly?
2. **Forward pass**: Does PyTorch's HuggingFace model match Llaminar's implementation?
3. **Hook capture**: Are the hooks capturing the right tensor values?

The fact that our earlier manual debug script (`debug_pytorch_snapshots.py`) showed correct embedding values suggests the issue might be in:
- Incremental decode with KV cache
- Hook-based capture vs direct forward pass
- Attention mechanism differences

## Code Changes

- `tests/test_parity_framework.cpp`: Enhanced shape comparison with squeeze + full dimension printing
