# QuantisedGemmKernel Superblock Integration

**Date**: 2025-12-01
**Status**: Complete

## Summary

Integrated the `unpack_superblock_to_int8()` API into `QuantisedGemmKernel::pack_weights_generic`. This optimization reduces virtual function call overhead by processing 256 elements (8 blocks) at a time instead of 32 elements.

## Changes

### `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h`

- Added `pack_single_block` helper method to encapsulate the packing logic for a 32-element block.
- Modified `pack_weights_generic` to check `unpackable->superblock_size()`.
- If superblock size is 256, the loop iterates over superblocks and calls `unpack_superblock_to_int8`, then packs the 8 sub-blocks using the helper.
- Fallback to `unpack_block_to_int8` for remaining blocks or non-superblock formats.

### Performance Results

Measured using `tests/v2/performance/kernels/Perf__QuantisedGemmPacking.cpp` on a 4096 x 4096 matrix.

| Format | Old Time (ms) | New Time (ms) | Speedup |
|--------|---------------|---------------|---------|
| Q4_K   | 8.24          | 7.58          | **1.09x** |
| Q6_K   | 6.11          | 5.95          | **1.03x** |
| IQ4_XS | 5.99          | 5.76          | **1.04x** |

**Analysis**:
- The speedup is primarily due to reducing virtual function calls from 24 calls (8 unpack + 8 scale + 8 min) to 1 call per 256 elements.
- This improves model loading time, as `QuantisedGemmKernel` is instantiated during pipeline initialization.
- Further speedups are expected when `unpack_superblock_to_int8` implementations are optimized with fused SIMD.

## Files Changed

- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h`
- `tests/v2/performance/kernels/Perf__QuantisedGemmPacking.cpp` (New benchmark)
- `tests/v2/CMakeLists.txt` (Registered benchmark)
