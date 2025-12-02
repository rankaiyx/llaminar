# Super-block Unpack API for IINT8Unpackable Interface

**Date**: 2025-12-01  
**Status**: Complete (Infrastructure for future optimization)

## Summary

Extended the `IINT8Unpackable` interface with a new `unpack_superblock_to_int8()` method that processes 256 elements (a full super-block) in one call instead of requiring 8 separate `unpack_block_to_int8()` calls. This API is designed to enable future fused SIMD optimizations.

## Changes

### Interface Updates (`src/v2/tensors/Tensors.h`)

Added to `IINT8Unpackable` interface:
```cpp
// Returns 32 for simple formats, 256 for super-block formats
virtual size_t superblock_size() const { return 32; }

// Unpack 256 elements + 8 scales + 8 mins in one call
virtual void unpack_superblock_to_int8(
    size_t row_idx,
    size_t superblock_idx,
    int8_t *output,
    float *scales = nullptr,
    float *mins = nullptr) const;
```

### Implementations Added (13 tensor classes)

**K-quant formats (asymmetric, have mins)**:
- `Q2_KTensor.cpp` - Uses `simd::transcode_q2_k_to_int8` per sub-block
- `Q3_KTensor.cpp` - Uses `simd::transcode_q3_k_to_int8` per sub-block
- `Q4_KTensor.cpp` - Uses `simd::transcode_q4_k_to_int8` per sub-block
- `Q5_KTensor.cpp` - Uses `simd::transcode_q5_k_to_int8` per sub-block
- `Q6_KTensor.cpp` - Uses `simd::transcode_q6_k_to_int8` per sub-block

**IQuant formats (symmetric, no mins)**:
- `IQ1_STensor.cpp` - Uses `simd::decode_iq1s_to_q8_0` per sub-block
- `IQ1_MTensor.cpp` - Uses `simd::decode_iq1m_to_q8_0` per sub-block
- `IQ2_XXSTensor.cpp` - Uses `simd::decode_iq2xxs_to_q8_0` per sub-block
- `IQ2_XSTensor.cpp` - Uses `simd::decode_iq2xs_to_q8_0` per sub-block
- `IQ2_STensor.cpp` - Uses `simd::decode_iq2s_to_q8_0` per sub-block
- `IQ3_XXSTensor.cpp` - Uses `simd::decode_iq3xxs_to_q8_0` per sub-block
- `IQ3_STensor.cpp` - Uses `simd::decode_iq3s_to_q8_0` per sub-block
- `IQ4_XSTensor.cpp` - Uses `simd::unpack_iq4_xs_to_int8` + `get_iq4_xs_scale` per sub-block

### Benchmark Added (`tests/v2/performance/tensors/Perf__IINT8Unpackable.cpp`)

New test `IINT8UnpackablePerf.SuperblockAPIComparison` comparing:
- Old API: 8 × `unpack_block_to_int8()` calls per super-block
- New API: 1 × `unpack_superblock_to_int8()` call per super-block

## Performance Results

Matrix size: 2048 × 2048 = 4.19M elements

| Format  | Old (8 calls) | New (1 call) | Speedup |
|---------|---------------|--------------|---------|
| Q4_K    | 768 µs        | 835 µs       | 0.92x   |
| Q6_K    | 1697 µs       | 1529 µs      | **1.11x** |
| IQ3_S   | 4122 µs       | 4615 µs      | 0.89x   |
| IQ4_XS  | 562 µs        | 696 µs       | 0.81x   |

**Key Insight**: The superblock API currently provides ~parity or slight slowdown because the implementations just loop over 8 sub-blocks internally. The performance benefit would come from:

1. **Fused SIMD implementations** that process all 256 elements in a single vectorized pass
2. **Better cache locality** by keeping all sub-block data hot during one call
3. **Reduced function call overhead** for very hot paths

## Why Q6_K Shows 1.11x Speedup

Q6_K's superblock implementation uses the existing SIMD transcode function efficiently:
- `transcode_q6_k_to_int8` already uses AVX512/AVX2 for 32-element sub-blocks
- The loop over 8 sub-blocks benefits from cache warming effects
- Less branching overhead compared to 8 separate interface calls

## Future Optimization Path

To realize performance gains from this API:

1. **Implement fused transcode functions** in `SIMDHelpers.h`:
   ```cpp
   void transcode_q4_k_superblock_to_int8(
       const Q4_KBlock& block,
       int8_t* output_256,
       float* scales_8,
       float* mins_8);
   ```

2. **Use wider SIMD registers** (AVX512 can process 64 elements at once, potentially handling 4 sub-blocks per instruction)

3. **Amortize scale computations** across sub-blocks when possible

## Files Changed

- `src/v2/tensors/Tensors.h` - Interface + declarations
- `src/v2/tensors/Q2_KTensor.cpp` - Implementation
- `src/v2/tensors/Q3_KTensor.cpp` - Implementation
- `src/v2/tensors/Q4_KTensor.cpp` - Implementation
- `src/v2/tensors/Q5_KTensor.cpp` - Implementation
- `src/v2/tensors/Q6_KTensor.cpp` - Implementation
- `src/v2/tensors/IQ1_STensor.cpp` - Implementation
- `src/v2/tensors/IQ1_MTensor.cpp` - Implementation
- `src/v2/tensors/IQ2_XXSTensor.cpp` - Implementation
- `src/v2/tensors/IQ2_XSTensor.cpp` - Implementation
- `src/v2/tensors/IQ2_STensor.cpp` - Implementation
- `src/v2/tensors/IQ3_XXSTensor.cpp` - Implementation
- `src/v2/tensors/IQ3_STensor.cpp` - Implementation
- `src/v2/tensors/IQ4_XSTensor.cpp` - Implementation
- `tests/v2/performance/tensors/Perf__IINT8Unpackable.cpp` - Benchmark

## Testing

- All V2 unit tests pass
- Performance benchmark validates API correctness
- Release and debug builds compile cleanly
