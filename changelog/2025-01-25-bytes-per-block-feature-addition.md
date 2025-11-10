# Critical Feature Addition: `bytes_per_block` for Bandwidth Modeling

**Date**: November 25, 2025  
**Author**: David Sanftenberg  
**Context**: CPU ML Heuristic Framework - Multi-Quant Support

## Problem Statement

The quantization features (`quant_block_size`, `bits_per_weight`) capture the **logical** structure of quantization blocks, but miss a critical dimension: **physical memory bandwidth requirements**.

**Example**: Q8_0 and Q4_0 both have `block_size=32`, but Q8_0 loads **34 bytes** vs Q4_0's **18 bytes** per block - nearly **2× the bandwidth!**

## Solution: Add `bytes_per_block` Feature

### Physical Block Sizes (from `src/v2/tensors/Tensors.h`)

```cpp
struct IQ4_NLBlock { uint16_t d; uint8_t qs[16]; };      // 18 bytes
struct Q4_0Block    { uint16_t d; uint8_t qs[16]; };      // 18 bytes
struct Q8_0Block    { uint16_t d; int8_t qs[32]; };       // 34 bytes
struct Q6_KBlock    { /* complex layout */ };             // 210 bytes
// FP16: No struct, 2 bytes per element
```

### Bandwidth Impact Analysis

| Format | Elements/Block | **Bytes/Block** | Bytes/Element | **Cache Lines/Block** | **Bandwidth vs IQ4_NL** |
|--------|---------------|-----------------|---------------|----------------------|------------------------|
| IQ4_NL | 32 | **18** | 0.5625 | 0.28 (fits in <1 line) | **1.0× (baseline)** |
| Q4_0   | 32 | **18** | 0.5625 | 0.28 (fits in <1 line) | **1.0× (identical!)** |
| Q8_0   | 32 | **34** | 1.0625 | 0.53 (~half a line) | **1.9× higher** |
| Q6_K   | 256 | **210** | 0.8203 | 3.28 (>3 lines!) | **11.7× higher** |
| FP16   | 1 | **2** | 2.0 | 0.03 (32/line) | 0.11× (no decode) |

### Key Insights

1. **Same `block_size`, different bandwidth**: Q8_0 vs Q4_0 have identical logical block size (32) but Q8_0 requires **89% more bandwidth** (34 vs 18 bytes)

2. **Cache line efficiency**:
   - IQ4_NL/Q4_0: 18 bytes → **3.5 blocks per 64B cache line** (excellent!)
   - Q8_0: 34 bytes → **1.9 blocks per cache line** (moderate)
   - Q6_K: 210 bytes → **0.3 blocks per cache line** (poor, requires 3.3 lines/block)

3. **Independent of `quant_block_size`**: This affects microkernel bandwidth even if decode happens at the same granularity

4. **Prefetch effectiveness**: Larger `bytes_per_block` means:
   - More cache lines to prefetch
   - Higher risk of evicting useful data
   - Different optimal prefetch distance

5. **IQ4_NL vs Q4_0 should prefer identical tiles**: Both have **exactly the same physical layout** (18 bytes, 32 elements)

## Implementation

### 1. Updated `QuantFormat` Dataclass

```python
@dataclass
class QuantFormat:
    name: str                # "IQ4_NL", "Q6_K", etc.
    model_path: str          # Path to .gguf file
    block_size: int          # Logical: 32, 256, 1
    bits_per_weight: float   # Effective: 4.5, 6.6, 16.0
    bytes_per_block: int     # Physical: 18, 34, 210, 2 (NEW!)
```

### 2. Format Definitions with Byte Sizes

```python
QUANT_FORMATS = [
    QuantFormat("IQ4_NL", "models/qwen2.5-0.5b-instruct-iq4_nl.gguf", 32, 4.5, 18),
    QuantFormat("Q4_0",   "models/qwen2.5-0.5b-instruct-q4_0.gguf",   32, 4.5, 18),
    QuantFormat("Q8_0",   "models/qwen2.5-0.5b-instruct-q8_0.gguf",   32, 8.5, 34),
    QuantFormat("Q6_K",   "models/qwen2.5-0.5b-instruct-q6_k.gguf",   256, 6.6, 210),
    QuantFormat("FP16",   "models/qwen2.5-0.5b-instruct-fp16.gguf",   1, 16.0, 2),
]
```

### 3. CSV Output (Now 32 Columns)

```csv
# Added column (total 5 quant features):
quant_format,quant_block_size,bits_per_weight,bytes_per_block,quant_alignment
IQ4_NL,32,4.5,18,0
Q8_0,32,8.5,34,0
Q6_K,256,6.6,210,0
```

### 4. Training Features (Now 27 Total)

```python
feature_cols = [
    # ... existing features ...
    'quant_block_size',   # Logical
    'bits_per_weight',    # Compression ratio
    'bytes_per_block',    # Physical (NEW - CRITICAL!)
    'quant_alignment',    # K % block_size
    # ... rest ...
]
# Total: 24 base + 3 log = 27 features (was 26)
```

## Expected ML Model Behavior

The neural network should learn:

1. **Bandwidth-sensitive tile selection**:
   - Smaller `bytes_per_block` → Can afford larger tiles (more data per cache line)
   - Larger `bytes_per_block` → Prefer smaller tiles to stay in cache

2. **Prefetch distance tuning**:
   - Q6_K (210 bytes) → Needs larger prefetch distance to hide latency
   - IQ4_NL (18 bytes) → Can use shorter prefetch distance

3. **IQ4_NL/Q4_0 convergence**:
   - These formats should select **identical optimal tiles** (same physical layout)
   - Model learns this from data, not hardcoded rules

4. **Q8_0 vs Q4_0 divergence**:
   - Despite same `block_size=32`, bandwidth difference (34 vs 18 bytes) should lead to different tile preferences
   - Q8_0 may prefer smaller tiles to reduce cache pressure

## Validation Strategy

### Post-Training Analysis

1. **Per-format tile distribution**:
   ```python
   for fmt in ['IQ4_NL', 'Q4_0', 'Q8_0', 'Q6_K', 'FP16']:
       top_tiles = get_top_tiles(model, format=fmt)
       print(f"{fmt}: {top_tiles}")
   ```
   Expected: IQ4_NL ≈ Q4_0 (identical), Q8_0 different, Q6_K very different

2. **Bandwidth correlation**:
   ```python
   correlation = np.corrcoef(df['bytes_per_block'], df['optimal_tile_area'])
   # Expected: Negative correlation (more bytes → smaller tiles)
   ```

3. **Prefetch distance vs bytes_per_block**:
   ```python
   # Group by bytes_per_block, compare optimal prefetch_dist
   # Expected: 210 bytes (Q6_K) → larger prefetch_dist
   ```

## Files Modified

1. **`src/v2/kernels/cpu/python/benchmark_cpu_gemm.py`**:
   - Added `bytes_per_block` field to `QuantFormat`
   - Updated format definitions with byte sizes
   - Added `bytes_per_block` to CSV output (column 8)
   - Added comments explaining bandwidth impact

2. **`src/v2/kernels/cpu/python/train_cpu_gemm_heuristic.py`**:
   - Added `bytes_per_block` to `feature_cols` (now 27 total)
   - Model input size auto-adapts to 27 features

3. **`changelog/2025-01-25-cpu-ml-heuristic-multi-quant-support.md`**:
   - Updated feature count: 26 → 27
   - Added bandwidth impact analysis table
   - Documented IQ4_NL/Q4_0 convergence expectation

## Impact on Data Volume

**No change**: Still ~185K benchmarks
- Same quant formats tested
- Same shapes × variants
- Just 1 additional column in CSV
- Negligible impact on runtime (<0.1%)

## Next Steps

1. ✅ **Complete model downloads** (14B Q8_0/Q6_K/FP16, 32B all formats)
2. ✅ **Run benchmark suite** (~63-76 hours)
3. ✅ **Train neural network** with 27 features
4. **Validate bandwidth hypothesis**:
   - Check IQ4_NL vs Q4_0 tile convergence
   - Check Q8_0 divergence from Q4_0 despite same block_size
   - Check Q6_K extreme case (210 bytes)
5. **Export C++ header** with trained weights
6. **Integrate** into SmartGemmSearch.cpp
7. **Verify** L1 miss rate <5% in production

## Conclusion

Adding `bytes_per_block` captures a **critical dimension** that was missing from the original feature set:

- **Before**: Model only knew logical block structure (`block_size=32`)
- **After**: Model knows **physical bandwidth requirements** (18 vs 34 vs 210 bytes)

This should enable the ML model to learn:
- Why Q8_0 and Q4_0 need different tiles despite same `block_size`
- Why Q6_K needs very different prefetch strategies (3.3 cache lines/block!)
- Why IQ4_NL and Q4_0 should converge to identical tiles (physically identical)

**Cost**: Just 1 additional CSV column, negligible runtime impact  
**Benefit**: Captures memory bandwidth dimension critical for tile selection

This was an excellent catch during design review! 🎯
