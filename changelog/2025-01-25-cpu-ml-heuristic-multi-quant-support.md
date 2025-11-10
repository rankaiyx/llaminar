# CPU ML Heuristic: Multi-Quantization Support (Exhaustive Strategy)

**Date**: November 9, 2025  
**Author**: David Sanftenberg  
**Status**: In Progress - Downloading models

## Overview

Enhanced the CPU GEMM ML heuristic framework to capture the impact of quantization formats on optimal tile selection. This critical enhancement was motivated by the observation that **quantization block size significantly affects tile performance**.

**Exhaustive Strategy**: After discovering that mradermacher and bartowski provide single-file quantized models for ALL sizes, we adopted an **exhaustive approach**:
- **ALL model sizes (0.5B-32B)**: Full or near-full quant coverage (4-5 formats each)
- **Learns quant × size interactions**: How block_size impact varies across problem sizes
- **Result**: ~185K data points (was 145K hybrid, was 214K original), **comprehensive and realistic**

## Motivation

**Key Insight**: Different quantization formats have different block sizes, which impact:
1. **Memory access patterns**: 32-byte blocks vs 256-byte blocks have different cache behavior
2. **Alignment requirements**: Tiles that align well with block boundaries perform better
3. **Decode overhead**: Smaller blocks → more frequent scale/offset lookups

**Example**:
- IQ4_NL (block_size=32): May prefer tiles with K dimensions divisible by 32
- Q6_K (block_size=256): May prefer larger tiles that amortize decode overhead
- FP16 (block_size=1, continuous data): Likely prefers larger tiles (no decode overhead)

**Previous limitation**: Manual heuristics and initial ML framework **completely ignored** this dimension, assuming all quantization formats behaved identically.

## Implementation

### 1. Selected Quantization Formats (Exhaustive Strategy)

**Full or near-full quant coverage for ALL sizes**:

| Size | Formats | Files | Source |
|------|---------|-------|--------|
| 0.5B | IQ4_NL, Q4_0, Q8_0, Q6_K, FP16 (5) | Single files | Already have |
| 1.5B | IQ4_NL, Q4_0, Q8_0, Q6_K, FP16 (5) | Single files | mradermacher + Official Qwen |
| 3B | IQ4_NL, Q4_0, Q8_0, Q6_K (4) | Single files | mradermacher + Official Qwen |
| 7B | IQ4_NL, Q4_0, Q8_0, Q6_K (4) | IQ4_NL single, others partial | mradermacher + Official Qwen |
| **14B** | **IQ4_NL, Q4_0, Q8_0, Q6_K, FP16 (5)** | **Single files** | **mradermacher + bartowski** |
| **32B** | **IQ4_NL, Q4_0, Q8_0, Q6_K (4)** | **Single files** | **mradermacher** |

**Quantization format details**:

| Format | Block Size | Bits/Weight | Example Model | Coverage |
|--------|-----------|-------------|---------------|----------|
| IQ4_NL | 32 | 4.5 | `Qwen2.5-14B-Instruct.IQ4_NL.gguf` | **All sizes** (0.5B-32B) |
| Q4_0 | 32 | 4.5 | `Qwen2.5-14B-Instruct.Q4_0.gguf` | **All sizes** (0.5B-32B) |
| Q8_0 | 32 | 8.5 | `Qwen2.5-14B-Instruct.Q8_0.gguf` | **All sizes** (0.5B-32B) |
| Q6_K | 256 | 6.6 | `Qwen2.5-14B-Instruct.Q6_K.gguf` | **All sizes** (0.5B-32B) |
| FP16 | 1 | 16.0 | `Qwen2.5-14B-Instruct-f16.gguf` | 0.5B, 1.5B, 14B |

**Block size coverage**: {1, 32, 256} - Full spectrum at ALL model sizes

**Key advantage**: We can now learn how quant_block_size impact **changes** with problem size:
- Does Q6_K (block=256) benefit more or less from certain tiles as models scale?
- How does decode overhead scale differently for IQ4_NL vs Q4_0?
- Does FP16 (no decode) show different tile preferences at 14B vs 0.5B?

### 2. New Features (3 added, total now 23 base features)

**Updated feature set**:
```python
feature_cols = [
    # Matrix dimensions (4)
    'm', 'n', 'k', 'problem_size',
    
    # Quantization features (3) ← NEW
    'quant_block_size',    # 32, 256, or 1
    'bits_per_weight',     # 4.5, 6.6, 8.5, 16.0
    
    # Variant parameters (6)
    'is_avx512', 'tile_m', 'tile_n', 'tile_area',
    'unroll_k', 'prefetch_dist',
    
    # Derived cache/alignment features (7)
    'l1_fit_ratio', 'l2_fit_ratio',
    'm_alignment', 'n_alignment', 'm_n_ratio',
    'quant_alignment',     # k % quant_block_size ← NEW
    
    # Additional derived features (2)
    'tile_bytes', 'working_set_bytes',
    
    # Log-scale features (3, added by extract_features)
    'log_m', 'log_n', 'log_k',
]
```

**Total features**: 23 base + 3 log-scale = **26 features** (was 20 base + 3 log = 23)

**Note**: `quant_format` (string: "IQ4_NL", "Q6_K", etc.) is **not** used as a feature directly. Instead:
- Categorical encoding would explode feature space
- Block size and bits/weight are the **continuous** numerical properties that matter
- Model learns quant impact via numerical features, not string labels

### 3. Updated Benchmark Script (`benchmark_cpu_gemm.py`)

**Key changes**:

1. **QuantFormat dataclass**:
```python
@dataclass
class QuantFormat:
    name: str                # "IQ4_NL", "Q6_K", etc.
    model_path: str          # Path to .gguf file
    block_size: int          # Logical: 32, 256, 1
    bits_per_weight: float   # Effective: 4.5, 6.6, 16.0
    bytes_per_block: int     # Physical: 18, 34, 210, 2 (NEW!)
```

2. **Triple-nested loop structure**:
```python
for quant_fmt in QUANT_FORMATS:  # 5 formats
    for shape in shapes:          # 35 shapes (≤32B)
        for variant in variants:  # 1225 variants
            # Total: ~185K benchmarks
```

3. **Updated CSV schema** (now 32 columns):
```csv
test_name,m,n,k,problem_size,
quant_format,quant_block_size,bits_per_weight,bytes_per_block,quant_alignment,  ← NEW (5 columns)
isa,tile_m,tile_n,unroll_k,prefetch_dist,variant_name,
is_avx512,tile_area,tile_bytes,l1_fit_ratio,
working_set_bytes,l2_fit_ratio,
m_alignment,n_alignment,m_n_ratio,
log_m,log_n,log_k,
gflops,time_ms,success
```

4. **Derived features**:
```python
quant_alignment = shape.k % quant_fmt.block_size  # K % block_size alignment
bytes_per_block = quant_fmt.bytes_per_block       # Physical packed size (CRITICAL!)
```

**Why `bytes_per_block` is critical**:

| Format | Block Size | Bytes/Block | Cache Lines/Block | Bandwidth vs IQ4_NL |
|--------|-----------|-------------|-------------------|---------------------|
| IQ4_NL | 32 | **18** | 0.28 | **1.0× (baseline)** |
| Q4_0 | 32 | **18** | 0.28 | **1.0× (identical!)** |
| Q8_0 | 32 | **34** | 0.53 | **1.9× higher** |
| Q6_K | 256 | **210** | 3.28 | **11.7× higher** |

**Key insights**:
- **Q8_0 vs Q4_0**: Same block size (32), but **nearly 2× bandwidth requirement!**
- **Q6_K**: Requires **3.3 cache lines per block** - prefetch effectiveness critical
- **IQ4_NL/Q4_0**: Identical physical layout - should prefer same tiles

This is **independent** of `quant_block_size` and affects:
- Memory bandwidth (bytes loaded per K-element)
- Cache line efficiency
- Prefetch effectiveness
- SIMD register packing

### 4. Updated Training Script (`train_cpu_gemm_heuristic.py`)

**Key changes**:

1. **Updated feature_cols** (24 base features, was 20):
   - Added: `quant_block_size`, `bits_per_weight`, **`bytes_per_block`**, `quant_alignment`
   
2. **Model input size auto-adapts**:
```python
input_size = X_train.shape[1]  # Automatically becomes 27 (24 base + 3 log)
model = CPUGemmPredictor(input_size=input_size, hidden_sizes=[128, 64, 32])
```

3. **No categorical encoding needed**: Numerical features only

## Impact on Benchmarking

### Data Volume (Revised Hybrid Strategy)

**Previous (original multi-quant plan)**:
- 35 shapes × 1225 variants × 5 quant formats = **214,375 benchmarks**
- Runtime: ~60-90 hours

**Current (hybrid strategy)**:
```
Small models (full quant coverage):
  - 0.5B:  6 tests × 1225 variants × 5 quants = 36,750
  - 1.5B:  6 tests × 1225 variants × 5 quants = 36,750
  - 3B:    5 tests × 1225 variants × 4 quants = 24,500
  - 7B:    6 tests × 1225 variants × 4 quants = 29,400

Large models (single quant):
  - 14B:   5 tests × 1225 variants × 1 quant = 6,125
  - 32B:   4 tests × 1225 variants × 1 quant = 4,900
  - Edge:  6 tests × 1225 variants × 1 quant = 7,350

Total: ~145,775 benchmarks (was 214K)
Runtime: ~40-60 hours (was 60-90 hours)
```

**Trade-off analysis**:
- ✅ **Captures critical dimension**: Quant block size relationship learned from small models
- ✅ **Comprehensive block coverage**: All block sizes {1, 32, 256} via 0.5B-7B
- ✅ **Size diversity**: Large models (14B, 32B) provide problem_size scaling data
- ✅ **Production relevance**: Tests actual available model formats
- ✅ **Faster execution**: 40-60 hours vs 60-90 hours (30% reduction)
- ✅ **More realistic**: Uses actual model weights, not synthetic data
- ✅ **Model availability**: All models downloadable from HuggingFace
- ⚙️ **Mitigation**: Unattended execution, incremental CSV saving, can pause/resume

### Expected Findings

**Hypothesis**: The ML model will discover:
1. **Block-aligned tiles perform better**:
   - IQ4_NL/Q4_0/Q8_0 (block=32): Prefer K dimensions divisible by 32
   - Q6_K (block=256): Prefer K dimensions divisible by 256
   
2. **Decode overhead matters**:
   - Small blocks (32) → more decode calls → prefer larger tiles to amortize
   - Large blocks (256) → fewer decode calls → can use smaller tiles
   
3. **Unquantized is different**:
   - FP16 (block=1) → continuous data, no decode → likely prefers different tiles

4. **Precision impacts cache**:
   - Higher bits/weight → larger memory footprint → different cache ratios
   - Q8_0 (8.5 bpw) vs Q4_0 (4.5 bpw) may prefer different tile sizes

**Validation**: Compare top-5 accuracy **across** quant formats vs **within** quant formats.

## File Changes

### Modified Files

1. **`src/v2/kernels/cpu/python/benchmark_cpu_gemm.py`**:
   - Added `QuantFormat` dataclass
   - Added `QUANT_FORMATS` list (5 formats)
   - Updated CSV schema: +4 columns (quant_format, quant_block_size, bits_per_weight, quant_alignment)
   - Changed loop structure: quant_formats → shapes → variants
   - Updated total count: 35 × 1225 × 5 = 214,375
   - Fixed main() argparse: removed `--model` (now iterates via QUANT_FORMATS)
   - Validate all model files exist at startup

2. **`src/v2/kernels/cpu/python/train_cpu_gemm_heuristic.py`**:
   - Updated `feature_cols` in main(): 23 base features (was 20)
   - Added: `quant_block_size`, `bits_per_weight`, `quant_alignment`
   - Model auto-adapts to 26 input features (23 base + 3 log)

### Unchanged (Ready to Use)

- `src/v2/kernels/cpu/python/validate_cpu_heuristic.py` ✓ Ready
- `src/v2/kernels/cpu/python/export_cpu_heuristic.py` ✓ Ready
- `src/v2/kernels/cpu/python/build_cpu_ml_heuristic.sh` ✓ Ready
- `tests/v2/performance/Perf__CpuGemmHeuristicValidation.cpp` ✓ Compiled

## Testing Strategy

### 1. Quick Smoke Test (Before Full Run)

Test single shape/variant/quant combination:

```bash
# Build test binary
cmake --build build_v2_release --target v2_benchmark_cpu_gemm_variant --parallel

# Test single benchmark (override shapes to minimal set)
python3 src/v2/kernels/cpu/python/benchmark_cpu_gemm.py \
  --shapes "1x896x896" \
  --output cpu_gemm_smoke_test.csv

# Verify CSV output
head -n 20 cpu_gemm_smoke_test.csv
wc -l cpu_gemm_smoke_test.csv  # Should be ~6K lines (1 shape × 1225 variants × 5 quants)
```

**Expected output**:
- CSV header with 30 columns (including new quant_* columns)
- ~6,125 rows (header + 1 × 1225 × 5)
- quant_format values: IQ4_NL, Q4_0, Q8_0, Q6_K, FP16
- quant_block_size values: 32, 32, 32, 256, 1
- bits_per_weight values: 4.5, 4.5, 8.5, 6.6, 16.0

### 2. Full Benchmark Run

```bash
# Unattended overnight run
nohup python3 src/v2/kernels/cpu/python/benchmark_cpu_gemm.py \
  --output cpu_gemm_benchmark_data.csv \
  > benchmark.log 2>&1 &

# Monitor progress
tail -f benchmark.log
```

**Expected runtime**: ~60-90 hours (unattended)

### 3. Training

```bash
# Train model (after benchmark completes)
python3 src/v2/kernels/cpu/python/train_cpu_gemm_heuristic.py \
  --data cpu_gemm_benchmark_data.csv \
  --output cpu_gemm_heuristic.onnx \
  --epochs 200

# Expected:
# - Model accepts 26 features (23 base + 3 log)
# - Training on ~171K points (214K × 80%)
# - Validation on ~43K points (214K × 20%)
```

### 4. Validation

```bash
# Validate generalization
python3 src/v2/kernels/cpu/python/validate_cpu_heuristic.py \
  --model cpu_gemm_heuristic.onnx \
  --data cpu_gemm_benchmark_data.csv

# Check metrics:
# - Top-5 accuracy (target: >85%)
# - Per-quant breakdown (does model generalize across formats?)
# - Generalization gap (seen vs unseen sizes)
```

## Expected Benefits

1. **Quantization-aware tile selection**:
   - Model learns block_size → optimal tile relationship
   - Better performance across all quant formats
   
2. **Reduced manual tuning**:
   - No need for separate heuristics per quant format
   - ML model automatically adapts
   
3. **Production relevance**:
   - Tests actual formats users deploy (IQ4_NL, Q6_K, FP16, etc.)
   - Validates performance across precision spectrum

4. **Scientific rigor**:
   - Systematic exploration of quant dimension
   - Quantifies impact (e.g., "Q6_K prefers 20% larger tiles than IQ4_NL")

## Risks and Mitigations

**Risk 1**: 5× longer benchmark time  
**Mitigation**: Unattended execution, incremental CSV saving, can pause/resume

**Risk 2**: Model may not generalize across quant formats  
**Mitigation**: Validation script includes per-quant accuracy breakdown

**Risk 3**: Feature engineering may be insufficient  
**Mitigation**: 
- Smoke test first (single shape)
- If accuracy poor, can add interaction terms (e.g., `tile_m × quant_block_size`)

**Risk 4**: Some model files may be missing  
**Mitigation**: Script validates all 5 models exist before starting

## Next Steps

1. ✅ **Complete** - Multi-quant benchmark script
2. ✅ **Complete** - Updated training script (26 features)
3. ✅ **Complete** - Identified model availability issue
4. ✅ **Complete** - Created hybrid strategy (full quant for small, single quant for large)
5. ✅ **Complete** - Created HuggingFace download script
6. ⚙️ **In Progress** - Downloading models from HuggingFace (~20-30 GB)
7. ⚙️ **TODO** - Add 1.5B, 3B test cases to BENCHMARK_SHAPES
8. ⚙️ **TODO** - Update QUANT_FORMATS with actual downloaded filenames
9. ⚙️ **TODO** - Run smoke test (single shape)
10. ⚙️ **TODO** - Run full benchmark (~40-60 hours)
11. ⚙️ **TODO** - Collect perf profiling for subset (Qwen 0.5B/1.5B/3B)
12. ⚙️ **TODO** - Train neural network
13. ⚙️ **TODO** - Validate generalization (per-quant + per-size breakdown)
14. ⚙️ **TODO** - Export C++ header
15. ⚙️ **TODO** - Integrate into SmartGemmSearch.cpp
16. ⚙️ **TODO** - Validate performance recovery (L1 miss <5%, GFLOPS >1000)

## Conclusion

The **hybrid multi-quantization strategy** captures a **critical dimension** of the CPU GEMM performance landscape that was completely missing from manual heuristics. By adopting a pragmatic approach:

**Full quant coverage (small models)**:
- 0.5B, 1.5B, 3B, 7B: 5 quant formats each
- Learns: How block_size ∈ {1, 32, 256} affects optimal tile selection
- Learns: How decode overhead trades off with tile size
- Learns: How precision (bits/weight) impacts cache behavior

**Single quant coverage (large models)**:
- 14B, 32B: IQ4_NL only
- Learns: How problem_size scaling affects tile selection
- Provides: Diverse m/n/k combinations for interpolation

**Result**: The ML model learns **both** dimensions (quant impact + size scaling) with realistic data in reasonable time.

This revised 145K-point dataset (was 214K) is:
- **More practical**: Uses downloadable models from HuggingFace
- **More realistic**: Real weight tensors, not synthetic data
- **Faster**: 40-60 hours vs 60-90 hours (30% reduction)
- **Comprehensive**: Full block_size spectrum + full problem_size range

**Status**: Downloading models from HuggingFace. Framework complete, ready for data collection after downloads finish.

---

## Model Download Status

**Download script**: `./download_qwen_models.sh`  
**Log file**: `download_models.log`  
**Progress**: Check with `tail -f download_models.log`

**Downloaded models**:
- mradermacher: IQ4_NL for all sizes (1.5B, 3B, 7B, 14B, 32B)
- Official Qwen: Q4_0, Q8_0, Q6_K, FP16 for 1.5B, 3B, 7B

**Total download size**: ~20-30 GB (varies by quant format)
