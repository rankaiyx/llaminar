# Weight Snapshot System - 2025-01-27

## Overview

Added comprehensive weight snapshot capture to PyTorch reference generation to eliminate uncertainty about weight loading, quantization, and MPI sharding differences between PyTorch and Llaminar.

## Motivation

During debugging of token divergence issues, we repeatedly encountered the question: **"Are the weights the same?"**

Without captured weight snapshots, we couldn't definitively answer:
- Are weights loaded correctly from GGUF?
- Does MPI sharding match PyTorch's distribution?
- Are there quantization/dequantization errors?
- Do orientation/transpose issues affect weight matrices?

**Solution**: Capture PyTorch's weights once, compare element-by-element with Llaminar's loaded weights before every parity test run.

## Implementation

### PyTorch Side (generate_incremental_decode_snapshots.py)

Added `save_model_weights()` function that:
1. Loads the HuggingFace model from GGUF
2. Extracts Q, K, V, O projection weights from all 24 layers
3. Saves as NumPy `.npy` files in `pytorch_snapshots_mapped/weights/`

**Weight Files Created** (96 total files):
```
pytorch_snapshots_mapped/
  weights/
    layer0_Q_WEIGHT.npy  # [896, 896] - FP32
    layer0_K_WEIGHT.npy  # [128, 896] - FP32
    layer0_V_WEIGHT.npy  # [128, 896] - FP32
    layer0_O_WEIGHT.npy  # [896, 896] - FP32
    layer1_Q_WEIGHT.npy
    ...
    layer23_O_WEIGHT.npy
```

**Weight Shapes** (for Qwen2-0.5B):
- Q projection: `[num_heads * head_dim, d_model]` = `[896, 896]`
- K projection: `[num_kv_heads * head_dim, d_model]` = `[128, 896]`
- V projection: `[num_kv_heads * head_dim, d_model]` = `[128, 896]`
- O projection: `[d_model, num_heads * head_dim]` = `[896, 896]`

**PyTorch Weight Layout**: `[out_features, in_features]`  
**Usage**: `output = input @ weight.T` (transpose required)

### Llaminar Side (test_parity_framework.cpp - TODO)

Will add weight verification function:
```cpp
bool verify_weights(
    const std::string& pytorch_weights_dir,
    const std::shared_ptr<MPIAttentionKernel>& attn_kernel,
    int layer_idx,
    bool verbose = false
) {
    // 1. Load PyTorch weight from .npy file
    auto pytorch_k_weight = load_numpy_array(
        pytorch_weights_dir + "/layer" + std::to_string(layer_idx) + "_K_WEIGHT.npy"
    );
    
    // 2. Get Llaminar's loaded weight (may be MPI-sharded)
    auto llaminar_k_weight = attn_kernel->get_k_weight();  // Need to expose this
    
    // 3. Account for MPI sharding
    // PyTorch: Full [128, 896] weight
    // Llaminar rank 0: Slice [64, 896] (first KV head)
    // Llaminar rank 1: Slice [64, 896] (second KV head)
    
    // 4. Compare element-by-element
    float max_diff = 0.0f;
    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < llaminar_k_weight->size(); ++i) {
        float pytorch_val = pytorch_k_weight[rank_offset + i];
        float llaminar_val = llaminar_k_weight->data()[i];
        float diff = std::abs(pytorch_val - llaminar_val);
        max_diff = std::max(max_diff, diff);
        sum_sq_diff += diff * diff;
    }
    
    float rel_l2 = std::sqrt(sum_sq_diff / llaminar_k_weight->size()) / 
                   pytorch_k_weight_norm;
    
    // 5. Report mismatch if significant
    if (max_diff > 1e-6f || rel_l2 > 1e-6f) {
        LOG_ERROR("Layer " << layer_idx << " K weight mismatch: "
                  << "max_diff=" << max_diff << " rel_l2=" << rel_l2);
        return false;
    }
    
    return true;
}
```

## Usage

### Generating Weight Snapshots

```bash
# Generate PyTorch snapshots WITH weight snapshots
python3 python/reference/generate_incremental_decode_snapshots.py \
  -m models/Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf \
  --prefill-tokens "1,2,3,4,5" \
  --num-decode-tokens 3 \
  -o pytorch_snapshots_mapped \
  -v

# Verify weights were saved
ls -lh pytorch_snapshots_mapped/weights/

# Check layer 0 K weight
python3 -c "
import numpy as np
k = np.load('pytorch_snapshots_mapped/weights/layer0_K_WEIGHT.npy')
print(f'Shape: {k.shape}')
print(f'First row: {k[0,:10]}')
print(f'Stats: min={k.min():.6f} max={k.max():.6f} mean={k.mean():.6f}')
"
```

### Parity Test with Weight Verification (TODO)

```bash
# Run parity test with automatic weight verification
mpirun -np 2 ./build/test_parity_framework \
  --gtest_filter="ParityFramework.TrueIncrementalDecodeVsPyTorch"

# Output will include:
# [WEIGHT_VERIFY] Layer 0 Q: ✓ Match (max_diff=0.000000, rel_l2=0.000000)
# [WEIGHT_VERIFY] Layer 0 K: ✓ Match (max_diff=0.000000, rel_l2=0.000000)
# [WEIGHT_VERIFY] Layer 0 V: ✓ Match (max_diff=0.000000, rel_l2=0.000000)
# [WEIGHT_VERIFY] Layer 0 O: ✓ Match (max_diff=0.000000, rel_l2=0.000000)
# ...
# [WEIGHT_VERIFY] All 96 weights verified ✓
```

## Debugging Weight Mismatches

If weight verification fails:

### 1. Check Weight File Integrity
```bash
python3 -c "
import numpy as np
import sys

try:
    k = np.load('pytorch_snapshots_mapped/weights/layer0_K_WEIGHT.npy')
    print(f'✓ Layer 0 K weight loaded: shape={k.shape} dtype={k.dtype}')
except Exception as e:
    print(f'✗ Failed to load: {e}')
    sys.exit(1)
"
```

### 2. Compare Llaminar's Loaded Weights
```cpp
// Add temporary debug output in MPIAttentionKernel.cpp
if (layer_index_ == 0 && rank == 0) {
    LOG_INFO("[DEBUG] local_wk shape: [" << local_wk->shape()[0] << ", " 
             << local_wk->shape()[1] << "]");
    LOG_INFO("[DEBUG] local_wk[0,:10]: ");
    for (int i = 0; i < 10; ++i) {
        std::cerr << local_wk->data()[i] << " ";
    }
    std::cerr << std::endl;
}
```

### 3. Check for Transpose Issues
PyTorch Linear layer weights are `[out_features, in_features]`.  
Llaminar may store as `[in_features, out_features]` (row-major).  
**Solution**: Apply transpose when comparing or adjust indexing.

### 4. Check for MPI Sharding
```cpp
// Verify MPI sharding matches PyTorch layout
int rank = getRank();
int world_size = getSize();

// Expected: Rank 0 gets first half of KV heads, rank 1 gets second half
int expected_kv_heads_per_rank = total_kv_heads / world_size;
int expected_k_weight_rows = expected_kv_heads_per_rank * head_dim;

ASSERT_EQ(local_wk->shape()[0], expected_k_weight_rows);
```

### 5. Check for Quantization
Model is FP32, so no quantization errors expected.  
If using quantized model (.gguf with Q4_0/Q6_K):
- Check dequant output matches PyTorch's loaded weights
- Use `LLAMINAR_DEQUANT_STATS=1` environment variable
- Compare dequant block-by-block

## Expected Output (Sample)

```
Saving model weights from models/Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf
  Extracting weights from 24 layers...
    Layer 0 weight shapes:
      Q: (896, 896)
      K: (128, 896)
      V: (128, 896)
      O: (896, 896)
      Q[0,:10]: [-0.00227356 -0.00500488  0.01879883  0.01245117  0.00408936 -0.00915527
  0.00619507 -0.00179291  0.01452637 -0.00592041]
      K[0,:10]: [ 0.0300293  -0.10009766 -0.00177765 -0.00842285  0.02160645 -0.10449219
 -0.01324463  0.00860596 -0.07226562 -0.03320312]
  ✓ Saved weights for 24 layers to pytorch_snapshots_mapped/weights/
```

## Benefits

✅ **Eliminates Weight Loading Uncertainty**: Compare weights once, verify every test run  
✅ **Catches MPI Sharding Issues**: Verify each rank gets correct weight slice  
✅ **Detects Transpose Bugs**: Spot orientation mismatches immediately  
✅ **No Performance Impact**: Weights verified once before inference, not during  
✅ **Quantization Debugging**: Compare dequant output with FP32 reference  
✅ **Regression Prevention**: Weight changes cause immediate test failure  

## Next Steps

1. ✅ **PyTorch weight snapshot generation** - COMPLETE
2. ✅ **Regenerate official snapshots** - COMPLETE  
3. ⏳ **Add weight verification to test_parity_framework.cpp** - IN PROGRESS
4. ⏳ **Expose weight getters in MPIAttentionKernel**
5. ⏳ **Add numpy .npy loader to C++ test framework**
6. ⏳ **Integrate weight verification into parity test workflow**

## Files Modified

- `python/reference/generate_incremental_decode_snapshots.py`: Added `save_model_weights()`
- `pytorch_snapshots_mapped/weights/`: 96 new weight files (24 layers × 4 weights)
- `changelog/2025-01-27_weight_snapshot_system.md`: This document

## Related Issues

- **Token Divergence Root Cause**: K_PROJECTION mismatch suspected (now verifiable with weights)
- **RoPE Investigation**: RoPE verified correct, issue is upstream in projections
- **FP32 vs Quantization**: Confirmed using FP32 model (no quant errors)
