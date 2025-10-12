# Weight Contract Validation Logging Enhancement
**Date**: January 27, 2025  
**Feature**: Enhanced weight validation logging during model loading

## Summary

Added detailed logging to the weight contract validation system to provide visibility into which weights are being validated during model loading. This addresses the token divergence investigation by ensuring all weights are validated and logged.

## Changes Made

### 1. Enhanced `WeightShapeContract::validate_with_mpi()` in `src/weight_contracts.h`

**Added detailed per-weight validation logging:**
```cpp
// Log successful validation with details
std::stringstream log_msg;
log_msg << "[WeightContract] ✓ " << weight_name;
if (layer_index >= 0)
{
    log_msg << " (layer " << layer_index << ")";
}
log_msg << ": shape=[";
for (size_t i = 0; i < actual.size(); ++i)
{
    if (i > 0) log_msg << ", ";
    log_msg << actual[i];
}
log_msg << "]";
if (mpi_size > 1)
{
    log_msg << " (rank " << mpi_rank << "/" << mpi_size;
    if (slice_type != WeightSliceType::REPLICATED)
    {
        log_msg << ", sliced ";
        if (slice_type == WeightSliceType::ROW_SLICED)
        {
            log_msg << "rows";
        }
        else if (slice_type == WeightSliceType::COL_SLICED)
        {
            log_msg << "cols";
        }
    }
    log_msg << ")";
}
LOG_DEBUG(log_msg.str());
```

**Output example** (single rank):
```
[WeightContract] ✓ token_embedding: shape=[151669, 896]
[WeightContract] ✓ output_norm: shape=[896]
[WeightContract] ✓ lm_head: shape=[151669, 896]
```

**Output example** (MPI distributed, 2 ranks):
```
[WeightContract] ✓ token_embedding: shape=[151669, 896] (rank 0/2)
[WeightContract] ✓ attn_q.weight (layer 0): shape=[448, 896] (rank 0/2, sliced rows)
[WeightContract] ✓ attn_q.weight (layer 0): shape=[448, 896] (rank 1/2, sliced rows)
```

### 2. Enhanced `ModelWeightContracts::validate_global_with_mpi()` 

**Added summary logging:**
```cpp
LOG_INFO("[WeightContract] Validating " << global_weights.size() << " global weights (rank " 
         << mpi_rank << "/" << mpi_size << ")");

// ... validation loop ...

LOG_INFO("[WeightContract] ✓ Global weights validated: " << validated_count << "/" << global_weights.size());
```

**Output example:**
```
[INFO] [WeightContract] Validating 3 global weights (rank 0/2)
[DEBUG] [WeightContract] ✓ token_embedding: shape=[151669, 896] (rank 0/2)
[DEBUG] [WeightContract] ✓ output_norm: shape=[896] (rank 0/2)
[DEBUG] [WeightContract] ✓ lm_head: shape=[151669, 896] (rank 0/2)
[INFO] [WeightContract] ✓ Global weights validated: 3/3
```

### 3. Enhanced `QwenModelWeights::validate_with_mpi()` in `src/qwen_pipeline_adapter.h`

**Added layer validation progress logging:**
```cpp
int n_layers = layer_count();
LOG_INFO("[WeightContract] Validating " << n_layers << " layers (rank " 
         << mpi_rank << "/" << mpi_size << ")");

for (int layer = 0; layer < n_layers; ++layer)
{
    contracts.validate_layer_with_mpi(layer, ...);
    
    // Log progress every 5 layers
    if ((layer + 1) % 5 == 0 || layer == n_layers - 1)
    {
        LOG_INFO("[WeightContract] ✓ Validated layers 0-" << layer << " (" << (layer + 1) 
                 << "/" << n_layers << ")");
    }
}
```

**Output example:**
```
[INFO] [WeightContract] Validating 24 layers (rank 0/2)
[INFO] [WeightContract] ✓ Validated layers 0-4 (5/24)
[INFO] [WeightContract] ✓ Validated layers 0-9 (10/24)
[INFO] [WeightContract] ✓ Validated layers 0-14 (15/24)
[INFO] [WeightContract] ✓ Validated layers 0-19 (20/24)
[INFO] [WeightContract] ✓ Validated layers 0-23 (24/24)
```

### 4. Added missing include in `src/weight_contracts.h`

Added `#include "logger.h"` to enable LOG_INFO and LOG_DEBUG macros in the header file.

## Benefits

### 1. Visibility into Weight Loading
- Every weight tensor is now logged with its actual dimensions
- Easy to verify that weights match expected shapes
- MPI slicing information is clearly shown

### 2. Debugging Support
- Can quickly identify if a weight has wrong dimensions
- Shows exactly which rank has which weight slice
- Helps diagnose weight distribution issues in MPI mode

### 3. Embedding Divergence Investigation
For the current token divergence issue, we can now:
- **Verify embedding table dimensions** on each rank
- **Check if embedding weights are being sliced** (they shouldn't be - should be replicated)
- **Compare weight shapes** between ranks to ensure consistency
- **Confirm all weights passed validation** before execution

## Verification Log Level

The detailed per-weight logs use `LOG_DEBUG` level to avoid cluttering normal output. 
Summary logs use `LOG_INFO` level for production visibility.

**To see detailed weight validation:**
```bash
# Set log level to DEBUG
export LLAMINAR_LOG_LEVEL=DEBUG
./build/llaminar --model models/model.gguf --verbose
```

**To see only summaries:**
```bash
# Default INFO level
./build/llaminar --model models/model.gguf
```

## Example Full Output

When loading a Qwen 0.5B model with 24 layers on 2 MPI ranks:

```
[INFO] [WeightContract] Validating 3 global weights (rank 0/2)
[DEBUG] [WeightContract] ✓ token_embedding: shape=[151669, 896] (rank 0/2)
[DEBUG] [WeightContract] ✓ output_norm: shape=[896] (rank 0/2)
[DEBUG] [WeightContract] ✓ lm_head: shape=[151669, 896] (rank 0/2)
[INFO] [WeightContract] ✓ Global weights validated: 3/3

[INFO] [WeightContract] Validating 24 layers (rank 0/2)
[DEBUG] [WeightContract] ✓ attn_norm.weight (layer 0): shape=[896] (rank 0/2)
[DEBUG] [WeightContract] ✓ attn_q.weight (layer 0): shape=[448, 896] (rank 0/2, sliced rows)
[DEBUG] [WeightContract] ✓ attn_k.weight (layer 0): shape=[64, 896] (rank 0/2, sliced rows)
[DEBUG] [WeightContract] ✓ attn_v.weight (layer 0): shape=[64, 896] (rank 0/2, sliced rows)
[DEBUG] [WeightContract] ✓ attn_output.weight (layer 0): shape=[896, 448] (rank 0/2, sliced cols)
... (more weights for layer 0)
[INFO] [WeightContract] ✓ Validated layers 0-4 (5/24)
... (continue for all 24 layers)
[INFO] [WeightContract] ✓ Validated layers 0-23 (24/24)

[INFO] ✓ All weights validated with MPI slicing (rank 0/2)
```

## Next Steps for Embedding Investigation

With this logging in place, we can now:

1. **Run the parity test** and capture weight validation logs
2. **Verify embedding table** is `[151669, 896]` on both ranks
3. **Confirm it's marked as replicated** (not sliced)
4. **Check if both ranks log identical shapes** for all weights

If embedding weights validate correctly but embeddings still diverge, the issue is likely in:
- Token ID lookup logic in `MPIEmbeddingKernel`
- Embedding table memory corruption after loading
- Gather/broadcast operations in distributed mode

## Files Modified

- `src/weight_contracts.h`: Enhanced validation logging, added logger.h include
- `src/qwen_pipeline_adapter.h`: Added layer validation progress logging

## Build Status

✅ **Build successful** - all tests compile and link correctly

The logging overhead is minimal as it only runs once during model loading.
