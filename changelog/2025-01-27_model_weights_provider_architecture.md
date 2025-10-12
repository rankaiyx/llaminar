# Model Weights Provider Architecture

**Date**: 2025-01-27  
**Author**: David Sanftenberg  
**Status**: Implemented  
**Related**: Weight snapshot verification system

## Overview

Introduced `QwenModelWeightsProvider` class to establish proper architectural separation between:
- **Weight Loading** (`ModelLoader`) - Loads and dequantizes from GGUF
- **Weight Serving** (`QwenModelWeightsProvider`) - Type-safe access interface
- **Weight Verification** (future `WeightVerifier`) - Parity testing against PyTorch

This eliminates the previous pattern of adding weight getters directly to kernels, which violated separation of concerns.

## Motivation

During implementation of weight verification for parity testing, we discovered an architectural gap:

```cpp
// ❌ WRONG: Adding weight getters to kernels
class MPIAttentionKernel {
    std::shared_ptr<Tensor> get_k_weight() const;  // Leaky abstraction!
};
```

**Problems with kernel-based getters:**
1. **Violates single responsibility** - Kernels consume weights, shouldn't provide them
2. **Duplicates logic** - Each kernel reimplements similar accessors
3. **No MPI awareness** - Caller must manually handle rank slicing
4. **Testing difficulty** - Can't mock weight access without mocking entire kernel

**Solution**: Dedicated provider class that owns and serves weights with MPI awareness built-in.

## Design

### Class Hierarchy

```
QwenPipeline::ModelWeights (struct)
    ↓ owned by
QwenModelWeightsProvider (class)
    ↓ used by
Kernels, Tests, Verifiers
```

### Key Features

1. **Ownership**: Provider takes `unique_ptr` to ModelWeights, manages lifecycle
2. **Type Safety**: Named getters (`getKeyWeight(layer)`) prevent indexing errors
3. **MPI Metadata**: Exposes `isWeightSliced()`, `getLocalSliceInfo()` for verification
4. **Backward Compatibility**: `rawWeights()` accessor for gradual migration
5. **Bounds Checking**: Validates layer indices with clear error messages

### API Example

```cpp
// Initialize provider
auto weights = loadModelWeights_impl_bridge(*loader, config);
auto provider = std::make_unique<QwenModelWeightsProvider>(
    std::make_unique<QwenPipeline::ModelWeights>(std::move(weights)),
    mpi_ctx, config
);

// Type-safe access
auto k_weight = provider->getKeyWeight(0);  // Layer 0 K projection

// MPI metadata for verification
if (provider->isWeightSliced("K")) {
    auto [offset, count] = provider->getLocalSliceInfo("K");
    // Rank 0: offset=0, count=1 (KV head 0)
    // Rank 1: offset=1, count=1 (KV head 1)
}
```

## Implementation Details

### File Structure

- `src/model_weights_provider.h` - Class declaration with full API documentation
- `src/model_weights_provider.cpp` - Implementation with MPI slice calculations
- Added to `CMakeLists.txt` LLAMINAR_CORE_SOURCES

### Weight Categories

**Global Weights (Replicated across ranks):**
- Token embedding: `getTokenEmbedding()`
- Output normalization: `getOutputNorm()`
- LM head: `getLMHead()`

**Per-Layer Attention (Some sliced):**
- Norms: `getAttentionNorm(layer)` - replicated
- Projections: `getQueryWeight(layer)`, `getKeyWeight(layer)`, `getValueWeight(layer)` - **sliced by heads**
- Output: `getOutputWeight(layer)` - replicated
- Biases: `getQueryBias(layer)`, `getKeyBias(layer)`, `getValueBias(layer)` - sliced

**Per-Layer FFN (Some sliced):**
- Norm: `getFFNNorm(layer)` - replicated
- Projections: `getGateWeight(layer)`, `getUpWeight(layer)` - **sliced by hidden dim**
- Down: `getDownWeight(layer)` - **row-sliced**

### MPI Slicing Logic

Provider encapsulates the slicing calculations that were previously scattered throughout the codebase:

```cpp
// K/V weights sliced by KV heads
int local_kv_heads = config_.n_head_kv / mpi_ctx_.size;
int kv_head_offset = mpi_ctx_.rank * local_kv_heads;

// Q weights sliced by Q heads
int local_q_heads = config_.n_head / mpi_ctx_.size;
int head_offset = mpi_ctx_.rank * local_q_heads;

// FFN weights sliced by hidden dimension
int local_d_ff = config_.d_ff / mpi_ctx_.size;
int d_ff_offset = mpi_ctx_.rank * local_d_ff;
```

This metadata is critical for weight verification - tests can now ask "what portion of the full weight is stored locally?" instead of duplicating this logic.

## Benefits

### For Weight Verification

```cpp
// Verification knows exactly what to compare
auto provider = /* ... */;
if (provider->isWeightSliced("K")) {
    auto [offset, count] = provider->getLocalSliceInfo("K");
    
    // Load full PyTorch K weight [num_kv_heads * head_dim, d_model]
    auto pytorch_k = load_pytorch_weight("layer0_K_WEIGHT.npy");
    
    // Extract local slice matching this rank
    auto pytorch_local = extract_rows(pytorch_k, 
                                      offset * head_dim, 
                                      count * head_dim);
    
    // Compare with Llaminar's local weight
    auto llaminar_k = provider->getKeyWeight(0);
    compare_tensors(pytorch_local, llaminar_k);
}
```

### For Testing

```cpp
// Can mock provider without mocking entire pipeline
class MockWeightsProvider : public QwenModelWeightsProvider {
    // Override specific getters for test scenarios
};
```

### For Future Extensions

- Easy to add new weight types (e.g., LoRA adapters)
- Natural place for weight quantization metadata
- Can extend with `getWeightDtype()`, `getWeightQuantization()` methods
- Could support hot-swapping weights for model merging

## Migration Path

### Phase 1: ✅ Provider Infrastructure (Current)
- Implement `QwenModelWeightsProvider` class
- Build and test compilation
- Document API and usage patterns

### Phase 2: Weight Verification (Next)
- Create `WeightVerifier` class using provider
- Implement PyTorch weight comparison
- Add to parity test framework

### Phase 3: Pipeline Integration (Future)
- Modify `QwenPipeline` to use provider
- Update `loadWeights()` to return wrapped provider
- Gradual migration of direct `ModelWeights` access

### Phase 4: Kernel Refactor (Optional)
- Remove any kernel-based weight access patterns
- Kernels only receive weights as execute() inputs
- Provider stays in pipeline layer

## Testing

### Build Verification
```bash
cmake --build build --target llaminar_core --parallel
# ✅ Compiles successfully
```

### Integration Tests (TODO)
- Unit test provider construction and getters
- Test MPI slicing calculations for all weight types
- Verify bounds checking throws on invalid layer index
- Test with different world sizes (1, 2, 4 ranks)

## Related Work

- **Weight Snapshot System** (`2025-01-27_weight_snapshot_system.md`): PyTorch reference generation
- **RoPE Investigation** (`2025-01-27_rope_investigation_summary.md`): Root cause analysis needing verification
- **Weight Role Classification** (`src/weights/weight_roles.{h,cpp}`): Complements provider with role metadata

## Future Enhancements

### Multi-Architecture Support

```cpp
// Abstract base class for different models
class IModelWeightsProvider {
    virtual std::shared_ptr<TensorBase> getKeyWeight(int layer) const = 0;
    // ...
};

class QwenModelWeightsProvider : public IModelWeightsProvider { /* ... */ };
class LlamaModelWeightsProvider : public IModelWeightsProvider { /* ... */ };
```

### Weight Statistics

```cpp
// Add optional weight profiling
struct WeightStats {
    float min, max, mean, stddev;
    size_t nan_count, inf_count;
};

WeightStats getWeightStats(int layer, const std::string& type) const;
```

### Dynamic Weight Updates

```cpp
// Support for LoRA, adapters, quantization
void updateWeight(int layer, const std::string& type, 
                  std::shared_ptr<TensorBase> new_weight);
```

## Conclusion

The `QwenModelWeightsProvider` establishes proper architectural boundaries between weight loading, serving, and verification. This makes the codebase more maintainable, testable, and extensible while specifically enabling the weight verification system needed to debug the K_PROJECTION mismatch in parity testing.

**Status**: ✅ Infrastructure complete, ready for verification implementation  
**Next**: Implement `WeightVerifier` class using this provider
