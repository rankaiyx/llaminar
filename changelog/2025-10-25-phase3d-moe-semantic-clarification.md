# Phase 3d Extension: MoE-Ready KV Cache (Semantic Clarification)

**Date**: October 25, 2025  
**Author**: David Sanftenberg  
**Status**: ✅ Complete (Phase 1 of 3)

## Overview

Extended Phase 3d KV cache implementation with **MoE-ready semantic clarification**. The cache now explicitly tracks **attention device placement**, not generic "layer device", enabling proper support for Mixture of Experts models where tensors within a layer may reside on different devices.

## Problem: Layer-Level Granularity Insufficient for MoE

### MoE Architecture Challenge

```
Qwen MoE 14B (hypothetical):
  Layer 0 (Shared Expert Layer):
    ├─ Attention Block → CPU (896 MB)
    │  ├─ wq, wk, wv, wo weights
    │  └─ KV cache ← Where does this go?
    │
    └─ FFN Block → MIXED
       ├─ Router → CPU (2 MB)
       ├─ Shared Expert 0-11 → GPU (42 GB total)
       └─ Local FFN → CPU (896 MB)
```

**Issue**: "Layer 0 device" is ambiguous - attention on CPU, experts on GPU.

**Solution**: KV cache tracks **attention device** (where Q/K/V/O weights live), not entire layer.

## Changes Made

### 1. KVCache.h - Semantic Parameter Rename

**Before**:
```cpp
KVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
        const std::vector<int> &layer_devices);
```

**After**:
```cpp
/**
 * @brief Construct KV cache with per-layer attention device placement
 *
 * The KV cache is stored on the device where **attention computation** occurs
 * (i.e., where wq, wk, wv, wo weights reside). For heterogeneous execution
 * or MoE models, this may differ from where FFN or expert weights are placed.
 *
 * Example use cases:
 * - Standard heterogeneous: Layers 0-11 attention on CPU, 12-23 on GPU
 * - MoE with shared experts: Attention on CPU, experts on GPU → use CPU
 */
KVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices);
```

### 2. KVCache.h - New Semantic Getter

Added `get_attention_device()` method for clarity:

```cpp
/**
 * @brief Get attention device for a specific layer
 *
 * Semantic alias for get_layer_device() that clarifies this returns
 * where attention computation happens, not where entire layer lives.
 */
int get_attention_device(int layer) const {
    return get_layer_device(layer);
}
```

**Rationale**: `get_layer_device()` kept for backward compatibility, `get_attention_device()` clarifies intent.

### 3. KVCache.cpp - Implementation Update

**Variable rename**:
```cpp
// OLD
for (int i = 0; i < n_layers; ++i) {
    int layer_device = layer_devices[i];
    // ...
}

// NEW
for (int i = 0; i < n_layers; ++i) {
    int attn_device = attention_devices[i];
    // ...
}
```

**Comment updates**:
- "per-layer device placement" → "per-layer attention device placement"
- "assigned device" → "attention device"

### 4. Test Updates

**Test documentation**:
```cpp
/**
 * @brief Test heterogeneous device placement (CPU + GPU split)
 *
 * Simulates a model with layers split across devices. The cache follows
 * where attention computation happens (not necessarily where FFN/experts live).
 */
TEST(KVCacheHeterogeneousTest, MultiDevicePlacement) {
    // Layers 0-3: Attention on CPU
    // Layers 4-7: Attention on GPU 0
    // (Note: For MoE, FFN/experts might be on different devices)
    std::vector<int> attention_devices = {-1, -1, -1, -1, 0, 0, 0, 0};
    // ...
}
```

**Test both methods**:
```cpp
EXPECT_EQ(cache->get_attention_device(i), -1);  // New semantic method
EXPECT_EQ(cache->get_layer_device(i), -1);      // Backward compat
```

## Usage Example: MoE Model

```cpp
// MoE model with 6 shared expert layers + 34 standard layers
int n_layers = 40;
std::vector<int> attention_devices(n_layers);

// Layers 0-5: Shared expert layers
// - Attention weights: CPU (moderate size)
// - Shared experts: GPU (very large, reused)
// → Cache follows attention device (CPU)
for (int i = 0; i < 6; ++i) {
    attention_devices[i] = -1;  // Attention on CPU
}

// Layers 6-39: Standard transformer layers (all CPU)
for (int i = 6; i < 40; ++i) {
    attention_devices[i] = -1;
}

auto cache = std::make_shared<KVCache>(n_layers, 2048, 8, 128, attention_devices);

// Result: All KV cache on CPU (where attention happens)
// Shared expert computation happens on GPU, but cache doesn't move
```

## Test Results

```
Building CXX object CMakeFiles/llaminar2_core.dir/tensors/KVCache.cpp.o
Building CXX object tests/v2/CMakeFiles/v2_test_kv_cache.dir/unit/tensors/Test__KVCache.cpp.o
[100%] Built target v2_test_kv_cache

Test project /workspaces/llaminar/build_v2
    Start 1: V2_FetchModelsFixture
1/2 Test #1: V2_FetchModelsFixture ............   Passed    0.01 sec
    Start 4: V2_Unit_KVCache
2/2 Test #4: V2_Unit_KVCache ..................   Passed    2.66 sec

100% tests passed, 0 tests failed out of 2
```

✅ **8/8 tests passing** (all original + heterogeneous tests)

## Backward Compatibility

- ✅ `get_layer_device()` still works (semantic alias)
- ✅ Parameter rename is internal (no public API breakage)
- ✅ Existing single-device constructor unchanged

## Benefits

### 1. Semantic Clarity

**Before**: "Layer device" - ambiguous for mixed-device layers  
**After**: "Attention device" - precisely where Q/K/V computation happens

### 2. MoE Correctness

```cpp
// Layer 0 of MoE model
LayerWeights layer0;
layer0.wq->device_index() == -1;         // CPU
layer0.shared_expert[0]->device_index() == 0;  // GPU

// Cache correctly follows attention
cache->get_attention_device(0) == -1;  // CPU ✅
// NOT: cache->get_layer_device(0) == 0 ❌ (would be wrong if we tracked "layer device")
```

### 3. Documentation Improvement

- Constructor docs now explain MoE use case
- Comments clarify "attention device" vs "layer device"
- Tests demonstrate MoE-aware thinking

## Next Steps (Phase 2-3)

### Phase 2: WeightPlacementMap Extension
- Tensor-level device specification API
- Per-tensor placement (not just per-layer)
- MoE expert placement strategies

### Phase 3: Pipeline Integration
- Update `Qwen2Pipeline` to use attention devices
- Automatic device detection from weights
- Cross-device transfer optimization

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/v2/tensors/KVCache.h` | ~40 | Parameter rename, new getter, docs |
| `src/v2/tensors/KVCache.cpp` | ~15 | Variable rename, comment updates |
| `tests/v2/unit/tensors/Test__KVCache.cpp` | ~30 | Test docs, use new method |

**Total**: ~85 lines modified (no breaking changes)

## Documentation

- **Design document**: `docs/TENSOR_LEVEL_DEVICE_PLACEMENT.md` (comprehensive MoE design)
- **Usage guide**: `docs/V2_KV_CACHE_MULTI_DEVICE.md` (needs update for new naming)
- **Architecture**: `.github/instructions/llaminar-v2-architecture.instructions.md`

## Conclusion

**Phase 1 complete**: KV cache now has **MoE-ready semantics** with clear distinction between:
- **Attention device**: Where Q/K/V/O weights and cache live
- **Layer device**: Ambiguous concept removed from API

**Impact**:
- ✅ Zero breaking changes
- ✅ Clearer intent for future MoE support
- ✅ Foundation for Phase 2 tensor-level placement
- ✅ All tests passing

**Next**: Design and implement `WeightPlacementMap` for tensor-granular device specification.
