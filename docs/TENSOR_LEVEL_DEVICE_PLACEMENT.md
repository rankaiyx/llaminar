# V2 Tensor-Level Device Placement for MoE Models

**Author**: David Sanftenberg  
**Date**: October 25, 2025  
**Status**: 🔄 In Progress (Phase 2 ✅ Complete, Phase 3 📋 Pending)

**Phase Status**:
- ✅ **Phase 1**: KV Cache Semantic Clarification (Complete)
- ✅ **Phase 2**: Tensor-Level WeightPlacementMap (Complete - Oct 25, 2025)
- 📋 **Phase 3**: Pipeline Integration (Pending)

## Problem Statement

**Current limitation**: Layer-level device granularity is insufficient for **Mixture of Experts (MoE)** models.

### MoE Architecture Example

```
Qwen MoE 14B (hypothetical):
  - 40 layers total
  - Layers 0-5: Shared expert layers
  - Layers 6-39: Standard transformer layers

Layer 0 (Shared Expert Layer):
  ├─ Attention Block
  │  ├─ wq, wk, wv, wo: 896 MB → CPU (fits comfortably)
  │  └─ KV cache: Follows attention device
  │
  └─ FFN Block (MIXED DEVICE)
     ├─ Router network: 2 MB → CPU
     ├─ Shared Expert 0 (gate/up/down): 3.5 GB → GPU ⚡ (large, reused)
     ├─ Shared Expert 1 (gate/up/down): 3.5 GB → GPU ⚡ (large, reused)
     ├─ Shared Expert 2 (gate/up/down): 3.5 GB → GPU ⚡ (large, reused)
     ├─ ... (12 total shared experts)
     └─ Local FFN (gate/up/down): 896 MB → CPU
```

**Total Layer 0 memory**: ~45 GB (12 × 3.5 GB experts + 2 GB other)  
**Available GPU memory**: 24 GB  
**Solution**: Only shared experts on GPU, rest on CPU

### The Core Issue

**Layer-level device placement**:
```cpp
layer_devices[0] = ???;  // What device is layer 0?
// - Attention weights: CPU
// - Shared experts: GPU
// - Local FFN: CPU
// ❌ No single device represents this layer
```

**KV cache confusion**:
```cpp
kv_cache_->append_kv(layer_idx, K, V);
// Which device should cache[0] be on?
// - Attention computed on CPU → cache on CPU ✅
// - But shared experts on GPU → misleading if we say "layer 0 device = GPU"
```

## Solution Architecture

### 1. Semantic Clarification: KV Cache Tracks Attention Device

**Rename for clarity**:
```cpp
// OLD (ambiguous)
KVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
        const std::vector<int> &layer_devices);

// NEW (clear intent)
KVCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices);  // Where Q/K/V computation happens
```

**Documentation update**:
```cpp
/**
 * @brief Construct KV cache with per-layer attention device placement
 *
 * The KV cache is stored on the device where attention computation occurs
 * (i.e., where wq, wk, wv, wo weights reside). For MoE models, this may
 * differ from where FFN or expert weights are placed.
 *
 * @param attention_devices Device where attention is computed per layer
 *                          (length = n_layers, -1 = CPU, ≥0 = GPU)
 *
 * Example (MoE with shared experts on GPU):
 *   Layer 0: attention on CPU, experts on GPU → attention_devices[0] = -1
 *   Layer 1: attention on CPU, experts on GPU → attention_devices[1] = -1
 *   Layer 6: all weights on CPU → attention_devices[6] = -1
 */
```

### 2. Pipeline: Tensor-Level Device Tracking

**Refactor `LayerWeights` to track device per tensor group**:

```cpp
// src/v2/pipelines/qwen/Qwen2Pipeline.h
struct LayerWeights {
    // Attention block (device tracked per tensor)
    std::shared_ptr<TensorBase> wq;        // Query projection
    std::shared_ptr<TensorBase> wk;        // Key projection
    std::shared_ptr<TensorBase> wv;        // Value projection
    std::shared_ptr<TensorBase> wo;        // Output projection
    std::shared_ptr<TensorBase> attn_norm; // Pre-attention norm
    
    // FFN block (device tracked per tensor)
    std::shared_ptr<TensorBase> gate_proj; // FFN gate projection
    std::shared_ptr<TensorBase> up_proj;   // FFN up projection
    std::shared_ptr<TensorBase> down_proj; // FFN down projection
    std::shared_ptr<TensorBase> ffn_norm;  // Pre-FFN norm
    
    // MoE-specific (optional, nullptr for non-MoE layers)
    struct MoEWeights {
        std::shared_ptr<TensorBase> router;  // Router network
        std::vector<ExpertWeights> shared_experts;  // Shared across layers
        std::vector<ExpertWeights> local_experts;   // Layer-specific
    };
    std::shared_ptr<MoEWeights> moe;  // nullptr for standard layers
    
    // Device query methods
    int attention_device() const {
        // Return device where Q/K/V live (assume all co-located)
        return wq->device_index();
    }
    
    int ffn_device() const {
        // Return device where FFN lives (for non-MoE)
        return gate_proj->device_index();
    }
    
    bool is_moe() const { return moe != nullptr; }
};

struct ExpertWeights {
    std::shared_ptr<TensorBase> gate;
    std::shared_ptr<TensorBase> up;
    std::shared_ptr<TensorBase> down;
    
    int device() const { return gate->device_index(); }
};
```

### 3. Weight Placement Map: Tensor-Granular Specification

**Current design** (layer-level):
```cpp
class WeightPlacementMap {
    std::map<std::string, int> layer_devices_;  // "layer_0" → device
};
```

**New design** (tensor-level):
```cpp
class WeightPlacementMap {
public:
    // Specify device for specific tensor in specific layer
    void set_tensor_device(int layer_idx, const std::string &tensor_name, int device) {
        tensor_devices_[make_key(layer_idx, tensor_name)] = device;
    }
    
    int get_tensor_device(int layer_idx, const std::string &tensor_name) const {
        auto it = tensor_devices_.find(make_key(layer_idx, tensor_name));
        return (it != tensor_devices_.end()) ? it->second : default_device_;
    }
    
    // Convenience: Set all attention tensors for a layer
    void set_attention_device(int layer_idx, int device) {
        set_tensor_device(layer_idx, "wq", device);
        set_tensor_device(layer_idx, "wk", device);
        set_tensor_device(layer_idx, "wv", device);
        set_tensor_device(layer_idx, "wo", device);
        set_tensor_device(layer_idx, "attn_norm", device);
    }
    
    // Convenience: Set all FFN tensors for a layer
    void set_ffn_device(int layer_idx, int device) {
        set_tensor_device(layer_idx, "gate_proj", device);
        set_tensor_device(layer_idx, "up_proj", device);
        set_tensor_device(layer_idx, "down_proj", device);
        set_tensor_device(layer_idx, "ffn_norm", device);
    }
    
    // MoE-specific
    void set_shared_expert_device(int expert_idx, int device) {
        shared_expert_devices_[expert_idx] = device;
    }
    
private:
    std::map<std::string, int> tensor_devices_;  // "layer_0:wq" → device
    std::map<int, int> shared_expert_devices_;   // expert_idx → device
    int default_device_ = -1;
    
    std::string make_key(int layer, const std::string &tensor) const {
        return "layer_" + std::to_string(layer) + ":" + tensor;
    }
};
```

### 4. Usage Example: MoE Model Configuration

```cpp
// Configure MoE model with granular placement
auto placement_map = std::make_shared<WeightPlacementMap>();
placement_map->set_default_device(-1);  // CPU default

// Layers 0-5: Shared expert layers
for (int layer = 0; layer < 6; ++layer) {
    // Attention stays on CPU (moderate size)
    placement_map->set_attention_device(layer, -1);
    
    // Local FFN stays on CPU
    placement_map->set_ffn_device(layer, -1);
    
    // Shared experts go to GPU (large, reused)
    for (int expert = 0; expert < 12; ++expert) {
        placement_map->set_shared_expert_device(expert, 0);  // GPU 0
    }
}

// Layers 6-39: Standard transformer layers (all CPU)
for (int layer = 6; layer < 40; ++layer) {
    placement_map->set_attention_device(layer, -1);
    placement_map->set_ffn_device(layer, -1);
}

// Create pipeline with placement map
auto pipeline = std::make_shared<Qwen2Pipeline>(
    model_ctx, mpi_ctx, -1, placement_map);

// KV cache initialized with attention devices
std::vector<int> attention_devices(40, -1);  // All attention on CPU
auto kv_cache = std::make_shared<KVCache>(40, 2048, 8, 128, attention_devices);
```

**Result**:
```
[WeightPlacementMap] Layer 0:
  Attention (CPU): wq, wk, wv, wo, attn_norm → 896 MB
  FFN (CPU): gate, up, down, norm → 896 MB
  Shared Experts (GPU 0): 12 experts × 3.5 GB → 42 GB

[WeightPlacementMap] Layer 1:
  (same as Layer 0)

[WeightPlacementMap] Layer 6:
  Attention (CPU): 896 MB
  FFN (CPU): 896 MB

[KVCache] Attention device allocation:
  CPU: 40 layers × 2.4 MB/layer = 96 MB (all attention on CPU)
```

## Implementation Plan

### Phase 1: Semantic Clarification (Quick Win)

**Goal**: Make current code MoE-ready without breaking changes

1. **Rename parameter** `layer_devices` → `attention_devices` in KVCache constructor
2. **Update documentation** to clarify KV cache tracks attention device
3. **Add getter**: `get_attention_device(layer)` (alias for `get_layer_device`)
4. **Update tests** with new naming

**Files to modify**:
- `src/v2/tensors/KVCache.{h,cpp}` (parameter rename, docs)
- `tests/v2/unit/tensors/Test__KVCache.cpp` (update tests)
- `docs/V2_KV_CACHE_MULTI_DEVICE.md` (update documentation)

**Impact**: Zero breaking changes (old method still works), clearer semantics

### Phase 2: WeightPlacementMap Refactor (Medium)

**Goal**: Enable tensor-level device specification

1. **Extend `WeightPlacementMap`** with tensor-level API
2. **Update `Qwen2Pipeline` constructor** to use placement map
3. **Add `LayerWeights::attention_device()` method**
4. **Create MoE weight structure** (optional, for future)

**Files modified** (Phase 2 ✅ Complete - Oct 25, 2025):
- ✅ `src/v2/loaders/WeightPlacementMap.{h,cpp}` (+74 lines header, +81 lines impl)
  - `setAttentionDevice()`, `getAttentionDevice()` - Attention block convenience
  - `setFFNDevice()`, `getFFNDevice()` - FFN block convenience
  - `setSharedExpertDevice()`, `getSharedExpertDevice()` - Shared expert placement
  - `setLocalExpertDevice()`, `getLocalExpertDevice()` - Local expert placement
- ✅ `tests/v2/unit/loaders/Test__WeightPlacementMap.cpp` (160 lines, 9 tests)
- ✅ `tests/v2/CMakeLists.txt` (test integration with `$<TARGET_FILE:>`)

**Test Results**: ✅ 9/9 tests passing (3.09s)

**See**: [changelog/2025-10-25-phase3d-tensor-level-placement.md](../changelog/2025-10-25-phase3d-tensor-level-placement.md)

### Phase 3: Pipeline Integration (Pending)

**Goal**: Integrate WeightPlacementMap into Qwen2Pipeline for MoE-ready execution

1. **Update Qwen2Pipeline** to use block-level methods:
   ```cpp
   // BEFORE:
   int attn_device = placement_map_ ? getWeightDevice("attn_q", -1) : device_idx_;
   
   // AFTER:
   int attn_device = placement_map_ ? placement_map_->getAttentionDevice(layer_idx) : device_idx_;
   ```

2. **Auto-detect attention devices** from loaded weights for KV cache initialization

3. **Device-aware operator dispatch** based on tensor device

4. **Create end-to-end MoE example** demonstrating heterogeneous execution

**Files to modify**:
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` (use getAttentionDevice/getFFNDevice)
- `src/v2/pipelines/PipelineBase.cpp` (auto-detect attention devices)
- `tests/v2/integration/` (end-to-end MoE placement test)

### Phase 4: Advanced Device Orchestration (Long-term)

**Goal**: Automatic device orchestration for mixed-device layers

1. **Kernel device dispatch** based on tensor device
2. **Automatic transfers** when inputs/outputs on different devices
3. **Device-aware operator fusion** (avoid unnecessary transfers)
4. **Memory-aware placement strategy** (auto-select which experts go to GPU)

**This is Phase 5+ work** (after batch processing)

## Open Questions

### Q1: What if attention tensors (Q/K/V/O) are on different devices?

**Answer**: Enforce constraint that all attention tensors for a layer co-locate.

```cpp
void WeightPlacementMap::setAttentionDevice(int layer, int device) {
    // All or nothing: Q/K/V/O must be on same device
    set_tensor_device(layer, "wq", device);
    set_tensor_device(layer, "wk", device);
    set_tensor_device(layer, "wv", device);
    set_tensor_device(layer, "wo", device);
    set_tensor_device(layer, "attn_norm", device);
}
```

**Rationale**: Attention computation requires Q/K/V co-located anyway.

**Status**: ✅ Implemented (Phase 2)

### Q2: Should KV cache auto-detect device from weights?

**Option A**: Explicit attention_devices vector (current - Phase 1)

```cpp
std::vector<int> attention_devices = compute_attention_devices(weights);
auto cache = std::make_shared<KVCache>(..., attention_devices);
```

**Option B**: Auto-detect from pipeline
```cpp
auto cache = pipeline->create_kv_cache();  // Auto-detects from weights
```

**Recommendation**: Start with **Option A** (explicit), move to **Option B** later.

### Q3: How do we handle expert selection routing?

**MoE forward pass**:
```cpp
// Router runs on CPU (small network)
auto router_logits = router_network(input);  // CPU
auto expert_ids = select_experts(router_logits);  // CPU

// Experts run on GPU (large networks)
for (int expert_id : expert_ids) {
    // Transfer input CPU → GPU
    auto expert_input_gpu = transfer_to_device(input, 0);
    
    // Compute on GPU
    auto expert_output_gpu = shared_experts[expert_id].forward(expert_input_gpu);
    
    // Transfer output GPU → CPU
    auto expert_output_cpu = transfer_to_device(expert_output_gpu, -1);
    
    // Accumulate on CPU
    accumulate(output, expert_output_cpu);
}
```

**Optimization**: Batch transfers, fuse expert computation on GPU.

## Next Steps

1. **Immediate**: Implement Phase 1 (semantic clarification) ← ~1 hour
2. **Short-term**: Design WeightPlacementMap API (Phase 2) ← ~2-3 hours
3. **Document**: MoE placement strategies and transfer overhead analysis

## References

- **Current KV Cache**: `src/v2/tensors/KVCache.{h,cpp}`
- **Pipeline Weights**: `src/v2/pipelines/qwen/Qwen2Pipeline.h` (LayerWeights)
- **MoE Papers**: 
  - Mixtral 8x7B: https://arxiv.org/abs/2401.04088
  - Switch Transformers: https://arxiv.org/abs/2101.03961

---

**TL;DR**: 
- ✅ **KV cache**: Track attention device (rename `layer_devices` → `attention_devices`)
- 🔄 **Pipeline**: Add tensor-level `WeightPlacementMap` for MoE models
- 📋 **Future**: Auto device orchestration with cross-device transfer optimization
