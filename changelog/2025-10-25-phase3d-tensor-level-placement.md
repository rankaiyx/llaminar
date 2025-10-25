# Phase 3d Extension: Tensor-Level Device Placement (Phase 2 Complete)

**Date**: October 25, 2025  
**Component**: V2 Device Management - WeightPlacementMap  
**Scope**: MoE-Ready Tensor-Level Device Placement  
**Status**: ✅ Phase 2 Complete

## Summary

Completed Phase 2 of MoE device placement architecture by implementing **block-level and MoE-specific device placement methods** in `WeightPlacementMap`. This enables fine-grained control over device placement at the tensor level, critical for MoE models where different weight types within a layer may reside on different devices.

**Test Results**: ✅ 9/9 tests passing (3.09s)

## Context

From [Phase 1](./2025-10-25-phase3d-moe-semantic-clarification.md), we established that KV cache tracks **attention device** placement. Phase 2 extends this by providing the infrastructure for pipelines to specify and query tensor-level device placement.

**MoE Use Case**:
```
Layer 0:
  - Attention weights (Q/K/V/O): CPU (896 MB)
  - Shared Expert 0-11: GPU (42 GB - reused across layers)
  - Local FFN: CPU (896 MB)
```

Existing layer-level APIs (`setLayerDevice(layer, device)`) insufficient → Need tensor-level control.

## Implementation Changes

### 1. Enhanced `src/v2/loaders/WeightPlacementMap.h` (74 lines added)

**Block-Level Methods** (Attention, FFN convenience):
```cpp
// Set all attention tensors for a layer to specific device
void setAttentionDevice(int layer_idx, int device_idx);
int getAttentionDevice(int layer_idx) const;

// Set all FFN tensors for a layer to specific device
void setFFNDevice(int layer_idx, int device_idx);
int getFFNDevice(int layer_idx) const;
```

**MoE-Specific Methods** (Expert placement):
```cpp
// Shared experts (reused across layers, typically on GPU)
void setSharedExpertDevice(int expert_idx, int device_idx);
int getSharedExpertDevice(int expert_idx) const;

// Local experts (layer-specific, typically on CPU/edge devices)
void setLocalExpertDevice(int layer_idx, int expert_idx, int device_idx);
int getLocalExpertDevice(int layer_idx, int expert_idx) const;
```

**New Private Members**:
```cpp
std::unordered_map<int, int> shared_expert_to_device_;  // Expert index → device
std::unordered_map<std::string, int> local_expert_to_device_;  // "layer_X:expert_Y" → device
```

### 2. Enhanced `src/v2/loaders/WeightPlacementMap.cpp` (81 lines added)

**Implementation Pattern**:
- Block-level methods set multiple tensor-specific mappings
- MoE methods use both direct maps and pattern-based rules
- Getters use existing `getDeviceForWeight()` priority logic

**Example Implementation**:
```cpp
void WeightPlacementMap::setAttentionDevice(int layer_idx, int device_idx) {
    std::string base = "blk." + std::to_string(layer_idx) + ".";
    setTensorDevice(base + "attn_q.weight", device_idx);
    setTensorDevice(base + "attn_k.weight", device_idx);
    setTensorDevice(base + "attn_v.weight", device_idx);
    setTensorDevice(base + "attn_output.weight", device_idx);
    setTensorDevice(base + "attn_norm.weight", device_idx);
}

int WeightPlacementMap::getAttentionDevice(int layer_idx) const {
    std::string attn_q_name = "blk." + std::to_string(layer_idx) + ".attn_q.weight";
    return getDeviceForWeight(attn_q_name, layer_idx);
}
```

**MoE Pattern Matching**:
```cpp
void WeightPlacementMap::setSharedExpertDevice(int expert_idx, int device_idx) {
    shared_expert_to_device_[expert_idx] = device_idx;
    
    // Set pattern for all tensors matching this expert
    std::string pattern = "shared_expert." + std::to_string(expert_idx) + ".*";
    setPatternDevice(pattern, device_idx);
}
```

### 3. Comprehensive Test Suite (`tests/v2/unit/loaders/Test__WeightPlacementMap.cpp`)

**File**: 160 lines, 9 test cases  
**Coverage**: Block-level placement, MoE scenarios, existing API compatibility

**Test Cases**:
1. ✅ `AttentionDevicePlacement` - Verify attention block placement
2. ✅ `FFNDevicePlacement` - Verify FFN block placement
3. ✅ `MixedAttentionFFNPlacement` - Different devices per block
4. ✅ `SharedExpertPlacement` - Shared expert GPU placement
5. ✅ `LocalExpertPlacement` - Layer-specific expert placement
6. ✅ `MoEHeterogeneousPlacement` - Realistic MoE scenario (24 layers, 8 experts)
7. ✅ `LayerRangePlacement` - Existing API compatibility
8. ✅ `PatternBasedPlacement` - Pattern matching still works
9. ✅ `ClearResetsAllMaps` - Cleanup verifies all maps cleared

**Example Realistic Test**:
```cpp
TEST_F(WeightPlacementMapTest, MoEHeterogeneousPlacement) {
    int n_layers = 24;
    int n_experts = 8;
    
    // All attention/FFN on CPU
    for (int i = 0; i < n_layers; ++i) {
        map_->setAttentionDevice(i, -1);
        map_->setFFNDevice(i, -1);
    }
    
    // Shared experts on GPU
    for (int i = 0; i < n_experts; ++i) {
        map_->setSharedExpertDevice(i, 0);
    }
    
    // Verify placements
    EXPECT_EQ(map_->getAttentionDevice(0), -1);  // CPU
    EXPECT_EQ(map_->getFFNDevice(0), -1);        // CPU
    EXPECT_EQ(map_->getSharedExpertDevice(0), 0); // GPU
}
```

### 4. CMake Integration (`tests/v2/CMakeLists.txt`)

**Critical Fix**: Use `$<TARGET_FILE:target>` for executable paths in CTest  
**Issue**: MPI working directory is project root, executables in `build_v2/tests/v2/`  
**Solution**: CMake generator expression resolves full path at build time

```cmake
add_executable(v2_test_weight_placement_map unit/loaders/Test__WeightPlacementMap.cpp)
target_link_libraries(v2_test_weight_placement_map llaminar2_core GTest::gtest_main)
add_v2_test(V2_Unit_WeightPlacementMap
    COMMAND $<TARGET_FILE:v2_test_weight_placement_map>  # Full path
    LABELS "V2;Unit;DeviceManagement;WeightPlacement;Heterogeneous;MoE"
)
```

## Files Modified/Created

**Enhanced Existing Implementation**:
- `src/v2/loaders/WeightPlacementMap.h` (+74 lines) - Method declarations
- `src/v2/loaders/WeightPlacementMap.cpp` (+81 lines) - Method implementations

**New Test Suite**:
- `tests/v2/unit/loaders/Test__WeightPlacementMap.cpp` (160 lines) - 9 comprehensive tests

**Build Configuration**:
- `tests/v2/CMakeLists.txt` (modified) - Test definition with fixed paths

**Duplicate Cleanup**:
- ❌ Removed `src/v2/utils/WeightPlacementMap.h` (duplicate created by mistake)

## Test Results

```
Test project /workspaces/llaminar/build_v2
    Start 1: V2_FetchModelsFixture
1/2 Test #1: V2_FetchModelsFixture ............   Passed    0.00 sec
    Start 5: V2_Unit_WeightPlacementMap
2/2 Test #5: V2_Unit_WeightPlacementMap .......   Passed    3.09 sec

100% tests passed, 0 tests failed out of 2

Test Summary:
  AttentionDevicePlacement ............... PASSED (0 ms)
  FFNDevicePlacement ..................... PASSED (0 ms)
  MixedAttentionFFNPlacement ............. PASSED (0 ms)
  SharedExpertPlacement .................. PASSED (0 ms)
  LocalExpertPlacement ................... PASSED (0 ms)
  MoEHeterogeneousPlacement .............. PASSED (0 ms)
  LayerRangePlacement .................... PASSED (0 ms)
  PatternBasedPlacement .................. PASSED (0 ms)
  ClearResetsAllMaps ..................... PASSED (0 ms)
```

## Architecture Decisions

### 1. Enhance Existing vs Create New

**Decision**: Enhanced existing `src/v2/loaders/WeightPlacementMap`  
**Rationale**:
- Already integrated into `PipelineBase` and `Qwen2Pipeline`
- Solid foundation (pattern matching, layer ranges)
- Adding methods > duplicating infrastructure

### 2. Block-Level + MoE-Specific Methods

**Decision**: Separate methods for attention, FFN, and expert placement  
**Rationale**:
- **Ergonomics**: `setAttentionDevice(0, GPU)` clearer than 5 tensor calls
- **MoE Semantics**: Shared experts fundamentally different from local FFN
- **Backward Compatibility**: Existing `setLayerDevice()` still works

### 3. Pattern Matching for Experts

**Decision**: MoE methods set both direct maps AND pattern rules  
**Rationale**:
- Direct map: Fast lookup for expert index
- Pattern: Catches all expert tensors (gate, up, down, etc.)
- Redundancy intentional → robust placement

## API Usage Example

```cpp
// Create placement map (CPU default)
auto placement_map = std::make_shared<WeightPlacementMap>(-1);

// MoE heterogeneous placement
int n_layers = 24;
int n_experts = 8;

// Attention/FFN on CPU (moderate size, frequent access)
for (int i = 0; i < n_layers; ++i) {
    placement_map->setAttentionDevice(i, -1);  // CPU
    placement_map->setFFNDevice(i, -1);        // CPU
}

// Shared experts on GPU (large, reused, parallel)
for (int i = 0; i < n_experts; ++i) {
    placement_map->setSharedExpertDevice(i, 0);  // GPU 0
}

// Initialize pipeline with placement map
auto pipeline = std::make_unique<Qwen2Pipeline>(
    model_loader,
    placement_map,  // Device placement specification
    kv_cache
);

// Pipeline queries placement during weight loading
int attn_device = placement_map->getAttentionDevice(layer_idx);  // -1 (CPU)
int expert_device = placement_map->getSharedExpertDevice(0);     // 0 (GPU)
```

## Integration Points

**Current State**:
- ✅ `PipelineBase` has `placement_map_` member
- ✅ `Qwen2Pipeline` uses `placement_map_` for device queries (20+ references)
- ✅ `KVCache` uses `attention_devices` (Phase 1)

**Future Work** (Phase 3):
- Auto-detect attention devices from loaded weights
- Initialize KV cache with placement map devices
- Update `Qwen2Pipeline` to use `getAttentionDevice()` / `getFFNDevice()` methods
- Implement device-aware operator dispatch

## Lessons Learned

1. **Search Before Creating**: Could have avoided duplicate by searching for "WeightPlacementMap" in `src/` earlier
2. **CTest Working Directory**: MPI tests need full executable paths when working directory ≠ build directory
3. **CMake Generator Expressions**: `$<TARGET_FILE:target>` cleaner than manual path construction
4. **Incremental Testing**: Direct execution (`./build_v2/tests/v2/test`) faster than CTest for debugging

## Next Steps (Phase 3)

1. **Update Qwen2Pipeline** to use new block-level methods:
   ```cpp
   // BEFORE:
   int attn_device = placement_map_ ? getWeightDevice("attn_q", -1) : device_idx_;
   
   // AFTER:
   int attn_device = placement_map_ ? placement_map_->getAttentionDevice(layer_idx) : device_idx_;
   ```

2. **Auto-Detect Attention Devices** from loaded weights for KV cache initialization

3. **End-to-End MoE Example** demonstrating heterogeneous execution

4. **Performance Testing** with real MoE models (Mixtral, Qwen2-MoE)

## References

- **Phase 1**: [2025-10-25-phase3d-moe-semantic-clarification.md](./2025-10-25-phase3d-moe-semantic-clarification.md)
- **Design Document**: [TENSOR_LEVEL_DEVICE_PLACEMENT.md](../docs/TENSOR_LEVEL_DEVICE_PLACEMENT.md)
- **V2 Architecture**: [.github/instructions/llaminar-v2-architecture.instructions.md](../.github/instructions/llaminar-v2-architecture.instructions.md)
- **Issue**: "How will the kv cache work with multiple devices?"
- **Critical Insight**: "I think we need to be more.. granular, though?" (MoE requires tensor-level placement)
