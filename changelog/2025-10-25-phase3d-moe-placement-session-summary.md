# Phase 3d Extension: MoE Device Placement (Session Summary)

**Date**: October 25, 2025  
**Scope**: V2 KV Cache + WeightPlacementMap MoE Readiness  
**Status**: ✅ Phase 1 + Phase 2 Complete

## Executive Summary

Completed **2 phases** of MoE-ready device placement architecture:

1. **Phase 1** (✅ Complete): KV Cache Semantic Clarification
   - Renamed `layer_devices` → `attention_devices` in KVCache
   - Added `get_attention_device()` method for explicit semantics
   - Updated all tests and documentation
   - **Result**: ✅ 8/8 tests passing (2.66s)

2. **Phase 2** (✅ Complete): Tensor-Level WeightPlacementMap
   - Added block-level methods (`setAttentionDevice`, `setFFNDevice`)
   - Added MoE methods (`setSharedExpertDevice`, `setLocalExpertDevice`)
   - Comprehensive test suite (9 test cases)
   - **Result**: ✅ 9/9 tests passing (3.09s)

**Total Implementation**: 155 lines C++ code, 160 lines tests, 3 documentation files

## Context: The Problem

**User Question**: *"how will the kv cache work with multiple devices (for example, CPU + GPU inference with layers split across both?)"*

**Critical Insight** (User): *"I think we need to be more.. granular, though?"*

**MoE Use Case**:
```
Layer 0 in MoE Model:
  - Attention (Q/K/V/O): CPU (896 MB)
  - Shared Expert 0-11: GPU (42 GB - too large for single device, reused across layers)
  - Local FFN: CPU (896 MB)

Question: What device is "layer 0"?
Answer: No single device - need tensor-level granularity!
```

**Layer-level placement is insufficient** for MoE models where different weight types within a layer reside on different devices.

## Implementation Overview

### Phase 1: KV Cache Semantic Clarification (Oct 25, 2025)

**Problem**: KV cache should track where **attention computation** happens, not ambiguous "layer device"

**Solution**: Rename parameter and add semantic getter

**Files Modified**:
- `src/v2/tensors/KVCache.{h,cpp}` (parameter renamed, docs updated)
- `tests/v2/unit/tensors/Test__KVCache.cpp` (tests updated)
- `docs/V2_KV_CACHE_MULTI_DEVICE.md` (semantic clarification)
- `changelog/2025-10-25-phase3d-moe-semantic-clarification.md` (Phase 1 changelog)
- `docs/TENSOR_LEVEL_DEVICE_PLACEMENT.md` (design document created)

**Key Change**:
```cpp
// BEFORE (ambiguous):
KVCache(int n_layers, ..., const std::vector<int> &layer_devices);

// AFTER (explicit semantics):
KVCache(int n_layers, ..., const std::vector<int> &attention_devices);

// New method:
int get_attention_device(int layer) const;  // KV cache follows attention computation
```

**Test Results**: ✅ 8/8 tests passing (2.66s)

**Changelog**: [2025-10-25-phase3d-moe-semantic-clarification.md](./2025-10-25-phase3d-moe-semantic-clarification.md)

### Phase 2: Tensor-Level WeightPlacementMap (Oct 25, 2025)

**Problem**: Existing WeightPlacementMap only supported layer-level placement

**Solution**: Add block-level and MoE-specific placement methods

**Files Modified**:
- `src/v2/loaders/WeightPlacementMap.h` (+74 lines) - Method declarations
- `src/v2/loaders/WeightPlacementMap.cpp` (+81 lines) - Method implementations
- `tests/v2/unit/loaders/Test__WeightPlacementMap.cpp` (160 lines, 9 tests)
- `tests/v2/CMakeLists.txt` (test integration, fixed executable paths)
- `changelog/2025-10-25-phase3d-tensor-level-placement.md` (Phase 2 changelog)
- `docs/TENSOR_LEVEL_DEVICE_PLACEMENT.md` (updated with Phase 2 completion)

**New API**:

**Block-Level Methods** (Convenience):
```cpp
void setAttentionDevice(int layer_idx, int device_idx);  // Sets wq, wk, wv, wo, attn_norm
int getAttentionDevice(int layer_idx) const;

void setFFNDevice(int layer_idx, int device_idx);        // Sets gate, up, down, ffn_norm
int getFFNDevice(int layer_idx) const;
```

**MoE-Specific Methods**:
```cpp
void setSharedExpertDevice(int expert_idx, int device_idx);
int getSharedExpertDevice(int expert_idx) const;

void setLocalExpertDevice(int layer_idx, int expert_idx, int device_idx);
int getLocalExpertDevice(int layer_idx, int expert_idx) const;
```

**Example Usage**:
```cpp
auto map = std::make_shared<WeightPlacementMap>(-1);  // CPU default

// MoE heterogeneous placement
for (int i = 0; i < 24; ++i) {
    map->setAttentionDevice(i, -1);  // Attention on CPU
    map->setFFNDevice(i, -1);        // Local FFN on CPU
}

for (int i = 0; i < 8; ++i) {
    map->setSharedExpertDevice(i, 0);  // Shared experts on GPU 0
}
```

**Test Results**: ✅ 9/9 tests passing (3.09s)

**Test Coverage**:
1. ✅ AttentionDevicePlacement - Verify attention block placement
2. ✅ FFNDevicePlacement - Verify FFN block placement
3. ✅ MixedAttentionFFNPlacement - Different devices per block
4. ✅ SharedExpertPlacement - Shared expert GPU placement
5. ✅ LocalExpertPlacement - Layer-specific expert placement
6. ✅ MoEHeterogeneousPlacement - Realistic MoE scenario (24 layers, 8 experts)
7. ✅ LayerRangePlacement - Existing API compatibility
8. ✅ PatternBasedPlacement - Pattern matching still works
9. ✅ ClearResetsAllMaps - Cleanup verification

**Changelog**: [2025-10-25-phase3d-tensor-level-placement.md](./2025-10-25-phase3d-tensor-level-placement.md)

## Technical Highlights

### 1. Semantic Precision (Phase 1)

**Insight**: KV cache doesn't belong to a "layer" - it belongs to **attention computation**

**Impact**:
- Clearer API for MoE where attention and experts may be on different devices
- Documentation now explicit: "KV cache resides where Q/K/V computation happens"
- Backward compatible: `get_layer_device()` still works

### 2. Convenience API Design (Phase 2)

**Insight**: Block-level methods reduce boilerplate and prevent errors

**Impact**:
```cpp
// WITHOUT convenience methods (error-prone):
map->setTensorDevice("blk.0.attn_q.weight", 0);
map->setTensorDevice("blk.0.attn_k.weight", 0);
map->setTensorDevice("blk.0.attn_v.weight", 0);
map->setTensorDevice("blk.0.attn_output.weight", 0);
map->setTensorDevice("blk.0.attn_norm.weight", 0);

// WITH convenience methods (clear intent):
map->setAttentionDevice(0, 0);
```

### 3. Pattern Matching for MoE (Phase 2)

**Insight**: Experts have multiple tensors (gate, up, down) - pattern matching catches all

**Implementation**:
```cpp
void WeightPlacementMap::setSharedExpertDevice(int expert_idx, int device_idx) {
    shared_expert_to_device_[expert_idx] = device_idx;  // Direct map
    
    // Pattern catches all expert tensors
    std::string pattern = "shared_expert." + std::to_string(expert_idx) + ".*";
    setPatternDevice(pattern, device_idx);
}
```

**Impact**: Robust placement - catches `shared_expert.0.gate.weight`, `shared_expert.0.up.weight`, etc.

### 4. CMake Integration Fix (Phase 2)

**Problem**: CTest couldn't find executables (working directory ≠ build directory)

**Solution**: Use CMake generator expressions for full paths
```cmake
# BEFORE (broken):
add_v2_test(V2_Unit_WeightPlacementMap COMMAND v2_test_weight_placement_map ...)

# AFTER (works):
add_v2_test(V2_Unit_WeightPlacementMap COMMAND $<TARGET_FILE:v2_test_weight_placement_map> ...)
```

**Impact**: Tests now run correctly via CTest despite MPI working directory being project root

## Architecture Decisions

### 1. Enhance Existing vs Create New

**Decision**: Enhanced existing `src/v2/loaders/WeightPlacementMap`

**Alternatives Considered**:
- Create new `TensorPlacementMap` → Rejected (duplication)
- Create new file in `src/v2/utils/` → Tried, then reverted (integration work)

**Rationale**:
- Already integrated into `PipelineBase` and `Qwen2Pipeline`
- Solid foundation (pattern matching, layer ranges)
- Adding methods > duplicating infrastructure

### 2. Separate Block-Level and MoE Methods

**Decision**: Distinct methods for attention, FFN, and expert placement

**Alternatives Considered**:
- Generic `setBlockDevice(layer, block_type, device)` → Rejected (less type-safe)
- Only MoE methods, no block convenience → Rejected (ergonomics)

**Rationale**:
- **Ergonomics**: `setAttentionDevice(0, GPU)` clearer than 5 tensor calls
- **MoE Semantics**: Shared experts fundamentally different from local FFN
- **Backward Compatibility**: Existing `setLayerDevice()` still works

### 3. Redundant Pattern + Direct Map for MoE

**Decision**: MoE methods set BOTH direct map AND pattern rules

**Implementation**:
```cpp
void setSharedExpertDevice(int expert_idx, int device_idx) {
    shared_expert_to_device_[expert_idx] = device_idx;  // Fast lookup
    setPatternDevice("shared_expert." + ..., device_idx);  // Catch all tensors
}
```

**Rationale**: Redundancy intentional → robust placement catches all expert tensors

## Lessons Learned

### 1. Search Before Creating

**Mistake**: Created `src/v2/utils/WeightPlacementMap.h` before discovering `src/v2/loaders/WeightPlacementMap.h`

**Lesson**: `grep -r "WeightPlacementMap" src/` would have saved ~1 hour

**Impact**: Had to delete duplicate, move tests, update CMake paths

### 2. CTest Working Directory Gotcha

**Problem**: All V2 unit tests were failing silently (hadn't run CTest in a while)

**Root Cause**: MPI working directory is project root (`WORKING_DIRECTORY`), executables in `build_v2/tests/v2/`

**Solution**: Use `$<TARGET_FILE:target>` CMake generator expression for full paths

**Impact**: Critical for MPI tests where `mpirun` resolves executables differently than direct execution

### 3. Incremental Testing is Faster

**Discovery**: Direct execution (`./build_v2/tests/v2/test`) ~instant vs CTest setup ~3s

**Practice**: Test directly during development, CTest for CI validation

**Impact**: Faster feedback loop during test writing

### 4. Semantic Precision Matters

**Insight**: "layer device" was ambiguous - MoE exposed this immediately

**Lesson**: When user says "I think we need to be more granular", believe them!

**Impact**: Phase 1 semantic clarification prevented confusion in Phase 2

## Integration Points

**Current State**:
- ✅ `PipelineBase` has `placement_map_` member (existing)
- ✅ `Qwen2Pipeline` uses `placement_map_` for device queries (20+ references)
- ✅ `KVCache` uses `attention_devices` (Phase 1)
- ✅ `WeightPlacementMap` has block-level and MoE methods (Phase 2)

**Next Steps** (Phase 3):
1. Update `Qwen2Pipeline` to use `getAttentionDevice()` / `getFFNDevice()`
2. Auto-detect attention devices from loaded weights
3. Initialize KV cache with placement map devices
4. Create end-to-end MoE placement example

## Performance Considerations

**No performance impact from Phase 1/2**:
- KV cache parameter rename: Zero overhead (same data structure)
- WeightPlacementMap methods: Construction-time only (not in hot path)
- Tests run in 3-6 seconds (acceptable for unit tests)

**Future Performance Work** (Phase 3+):
- Device-aware operator dispatch
- Minimize cross-device transfers
- Memory-aware placement strategies

## Documentation Created

1. **Design Document**: `docs/TENSOR_LEVEL_DEVICE_PLACEMENT.md` (285 lines)
   - Problem statement with MoE use case
   - Solution architecture
   - Implementation phases
   - Updated with Phase 1 and Phase 2 completion status

2. **Phase 1 Changelog**: `changelog/2025-10-25-phase3d-moe-semantic-clarification.md` (230 lines)
   - KV cache semantic clarification
   - Files modified, test results
   - API changes and rationale

3. **Phase 2 Changelog**: `changelog/2025-10-25-phase3d-tensor-level-placement.md` (420 lines)
   - WeightPlacementMap enhancement
   - Comprehensive API documentation
   - Test results and coverage
   - Integration examples

4. **Session Summary**: This document (comprehensive record)

## Statistics

**Code Changes**:
- C++ Implementation: 155 lines (74 header + 81 implementation)
- Test Code: 160 lines (9 test cases)
- Documentation: 935 lines (3 documents)

**Test Coverage**:
- Phase 1: 8/8 tests passing (2.66s)
- Phase 2: 9/9 tests passing (3.09s)
- **Total**: 17/17 tests passing ✅

**Files Modified/Created**:
- Modified: 7 files
- Created: 5 files (3 docs, 1 test, 1 changelog)
- Deleted: 1 file (duplicate)

## Next Steps (Phase 3)

**Short-term** (1-2 sessions):
1. Update `Qwen2Pipeline` to use block-level methods
   ```cpp
   // Replace:
   int attn_device = placement_map_ ? getWeightDevice("attn_q", -1) : device_idx_;
   // With:
   int attn_device = placement_map_ ? placement_map_->getAttentionDevice(layer_idx) : device_idx_;
   ```

2. Auto-detect attention devices from loaded weights
   ```cpp
   std::vector<int> detect_attention_devices(const WeightPlacementMap& map, int n_layers) {
       std::vector<int> devices(n_layers);
       for (int i = 0; i < n_layers; ++i) {
           devices[i] = map.getAttentionDevice(i);
       }
       return devices;
   }
   ```

3. Initialize KV cache with auto-detected devices
   ```cpp
   auto attn_devices = detect_attention_devices(*placement_map_, n_layers);
   kv_cache_ = std::make_shared<KVCache>(n_layers, max_seq_len, n_kv_heads, head_dim, attn_devices);
   ```

**Long-term** (Phase 4+):
- Device-aware operator dispatch
- Automatic cross-device transfers
- Memory-aware placement strategies
- MoE model support (Mixtral, Qwen2-MoE)
- Performance testing and optimization

## References

**Design Documents**:
- [TENSOR_LEVEL_DEVICE_PLACEMENT.md](../docs/TENSOR_LEVEL_DEVICE_PLACEMENT.md) - Master design document
- [V2_KV_CACHE_MULTI_DEVICE.md](../docs/V2_KV_CACHE_MULTI_DEVICE.md) - KV cache multi-device documentation
- [llaminar-v2-architecture.instructions.md](../.github/instructions/llaminar-v2-architecture.instructions.md) - V2 architecture overview

**Changelogs**:
- [2025-10-25-phase3d-moe-semantic-clarification.md](./2025-10-25-phase3d-moe-semantic-clarification.md) - Phase 1 complete
- [2025-10-25-phase3d-tensor-level-placement.md](./2025-10-25-phase3d-tensor-level-placement.md) - Phase 2 complete

**User Insights**:
- Original question: "how will the kv cache work with multiple devices?"
- Critical insight: "I think we need to be more.. granular, though?"
- MoE scenario: "shared MoE tensors on GPU whilst keeping other tensors (ffn, etc) on CPU"

## Conclusion

Successfully completed **2 phases** of MoE-ready device placement:

1. ✅ **Phase 1**: Semantic clarification - KV cache tracks attention device
2. ✅ **Phase 2**: Infrastructure - WeightPlacementMap supports tensor-level placement

**Impact**: V2 infrastructure now ready for MoE heterogeneous execution

**Next**: Phase 3 integration will connect placement map to pipeline execution
