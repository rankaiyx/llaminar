# PrefillProvider Parity Testing Quick Start

## Overview

The `PrefillProvider` abstraction enables **stage-by-stage** parity testing against PyTorch references, allowing you to pinpoint the exact layer and stage where divergence begins.

## Quick Example

### Before (QwenPipeline - end-to-end only)
```cpp
TEST(PrefillParity, EndToEndOnly) {
    QwenPipeline pipeline(config);
    pipeline.execute(tokens, weights, output);
    
    // Compare final output only - if divergence found, no idea where it started!
    compareWithPyTorch(output, pytorch_final_output);
    // ❌ Divergence at layer 15? Layer 20? We don't know!
}
```

### After (OpenBLASPrefillProvider - stage-by-stage)
```cpp
TEST(OpenBLASPrefillVsPyTorch, StageByStage) {
    // Enable snapshot capture
    setenv("LLAMINAR_PARITY_CAPTURE", "1", 1);
    
    // Create provider
    auto provider = std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
    
    // Execute with snapshot capture
    PrefillMetrics metrics;
    provider->execute(tokens, weights, output, ctx, metrics);
    
    // Load PyTorch reference snapshots
    PyTorchSnapshotLoader pytorch("pytorch_snapshots/");
    
    // Compare stage-by-stage
    compareSnapshot(PipelineStage::EMBEDDING, -1, pytorch);  // ✅ PASS
    
    for (int layer = 0; layer < 28; ++layer) {
        compareSnapshot(PipelineStage::ATTENTION_NORM, layer, pytorch);      // ✅ PASS
        compareSnapshot(PipelineStage::ATTENTION_OUTPUT, layer, pytorch);    // ✅ PASS
        compareSnapshot(PipelineStage::ATTENTION_RESIDUAL, layer, pytorch);  // ✅ PASS
        compareSnapshot(PipelineStage::FFN_NORM, layer, pytorch);            // ✅ PASS
        compareSnapshot(PipelineStage::FFN_DOWN, layer, pytorch);            
        // ❌ FAIL at layer 15, stage FFN_DOWN - DIVERGENCE FOUND!
        // rel_l2=0.05, max_abs=0.12
        break;  // Stop at first divergence
    }
    
    // Now you know: divergence starts at layer 15, FFN down projection
    // You can focus debugging on that specific operation!
}
```

## Snapshot Stages Available

### Global Stages
- `EMBEDDING` (layer=-1): After token embedding lookup
- `FINAL_NORM` (layer=-1): After final RMSNorm
- `LM_HEAD` (layer=-1): After language model head projection

### Per-Layer Stages (layer=0..27 for Qwen-0.5B)
- `ATTENTION_NORM`: After attention RMSNorm (input to Q/K/V)
- `ATTENTION_OUTPUT`: After attention output projection W_o
- `ATTENTION_RESIDUAL`: After attention residual add
- `FFN_NORM`: After FFN RMSNorm (input to gate/up)
- `FFN_DOWN`: After FFN down projection
- `FFN_RESIDUAL`: After FFN residual add (final layer output)

**Total**: 3 global + (6 × 28 layers) = **171 snapshots** for full prefill

## Generating PyTorch Reference Snapshots

```bash
# 1. Run PyTorch reference implementation with snapshot capture
cd python/reference
python run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --capture-stages all \
    --output pytorch_snapshots.npz

# 2. Convert NPZ to individual .npy files (for easier loading)
python ../tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/

# 3. Directory structure:
# pytorch_snapshots/
#   embedding_-1.npy              (shape: [5, 896])
#   attention_norm_0.npy          (shape: [5, 896])
#   attention_output_0.npy        (shape: [5, 896])
#   ...
#   ffn_residual_27.npy           (shape: [5, 896])
#   final_norm_-1.npy             (shape: [5, 896])
#   lm_head_-1.npy                (shape: [5, 151936])
```

## Running Parity Tests

### Environment Setup
```bash
export PYTORCH_SNAPSHOT_DIR=pytorch_snapshots/
export PYTORCH_SNAPSHOT_TOKENS=1,2,3,4,5
export LLAMINAR_PARITY_CAPTURE=1  # Enable snapshot capture
```

### Run OpenBLAS Prefill Parity Test
```bash
mpirun -np 2 ./build/test_parity_framework \
    --gtest_filter="ParityFramework.OpenBLASPrefillVsPyTorch"
```

**Expected Output**:
```
[==========] Running 1 test from 1 test suite.
[----------] 1 test from ParityFramework
[ RUN      ] ParityFramework.OpenBLASPrefillVsPyTorch
[INFO] OpenBLASPrefillProvider: 5 tokens, 28 layers, 12.5ms total, 171 snapshots
[INFO] Comparing EMBEDDING (layer=-1): rel_l2=0.0001, max_abs=0.0002 ✅
[INFO] Comparing ATTENTION_NORM (layer=0): rel_l2=0.0001, max_abs=0.0003 ✅
[INFO] Comparing ATTENTION_OUTPUT (layer=0): rel_l2=0.0002, max_abs=0.0005 ✅
...
[INFO] All 171 snapshots within tolerance ✅
[       OK ] ParityFramework.OpenBLASPrefillVsPyTorch (125 ms)
[----------] 1 test from ParityFramework (125 ms total)
```

**If Divergence Found**:
```
[INFO] Comparing FFN_DOWN (layer=15): rel_l2=0.0523, max_abs=0.1245 ❌
[ERROR] Divergence at layer 15, stage FFN_DOWN
[ERROR] Expected max_abs < 0.01, got 0.1245 (12.45x over tolerance)
[ERROR] First divergence found - stopping comparison
[  FAILED  ] ParityFramework.OpenBLASPrefillVsPyTorch (98 ms)
```

## Debugging Workflow

### Step 1: Identify Divergence Location
Run parity test to find first diverging stage:
```cpp
// Output: Divergence at layer 15, stage FFN_DOWN
```

### Step 2: Inspect Snapshots
```cpp
// Load both snapshots
auto llaminar_snapshot = loadSnapshot(PipelineStage::FFN_DOWN, 15);
auto pytorch_snapshot = loadPyTorchSnapshot("ffn_down_15.npy");

// Compute diff
auto diff = llaminar_snapshot - pytorch_snapshot;

// Analyze diff statistics
auto stats = computeStats(diff);
LOG_INFO("Diff stats: mean=" << stats.mean 
         << ", std=" << stats.std
         << ", max_abs=" << stats.max_abs
         << ", rel_l2=" << stats.rel_l2);

// Visualize diff heatmap (Python helper)
exportDiffHeatmap(diff, "ffn_down_15_diff.png");
```

### Step 3: Isolate Problematic Operation
Focus on the specific operation that diverged:
```cpp
// FFN_DOWN is the down projection: swiglu_out @ w_down
// Check inputs to this operation:

// 1. Check SwiGLU output (input to down projection)
compareSnapshot(PipelineStage::FFN_SWIGLU, 15, pytorch);
// If this matches, problem is in down projection

// 2. Check weight matrix
auto w_down = weights.w_down[15];
compareWeightsWithPyTorch(w_down, pytorch_weights.w_down[15]);
// If weights match, problem is in matmul execution

// 3. Test matmul in isolation
testMatMulIsolated(swiglu_out, w_down, expected_output);
```

### Step 4: Root Cause Analysis

Common divergence causes and fixes:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Early divergence (layer 0-2) | Weight loading issue | Check weight transpose, quantization |
| Gradual drift (increases with layer) | Numerical precision accumulation | Check float32 vs float16, epsilon values |
| Sudden spike at specific layer | Incorrect weight for that layer | Verify weight slice indexing |
| Divergence in attention only | RoPE position encoding issue | Check n_past, sequence position |
| Divergence in FFN only | SwiGLU or matmul issue | Test SwiGLU activation separately |

## Tolerance Calibration

Start with strict tolerances and relax if needed:

```cpp
struct ToleranceConfig {
    double max_abs_threshold = 0.01;      // Absolute error threshold
    double rel_l2_threshold = 0.001;      // Relative L2 norm threshold
    
    bool within_tolerance(double max_abs, double rel_l2) const {
        return max_abs < max_abs_threshold && rel_l2 < rel_l2_threshold;
    }
};

// Example tolerance evolution:
// Initial (strict):    max_abs=0.001, rel_l2=0.0001
// After layer 10:      max_abs=0.01,  rel_l2=0.001   (allow slight accumulation)
// After layer 20:      max_abs=0.05,  rel_l2=0.005   (more accumulation)
// Final output:        max_abs=0.1,   rel_l2=0.01    (reasonable for end result)
```

## Advanced Usage

### Capture Subset of Stages
```cpp
// Only capture attention stages (skip FFN to reduce overhead)
auto provider = std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);

// Override captureSnapshot to filter stages
class SelectiveOpenBLASProvider : public OpenBLASPrefillProvider {
    void captureSnapshot(PipelineStage stage, int layer_idx, 
                        const float* data, int seq_len, int feature_dim) override {
        // Only capture attention stages
        if (stage == PipelineStage::ATTENTION_NORM ||
            stage == PipelineStage::ATTENTION_OUTPUT ||
            stage == PipelineStage::ATTENTION_RESIDUAL) {
            OpenBLASPrefillProvider::captureSnapshot(stage, layer_idx, data, seq_len, feature_dim);
        }
    }
};
```

### Compare Multiple Backends
```cpp
TEST(MultiBackendParity, OpenBLASvsCOSMA) {
    // Execute with OpenBLAS
    auto openblas = std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
    openblas->execute(tokens, weights, output_openblas, ctx, metrics);
    
    // Execute with COSMA
    auto cosma = std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
    cosma->execute(tokens, weights, output_cosma, ctx, metrics);
    
    // Compare OpenBLAS vs COSMA stage-by-stage
    for (int layer = 0; layer < 28; ++layer) {
        auto openblas_snap = getSnapshot("OpenBLAS", PipelineStage::FFN_DOWN, layer);
        auto cosma_snap = getSnapshot("COSMA", PipelineStage::FFN_DOWN, layer);
        
        auto diff = computeDiff(openblas_snap, cosma_snap);
        LOG_INFO("Layer " << layer << " OpenBLAS vs COSMA: rel_l2=" << diff.rel_l2);
    }
}
```

## Performance Impact

### Debug Builds (snapshot capture enabled)
- **Memory overhead**: ~171 snapshots × 5 tokens × 896 floats × 4 bytes ≈ **3 MB**
- **Time overhead**: ~5-10% (snapshot copy operations)
- **Recommended**: Development and parity testing only

### Release Builds (snapshot capture compiled out)
- **Memory overhead**: 0 bytes (compiled out with `#ifdef NDEBUG`)
- **Time overhead**: 0% (compiler optimizes away empty functions)
- **Recommended**: Production inference

## Summary

✅ **Precise Divergence Detection** - Know exact layer and stage where divergence starts  
✅ **171 Comparison Points** - Comprehensive coverage of entire prefill pipeline  
✅ **Zero Production Overhead** - Compiled out in release builds  
✅ **PyTorch Reference** - Ground truth validation against reference implementation  
✅ **Incremental Debugging** - Fix divergences layer-by-layer, stage-by-stage  

This enables a **systematic debugging workflow**: detect → isolate → analyze → fix, rather than guessing where problems might be.
