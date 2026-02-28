# GPU-Side Argmax Decode Sampling

**Date**: 2026-02-28  
**Branch**: `tensor-parallel`  
**Impact**: TP=2 decode throughput **91.74 → 95.53 tok/s** (7B Q8_0, 20-token), **122.4% TP efficiency** (exceeds 120% target)

## Summary

Eliminated two remaining per-token overheads in the TP decode path:
1. **D2H logits gather** (~286 µs/token): 608 KB transfer from 2 GPUs → host combined buffer
2. **CPU argmax scan** (~170 µs/token): `std::max_element` over 152,064 floats

Replaced them with a single HIP argmax kernel that runs entirely on-device, transferring only 8 bytes (float + int) per device per token.

## Performance Results

### 20-token decode (The capital of France is):
| Config | Before | After | Improvement |
|--------|--------|-------|-------------|
| TP=2 decode | 91.74 tok/s | 95.53 tok/s | +4.1% |
| Single GPU | 78.02 tok/s | 78.02 tok/s | (unchanged) |
| TP efficiency | 116.9% | **122.4%** | +5.5pp |

### 128-token decode (default prompt):
| Config | Before | After |
|--------|--------|-------|
| TP=2 decode | ~85 tok/s | 87.48 tok/s |
| Single GPU | 74.31 tok/s | 74.31 tok/s |
| TP efficiency | ~114% | **117.7%** |

## Code Changes

### New Files
- **`src/v2/kernels/rocm/ops/ROCmArgmaxKernels.hip`**: HIP argmax kernel using single-block 1024-thread shared memory reduction. For vocab_local ~76K, each thread scans ~74 elements, then tree reduction. Returns (max_value, max_index) via device memory.

### Modified Files (8 files)

- **`src/v2/CMakeLists.txt`**: Added ROCmArgmaxKernels.hip to HIP sources.

- **`src/v2/backends/IBackend.h`**: Added `argmaxF32()` virtual method (default returns false).

- **`src/v2/backends/rocm/ROCmBackend.h`**: Added `argmaxF32()` override with lazy per-device result buffer allocation (`ArgmaxDeviceBuffers` struct).

- **`src/v2/backends/rocm/ROCmBackend.cpp`**: Implemented `argmaxF32()` — lazy 8-byte device buffer alloc, kernel launch on default stream, `hipDeviceSynchronize()`, 8-byte D2H copy.

- **`src/v2/execution/local_execution/orchestrators/IInferenceRunner.h`**: Added `sampleGreedyOnDevice()` (returns -1 default) and `setSkipLogitsGatherDecode(bool)` (no-op default).

- **`src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h/.cpp`**: 
  - `sampleGreedyOnDevice()`: Collects per-device GPU logits pointers, calls `argmaxF32()` on each, picks global winner.
  - `setSkipLogitsGatherDecode(bool)`: Sets flag to skip `gatherLogits()` during decode.
  - `forwardTP()`: Conditional `gatherLogits` skip when flag set and seq_len==1.

- **`src/v2/execution/runner/IOrchestrationRunner.h`**: Added `sampleGreedyOnDevice()` and `setSkipLogitsGatherDecode()` with defaults.

- **`src/v2/execution/runner/OrchestrationRunner.h/.cpp`**: Forwarding implementations to `runner_`.

- **`src/v2/Main.cpp`**: Added `sampleGreedyOnDevice()` and `setSkipLogitsGatherDecode()` to `BenchmarkRunnerAdapter`.

- **`src/v2/utils/BenchmarkRunner.cpp`**: 
  - `run()`: Calls `setSkipLogitsGatherDecode(true)` before warmup.
  - `runDecode()`: Tries `sampleGreedyOnDevice()` first; falls back to CPU path if returns -1.

## Design Decisions

1. **Single-block kernel**: For 76K elements, a single block with 1024 threads is sufficient (74 elements/thread + tree reduction). No need for multi-block reduction complexity.

2. **Fallback to CPU**: `sampleGreedyOnDevice()` returns -1 on any failure, triggering the existing CPU sampling path. This ensures robustness for non-TP runners or CUDA backends that don't implement argmax.

3. **hipDeviceSynchronize**: Used for correctness — ensures graph replay + argmax kernel are complete before reading results. The second device sync is nearly free since device 1 runs concurrently during device 0 sync.

4. **Benchmark-only enablement**: `setSkipLogitsGatherDecode(true)` is set only by BenchmarkRunner. Interactive/streaming use cases still use the full gatherLogits path since they need host-side logits for temperature/top-k sampling.

## Overhead Breakdown (per decode token, TP=2)

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| D2H logits gather | 286 µs | 0 µs (skipped) | 286 µs |
| CPU argmax scan | 170 µs | 0 µs (eliminated) | 170 µs |
| GPU argmax kernel | - | ~10 µs | - |
| D2H result (8 bytes × 2 devices) | - | ~5 µs | - |
| hipDeviceSynchronize (2 devices) | - | ~30 µs | - |
| **Net savings** | **456 µs** | **~45 µs** | **~411 µs** |

## Next Steps

- Consider extending GPU-side sampling to top-k/top-p for non-benchmark paths
- Investigate prefill TP efficiency (currently 25-40% vs single GPU)
- Profile remaining per-token overhead sources
