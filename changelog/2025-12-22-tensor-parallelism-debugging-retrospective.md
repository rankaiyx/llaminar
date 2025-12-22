# Tensor Parallelism Debugging Retrospective

**Date**: December 22, 2025  
**Feature**: Tensor Parallelism (TP) for 2-rank MPI execution  
**Status**: ✅ Working - Single-rank and 2-rank produce identical output

## Executive Summary

This document captures the debugging journey to get tensor parallelism working end-to-end, including all issues encountered, their root causes, and architectural recommendations to prevent similar issues in the future.

---

## Issues Encountered

### Issue 1: Bias Extraction from TensorSlice (Silent Data Corruption)

**Symptom**: Incorrect inference output with TP enabled, but no crash or error.

**Root Cause**: When extracting bias for sharded attention projections, the code attempted to use `dynamic_cast<FP32Tensor*>` on a `TensorSlice` object, which returned `nullptr`. The null check was missing, so no bias was passed to the GEMM.

**Location**: `src/v2/pipelines/qwen/Qwen2Graph.cpp` (lines ~940-960) and `src/v2/pipelines/qwen/Qwen2Pipeline.cpp`

**Original Code Pattern**:
```cpp
// WRONG: TensorSlice is not an FP32Tensor, dynamic_cast returns nullptr
auto* bias_fp32 = dynamic_cast<FP32Tensor*>(bias_slice.get());
const float* bias_data = bias_fp32 ? bias_fp32->data() : nullptr;
// bias_data is always nullptr for sliced tensors!
```

**Fix**:
```cpp
// CORRECT: TensorSlice exposes data() directly
const float* bias_data = bias_slice ? bias_slice->data() : nullptr;
```

**Why This Was Hard to Debug**:
- No crash, no error message
- Output looked "reasonable" but was numerically wrong
- Required comparing intermediate activations between single-rank and 2-rank

---

### Issue 2: Segfault from Debug Logging (Use-After-Move)

**Symptom**: Segmentation fault during inference with debug logging enabled.

**Root Cause**: Debug logging code was accessing tensor data after the tensor had been moved/released:

```cpp
// In ComputeStage.cpp around lines 589-617
LOG_INFO("[FusedQKVGEMMStage] Q output[0:8]=" << formatFloatArray(q_output->data(), 8));
// ^ q_output may have been invalidated by buffer aliasing
```

**Fix**: Removed the debug logging code that accessed potentially-invalidated tensors.

**Why This Was Hard to Debug**:
- Only manifested with DEBUG log level enabled
- Segfault location didn't clearly indicate the root cause
- Required GDB backtrace to identify the access pattern

---

### Issue 3: Missing Bias in FusedQKVGEMMStage GEMM Call

**Symptom**: Q/K/V projections producing incorrect values.

**Root Cause**: The `FusedQKVGEMMStage` was calling the GEMM kernel without passing the bias tensor, even when bias was available:

```cpp
// WRONG: bias parameter missing
kernel->multiply(input, output, m, n, k);

// CORRECT: bias parameter included
kernel->multiply(input, output, m, n, k, 1.0f, 0.0f, bias_data);
```

**Why This Was Hard to Debug**:
- GEMM succeeded without error
- Output shape was correct
- Only numerical comparison revealed the issue

---

### Issue 4: False Alarm - Apparent Output Divergence

**Symptom**: Single-rank and 2-rank appeared to produce different text outputs.

**Root Cause**: Multiple confounding factors:

1. **Different prompts used in different test runs** - Copy/paste errors led to comparing outputs from different prompts
2. **Non-deterministic sampling** - Without `-t 0` (greedy sampling), temperature-based sampling produces different outputs each run
3. **Logging noise** - Verbose INFO-level logging obscured the actual generated text

**Resolution**: Systematic testing with:
- Consistent prompts
- Greedy sampling (`-t 0`)
- ERROR log level for clean output
- Token-by-token comparison using `grep "Sampled token"`

---

### Issue 5: AllReduce Verification Difficulty

**Symptom**: Uncertainty about whether AllReduce was actually being called.

**Root Cause**: No clear logging indicating AllReduce execution and results.

**Resolution**: Added temporary logging to `AllreduceStage::execute()`:
```cpp
LOG_INFO("[AllreduceStage] Execute: buffer=" << buffer_ << " count=" << count_ 
         << " has_comm=" << (comm_ != MPI_COMM_NULL));
int result = MPI_Allreduce(MPI_IN_PLACE, buffer_, count_, MPI_FLOAT, MPI_SUM, comm_);
LOG_INFO("[AllreduceStage] MPI_Allreduce returned " << result);
```

---

## Architectural Recommendations

### 1. Unified Tensor Data Access Interface

**Problem**: `TensorBase`, `FP32Tensor`, `TensorSlice`, and quantized tensors all have different ways to access raw data. This leads to fragile `dynamic_cast` chains.

**Recommendation**: Add a universal `data()` method to `TensorBase`:

```cpp
class TensorBase {
public:
    // Returns pointer to raw data (FP32 for activation tensors, native format for quantized)
    virtual const void* raw_data() const = 0;
    
    // Returns FP32 data pointer (dequantizes on-demand for quantized tensors, or nullptr)
    virtual const float* fp32_data() const { return nullptr; }
    
    // For activation tensors that are always FP32
    virtual float* mutable_fp32_data() { return nullptr; }
};

class FP32Tensor : public TensorBase {
    const float* fp32_data() const override { return data_; }
    float* mutable_fp32_data() override { return data_; }
};

class TensorSlice : public TensorBase {
    const float* fp32_data() const override { return slice_data_; }  // Already FP32
};
```

**Benefit**: Eliminates `dynamic_cast` usage for data access, making bugs compile-time errors instead of runtime silent failures.

---

### 2. Explicit Bias Handling in GEMM Stages

**Problem**: Bias is optionally passed to GEMM, and it's easy to forget or mishandle it.

**Recommendation**: Make bias explicit in stage construction:

```cpp
struct GEMMStageConfig {
    TensorBase* weight;           // Required
    TensorBase* bias;             // Optional, nullptr if not used
    bool requires_bias;           // If true, assert bias != nullptr
    std::string debug_name;       // For logging
};

class GEMMStage : public ComputeStage {
public:
    GEMMStage(GEMMStageConfig config) {
        if (config.requires_bias && !config.bias) {
            throw std::runtime_error("GEMM stage '" + config.debug_name + "' requires bias but none provided");
        }
    }
};
```

**Benefit**: Fails fast with clear error message instead of producing silently wrong results.

---

### 3. Stage Execution Tracing Infrastructure

**Problem**: Debugging required adding ad-hoc logging, then removing it.

**Recommendation**: Built-in tracing infrastructure controlled by environment variable:

```cpp
// In ComputeStage.h
class ComputeStage {
protected:
    void traceInput(const std::string& name, const float* data, size_t count) {
        if (debugEnv().execution.trace_stages) {
            LOG_TRACE("[" << stageName() << "] " << name << "[0:8]=" 
                      << formatFloatArray(data, std::min(count, 8UL)));
        }
    }
    
    void traceOutput(const std::string& name, const float* data, size_t count) {
        if (debugEnv().execution.trace_stages) {
            LOG_TRACE("[" << stageName() << "] " << name << "[0:8]=" 
                      << formatFloatArray(data, std::min(count, 8UL)));
        }
    }
};

// Usage in stage implementations
void FusedQKVGEMMStage::execute() {
    traceInput("input", input_->fp32_data(), input_->numel());
    // ... do work ...
    traceOutput("Q", q_output_->fp32_data(), q_output_->numel());
    traceOutput("K", k_output_->fp32_data(), k_output_->numel());
    traceOutput("V", v_output_->fp32_data(), v_output_->numel());
}
```

**Environment Variable**: `LLAMINAR_TRACE_STAGES=1`

**Benefit**: Always-available debugging without code changes. Safe to leave in production (zero cost when disabled).

---

### 4. Tensor Validity Assertions

**Problem**: Use-after-move/use-after-free bugs are silent until they segfault.

**Recommendation**: Debug-mode validity tracking:

```cpp
class TensorBase {
#ifndef NDEBUG
    mutable bool valid_ = true;
    
    void invalidate() { valid_ = false; }
    void assertValid() const {
        assert(valid_ && "Accessing invalidated tensor");
    }
#else
    void invalidate() {}
    void assertValid() const {}
#endif

public:
    const float* data() const {
        assertValid();
        return data_;
    }
};

// In buffer manager when aliasing/reusing buffers
void BufferManager::releaseBuffer(TensorBase* tensor) {
    tensor->invalidate();
    // ... return to pool ...
}
```

**Benefit**: Catches use-after-invalidation bugs immediately with clear assertion message.

---

### 5. Deterministic Testing Mode

**Problem**: Non-deterministic sampling makes debugging difficult.

**Recommendation**: Add `--deterministic` flag that:
- Forces temperature=0 (greedy sampling)
- Seeds RNG with fixed value
- Disables any stochastic operations

```cpp
// In ArgParser
if (args.deterministic) {
    args.temperature = 0.0f;
    args.seed = 42;
    LOG_INFO("Deterministic mode: temperature=0, seed=42");
}
```

**Benefit**: Reproducible outputs for debugging and testing.

---

### 6. MPI Operation Logging

**Problem**: Hard to verify MPI collective operations are happening correctly.

**Recommendation**: Structured MPI logging in DebugEnv:

```cpp
// In DebugEnv.h
struct MPIConfig {
    bool log_collectives = false;      // LLAMINAR_MPI_LOG_COLLECTIVES
    bool verify_allreduce = false;     // LLAMINAR_MPI_VERIFY_ALLREDUCE
};

// In AllreduceStage::execute()
if (debugEnv().mpi.log_collectives) {
    LOG_DEBUG("[MPI] AllReduce: count=" << count_ << " buffer=" << buffer_);
}

if (debugEnv().mpi.verify_allreduce) {
    // Compute local checksum before
    float sum_before = std::accumulate(buffer_, buffer_ + count_, 0.0f);
    
    MPI_Allreduce(MPI_IN_PLACE, buffer_, count_, MPI_FLOAT, MPI_SUM, comm_);
    
    // Compute checksum after
    float sum_after = std::accumulate(buffer_, buffer_ + count_, 0.0f);
    LOG_DEBUG("[MPI] AllReduce verified: sum_before=" << sum_before 
              << " sum_after=" << sum_after);
}
```

**Benefit**: Easy verification of MPI correctness without code changes.

---

### 7. Tensor Shape Validation at Stage Boundaries

**Problem**: Shape mismatches between stages cause subtle bugs.

**Recommendation**: Explicit shape contracts:

```cpp
class ComputeStage {
protected:
    struct TensorContract {
        std::string name;
        std::vector<size_t> expected_shape;
        TensorBase* tensor;
    };
    
    void validateInputs(std::initializer_list<TensorContract> contracts) {
        for (const auto& c : contracts) {
            if (c.tensor->shape() != c.expected_shape) {
                throw std::runtime_error(
                    stageName() + ": " + c.name + " shape mismatch. "
                    "Expected " + shapeToString(c.expected_shape) + 
                    " got " + shapeToString(c.tensor->shape())
                );
            }
        }
    }
};

// Usage
void FusedQKVGEMMStage::execute() {
    validateInputs({
        {"input", {batch_size_, d_model_}, input_},
        {"wq", {local_n_heads_ * head_dim_, d_model_}, wq_},
        // ...
    });
}
```

**Benefit**: Early, clear errors instead of garbage output.

---

### 8. Comparison Testing Infrastructure

**Problem**: Had to manually compare single-rank vs 2-rank outputs.

**Recommendation**: Built-in comparison mode:

```bash
# Run comparison test
./run_llaminar.sh --compare-ranks \
    -m model.gguf \
    -p "Test prompt" \
    -n 20 \
    --reference-ranks 1 \
    --test-ranks 2
```

Implementation:
1. Run inference with 1 rank, capture token sequence
2. Run inference with 2 ranks, capture token sequence  
3. Compare and report differences

**Benefit**: One-command verification of TP correctness.

---

## Testing Checklist for Future TP Changes

1. ✅ Run with single rank, capture output
2. ✅ Run with 2 ranks, capture output
3. ✅ Compare token-by-token (use `grep "Sampled token"` with DEBUG logging)
4. ✅ Verify AllReduce is called (check logs for MPI_Allreduce)
5. ✅ Test with different sequence lengths (1, 10, 100 tokens)
6. ✅ Test with different batch sizes
7. ✅ Always use greedy sampling (`-t 0`) for deterministic comparison

---

## Files Modified During This Debug Session

| File | Change |
|------|--------|
| `src/v2/pipelines/qwen/Qwen2Graph.cpp` | Fixed bias extraction from TensorSlice |
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | Fixed bias extraction from TensorSlice |
| `src/v2/execution/ComputeStage.cpp` | Removed debug logging causing segfault |
| `src/v2/execution/ComputeStage.cpp` | Added bias to GEMM call in FusedQKVGEMMStage |

---

## Conclusion

The tensor parallelism implementation is now verified working. The key lessons learned:

1. **Type safety matters** - Using `dynamic_cast` for data access is fragile
2. **Fail fast** - Silent failures are the worst kind of bug
3. **Built-in tracing** - Debug infrastructure should be permanent, not ad-hoc
4. **Deterministic testing** - Non-deterministic behavior makes debugging exponentially harder
5. **Comparison testing** - Automated comparison between configurations catches regressions

Implementing the architectural recommendations above would significantly reduce debugging time for future issues.
