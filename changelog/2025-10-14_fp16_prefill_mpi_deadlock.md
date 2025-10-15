# FP16 Prefill MPI Deadlock Investigation

**Date:** October 14, 2025  
**Author:** David Sanftenberg (via GitHub Copilot)  
**Issue:** MPI deadlock during FP16 prefill in RMSNorm computation  
**Status:** 🔍 INVESTIGATING

## Problem Summary

After fixing the OpenBLAS thread pool cleanup issue, discovered a **different hang** during FP16 prefill phase. The hang occurs during the first transformer layer's attention normalization.

## Symptoms

- **Hang location**: `MPIRMSNormKernel::computeDistributedRMSNorm` line 532 (MPI_Allreduce)
- **Test**: FP16 model (`qwen2.5-0.5b-instruct-fp16.gguf`)
- **Phase**: Prefill (5-token prompt)
- **Rank 0 state**: Stuck waiting in MPI_Allreduce for sum_sq statistics
- **Rank 1 state**: Unknown (no logs printed during FP16 prefill)

## Backtrace (Rank 0)

```
#0  opal_progress () from libopen-pal.so.40
#1  ompi_request_default_wait () from libmpi.so.40
#2  ompi_coll_base_sendrecv_actual () from libmpi.so.40
#3  ompi_coll_base_allreduce_intra_recursivedoubling () from libmpi.so.40
#4  PMPI_Allreduce () from libmpi.so.40
#5  PerfAllreduce (count=1, MPI_DOUBLE, MPI_SUM) at perf_counters.h:391
#6  MPIRMSNormKernel::computeDistributedRMSNorm (local_seq_len=3, hidden_size=896, global_seq_len=5)
    at MPIRMSNormKernel.cpp:532
#7  MPIRMSNormKernel::execute (inputs, outputs) at MPIRMSNormKernel.cpp:309
#8  PrefillProviderBaseImpl::executeKernel (kernel_name="rmsnorm") at prefill_provider_base_impl.cpp:397
#9  OpenBLASPrefillProvider::executeAttentionBlock (layer_idx=0) at openblas_prefill_provider.cpp:206
#10 PrefillProviderBaseImpl::executeTransformerLayer (layer_idx=0) at prefill_provider_base_impl.cpp:187
#11 PrefillProviderBaseImpl::execute (tokens=[5 tokens]) at prefill_provider_base_impl.cpp:81
```

## Analysis

### Sequence Distribution
- Prompt length: 5 tokens
- Rank 0: `local_seq_len=3`, `offset=0` ✅ (logged)
- Rank 1: Unknown (no logs) ❌

### Code Path
1. `MPIRMSNormKernel::execute` determines `feature_sharded=false`
2. Calls `distributeInput(global_input, local_input, global_seq_len=5, hidden_size=896)`
3. Calls `computeDistributedRMSNorm(local_input, ..., local_seq_len, hidden_size, global_seq_len=5)`
4. Inside `computeDistributedRMSNorm`, **unconditional** MPI_Allreduce at line 532:
   ```cpp
   checkMPIError(PerfAllreduce(&local_sum_sq_total, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, getComm()),
                 "MPI_Allreduce for RMS statistics");
   ```

### Root Cause Hypothesis

**MPI collective operations must be called by ALL ranks**. If rank 1 doesn't reach the `PerfAllreduce` call, rank 0 will deadlock.

Possible causes:
1. **Rank 1 crashed/exception** before reaching Allreduce
2. **Rank 1 took different code path** (e.g., `feature_sharded=true` while rank 0 has `false`)
3. **Rank 1 stuck in `distributeInput`** (MPI communication deadlock earlier)
4. **Asymmetric tensor distribution** causing rank 1 to early-return or skip the computation

## Differences from OpenBLAS Hang

| Aspect | OpenBLAS Thread Hang | MPI Prefill Deadlock |
|--------|---------------------|---------------------|
| **Phase** | Decode (autoregressive) | Prefill (initial prompt) |
| **Location** | Final LM head matmul | First layer attn_norm RMSNorm |
| **Cause** | Thread pool state corruption | MPI collective synchronization |
| **Fix** | SetUp/TearDown cleanup | TBD |
| **Models affected** | FP16/FP32 sequential tests | FP16 standalone |

## Evidence

### Gemini FP32 Test (Working)
- Decode phase: Rank 1 has `local_seq_len=0` (OK, participates in Allreduce with 0 contribution)
- Test completes successfully (32s)

### FP16 Test (Hanging)
- Prefill phase: Rank 0 has `local_seq_len=3`
- Prefill phase: Rank 1 has **no logs** (never reached the logged code)
- Hangs indefinitely at first Allreduce

## Next Steps

1. **Add instrumentation** to `distributeInput` to see if rank 1 reaches it
2. **Check MPI_Scatterv** in `distributeInput` - might be deadlocking
3. **Verify row distribution logic** - `getRowDistribution(5)` should return `(3,0)` for rank 0, `(2,3)` for rank 1
4. **Add barrier before `computeDistributedRMSNorm`** to ensure both ranks reach it
5. **Check for exceptions/crashes** in rank 1 MPI process

## Workaround

For now, **skip FP16/FP32 models in automated parity tests** or **run them without prefill** (decode-only mode).

## Related Issues

- ✅ OpenBLAS thread cleanup fixed (separate issue)
- ❓ Why does FP32 Gemini work but FP16 Qwen hangs?
- ❓ Does Q8_0 (also unquantized-like) have similar issues?
