# MPIAttentionOrchestrator ⇄ CpuAttentionKernelT Refactor Plan

## Goals

1. **Single Source of Truth for CPU Attention:**
   - Eliminate duplicated attention pipelines across `GQAAttention` and `MpiAttentionOrchestrator`.
   - Make `CpuAttentionKernelT<T>` (via `ITensorAttention`) the canonical implementation for CPU attention math (Q·K^T, scaling, masking, softmax, scores·V).

2. **Preserve Existing MPI Responsibilities:**
   - Keep `MpiAttentionOrchestrator` responsible for:
     - MPI strategy dispatch (`None`, `TensorParallel`, `SequenceParallel` (future), etc.).
     - Rank-local head partitioning (`get_local_slice`), and MPI collectives (`allreduce_sum`, barriers, staging).
   - Ensure behavior and logging semantics remain consistent where possible.

3. **Improve Reuse Across Single-Rank and MPI Paths:**
   - `GQAAttention` becomes a reusable single-rank attention front-end that always calls `ITensorAttention` under the hood.
   - `MpiAttentionOrchestrator::compute` delegates to `GQAAttention::compute` for non-MPI / single-rank paths.
   - `MpiAttentionOrchestrator::compute_tensor_parallel` uses the same kernel-backed per-rank path, with MPI-specific slicing and allreduce around it.

---

## Current State Overview (Before Refactor)

### 1. CPU Attention Kernel

**Files:**
- `src/v2/kernels/cpu/CpuAttentionKernelT.h`
- `src/v2/tensors/TensorKernels.h` (`ITensorAttention` interface)
- `src/v2/tensors/FP32Tensor.cpp`, `BF16Tensor.cpp`, `FP16Tensor.cpp`, etc.

**Key Points:**
- `ITensorAttention` defines:
  - `compute(...)` for single-sequence attention:
    - Inputs: `const float *Q, *K, *V`, `float *output`, shape params, masking, workspaces, `mpi_ctx`, `device_idx`.
  - `compute_batch(...)` for batched attention (currently available, used for future expansion).
- `CpuAttentionKernelT<TensorType>` implements `ITensorAttention`.
  - Internally handles:
    - Precision conversion (BF16/FP16/Q8_1/INT32/Q8_0 → FP32) where needed.
    - K/V broadcasting (GQA/MQA) where appropriate.
    - Q·K^T GEMM, scaling, masking, softmax, scores·V GEMM.
- Activation tensors wire into this kernel via `createAttention()`:
  - `FP32Tensor::createAttention()` → `std::make_unique<CpuAttentionKernelT<FP32Tensor>>()`.
  - `BF16Tensor::createAttention()` → `CpuAttentionKernelT<BF16Tensor>`.
  - `FP16Tensor::createAttention()` → `CpuAttentionKernelT<FP16Tensor>`.
  - `Q8_1Tensor::createAttention()` currently maps to `CPUAttention` (to be aligned later if needed).

### 2. GQAAttention (Single-Rank Attention Front-End)

**Files:**
- `src/v2/pipelines/attention/GQAAttention.h`
- `src/v2/pipelines/attention/GQAAttention.cpp`

**Responsibilities Today:**
- Implements attention for non-MPI paths (single rank) with GQA/MHA/MQA semantics.
- Handles:
  - Input validation (`validate_inputs`).
  - K/V broadcasting via `broadcast_kv_heads_if_needed`.
  - Manual per-head pipeline using helpers:
    - `extract_head_data`
    - `compute_attention_scores`
    - `scale_scores_inplace`
    - `apply_attention_mask`
    - `apply_softmax`
    - `compute_context_from_scores`
    - `write_context_to_output`
- Uses workspaces from `GQAAttentionConfig`:
  - `workspace_scores`, `workspace_qkv_buffer`, `workspace_context`, `workspace_mask`.

### 3. MpiAttentionOrchestrator (MPI-Aware Front-End)

**Files:**
- `src/v2/pipelines/attention/MpiAttentionOrchestrator.h`
- `src/v2/pipelines/attention/MpiAttentionOrchestrator.cpp`

**Current Responsibilities:**
- Defines `MpiAttentionConfig` (mirrors many fields from `GQAAttentionConfig` plus MPI-specific ones).
- Provides methods:
  - `compute(...)` — single-rank implementation; largely parallel to `GQAAttention::compute`.
  - `compute_batch(...)` — batched attention with padding/masking; similar to `GQAAttention::compute_batch`.
  - `compute_mpi(...)` — dispatch to appropriate MPI strategy:
    - `MPIStrategy::None` or 1 rank → `compute(...)`.
    - `MPIStrategy::TensorParallel` → `compute_tensor_parallel(...)`.
    - `MPIStrategy::SequenceParallel`, `Hybrid` → TODO / not yet implemented.
  - `compute_tensor_parallel(...)` — Tensor-parallel attention:
    - Validates MPI context, head divisibility.
    - Uses `mpi_ctx->get_local_slice` to compute `(start_head, local_n_heads)`.
    - Broadcasts K/V if needed, into local `K_broadcast` / `V_broadcast`.
    - For each local head:
      - Extracts `Q_h`, `K_h` with `extract_head_data`.
      - Calls `compute_attention_scores` to build local `scores`.
    - Scales scores, applies causal/padding masks, runs softmax.
    - For each local head:
      - Extracts `V_h`, computes context via `compute_context_from_scores`.
      - Writes context into `local_output` (rank-local [tokens, local_heads, head_dim]).
    - Packs `local_output` into a full-head `send_buffer` and runs `allreduce_sum` to produce the final `output`.
- **Note:** `compute` and `compute_tensor_parallel` both reimplement attention math independently from `CpuAttentionKernelT<T>`.

---

## Target Architecture (After Refactor)

### High-Level Design

1. **`ITensorAttention` / `CpuAttentionKernelT<T>` own all CPU attention math.**
   - No direct GEMM/masking/softmax calls in `GQAAttention` or `MpiAttentionOrchestrator`.

2. **`GQAAttention` becomes a slim front-end:**
   - Performs input validation, K/V broadcasting, workspace wiring.
   - Delegates math to `ITensorAttention::compute` / `compute_batch` via `output`'s `createAttention()`.

3. **`MpiAttentionOrchestrator` uses `GQAAttention` and the kernel:**
   - `compute(...)` → constructs a `GQAAttentionConfig` and calls `GQAAttention::compute` (single-rank path).
   - `compute_batch(...)` → constructs `GQAAttentionConfig` and calls `GQAAttention::compute_batch`.
   - `compute_mpi(...)` → dispatch stays the same, but the bodies of `compute`/`compute_tensor_parallel` are kernel-backed.
   - `compute_tensor_parallel(...)`:
     - Keeps MPI-specific logic (head partitioning, collectives).
     - Delegates rank-local attention math to `ITensorAttention::compute` over rank-local Q/K/V slices.

---

## Step-by-Step Refactor Plan

### Step 1: Introduce a Unified Helper for Kernel-Based Attention in GQAAttention

**Goal:** Centralize the `ITensorAttention` call in `GQAAttention` for single-rank paths.

1. **In `GQAAttention.cpp`, after validation and K/V broadcasting in `compute(...)`:**
   - Remove / disable the manual per-head blocks (`#pragma omp parallel` loops, `compute_attention_scores`, `scale_scores_inplace`, `apply_attention_mask`, `apply_softmax`, `compute_context_from_scores`).
   - Introduce a small internal helper (static, file-local) or inline code that:

     ```cpp
     // Pseudocode
     auto *act_out = dynamic_cast<IActivationTensor *>(output);
     if (!act_out) {
         LOG_ERROR("[GQAAttention] output tensor does not implement IActivationTensor");
         return false;
     }

     auto kernel = act_out->createAttention();
     if (!kernel) {
         LOG_ERROR("[GQAAttention] createAttention() returned null");
         return false;
     }

     const float *Q_ptr = Q->data();
     const float *K_ptr = K_expanded; // broadcasted if needed
     const float *V_ptr = V_expanded;
     float *out_ptr = output->mutable_data();

     return kernel->compute(
         Q_ptr,
         K_ptr,
         V_ptr,
         out_ptr,
         /*seq_len=*/seq_len,
         /*n_heads=*/config.n_heads,
         /*n_kv_heads=*/config.n_kv_heads,
         /*head_dim=*/config.head_dim,
         /*causal=*/config.causal || sequence_lengths != nullptr,
         /*window_size=*/config.window_size,
         config.workspace_scores.get(),
         config.workspace_qkv_buffer.get(),
         config.workspace_context.get(),
         config.workspace_mask.get(),
         /*use_bf16=*/(config.precision == ActivationPrecision::BF16),
         /*mpi_ctx=*/nullptr,
         /*device_idx=*/-1);
     ```

2. **In `GQAAttention::compute_batch(...)`:**
   - After `broadcast_kv_heads_if_needed` and construction of the combined batch mask, remove the manual `#pragma omp parallel` loops that directly compute scores, apply masking, and compute contexts.
   - Use an analogous invocation of `ITensorAttention::compute_batch`:

     ```cpp
     auto *act_out = dynamic_cast<IActivationTensor *>(output);
     auto kernel = act_out->createAttention();

     const float *Q_ptr = Q->data();
     const float *K_ptr = K_expanded;
     const float *V_ptr = V_expanded;
     float *out_ptr = output->mutable_data();

     return kernel->compute_batch(
         Q_ptr, K_ptr, V_ptr, out_ptr,
         batch_size, seq_len,
         config.n_heads, config.n_kv_heads,
         config.head_dim,
         /*causal=*/config.causal,
         /*window_size=*/config.window_size,
         config.workspace_scores.get(),
         config.workspace_qkv_buffer.get(),
         config.workspace_context.get(),
         config.workspace_mask.get(),
         /*use_bf16=*/(config.precision == ActivationPrecision::BF16),
         /*mpi_ctx=*/nullptr,
         /*device_idx=*/-1);
     ```

3. **Adjust Tests (if any depend on internal helpers):**
   - If unit tests for `GQAAttention` currently inspect intermediate buffers (`scores`, `mask`, etc.), update them to only validate final outputs or to go through the kernel instead of internal helper functions.

**Outcome:** All single-rank attention invoked via `GQAAttention` now runs through `CpuAttentionKernelT<T>`.

---

### Step 2: Make `MpiAttentionOrchestrator::compute` / `compute_batch` Call GQAAttention

**Goal:** Avoid duplicating non-MPI attention logic in `MpiAttentionOrchestrator`.

1. **In `MpiAttentionOrchestrator::compute(...)`:**
   - After calling `validate_inputs` and extracting `seq_len`, construct a `GQAAttentionConfig` from `MpiAttentionConfig`:

     ```cpp
     GQAAttentionConfig gqa_cfg;
     gqa_cfg.n_heads = config.n_heads;
     gqa_cfg.n_kv_heads = config.n_kv_heads;
     gqa_cfg.head_dim = config.head_dim;
     gqa_cfg.causal = config.causal;
     gqa_cfg.window_size = config.window_size;
     gqa_cfg.precision = config.precision;
     gqa_cfg.mpi_ctx = config.mpi_ctx;            // can be nullptr for single-rank
     gqa_cfg.mpi_strategy = MPIStrategy::None;    // force single-rank behavior
     gqa_cfg.verbose_logging = config.verbose_logging;

     gqa_cfg.workspace_scores = config.workspace_scores;
     gqa_cfg.workspace_qkv_buffer = config.workspace_qkv_buffer;
     gqa_cfg.workspace_context = config.workspace_context;
     gqa_cfg.workspace_mask = config.workspace_mask;
     ```

   - Replace the manual attention pipeline in `MpiAttentionOrchestrator::compute` with:

     ```cpp
     return GQAAttention::compute(Q, K, V, output, gqa_cfg, batch_size, sequence_lengths);
     ```

2. **In `MpiAttentionOrchestrator::compute_batch(...)`:**
   - Similarly, map `MpiAttentionConfig` → `GQAAttentionConfig`.
   - Delegate to `GQAAttention::compute_batch`:

     ```cpp
     return GQAAttention::compute_batch(Q, K, V, output, actual_lengths, batch_size, seq_len, gqa_cfg);
     ```

3. **Cleanup / Documentation:**
   - Remove or deprecate the now-unused internal helper calls in `MpiAttentionOrchestrator` that duplicate `GQAAttention` behavior.
   - Update docstrings in `MpiAttentionOrchestrator.h` to reflect that `compute` uses `GQAAttention` and `ITensorAttention` under the hood.

**Outcome:** All non-MPI attention paths through `MpiAttentionOrchestrator` now reuse `GQAAttention` and the CPU kernel, with no duplicated math.

---

### Step 3: Refactor `compute_mpi` to Keep Dispatch but Use Kernel-Backed Implementations

**Goal:** Ensure MPI dispatcher remains the same conceptually but now routes into kernel-backed implementations.

1. **`MpiAttentionOrchestrator::compute_mpi(...)` remains the main switch:**
   - Keep the early-exit fast path:

     ```cpp
     if (!config.mpi_ctx || config.mpi_ctx->world_size() == 1 || config.mpi_strategy == MPIStrategy::None) {
         return compute(Q, K, V, output, config, batch_size, sequence_lengths);
     }
     ```

   - Keep the MPI strategy dispatch as-is, but rely on refactored `compute` / `compute_tensor_parallel`:

     ```cpp
     switch (config.mpi_strategy) {
     case MPIStrategy::TensorParallel:
         return compute_tensor_parallel(Q, K, V, output, config, batch_size, sequence_lengths);
     case MPIStrategy::SequenceParallel:
         // TODO
     case MPIStrategy::PipelineParallel:
         return compute(Q, K, V, output, config, batch_size, sequence_lengths);
     case MPIStrategy::Hybrid:
         // TODO
     default:
         // Error
     }
     ```

2. **No change in surface API:**
   - The function signatures and expected behaviors remain identical.
   - Internal math will now be driven by `CpuAttentionKernelT<T>` via `GQAAttention` or via rank-local kernel calls.

**Outcome:** MPI strategy routing remains stable; internals are simplified and unified.

---

### Step 4: Implement Rank-Local Kernel-Based Path in `compute_tensor_parallel`

**Goal:** Replace manual Q·K^T/mask/softmax/scores·V in `compute_tensor_parallel` with a rank-local `ITensorAttention::compute` call, while preserving MPI slicing + allreduce behavior.

1. **Retain MPI-Specific Logic:**
   - Keep:
     - MPI context validation.
     - Head divisibility checks (`config.n_heads % world_size == 0`).
     - `get_local_slice` to obtain `(start_head, local_n_heads)`.
     - `broadcast_kv_heads_if_needed` to create `K_expanded`, `V_expanded`.
     - Allreduce staging (`send_buffer`, `allreduce_sum`, optional GPU staging via `MPIStager`).

2. **Replace Manual Score/Context Computation with Kernel Calls:**
   - After broadcasting K/V and computing `total_tokens` and `seq_len`:

     **(a) Materialize Rank-Local Q/K/V Slices:**

     - Allocate contiguous FP32 buffers for rank-local heads:

       ```cpp
       std::vector<float> Q_local(total_tokens * local_n_heads * config.head_dim);
       std::vector<float> K_local(total_tokens * local_n_heads * config.head_dim);
       std::vector<float> V_local(total_tokens * local_n_heads * config.head_dim);
       std::vector<float> out_local(total_tokens * local_n_heads * config.head_dim);
       ```

     - Use existing `extract_head_data` to pack each local head into `Q_local`/`K_local`/`V_local`:

       ```cpp
       for (size_t local_h = 0; local_h < local_n_heads; ++local_h) {
           size_t global_h = start_head + local_h;

           float *Q_h = Q_local.data() + local_h * total_tokens * config.head_dim;
           float *K_h = K_local.data() + local_h * total_tokens * config.head_dim;
           float *V_h = V_local.data() + local_h * total_tokens * config.head_dim;

           extract_head_data(Q_data, Q_h, total_tokens, config.head_dim, config.n_heads, global_h, 0);
           extract_head_data(K_expanded, K_h, total_tokens, config.head_dim, config.n_heads, global_h, 0);
           extract_head_data(V_expanded, V_h, total_tokens, config.head_dim, config.n_heads, global_h, 0);
       }
       ```

     **(b) Invoke `ITensorAttention::compute` for Rank-Local Heads:**

     - Obtain an attention kernel for the activation precision currently in use. Since `ITensorAttention::compute` takes `float*` buffers, we can use one of:
       - A dummy `FP32Tensor` view solely to call `createAttention()`.
       - Or reusing the actual `output` tensor if it implements `IActivationTensor` and we are comfortable that its native precision matches the activation path.

     - Example (using `output` as the activation source):

       ```cpp
       auto *act_out = dynamic_cast<IActivationTensor *>(output);
       if (!act_out) {
           LOG_ERROR("[MpiAttentionOrchestrator] output tensor does not implement IActivationTensor");
           return false;
       }
       auto kernel = act_out->createAttention();

       bool local_ok = kernel->compute(
           Q_local.data(),
           K_local.data(),
           V_local.data(),
           out_local.data(),
           /*seq_len=*/total_tokens,
           /*n_heads=*/static_cast<int>(local_n_heads),
           /*n_kv_heads=*/config.n_kv_heads,
           /*head_dim=*/config.head_dim,
           /*causal=*/config.causal || sequence_lengths != nullptr,
           /*window_size=*/config.window_size,
           /*workspace_scores=*/nullptr,   // rank-local; can be wired later if needed
           /*workspace_buffer=*/nullptr,
           /*workspace_context=*/nullptr,
           /*workspace_mask=*/nullptr,
           /*use_bf16=*/(config.precision == ActivationPrecision::BF16),
           /*mpi_ctx=*/config.mpi_ctx.get(),
           /*device_idx=*/-1);
       ```

     - Propagate failure across ranks via `allreduce_sum` (as is already done today for GEMM failures).

   **(c) Reuse Existing Allreduce Packing Logic:**

   - Instead of filling `local_output` from per-head contexts, `out_local` now holds rank-local attention outputs.
   - Reuse the existing packing & allreduce code, substituting `local_output` with `out_local`:

     ```cpp
     // Copy local heads to correct position in send buffer
     #pragma omp parallel for collapse(2) if (total_tokens * local_n_heads > 64)
     for (int t = 0; t < total_tokens; ++t) {
         for (size_t local_h = 0; local_h < local_n_heads; ++local_h) {
             size_t global_h = start_head + local_h;
             for (int d = 0; d < config.head_dim; ++d) {
                 send_buffer[t * config.n_heads * config.head_dim + global_h * config.head_dim + d] =
                     out_local[t * local_n_heads * config.head_dim + local_h * config.head_dim + d];
             }
         }
     }
     ```

   - The rest of the staging + `allreduce_sum` + barrier logic can remain unchanged.

**Outcome:** Rank-local attention computation in Tensor-Parallel mode now runs entirely through `CpuAttentionKernelT<T>`, while MPI slicing and aggregation behavior remains intact.

---

### Step 5: Alignment and Cleanup for `Q8_1Tensor` / INT8 Paths (Optional Follow-Up)

**Goal:** Ensure quantized activation paths (Q8_1/INT32) are also unified under `CpuAttentionKernelT<T>` where appropriate.

1. **Audit `Q8_1Tensor::createAttention()`:**
   - Currently returns `CPUAttention` (legacy kernel). Plan to switch this to `CpuAttentionKernelT<Q8_1Tensor>` once that specialization is fully implemented, or maintain a compatibility path until it is.

2. **Ensure `CpuAttentionKernelT<T>` Specializations Cover All Intended Activation Types:**
   - Confirm extern template instantiations for:
     - `CpuAttentionKernelT<FP32Tensor>`
     - `CpuAttentionKernelT<BF16Tensor>`
     - `CpuAttentionKernelT<FP16Tensor>`
     - `CpuAttentionKernelT<INT32Tensor>` (if used for intermediate accumulators)
     - `CpuAttentionKernelT<Q8_0Tensor>` / `CpuAttentionKernelT<Q8_1Tensor>` as the design evolves.

3. **Update Documentation and Comments:**
   - In `TensorKernels.h` and the tensor files, document that `createAttention()` is the canonical path for CPU attention and that `MpiAttentionOrchestrator` / `GQAAttention` assume this.

---

## Testing and Validation Plan

1. **Unit Tests:**
   - Run all V2 unit tests:

     ```bash
     cd /workspaces/llaminar
     ctest --test-dir build_v2 --output-on-failure -R '^V2_Unit_'
     ```

   - Add or extend unit tests for:
     - `GQAAttention::compute` and `compute_batch` verifying results with different `ActivationPrecision` settings.
     - `MpiAttentionOrchestrator::compute` to ensure behavior matches old path for single-rank runs.

2. **MPI Tensor-Parallel Tests:**
   - Use existing MPI correctness tests (if present) or add focused tests around `compute_tensor_parallel`:
     - Compare `compute_mpi` (TensorParallel, world_size > 1) vs `compute` (single rank) outputs on small models and short sequences.

3. **Performance Sanity Checks:**
   - Measure attention block runtime before vs. after refactor for representative shapes:
     - Small seq_len / few heads.
     - Long seq_len / many heads.
   - Ensure there are no major regressions; minor differences due to different tiling or kernel behavior should be documented.

4. **Logging Verification:**
   - Spot-check logs for single-rank and MPI runs to verify that:
     - Key messages (e.g., MPI slice information, allreduce summaries) still appear.
     - New error messages from `CpuAttentionKernelT<T>` are informative and consistent.

---

## Rollout Strategy

1. **Phase 1 – Single-Rank Consolidation:**
   - Implement Step 1 and Step 2.
   - Run all unit tests; validate functional equivalence of `GQAAttention` and `MpiAttentionOrchestrator::compute` vs prior behavior.

2. **Phase 2 – Tensor-Parallel Integration:**
   - Implement Step 4 (kernel-backed `compute_tensor_parallel`).
   - Run MPI tests and small-scale integration tests with 2–4 ranks.

3. **Phase 3 – Quantized Path Alignment (Optional):**
   - Implement Step 5 incrementally, starting with FP32/BF16/FP16 stable paths, then INT32/Q8_1 as needed.

4. **Phase 4 – Cleanup & Documentation:**
   - Remove dead helper functions that are no longer used.
   - Finalize this document with any deviations from the plan discovered during implementation.
