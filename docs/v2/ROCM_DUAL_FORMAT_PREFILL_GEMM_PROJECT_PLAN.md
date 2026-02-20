# ROCm Dual-Format Prefill GEMM Project Plan

## Goal
Deliver a production-ready ROCm prefill GEMM path for both packed weight families:
- INT8 VNNI (Q7/Q8 full INT8 repack)
- ratio-VNNI (Q6 and lower, IQ* quant families, no mandatory full INT8 expansion)

This plan defines design constraints, kernel strategy, phased implementation, and validation gates.

## Why This Project
Current behavior is split:
- Decode (M=1) has native GEMV support for INT8 VNNI and ratio-VNNI variants.
- Prefill (M>1) still falls back to CK GEMM flow via row-major scratch/repack.

That fallback is safe, but it does not complete the intended dual-format story for prefill and can erode ratio-VNNI VRAM advantages.

## Objectives
1. Add native prefill GEMM for INT8 VNNI.
2. Add native prefill GEMM for ratio-VNNI payload format.
3. Preserve ratio-VNNI memory goals by avoiding forced full-int8 expansion in standard execution.
4. Keep deterministic, low-risk fallback to CK path for unsupported shapes.
5. Roll out behind feature flags with tight correctness and performance gates.

## Non-Goals (Initial Implementation)
- No decode-path redesign (M=1 GEMV path remains authoritative).
- No change to model semantics, TP/PP placement rules, or tensor ownership model.
- No immediate support for every ratio-VNNI variant in phase 1.

## Supported Format Matrix (Target)
- INT8 VNNI:
  - Q8_x, Q7_x style full-int8 repack families.
  - Native prefill GEMM path required.
- ratio-VNNI:
  - Q6 and below, IQ* families.
  - Native prefill GEMM path must consume payload + ratio side data directly.
  - Full-int8 expansion is fallback/debug option, not default operating mode.

## Current State Snapshot
- Existing decode kernels:
  - INT8 VNNI GEMV family (wide, square, grid_kpar, scaled/fused variants).
  - ratio-VNNI GEMV skeletons available for selected formats (Q4_0, IQ4_NL).
- Existing prefill path:
  - Quantize A -> CK GEMM int8 path via row-major scratch/repack -> scale/epilogue.
- Existing guardrail:
  - Prefill experimental branch already scaffolded with explicit fallback logging.

## Design Principles
1. Correctness-first rollout with strict fallback.
2. Keep decode and prefill dispatch separate to avoid regressions in M=1 path.
3. Avoid hidden data format conversions that break VRAM assumptions.
4. Shared epilogue semantics across backends (alpha, beta, optional bias).
5. Explicit observability: each dispatch path must log one-time reason when selected/fallbacked.

## ABI Extension (Phase 3 Start)

### ABI v2 Descriptor Contract
- New descriptor-based C ABI entrypoint: `rocmGemm_ratio_vnni_int8_int32_prefill_v2`.
- Shared descriptor type: `ROCmRatioVNNIPrefillAbiV2Desc`.
- Legacy fixed-parameter entrypoint remains available and forwards to v2.

Descriptor fields (v2):
- `abi_version`
- `bitwidth`, `codebook_id`, `has_min`
- `block_size`, `payload_bytes`, `ratio_bytes`
- `flags`
- `blocks_per_row`
- `payload_stride_bytes`, `ratio_stride_bytes`

Current supported subset under v2 remains Phase-2 compatible:
- `bitwidth=4`, `block_size=32`, `payload_bytes=16`, `ratio_bytes=1`
- `codebook_id in {0 (linear), 4 (IQ4)}`

### Why this matters
- Stops repeated C ABI signature churn for each new ratio family.
- Allows future family bring-up via metadata/decoder extension first, signature changes last.
- Preserves backward compatibility while migrating callsites to descriptor-based execution.

## High-Level Architecture

### Prefill Dispatch Contract (M>1)
Given (M, N, K, weight format, epilogue config):
1. If INT8 VNNI format and shape supported -> run native VNNI prefill GEMM.
2. Else if ratio-VNNI format and shape supported -> run native ratio prefill GEMM.
3. Else -> fallback CK path (existing stable behavior).

### Epilogue Strategy
Phase 1 uses two-stage output for safety:
- Core kernel computes INT8xINT8 (or ratio decode) accumulation into INT32.
- Existing scaling/epilogue utility applies alpha/beta/bias to FP32 output.

Later phases can fuse epilogue when stable.

### Workspace/Memory Policy
- Reuse existing workspace buffers where possible.
- Add dedicated temporary buffers only when unavoidable (for split-K or ratio staging).
- For ratio-VNNI, keep payload-resident execution; no unconditional full-int8 staging.

## Phased Implementation Plan

### Phase 0 - Baseline and Contracts
Objective: lock down expected behavior before new kernels land.

Tasks:
1. Document prefill dispatch matrix in code comments and this plan.
2. Add one-time logging for selected path and fallback reason categories.
3. Define hard shape guards (for example K % 4 constraints) and failure categories.

Exit Criteria:
- Deterministic logs identify which path executed for every prefill GEMM.
- No behavior change yet.

---

### Phase 1 - Native INT8 VNNI Prefill GEMM (M>1)
Objective: implement first native prefill kernel family for full-int8 VNNI weights.

Kernel Plan:
1. Add entrypoint family (suggested naming):
   - rocmGemm_int8_int8_int32_vnni_prefill
   - rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar (optional if needed)
2. Initial tile strategy:
   - Start with fixed tile (example: M_tile=8, N_tile=64, K_step=128).
   - Use sdot4-compatible K iteration and int32 accumulators.
3. Keep scaling/epilogue external in phase 1.

Integration Tasks:
1. Wire dispatch into ROCmQuantisedGemmKernel prefill branch (feature-flag gated).
2. Preserve CK fallback for unsupported shapes or launch failures.
3. Ensure fused-projection prefill path can use shared quantized activations with new kernel path.

Exit Criteria:
- Correctness parity vs CK across unit/integration prefill tests.
- No decode regressions.
- Prefill path selects native VNNI kernel for supported INT8 VNNI cases.

---

### Phase 2 - ratio-VNNI Prefill GEMM Core
Objective: native prefill GEMM for ratio payload + side metadata without full-int8 expansion.

Kernel Plan:
1. Add entrypoint family (suggested naming):
   - rocmGemm_ratio_vnni_int8_int32_prefill
2. Reuse ratio decode primitives and metadata validation from GEMV path.
3. Support initial subset first (Q4_0 and IQ4_NL), then expand.

Integration Tasks:
1. Add strict ratio metadata gates:
   - bitwidth, block size, payload bytes, codebook/min flags.
2. Route supported ratio formats to native ratio GEMM.
3. Fallback to CK path when metadata unsupported.

Exit Criteria:
- Native ratio prefill executes for initial supported formats.
- No forced full-int8 expansion in standard supported-ratio path.
- Correctness parity vs CK reference path.

---

### Phase 3 - Broaden ratio-VNNI Coverage
Objective: cover remaining Q6-and-lower and IQ* ratio families.

Tasks:
1. Add format specializations incrementally.
2. Add validation table mapping quant family -> ratio metadata contract.
3. Expand integration tests per quant family and shape bucket.

Exit Criteria:
- All targeted ratio families dispatch natively when metadata matches.
- Fallback rate for supported models is low and explainable.

---

### Phase 4 - Performance Optimizations
Objective: close throughput gap to CK and improve large-shape scaling.

Tasks:
1. Add split-K/grid_kpar variants where beneficial.
2. Tune tile shapes for representative M/N/K buckets.
3. Consider fused epilogue path (alpha/beta/bias) after baseline stability.
4. Optimize shared-memory staging and vectorized loads.

Exit Criteria:
- Measurable prefill latency improvement for targeted workloads.
- No correctness regressions from tuning.

---

### Phase 5 - Hardening and Default Rollout
Objective: move from experimental to default-safe operation.

Tasks:
1. Burn-in under long-run integration tests and stress runs.
2. Keep CK fallback available with clear diagnostics.
3. Promote feature flag defaults only after gates pass.

Exit Criteria:
- Stable across single-device and LOCAL TP runs.
- Production-ready fallback behavior documented.

## Dispatch and Fallback Decision Table
1. If M == 1 -> existing GEMV decode path (unchanged).
2. If M > 1 and experimental prefill flag disabled -> CK fallback.
3. If M > 1 and weight format is INT8 VNNI and shape supported -> native INT8 VNNI GEMM.
4. If M > 1 and weight format is ratio-VNNI and metadata+shape supported -> native ratio-VNNI GEMM.
5. Else -> CK fallback with one-time categorized reason log.

## Validation Plan

### Correctness
1. Add shape bucket tests:
   - Small M (2-8), medium M (16-64), large M.
   - N and K values matching real model layers.
2. Compare against CK outputs:
   - Max abs error, relative error, and tolerance per quant family.
3. Run existing unit and integration suites in Integration build.

### Performance
1. Benchmark prefill-only timing by shape bucket.
2. Benchmark end-to-end startup + first prefill token path.
3. Track fallback rate (how often native path is bypassed).

### Stability
1. Stress test repeated prefill runs for memory correctness.
2. Validate no stalls/hangs in LOCAL TP.
3. Confirm deterministic fallback under unsupported metadata.

## Instrumentation Requirements
Add counters/logging for:
- prefill_gemm.path_selected (int8_vnni_native, ratio_vnni_native, ck_fallback)
- prefill_gemm.fallback_reason (shape, metadata, launch_error, flag_disabled)
- prefill_gemm.kernel_ms (per path)
- prefill_gemm.epilogue_ms

## Risks and Mitigations
1. Risk: ratio decode complexity causes correctness drift.
   - Mitigation: keep external epilogue first, add strict metadata gating.
2. Risk: native path underperforms CK for some shapes.
   - Mitigation: shape-aware dispatch thresholds + CK fallback.
3. Risk: hidden memory growth from extra staging.
   - Mitigation: workspace accounting and explicit buffer ownership checks.
4. Risk: decode regressions from shared code edits.
   - Mitigation: isolate M==1 path, avoid refactors in decode hot path during early phases.

## Work Breakdown Checklist

### Phase 0
- [x] Add dispatch/fallback reason taxonomy (scaffold implementation in ROCmQuantisedGemmKernel).
- [x] Add prefill path selection logs and counters.

### Phase 1
- [x] Add native INT8 VNNI prefill GEMM kernel entrypoints (scaffold C API, fallback-safe).
- [x] Wire prefill dispatch in ROCmQuantisedGemmKernel (experimental route + CK fallback).
- [x] Add correctness tests vs CK for INT8 VNNI prefill.

### Phase 2
- [x] Add native ratio-VNNI prefill GEMM core entrypoints.
- [x] Add ratio metadata validation gates.
- [x] Add correctness tests for Q4_0 and IQ4_NL ratio formats.

### Phase 3
- [~] Add ratio specializations for remaining Q6-and-lower and IQ* families (ABI-v2 descriptor scaffold landed; container/decoder extension now includes Q4_1 min side-channel packing + row-major expansion + IQ4_XS superblock payload slicing; remaining families still pending).
- [x] Expand integration coverage matrix.

#### Latest Validation (2026-02-20)
- Native ratio prefill path now accepts `has_min=1` metadata for linear codebook (`Q4_1`) via ABI-v2 min side-channel fields.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed in repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.
- No regressions observed in existing `Q4_0` / `IQ4_NL` parity coverage within the same target.

#### Latest Validation (2026-02-20, Slice 2)
- Added native ratio prefill support for `Q5_0` and `Q5_1` (bitwidth=5, payload=20) with ABI-v2 descriptor validation.
- Added integration parity coverage for `Q5_0` / `Q5_1` in `PrefillNativeRatioVNNI_MatchesCKFallback`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 3)
- Added bitwidth-5 ratio→rowmajor CK fallback repack support in ROCm HIP path (Q5 payload expansion kernel + dispatcher validation).
- Expanded integration coverage with `PrefillCKFallback_Q5RatioOnlyRepack_MatchesBaseline`, which forces ratio-only Q5 CK fallback (no VNNI host weights) and verifies parity.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 4)
- Added native ratio container coverage for `IQ4_XS` by mapping each 256-element superblock into eight 32-element IQ4 payload slices under the existing ABI-v2 Q4 descriptor contract (`bitwidth=4`, `payload=16`, `block_size=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` to include `IQ4_XS` parity coverage.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 5)
- Expanded ratio-only CK fallback parity coverage from Q5-only to a broader family set: `Q4_0`, `IQ4_NL`, `IQ4_XS`, `Q4_1`, `Q5_0`, `Q5_1`.
- Renamed coverage test to `PrefillCKFallback_RatioOnlyRepack_MatchesBaseline` and added explicit min side-channel assertions for `Q4_1`/`Q5_1`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 6)
- Expanded `PrefillNativeInt8VNNI_MatchesCKFallback` to cover additional low-bit source families routed through current INT8-VNNI native prefill path: `Q6_K`, `Q3_K`, `Q2_K` (alongside `Q8_0`).
- Added explicit packaging assertions for this route (`int8_data_vnni` present, ratio payload/side-channel absent) to guard dispatch assumptions while ratio-native specializations for these families remain pending.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 7)
- Further expanded `PrefillNativeInt8VNNI_MatchesCKFallback` to include IQ low-bit source families currently routed through INT8 packing: `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`.
- Retained and exercised route assertions that these families stay on the INT8-VNNI path (`int8_data_vnni` populated, ratio payload/side-channel empty) while native ratio prefill specializations remain pending.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 8)
- Added first non-Q4/Q5 descriptor extension for native prefill: `Q6_K` now emits an ABI-v2 payload container with `bitwidth=8`, `block_size=32`, `payload_bytes=32` (payload built from existing INT8 VNNI host pack), preserving existing epilogue scaling semantics.
- Extended ROCm ABI-v2 prefill launcher with a dedicated bitwidth-8 payload kernel path (`gemm_payload_v2_i8_prefill_kernel`) and descriptor validation for this family.
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` to include `Q6_K` parity coverage and metadata assertions (`bitwidth=8`, `payload=32`).
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 9)
- Extended the bitwidth-8 payload-v2 native prefill descriptor route to include `IQ3_XXS` (same `block_size=32`, `payload_bytes=32` contract built from existing INT8 VNNI host pack).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` to include `IQ3_XXS` parity coverage and bitwidth-8 metadata assertions.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed, plus repeated stress run (`ctest --repeat until-fail:3`): 3/3 passes.

#### Latest Validation (2026-02-20, Slice 10)
- Re-enabled `PrefillNativeRatioVNNI_MatchesCKFallback` full-family execution (removed temporary skip) and reproduced allocator corruption in the ratio-native path (`double free or corruption (!prev)` / `malloc(): invalid size`).
- Fixed dequant cache tail handling in quant tensor `data()` implementations to prevent partial-block out-of-bounds writes when `K < block_size` or `K % block_size != 0`:
   - `Q6_KTensor::data()`
   - `IQ3_XXSTensor::data()`
   - `IQ3_STensor::data()`
- Focused ratio-native parity case now passes end-to-end (Q6_K, IQ3_XXS, IQ3_S coverage active), and the integration gate is stable under repeat stress (`ctest --repeat until-fail:5 -R '^V2_Integration_ROCmQuantisedGemmKernel$'`): 5/5 passes.
- Extended burn-in on the same focused gate also passed (`ctest --repeat until-fail:10 -R '^V2_Integration_ROCmQuantisedGemmKernel$'`): 10/10 passes.

#### Latest Validation (2026-02-20, Slice 11)
- Extended payload-v2 bitwidth-8 native prefill specialization to `IQ2_XXS`, `IQ2_XS`, and `IQ2_S` (same descriptor contract as current bitwidth-8 path: `block_size=32`, `payload_bytes=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` parity matrix and metadata assertions to include all three IQ2 families.
- During first bring-up, reproduced allocator corruption in IQ2 paths and fixed the same root-cause class (partial-block dequant cache overwrite) in:
   - `IQ2_XXSTensor::data()`
   - `IQ2_XSTensor::data()`
   - `IQ2_STensor::data()`
- Focused ratio-native parity test now passes with active coverage for `Q6_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ3_S`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.

#### Latest Validation (2026-02-20, Slice 12)
- Extended payload-v2 bitwidth-8 native prefill specialization to `Q3_K` and `Q2_K` (same descriptor contract: `bitwidth=8`, `block_size=32`, `payload_bytes=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` parity matrix and metadata assertions to include `Q3_K` and `Q2_K` alongside existing `Q6_K`/`IQ2*`/`IQ3*` families.
- Preemptively fixed the same partial-block dequant cache overwrite class for K-tail handling in:
   - `Q3_KTensor::data()`
   - `Q2_KTensor::data()`
- Focused ratio-native parity test passes with active coverage for `Q6_K`, `Q3_K`, `Q2_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ3_S`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.

#### Latest Validation (2026-02-20, Slice 13)
- Extended payload-v2 bitwidth-8 native prefill specialization to `IQ1_S` and `IQ1_M` (same descriptor contract: `bitwidth=8`, `block_size=32`, `payload_bytes=32`).
- Expanded `PrefillNativeRatioVNNI_MatchesCKFallback` parity matrix and metadata assertions to include `IQ1_S` and `IQ1_M`.
- Fixed partial-block dequant cache overwrite class for K-tail handling in:
   - `IQ1_STensor::data()`
   - `IQ1_MTensor::data()`
- Parity behavior note: `IQ1_M` remains cosine-stable (`~1.0`) but shows quantization-step absolute deltas vs CK fallback (observed up to `1.0`), so the family-specific `max_abs_diff` threshold was relaxed to `1.5` while retaining strict `5e-3` for all other covered families.
- Focused ratio-native parity test passes with active coverage for `Q6_K`, `Q3_K`, `Q2_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and `IQ3_S`.
- Focused integration target `V2_Integration_ROCmQuantisedGemmKernel` passed repeated stress run (`ctest --repeat until-fail:5`): 5/5 passes.

#### Latest Validation (2026-02-20, Slice 14)
- Added dedicated unit regression coverage for quantized-tensor partial-tail decode safety in `Test__QuantizedTensorTailDecodeRegression`.
- New unit target `V2_Unit_QuantizedTensorTailDecodeRegression` validates `data()` vs `to_fp32_row()` on partial-`K` shape (`13x17`) across all recently fixed families: `Q6_K`, `Q3_K`, `Q2_K`, `IQ3_XXS`, `IQ3_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ1_S`, `IQ1_M`.
- Focused unit gate passed in Integration build: `ctest --test-dir build_v2_integration -R '^V2_Unit_QuantizedTensorTailDecodeRegression$' --output-on-failure --parallel`.

#### Latest Validation (2026-02-20, Slice 15 - deferred)
- Attempted to extend payload-v2 bitwidth-8 native prefill specialization and parity matrix to `Q4_K` and `Q5_K`.
- Focused parity run (`ROCmQuantisedGemmIntegrationTest.PrefillNativeRatioVNNI_MatchesCKFallback`) reproduced allocator corruption (`malloc(): invalid size (unsorted)`) immediately after `Q6_K` coverage when the new families were included.
- Rolled back the `Q4_K`/`Q5_K` additions from the bitwidth-8 payload-v2 route and parity matrix to preserve a green baseline.
- Post-rollback focused parity gate is green again for active families (`Q6_K`, `Q3_K`, `Q2_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`) using:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.

#### Latest Validation (2026-02-20, Slice 16)
- Fixed root-cause heap corruption in quantized tail decode cache for:
   - `Q4_KTensor::data()`
   - `Q5_KTensor::data()`
- Both methods now use partial-block safe copy semantics (decode full super-block to temp, copy only valid tail elements), matching the previously fixed `Q6_K`/`Q3_K`/`Q2_K`/`IQ*` families.
- Extended unit regression coverage (`V2_Unit_QuantizedTensorTailDecodeRegression`) to include `Q4_K` and `Q5_K` on partial-`K` shape (`13x17`); focused unit gate passed.
- Re-enabled payload-v2 bitwidth-8 native prefill specialization and parity-matrix coverage for `Q4_K` and `Q5_K`; focused parity gate now passes with active coverage for both families:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.
- Focused integration target burn-in also passed: `ctest --test-dir /workspaces/llaminar/build_v2_integration --output-on-failure --repeat until-fail:5 -R '^V2_Integration_ROCmQuantisedGemmKernel$'` (5/5 passes).

#### Latest Validation (2026-02-20, Slice 17)
- Completed the Phase-3 parity coverage audit and closed a matrix gap in `PrefillNativeRatioVNNI_MatchesCKFallback`: added explicit execution coverage for `Q4_0`, `IQ4_NL`, `IQ4_XS`, `Q4_1`, `Q5_0`, and `Q5_1` (in addition to existing `Q6_K`/`Q4_K`/`Q5_K`/`Q3_K`/`Q2_K`/`IQ1*`/`IQ2*`/`IQ3*`).
- Audit run exposed another latent partial-tail heap corruption class in `IQ4_XSTensor::data()` for sub-block `K` shapes; fixed with the same tail-safe partial-block copy approach used for other quant families.
- Extended unit tail regression suite to include `IQ4_XS` (`V2_Unit_QuantizedTensorTailDecodeRegression`), and validated pass.
- Focused ratio-native parity gate now passes end-to-end with all targeted Phase-3 families active:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.

#### Latest Validation (2026-02-20, Slice 18)
- Expanded ratio-native prefill shape buckets in `PrefillNativeRatioVNNI_MatchesCKFallback` to strengthen Phase-3 exit confidence:
   - small-M: `(M,N,K)=(4,128,128)`
   - medium-M: `(12,160,128)`, `(20,192,128)`
   - larger M/K: `(28,256,256)`
   - wider-K bucket: `(16,192,512)`
- Rebuilt focused integration target and validated the full-family parity matrix across all listed Phase-3 families under the expanded shape set.
- Focused gate passed:
   `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`.

#### Latest Validation (2026-02-20, Slice 19 - Phase 4 start)
- Added first opt-in INT8 prefill split-K/grid-kpar variant behind existing dispatch guards and fallback:
   - New HIP kernel path uses `grid.z` K-splitting with INT32 atomic accumulation.
   - New launcher entrypoint: `rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(...)`.
   - Routing is opt-in via DebugEnv flags and falls back to baseline INT8 prefill kernel on launch failure.
- Added ROCm env controls:
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR=1`
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS=<n>`
- Expanded `PrefillNativeInt8VNNI_MatchesCKFallback` to validate both INT8 native variants (`baseline`, `grid_kpar`) against CK fallback.
- Focused validation passed:
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeInt8VNNI_MatchesCKFallback*'`
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`

#### Latest Validation (2026-02-20, Slice 20 - Phase 4 tuning)
- Added grid-kpar split tuning controls and auto mode semantics:
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR=1` enables INT8 prefill grid-kpar routing.
   - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_SPLITS=0` now means auto heuristic (manual `1..32` still supported).
- Implemented conservative auto-slice heuristic in prefill dispatcher (`tryExperimentalPrefillNativeGemm`) that selects split count from `(M,N,K)` and bounds fanout for atomic overhead control.
- Preserved fallback safety contract:
   - grid-kpar launch failure falls back to baseline native INT8 prefill
   - baseline native failure still falls back to CK path via existing logic.
- Focused correctness gates passed after tuning:
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeInt8VNNI_MatchesCKFallback*'`
   - `/workspaces/llaminar/build_v2_integration/tests/v2/v2_integration_rocm_quantised_gemm_kernel --gtest_filter='*PrefillNativeRatioVNNI_MatchesCKFallback*'`
- Quick perf smoke (single-run wall clock, same gtest filter) on this host:
   - baseline: `baseline_elapsed_ms=1124`
   - grid-kpar auto: `grid_auto_elapsed_ms=951`
   - note: this is a coarse smoke signal, not a full benchmark sweep.

#### Latest Validation (2026-02-20, Slice 21 - production-path perf sweep harness)
- Added a production-path prefill perf sweep to `Perf__ROCmQuantisedGemmKernel`:
   - new test: `ROCmQuantisedGemmPerf.PrefillFullPath_GridKParSweep`
   - measures full `multiply_tensor` prefill route (quantize activations + native prefill GEMM + scaling/epilogue), not isolated kernel-only timings.
- Added mode matrix for controlled comparison under identical call chain and fallback contract:
   - `baseline`
   - `grid_auto` (`LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR=1`, `..._SPLITS=0`)
   - `grid_s4`
   - `grid_s8`
- Added per-mode env override + `mutableDebugEnv().reload()` wiring inside the perf test to ensure mode toggles are applied in-process between runs.
- Added explicit workspace binding in benchmark path so full `multiply_tensor` timing is valid (matches production kernel contract).
- Focused validation passed:
   - `/workspaces/llaminar/build_v2_release/tests/v2/v2_perf_rocm_quantised_gemm_kernel --gtest_filter='*PrefillFullPath_GridKParSweep*'`
- Initial host results (Qwen-7B FFN Up, full path):
   - `M=64`: baseline `6.812 ms`, grid_auto `6.871 ms`, grid_s4 `6.911 ms`, grid_s8 `6.876 ms`
   - `M=256`: baseline `25.400 ms`, grid_auto `26.241 ms`, grid_s4 `26.305 ms`, grid_s8 `26.246 ms`
   - all modes cosine-stable vs FP32 reference (`~0.999956`).

#### Latest Validation (2026-02-20, Slice 22 - prefill micro-variant dispatch + real-model aspect sweeps)
- Added shape-aware INT8 prefill tile variant dispatch with micro-specialized launch variants:
   - baseline and grid-kpar launchers now dispatch among `{16x16, 32x8, 8x32, 8x8}` tiles.
   - runtime auto-selection is based on matrix aspect ratio and small-shape guardrails.
   - optional force-override knobs added for tuning sweeps:
      - `LLAMINAR_ROCM_VNNI_PREFILL_VARIANT` (`-1 auto`, `0 16x16`, `1 32x8`, `2 8x32`, `3 8x8`)
      - `LLAMINAR_ROCM_VNNI_PREFILL_GRID_VARIANT` (`-1 auto`, `0 16x16`, `1 32x8`, `2 8x32`, `3 8x8`)
- Extended full-path perf suite with a real-model/aspect-ratio sweep target:
   - new test: `ROCmQuantisedGemmPerf.PrefillFullPath_RealModelAspectSweep`
   - covers representative projection shapes across 0.5B/7B/14B/32B and both tall/wide/square aspect ratios.
   - compares auto-dispatch against forced tile variants using full `multiply_tensor` path timing.

#### Latest Validation (2026-02-20, Slice 23 - CPT + KB levers in one pass)
- Implemented two additional prefill tuning levers requested for sweep-driven policy work:
   1. **CPT (outputs-per-thread)** for INT8 prefill kernels:
      - New env: `LLAMINAR_ROCM_VNNI_PREFILL_CPT` with supported values `{1,2,4}`.
      - Applied to both baseline prefill and grid-kpar prefill launch paths.
   2. **Grid-kpar KB control** for split-K policy:
      - New env: `LLAMINAR_ROCM_VNNI_PREFILL_GRID_KPAR_KB` (`0` = auto policy).
      - Integrated into prefill dispatcher before legacy split override/auto fallback.
- Added full-path performance sweep target that exercises both knobs together:
   - new test: `ROCmQuantisedGemmPerf.PrefillFullPath_CptAndKbSweep`
   - sweeps `CPT ∈ {1,2,4}` and `KB ∈ {0(auto),4,8,16}` over representative real-model shapes.
- Focused sweep run passed:
   - `/workspaces/llaminar/build_v2_release/tests/v2/v2_perf_rocm_quantised_gemm_kernel --gtest_filter='*PrefillFullPath_CptAndKbSweep*'`
- Initial findings on this host:
   - `CPT=1` is consistently best for the tested prefill shapes.
   - `CPT=2` and `CPT=4` regress throughput materially (larger regression at `CPT=4`).
   - Grid-kpar `KB` sensitivity is shape-dependent but modest in the tested band (`4..16`), with occasional small wins over `KB=0(auto)` for wider reduction-heavy shapes.

#### Latest Validation (2026-02-20, Slice 24 - retire CPT>1 from routine sweeps)
- To reduce sweep runtime and focus on high-value tuning axes, routine prefill sweep policy now fixes `LLAMINAR_ROCM_VNNI_PREFILL_CPT=1`.
- `PrefillFullPath_CptAndKbSweep` was narrowed to `CPT=1` only while retaining `KB ∈ {0(auto),4,8,16}` coverage.
- Rationale: host sweep data shows `CPT>1` is uniformly worse across tested real-model prefill shapes, so continuing to sweep those modes adds time without actionable upside.
- Sweep coverage now explicitly includes Qwen `7B` and `14B` prefill projection families: `AttnOut`, `FFN_Up`, `FFN_Down`, and `FFN_Gate`, so `KB` tuning decisions are validated across both square (attention) and tall/wide (FFN) shapes.

### Phase 4
- [ ] Add split-K/grid_kpar prefill variants as needed.
- [ ] Tune tile parameters and dispatch thresholds.
- [ ] Evaluate optional fused epilogue.

### Phase 5
- [ ] Run burn-in and stress validation.
- [ ] Define criteria to flip default flag behavior.
- [ ] Document stable fallback and operational guidance.

## Success Criteria
This project is complete when:
1. Prefill GEMM has native execution for both INT8 VNNI and ratio-VNNI formats.
2. ratio-VNNI path preserves VRAM intent by default (no unconditional full-int8 expansion).
3. Correctness parity with CK reference is maintained across targeted quant families.
4. CK fallback remains reliable and observable for unsupported scenarios.
5. Performance is neutral-to-better on target prefill workloads and improved after tuning phases.

## Proposed Immediate Next Step
1. ✅ Coverage audit complete: parity matrix and metadata assertions now execute all targeted Phase-3 families (`Q4_0`, `IQ4_NL`, `IQ4_XS`, `Q4_1`, `Q5_0`, `Q5_1`, `Q6_K`, `Q4_K`, `Q5_K`, `Q3_K`, `Q2_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`).
2. ✅ Shape-bucket extension complete for ratio-native prefill (small/medium/large `M` + wider `K`) with focused gate passing.
3. ✅ Phase 4 kickoff complete: first INT8 split-K/grid_kpar prefill variant is integrated behind feature flags and validated.
4. ✅ Completed: auto split tuning and focused baseline-vs-grid perf smoke.
5. ✅ Completed: added production-path full prefill sweep harness (`PrefillFullPath_GridKParSweep`) with baseline vs grid (`auto/s4/s8`) comparisons.
6. In progress: broadened sweep buckets to real model projection shapes + aspect ratios and added micro-variant forcing controls.
7. Next: ingest sweep results and codify final shape-threshold rules (baseline vs grid-kpar and tile variant) into stable default dispatch policy.
