# TurboQuant Debug Handover

Date: 2026-03-26

Scope: handoff for continued debugging of TurboQuant KV-cache quantization, with emphasis on the mismatch between the paper's predicted `Q_prod` behavior and current end-to-end decode parity in Llaminar V2.

## 1. Current Status

This branch now contains:

- Per-layer and per-head derived TurboQuant contexts.
- Two runtime modes for TurboQuant storage:
  - `Q_prod` style inner-product path.
  - `scalar-full` direct reconstruction path.
- Qwen2 single-device parity coverage for `TQ4` and `TQ3` KV cache.
- Qwen3 single-device parity coverage for `TQ4` and `TQ3` KV cache.

Current validated outcomes:

- Qwen2, `head_dim=64`:
  - `Q_prod` for keys was not acceptable in end-to-end decode parity.
  - Runtime was changed to use `scalar-full` for keys when `head_dim < 128`.
  - Values always use `scalar-full`.
- Qwen3, `head_dim=128`:
  - New parity cases were added to test the claim that `Q_prod` should be acceptable at `head_dim=128`.
  - Result: both `TQ4` and `TQ3` pass prefill but fail decode badly.

This means the working hypothesis that `head_dim=128` is sufficient for the current `Q_prod` implementation is not supported by the actual parity gate.

## 2. Implementation Files Touched

Primary implementation files touched in this debugging pass:

- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp`
- `src/v2/kernels/cpu/turboquant/TurboQuantContext.h`
- `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h`
- `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h`
- `src/v2/tensors/TQ4Tensor.h`
- `src/v2/tensors/TQ4Tensor.cpp`
- `src/v2/tensors/TQ3Tensor.h`
- `src/v2/tensors/TQ3Tensor.cpp`

Supporting test files touched:

- `tests/v2/integration/parity/qwen2/Test__Qwen2_SingleDevice_Parity.cpp`
- `tests/v2/integration/parity/qwen3/Test__Qwen3_SingleDevice_Parity.cpp`
- `tests/v2/unit/Test__TurboQuantRoundtrip.cpp`
- `tests/v2/unit/kernels/cpu/Test__CPURingKVCache_TurboQuant.cpp`

## 3. Key Code Anchors

The main runtime decision point is now in `KVCacheAppendStage`:

- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp:950`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp:951`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp:965`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp:990`

That logic currently does:

- derive a layer-specific TurboQuant context,
- choose `use_inner_product_keys = (head_dim >= 128)`,
- route keys to `quantize_from_fp32()` for larger heads,
- route keys to `quantize_from_fp32_scalar()` for smaller heads,
- route values to `quantize_from_fp32_scalar()` unconditionally.

The decode-side dequantization path is in `AttentionComputeStage`:

- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp:239`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp:245`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp:250`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp:306`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp:345`

The quantize/dequantize primitives are here:

- `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:47`
- `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:97`
- `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:164`
- `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:245`
- `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:64`
- `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:124`
- `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:170`
- `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:224`

The context derivation logic is here:

- `src/v2/kernels/cpu/turboquant/TurboQuantContext.h:64`
- `src/v2/kernels/cpu/turboquant/TurboQuantContext.h:74`
- `src/v2/kernels/cpu/turboquant/TurboQuantContext.h:83`

The tensor wrappers that expose scalar-full entry points are here:

- `src/v2/tensors/TQ4Tensor.cpp:244`
- `src/v2/tensors/TQ4Tensor.cpp:336`
- `src/v2/tensors/TQ4Tensor.cpp:378`
- `src/v2/tensors/TQ3Tensor.cpp:237`
- `src/v2/tensors/TQ3Tensor.cpp:329`
- `src/v2/tensors/TQ3Tensor.cpp:371`

The newly added Qwen3 parity configs are here:

- `tests/v2/integration/parity/qwen3/Test__Qwen3_SingleDevice_Parity.cpp:84`
- `tests/v2/integration/parity/qwen3/Test__Qwen3_SingleDevice_Parity.cpp:101`

## 4. Paper Source and License

Paper:

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
- Authors: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
- arXiv: `2504.19874`
- Abstract page: `https://arxiv.org/abs/2504.19874`
- HTML: `https://arxiv.org/html/2504.19874v1`
- ar5iv mirror used for readable section extraction: `https://ar5iv.labs.arxiv.org/html/2504.19874`
- License shown on arXiv page: `CC BY 4.0`

Terminology note:

- The paper names the direct reconstruction method `TurboQuant_mse` / `Q_mse`.
- This repo currently uses the name `scalar-full` for the engineering mode that spends the full per-coordinate bit budget on direct reconstruction instead of a residual QJL sketch.
- `scalar-full` is therefore our implementation-side analogue of the paper's MSE-oriented path, not a paper term.

## 5. Paper Claims and Techniques Relevant to This Debugging Pass

The excerpts below are included because they are the precise claims the implementation is trying to realize. They should be used as the standard for consistency checks.

### 5.1 Problem Definition and Objectives

Verbatim excerpt from the paper's problem statement:

> "Formally, our goal is to design a quantization map, denoted as `Q: R^d -> {0,1}^B`, that transforms `d`-dimensional vectors to a binary string of `B` bits. If we set `B = b * d` for some `b >= 0`, this quantizer will have a bit-width of `b`, representing the average number of bits used to encode each real-valued coordinate of `R^d`. Crucially, we require an inverse map, `Q^{-1}: {0,1}^B -> R^d` that performs dequantization, approximately reconstructing original vectors from their quantized representations." 

> "(MSE) `D_mse := E_Q[||x - Q^{-1}(Q(x))||_2^2]`"

> "(inner-prod error) `D_prod := E_Q[|<y, x> - <y, Q^{-1}(Q(x))>|^2]`."

> "Furthermore, for inner-product quantizers, we require unbiasedness of the inner product estimator, a desirable property for numerous applications. More precisely, we require: `(unbiased inner-prod) E_Q[<y, Q^{-1}(Q(x))>] = <y, x>`."

### 5.2 MSE TurboQuant, Which Best Matches Our `scalar-full`

Verbatim excerpts describing the MSE-oriented method:

> "Our first VQ algorithm is designed to minimize MSE distortion. To achieve this, we apply a random rotation to the input vectors, thereby inducing a Beta distribution on each coordinate, irrespective of the input vectors themselves. In high dimensions `d`, the distribution of each coordinate converges to a Gaussian distribution `N(0, 1/d)` due to concentration of measure and the central limit theorem. Furthermore, any two distinct coordinates become nearly uncorrelated and, more importantly, almost independent. This near-independence is a crucial aspect that simplifies our quantization design. It allows us to quantize each coordinate using optimal scalar quantization, disregarding interactions or correlations between different coordinates, while still achieving near-optimal distortion."

> "We find optimal scalar quantizers for random variables with Beta distributions by solving a continuous 1-dimensional k-means problem using the Max-Lloyd algorithm. We precompute and store these optimal codebooks for a range of practically useful bit-widths, to enable efficient subsequent invocations of our TurboQuant algorithm."

> "Let `x in S^{d-1}` be a (worst-case) vector on the unit sphere in dimension `d`. We aim to quantize `x` to `b` bits per coordinate while minimizing the reconstruction MSE. We start by randomizing this vector by multiplying it with a random rotation matrix `Pi in R^{d x d}`."

> "The resulting rotated vector, `Pi * x`, is uniformly distributed on the unit sphere `S^{d-1}`. As shown in Lemma 1, each coordinate of `Pi * x` follows a Beta distribution, which converges to a normal distribution in high dimensions. Furthermore, in high dimensions, distinct coordinates of `Pi * x` become nearly independent, allowing us to apply optimal scalar quantizers to each coordinate independently."

> "The optimal scalar quantization problem, given a known probability distribution, can be framed as a continuous k-means problem in dimension one. Specifically, we aim to partition the interval `[-1, 1]` into `2^b` clusters/buckets."

> "Therefore the quantizer `Q_mse: R^d -> {0,1}^{b*d}` first computes `Pi * x` and then computes and stores the indices of the nearest centroids to each coordinate of this vector. The dequantization map `Q_mse^{-1}: {0,1}^{b*d} -> R^d` reconstructs the vector by retrieving the centroids corresponding to the stored indices and then rotating the result back to the original basis through multiplication with `Pi^T`."

Theorem-level distortion claim:

> "For any bit-width `b >= 1` and any vector `x in S^{d-1}`, the procedure `Quant_mse(x)` outputs an index vector `idx in [2^b]^d`. When this index vector is passed to the primitive `DeQuant_mse(idx)`, it produces a reconstructed vector `x~ in R^d` that satisfies the following distortion bounds:"

> "MSE defined as `D_mse := E_{x~}[||x - x~||_2^2]` is bounded by `D_mse <= (3*pi/2) * 1/(4^b)` for any `b >= 0`."

> "For small bit-widths, specifically `b = 1,2,3,4` the MSE exhibits finer-grained distortion values: `D_mse ≈ 0.36, 0.117, 0.03, 0.009`, respectively."

### 5.3 QJL Primitive Used by `Q_prod`

Verbatim excerpts describing QJL:

> "For any positive integer `d` the QJL map `Q_qjl: R^d -> {-1,+1}^d` is defined as: `Q_qjl(x) := sign(S * x)` for any `x in R^d`, where `S in R^{d x d}` is a random matrix with i.i.d. entries sampled from the normal distribution `N(0,1)` and the sign function is applied entry-wise to its vector input."

> "The inverse/dequantization map `Q_qjl^{-1}: {-1,+1}^d -> R^d` is defined as: `Q_qjl^{-1}(z) := (pi/2d) * S^T * z` for any `z in {-1,+1}^d`."

> "Unbiased: `E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y, x>`."

> "Variance Bound: `Var(<y, Q_qjl^{-1}(Q_qjl(x))>) <= (pi/2d) * ||y||_2^2`."

### 5.4 Inner-Product TurboQuant, Which Best Matches Our `Q_prod`

Verbatim excerpts describing the paper's `Q_prod` technique:

> "We show that the MSE optimized quantizers are biased for inner product estimation and thus a different VQ scheme is needed to get an unbiased inner product quantizer. Our solution is a two stage algorithm that first applies the abovementioned `Q_mse` with a bit-width one less than our target budget and then apply a QJL on the residual error. This is proved to be unbiased and also has nearly optimal inner product error rate."

> "As previously stated, we design two VQ algorithms: one optimized for minimizing MSE and the other for minimizing inner product error. We show that MSE-optimal quantizers do not necessarily provide unbiased inner product estimates, particularly exhibiting significant bias at lower bit-widths. Our solution for inner product quantization is a two-stage algorithm. First, we apply the MSE-optimal quantizer using one less bit than the desired bit-width budget, thus minimizing the L2 norm of the residuals. Next we apply an unbiased and optimal single-bit quantizer to the residual."

> "To address this bias, we propose a solution that combines `TurboQuant_mse` with an instance of QJL. Specifically, let `Q_mse` be the quantization map corresponding to `TurboQuant_mse` with a bit-width of `b-1`. For any `x in S^{d-1}` the residual vector, defined as `r := x - Q_mse^{-1}(Q_mse(x))`, has a small L2 norm. We can then apply the QJL transform to this residual."

> "`<y, Q_mse^{-1}(Q_mse(x))> + ||r||_2 * <y, Q_qjl^{-1}(Q_qjl(r))>`."

> "More formally, the quantization map `Q_prod: S^{d-1} -> [2^{b-1}]^d x {-1,1}^d x R` is defined as: `Q_prod(x) = [Q_mse(x), Q_qjl(x - Q_mse^{-1}(Q_mse(x))), ||x - Q_mse^{-1}(Q_mse(x))||_2]`."

Theorem-level claim for `Q_prod`:

> "For any bit-width `b >= 1` and any vector `x in S^{d-1}`, the procedure `Quant_prod(x)` outputs an index vector `idx in [2^{b-1}]^d` along with a sign vector `qjl in {-1,1}^d` and a positive number `gamma >= 0`. When these vectors and the scalar value are passed to the primitive `DeQuant_prod(idx, qjl, gamma)`, it produces a reconstructed vector `x~` that satisfies the following distortion bounds:"

> "Expected inner-product `E_{x~}[<y, x~>] = <y, x>`."

> "Inner-product distortion defined as `D_prod := E_{x~}[|<y, x> - <y, x~>|^2]` is bounded by `D_prod <= (3*pi/2) * (||y||_2^2 / d) * 1/(4^b)` for any `b >= 0`."

> "For small bit-widths, specifically `b = 1,2,3,4`, `D_prod` exhibits finer-grained distortion values: `D_prod ≈ 1.57/d, 0.56/d, 0.18/d, 0.047/d`, respectively."

### 5.5 Lower-Bound Framing That Matters Here

Verbatim excerpt:

> "As demonstrated by our lower bounds, TurboQuant’s MSE distortion is provably within a factor of at most `3*pi/2 ≈ 2.7` of the information-theoretical lower bound."

This matters because our current decode failures are much larger than what the paper predicts for the intended `Q_prod` regime, even after allowing for constant factors.

## 6. Our Matching Implementations

### 6.1 `Q_prod` in Our Code

Our `Q_prod`-style path corresponds to:

- `turboquant_quantize_tq4()` in `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:164`
- `turboquant_quantize_tq3()` in `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:245`
- `turboquant_dequantize_tq4()` in `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:170`
- `turboquant_dequantize_tq3()` in `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:224`

Implementation details:

- We compute `norm = ||input||_2` and quantize the normalized unit vector.
- We rotate the unit vector with `apply_rotation()`.
- We allocate `b-1` bits per coordinate to the MSE stage:
  - `TQ4` uses `tq3_nearest_centroid()` and stores 3-bit centroid indices.
  - `TQ3` uses `tq2_nearest_centroid()` and stores 2-bit centroid indices.
- We reconstruct the MSE part, inverse-rotate it, and compute the residual in the original basis.
- We store `residual_norm = ||residual||_2`.
- We normalize the residual and quantize it with `qjl_quantize_signs()`.
- On dequantization we reconstruct the MSE part, inverse-rotate it, reconstruct the QJL residual estimate with `qjl_dequantize_unit()`, multiply by `residual_norm`, add, then multiply by the outer `norm`.

This matches the paper structurally.

### 6.2 `scalar-full` in Our Code

Our `scalar-full` path corresponds to:

- `turboquant_quantize_tq4_scalar_full()` in `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:47`
- `turboquant_quantize_tq3_scalar_full()` in `src/v2/kernels/cpu/turboquant/TurboQuantQuantize.h:97`
- `turboquant_dequantize_tq4_scalar_full()` in `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:64`
- `turboquant_dequantize_tq3_scalar_full()` in `src/v2/kernels/cpu/turboquant/TurboQuantDequantize.h:124`

Implementation details:

- We still rotate the normalized input.
- We do not produce a residual sketch.
- Instead, we use all available bits for direct coordinate reconstruction in the rotated domain.
- We encode the extra top bit(s) into `qjl_signs` and mark the block with `residual_norm < 0.0f` as a sentinel that this block is not a `Q_prod` block.
- Dequantization checks the sentinel and reconstructs directly from the full scalar codebook rather than adding a QJL residual estimate.

This is not a paper term, but it is the closest repo-side engineering analogue of the paper's `Q_mse` path, adapted to our block layout.

## 7. Current Parity Evidence

### 7.1 Qwen2 (`head_dim=64`)

Qwen2 parity configs for TurboQuant are in:

- `tests/v2/integration/parity/qwen2/Test__Qwen2_SingleDevice_Parity.cpp:175`
- `tests/v2/integration/parity/qwen2/Test__Qwen2_SingleDevice_Parity.cpp:192`

Observed result from prior debugging:

- `Q_prod` on keys at `head_dim=64` was not good enough for end-to-end decode parity.
- Runtime now uses `scalar-full` for keys when `head_dim < 128`.

### 7.2 Qwen3 (`head_dim=128`)

New parity configs were added in:

- `tests/v2/integration/parity/qwen3/Test__Qwen3_SingleDevice_Parity.cpp:84`
- `tests/v2/integration/parity/qwen3/Test__Qwen3_SingleDevice_Parity.cpp:101`

Focused validation command used:

```bash
cmake --build build_v2_integration --parallel && \
./build_v2_integration/tests/v2/v2_integration_parity_qwen3_single_device \
  --gtest_filter='*PrefillParity/Qwen3_CPU_KV_TQ4:*DecodeParity/Qwen3_CPU_KV_TQ4:*PrefillParity/Qwen3_CPU_KV_TQ3:*DecodeParity/Qwen3_CPU_KV_TQ3'
```

Observed result:

- `Qwen3_CPU_KV_TQ4` prefill: passed.
- `Qwen3_CPU_KV_TQ3` prefill: passed.
- `Qwen3_CPU_KV_TQ4` decode: failed.
- `Qwen3_CPU_KV_TQ3` decode: failed.

Key decode summaries:

- `Qwen3_CPU_KV_TQ4`: `Steps=0/5  AvgCosine=0.6733  Top1=40.0%  RefInTop3=2/5  Top5=100.0%`
- `Qwen3_CPU_KV_TQ3`: `Steps=0/5  AvgCosine=0.4886  Top1=40.0%  RefInTop3=2/5  Top5=60.0%`

Interpretation:

- At `head_dim=128`, our current `Q_prod` implementation is still not preserving decode quality in Qwen3.
- The failure is not a broad pipeline break, because prefill is green.
- The failure is specifically in incremental decode behavior with cached quantized K/V.

## 8. Why This Is Suspicious Relative to the Paper

If the paper's `Q_prod` logic is implemented correctly, the small-bit distortion claims at `b=4` and `b=3` are much smaller than what the parity failures imply.

For unit-norm query vectors, the paper's fine-grained `Q_prod` bounds imply approximately:

- `TQ4` / `b=4`: `D_prod ≈ 0.047 / d`
- `TQ3` / `b=3`: `D_prod ≈ 0.18 / d`

At `d = 128`, those become approximately:

- `TQ4`: `0.000367`
- `TQ3`: `0.001406`

Even allowing for non-unit query norms and model sensitivity, current decode divergence is far worse than what these asymptotic predictions suggest. That does not prove the paper should pass our parity gate automatically, but it strongly suggests that we should keep searching for implementation mismatches before concluding the theory is irrelevant.

## 9. Highest-Value Inconsistency Checks for the Next Agent

### 9.1 Residual-Sketch Scaling Consistency

Paper intent:

- `Q_prod` stores an `(b-1)`-bit MSE approximation.
- It applies QJL to the residual.
- Dequantization reconstructs the residual estimator and scales it by the residual norm.

Things to verify in code:

- Whether `qjl_quantize_signs()` and `qjl_dequantize_unit()` exactly implement the paper's `Q_qjl` and `Q_qjl^{-1}` scaling.
- Whether the `pi/(2d)` factor is represented correctly and only once.
- Whether the residual is normalized in the correct basis before sign quantization.
- Whether the final residual contribution is multiplied by the correct norm exactly once.

### 9.2 Basis Consistency of the Residual

The paper's presentation is easy to misread here.

Things to verify:

- Our residual is formed after inverse-rotating the MSE reconstruction back to original basis.
- QJL is then applied to that residual in the original basis.
- Dequantization adds a QJL residual estimate in that same basis.

This is structurally plausible, but it is worth checking whether the intended paper interpretation instead assumes a different placement of the residual sketch relative to the rotation.

### 9.3 Context Derivation Strategy

Current code derives contexts twice:

- global context -> layer context
- layer context -> head context

Relevant files:

- `src/v2/kernels/cpu/turboquant/TurboQuantContext.h:64`
- `src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp:950`
- `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp:239`

Questions to answer:

- Does the paper require one fixed random rotation and projection per quantizer instance, rather than nested re-derivation?
- Is head-specific randomization helping or hurting unbiasedness and variance?
- Are we accidentally making quantization and dequantization consistent with each other but inconsistent with the theoretical regime the paper analyzes?

### 9.4 `Q_mse` Versus Our `scalar-full`

Our `scalar-full` is not just the paper's `Q_mse` dropped into a block type. It also reuses storage fields:

- low bits go into `mse_indices`
- top bit(s) go into `qjl_signs`
- `residual_norm < 0` is a sentinel for direct reconstruction mode

That is fine as an engineering representation, but the next agent should confirm that:

- no decode path accidentally interprets a scalar-full block as a residual-sketch block,
- no ring-cache copy or view path strips or mutates the sign-bit payload,
- no tensor dequant cache uses a stale context when re-reading scalar-full rows.

### 9.5 Decode-Only Failure Focus

Because prefill passes and decode fails, the next agent should focus on:

- incremental append semantics,
- partial-row dequantization and persistence,
- cached-FP32 buffer reuse in `AttentionComputeStage`,
- position / RoPE / per-step sensitivity in Qwen3.

The most likely productive comparison is not random synthetic vectors. It is actual Qwen3 decode-step K vectors, queried by the exact Q vectors used in the failing step.

## 10. Recommended Next Debugging Loop

1. Add a Qwen3-specific regression similar to the existing Qwen2 decode-logit regression in `tests/v2/unit/Test__TurboQuantRoundtrip.cpp`.

2. For one failing decode step, compare on actual model activations:

- FP32 key vector
- `Q_prod` dequantized key vector
- `scalar-full` dequantized key vector
- per-head dot product errors against the exact query vector
- resulting softmax rank shifts

3. Verify whether `Q_prod` is already losing at the raw attention-score level before any softmax or LM head effects.

4. Temporarily force Qwen3 `head_dim=128` keys to `scalar-full` and rerun the same focused parity slice.

5. If scalar-full recovers Qwen3 decode parity, the problem is almost certainly in the current `Q_prod` implementation or its variance behavior, not in the surrounding KV-cache plumbing.

6. If scalar-full still fails, the problem is in the broader TurboQuant integration path for Qwen3 decode rather than specifically in residual QJL correction.

## 11. Practical Commands

Build integration tests:

```bash
cmake --build build_v2_integration --parallel
```

Run just the Qwen3 TurboQuant parity cases:

```bash
./build_v2_integration/tests/v2/v2_integration_parity_qwen3_single_device \
  --gtest_filter='*PrefillParity/Qwen3_CPU_KV_TQ4:*DecodeParity/Qwen3_CPU_KV_TQ4:*PrefillParity/Qwen3_CPU_KV_TQ3:*DecodeParity/Qwen3_CPU_KV_TQ3'
```

Run Qwen2 TurboQuant parity cases for comparison:

```bash
./build_v2_integration/tests/v2/v2_integration_parity_qwen2_single_device \
  --gtest_filter='*PrefillParity/CPU_KV_TQ4:*DecodeParity/CPU_KV_TQ4:*PrefillParity/CPU_KV_TQ3:*DecodeParity/CPU_KV_TQ3'
```

## 12. Bottom Line for the Next Agent

Do not assume the paper is wrong and do not assume the parity gate is too strict.

The right next move is to iterate on implementation consistency against the paper's `Q_mse` and `Q_prod` constructions until one of these becomes clearly true:

- the current `Q_prod` implementation is inconsistent with the paper and can be fixed,
- the current `Q_prod` implementation is consistent with the paper but still unsuitable for this LLM decode setting,
- or the decode failure is actually caused by an integration bug outside the core quantizer math.

At this point, the evidence strongly favors continuing the inconsistency hunt.