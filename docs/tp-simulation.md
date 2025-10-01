# Tensor Parallel (TP) Simulation

The TP simulation facility provides an in-process emulation of a row- or column-partitioned
attention output projection so that correctness, stitching logic, and heuristic decisions can be
iterated without launching a true multi-rank tensor parallel (TP) deployment.

## Rationale
- Rapid feedback: prototype partition heuristics and reconstruction logic without MPI collectives.
- Deterministic validation: identical random seeds yield reproducible comparisons against baseline.
- Safety net: integration tests ensure refactors to the projection path preserve parity.

## High-Level Flow
```
Baseline:  Y = A [M x K]  *  W_O [K x K]

Column Sim:
  For each part p: take columns [n_off : n_off + n_local) of W_O -> Wp
    Yp = A * Wp    (# columns reduced)
    Stitch columns into Y

Row Sim:
  For each part p: take rows [m_off : m_off + m_local) of A -> Ap
    Yp = Ap * W_O
    Stitch rows into Y
```

## Environment Flags
| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_TP_WO_SIM_ENABLE` | Enable simulation path in the attention output projection. | Off |
| `LLAMINAR_TP_WO_SIM_PARTITIONS` | Number of partitions (>=2). | 2 |
| `LLAMINAR_TP_WO_SIM_MODE` | `row`, `col`, or `auto` (heuristic). | `auto` |

Heuristic (`auto`) precedence:
1. If `d_model % parts == 0` -> column mode
2. Else if `seq_len % parts == 0` -> row mode
3. Else fallback to column

## Reconstruction Guarantees
Floating point differences are limited to rounding variance; tests assert:
- `max_abs < 1e-5`
- `rel_l2 < 1e-6`

## Key Components
| File | Responsibility |
|------|----------------|
| `src/kernels/MPIAttentionKernel.cpp` | Houses simulation entry + baseline bypass gating. |
| `src/tensors/tp_output_projection_executor.h` | Partition executor (row/col loops, helpers). |
| `tests/test_attention_tp_sim_parity.cpp` | Direct reconstruction parity (row/col/auto). |
| `tests/test_attention_tp_sim_integration.cpp` | Env-driven kernel parity test with MPI guard. |

## Test Strategy
1. Pure logic parity (manual reconstruction vs baseline GEMM).
2. Kernel integration parity (env toggles + internal path selection).
3. Future: performance counters & heuristic logging snapshots.

## Extending Simulation
| Goal | Considerations |
|------|----------------|
| Hybrid Head+TP Split | Requires combined partition spec; preserve deterministic stitching order. |
| Real Multi-Rank TP | Replace serial loop with per-rank compute + Allreduce/Allgather. |
| Streaming / Overlap | Introduce tile iterator for W_O columns; measure cache effects. |
| Mixed Precision | Keep accumulation dtype consistent with baseline to maintain parity thresholds. |

## Notes
- Simulation currently assumes all heads are locally owned (HEAD_WISE distribution) and focuses on output projection only.
- Keep environment parsing centralized: extend `debug_env.[h|cpp]` rather than ad-hoc getenv calls.
- When adding experimental flags, document them here early to avoid drift.

Maintainer: Update this document when heuristic logic or reconstruction contracts change.
