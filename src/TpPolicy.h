/**
 * @file tp_policy.h
 * @brief Tensor parallel (TP) policy helper for deciding BLAS threading & outer parallel strategy.
 * @author David Sanftenberg
 */
#pragma once
#include <cstddef>
#include <cstdlib>
#include <string>
#include <algorithm>

namespace llaminar
{

    struct TPPolicyDecision
    {
        int partitions = 1;            // how many column partitions (already chosen upstream)
        int blas_threads = 1;          // threads to give each partition's GEMM
        bool outer_parallel = false;   // whether to wrap partition loop in OpenMP parallel for
        bool force_single_pass = true; // if true, single loop; else may chunk
    };

    /**
     * Environment driven TP policy.
     * Env vars:
     *  LLAMINAR_TP_FORCE_BLAS_THREADS   -> int override for per-partition BLAS threads
     *  LLAMINAR_TP_OUTER_PARALLEL       -> when set to non-zero, attempt OpenMP outer parallel loop
     *  LLAMINAR_TP_DISABLE_OUTER_PAR    -> force disable outer parallel
     *  LLAMINAR_TP_MAX_BLAS_THREADS     -> cap for computed blas threads
     *  LLAMINAR_SEQ_LEN_HINT            -> guides small decode heuristics
     */
    inline TPPolicyDecision compute_tp_policy(int partitions, size_t seq_len, size_t k_dim, size_t d_model)
    {
        TPPolicyDecision d;
        d.partitions = std::max(1, partitions);
        // Base threads: try to infer from OMP_NUM_THREADS / partitions
        int omp_threads = 1;
        // Prefer snapshot OMP thread force if configured; else consult runtime env once (outer code already set via configureGlobalOpenMPThreads)
        const auto &snap = ::llaminar::debugEnv();
        if (snap.threading.force_threads > 0)
        {
            omp_threads = snap.threading.force_threads;
        }
        else if (const char *p = std::getenv("OMP_NUM_THREADS"))
        { // fallback (not hot loop)
            omp_threads = std::max(1, atoi(p));
        }
        int suggested = std::max(1, omp_threads / d.partitions);
        // If partitions large and suggested collapses to 1 but matrix is sizable, allow bump.
        long double work = static_cast<long double>(seq_len) * k_dim * (d_model / d.partitions);
        if (suggested == 1 && work > 32.0L * 1024 * 1024)
        {                                         // ~32M mults per partition
            suggested = std::min(omp_threads, 2); // modest bump
        }
        if (snap.tp_policy.force_blas_threads > 0)
            suggested = snap.tp_policy.force_blas_threads;
        if (snap.tp_policy.max_blas_threads > 0)
            suggested = std::min(suggested, snap.tp_policy.max_blas_threads);
        // Sequence length hint: if tiny decode (<256) force 1
        if (snap.tp_policy.seq_len_hint > 0 && snap.tp_policy.seq_len_hint < 256)
            suggested = 1;
        d.blas_threads = suggested;
        bool want_outer = snap.tp_policy.outer_parallel;
        if (snap.tp_policy.disable_outer_par)
            want_outer = false;
#ifdef _OPENMP
        if (want_outer && d.blas_threads == 1 && d.partitions > 1)
        {
            d.outer_parallel = true;
        }
#endif
        return d;
    }

} // namespace llaminar
