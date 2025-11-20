/**
 * @file ThreadingUtils.h
 * @brief Utilities for managing thread counts and OpenMP settings.
 *
 * @author David Sanftenberg
 * @date 2025-10-26
 */

#pragma once

#include <omp.h>

namespace llaminar2
{
    namespace utils
    {

        /**
         * @brief RAII guard to temporarily set OpenMP threads to 1.
         *
         * Useful for small workloads (e.g. single-token decode) where
         * thread synchronization overhead outweighs parallel speedup.
         */
        struct ScopedSingleThread
        {
            int original_threads;
            bool active;

            explicit ScopedSingleThread(bool enable) : active(enable)
            {
                if (active)
                {
                    original_threads = omp_get_max_threads();
                    if (original_threads > 1)
                    {
                        omp_set_num_threads(1);
                    }
                    else
                    {
                        // If already 1, no need to restore
                        active = false;
                    }
                }
            }

            ~ScopedSingleThread()
            {
                if (active)
                {
                    omp_set_num_threads(original_threads);
                }
            }

            // Prevent copying
            ScopedSingleThread(const ScopedSingleThread &) = delete;
            ScopedSingleThread &operator=(const ScopedSingleThread &) = delete;
        };

    } // namespace utils
} // namespace llaminar2
