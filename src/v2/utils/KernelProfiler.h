/**
 * @file KernelProfiler.h
 * @brief Per-kernel timing infrastructure for decode performance analysis
 * @author David Sanftenberg
 *
 * Provides thread-safe, low-overhead profiling for kernel operations.
 * Enable via LLAMINAR_PROFILE_KERNELS=1 environment variable.
 *
 * Usage:
 *   // At start of kernel operation
 *   KERNEL_PROFILE_BEGIN(KernelType::GEMM_Q8);
 *
 *   // ... kernel work ...
 *
 *   // At end of kernel operation
 *   KERNEL_PROFILE_END(KernelType::GEMM_Q8);
 *
 *   // Or use RAII scoped timing:
 *   {
 *       KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);
 *       // ... kernel work ...
 *   }
 *
 * At end of inference, call KernelProfiler::printSummary() to see results.
 */
#pragma once

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>

#include "DebugEnv.h"

namespace llaminar2
{

    /**
     * @brief Kernel type categories for profiling
     */
    enum class KernelType : uint8_t
    {
        // GEMM variants
        GEMM_FP32 = 0, ///< FP32 GEMM (OpenBLAS fallback)
        GEMM_Q8,       ///< Q8_1 quantized GEMM (JIT microkernel)
        GEMM_IQ4,      ///< IQ4_NL quantized GEMM

        // Attention
        ATTENTION,      ///< Full attention block (Q*K^T, softmax, *V)
        ATTENTION_QK,   ///< Q*K^T matmul only
        ATTENTION_SOFT, ///< Softmax only
        ATTENTION_V,    ///< Attention * V only

        // Normalization
        RMS_NORM, ///< RMS normalization

        // Activation / FFN
        SWIGLU,   ///< SwiGLU activation
        FFN_UP,   ///< FFN up projection
        FFN_DOWN, ///< FFN down projection
        FFN_GATE, ///< FFN gate projection

        // Positional encoding
        ROPE, ///< Rotary position embedding

        // Quantization
        QUANTIZE_Q8, ///< FP32 -> Q8_1 quantization

        // Misc
        EMBEDDING,    ///< Token embedding lookup
        LM_HEAD,      ///< Final LM head projection
        RESIDUAL_ADD, ///< Residual connection add
        SOFTMAX,      ///< Standalone softmax

        COUNT ///< Sentinel for array sizing
    };

    /**
     * @brief Get human-readable name for a kernel type
     */
    inline const char *kernelTypeName(KernelType type)
    {
        switch (type)
        {
        case KernelType::GEMM_FP32:
            return "GEMM_FP32";
        case KernelType::GEMM_Q8:
            return "GEMM_Q8";
        case KernelType::GEMM_IQ4:
            return "GEMM_IQ4";
        case KernelType::ATTENTION:
            return "ATTENTION";
        case KernelType::ATTENTION_QK:
            return "ATTENTION_QK";
        case KernelType::ATTENTION_SOFT:
            return "ATTENTION_SOFT";
        case KernelType::ATTENTION_V:
            return "ATTENTION_V";
        case KernelType::RMS_NORM:
            return "RMS_NORM";
        case KernelType::SWIGLU:
            return "SWIGLU";
        case KernelType::FFN_UP:
            return "FFN_UP";
        case KernelType::FFN_DOWN:
            return "FFN_DOWN";
        case KernelType::FFN_GATE:
            return "FFN_GATE";
        case KernelType::ROPE:
            return "ROPE";
        case KernelType::QUANTIZE_Q8:
            return "QUANTIZE_Q8";
        case KernelType::EMBEDDING:
            return "EMBEDDING";
        case KernelType::LM_HEAD:
            return "LM_HEAD";
        case KernelType::RESIDUAL_ADD:
            return "RESIDUAL_ADD";
        case KernelType::SOFTMAX:
            return "SOFTMAX";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Thread-safe kernel profiling accumulator
     *
     * Uses atomic operations for thread-safe accumulation without locks.
     * Each kernel type has its own accumulator to avoid false sharing.
     */
    class KernelProfiler
    {
    public:
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;
        using Duration = std::chrono::nanoseconds;

        /**
         * @brief Per-kernel-type timing statistics
         */
        struct alignas(64) KernelStats // Cache-line aligned to prevent false sharing
        {
            std::atomic<uint64_t> total_ns{0};        ///< Total time in nanoseconds
            std::atomic<uint64_t> call_count{0};      ///< Number of calls
            std::atomic<uint64_t> max_ns{0};          ///< Maximum single call time
            std::atomic<uint64_t> min_ns{UINT64_MAX}; ///< Minimum single call time
        };

        /**
         * @brief Check if profiling is enabled (from DebugEnv)
         */
        static bool isEnabled()
        {
            // Use DebugEnv for centralized configuration
            return debugEnv().profile.enabled;
        }

        /**
         * @brief Record a kernel execution time
         * @param type Kernel type
         * @param duration_ns Duration in nanoseconds
         */
        static void record(KernelType type, uint64_t duration_ns)
        {
            if (!isEnabled())
                return;

            auto &stats = getStats(type);
            stats.total_ns.fetch_add(duration_ns, std::memory_order_relaxed);
            stats.call_count.fetch_add(1, std::memory_order_relaxed);

            // Update max (lock-free)
            uint64_t current_max = stats.max_ns.load(std::memory_order_relaxed);
            while (duration_ns > current_max &&
                   !stats.max_ns.compare_exchange_weak(current_max, duration_ns,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed))
            {
            }

            // Update min (lock-free)
            uint64_t current_min = stats.min_ns.load(std::memory_order_relaxed);
            while (duration_ns < current_min &&
                   !stats.min_ns.compare_exchange_weak(current_min, duration_ns,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed))
            {
            }
        }

        /**
         * @brief Get stats for a kernel type (for external access)
         */
        static KernelStats &getStats(KernelType type)
        {
            return getInstance().stats_[static_cast<size_t>(type)];
        }

        /**
         * @brief Reset all statistics
         */
        static void reset()
        {
            auto &inst = getInstance();
            for (auto &stats : inst.stats_)
            {
                stats.total_ns.store(0, std::memory_order_relaxed);
                stats.call_count.store(0, std::memory_order_relaxed);
                stats.max_ns.store(0, std::memory_order_relaxed);
                stats.min_ns.store(UINT64_MAX, std::memory_order_relaxed);
            }
        }

        /**
         * @brief Print formatted summary of all kernel timings
         * @param total_tokens Total tokens processed (for tok/s calculation)
         * @return Formatted string with timing breakdown
         */
        static std::string getSummary(uint64_t total_tokens = 0)
        {
            if (!isEnabled())
            {
                return "[Kernel profiling disabled. Set LLAMINAR_PROFILE_KERNELS=1 to enable]\n";
            }

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);

            oss << "\n";
            oss << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
            oss << "║                         KERNEL PROFILING SUMMARY                             ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
            oss << "║  Kernel Type      │  Calls   │  Total (ms)  │  Avg (µs)  │  Min/Max (µs)     ║\n";
            oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

            uint64_t grand_total_ns = 0;

            for (size_t i = 0; i < static_cast<size_t>(KernelType::COUNT); ++i)
            {
                const auto &stats = getStats(static_cast<KernelType>(i));
                uint64_t count = stats.call_count.load(std::memory_order_relaxed);
                if (count == 0)
                    continue;

                uint64_t total_ns = stats.total_ns.load(std::memory_order_relaxed);
                uint64_t min_ns = stats.min_ns.load(std::memory_order_relaxed);
                uint64_t max_ns = stats.max_ns.load(std::memory_order_relaxed);

                double total_ms = total_ns / 1e6;
                double avg_us = (total_ns / static_cast<double>(count)) / 1e3;
                double min_us = min_ns / 1e3;
                double max_us = max_ns / 1e3;

                grand_total_ns += total_ns;

                // Format: kernel name (16 chars), calls (8), total ms (12), avg µs (10), min/max
                oss << "║  " << std::left << std::setw(16) << kernelTypeName(static_cast<KernelType>(i))
                    << "│" << std::right << std::setw(8) << count << "  "
                    << "│" << std::setw(12) << total_ms << "  "
                    << "│" << std::setw(10) << avg_us << "  "
                    << "│" << std::setw(8) << min_us << "/" << std::setw(8) << max_us << " ║\n";
            }

            oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";

            double grand_total_ms = grand_total_ns / 1e6;
            oss << "║  TOTAL KERNEL TIME: " << std::setw(10) << grand_total_ms << " ms";

            if (total_tokens > 0)
            {
                double toks_per_sec = (total_tokens * 1e9) / static_cast<double>(grand_total_ns);
                oss << "   (" << std::setw(6) << toks_per_sec << " kernel tok/s)";
            }
            oss << std::setw(20) << " " << "║\n";

            oss << "╚══════════════════════════════════════════════════════════════════════════════╝\n";

            return oss.str();
        }

        /**
         * @brief Print summary to stderr
         */
        static void printSummary(uint64_t total_tokens = 0)
        {
            fprintf(stderr, "%s", getSummary(total_tokens).c_str());
        }

    private:
        KernelProfiler() = default;

        static KernelProfiler &getInstance()
        {
            static KernelProfiler instance;
            return instance;
        }

        std::array<KernelStats, static_cast<size_t>(KernelType::COUNT)> stats_;
    };

    /**
     * @brief RAII scoped timer for kernel profiling
     */
    class ScopedKernelTimer
    {
    public:
        explicit ScopedKernelTimer(KernelType type)
            : type_(type), enabled_(KernelProfiler::isEnabled())
        {
            if (enabled_)
            {
                start_ = KernelProfiler::Clock::now();
            }
        }

        ~ScopedKernelTimer()
        {
            if (enabled_)
            {
                auto end = KernelProfiler::Clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
                KernelProfiler::record(type_, static_cast<uint64_t>(duration.count()));
            }
        }

        // Non-copyable, non-movable
        ScopedKernelTimer(const ScopedKernelTimer &) = delete;
        ScopedKernelTimer &operator=(const ScopedKernelTimer &) = delete;

    private:
        KernelType type_;
        bool enabled_;
        KernelProfiler::TimePoint start_;
    };

    /**
     * @brief Manual begin/end timing (for cases where RAII doesn't fit)
     */
    class ManualKernelTimer
    {
    public:
        void begin()
        {
            if (KernelProfiler::isEnabled())
            {
                start_ = KernelProfiler::Clock::now();
            }
        }

        void end(KernelType type)
        {
            if (KernelProfiler::isEnabled())
            {
                auto end = KernelProfiler::Clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
                KernelProfiler::record(type, static_cast<uint64_t>(duration.count()));
            }
        }

    private:
        KernelProfiler::TimePoint start_;
    };

} // namespace llaminar2

// ============================================================================
// Convenience macros for kernel profiling
// ============================================================================

/**
 * @brief Scoped kernel profiling (RAII-based)
 *
 * Usage:
 *   void myKernel() {
 *       KERNEL_PROFILE_SCOPE(KernelType::GEMM_Q8);
 *       // ... kernel work, automatically timed until scope exit ...
 *   }
 */
#define KERNEL_PROFILE_SCOPE(kernel_type) \
    ::llaminar2::ScopedKernelTimer _kernel_timer_##__LINE__(kernel_type)

/**
 * @brief Manual kernel profiling begin
 *
 * Usage:
 *   KERNEL_PROFILE_BEGIN(timer_name);
 *   // ... kernel work ...
 *   KERNEL_PROFILE_END(timer_name, KernelType::GEMM_Q8);
 */
#define KERNEL_PROFILE_BEGIN(timer_name)       \
    ::llaminar2::ManualKernelTimer timer_name; \
    timer_name.begin()

#define KERNEL_PROFILE_END(timer_name, kernel_type) \
    timer_name.end(kernel_type)
