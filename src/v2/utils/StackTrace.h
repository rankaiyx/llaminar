#pragma once

/**
 * @file StackTrace.h
 * @brief C++23 stacktrace support with fallback for older compilers
 * @author David Sanftenberg
 *
 * Provides cross-platform stack trace capture for diagnostics and debugging.
 * Uses C++23 <stacktrace> when available (GCC 13+), falls back to empty string otherwise.
 *
 * Usage:
 *   @code
 *   std::string trace = llaminar2::captureStackTrace();
 *   LOG_ERROR("Unexpected transfer detected:\n" << trace);
 *   @endcode
 */

#include <string>
#include <sstream>

// Check for C++23 stacktrace support
// GCC 13+ supports <stacktrace> with -lstdc++_libbacktrace
// BUT: We disable it when using clang/HIP because the ROCm linker can't link against it
// Also disabled via CMake LLAMINAR_DISABLE_STACKTRACE for ROCm builds
#if !defined(LLAMINAR_DISABLE_STACKTRACE) && __has_include(<stacktrace>) && (__cplusplus >= 202302L || (defined(__GNUC__) && __GNUC__ >= 13)) && !defined(__clang__)
#define LLAMINAR_HAS_STACKTRACE 1
#include <stacktrace>
#else
#define LLAMINAR_HAS_STACKTRACE 0
#endif

namespace llaminar2
{

    /**
     * @brief Capture current stack trace as a string
     *
     * @param skip_frames Number of frames to skip from top (default: 1 to skip this function)
     * @param max_frames Maximum number of frames to capture (default: 32)
     * @return Stack trace as multi-line string, or empty string if not supported
     */
    inline std::string captureStackTrace([[maybe_unused]] int skip_frames = 1,
                                         [[maybe_unused]] int max_frames = 32)
    {
#if LLAMINAR_HAS_STACKTRACE
        try
        {
            auto trace = std::stacktrace::current();
            std::ostringstream oss;

            int frame_count = 0;
            for (const auto &entry : trace)
            {
                // Skip the first N frames (usually this function and the caller)
                if (skip_frames > 0)
                {
                    --skip_frames;
                    continue;
                }

                if (frame_count >= max_frames)
                    break;

                oss << "  #" << frame_count << " ";

                // Get description (may include function name and location)
                std::string desc = entry.description();
                if (!desc.empty())
                {
                    oss << desc;
                }
                else
                {
                    oss << "??";
                }

                // Add source location if available
                std::string file = entry.source_file();
                int line = static_cast<int>(entry.source_line());
                if (!file.empty())
                {
                    oss << " at " << file;
                    if (line > 0)
                    {
                        oss << ":" << line;
                    }
                }

                oss << "\n";
                ++frame_count;
            }

            return oss.str();
        }
        catch (...)
        {
            return "(stacktrace capture failed)";
        }
#else
        return "(C++23 stacktrace not available - need GCC 13+ or compile without clang)";
#endif
    }

    /**
     * @brief Exception with embedded stack trace for transfer violations
     */
    class TransferViolationException : public std::runtime_error
    {
    public:
        TransferViolationException(const std::string &message, const std::string &stacktrace)
            : std::runtime_error(formatMessage(message, stacktrace)),
              message_(message),
              stacktrace_(stacktrace) {}

        const std::string &getStackTrace() const { return stacktrace_; }
        const std::string &getRawMessage() const { return message_; }

    private:
        static std::string formatMessage(const std::string &msg, const std::string &trace)
        {
            return msg + "\nStack trace:\n" + trace;
        }

        std::string message_;
        std::string stacktrace_;
    };

} // namespace llaminar2
