/**
 * @file CoherenceAuditLog.h
 * @brief Per-tensor ring buffer recording coherence state transitions.
 *
 * When LLAMINAR_ASSERTIONS_ACTIVE and LLAMINAR_COHERENCE_AUDIT=1,
 * every call to applyCoherenceOp_() records an entry. The log is
 * dumped automatically when TensorVerification detects a failure.
 *
 * Zero overhead when disabled (no string formatting, no allocation).
 */

#pragma once

#include "CoherenceState.h"
#include "utils/Assertions.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <ostream>
#include <thread>

namespace llaminar2
{

    struct CoherenceAuditEntry
    {
        TensorCoherenceState from_state = TensorCoherenceState::INVALID;
        TensorCoherenceState to_state = TensorCoherenceState::INVALID;
        CoherenceOp op{};
        const char *caller = nullptr; ///< __builtin_FUNCTION() of the callsite
        uint64_t timestamp_ns = 0;
        uint32_t thread_id = 0;
        bool valid_transition = true;
    };

    class CoherenceAuditLog
    {
    public:
        static constexpr size_t RING_SIZE = 32;

        void record(TensorCoherenceState from, TensorCoherenceState to,
                    CoherenceOp op, const char *caller, bool valid)
        {
            auto &e = ring_[head_ % RING_SIZE];
            e.from_state = from;
            e.to_state = to;
            e.op = op;
            e.caller = caller;
            e.timestamp_ns = static_cast<uint64_t>(
                std::chrono::steady_clock::now().time_since_epoch().count());
            e.thread_id = static_cast<uint32_t>(
                std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFFFFF);
            e.valid_transition = valid;
            ++head_;
            if (count_ < RING_SIZE)
                ++count_;
        }

        /**
         * @brief Dump the ring buffer to an output stream.
         */
        void dump(std::ostream &os, const char *tensor_name, const void *tensor_ptr) const
        {
            os << "\n[COHERENCE_AUDIT] Tensor '" << (tensor_name ? tensor_name : "<unnamed>")
               << "' (" << tensor_ptr << ") — last " << count_ << " transitions:\n";

            // Compute base timestamp for relative display
            uint64_t base_ts = 0;
            if (count_ > 0)
            {
                size_t oldest = (head_ >= count_) ? (head_ - count_) : 0;
                base_ts = ring_[oldest % RING_SIZE].timestamp_ns;
            }

            for (size_t i = 0; i < count_; ++i)
            {
                size_t idx = (head_ - count_ + i) % RING_SIZE;
                const auto &e = ring_[idx];
                double rel_ms = static_cast<double>(e.timestamp_ns - base_ts) / 1e6;
                os << "  #" << i
                   << "  " << to_string(e.from_state) << " -> " << to_string(e.to_state)
                   << "  op=" << to_string(e.op)
                   << "  caller=" << (e.caller ? e.caller : "?")
                   << "  t=+" << rel_ms << "ms"
                   << "  tid=0x" << std::hex << e.thread_id << std::dec;
                if (!e.valid_transition)
                    os << "  *** INVALID ***";
                os << "\n";
            }
        }

        size_t count() const { return count_; }

        bool hasTransition(TensorCoherenceState from, TensorCoherenceState to) const
        {
            for (size_t i = 0; i < count_; ++i)
            {
                size_t idx = (head_ - count_ + i) % RING_SIZE;
                if (ring_[idx].from_state == from && ring_[idx].to_state == to)
                    return true;
            }
            return false;
        }

    private:
        std::array<CoherenceAuditEntry, RING_SIZE> ring_{};
        size_t head_ = 0;
        size_t count_ = 0;
    };

} // namespace llaminar2
