/**
 * @file NullWeightStreamer.h
 * @brief No-op weight streamer for RESIDENT mode (weights fit in GPU VRAM)
 * @author David Sanftenberg
 * @date January 2026
 *
 * This file provides a null implementation of IWeightStreamer that is used when
 * weights are permanently resident on device (Option B - RESIDENT mode). This
 * allows the codebase to have consistent streaming hooks without any overhead
 * when streaming isn't actually needed.
 *
 * Design Philosophy:
 * - All methods are inline no-ops or return appropriate defaults
 * - Marked `final` to enable devirtualization by the compiler
 * - Zero runtime overhead when inlined (no virtual dispatch)
 * - Header-only implementation (no .cpp file needed)
 *
 * Usage:
 *   // When weights fit in VRAM, use NullWeightStreamer
 *   auto streamer = std::make_unique<NullWeightStreamer>(vram_budget);
 *   
 *   // All methods work but do nothing:
 *   streamer->ensureLayerOnDevice(0, gpu);  // Always returns true
 *   streamer->prefetchLayer(1, gpu);         // No-op
 *   streamer->isLayerCached(0, gpu);        // Always returns true
 *
 * @see IWeightStreamer.h for the interface definition
 * @see docs/v2/OPTION_B_WEIGHT_STREAMING_DESIGN.md for design details
 */

#pragma once

#include "IWeightStreamer.h"

namespace llaminar2
{

    /**
     * @brief No-op weight streamer for when weights are permanently resident on device
     *
     * This implementation is used when the model's weights fit entirely in GPU VRAM.
     * All methods are trivial no-ops that return appropriate default values:
     * - ensureLayerOnDevice: Always returns true (weights already resident)
     * - isLayerCached: Always returns true (everything is resident)
     * - prefetch/release/evict: No-ops (nothing to stream)
     * - stats: Returns empty/zero statistics
     *
     * Marking the class `final` enables the compiler to devirtualize calls when
     * the concrete type is known, eliminating virtual dispatch overhead.
     */
    class NullWeightStreamer final : public IWeightStreamer
    {
    public:
        /**
         * @brief Construct a null weight streamer
         *
         * @param memory_budget Optional memory budget to report (defaults to 0)
         *                      This is purely informational - no memory management occurs
         */
        explicit NullWeightStreamer(size_t memory_budget = 0) noexcept
            : budget_(memory_budget)
        {
        }

        ~NullWeightStreamer() override = default;

        // Non-copyable, non-movable (interface contract)
        NullWeightStreamer(const NullWeightStreamer&) = delete;
        NullWeightStreamer& operator=(const NullWeightStreamer&) = delete;
        NullWeightStreamer(NullWeightStreamer&&) = delete;
        NullWeightStreamer& operator=(NullWeightStreamer&&) = delete;

        // =====================================================================
        // Layer Management - All no-ops or return success
        // =====================================================================

        /**
         * @brief Always returns true - weights are already resident
         */
        bool ensureLayerOnDevice(int /*layer_idx*/, DeviceId /*device*/) override
        {
            return true;
        }

        /**
         * @brief No-op - nothing to prefetch when weights are resident
         */
        void prefetchLayer(int /*layer_idx*/, DeviceId /*device*/) override
        {
            // No-op: weights are already on device
        }

        /**
         * @brief No-op - nothing to release when weights are resident
         */
        void releaseLayer(int /*layer_idx*/) override
        {
            // No-op: weights remain on device
        }

        // =====================================================================
        // Phase Management - No-op
        // =====================================================================

        /**
         * @brief No-op - phase transitions don't affect resident weights
         */
        void onPhaseTransition(InferencePhase /*old_phase*/,
                               InferencePhase /*new_phase*/) override
        {
            // No-op: resident weights don't change behavior by phase
        }

        // =====================================================================
        // Memory Management - Returns zeros/defaults
        // =====================================================================

        /**
         * @brief Returns 0 - no streaming cache exists
         */
        size_t currentDeviceMemoryUsage() const override
        {
            return 0;
        }

        /**
         * @brief Returns the configured memory budget
         */
        size_t memoryBudget() const override
        {
            return budget_;
        }

        /**
         * @brief Returns false - nothing to evict when weights are resident
         */
        bool evictLayer(int /*layer_idx*/) override
        {
            return false;
        }

        /**
         * @brief No-op - no cache to clear
         */
        void clearCache() override
        {
            // No-op: no streaming cache exists
        }

        // =====================================================================
        // Synchronization - No-op
        // =====================================================================

        /**
         * @brief No-op - no pending transfers when weights are resident
         */
        void synchronize() override
        {
            // No-op: no async transfers in progress
        }

        // =====================================================================
        // Diagnostics - Return defaults indicating everything is cached
        // =====================================================================

        /**
         * @brief Always returns true - all layers are resident
         */
        bool isLayerCached(int /*layer_idx*/, DeviceId /*device*/) const override
        {
            return true;
        }

        /**
         * @brief Always returns false - no prefetch operations occur
         */
        bool isPrefetchInProgress(int /*layer_idx*/) const override
        {
            return false;
        }

        /**
         * @brief Returns empty statistics - no streaming activity
         */
        StreamingStats stats() const override
        {
            return StreamingStats{};
        }

        /**
         * @brief No-op - no statistics to reset
         */
        void resetStats() override
        {
            // No-op: no statistics accumulated
        }

    private:
        size_t budget_; ///< Configured memory budget (informational only)
    };

} // namespace llaminar2
