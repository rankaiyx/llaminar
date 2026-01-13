/**
 * @file StreamingConfigFromEnv.h
 * @brief Helper function to create StreamingConfig from environment variables
 * @author GitHub Copilot
 * @date January 2026
 *
 * This header provides the implementation of createStreamingConfigFromEnv(),
 * which bridges the DebugEnv system with the StreamingConfig used by
 * IWeightStreamer implementations.
 *
 * @see IWeightStreamer.h for StreamingConfig struct
 * @see DebugEnv.h for StreamingEnv configuration
 */

#pragma once

#include "IWeightStreamer.h"
#include "../utils/DebugEnv.h"

namespace llaminar2
{

    /**
     * @brief Create a StreamingConfig from environment variables
     *
     * Maps the StreamingEnv configuration from DebugEnv to the StreamingConfig
     * structure used by IWeightStreamer implementations.
     *
     * Environment Variable Mapping:
     * - LLAMINAR_WEIGHT_STREAMING → (used to check if streaming is enabled)
     * - LLAMINAR_STREAM_MEMORY_MB → gpu_memory_budget (converted to bytes)
     * - LLAMINAR_STREAM_PREFETCH_DEPTH → prefetch_depth
     * - LLAMINAR_STREAM_EVICTION_POLICY → eviction_policy (lru/fifo/none)
     * - LLAMINAR_STREAM_VERBOSE → log_transfer_stats
     *
     * @return StreamingConfig populated from environment variables
     *
     * @code
     *   // Usage example:
     *   if (debugEnv().streaming.enabled) {
     *       auto config = createStreamingConfigFromEnv();
     *       auto streamer = std::make_unique<LayerWeightStreamer>(config, weight_manager);
     *   }
     * @endcode
     */
    inline StreamingConfig createStreamingConfigFromEnv()
    {
        const auto &env = debugEnv().streaming;
        StreamingConfig config;

        // Memory budget: convert MB to bytes (0 = auto-detect)
        if (env.memory_budget_mb > 0)
        {
            config.gpu_memory_budget = env.memory_budget_mb * 1024 * 1024;
        }
        else
        {
            config.gpu_memory_budget = 0; // 0 = auto-detect
        }

        // Prefetch depth
        config.prefetch_depth = static_cast<size_t>(env.prefetch_depth);
        config.enable_prefetch = (env.prefetch_depth > 0);

        // Eviction policy
        if (env.eviction_policy == "lru")
        {
            config.eviction_policy = StreamingEvictionPolicy::LRU;
        }
        else if (env.eviction_policy == "fifo")
        {
            config.eviction_policy = StreamingEvictionPolicy::FIFO;
        }
        else if (env.eviction_policy == "none")
        {
            config.eviction_policy = StreamingEvictionPolicy::NONE;
        }
        // Default is LRU (already set in StreamingConfig)

        // Verbose logging
        config.log_transfer_stats = env.verbose;

        return config;
    }

    /**
     * @brief Check if weight streaming is enabled via environment
     *
     * Convenience function to check LLAMINAR_WEIGHT_STREAMING without
     * creating a full StreamingConfig.
     *
     * @return true if weight streaming is enabled
     */
    inline bool isWeightStreamingEnabled()
    {
        return debugEnv().streaming.enabled;
    }

} // namespace llaminar2
