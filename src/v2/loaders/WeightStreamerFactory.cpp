/**
 * @file WeightStreamerFactory.cpp
 * @brief Implementation of WeightStreamerFactory
 * @author GitHub Copilot
 * @date January 2026
 */

#include "WeightStreamerFactory.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    std::unique_ptr<IWeightStreamer> WeightStreamerFactory::createFromEnv(
        std::shared_ptr<WeightManager> weight_manager,
        int num_layers)
    {
        if (isWeightStreamingEnabled())
        {
            // Streaming mode - validate required parameters
            if (!weight_manager)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::createFromEnv: weight_manager is required when "
                    "LLAMINAR_WEIGHT_STREAMING=1");
            }
            if (num_layers <= 0)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::createFromEnv: num_layers must be > 0 when "
                    "LLAMINAR_WEIGHT_STREAMING=1, got: " +
                    std::to_string(num_layers));
            }

            // Create config from environment variables
            StreamingConfig config = createStreamingConfigFromEnv();

            LOG_DEBUG("[WeightStreamerFactory] Creating LayerWeightStreamer (streaming mode)");
            LOG_DEBUG("[WeightStreamerFactory]   num_layers=" << num_layers);
            LOG_DEBUG("[WeightStreamerFactory]   gpu_memory_budget="
                      << (config.gpu_memory_budget / (1024 * 1024)) << " MB");
            LOG_DEBUG("[WeightStreamerFactory]   prefetch_depth=" << config.prefetch_depth);

            return std::make_unique<LayerWeightStreamer>(
                std::move(weight_manager), num_layers, config);
        }
        else
        {
            // Resident mode - no streaming needed
            LOG_DEBUG("[WeightStreamerFactory] Creating NullWeightStreamer (resident mode)");
            return std::make_unique<NullWeightStreamer>();
        }
    }

    std::unique_ptr<IWeightStreamer> WeightStreamerFactory::create(
        WeightResidencyMode mode,
        std::shared_ptr<WeightManager> weight_manager,
        int num_layers,
        const StreamingConfig &config)
    {
        switch (mode)
        {
        case WeightResidencyMode::RESIDENT:
            LOG_DEBUG("[WeightStreamerFactory] Creating NullWeightStreamer (RESIDENT mode)");
            return std::make_unique<NullWeightStreamer>();

        case WeightResidencyMode::STREAMING:
        {
            // Validate required parameters for streaming mode
            if (!weight_manager)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::create: weight_manager is required for STREAMING mode");
            }
            if (num_layers <= 0)
            {
                throw std::invalid_argument(
                    "WeightStreamerFactory::create: num_layers must be > 0 for STREAMING mode, got: " +
                    std::to_string(num_layers));
            }

            LOG_DEBUG("[WeightStreamerFactory] Creating LayerWeightStreamer (STREAMING mode)");
            LOG_DEBUG("[WeightStreamerFactory]   num_layers=" << num_layers);
            LOG_DEBUG("[WeightStreamerFactory]   gpu_memory_budget="
                      << (config.gpu_memory_budget / (1024 * 1024)) << " MB");

            return std::make_unique<LayerWeightStreamer>(
                std::move(weight_manager), num_layers, config);
        }

        case WeightResidencyMode::UNIFIED:
            // Unified memory mode - let driver handle placement
            // Use NullWeightStreamer since no explicit streaming is needed
            LOG_DEBUG("[WeightStreamerFactory] Creating NullWeightStreamer (UNIFIED mode)");
            return std::make_unique<NullWeightStreamer>();

        default:
            // Should not happen, but provide fallback
            LOG_WARN("[WeightStreamerFactory] Unknown WeightResidencyMode, defaulting to NullWeightStreamer");
            return std::make_unique<NullWeightStreamer>();
        }
    }

    WeightResidencyMode WeightStreamerFactory::detectResidencyMode(
        size_t model_memory_bytes,
        size_t available_vram_bytes,
        float headroom_fraction)
    {
        // Clamp headroom fraction to valid range [0, 1)
        if (headroom_fraction < 0.0f)
        {
            headroom_fraction = 0.0f;
        }
        if (headroom_fraction >= 1.0f)
        {
            headroom_fraction = 0.99f; // Leave at least 1% for weights
        }

        // Calculate available memory for weights after headroom
        const size_t available_for_weights = static_cast<size_t>(
            available_vram_bytes * (1.0f - headroom_fraction));

        // Edge case: no VRAM available
        if (available_for_weights == 0)
        {
            LOG_DEBUG("[WeightStreamerFactory] detectResidencyMode: no VRAM available → STREAMING");
            return WeightResidencyMode::STREAMING;
        }

        // Edge case: no model memory needed
        if (model_memory_bytes == 0)
        {
            LOG_DEBUG("[WeightStreamerFactory] detectResidencyMode: no model memory → RESIDENT");
            return WeightResidencyMode::RESIDENT;
        }

        // Simple threshold check
        if (model_memory_bytes <= available_for_weights)
        {
            LOG_DEBUG("[WeightStreamerFactory] detectResidencyMode: model fits ("
                      << (model_memory_bytes / (1024 * 1024)) << " MB <= "
                      << (available_for_weights / (1024 * 1024)) << " MB) → RESIDENT");
            return WeightResidencyMode::RESIDENT;
        }
        else
        {
            LOG_DEBUG("[WeightStreamerFactory] detectResidencyMode: model exceeds VRAM ("
                      << (model_memory_bytes / (1024 * 1024)) << " MB > "
                      << (available_for_weights / (1024 * 1024)) << " MB) → STREAMING");
            return WeightResidencyMode::STREAMING;
        }
    }

} // namespace llaminar2
