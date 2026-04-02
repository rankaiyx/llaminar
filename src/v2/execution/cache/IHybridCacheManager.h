/**
 * @file IHybridCacheManager.h
 * @brief Interface for hybrid KV/GDN state management
 *
 * Provides an abstract interface for managing heterogeneous layer state
 * (KV cache + GDN recurrence/conv state). Enables mocking in unit tests
 * and decouples stages from the concrete HybridCacheManager implementation.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{

    class IKVCache;
    struct GDNLayerState;
    enum class LayerStateType;

    /**
     * @brief Abstract interface for hybrid cache management
     *
     * Stages that need to query layer state type or access GDN state
     * should depend on this interface, not the concrete HybridCacheManager.
     */
    class IHybridCacheManager
    {
    public:
        virtual ~IHybridCacheManager() = default;

        /// Get the state type for a specific layer
        virtual LayerStateType getLayerStateType(int layer_idx) const = 0;

        /// Get the KV cache (for full-attention layers)
        virtual IKVCache *kvCache() const = 0;

        /// Get mutable GDN state for a specific layer (nullptr if not a GDN layer)
        virtual GDNLayerState *getGDNState(int layer_idx) = 0;

        /// Get const GDN state for a specific layer (nullptr if not a GDN layer)
        virtual const GDNLayerState *getGDNState(int layer_idx) const = 0;

        /// Reset all state (for new sequence)
        virtual void reset() = 0;

        /// Total memory for GDN states
        virtual size_t gdnMemoryBytes() const = 0;

        /// Number of KV cache layers
        virtual int kvCacheLayerCount() const = 0;

        /// Number of GDN state layers
        virtual int gdnLayerCount() const = 0;

        /// Total layer count
        virtual int totalLayers() const = 0;

        /// Map from layer_idx to KV cache layer index (-1 if not a KV cache layer)
        virtual int toKVCacheLayerIdx(int layer_idx) const = 0;
    };

} // namespace llaminar2
