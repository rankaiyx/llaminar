/**
 * @file ExpertWeightCache.h
 * @brief LRU cache for MoE expert weights with GPU offloading support
 *
 * When the model has more experts than fit in device memory, the cache
 * manages loading/evicting expert weights on demand. Expert weights are
 * stored as opaque blobs identified by (layer, expert_id).
 *
 * Design goals:
 * - O(1) hit/miss lookup
 * - LRU eviction policy (with FIFO option)
 * - Testable without real GPU memory via IExpertWeightStorage interface
 * - Thread-safe for concurrent expert execution
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Key for an expert's weight set
     */
    struct ExpertWeightKey
    {
        int layer_idx;
        int expert_id;

        bool operator==(const ExpertWeightKey &o) const
        {
            return layer_idx == o.layer_idx && expert_id == o.expert_id;
        }
    };

    /**
     * @brief Hash for ExpertWeightKey
     */
    struct ExpertWeightKeyHash
    {
        size_t operator()(const ExpertWeightKey &k) const
        {
            return std::hash<int>()(k.layer_idx) ^ (std::hash<int>()(k.expert_id) << 16);
        }
    };

    /**
     * @brief Opaque handle to loaded expert weights
     *
     * The cache returns these handles. Callers hold them during expert
     * execution; the handle prevents eviction while in use.
     */
    struct ExpertWeightHandle
    {
        ExpertWeightKey key;
        const void *gate_w = nullptr; ///< Pointer to gate projection weights
        const void *up_w = nullptr;   ///< Pointer to up projection weights
        const void *down_w = nullptr; ///< Pointer to down projection weights
        size_t total_bytes = 0;       ///< Total memory occupied by this expert

        bool valid() const { return gate_w != nullptr; }
    };

    /**
     * @brief Interface for actual weight storage (host, device, etc.)
     *
     * Mockable for testing without real GPU memory.
     */
    class IExpertWeightStorage
    {
    public:
        virtual ~IExpertWeightStorage() = default;

        /// Load expert weights into storage, return handle
        virtual ExpertWeightHandle load(const ExpertWeightKey &key) = 0;

        /// Release expert weights from storage
        virtual void release(const ExpertWeightHandle &handle) = 0;

        /// Bytes per expert (for capacity planning)
        virtual size_t bytesPerExpert() const = 0;
    };

    /**
     * @brief Eviction policy
     */
    enum class EvictionPolicy
    {
        LRU,  ///< Least Recently Used
        FIFO, ///< First In, First Out
    };

    /**
     * @brief Cache statistics
     */
    struct ExpertCacheStats
    {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t current_entries = 0;
        size_t current_bytes = 0;
        size_t capacity_bytes = 0;

        double hitRate() const
        {
            uint64_t total = hits + misses;
            return total > 0 ? static_cast<double>(hits) / total : 0.0;
        }
    };

    /**
     * @brief LRU/FIFO cache for expert weights
     *
     * Thread-safe. Callers acquire handles via get(), which either
     * returns a cached entry or loads a new one (potentially evicting).
     */
    class ExpertWeightCache
    {
    public:
        /**
         * @param storage       Backend that loads/releases weights
         * @param capacity_bytes Maximum memory budget for cached weights
         * @param policy        Eviction policy (default LRU)
         */
        ExpertWeightCache(
            std::shared_ptr<IExpertWeightStorage> storage,
            size_t capacity_bytes,
            EvictionPolicy policy = EvictionPolicy::LRU);

        /**
         * @brief Get expert weights, loading if necessary
         *
         * @param key Expert to fetch
         * @return Handle to weights (valid until next call that might evict)
         */
        ExpertWeightHandle get(const ExpertWeightKey &key);

        /**
         * @brief Prefetch experts into cache without returning handles
         */
        void prefetch(const std::vector<ExpertWeightKey> &keys);

        /**
         * @brief Explicitly evict an expert from cache
         */
        bool evict(const ExpertWeightKey &key);

        /**
         * @brief Clear all cached entries
         */
        void clear();

        /**
         * @brief Get current cache statistics
         */
        ExpertCacheStats stats() const;

        /**
         * @brief Check if expert is currently cached
         */
        bool isCached(const ExpertWeightKey &key) const;

        /**
         * @brief Number of currently cached experts
         */
        size_t size() const;

    private:
        void evictIfNeeded(size_t needed_bytes);
        void touchEntry(const ExpertWeightKey &key);

        struct CacheEntry
        {
            ExpertWeightHandle handle;
        };

        std::shared_ptr<IExpertWeightStorage> storage_;
        size_t capacity_bytes_;
        EvictionPolicy policy_;

        // LRU list: front = most recently used, back = least recently used
        using LRUList = std::list<ExpertWeightKey>;
        LRUList lru_list_;

        // Map from key to (cache entry, LRU list iterator)
        using CacheMap = std::unordered_map<
            ExpertWeightKey,
            std::pair<CacheEntry, LRUList::iterator>,
            ExpertWeightKeyHash>;
        CacheMap cache_;

        mutable std::mutex mutex_;
        size_t current_bytes_ = 0;

        // Stats
        mutable uint64_t hits_ = 0;
        mutable uint64_t misses_ = 0;
        uint64_t evictions_ = 0;
    };

} // namespace llaminar2
