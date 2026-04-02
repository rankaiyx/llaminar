/**
 * @file ExpertWeightCache.cpp
 * @brief LRU/FIFO expert weight cache implementation
 */

#include "ExpertWeightCache.h"
#include <stdexcept>

namespace llaminar2
{

    ExpertWeightCache::ExpertWeightCache(
        std::shared_ptr<IExpertWeightStorage> storage,
        size_t capacity_bytes,
        EvictionPolicy policy)
        : storage_(std::move(storage)), capacity_bytes_(capacity_bytes), policy_(policy)
    {
        if (!storage_)
            throw std::invalid_argument("ExpertWeightCache: storage must not be null");
    }

    ExpertWeightHandle ExpertWeightCache::get(const ExpertWeightKey &key)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            // Cache hit
            hits_++;
            if (policy_ == EvictionPolicy::LRU)
                touchEntry(key);
            return it->second.first.handle;
        }

        // Cache miss
        misses_++;

        // Load from storage
        ExpertWeightHandle handle = storage_->load(key);
        if (!handle.valid())
            return handle; // Load failed

        // Evict as needed
        evictIfNeeded(handle.total_bytes);

        // Insert into cache
        lru_list_.push_front(key);
        CacheEntry entry{handle};
        cache_[key] = {entry, lru_list_.begin()};
        current_bytes_ += handle.total_bytes;

        return handle;
    }

    void ExpertWeightCache::prefetch(const std::vector<ExpertWeightKey> &keys)
    {
        for (const auto &key : keys)
            get(key); // get() handles caching internally
    }

    bool ExpertWeightCache::evict(const ExpertWeightKey &key)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = cache_.find(key);
        if (it == cache_.end())
            return false;

        // Release from storage
        storage_->release(it->second.first.handle);
        current_bytes_ -= it->second.first.handle.total_bytes;

        // Remove from LRU list
        lru_list_.erase(it->second.second);
        cache_.erase(it);
        evictions_++;

        return true;
    }

    void ExpertWeightCache::clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &[key, entry_pair] : cache_)
            storage_->release(entry_pair.first.handle);

        cache_.clear();
        lru_list_.clear();
        current_bytes_ = 0;
    }

    ExpertCacheStats ExpertWeightCache::stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return ExpertCacheStats{
            .hits = hits_,
            .misses = misses_,
            .evictions = evictions_,
            .current_entries = static_cast<uint64_t>(cache_.size()),
            .current_bytes = current_bytes_,
            .capacity_bytes = capacity_bytes_,
        };
    }

    bool ExpertWeightCache::isCached(const ExpertWeightKey &key) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.find(key) != cache_.end();
    }

    size_t ExpertWeightCache::size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }

    void ExpertWeightCache::evictIfNeeded(size_t needed_bytes)
    {
        // mutex_ already held by caller
        while (current_bytes_ + needed_bytes > capacity_bytes_ && !lru_list_.empty())
        {
            // Evict from the back (LRU) or front (FIFO won't re-touch)
            const ExpertWeightKey &victim = lru_list_.back();

            auto it = cache_.find(victim);
            if (it != cache_.end())
            {
                storage_->release(it->second.first.handle);
                current_bytes_ -= it->second.first.handle.total_bytes;
                cache_.erase(it);
            }
            lru_list_.pop_back();
            evictions_++;
        }
    }

    void ExpertWeightCache::touchEntry(const ExpertWeightKey &key)
    {
        // mutex_ already held by caller
        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            // Move to front of LRU list
            lru_list_.erase(it->second.second);
            lru_list_.push_front(key);
            it->second.second = lru_list_.begin();
        }
    }

} // namespace llaminar2
