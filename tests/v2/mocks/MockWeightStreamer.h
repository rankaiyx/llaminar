/**
 * @file MockWeightStreamer.h
 * @brief Mock weight streamer for unit testing weight streaming logic
 *
 * This mock enables:
 * - Testing streaming logic without actual GPU memory management
 * - Simulating prefetch and eviction behavior
 * - Tracking streaming operations for verification
 * - Testing different eviction policies (LRU, FIFO)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "loaders/IWeightStreamer.h"
#include "backends/DeviceId.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <functional>
#include <chrono>
#include <memory>

namespace llaminar2::test {

/**
 * @brief Builder for MockWeightStreamer
 */
class MockWeightStreamerBuilder;

/**
 * @brief Mock weight streamer for testing streaming behavior
 *
 * Provides controllable behavior for testing:
 * - Configurable memory budget and layer sizes
 * - Callbacks for prefetch/evict operations
 * - LRU or FIFO eviction policy simulation
 * - Statistics tracking for verification
 *
 * Usage:
 * @code
 * auto mock = MockWeightStreamerBuilder()
 *     .setNumLayers(24)
 *     .setLayerSize(100 * 1024 * 1024)  // 100MB per layer
 *     .setMemoryBudget(300 * 1024 * 1024)  // 300MB budget (3 layers)
 *     .setEvictionPolicy(StreamingEvictionPolicy::LRU)
 *     .build();
 *
 * // Set callbacks to track operations
 * std::vector<int> prefetched_layers;
 * mock->set_prefetch_callback([&](int layer) { prefetched_layers.push_back(layer); });
 *
 * // Test streaming
 * mock->ensureLayerOnDevice(0, DeviceId::cuda(0));
 * mock->prefetchLayer(1, DeviceId::cuda(0));
 * @endcode
 */
class MockWeightStreamer : public IWeightStreamer {
public:
    // =========================================================================
    // Factory Methods
    // =========================================================================

    /**
     * @brief Create a streaming-disabled mock (all layers resident)
     */
    static std::shared_ptr<MockWeightStreamer> createResident();

    /**
     * @brief Create a streaming mock with default configuration
     * @param num_layers Total layers in model
     * @param cache_capacity Number of layers that fit in cache
     */
    static std::shared_ptr<MockWeightStreamer> createStreaming(
        int num_layers,
        int cache_capacity = 3);

    // =========================================================================
    // Construction
    // =========================================================================

    MockWeightStreamer();
    ~MockWeightStreamer() override = default;

    // =========================================================================
    // IWeightStreamer Implementation - Layer Management
    // =========================================================================

    bool ensureLayerOnDevice(int layer_idx, DeviceId device) override;
    void prefetchLayer(int layer_idx, DeviceId device) override;
    void releaseLayer(int layer_idx) override;

    // =========================================================================
    // IWeightStreamer Implementation - Phase Management
    // =========================================================================

    void onPhaseTransition(InferencePhase old_phase, InferencePhase new_phase) override;

    // =========================================================================
    // IWeightStreamer Implementation - Memory Management
    // =========================================================================

    size_t currentDeviceMemoryUsage() const override;
    size_t memoryBudget() const override { return memory_budget_; }
    bool evictLayer(int layer_idx) override;
    void clearCache() override;

    // =========================================================================
    // IWeightStreamer Implementation - Synchronization
    // =========================================================================

    void synchronize() override;

    // =========================================================================
    // IWeightStreamer Implementation - Diagnostics
    // =========================================================================

    bool isLayerCached(int layer_idx, DeviceId device) const override;
    bool isPrefetchInProgress(int layer_idx) const override;
    StreamingStats stats() const override { return stats_; }
    void resetStats() override;

    // =========================================================================
    // Mock Configuration API
    // =========================================================================

    /**
     * @brief Enable or disable streaming mode
     * When disabled, all layers are considered resident
     */
    void setStreamingEnabled(bool enabled) { streaming_enabled_ = enabled; }
    bool isStreamingEnabled() const { return streaming_enabled_; }

    /**
     * @brief Set number of layers in the model
     */
    void setNumLayers(int num_layers) { num_layers_ = num_layers; }
    int numLayers() const { return num_layers_; }

    /**
     * @brief Set memory budget for the cache
     */
    void setMemoryBudget(size_t budget) { memory_budget_ = budget; }

    /**
     * @brief Set memory size per layer (for eviction calculations)
     */
    void setLayerSize(size_t size) { layer_size_ = size; }
    size_t layerSize() const { return layer_size_; }

    /**
     * @brief Set eviction policy
     */
    void setEvictionPolicy(StreamingEvictionPolicy policy) { eviction_policy_ = policy; }
    StreamingEvictionPolicy evictionPolicy() const { return eviction_policy_; }

    /**
     * @brief Set prefetch depth (how many layers to prefetch ahead)
     */
    void setPrefetchDepth(size_t depth) { prefetch_depth_ = depth; }
    size_t prefetchDepth() const { return prefetch_depth_; }

    // =========================================================================
    // Callback Configuration
    // =========================================================================

    /**
     * @brief Set callback invoked when a layer is prefetched
     */
    void set_prefetch_callback(std::function<void(int layer)> cb) {
        prefetch_callback_ = std::move(cb);
    }

    /**
     * @brief Set callback invoked when a layer is evicted
     */
    void set_evict_callback(std::function<void(int layer)> cb) {
        evict_callback_ = std::move(cb);
    }

    /**
     * @brief Set callback invoked when a layer is loaded (ensureLayerOnDevice)
     */
    void set_load_callback(std::function<void(int layer)> cb) {
        load_callback_ = std::move(cb);
    }

    // =========================================================================
    // Test Inspection API
    // =========================================================================

    /**
     * @brief Get list of currently cached layers (ordered by cache order)
     */
    const std::list<int>& cachedLayers() const { return lru_order_; }

    /**
     * @brief Get number of cached layers
     */
    size_t cachedLayerCount() const { return cached_layers_.size(); }

    /**
     * @brief Get layers currently being prefetched
     */
    const std::unordered_set<int>& prefetchingLayers() const { return prefetching_; }

    /**
     * @brief Get ordered list of load operations (for verification)
     */
    const std::vector<int>& loadHistory() const { return load_history_; }

    /**
     * @brief Get ordered list of evict operations (for verification)
     */
    const std::vector<int>& evictHistory() const { return evict_history_; }

    /**
     * @brief Get ordered list of prefetch operations (for verification)
     */
    const std::vector<int>& prefetchHistory() const { return prefetch_history_; }

    /**
     * @brief Clear operation histories (but keep cache state)
     */
    void clearHistory();

    /**
     * @brief Simulate prefetch completion for a layer
     */
    void completePrefetch(int layer_idx);

    /**
     * @brief Force a layer into the cache (for test setup)
     */
    void forceCache(int layer_idx);

private:
    // =========================================================================
    // Internal Methods
    // =========================================================================

    /**
     * @brief Evict layers to make room for new layer
     * @return true if sufficient space was freed
     */
    bool evictToFreeMemory();

    /**
     * @brief Evict least recently used layer
     */
    int evictLRU();

    /**
     * @brief Evict oldest (first-in) layer
     */
    int evictFIFO();

    /**
     * @brief Update LRU tracking for a layer (move to front)
     */
    void touchLayer(int layer_idx);

    // =========================================================================
    // Configuration
    // =========================================================================

    bool streaming_enabled_ = true;
    int num_layers_ = 24;
    size_t memory_budget_ = 0;  // 0 = unlimited
    size_t layer_size_ = 100 * 1024 * 1024;  // 100MB default
    StreamingEvictionPolicy eviction_policy_ = StreamingEvictionPolicy::LRU;
    size_t prefetch_depth_ = 1;

    // =========================================================================
    // Cache State
    // =========================================================================

    std::unordered_set<int> cached_layers_;
    std::list<int> lru_order_;  // Front = MRU, Back = LRU
    std::unordered_map<int, std::list<int>::iterator> lru_map_;
    std::unordered_set<int> prefetching_;

    // =========================================================================
    // Callbacks
    // =========================================================================

    std::function<void(int)> prefetch_callback_;
    std::function<void(int)> evict_callback_;
    std::function<void(int)> load_callback_;

    // =========================================================================
    // Operation History (for verification)
    // =========================================================================

    std::vector<int> load_history_;
    std::vector<int> evict_history_;
    std::vector<int> prefetch_history_;

    // =========================================================================
    // Statistics
    // =========================================================================

    mutable StreamingStats stats_;
    InferencePhase current_phase_ = InferencePhase::PREFILL;
};

/**
 * @brief Fluent builder for MockWeightStreamer
 */
class MockWeightStreamerBuilder {
public:
    MockWeightStreamerBuilder();

    MockWeightStreamerBuilder& setStreamingEnabled(bool enabled);
    MockWeightStreamerBuilder& setNumLayers(int num_layers);
    MockWeightStreamerBuilder& setMemoryBudget(size_t budget);
    MockWeightStreamerBuilder& setLayerSize(size_t size);
    MockWeightStreamerBuilder& setEvictionPolicy(StreamingEvictionPolicy policy);
    MockWeightStreamerBuilder& setPrefetchDepth(size_t depth);

    std::shared_ptr<MockWeightStreamer> build();

private:
    std::shared_ptr<MockWeightStreamer> mock_;
};

// =============================================================================
// IMPLEMENTATION
// =============================================================================

inline MockWeightStreamer::MockWeightStreamer() = default;

inline std::shared_ptr<MockWeightStreamer> MockWeightStreamer::createResident() {
    auto mock = std::make_shared<MockWeightStreamer>();
    mock->setStreamingEnabled(false);
    return mock;
}

inline std::shared_ptr<MockWeightStreamer> MockWeightStreamer::createStreaming(
    int num_layers, int cache_capacity) {
    auto mock = std::make_shared<MockWeightStreamer>();
    mock->setStreamingEnabled(true);
    mock->setNumLayers(num_layers);
    // Set budget to hold exactly cache_capacity layers
    size_t layer_size = 100 * 1024 * 1024;  // 100MB
    mock->setLayerSize(layer_size);
    mock->setMemoryBudget(cache_capacity * layer_size);
    return mock;
}

inline bool MockWeightStreamer::ensureLayerOnDevice(int layer_idx, DeviceId /*device*/) {
    if (!streaming_enabled_) {
        return true;  // All layers resident
    }

    if (layer_idx < 0 || layer_idx >= num_layers_) {
        return false;
    }

    // Check if already cached
    if (cached_layers_.count(layer_idx)) {
        touchLayer(layer_idx);
        stats_.cache_hits++;
        return true;
    }

    // Track as cache miss
    stats_.cache_misses++;
    
    // Check if prefetch was in progress (will count as prefetch hit)
    if (prefetching_.count(layer_idx)) {
        prefetching_.erase(layer_idx);
        stats_.prefetch_hits++;
    }

    // Evict if needed
    if (memory_budget_ > 0) {
        while (currentDeviceMemoryUsage() + layer_size_ > memory_budget_) {
            if (!evictToFreeMemory()) {
                return false;  // Can't evict enough
            }
        }
    }

    // Load the layer
    cached_layers_.insert(layer_idx);
    lru_order_.push_front(layer_idx);
    lru_map_[layer_idx] = lru_order_.begin();

    load_history_.push_back(layer_idx);
    stats_.layers_transferred++;
    stats_.bytes_transferred += layer_size_;
    stats_.current_layers_cached = cached_layers_.size();
    stats_.current_cache_size = currentDeviceMemoryUsage();
    stats_.peak_cache_size = std::max(stats_.peak_cache_size, stats_.current_cache_size);

    if (load_callback_) {
        load_callback_(layer_idx);
    }

    return true;
}

inline void MockWeightStreamer::prefetchLayer(int layer_idx, DeviceId /*device*/) {
    if (!streaming_enabled_) {
        return;  // No-op when resident
    }

    if (layer_idx < 0 || layer_idx >= num_layers_) {
        return;
    }

    // Already cached or prefetching
    if (cached_layers_.count(layer_idx) || prefetching_.count(layer_idx)) {
        return;
    }

    prefetching_.insert(layer_idx);
    prefetch_history_.push_back(layer_idx);

    if (prefetch_callback_) {
        prefetch_callback_(layer_idx);
    }
}

inline void MockWeightStreamer::releaseLayer(int layer_idx) {
    // Hint that layer is no longer needed - moves to end of LRU
    if (cached_layers_.count(layer_idx)) {
        // Move to back of LRU (make it eviction candidate)
        auto it = lru_map_.find(layer_idx);
        if (it != lru_map_.end()) {
            lru_order_.erase(it->second);
            lru_order_.push_back(layer_idx);
            lru_map_[layer_idx] = std::prev(lru_order_.end());
        }
    }
}

inline void MockWeightStreamer::onPhaseTransition(InferencePhase /*old_phase*/, InferencePhase new_phase) {
    current_phase_ = new_phase;
}

inline size_t MockWeightStreamer::currentDeviceMemoryUsage() const {
    return cached_layers_.size() * layer_size_;
}

inline bool MockWeightStreamer::evictLayer(int layer_idx) {
    if (!cached_layers_.count(layer_idx)) {
        return false;
    }

    cached_layers_.erase(layer_idx);
    auto it = lru_map_.find(layer_idx);
    if (it != lru_map_.end()) {
        lru_order_.erase(it->second);
        lru_map_.erase(it);
    }

    evict_history_.push_back(layer_idx);
    stats_.layers_evicted++;
    stats_.bytes_evicted += layer_size_;
    stats_.current_layers_cached = cached_layers_.size();
    stats_.current_cache_size = currentDeviceMemoryUsage();

    if (evict_callback_) {
        evict_callback_(layer_idx);
    }

    return true;
}

inline void MockWeightStreamer::clearCache() {
    while (!lru_order_.empty()) {
        evictLayer(lru_order_.back());
    }
    prefetching_.clear();
}

inline void MockWeightStreamer::synchronize() {
    // Complete all pending prefetches
    // Make a copy since completePrefetch modifies prefetching_
    std::vector<int> pending(prefetching_.begin(), prefetching_.end());
    for (int layer : pending) {
        completePrefetch(layer);
    }
}

inline bool MockWeightStreamer::isLayerCached(int layer_idx, DeviceId /*device*/) const {
    if (!streaming_enabled_) {
        return true;  // All resident
    }
    return cached_layers_.count(layer_idx) > 0;
}

inline bool MockWeightStreamer::isPrefetchInProgress(int layer_idx) const {
    return prefetching_.count(layer_idx) > 0;
}

inline void MockWeightStreamer::resetStats() {
    stats_ = StreamingStats{};
    stats_.current_layers_cached = cached_layers_.size();
    stats_.current_cache_size = currentDeviceMemoryUsage();
}

inline void MockWeightStreamer::clearHistory() {
    load_history_.clear();
    evict_history_.clear();
    prefetch_history_.clear();
}

inline void MockWeightStreamer::completePrefetch(int layer_idx) {
    if (!prefetching_.count(layer_idx)) {
        return;
    }

    prefetching_.erase(layer_idx);

    // Evict if needed
    if (memory_budget_ > 0) {
        while (currentDeviceMemoryUsage() + layer_size_ > memory_budget_) {
            if (!evictToFreeMemory()) {
                // Can't fit, discard prefetch
                stats_.prefetch_wasted++;
                return;
            }
        }
    }

    // Add to cache
    cached_layers_.insert(layer_idx);
    lru_order_.push_front(layer_idx);
    lru_map_[layer_idx] = lru_order_.begin();

    stats_.layers_transferred++;
    stats_.bytes_transferred += layer_size_;
    stats_.current_layers_cached = cached_layers_.size();
    stats_.current_cache_size = currentDeviceMemoryUsage();
    stats_.peak_cache_size = std::max(stats_.peak_cache_size, stats_.current_cache_size);
}

inline void MockWeightStreamer::forceCache(int layer_idx) {
    if (cached_layers_.count(layer_idx)) {
        return;
    }

    cached_layers_.insert(layer_idx);
    lru_order_.push_front(layer_idx);
    lru_map_[layer_idx] = lru_order_.begin();
    stats_.current_layers_cached = cached_layers_.size();
    stats_.current_cache_size = currentDeviceMemoryUsage();
}

inline bool MockWeightStreamer::evictToFreeMemory() {
    if (cached_layers_.empty()) {
        return false;
    }

    int evicted = -1;
    switch (eviction_policy_) {
        case StreamingEvictionPolicy::LRU:
            evicted = evictLRU();
            break;
        case StreamingEvictionPolicy::FIFO:
            evicted = evictFIFO();
            break;
        case StreamingEvictionPolicy::NONE:
            return false;  // No eviction allowed
    }

    return evicted >= 0;
}

inline int MockWeightStreamer::evictLRU() {
    if (lru_order_.empty()) {
        return -1;
    }
    int layer = lru_order_.back();
    evictLayer(layer);
    return layer;
}

inline int MockWeightStreamer::evictFIFO() {
    // FIFO: evict oldest loaded (back of FIFO order list)
    // Unlike LRU, FIFO doesn't reorder on access, so we track insertion order separately
    if (lru_order_.empty()) {
        return -1;
    }
    // For FIFO, we don't touch layers on access, so back is truly oldest
    int layer = lru_order_.back();
    evictLayer(layer);
    return layer;
}

inline void MockWeightStreamer::touchLayer(int layer_idx) {
    // Only update order for LRU policy (FIFO doesn't reorder on access)
    if (eviction_policy_ != StreamingEvictionPolicy::LRU) {
        return;
    }
    auto it = lru_map_.find(layer_idx);
    if (it != lru_map_.end()) {
        lru_order_.erase(it->second);
        lru_order_.push_front(layer_idx);
        lru_map_[layer_idx] = lru_order_.begin();
    }
}

// =============================================================================
// Builder Implementation
// =============================================================================

inline MockWeightStreamerBuilder::MockWeightStreamerBuilder()
    : mock_(std::make_shared<MockWeightStreamer>()) {}

inline MockWeightStreamerBuilder& MockWeightStreamerBuilder::setStreamingEnabled(bool enabled) {
    mock_->setStreamingEnabled(enabled);
    return *this;
}

inline MockWeightStreamerBuilder& MockWeightStreamerBuilder::setNumLayers(int num_layers) {
    mock_->setNumLayers(num_layers);
    return *this;
}

inline MockWeightStreamerBuilder& MockWeightStreamerBuilder::setMemoryBudget(size_t budget) {
    mock_->setMemoryBudget(budget);
    return *this;
}

inline MockWeightStreamerBuilder& MockWeightStreamerBuilder::setLayerSize(size_t size) {
    mock_->setLayerSize(size);
    return *this;
}

inline MockWeightStreamerBuilder& MockWeightStreamerBuilder::setEvictionPolicy(StreamingEvictionPolicy policy) {
    mock_->setEvictionPolicy(policy);
    return *this;
}

inline MockWeightStreamerBuilder& MockWeightStreamerBuilder::setPrefetchDepth(size_t depth) {
    mock_->setPrefetchDepth(depth);
    return *this;
}

inline std::shared_ptr<MockWeightStreamer> MockWeightStreamerBuilder::build() {
    return mock_;
}

} // namespace llaminar2::test
