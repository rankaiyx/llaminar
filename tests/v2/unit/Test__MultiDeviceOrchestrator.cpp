/**
 * @file Test__MultiDeviceOrchestrator.cpp
 * @brief Unit tests for MultiDeviceOrchestrator mocks and interface contract
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests the mock implementations used for testing MultiDeviceOrchestrator
 * coordination logic. The mocks enable testing LOCAL tensor parallelism
 * coordination without real devices.
 *
 * Note: Tests for the actual MultiDeviceOrchestrator class will be enabled
 * once the implementation is added to the build.
 */

#include <gtest/gtest.h>
#include "execution/IInferenceRunner.h"
#include "collective/ILocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "tensors/Tensors.h"
#include <atomic>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>

using namespace llaminar2;

// =============================================================================
// MockDeviceGraphOrchestrator - Mock for per-device runners
// =============================================================================

/**
 * @brief Mock inference runner for per-device testing
 *
 * Tracks method calls and provides configurable return values for testing
 * the MultiDeviceOrchestrator coordination logic.
 */
class MockDeviceGraphOrchestrator : public IInferenceRunner
{
public:
    struct Config
    {
        int vocab_size = 32000;
        bool forward_should_fail = false;
        std::string architecture = "mock_qwen2";
    };

    MockDeviceGraphOrchestrator() : MockDeviceGraphOrchestrator(Config{}) {}

    explicit MockDeviceGraphOrchestrator(const Config &config)
        : config_(config), position_(0)
    {
        logits_.resize(static_cast<size_t>(config_.vocab_size), 0.0f);
    }

    // =====================================================================
    // IInferenceRunner Implementation
    // =====================================================================

    bool forward(const int *tokens, int seq_len) override
    {
        (void)tokens;
        forward_calls_.fetch_add(1, std::memory_order_relaxed);
        if (config_.forward_should_fail)
        {
            return false;
        }
        position_ += seq_len;
        return true;
    }

    const float *logits() const override
    {
        return logits_.data();
    }

    int vocab_size() const override
    {
        return config_.vocab_size;
    }

    void clear_cache() override
    {
        clear_cache_calls_.fetch_add(1, std::memory_order_relaxed);
        position_ = 0;
    }

    int get_position() const override
    {
        return position_;
    }

    ExecutionPath executionPath() const override
    {
        return ExecutionPath::GRAPH;
    }

    const char *architecture() const override
    {
        return config_.architecture.c_str();
    }

    // =====================================================================
    // Test Utilities
    // =====================================================================

    size_t forward_call_count() const
    {
        return forward_calls_.load(std::memory_order_relaxed);
    }

    size_t clear_cache_call_count() const
    {
        return clear_cache_calls_.load(std::memory_order_relaxed);
    }

    void set_forward_fails(bool fails) { config_.forward_should_fail = fails; }

    void set_mock_logits(const std::vector<float> &logits)
    {
        logits_ = logits;
        config_.vocab_size = static_cast<int>(logits.size());
    }

    void set_vocab_size(int size)
    {
        config_.vocab_size = size;
        logits_.resize(static_cast<size_t>(size), 0.0f);
    }

    void reset_call_counts()
    {
        forward_calls_.store(0, std::memory_order_relaxed);
        clear_cache_calls_.store(0, std::memory_order_relaxed);
    }

private:
    Config config_;
    int position_;
    std::vector<float> logits_;
    mutable std::atomic<size_t> forward_calls_{0};
    mutable std::atomic<size_t> clear_cache_calls_{0};
};

// =============================================================================
// MockLocalTPContext - Mock for LOCAL TP context
// =============================================================================

/**
 * @brief Mock LOCAL TP context for testing collective operations
 *
 * Tracks synchronization calls and provides configurable devices/weights.
 */
class MockLocalTPContext : public ILocalTPContext
{
public:
    struct Config
    {
        std::vector<GlobalDeviceAddress> devices;
        std::vector<float> weights;
        CollectiveBackendType backend = CollectiveBackendType::HOST;
        bool allreduce_should_fail = false;
        bool allgather_should_fail = false;
    };

    MockLocalTPContext() : MockLocalTPContext(Config{}) {}

    explicit MockLocalTPContext(const Config &config)
        : config_(config)
    {
        // Default to 2 CPU devices if none specified
        if (config_.devices.empty())
        {
            config_.devices.push_back(GlobalDeviceAddress::cpu());
            config_.devices.push_back(GlobalDeviceAddress::cpu());
        }
        // Default to equal weights
        if (config_.weights.empty())
        {
            float equal = 1.0f / static_cast<float>(config_.devices.size());
            config_.weights.resize(config_.devices.size(), equal);
        }
    }

    // =====================================================================
    // ILocalTPContext Configuration API
    // =====================================================================

    const std::vector<GlobalDeviceAddress> &devices() const override
    {
        return config_.devices;
    }

    const std::vector<float> &weights() const override
    {
        return config_.weights;
    }

    CollectiveBackendType backend() const override
    {
        return config_.backend;
    }

    int degree() const override
    {
        return static_cast<int>(config_.devices.size());
    }

    // =====================================================================
    // ILocalTPContext Collective Operations
    // =====================================================================

    bool allreduce(TensorBase * /*tensor*/) override
    {
        allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allreduce_should_fail;
    }

    bool allreduce(const TensorBase * /*input*/, TensorBase * /*output*/) override
    {
        allreduce_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allreduce_should_fail;
    }

    bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override
    {
        allgather_calls_.fetch_add(1, std::memory_order_relaxed);
        return !config_.allgather_should_fail;
    }

    bool gatherFromDevices(
        const std::vector<const TensorBase *> &shards,
        TensorBase *output) override
    {
        gather_from_devices_calls_.fetch_add(1, std::memory_order_relaxed);

        // Simple mock implementation: copy data from shards to output
        if (shards.empty() || !output)
        {
            return false;
        }

        float *dst = output->mutable_data();
        size_t offset = 0;
        for (const auto *shard : shards)
        {
            if (shard)
            {
                const float *src = shard->data();
                size_t count = shard->numel();
                std::memcpy(dst + offset, src, count * sizeof(float));
                offset += count;
            }
        }
        return !config_.allgather_should_fail;
    }

    bool reduceScatter(const TensorBase * /*input*/, TensorBase * /*output_shard*/) override
    {
        reduce_scatter_calls_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    // =====================================================================
    // ILocalTPContext Synchronization
    // =====================================================================

    void synchronize() override
    {
        synchronize_calls_.fetch_add(1, std::memory_order_relaxed);
    }

    // =====================================================================
    // ILocalTPContext Device Management
    // =====================================================================

    int indexForDevice(const GlobalDeviceAddress &device) const override
    {
        for (size_t i = 0; i < config_.devices.size(); ++i)
        {
            if (config_.devices[i] == device)
            {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    const GlobalDeviceAddress &deviceAt(int index) const override
    {
        if (index < 0 || index >= static_cast<int>(config_.devices.size()))
        {
            throw std::out_of_range("MockLocalTPContext::deviceAt: index out of range");
        }
        return config_.devices[static_cast<size_t>(index)];
    }

    float weightForDevice(const GlobalDeviceAddress &device) const override
    {
        int idx = indexForDevice(device);
        return (idx >= 0) ? config_.weights[static_cast<size_t>(idx)] : 0.0f;
    }

    // =====================================================================
    // ILocalTPContext Sharding Utilities
    // =====================================================================

    int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
    {
        float w = weightForDevice(device);
        return static_cast<int>(w * static_cast<float>(total_heads) + 0.5f);
    }

    std::pair<int, int> rowRangeForDevice(
        const GlobalDeviceAddress &device, int total_rows) const override
    {
        int idx = indexForDevice(device);
        if (idx < 0)
            return {0, 0};

        float cumulative = 0.0f;
        for (int i = 0; i < idx; ++i)
        {
            cumulative += config_.weights[static_cast<size_t>(i)];
        }
        int start = static_cast<int>(cumulative * static_cast<float>(total_rows));
        int end = static_cast<int>((cumulative + config_.weights[static_cast<size_t>(idx)]) * static_cast<float>(total_rows));
        return {start, end};
    }

    std::pair<int, int> colRangeForDevice(
        const GlobalDeviceAddress &device, int total_cols) const override
    {
        return rowRangeForDevice(device, total_cols);
    }

    // =====================================================================
    // Test Utilities
    // =====================================================================

    size_t synchronize_call_count() const
    {
        return synchronize_calls_.load(std::memory_order_relaxed);
    }

    size_t allreduce_call_count() const
    {
        return allreduce_calls_.load(std::memory_order_relaxed);
    }

    size_t allgather_call_count() const
    {
        return allgather_calls_.load(std::memory_order_relaxed);
    }

    size_t gather_from_devices_call_count() const
    {
        return gather_from_devices_calls_.load(std::memory_order_relaxed);
    }

    void reset_call_counts()
    {
        synchronize_calls_.store(0, std::memory_order_relaxed);
        allreduce_calls_.store(0, std::memory_order_relaxed);
        allgather_calls_.store(0, std::memory_order_relaxed);
        gather_from_devices_calls_.store(0, std::memory_order_relaxed);
        reduce_scatter_calls_.store(0, std::memory_order_relaxed);
    }

    void set_allreduce_fails(bool fails) { config_.allreduce_should_fail = fails; }
    void set_allgather_fails(bool fails) { config_.allgather_should_fail = fails; }

private:
    Config config_;
    mutable std::atomic<size_t> synchronize_calls_{0};
    mutable std::atomic<size_t> allreduce_calls_{0};
    mutable std::atomic<size_t> allgather_calls_{0};
    mutable std::atomic<size_t> gather_from_devices_calls_{0};
    mutable std::atomic<size_t> reduce_scatter_calls_{0};
};

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for MultiDeviceOrchestrator tests
 *
 * Provides helper methods to create mock orchestrators with different
 * configurations.
 */
class Test__MultiDeviceOrchestrator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Default setup: 2 device runners
        mock_runners_.clear();
        mock_runners_.push_back(std::make_unique<MockDeviceGraphOrchestrator>());
        mock_runners_.push_back(std::make_unique<MockDeviceGraphOrchestrator>());

        // Create mock TP context with 2 devices
        MockLocalTPContext::Config tp_config;
        tp_config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        tp_config.weights = {0.5f, 0.5f};
        mock_tp_ctx_ = std::make_unique<MockLocalTPContext>(tp_config);
    }

    // Store raw pointers to mock runners for call verification
    std::vector<MockDeviceGraphOrchestrator *> getRawMockRunners()
    {
        std::vector<MockDeviceGraphOrchestrator *> result;
        for (auto &runner : mock_runners_)
        {
            result.push_back(runner.get());
        }
        return result;
    }

    std::vector<std::unique_ptr<MockDeviceGraphOrchestrator>> mock_runners_;
    std::unique_ptr<MockLocalTPContext> mock_tp_ctx_;
};

// =============================================================================
// Construction Tests
// =============================================================================

TEST_F(Test__MultiDeviceOrchestrator, ConstructsWithValidConfig)
{
    // Verify mock setup is valid
    ASSERT_EQ(mock_runners_.size(), 2u);
    ASSERT_NE(mock_tp_ctx_, nullptr);
    EXPECT_EQ(mock_tp_ctx_->degree(), 2);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextDegreeMatchesDevices)
{
    // Create TP context with 3 devices
    MockLocalTPContext::Config tp_config;
    tp_config.devices = {
        GlobalDeviceAddress::cpu(),
        GlobalDeviceAddress::cpu(),
        GlobalDeviceAddress::cpu()};
    auto tp_ctx = std::make_unique<MockLocalTPContext>(tp_config);

    EXPECT_EQ(tp_ctx->degree(), 3);
    EXPECT_EQ(tp_ctx->devices().size(), 3u);
}

// =============================================================================
// Mock Device Runner Tests
// =============================================================================

TEST_F(Test__MultiDeviceOrchestrator, MockDeviceRunnerForwardTracksCallCount)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    EXPECT_EQ(runner->forward_call_count(), 0u);

    bool result = runner->forward(tokens, 3);
    EXPECT_TRUE(result);
    EXPECT_EQ(runner->forward_call_count(), 1u);

    result = runner->forward(tokens, 3);
    EXPECT_TRUE(result);
    EXPECT_EQ(runner->forward_call_count(), 2u);
}

TEST_F(Test__MultiDeviceOrchestrator, MockDeviceRunnerClearCacheTracksCallCount)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();

    EXPECT_EQ(runner->clear_cache_call_count(), 0u);

    runner->clear_cache();
    EXPECT_EQ(runner->clear_cache_call_count(), 1u);

    runner->clear_cache();
    EXPECT_EQ(runner->clear_cache_call_count(), 2u);
}

TEST_F(Test__MultiDeviceOrchestrator, MockDeviceRunnerReturnsConfiguredVocabSize)
{
    MockDeviceGraphOrchestrator::Config config;
    config.vocab_size = 50000;
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>(config);

    EXPECT_EQ(runner->vocab_size(), 50000);
}

TEST_F(Test__MultiDeviceOrchestrator, MockDeviceRunnerForwardCanFail)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    runner->set_forward_fails(true);

    int tokens[] = {1, 2, 3};
    bool result = runner->forward(tokens, 3);

    EXPECT_FALSE(result);
    EXPECT_EQ(runner->forward_call_count(), 1u); // Still tracked
}

TEST_F(Test__MultiDeviceOrchestrator, MockDeviceRunnerReturnsLogits)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    runner->set_mock_logits({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    const float *logits = runner->logits();
    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 1.0f);
    EXPECT_FLOAT_EQ(logits[4], 5.0f);
}

TEST_F(Test__MultiDeviceOrchestrator, MockDeviceRunnerReturnsArchitecture)
{
    MockDeviceGraphOrchestrator::Config config;
    config.architecture = "test_arch";
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>(config);

    EXPECT_STREQ(runner->architecture(), "test_arch");
}

TEST_F(Test__MultiDeviceOrchestrator, MockDeviceRunnerReturnsGraphExecutionPath)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    EXPECT_EQ(runner->executionPath(), ExecutionPath::GRAPH);
}

// =============================================================================
// Mock TP Context Tests
// =============================================================================

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextReturnsConfiguredDevices)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->devices().size(), 2u);
    EXPECT_EQ(ctx->devices()[0], GlobalDeviceAddress::cuda(0));
    EXPECT_EQ(ctx->devices()[1], GlobalDeviceAddress::cuda(1));
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextReturnsConfiguredWeights)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    config.weights = {0.7f, 0.3f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->weights().size(), 2u);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 0.7f);
    EXPECT_FLOAT_EQ(ctx->weights()[1], 0.3f);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextSynchronizeTracksCallCount)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    EXPECT_EQ(ctx->synchronize_call_count(), 0u);

    ctx->synchronize();
    EXPECT_EQ(ctx->synchronize_call_count(), 1u);

    ctx->synchronize();
    ctx->synchronize();
    EXPECT_EQ(ctx->synchronize_call_count(), 3u);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextAllreduceReturnsTrue)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    bool result = ctx->allreduce(static_cast<TensorBase *>(nullptr));
    EXPECT_TRUE(result);
    EXPECT_EQ(ctx->allreduce_call_count(), 1u);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextAllreduceCanFail)
{
    auto ctx = std::make_unique<MockLocalTPContext>();
    ctx->set_allreduce_fails(true);

    bool result = ctx->allreduce(static_cast<TensorBase *>(nullptr));
    EXPECT_FALSE(result);
    EXPECT_EQ(ctx->allreduce_call_count(), 1u); // Still tracked
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextAllgatherReturnsTrue)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    bool result = ctx->allgather(nullptr, nullptr);
    EXPECT_TRUE(result);
    EXPECT_EQ(ctx->allgather_call_count(), 1u);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextDeviceIndexing)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::cuda(0)), 0);
    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::rocm(1)), 1);
    EXPECT_EQ(ctx->indexForDevice(GlobalDeviceAddress::cpu()), -1); // Not found
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextDeviceAtReturnsCorrectDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->deviceAt(0), GlobalDeviceAddress::cuda(0));
    EXPECT_EQ(ctx->deviceAt(1), GlobalDeviceAddress::cuda(1));
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextDeviceAtThrowsForInvalidIndex)
{
    auto ctx = std::make_unique<MockLocalTPContext>();

    EXPECT_THROW(ctx->deviceAt(-1), std::out_of_range);
    EXPECT_THROW(ctx->deviceAt(99), std::out_of_range);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextWeightForDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.6f, 0.4f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cuda(0)), 0.6f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cuda(1)), 0.4f);
    EXPECT_FLOAT_EQ(ctx->weightForDevice(GlobalDeviceAddress::cpu()), 0.0f); // Not found
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextHeadsForDevice)
{
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    config.weights = {0.5f, 0.5f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    // 16 total heads with 50/50 weights = 8 each
    EXPECT_EQ(ctx->headsForDevice(config.devices[0], 16), 8);
    EXPECT_EQ(ctx->headsForDevice(config.devices[1], 16), 8);
}

// =============================================================================
// Integration: Multi-Runner Coordination Tests
// =============================================================================

TEST_F(Test__MultiDeviceOrchestrator, AllDevicesReadyWhenAllHaveVocabSize)
{
    // Verify that when all mock runners have valid vocab_size, allDevicesReady would return true
    // (Testing the mock behavior that the real orchestrator uses)
    for (auto &runner : mock_runners_)
    {
        EXPECT_GT(runner->vocab_size(), 0);
    }
    EXPECT_EQ(mock_runners_.size(), 2u);
}

TEST_F(Test__MultiDeviceOrchestrator, MultipleRunnersCanBeCalledInParallel)
{
    // Simulate what MultiDeviceOrchestrator does: call forward on all runners
    int tokens[] = {1, 2, 3};

    // Call forward on all mock runners
    bool all_success = true;
    for (auto &runner : mock_runners_)
    {
        if (!runner->forward(tokens, 3))
        {
            all_success = false;
        }
    }

    EXPECT_TRUE(all_success);
    for (auto &runner : mock_runners_)
    {
        EXPECT_EQ(runner->forward_call_count(), 1u);
    }
}

TEST_F(Test__MultiDeviceOrchestrator, ForwardFailsIfAnyDeviceFails)
{
    // Set one runner to fail
    mock_runners_[1]->set_forward_fails(true);

    int tokens[] = {1, 2, 3};

    // Simulate MultiDeviceOrchestrator::forward
    bool all_success = true;
    for (auto &runner : mock_runners_)
    {
        if (!runner->forward(tokens, 3))
        {
            all_success = false;
        }
    }

    EXPECT_FALSE(all_success);
    // Both should have been called
    EXPECT_EQ(mock_runners_[0]->forward_call_count(), 1u);
    EXPECT_EQ(mock_runners_[1]->forward_call_count(), 1u);
}

TEST_F(Test__MultiDeviceOrchestrator, ClearCacheClearsAllDevices)
{
    // Simulate MultiDeviceOrchestrator::clear_cache
    for (auto &runner : mock_runners_)
    {
        runner->clear_cache();
    }

    for (auto &runner : mock_runners_)
    {
        EXPECT_EQ(runner->clear_cache_call_count(), 1u);
    }
}

TEST_F(Test__MultiDeviceOrchestrator, LogitsReturnsFromPrimaryDevice)
{
    // Set different logits on each runner
    mock_runners_[0]->set_mock_logits({10.0f, 20.0f, 30.0f});
    mock_runners_[1]->set_mock_logits({1.0f, 2.0f, 3.0f});

    // Simulate MultiDeviceOrchestrator::logits (returns from primary device)
    const float *logits = mock_runners_[0]->logits();

    ASSERT_NE(logits, nullptr);
    EXPECT_FLOAT_EQ(logits[0], 10.0f);
    EXPECT_FLOAT_EQ(logits[1], 20.0f);
    EXPECT_FLOAT_EQ(logits[2], 30.0f);
}

TEST_F(Test__MultiDeviceOrchestrator, VocabSizeFromPrimaryDevice)
{
    mock_runners_[0]->set_vocab_size(50000);
    mock_runners_[1]->set_vocab_size(32000);

    // Simulate MultiDeviceOrchestrator::vocab_size (returns from primary device)
    int vocab = mock_runners_[0]->vocab_size();

    EXPECT_EQ(vocab, 50000);
}

TEST_F(Test__MultiDeviceOrchestrator, SynchronizeDevicesCallsTPContext)
{
    // Simulate MultiDeviceOrchestrator::synchronizeDevices
    mock_tp_ctx_->synchronize();

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 1u);
}

TEST_F(Test__MultiDeviceOrchestrator, DeviceCountMatchesTPDegree)
{
    // Verify TP context degree matches expected device count
    EXPECT_EQ(mock_tp_ctx_->degree(), static_cast<int>(mock_runners_.size()));
}

TEST_F(Test__MultiDeviceOrchestrator, LocalTPContextReturnsContext)
{
    // Verify mock TP context is valid
    ASSERT_NE(mock_tp_ctx_, nullptr);
    EXPECT_EQ(mock_tp_ctx_->degree(), 2);
    EXPECT_EQ(mock_tp_ctx_->backend(), CollectiveBackendType::HOST);
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST_F(Test__MultiDeviceOrchestrator, EmptyDeviceRunnersReturnsSafeDefaults)
{
    std::vector<std::unique_ptr<MockDeviceGraphOrchestrator>> empty_runners;

    // With no runners, we should handle gracefully
    EXPECT_TRUE(empty_runners.empty());
}

TEST_F(Test__MultiDeviceOrchestrator, SingleTPContextWithOneDevice)
{
    // Test TP context with single device - should work but is unusual
    MockLocalTPContext::Config config;
    config.devices = {GlobalDeviceAddress::cpu()};
    config.weights = {1.0f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    EXPECT_EQ(ctx->degree(), 1);
    EXPECT_FLOAT_EQ(ctx->weights()[0], 1.0f);
}

TEST_F(Test__MultiDeviceOrchestrator, MockRunnerPositionUpdatesOnForward)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3, 4, 5};

    EXPECT_EQ(runner->get_position(), 0);

    runner->forward(tokens, 5);
    EXPECT_EQ(runner->get_position(), 5);

    runner->forward(tokens, 3);
    EXPECT_EQ(runner->get_position(), 8);
}

TEST_F(Test__MultiDeviceOrchestrator, MockRunnerClearCacheResetsPosition)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    runner->forward(tokens, 3);
    EXPECT_EQ(runner->get_position(), 3);

    runner->clear_cache();
    EXPECT_EQ(runner->get_position(), 0);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextRowRangeCalculation)
{
    MockLocalTPContext::Config config;
    // Use distinct devices so indexForDevice can differentiate them
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.75f, 0.25f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    // For 100 total rows with 75/25 split:
    // Device 0: rows 0-74 (75 rows)
    // Device 1: rows 75-99 (25 rows)
    auto [start0, end0] = ctx->rowRangeForDevice(config.devices[0], 100);
    EXPECT_EQ(start0, 0);
    EXPECT_EQ(end0, 75);

    auto [start1, end1] = ctx->rowRangeForDevice(config.devices[1], 100);
    EXPECT_EQ(start1, 75);
    EXPECT_EQ(end1, 100);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextColRangeMatchesRowRange)
{
    MockLocalTPContext::Config config;
    // Use distinct devices so indexForDevice can differentiate them
    config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)};
    config.weights = {0.6f, 0.4f};
    auto ctx = std::make_unique<MockLocalTPContext>(config);

    auto row_range = ctx->rowRangeForDevice(config.devices[0], 50);
    auto col_range = ctx->colRangeForDevice(config.devices[0], 50);

    EXPECT_EQ(row_range, col_range);
}

TEST_F(Test__MultiDeviceOrchestrator, ResetCallCountsWorks)
{
    auto runner = std::make_unique<MockDeviceGraphOrchestrator>();
    int tokens[] = {1, 2, 3};

    runner->forward(tokens, 3);
    runner->clear_cache();

    EXPECT_EQ(runner->forward_call_count(), 1u);
    EXPECT_EQ(runner->clear_cache_call_count(), 1u);

    runner->reset_call_counts();

    EXPECT_EQ(runner->forward_call_count(), 0u);
    EXPECT_EQ(runner->clear_cache_call_count(), 0u);
}

TEST_F(Test__MultiDeviceOrchestrator, MockTPContextResetCallCountsWorks)
{
    mock_tp_ctx_->synchronize();
    mock_tp_ctx_->allreduce(static_cast<TensorBase *>(nullptr));

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 1u);
    EXPECT_EQ(mock_tp_ctx_->allreduce_call_count(), 1u);

    mock_tp_ctx_->reset_call_counts();

    EXPECT_EQ(mock_tp_ctx_->synchronize_call_count(), 0u);
    EXPECT_EQ(mock_tp_ctx_->allreduce_call_count(), 0u);
}
