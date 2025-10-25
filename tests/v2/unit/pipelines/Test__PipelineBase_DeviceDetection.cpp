/**
 * @file Test__PipelineBase_DeviceDetection.cpp
 * @brief Unit tests for PipelineBase device detection helpers (Phase 3)
 *
 * Tests the Phase 3 device detection infrastructure that enables MoE-ready
 * heterogeneous execution by detecting attention and FFN devices from WeightPlacementMap.
 */

#include "../../../../src/v2/pipelines/PipelineBase.h"
#include "../../../../src/v2/loaders/WeightPlacementMap.h"
#include "../../../../src/v2/loaders/ModelContext.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace llaminar2;

/**
 * @brief Mock pipeline for testing Phase 3 device detection helpers
 *
 * Minimal implementation that exposes base class helpers for testing.
 * No actual computation needed - just tests device detection logic.
 */
class MockPipelineForDeviceDetection : public PipelineBase
{
public:
    // Expose Phase 3 helpers for testing
    using PipelineBase::detectAttentionDevices;
    using PipelineBase::detectFFNDevices;

    // Minimal constructor
    MockPipelineForDeviceDetection(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        std::shared_ptr<WeightPlacementMap> placement_map)
        : PipelineBase(model_ctx, mpi_ctx, device_idx, placement_map) {}

    // Implement pure virtual methods with minimal stubs
    const char *architecture() const override { return "MockPipeline"; }
    bool forward(const int *tokens, int seq_len) override { return true; }
    bool transformer_layer(int layer_idx, int seq_len) override { return true; }
    std::vector<std::string> getAllWeightNames() const override { return {}; }
    ActivationBuffers createBuffersForDevice(int device_idx, int max_seq_len) override
    {
        return ActivationBuffers();
    }
};

class PipelineBaseDeviceDetectionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create minimal model context (dummy path - no actual file needed for device detection tests)
        model_ctx_ = ModelContext::createForTesting("/tmp/dummy.gguf");
        mpi_ctx_ = nullptr; // Single-node test
        device_idx_ = -1;   // CPU default
    }

    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    int device_idx_;
};

// ========== Phase 3: Attention Device Detection ==========

TEST_F(PipelineBaseDeviceDetectionTest, DetectAttentionDevices_UniformCPU)
{
    // All layers on CPU (default)
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    MockPipelineForDeviceDetection pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map);

    int n_layers = 24;
    auto attention_devices = pipeline.detectAttentionDevices(n_layers);

    // All layers should be on CPU (-1)
    ASSERT_EQ(attention_devices.size(), 24);
    for (int i = 0; i < n_layers; ++i)
    {
        EXPECT_EQ(attention_devices[i], -1) << "Layer " << i << " should be on CPU";
    }
}

TEST_F(PipelineBaseDeviceDetectionTest, DetectAttentionDevices_UniformGPU)
{
    // All layers on GPU 0
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    int n_layers = 24;
    for (int i = 0; i < n_layers; ++i)
    {
        placement_map->setAttentionDevice(i, 0); // GPU 0
    }

    MockPipelineForDeviceDetection pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map);
    auto attention_devices = pipeline.detectAttentionDevices(n_layers);

    // All layers should be on GPU 0
    ASSERT_EQ(attention_devices.size(), 24);
    for (int i = 0; i < n_layers; ++i)
    {
        EXPECT_EQ(attention_devices[i], 0) << "Layer " << i << " should be on GPU 0";
    }
}

TEST_F(PipelineBaseDeviceDetectionTest, DetectAttentionDevices_Split)
{
    // Layers 0-11 on CPU, 12-23 on GPU 0
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    int n_layers = 24;
    for (int i = 0; i < 12; ++i)
    {
        placement_map->setAttentionDevice(i, -1); // CPU
    }
    for (int i = 12; i < n_layers; ++i)
    {
        placement_map->setAttentionDevice(i, 0); // GPU 0
    }

    MockPipelineForDeviceDetection pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map);
    auto attention_devices = pipeline.detectAttentionDevices(n_layers);

    ASSERT_EQ(attention_devices.size(), 24);

    // Verify split
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_EQ(attention_devices[i], -1) << "Layer " << i << " should be on CPU";
    }
    for (int i = 12; i < 24; ++i)
    {
        EXPECT_EQ(attention_devices[i], 0) << "Layer " << i << " should be on GPU 0";
    }
}

TEST_F(PipelineBaseDeviceDetectionTest, DetectAttentionDevices_MultiGPU)
{
    // Layers 0-9 on CPU, 10-19 on GPU 0, 20-29 on GPU 1
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    int n_layers = 30;
    for (int i = 0; i < 10; ++i)
    {
        placement_map->setAttentionDevice(i, -1); // CPU
    }
    for (int i = 10; i < 20; ++i)
    {
        placement_map->setAttentionDevice(i, 0); // GPU 0
    }
    for (int i = 20; i < n_layers; ++i)
    {
        placement_map->setAttentionDevice(i, 1); // GPU 1
    }

    MockPipelineForDeviceDetection pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map);
    auto attention_devices = pipeline.detectAttentionDevices(n_layers);

    ASSERT_EQ(attention_devices.size(), 30);

    // Verify multi-GPU split
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(attention_devices[i], -1) << "Layer " << i << " should be on CPU";
    }
    for (int i = 10; i < 20; ++i)
    {
        EXPECT_EQ(attention_devices[i], 0) << "Layer " << i << " should be on GPU 0";
    }
    for (int i = 20; i < 30; ++i)
    {
        EXPECT_EQ(attention_devices[i], 1) << "Layer " << i << " should be on GPU 1";
    }
}

// ========== Phase 3: FFN Device Detection ==========

TEST_F(PipelineBaseDeviceDetectionTest, DetectFFNDevices_Uniform)
{
    // All FFN on GPU 0
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    int n_layers = 24;
    for (int i = 0; i < n_layers; ++i)
    {
        placement_map->setFFNDevice(i, 0); // GPU 0
    }

    MockPipelineForDeviceDetection pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map);
    auto ffn_devices = pipeline.detectFFNDevices(n_layers);

    ASSERT_EQ(ffn_devices.size(), 24);
    for (int i = 0; i < n_layers; ++i)
    {
        EXPECT_EQ(ffn_devices[i], 0) << "Layer " << i << " FFN should be on GPU 0";
    }
}

TEST_F(PipelineBaseDeviceDetectionTest, DetectFFNDevices_MixedWithAttention)
{
    // Attention on CPU, FFN on GPU (heterogeneous within layer)
    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    int n_layers = 24;
    for (int i = 0; i < n_layers; ++i)
    {
        placement_map->setAttentionDevice(i, -1); // Attention on CPU
        placement_map->setFFNDevice(i, 0);        // FFN on GPU 0
    }

    MockPipelineForDeviceDetection pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map);

    auto attention_devices = pipeline.detectAttentionDevices(n_layers);
    auto ffn_devices = pipeline.detectFFNDevices(n_layers);

    ASSERT_EQ(attention_devices.size(), 24);
    ASSERT_EQ(ffn_devices.size(), 24);

    // Verify heterogeneous placement
    for (int i = 0; i < n_layers; ++i)
    {
        EXPECT_EQ(attention_devices[i], -1) << "Layer " << i << " attention should be on CPU";
        EXPECT_EQ(ffn_devices[i], 0) << "Layer " << i << " FFN should be on GPU 0";
    }
}

// ========== MoE Scenario: Attention/FFN on CPU, Experts on GPU ==========

TEST_F(PipelineBaseDeviceDetectionTest, MoEHeterogeneousPlacement)
{
    // Realistic MoE scenario:
    // - Attention: CPU (moderate size)
    // - Local FFN: CPU (moderate size)
    // - Shared Experts: GPU (large, tested separately in WeightPlacementMap tests)

    auto placement_map = std::make_shared<WeightPlacementMap>(-1);

    int n_layers = 24;
    int n_experts = 8;

    // All attention on CPU
    for (int i = 0; i < n_layers; ++i)
    {
        placement_map->setAttentionDevice(i, -1);
    }

    // All local FFN on CPU
    for (int i = 0; i < n_layers; ++i)
    {
        placement_map->setFFNDevice(i, -1);
    }

    // Shared experts on GPU (tested in WeightPlacementMap, not via detection helpers)
    for (int i = 0; i < n_experts; ++i)
    {
        placement_map->setSharedExpertDevice(i, 0);
    }

    MockPipelineForDeviceDetection pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map);

    auto attention_devices = pipeline.detectAttentionDevices(n_layers);
    auto ffn_devices = pipeline.detectFFNDevices(n_layers);

    // Verify attention and FFN on CPU (experts tested elsewhere)
    ASSERT_EQ(attention_devices.size(), 24);
    ASSERT_EQ(ffn_devices.size(), 24);

    for (int i = 0; i < n_layers; ++i)
    {
        EXPECT_EQ(attention_devices[i], -1) << "Layer " << i << " attention should be on CPU";
        EXPECT_EQ(ffn_devices[i], -1) << "Layer " << i << " FFN should be on CPU";
    }

    // Expert placement verified via WeightPlacementMap API
    for (int i = 0; i < n_experts; ++i)
    {
        EXPECT_EQ(placement_map->getSharedExpertDevice(i), 0)
            << "Shared expert " << i << " should be on GPU 0";
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
