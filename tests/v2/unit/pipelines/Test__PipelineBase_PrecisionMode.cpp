/**
 * @file Test__PipelineBase_PrecisionMode.cpp
 * @brief Unit tests for PipelineBase precision mode handling
 *
 * Tests that WeightPrecision and ActivationPrecision flow correctly from
 * PipelineConfig through PipelineBase to attention kernels and other operations.
 *
 * Test file naming convention:
 *   File: Test__PipelineBase_PrecisionMode.cpp → Testing: PipelineBase precision handling
 *   Suite: TEST(Test__PipelineBase_PrecisionMode, ...) → Matches filename
 *
 * @author David Sanftenberg
 * @date 2025-11-05
 */

#include <gtest/gtest.h>
#include "pipelines/PipelineBase.h"
#include "pipelines/PipelineConfig.h"
#include "loaders/ModelContext.h"
#include "loaders/WeightPlacementMap.h"
#include "utils/MPIContext.h"
#include <memory>

using namespace llaminar2;

// =============================================================================
// TEST FIXTURE
// =============================================================================

/**
 * @brief Mock pipeline for testing precision mode
 *
 * Minimal implementation that allows testing precision configuration.
 */
class MockPipelineForPrecision : public PipelineBase
{
public:
    // Expose protected config for testing
    WeightPrecision getWeightPrecision() const { return config_.weight_precision; }
    ActivationPrecision getActivationPrecision() const { return config_.activation_precision; }

    // Constructor
    MockPipelineForPrecision(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        int device_idx,
        std::shared_ptr<WeightPlacementMap> placement_map,
        const PipelineConfig &config)
        : PipelineBase(model_ctx, mpi_ctx, device_idx, placement_map, config) {}

    // Implement all pure virtual methods with minimal stubs
    const char *architecture() const override { return "MockPrecisionPipeline"; }
    bool forward(const int * /*tokens*/, int /*seq_len*/) override { return true; }

    std::vector<std::string> getAllWeightNames() const override
    {
        return {};
    }

    ActivationBuffers createBuffersForDevice(int /*device_idx*/, int /*max_seq_len*/) override
    {
        return ActivationBuffers{};
    }

    bool transformer_layer(int /*layer_idx*/, int /*seq_len*/) override
    {
        return true;
    }
};

class Test__PipelineBase_PrecisionMode : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create minimal model context for testing
        model_ctx_ = ModelContext::createForTesting("test_precision.gguf");
        mpi_ctx_ = nullptr; // Single-node test
        device_idx_ = -1;   // CPU
        placement_map_ = std::make_shared<WeightPlacementMap>(-1);
    }

    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    int device_idx_;
    std::shared_ptr<WeightPlacementMap> placement_map_;
};

// =============================================================================
// WEIGHT PRECISION CONFIGURATION TESTS
// =============================================================================

/**
 * @brief Test NATIVE weight precision mode (default)
 */
TEST_F(Test__PipelineBase_PrecisionMode, WeightPrecision_NativeDefault)
{
    PipelineConfig config;
    config.weight_precision = WeightPrecision::NATIVE;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    // Verify weight precision is stored correctly
    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::NATIVE);
}
/**
 * @brief Test FP32 weight precision conversion
 */
TEST_F(Test__PipelineBase_PrecisionMode, WeightPrecision_ConvertToFP32)
{
    PipelineConfig config;
    config.weight_precision = WeightPrecision::CONVERT_TO_FP32;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::CONVERT_TO_FP32);
}

/**
 * @brief Test BF16 weight precision conversion
 */
TEST_F(Test__PipelineBase_PrecisionMode, WeightPrecision_ConvertToBF16)
{
    PipelineConfig config;
    config.weight_precision = WeightPrecision::CONVERT_TO_BF16;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::CONVERT_TO_BF16);
}

/**
 * @brief Test INT8 weight precision conversion
 */
TEST_F(Test__PipelineBase_PrecisionMode, WeightPrecision_ConvertToINT8)
{
    PipelineConfig config;
    config.weight_precision = WeightPrecision::CONVERT_TO_INT8;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::CONVERT_TO_INT8);
}

// =============================================================================
// ACTIVATION PRECISION CONFIGURATION TESTS
// =============================================================================

/**
 * @brief Test FP32 activation precision mode (default)
 */
TEST_F(Test__PipelineBase_PrecisionMode, ActivationPrecision_FP32Default)
{
    PipelineConfig config;
    config.activation_precision = ActivationPrecision::FP32;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    // Verify activation precision is stored correctly
    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::FP32);
}

/**
 * @brief Test BF16 activation precision mode
 */
TEST_F(Test__PipelineBase_PrecisionMode, ActivationPrecision_BF16)
{
    PipelineConfig config;
    config.activation_precision = ActivationPrecision::BF16;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::BF16);
}

/**
 * @brief Test FP16 activation precision mode
 */
TEST_F(Test__PipelineBase_PrecisionMode, ActivationPrecision_FP16)
{
    PipelineConfig config;
    config.activation_precision = ActivationPrecision::FP16;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::FP16);
}

/**
 * @brief Test Q8_1 activation precision mode (quantized 8-bit with per-block scaling)
 */
TEST_F(Test__PipelineBase_PrecisionMode, ActivationPrecision_Q8_1)
{
    PipelineConfig config;
    config.activation_precision = ActivationPrecision::Q8_1;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::Q8_1);
}

// =============================================================================
// COMBINED PRECISION CONFIGURATION TESTS
// =============================================================================

/**
 * @brief Test default PipelineConfig uses NATIVE weights and FP32 activations
 */
TEST_F(Test__PipelineBase_PrecisionMode, DefaultConfigIsNativeAndFP32)
{
    PipelineConfig config; // Default constructor

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    // Default should be NATIVE weights, FP32 activations
    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::NATIVE);
    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::FP32);
}

/**
 * @brief Test typical MIXED mode: quantized weights (NATIVE) + FP32 activations
 */
TEST_F(Test__PipelineBase_PrecisionMode, MixedMode_NativeWeightsFP32Activations)
{
    PipelineConfig config;
    config.weight_precision = WeightPrecision::NATIVE;       // Keep quantized weights
    config.activation_precision = ActivationPrecision::FP32; // FP32 computation

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::NATIVE);
    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::FP32);
}

/**
 * @brief Test INT8 weights with FP32 activations (INT8 inference mode)
 */
TEST_F(Test__PipelineBase_PrecisionMode, INT8Weights_FP32Activations)
{
    PipelineConfig config;
    config.weight_precision = WeightPrecision::CONVERT_TO_INT8;
    config.activation_precision = ActivationPrecision::FP32;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config);

    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::CONVERT_TO_INT8);
    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::FP32);
}

/**
 * @brief Test precision persists through config copy
 */
TEST_F(Test__PipelineBase_PrecisionMode, PrecisionPersistsThroughCopy)
{
    PipelineConfig config1;
    config1.weight_precision = WeightPrecision::CONVERT_TO_INT8;
    config1.activation_precision = ActivationPrecision::BF16;

    // Copy config
    PipelineConfig config2 = config1;

    MockPipelineForPrecision pipeline(model_ctx_, mpi_ctx_, device_idx_, placement_map_, config2);

    // Should preserve both precision settings
    EXPECT_EQ(pipeline.getWeightPrecision(), WeightPrecision::CONVERT_TO_INT8);
    EXPECT_EQ(pipeline.getActivationPrecision(), ActivationPrecision::BF16);
}

// =============================================================================
// PRECISION ENUM VALUE TESTS (standalone, don't need fixture)
// =============================================================================

/**
 * @brief Test WeightPrecision enum values are distinct
 */
TEST(WeightPrecisionEnum, EnumValuesDistinct)
{
    // Verify enum values are distinct
    EXPECT_NE(static_cast<int>(WeightPrecision::NATIVE), static_cast<int>(WeightPrecision::CONVERT_TO_FP32));
    EXPECT_NE(static_cast<int>(WeightPrecision::NATIVE), static_cast<int>(WeightPrecision::CONVERT_TO_BF16));
    EXPECT_NE(static_cast<int>(WeightPrecision::NATIVE), static_cast<int>(WeightPrecision::CONVERT_TO_FP16));
    EXPECT_NE(static_cast<int>(WeightPrecision::NATIVE), static_cast<int>(WeightPrecision::CONVERT_TO_INT8));

    EXPECT_NE(static_cast<int>(WeightPrecision::CONVERT_TO_FP32), static_cast<int>(WeightPrecision::CONVERT_TO_BF16));
    EXPECT_NE(static_cast<int>(WeightPrecision::CONVERT_TO_FP32), static_cast<int>(WeightPrecision::CONVERT_TO_INT8));
}

/**
 * @brief Test ActivationPrecision enum values are distinct
 */
TEST(ActivationPrecisionEnum, EnumValuesDistinct)
{
    // Verify enum values are distinct
    EXPECT_NE(static_cast<int>(ActivationPrecision::FP32), static_cast<int>(ActivationPrecision::BF16));
    EXPECT_NE(static_cast<int>(ActivationPrecision::FP32), static_cast<int>(ActivationPrecision::FP16));
    EXPECT_NE(static_cast<int>(ActivationPrecision::FP32), static_cast<int>(ActivationPrecision::Q8_1));

    EXPECT_NE(static_cast<int>(ActivationPrecision::BF16), static_cast<int>(ActivationPrecision::FP16));
    EXPECT_NE(static_cast<int>(ActivationPrecision::BF16), static_cast<int>(ActivationPrecision::Q8_1));
}

/**
 * @brief Test NATIVE is the first WeightPrecision value (for default initialization)
 */
TEST(WeightPrecisionEnum, NativeIsFirstValue)
{
    // NATIVE should be the first enum value (0) for default initialization
    EXPECT_EQ(static_cast<int>(WeightPrecision::NATIVE), 0);
}

/**
 * @brief Test FP32 is the first ActivationPrecision value (for default initialization)
 */
TEST(ActivationPrecisionEnum, FP32IsFirstValue)
{
    // FP32 should be the first enum value (0) for default initialization
    EXPECT_EQ(static_cast<int>(ActivationPrecision::FP32), 0);
}

// =============================================================================
// MAIN
// =============================================================================
