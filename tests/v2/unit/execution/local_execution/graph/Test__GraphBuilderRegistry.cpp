/**
 * @file Test__GraphBuilderRegistry.cpp
 * @brief Unit tests for GraphBuilderRegistry (self-registration pattern)
 * @author David Sanftenberg
 * @date March 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/GraphBuilderRegistry.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "models/GraphTypes.h"

using namespace llaminar2;

// ============================================================================
// Minimal stub graph builder for testing
// ============================================================================

class StubGraphBuilder : public IGraphBuilder
{
public:
    StubGraphBuilder(const GraphConfig &cfg, std::shared_ptr<IMPIContext> mpi)
        : config_(cfg), mpi_(std::move(mpi)) {}

    // Pure virtual overrides
    ComputeGraph buildForwardGraph(const ForwardInput &, ForwardOutput &) override { return {}; }
    ComputeGraph buildLayerGraph(const LayerContext &) override { return {}; }
    const GraphConfig &config() const override { return config_; }
    void setWeights(const ModelWeights &) override {}
    void setBuffers(const ModelBuffers &) override {}
    const ModelBuffers &buffers() const override
    {
        static ModelBuffers empty;
        return empty;
    }

    // Optional overrides
    int numLayers() const override { return config_.n_layers; }
    int hiddenDim() const override { return config_.d_model; }
    bool isInitialized() const override { return true; }
    std::string architectureName() const override { return "stub"; }

private:
    GraphConfig config_;
    std::shared_ptr<IMPIContext> mpi_;
};

// Register a test-only architecture via the registrar helper
static GraphBuilderRegistrar s_stub_reg("test_stub_arch",
                                        [](const GraphConfig &cfg, std::shared_ptr<IMPIContext> mpi)
                                        {
                                            return std::make_shared<StubGraphBuilder>(cfg, std::move(mpi));
                                        });

// ============================================================================
// Tests
// ============================================================================

class Test__GraphBuilderRegistry : public ::testing::Test
{
protected:
    GraphConfig makeMinimalConfig()
    {
        GraphConfig cfg{};
        cfg.n_layers = 4;
        cfg.d_model = 128;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 4;
        cfg.head_dim = 32;
        cfg.d_ff = 512;
        cfg.vocab_size = 1000;
        return cfg;
    }
};

TEST_F(Test__GraphBuilderRegistry, Create_KnownArch_Succeeds)
{
    auto cfg = makeMinimalConfig();
    auto builder = GraphBuilderRegistry::create("test_stub_arch", cfg, nullptr);
    ASSERT_NE(builder, nullptr);
    EXPECT_EQ(builder->architectureName(), "stub");
    EXPECT_EQ(builder->numLayers(), 4);
}

TEST_F(Test__GraphBuilderRegistry, Create_CaseInsensitive)
{
    auto cfg = makeMinimalConfig();
    auto builder = GraphBuilderRegistry::create("TEST_STUB_ARCH", cfg, nullptr);
    ASSERT_NE(builder, nullptr);
    EXPECT_EQ(builder->architectureName(), "stub");
}

TEST_F(Test__GraphBuilderRegistry, Create_UnknownArch_Throws)
{
    auto cfg = makeMinimalConfig();
    EXPECT_THROW(
        GraphBuilderRegistry::create("nonexistent_arch", cfg, nullptr),
        std::runtime_error);
}

TEST_F(Test__GraphBuilderRegistry, IsSupported_KnownArch)
{
    EXPECT_TRUE(GraphBuilderRegistry::isSupported("test_stub_arch"));
    EXPECT_TRUE(GraphBuilderRegistry::isSupported("TEST_STUB_ARCH"));
}

TEST_F(Test__GraphBuilderRegistry, IsSupported_UnknownArch)
{
    EXPECT_FALSE(GraphBuilderRegistry::isSupported("nonexistent_arch"));
}

TEST_F(Test__GraphBuilderRegistry, SupportedArchitectures_ContainsRegistered)
{
    auto archs = GraphBuilderRegistry::supportedArchitectures();
    EXPECT_FALSE(archs.empty());

    // Should contain our test stub
    bool found_stub = false;
    for (const auto &a : archs)
    {
        if (a == "test_stub_arch")
            found_stub = true;
    }
    EXPECT_TRUE(found_stub) << "test_stub_arch not found in supportedArchitectures()";
}
