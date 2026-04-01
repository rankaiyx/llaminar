/**
 * @file Test__SchemaFactoryRegistry.cpp
 * @brief Unit tests for SchemaFactoryRegistry self-registration pattern
 * @author David Sanftenberg
 * @date March 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "execution/local_execution/graph/GraphSchema.h"

using namespace llaminar2;

// ============================================================================
// Stub schema factory for self-registration testing
// ============================================================================

class StubSchemaFactory : public ISchemaFactory
{
public:
    GraphSchema createSchema() const override { return {}; }
    std::string architectureName() const override { return "test_stub_schema"; }
    WeightShardingConfig getWeightShardingConfig() const override
    {
        WeightShardingConfig cfg;
        cfg.patterns.push_back(WeightShardingPattern{".*", WeightShardingMode::Replicate, "test"});
        return cfg;
    }
    StageShardingConfig getStageShardingConfig() const override
    {
        StageShardingConfig cfg;
        cfg["test_stage"] = SnapshotShardingMode::REPLICATED;
        return cfg;
    }
    bool isWeightOptional(const std::string &) const override { return false; }
    std::vector<std::string> layerWeightSuffixes() const override { return {}; }
};

// Register a test-only architecture
static SchemaFactoryRegistrar s_stub_schema("test_stub_schema",
                                            []()
                                            { return std::make_unique<StubSchemaFactory>(); });

// ============================================================================
// Tests
// ============================================================================

class Test__SchemaFactoryRegistry : public ::testing::Test
{
};

TEST_F(Test__SchemaFactoryRegistry, IsSupported_Registered)
{
    EXPECT_TRUE(SchemaFactoryRegistry::isSupported("test_stub_schema"));
    EXPECT_TRUE(SchemaFactoryRegistry::isSupported("TEST_STUB_SCHEMA"));
}

TEST_F(Test__SchemaFactoryRegistry, IsSupported_Unknown)
{
    EXPECT_FALSE(SchemaFactoryRegistry::isSupported("nonexistent"));
}

TEST_F(Test__SchemaFactoryRegistry, GetFactory_Registered_Succeeds)
{
    auto factory = SchemaFactoryRegistry::getFactory("test_stub_schema");
    ASSERT_NE(factory, nullptr);
}

TEST_F(Test__SchemaFactoryRegistry, GetFactory_CaseInsensitive)
{
    auto factory = SchemaFactoryRegistry::getFactory("TEST_STUB_SCHEMA");
    ASSERT_NE(factory, nullptr);
}

TEST_F(Test__SchemaFactoryRegistry, GetFactory_Unknown_Throws)
{
    EXPECT_THROW(
        SchemaFactoryRegistry::getFactory("nonexistent"),
        std::runtime_error);
}

TEST_F(Test__SchemaFactoryRegistry, GetWeightShardingConfig_Registered)
{
    auto config = SchemaFactoryRegistry::getWeightShardingConfig("test_stub_schema");
    EXPECT_FALSE(config.patterns.empty());
}

TEST_F(Test__SchemaFactoryRegistry, SupportedArchitectures_ContainsRegistered)
{
    auto archs = SchemaFactoryRegistry::supportedArchitectures();
    EXPECT_FALSE(archs.empty());

    bool found = false;
    for (const auto &a : archs)
    {
        if (a == "test_stub_schema")
            found = true;
    }
    EXPECT_TRUE(found);
}

TEST_F(Test__SchemaFactoryRegistry, GetStageShardingConfig_Registered)
{
    auto config = SchemaFactoryRegistry::getStageShardingConfig("test_stub_schema");
    EXPECT_FALSE(config.empty());
}
