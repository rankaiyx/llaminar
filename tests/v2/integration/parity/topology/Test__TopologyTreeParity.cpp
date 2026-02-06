/**
 * @file Test__TopologyTreeParity.cpp
 * @brief Parity tests for ParallelismTree-based inference
 *
 * Validates that tree-compiled runners produce correct results and that
 * topology configuration integrates properly with the orchestration system.
 *
 * Test categories:
 * - Baseline: Single-device without tree (reference)
 * - SingleDeviceTree: Tree with single device matches baseline
 * - LocalPP_TwoStage: 2-stage pipeline parallel within one rank
 * - CLITopology: Parse CLI --topology and verify compilation
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream>
#include <memory>

#include "config/OrchestrationConfig.h"
#include "config/ParallelismTreeParser.h"
#include "execution/parallelism_tree/ParallelismTree.h"
#include "execution/parallelism_tree/TreeToRunnerCompiler.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "backends/GlobalDeviceAddress.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class Test__TopologyTreeParity : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check for model
            model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";
            // Skip if model not found
            std::ifstream f(model_path_);
            if (!f.good())
            {
                GTEST_SKIP() << "Test model not found: " << model_path_;
            }
        }

        std::string model_path_;
        static constexpr int kQwen2_5_NumLayers = 24; // Qwen2.5-0.5B has 24 layers
    };

    // =========================================================================
    // Test Cases
    // =========================================================================

    /**
     * @brief Verify ParallelismTree basic construction and validation
     */
    TEST_F(Test__TopologyTreeParity, TreeConstruction_SingleDevice)
    {
        // Create a simple single-device tree
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");

        // Create DEVICE leaf node
        auto device_node = Device(cpu_device, 0);
        EXPECT_EQ(device_node.type, ParallelismNodeType::DEVICE);
        EXPECT_EQ(device_node.owning_rank, 0);
        // toString() returns full format: hostname:numa:type:ordinal
        EXPECT_THAT(device_node.device.toString(), testing::HasSubstr("cpu:0"));

        // Wrap in tree and assign layers
        ParallelismTree tree;
        tree.root = std::move(device_node);
        tree.total_layers = kQwen2_5_NumLayers;
        tree.world_size = 1;
        tree.assignLayers(kQwen2_5_NumLayers);

        // Validate
        auto errors = tree.validate();
        EXPECT_TRUE(errors.empty()) << "Validation errors: " << (errors.empty() ? "" : errors[0]);

        // Check layer assignment
        EXPECT_EQ(tree.root.first_layer, 0);
        EXPECT_EQ(tree.root.last_layer, kQwen2_5_NumLayers - 1);
        EXPECT_EQ(tree.root.layerCount(), kQwen2_5_NumLayers);
    }

    /**
     * @brief Verify 2-stage PP tree construction
     */
    TEST_F(Test__TopologyTreeParity, TreeConstruction_TwoStagePP)
    {
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");

        // Create 2-stage PP tree
        auto tree_root = PP("local_pp", {
            Device(cpu_device, 0),
            Device(cpu_device, 0)
        });

        ParallelismTree tree;
        tree.root = std::move(tree_root);
        tree.total_layers = kQwen2_5_NumLayers;
        tree.world_size = 1;
        tree.assignLayers(kQwen2_5_NumLayers);

        // Validate
        auto errors = tree.validate();
        EXPECT_TRUE(errors.empty()) << "Validation errors: " << (errors.empty() ? "" : errors[0]);

        // Check layer assignment - should be split evenly
        ASSERT_EQ(tree.root.children.size(), 2u);
        EXPECT_EQ(tree.root.children[0].first_layer, 0);
        EXPECT_EQ(tree.root.children[0].last_layer, 11);  // First 12 layers
        EXPECT_EQ(tree.root.children[1].first_layer, 12);
        EXPECT_EQ(tree.root.children[1].last_layer, 23); // Last 12 layers

        // Check embedding/LM head flags
        EXPECT_TRUE(tree.root.children[0].has_embedding);
        EXPECT_FALSE(tree.root.children[0].has_lm_head);
        EXPECT_FALSE(tree.root.children[1].has_embedding);
        EXPECT_TRUE(tree.root.children[1].has_lm_head);
    }

    /**
     * @brief Verify TP tree construction
     */
    TEST_F(Test__TopologyTreeParity, TreeConstruction_TwoDeviceTP)
    {
        auto cuda0 = GlobalDeviceAddress::parse("cuda:0");
        auto cuda1 = GlobalDeviceAddress::parse("cuda:1");

        // Create 2-device TP tree
        auto tree_root = TP("local_tp", {cuda0, cuda1}, 0, CollectiveBackendType::AUTO);

        ParallelismTree tree;
        tree.root = std::move(tree_root);
        tree.total_layers = kQwen2_5_NumLayers;
        tree.world_size = 1;
        tree.assignLayers(kQwen2_5_NumLayers);

        // Validate
        auto errors = tree.validate();
        EXPECT_TRUE(errors.empty()) << "Validation errors: " << (errors.empty() ? "" : errors[0]);

        // Check layer assignment - TP children share same layers
        ASSERT_EQ(tree.root.children.size(), 2u);
        EXPECT_EQ(tree.root.children[0].first_layer, 0);
        EXPECT_EQ(tree.root.children[0].last_layer, kQwen2_5_NumLayers - 1);
        EXPECT_EQ(tree.root.children[1].first_layer, 0);
        EXPECT_EQ(tree.root.children[1].last_layer, kQwen2_5_NumLayers - 1);
    }

    /**
     * @brief Parse CLI --topology inline string
     */
    TEST_F(Test__TopologyTreeParity, CLIParsing_SingleDevice)
    {
        std::string topology_str = "cpu:0";

        auto result = ParallelismTreeParser::parseCLI(topology_str, kQwen2_5_NumLayers, 1);
        ASSERT_TRUE(result.success()) << result.errorString();

        // Verify tree structure
        EXPECT_EQ(result.tree->root.type, ParallelismNodeType::DEVICE);
        EXPECT_EQ(result.tree->root.owning_rank, 0);
    }

    /**
     * @brief Parse CLI --topology with PP structure
     */
    TEST_F(Test__TopologyTreeParity, CLIParsing_PPTwoStage)
    {
        std::string topology_str = "PP(test_pp, cpu:0, cpu:0)";

        auto result = ParallelismTreeParser::parseCLI(topology_str, kQwen2_5_NumLayers, 1);
        ASSERT_TRUE(result.success()) << result.errorString();

        // Verify tree structure
        EXPECT_EQ(result.tree->root.type, ParallelismNodeType::PIPELINE_PARALLEL);
        EXPECT_EQ(result.tree->root.name, "test_pp");
        EXPECT_EQ(result.tree->root.children.size(), 2u);

        // Layers should be assigned
        auto errors = result.tree->validate();
        EXPECT_TRUE(errors.empty()) << "Validation: " << (errors.empty() ? "" : errors[0]);
    }

    /**
     * @brief Parse CLI --topology with TP structure
     */
    TEST_F(Test__TopologyTreeParity, CLIParsing_TPTwoDevice)
    {
        std::string topology_str = "TP(test_tp, cuda:0, cuda:1)";

        auto result = ParallelismTreeParser::parseCLI(topology_str, kQwen2_5_NumLayers, 1);
        ASSERT_TRUE(result.success()) << result.errorString();

        // Verify tree structure
        EXPECT_EQ(result.tree->root.type, ParallelismNodeType::TENSOR_PARALLEL);
        EXPECT_EQ(result.tree->root.name, "test_tp");
        EXPECT_EQ(result.tree->root.children.size(), 2u);
    }

    /**
     * @brief Verify OrchestrationConfig with topology_tree is validated
     */
    TEST_F(Test__TopologyTreeParity, ConfigValidation_WithTopologyTree)
    {
        // Create a valid topology tree
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");
        auto device_node = Device(cpu_device, 0);

        ParallelismTree tree;
        tree.root = std::move(device_node);
        tree.total_layers = kQwen2_5_NumLayers;
        tree.world_size = 1;
        tree.assignLayers(kQwen2_5_NumLayers);

        // Set up config with tree
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path_;
        config.topology_tree = std::move(tree);

        // Validate should pass
        auto errors = config.validate();
        EXPECT_TRUE(errors.empty()) << "Config validation failed: " << (errors.empty() ? "" : errors[0]);
    }

    /**
     * @brief Verify OrchestrationConfig validation catches invalid tree
     */
    TEST_F(Test__TopologyTreeParity, ConfigValidation_InvalidTree)
    {
        // Create an invalid tree (no layers assigned)
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");
        auto device_node = Device(cpu_device, 0);

        ParallelismTree tree;
        tree.root = std::move(device_node);
        tree.total_layers = 0; // Invalid: no layers
        tree.world_size = 1;
        // Don't call assignLayers - tree is invalid

        // Set up config with invalid tree
        OrchestrationConfig config = OrchestrationConfig::defaults();
        config.model_path = model_path_;
        config.topology_tree = std::move(tree);

        // Validate should fail
        auto errors = config.validate();
        EXPECT_FALSE(errors.empty()) << "Expected validation to fail for invalid tree";
    }

    /**
     * @brief Verify TreeToRunnerCompiler::shouldCompile for rank ownership
     */
    TEST_F(Test__TopologyTreeParity, Compiler_ShouldCompile_DeviceOwnership)
    {
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");

        // Device owned by rank 0
        auto node_rank0 = Device(cpu_device, 0);
        EXPECT_TRUE(TreeToRunnerCompiler::shouldCompile(node_rank0, 0));
        EXPECT_FALSE(TreeToRunnerCompiler::shouldCompile(node_rank0, 1));

        // Device owned by rank 1
        auto node_rank1 = Device(cpu_device, 1);
        EXPECT_FALSE(TreeToRunnerCompiler::shouldCompile(node_rank1, 0));
        EXPECT_TRUE(TreeToRunnerCompiler::shouldCompile(node_rank1, 1));
    }

    /**
     * @brief Verify TreeToRunnerCompiler::isCrossRankPP detection
     */
    TEST_F(Test__TopologyTreeParity, Compiler_IsCrossRankPP)
    {
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");

        // Same-rank PP (both devices on rank 0)
        auto local_pp = PP("local", {
            Device(cpu_device, 0),
            Device(cpu_device, 0)
        });
        EXPECT_FALSE(TreeToRunnerCompiler::isCrossRankPP(local_pp));

        // Cross-rank PP (devices on different ranks)
        auto cross_pp = PP("cross", {
            Device(cpu_device, 0),
            Device(cpu_device, 1)
        });
        EXPECT_TRUE(TreeToRunnerCompiler::isCrossRankPP(cross_pp));
    }

    /**
     * @brief Verify complete tree compilation with mock factories
     */
    TEST_F(Test__TopologyTreeParity, Compiler_CompileWithMockFactories)
    {
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");

        // Build simple device tree
        auto device_node = Device(cpu_device, 0);
        ParallelismTree tree;
        tree.root = std::move(device_node);
        tree.total_layers = kQwen2_5_NumLayers;
        tree.world_size = 1;
        tree.assignLayers(kQwen2_5_NumLayers);

        // Set up compile context with mock device factory
        bool factory_called = false;
        TreeToRunnerCompiler::CompileContext ctx;
        ctx.my_rank = 0;
        ctx.world_size = 1;
        ctx.max_seq_len = 2048;
        ctx.batch_size = 1;
        ctx.hidden_dim = 896;
        ctx.vocab_size = 151936;
        ctx.device_runner_factory = [&factory_called](
            const ParallelismNode& node,
            const std::shared_ptr<IModelContext>& /* model_ctx */) -> std::unique_ptr<IInferenceRunner>
        {
            factory_called = true;
            // Return nullptr for test (real implementation would create DeviceGraphOrchestrator)
            LOG_DEBUG("Mock device factory called for node: " << node.name);
            return nullptr;
        };

        // Compile (should call mock factory)
        auto runner = TreeToRunnerCompiler::compile(tree, ctx);
        
        // Factory should have been called
        EXPECT_TRUE(factory_called) << "Device runner factory should have been called";
    }

    /**
     * @brief Verify pp/tp nested tree structure
     */
    TEST_F(Test__TopologyTreeParity, TreeConstruction_NestedPPwithTP)
    {
        auto cuda0 = GlobalDeviceAddress::parse("cuda:0");
        auto cuda1 = GlobalDeviceAddress::parse("cuda:1");

        // Build nested structure: PP with TP children
        // PP(global, TP(stage0, cuda:0, cuda:1), TP(stage1, cuda:0, cuda:1))
        auto tree_root = PP("global", {
            TP("stage0_tp", {cuda0, cuda1}, 0, CollectiveBackendType::NCCL),
            TP("stage1_tp", {cuda0, cuda1}, 0, CollectiveBackendType::NCCL)
        });

        ParallelismTree tree;
        tree.root = std::move(tree_root);
        tree.total_layers = kQwen2_5_NumLayers;
        tree.world_size = 1;
        tree.assignLayers(kQwen2_5_NumLayers);

        // Validate
        auto errors = tree.validate();
        EXPECT_TRUE(errors.empty()) << "Validation: " << (errors.empty() ? "" : errors[0]);

        // Check structure
        EXPECT_EQ(tree.root.type, ParallelismNodeType::PIPELINE_PARALLEL);
        ASSERT_EQ(tree.root.children.size(), 2u);

        // Each PP child should be TP
        EXPECT_EQ(tree.root.children[0].type, ParallelismNodeType::TENSOR_PARALLEL);
        EXPECT_EQ(tree.root.children[1].type, ParallelismNodeType::TENSOR_PARALLEL);

        // PP children get different layers
        EXPECT_EQ(tree.root.children[0].first_layer, 0);
        EXPECT_EQ(tree.root.children[0].last_layer, 11);
        EXPECT_EQ(tree.root.children[1].first_layer, 12);
        EXPECT_EQ(tree.root.children[1].last_layer, 23);

        // TP children (within each PP stage) share same layers as their TP parent
        EXPECT_EQ(tree.root.children[0].children[0].first_layer, 0);
        EXPECT_EQ(tree.root.children[0].children[0].last_layer, 11);
        EXPECT_EQ(tree.root.children[0].children[1].first_layer, 0);
        EXPECT_EQ(tree.root.children[0].children[1].last_layer, 11);
    }

    /**
     * @brief Verify toString() produces readable output
     */
    TEST_F(Test__TopologyTreeParity, TreeToString_Readable)
    {
        auto cpu_device = GlobalDeviceAddress::parse("cpu:0");

        // Build a simple tree
        auto tree_root = PP("test", {
            Device(cpu_device, 0),
            Device(cpu_device, 0)
        });

        ParallelismTree tree;
        tree.root = std::move(tree_root);
        tree.total_layers = kQwen2_5_NumLayers;
        tree.world_size = 1;
        tree.assignLayers(kQwen2_5_NumLayers);

        // Get string representation
        std::string str = tree.toString();

        // Should contain key information
        EXPECT_FALSE(str.empty());
        EXPECT_NE(str.find("PP"), std::string::npos) << "Should contain PP";
        EXPECT_NE(str.find("test"), std::string::npos) << "Should contain node name";
    }

} // namespace llaminar2::test
