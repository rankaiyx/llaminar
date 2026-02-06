/**
 * @file Test__CompoundShardInfo.cpp
 * @brief Unit tests for CompoundShardInfo multi-level shard descriptor
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for CompoundShardInfo which describes the sharding state at a
 * specific leaf device in a potentially nested parallelism tree:
 * - Layer range from PP assignment
 * - TP shard index and total (compound for nested TP)
 * - Work fraction (for proportional TP)
 *
 * All tests are pure data-structure tests — no MPI, no GPU, no model loading.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "execution/parallelism_tree/CompoundShardInfo.h"
#include "execution/parallelism_tree/ParallelismTree.h"
#include "backends/GlobalDeviceAddress.h"

using namespace llaminar2;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Build a simple single-device tree
 */
static ParallelismTree buildSingleDevice()
{
    auto root = Device(GlobalDeviceAddress::cuda(0), 0);
    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(24);
    return tree;
}

/**
 * @brief Build a 2-way TP tree
 *
 * TP(tp2, [cuda:0, cuda:1], rank=0)
 */
static ParallelismTree buildSimpleTP2()
{
    auto root = TP("tp2", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0);
    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(24);
    return tree;
}

/**
 * @brief Build a PP(TP, TP) tree: 2 pipeline stages, each with 2-way TP
 *
 * PP(root)
 * ├── TP(stage0, [cuda:0, cuda:1], rank=0)
 * └── TP(stage1, [cuda:2, cuda:3], rank=0)
 */
static ParallelismTree buildNestedPPTP()
{
    auto root = PP("root", {
                               TP("stage0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0),
                               TP("stage1", {GlobalDeviceAddress::cuda(2), GlobalDeviceAddress::cuda(3)}, 0),
                           });

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(24);
    return tree;
}

/**
 * @brief Build a 4-way TP tree with proportional weights
 *
 * TP(tp4, [cuda:0, cuda:1, cuda:2, cuda:3], rank=0, weights=[0.4, 0.3, 0.2, 0.1])
 */
static ParallelismTree buildProportionalTP4()
{
    ParallelismNode root;
    root.type = ParallelismNodeType::TENSOR_PARALLEL;
    root.name = "tp4";
    root.backend = CollectiveBackendType::NCCL;
    root.tp_weights = {0.4f, 0.3f, 0.2f, 0.1f};

    for (int i = 0; i < 4; ++i)
    {
        ParallelismNode leaf;
        leaf.type = ParallelismNodeType::DEVICE;
        leaf.name = "dev" + std::to_string(i);
        leaf.device = GlobalDeviceAddress::cuda(i);
        leaf.owning_rank = 0;
        root.children.push_back(std::move(leaf));
    }

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(24);
    return tree;
}

/**
 * @brief Build a 3-level deep nested TP tree: TP(TP(a,b), TP(c,d))
 *
 * TP(outer)
 * ├── TP(inner0, [cuda:0, cuda:1], rank=0)
 * └── TP(inner1, [cuda:2, cuda:3], rank=0)
 *
 * This creates 4 total shards across 2 levels of TP.
 */
static ParallelismTree buildDeepNestedTP()
{
    auto inner0 = TP("inner0", {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)}, 0);
    auto inner1 = TP("inner1", {GlobalDeviceAddress::cuda(2), GlobalDeviceAddress::cuda(3)}, 0);

    auto root = TP("outer", {inner0, inner1});

    ParallelismTree tree;
    tree.root = std::move(root);
    tree.world_size = 1;
    tree.assignLayers(24);
    return tree;
}

// =============================================================================
// Test Suite: Factory Methods
// =============================================================================

/**
 * @test SingleDevice
 * @brief Single device → 1 shard, full work, all layers
 */
TEST(Test__CompoundShardInfo, SingleDevice)
{
    auto tree = buildSingleDevice();
    auto info = CompoundShardInfo::fromLeaf(tree.root);

    EXPECT_EQ(info.layer_first, 0);
    EXPECT_EQ(info.layer_last, 23);
    EXPECT_EQ(info.layerCount(), 24);
    EXPECT_EQ(info.tp_shard_index, 0);
    EXPECT_EQ(info.tp_total_shards, 1);
    EXPECT_FLOAT_EQ(info.work_fraction, 1.0f);
    EXPECT_FALSE(info.isSharded());
    EXPECT_TRUE(info.has_embedding);
    EXPECT_TRUE(info.has_lm_head);
}

/**
 * @test SimpleTP2Way
 * @brief 2-device TP → shard 0 or 1 of 2, 0.5 work each
 */
TEST(Test__CompoundShardInfo, SimpleTP2Way)
{
    auto tree = buildSimpleTP2();
    auto leaves = tree.root.leafDevices();
    ASSERT_EQ(leaves.size(), 2u);

    // First device (cuda:0)
    auto info0 = CompoundShardInfo::fromTreePath(*leaves[0], tree);
    EXPECT_EQ(info0.layer_first, 0);
    EXPECT_EQ(info0.layer_last, 23);
    EXPECT_EQ(info0.tp_shard_index, 0);
    EXPECT_EQ(info0.tp_total_shards, 2);
    EXPECT_FLOAT_EQ(info0.work_fraction, 0.5f);
    EXPECT_TRUE(info0.isSharded());

    // Second device (cuda:1)
    auto info1 = CompoundShardInfo::fromTreePath(*leaves[1], tree);
    EXPECT_EQ(info1.layer_first, 0);
    EXPECT_EQ(info1.layer_last, 23);
    EXPECT_EQ(info1.tp_shard_index, 1);
    EXPECT_EQ(info1.tp_total_shards, 2);
    EXPECT_FLOAT_EQ(info1.work_fraction, 0.5f);
    EXPECT_TRUE(info1.isSharded());
}

/**
 * @test NestedTPPP
 * @brief PP(TP(cuda:0, cuda:1), TP(cuda:2, cuda:3)) → each device shards correctly
 *
 * Stage 0 has layers 0-11, stage 1 has layers 12-23.
 * Each TP domain has 2 shards.
 */
TEST(Test__CompoundShardInfo, NestedTPPP)
{
    auto tree = buildNestedPPTP();
    auto leaves = tree.root.leafDevices();
    ASSERT_EQ(leaves.size(), 4u);

    // Stage 0, device 0 (cuda:0) - layers 0-11, shard 0/2
    auto info0 = CompoundShardInfo::fromTreePath(*leaves[0], tree);
    EXPECT_EQ(info0.layer_first, 0);
    EXPECT_EQ(info0.layer_last, 11);
    EXPECT_EQ(info0.tp_shard_index, 0);
    EXPECT_EQ(info0.tp_total_shards, 2);
    EXPECT_TRUE(info0.has_embedding);
    EXPECT_FALSE(info0.has_lm_head);

    // Stage 0, device 1 (cuda:1) - layers 0-11, shard 1/2
    auto info1 = CompoundShardInfo::fromTreePath(*leaves[1], tree);
    EXPECT_EQ(info1.layer_first, 0);
    EXPECT_EQ(info1.layer_last, 11);
    EXPECT_EQ(info1.tp_shard_index, 1);
    EXPECT_EQ(info1.tp_total_shards, 2);
    EXPECT_TRUE(info1.has_embedding);
    EXPECT_FALSE(info1.has_lm_head);

    // Stage 1, device 0 (cuda:2) - layers 12-23, shard 0/2
    auto info2 = CompoundShardInfo::fromTreePath(*leaves[2], tree);
    EXPECT_EQ(info2.layer_first, 12);
    EXPECT_EQ(info2.layer_last, 23);
    EXPECT_EQ(info2.tp_shard_index, 0);
    EXPECT_EQ(info2.tp_total_shards, 2);
    EXPECT_FALSE(info2.has_embedding);
    EXPECT_TRUE(info2.has_lm_head);

    // Stage 1, device 1 (cuda:3) - layers 12-23, shard 1/2
    auto info3 = CompoundShardInfo::fromTreePath(*leaves[3], tree);
    EXPECT_EQ(info3.layer_first, 12);
    EXPECT_EQ(info3.layer_last, 23);
    EXPECT_EQ(info3.tp_shard_index, 1);
    EXPECT_EQ(info3.tp_total_shards, 2);
    EXPECT_FALSE(info3.has_embedding);
    EXPECT_TRUE(info3.has_lm_head);
}

/**
 * @test DeepNesting
 * @brief 3-level tree with TP at multiple levels → compound shard products
 *
 * TP(outer, [TP(a,b), TP(c,d)]) → 4 total shards
 * Device indices: 0, 1, 2, 3 → shard indices: 0, 1, 2, 3
 */
TEST(Test__CompoundShardInfo, DeepNesting)
{
    auto tree = buildDeepNestedTP();
    auto leaves = tree.root.leafDevices();
    ASSERT_EQ(leaves.size(), 4u);

    // All should have the same layer range (TP duplicates layers)
    for (const auto *leaf : leaves)
    {
        auto info = CompoundShardInfo::fromTreePath(*leaf, tree);
        EXPECT_EQ(info.layer_first, 0);
        EXPECT_EQ(info.layer_last, 23);
        EXPECT_EQ(info.tp_total_shards, 4) << "2 levels of 2-way TP = 4 shards";
    }

    // Check individual shard indices
    // Outer TP child 0 -> inner TP children 0,1 -> shard indices 0,1
    // Outer TP child 1 -> inner TP children 0,1 -> shard indices 2,3
    auto info0 = CompoundShardInfo::fromTreePath(*leaves[0], tree);
    EXPECT_EQ(info0.tp_shard_index, 0);
    EXPECT_FLOAT_EQ(info0.work_fraction, 0.25f);

    auto info1 = CompoundShardInfo::fromTreePath(*leaves[1], tree);
    EXPECT_EQ(info1.tp_shard_index, 1);
    EXPECT_FLOAT_EQ(info1.work_fraction, 0.25f);

    auto info2 = CompoundShardInfo::fromTreePath(*leaves[2], tree);
    EXPECT_EQ(info2.tp_shard_index, 2);
    EXPECT_FLOAT_EQ(info2.work_fraction, 0.25f);

    auto info3 = CompoundShardInfo::fromTreePath(*leaves[3], tree);
    EXPECT_EQ(info3.tp_shard_index, 3);
    EXPECT_FLOAT_EQ(info3.work_fraction, 0.25f);
}

/**
 * @test ProportionalWeights
 * @brief TP with tp_weights → work_fraction reflects weights
 */
TEST(Test__CompoundShardInfo, ProportionalWeights)
{
    auto tree = buildProportionalTP4();
    auto leaves = tree.root.leafDevices();
    ASSERT_EQ(leaves.size(), 4u);

    // Weights: 0.4, 0.3, 0.2, 0.1 (sum = 1.0)
    auto info0 = CompoundShardInfo::fromTreePath(*leaves[0], tree);
    EXPECT_NEAR(info0.work_fraction, 0.4f, 1e-5f);

    auto info1 = CompoundShardInfo::fromTreePath(*leaves[1], tree);
    EXPECT_NEAR(info1.work_fraction, 0.3f, 1e-5f);

    auto info2 = CompoundShardInfo::fromTreePath(*leaves[2], tree);
    EXPECT_NEAR(info2.work_fraction, 0.2f, 1e-5f);

    auto info3 = CompoundShardInfo::fromTreePath(*leaves[3], tree);
    EXPECT_NEAR(info3.work_fraction, 0.1f, 1e-5f);
}

// =============================================================================
// Test Suite: Validation
// =============================================================================

/**
 * @test ValidInfo
 * @brief Valid CompoundShardInfo passes validation
 */
TEST(Test__CompoundShardInfo, ValidInfo)
{
    CompoundShardInfo info;
    info.layer_first = 0;
    info.layer_last = 23;
    info.tp_shard_index = 1;
    info.tp_total_shards = 4;
    info.work_fraction = 0.25f;

    auto errors = info.validate();
    EXPECT_TRUE(errors.empty()) << "Valid info should have no errors";
}

/**
 * @test ValidationErrors_NegativeLayerFirst
 * @brief Negative layer_first is caught
 */
TEST(Test__CompoundShardInfo, ValidationErrors_NegativeLayerFirst)
{
    CompoundShardInfo info;
    info.layer_first = -1;
    info.layer_last = 23;

    auto errors = info.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_NE(errors[0].find("layer_first"), std::string::npos);
}

/**
 * @test ValidationErrors_LayerRangeInverted
 * @brief layer_first > layer_last is caught
 */
TEST(Test__CompoundShardInfo, ValidationErrors_LayerRangeInverted)
{
    CompoundShardInfo info;
    info.layer_first = 10;
    info.layer_last = 5;

    auto errors = info.validate();
    EXPECT_FALSE(errors.empty());
    bool found = false;
    for (const auto &err : errors)
    {
        if (err.find("layer_first") != std::string::npos &&
            err.find("layer_last") != std::string::npos)
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Should catch inverted layer range";
}

/**
 * @test ValidationErrors_TotalShardsZero
 * @brief tp_total_shards < 1 is caught
 */
TEST(Test__CompoundShardInfo, ValidationErrors_TotalShardsZero)
{
    CompoundShardInfo info;
    info.layer_first = 0;
    info.layer_last = 23;
    info.tp_total_shards = 0;

    auto errors = info.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_NE(errors[0].find("tp_total_shards"), std::string::npos);
}

/**
 * @test ValidationErrors_ShardIndexOutOfRange
 * @brief tp_shard_index >= tp_total_shards is caught
 */
TEST(Test__CompoundShardInfo, ValidationErrors_ShardIndexOutOfRange)
{
    CompoundShardInfo info;
    info.layer_first = 0;
    info.layer_last = 23;
    info.tp_shard_index = 4;
    info.tp_total_shards = 4;

    auto errors = info.validate();
    EXPECT_FALSE(errors.empty());
    bool found = false;
    for (const auto &err : errors)
    {
        if (err.find("tp_shard_index") != std::string::npos)
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Should catch shard index >= total";
}

/**
 * @test ValidationErrors_WorkFractionZero
 * @brief work_fraction <= 0 is caught
 */
TEST(Test__CompoundShardInfo, ValidationErrors_WorkFractionZero)
{
    CompoundShardInfo info;
    info.layer_first = 0;
    info.layer_last = 23;
    info.work_fraction = 0.0f;

    auto errors = info.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_NE(errors[0].find("work_fraction"), std::string::npos);
}

/**
 * @test ValidationErrors_WorkFractionTooLarge
 * @brief work_fraction > 1 is caught
 */
TEST(Test__CompoundShardInfo, ValidationErrors_WorkFractionTooLarge)
{
    CompoundShardInfo info;
    info.layer_first = 0;
    info.layer_last = 23;
    info.work_fraction = 1.5f;

    auto errors = info.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_NE(errors[0].find("work_fraction"), std::string::npos);
}

// =============================================================================
// Test Suite: Serialization and Comparison
// =============================================================================

/**
 * @test ToString
 * @brief Verify human-readable output
 */
TEST(Test__CompoundShardInfo, ToString)
{
    CompoundShardInfo info;
    info.layer_first = 5;
    info.layer_last = 15;
    info.tp_shard_index = 2;
    info.tp_total_shards = 4;
    info.work_fraction = 0.25f;
    info.has_embedding = false;
    info.has_lm_head = true;

    std::string str = info.toString();

    // Check for expected content
    EXPECT_NE(str.find("[5, 15]"), std::string::npos);
    EXPECT_NE(str.find("11 layers"), std::string::npos);
    EXPECT_NE(str.find("2 of 4"), std::string::npos);
    EXPECT_NE(str.find("sharded"), std::string::npos);
    EXPECT_NE(str.find("0.25"), std::string::npos);
    EXPECT_NE(str.find("has_lm_head: true"), std::string::npos);
}

/**
 * @test Equality
 * @brief Verify equality operator
 */
TEST(Test__CompoundShardInfo, Equality)
{
    CompoundShardInfo a;
    a.layer_first = 0;
    a.layer_last = 23;
    a.tp_shard_index = 1;
    a.tp_total_shards = 4;
    a.work_fraction = 0.25f;
    a.has_embedding = true;
    a.has_lm_head = false;

    CompoundShardInfo b = a; // Copy

    EXPECT_EQ(a, b);

    // Change each field and verify inequality
    b.layer_first = 1;
    EXPECT_NE(a, b);
    b.layer_first = 0;

    b.layer_last = 22;
    EXPECT_NE(a, b);
    b.layer_last = 23;

    b.tp_shard_index = 2;
    EXPECT_NE(a, b);
    b.tp_shard_index = 1;

    b.tp_total_shards = 8;
    EXPECT_NE(a, b);
    b.tp_total_shards = 4;

    b.work_fraction = 0.5f;
    EXPECT_NE(a, b);
    b.work_fraction = 0.25f;

    b.has_embedding = false;
    EXPECT_NE(a, b);
    b.has_embedding = true;

    b.has_lm_head = true;
    EXPECT_NE(a, b);
    b.has_lm_head = false;

    EXPECT_EQ(a, b); // Back to equal
}

/**
 * @test IsSharded
 * @brief Verify isSharded() predicate
 */
TEST(Test__CompoundShardInfo, IsSharded)
{
    CompoundShardInfo info;

    info.tp_total_shards = 1;
    EXPECT_FALSE(info.isSharded());

    info.tp_total_shards = 2;
    EXPECT_TRUE(info.isSharded());

    info.tp_total_shards = 100;
    EXPECT_TRUE(info.isSharded());
}

/**
 * @test LayerCount
 * @brief Verify layerCount() computation
 */
TEST(Test__CompoundShardInfo, LayerCount)
{
    CompoundShardInfo info;

    // Unassigned
    info.layer_first = -1;
    info.layer_last = -1;
    EXPECT_EQ(info.layerCount(), 0);

    // Single layer
    info.layer_first = 5;
    info.layer_last = 5;
    EXPECT_EQ(info.layerCount(), 1);

    // Multiple layers
    info.layer_first = 0;
    info.layer_last = 23;
    EXPECT_EQ(info.layerCount(), 24);
}

/**
 * @test FromLeafVsFromTreePath
 * @brief fromLeaf() gives simple defaults, fromTreePath() computes TP
 */
TEST(Test__CompoundShardInfo, FromLeafVsFromTreePath)
{
    auto tree = buildSimpleTP2();
    auto leaves = tree.root.leafDevices();
    ASSERT_EQ(leaves.size(), 2u);

    // fromLeaf just takes the leaf's layer range, defaults for TP
    auto fromLeaf = CompoundShardInfo::fromLeaf(*leaves[1]);
    EXPECT_EQ(fromLeaf.layer_first, 0);
    EXPECT_EQ(fromLeaf.layer_last, 23);
    EXPECT_EQ(fromLeaf.tp_shard_index, 0);    // Default
    EXPECT_EQ(fromLeaf.tp_total_shards, 1);   // Default
    EXPECT_FLOAT_EQ(fromLeaf.work_fraction, 1.0f); // Default

    // fromTreePath computes TP from tree structure
    auto fromPath = CompoundShardInfo::fromTreePath(*leaves[1], tree);
    EXPECT_EQ(fromPath.layer_first, 0);
    EXPECT_EQ(fromPath.layer_last, 23);
    EXPECT_EQ(fromPath.tp_shard_index, 1);   // Second child
    EXPECT_EQ(fromPath.tp_total_shards, 2);  // 2-way TP
    EXPECT_FLOAT_EQ(fromPath.work_fraction, 0.5f); // Equal split
}
