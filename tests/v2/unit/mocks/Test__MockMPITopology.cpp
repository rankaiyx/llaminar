/**
 * @file Test__MockMPITopology.cpp
 * @brief Unit tests for MockMPITopology
 *
 * Tests the mock MPI topology implementation for:
 * - Basic topology queries (rank, world_size, node_count)
 * - Work distribution (head ranges, column ranges, row ranges)
 * - Heterogeneous cluster configurations (CPU-only, GPU, mixed)
 * - Builder pattern for custom configurations
 * - ClusterInventory generation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "mocks/MockMPITopology.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Basic Construction Tests
// ============================================================================

TEST(Test__MockMPITopology, CreateSimple_SingleRank)
{
    auto mock = MockMPITopology::createSimple(0, 1);
    
    EXPECT_EQ(mock->rank(), 0);
    EXPECT_EQ(mock->world_size(), 1);
    EXPECT_EQ(mock->node_count(), 1);
    EXPECT_EQ(mock->ranks_per_node(), 1);
    EXPECT_TRUE(mock->is_coordinator());
    EXPECT_TRUE(mock->is_compute_participant());
    EXPECT_TRUE(mock->is_node_leader());
}

TEST(Test__MockMPITopology, CreateSimple_TwoRanks)
{
    auto rank0 = MockMPITopology::createSimple(0, 2);
    auto rank1 = MockMPITopology::createSimple(1, 2);
    
    EXPECT_EQ(rank0->rank(), 0);
    EXPECT_EQ(rank1->rank(), 1);
    EXPECT_EQ(rank0->world_size(), 2);
    EXPECT_EQ(rank1->world_size(), 2);
    
    EXPECT_TRUE(rank0->is_coordinator());
    EXPECT_FALSE(rank1->is_coordinator());
}

TEST(Test__MockMPITopology, CreateSimple_MultiNode)
{
    // 4 ranks across 2 nodes (2 ranks per node)
    auto rank0 = MockMPITopology::createSimple(0, 4, 2);
    auto rank1 = MockMPITopology::createSimple(1, 4, 2);
    auto rank2 = MockMPITopology::createSimple(2, 4, 2);
    auto rank3 = MockMPITopology::createSimple(3, 4, 2);
    
    EXPECT_EQ(rank0->node_count(), 2);
    EXPECT_EQ(rank0->ranks_per_node(), 2);
    
    // Node leaders
    EXPECT_TRUE(rank0->is_node_leader());   // local_rank 0 on node 0
    EXPECT_FALSE(rank1->is_node_leader());  // local_rank 1 on node 0
    EXPECT_TRUE(rank2->is_node_leader());   // local_rank 0 on node 1
    EXPECT_FALSE(rank3->is_node_leader());  // local_rank 1 on node 1
    
    // Verify node IDs
    EXPECT_EQ(rank0->placement().node_id, 0);
    EXPECT_EQ(rank1->placement().node_id, 0);
    EXPECT_EQ(rank2->placement().node_id, 1);
    EXPECT_EQ(rank3->placement().node_id, 1);
}

TEST(Test__MockMPITopology, InvalidRank_ThrowsException)
{
    EXPECT_THROW(MockMPITopology::createSimple(-1, 2), std::invalid_argument);
    EXPECT_THROW(MockMPITopology::createSimple(2, 2), std::invalid_argument);
    EXPECT_THROW(MockMPITopology::createSimple(0, 0), std::invalid_argument);
}

// ============================================================================
// Work Distribution Tests
// ============================================================================

TEST(Test__MockMPITopology, HeadRange_TwoRanks_EvenSplit)
{
    auto rank0 = MockMPITopology::createSimple(0, 2);
    auto rank1 = MockMPITopology::createSimple(1, 2);
    
    // 14 heads, 2 ranks -> 7 heads each
    WorkRange heads0 = rank0->get_head_range(14);
    WorkRange heads1 = rank1->get_head_range(14);
    
    EXPECT_EQ(heads0.start, 0);
    EXPECT_EQ(heads0.end, 7);
    EXPECT_EQ(heads0.size(), 7);
    
    EXPECT_EQ(heads1.start, 7);
    EXPECT_EQ(heads1.end, 14);
    EXPECT_EQ(heads1.size(), 7);
}

TEST(Test__MockMPITopology, HeadRange_TwoRanks_OddSplit)
{
    auto rank0 = MockMPITopology::createSimple(0, 2);
    auto rank1 = MockMPITopology::createSimple(1, 2);
    
    // 15 heads, 2 ranks -> 8 + 7
    WorkRange heads0 = rank0->get_head_range(15);
    WorkRange heads1 = rank1->get_head_range(15);
    
    EXPECT_EQ(heads0.start, 0);
    EXPECT_EQ(heads0.end, 8);   // First rank gets extra
    
    EXPECT_EQ(heads1.start, 8);
    EXPECT_EQ(heads1.end, 15);
    
    // Total coverage
    EXPECT_EQ(heads0.size() + heads1.size(), 15);
}

TEST(Test__MockMPITopology, ColumnRange_FourRanks)
{
    // 4096 columns across 4 ranks
    for (int r = 0; r < 4; ++r) {
        auto mock = MockMPITopology::createSimple(r, 4);
        WorkRange cols = mock->get_column_range(4096);
        
        EXPECT_EQ(cols.start, r * 1024);
        EXPECT_EQ(cols.end, (r + 1) * 1024);
        EXPECT_EQ(cols.size(), 1024);
    }
}

TEST(Test__MockMPITopology, RowRange_ThreeRanks)
{
    // 100 rows across 3 ranks -> 34 + 33 + 33
    auto rank0 = MockMPITopology::createSimple(0, 3);
    auto rank1 = MockMPITopology::createSimple(1, 3);
    auto rank2 = MockMPITopology::createSimple(2, 3);
    
    WorkRange rows0 = rank0->get_row_range(100);
    WorkRange rows1 = rank1->get_row_range(100);
    WorkRange rows2 = rank2->get_row_range(100);
    
    // First rank gets extra
    EXPECT_EQ(rows0.size(), 34);
    EXPECT_EQ(rows1.size(), 33);
    EXPECT_EQ(rows2.size(), 33);
    
    // No gaps, no overlaps
    EXPECT_EQ(rows0.end, rows1.start);
    EXPECT_EQ(rows1.end, rows2.start);
    EXPECT_EQ(rows2.end, 100);
}

TEST(Test__MockMPITopology, VocabRange_SingleRank)
{
    auto mock = MockMPITopology::createSimple(0, 1);
    WorkRange vocab = mock->get_vocab_range(152064);
    
    EXPECT_EQ(vocab.start, 0);
    EXPECT_EQ(vocab.end, 152064);
    EXPECT_EQ(vocab.size(), 152064);
}

// ============================================================================
// Builder Pattern Tests
// ============================================================================

TEST(Test__MockMPITopology, Builder_CPUOnlyCluster)
{
    auto mock = MockMPITopologyBuilder()
        .addCPUOnlyRank(0, 0)
        .addCPUOnlyRank(1, 0)
        .setLocalRank(0)
        .build();
    
    EXPECT_EQ(mock->rank(), 0);
    EXPECT_EQ(mock->world_size(), 2);
    EXPECT_EQ(mock->node_count(), 1);
    EXPECT_FALSE(mock->has_accelerator());
}

TEST(Test__MockMPITopology, Builder_MixedCPUAndGPU)
{
    auto mock = MockMPITopologyBuilder()
        .addCPUOnlyRank(0, 0)
        .addGPURank(1, 0, 0)  // Rank 1 has CUDA:0
        .setLocalRank(1)
        .build();
    
    EXPECT_EQ(mock->rank(), 1);
    EXPECT_TRUE(mock->has_accelerator());
    EXPECT_EQ(mock->get_device(), 0);  // CUDA device 0
    
    // Verify devices
    const auto& devices = mock->get_devices();
    EXPECT_EQ(devices.size(), 2);  // CPU + GPU
}

TEST(Test__MockMPITopology, Builder_HeterogeneousMultiNode)
{
    // Node 0: 2 CPU-only ranks
    // Node 1: 2 GPU ranks
    auto mock = MockMPITopologyBuilder()
        .addCPUOnlyRank(0, 0)
        .addCPUOnlyRank(1, 0)
        .addGPURank(2, 1, 0)
        .addGPURank(3, 1, 1)
        .setLocalRank(2)  // We are a GPU rank
        .build();
    
    EXPECT_EQ(mock->world_size(), 4);
    EXPECT_EQ(mock->node_count(), 2);
    EXPECT_TRUE(mock->has_accelerator());
    
    // Verify rank 0 is CPU-only
    const auto& p0 = mock->get_placement(0);
    bool rank0_has_gpu = false;
    for (const auto& dev : p0.devices) {
        if (dev.type != DeviceCapability::Type::CPU) {
            rank0_has_gpu = true;
        }
    }
    EXPECT_FALSE(rank0_has_gpu);
    
    // Verify rank 3 has GPU
    const auto& p3 = mock->get_placement(3);
    bool rank3_has_gpu = false;
    for (const auto& dev : p3.devices) {
        if (dev.type == DeviceCapability::Type::CUDA) {
            rank3_has_gpu = true;
        }
    }
    EXPECT_TRUE(rank3_has_gpu);
}

TEST(Test__MockMPITopology, Builder_ROCmGPU)
{
    auto mock = MockMPITopologyBuilder()
        .addROCmRank(0, 0, 0, 16.0f)
        .setLocalRank(0)
        .build();
    
    EXPECT_TRUE(mock->has_accelerator());
    
    const auto& devices = mock->get_devices();
    bool has_rocm = false;
    for (const auto& dev : devices) {
        if (dev.type == DeviceCapability::Type::ROCm) {
            has_rocm = true;
            EXPECT_EQ(dev.device_id, 0);
        }
    }
    EXPECT_TRUE(has_rocm);
}

TEST(Test__MockMPITopology, Builder_EmptyBuilder_ThrowsException)
{
    MockMPITopologyBuilder builder;
    EXPECT_THROW(builder.build(), std::invalid_argument);
}

TEST(Test__MockMPITopology, Builder_InvalidLocalRank_ThrowsException)
{
    MockMPITopologyBuilder builder;
    builder.addCPUOnlyRank(0, 0);
    builder.setLocalRank(5);  // Invalid
    EXPECT_THROW(builder.build(), std::invalid_argument);
}

// ============================================================================
// ClusterInventory Tests
// ============================================================================

TEST(Test__MockMPITopology, ClusterInventory_Basic)
{
    auto mock = MockMPITopology::createSimple(0, 2);
    const ClusterInventory& inv = mock->clusterInventory();
    
    EXPECT_EQ(inv.world_size, 2);
    EXPECT_EQ(inv.ranks.size(), 2);
    EXPECT_EQ(inv.ranks[0].rank, 0);
    EXPECT_EQ(inv.ranks[1].rank, 1);
}

TEST(Test__MockMPITopology, ClusterInventory_GPUAggregation)
{
    auto mock = MockMPITopologyBuilder()
        .addGPURank(0, 0, 0, 8.0f)
        .addGPURank(1, 0, 1, 16.0f)
        .setLocalRank(0)
        .build();
    
    const ClusterInventory& inv = mock->clusterInventory();
    
    EXPECT_TRUE(inv.hasAnyGPU());
    EXPECT_EQ(inv.total_gpus, 2);
    
    // Total GPU memory: 8GB + 16GB = 24GB
    size_t expected_memory = static_cast<size_t>(24) * 1024 * 1024 * 1024;
    EXPECT_EQ(inv.total_gpu_memory, expected_memory);
}

// ============================================================================
// Compute Weights Tests
// ============================================================================

TEST(Test__MockMPITopology, ComputeWeights_Homogeneous)
{
    auto mock = MockMPITopology::createSimple(0, 2);
    auto weights = mock->get_compute_weights();
    
    EXPECT_EQ(weights.size(), 2);
    EXPECT_FLOAT_EQ(weights[0], 1.0f);  // CPU-only
    EXPECT_FLOAT_EQ(weights[1], 1.0f);  // CPU-only
}

TEST(Test__MockMPITopology, ComputeWeights_Heterogeneous)
{
    auto mock = MockMPITopologyBuilder()
        .addCPUOnlyRank(0, 0)
        .addGPURank(1, 0, 0)  // GPU has 10.0f compute power by default
        .setLocalRank(0)
        .build();
    
    auto weights = mock->get_compute_weights();
    
    EXPECT_EQ(weights.size(), 2);
    EXPECT_FLOAT_EQ(weights[0], 1.0f);   // CPU-only
    EXPECT_FLOAT_EQ(weights[1], 11.0f);  // CPU (1.0) + GPU (10.0)
}

// ============================================================================
// SliceMetadata Tests
// ============================================================================

TEST(Test__MockMPITopology, SliceMetadata_RowParallel)
{
    auto mock = MockMPITopology::createSimple(0, 2);
    
    // 4096 rows, 896 cols, 2 ranks
    SliceMetadata meta = mock->createRowParallelMeta(4096, 896);
    
    EXPECT_EQ(meta.mode, SliceMode::ROW_PARALLEL);
    EXPECT_EQ(meta.original_rows, 4096);
    EXPECT_EQ(meta.original_cols, 896);
    EXPECT_EQ(meta.slice_start, 0);
    EXPECT_EQ(meta.slice_size(), 2048);  // Half the rows
    EXPECT_EQ(meta.world_size, 2);
    EXPECT_EQ(meta.rank, 0);
}

TEST(Test__MockMPITopology, SliceMetadata_ColumnParallel)
{
    auto rank1 = MockMPITopology::createSimple(1, 2);
    
    // 896 rows, 4096 cols, 2 ranks
    SliceMetadata meta = rank1->createColumnParallelMeta(896, 4096);
    
    EXPECT_EQ(meta.mode, SliceMode::COLUMN_PARALLEL);
    EXPECT_EQ(meta.original_rows, 896);
    EXPECT_EQ(meta.original_cols, 4096);
    EXPECT_EQ(meta.slice_start, 2048);  // Second half
    EXPECT_EQ(meta.slice_size(), 2048);
    EXPECT_EQ(meta.rank, 1);
}

// ============================================================================
// toString Tests
// ============================================================================

TEST(Test__MockMPITopology, ToString_ContainsBasicInfo)
{
    auto mock = MockMPITopology::createSimple(0, 2);
    std::string str = mock->to_string();
    
    EXPECT_NE(str.find("rank=0"), std::string::npos);
    EXPECT_NE(str.find("2"), std::string::npos);  // world_size
    EXPECT_NE(str.find("CPU"), std::string::npos);
}

// ============================================================================
// IMPITopology::createMock Static Factory Tests
// ============================================================================

TEST(Test__MockMPITopology, CreateMockViaInterface)
{
    std::vector<RankPlacement> placements(2);
    placements[0].rank = 0;
    placements[0].node_id = 0;
    placements[0].local_rank = 0;
    placements[0].devices.push_back(MockDevices::cpu());
    
    placements[1].rank = 1;
    placements[1].node_id = 0;
    placements[1].local_rank = 1;
    placements[1].devices.push_back(MockDevices::cpu());
    
    auto mock = IMPITopology::createMock(0, 2, placements);
    
    EXPECT_EQ(mock->rank(), 0);
    EXPECT_EQ(mock->world_size(), 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Test__MockMPITopology, GetPlacement_OutOfRange_ThrowsException)
{
    auto mock = MockMPITopology::createSimple(0, 2);
    
    EXPECT_THROW(mock->get_placement(-1), std::out_of_range);
    EXPECT_THROW(mock->get_placement(2), std::out_of_range);
}

TEST(Test__MockMPITopology, WorkRange_SingleElement)
{
    // Edge case: 1 head across 2 ranks
    auto rank0 = MockMPITopology::createSimple(0, 2);
    auto rank1 = MockMPITopology::createSimple(1, 2);
    
    WorkRange heads0 = rank0->get_head_range(1);
    WorkRange heads1 = rank1->get_head_range(1);
    
    // Only rank 0 gets the head
    EXPECT_EQ(heads0.size(), 1);
    EXPECT_EQ(heads1.size(), 0);
    EXPECT_TRUE(heads1.empty());
}

TEST(Test__MockMPITopology, WorkRange_ZeroElements)
{
    auto mock = MockMPITopology::createSimple(0, 2);
    WorkRange heads = mock->get_head_range(0);
    
    EXPECT_TRUE(heads.empty());
    EXPECT_EQ(heads.size(), 0);
}

TEST(Test__MockMPITopology, SetComputeParticipant)
{
    auto mock = MockMPITopology::createSimple(0, 2);
    
    EXPECT_TRUE(mock->is_compute_participant());
    
    mock->set_compute_participant(false);
    EXPECT_FALSE(mock->is_compute_participant());
    
    mock->set_compute_participant(true);
    EXPECT_TRUE(mock->is_compute_participant());
}
