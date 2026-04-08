/**
 * @file Test__GlobalPPTransferStage.cpp
 * @brief Unit tests for GlobalPPTransferStage
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for Global PP transfer stage (cross-rank MPI send/recv):
 * - Stage construction and properties
 * - Stage type and name generation
 * - Buffer requirements (SEND vs RECV asymmetry)
 * - Parameter validation (null tensor, null mpi_ctx, invalid peer_rank)
 * - Coherence policy (NONE)
 * - Dump info (direction, peer_rank, tag metadata)
 *
 * Note: Actual MPI send/recv requires integration tests with MPI_PROCS >= 2.
 * These unit tests validate the stage interface without MPI communication.
 */

#include <gtest/gtest.h>
#include <vector>

#include "execution/compute_stages/stages/GlobalPPTransferStage.h"
#include "execution/debug/BufferRole.h"
#include "tensors/TensorFactory.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__GlobalPPTransferStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create MPI context (rank 0, world_size 2, no real comm needed for unit tests)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);

        // Create test tensor (typical hidden state: batch=2, dim=896)
        TensorFactory factory(*mpi_ctx_);
        test_tensor_ = factory.createFP32({2, 896}, DeviceId::cpu());

        // Fill with known values for dump info validation
        for (size_t i = 0; i < test_tensor_->numel(); ++i)
        {
            test_tensor_->mutable_data()[i] = static_cast<float>(i) * 0.01f;
        }
    }

    std::shared_ptr<IMPIContext> mpi_ctx_;
    std::unique_ptr<FP32Tensor> test_tensor_;
};

// =============================================================================
// Stage Type and Properties Tests
// =============================================================================

/**
 * @test Stage type is GLOBAL_PP_TRANSFER
 */
TEST_F(Test__GlobalPPTransferStage, StageType)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;
    params.tag = 100;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_EQ(stage->type(), ComputeStageType::GLOBAL_PP_TRANSFER);
}

/**
 * @test Default send name includes peer rank and tag
 */
TEST_F(Test__GlobalPPTransferStage, DefaultSendName)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;
    params.tag = 42;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_EQ(stage->name(), "gpp_send_to_rank1_tag42");
}

/**
 * @test Default recv name includes peer rank and tag
 */
TEST_F(Test__GlobalPPTransferStage, DefaultRecvName)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::RECV;
    params.tensor = test_tensor_.get();
    params.peer_rank = 0;
    params.tag = 99;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_EQ(stage->name(), "gpp_recv_from_rank0_tag99");
}

/**
 * @test Custom name overrides default
 */
TEST_F(Test__GlobalPPTransferStage, CustomName)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;
    params.tag = 100;
    params.stage_name = "my_custom_transfer";

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_EQ(stage->name(), "my_custom_transfer");
}

/**
 * @test Coherence policy is NONE
 */
TEST_F(Test__GlobalPPTransferStage, CoherencePolicyNone)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_EQ(stage->coherencePolicy(), CoherencePolicy::NONE);
}

/**
 * @test Supports all backends (MPI handles communication internally)
 */
TEST_F(Test__GlobalPPTransferStage, SupportsAllBackends)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_TRUE(stage->supportsBackend(ComputeBackendType::CPU));
}

// =============================================================================
// Accessor Tests
// =============================================================================

/**
 * @test Accessors return correct values
 */
TEST_F(Test__GlobalPPTransferStage, Accessors)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::RECV;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;
    params.tag = 55;
    params.count = 100;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_EQ(stage->getDirection(), GlobalPPTransferParams::Direction::RECV);
    EXPECT_EQ(stage->getTensor(), test_tensor_.get());
    EXPECT_EQ(stage->getPeerRank(), 1);
    EXPECT_EQ(stage->getTag(), 55);
    EXPECT_EQ(stage->getCount(), 100u);
}

// =============================================================================
// Buffer Requirements Tests
// =============================================================================

/**
 * @test SEND direction: tensor is input
 */
TEST_F(Test__GlobalPPTransferStage, BufferRequirementsSend)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);
    auto reqs = stage->getBufferRequirements();

    EXPECT_FALSE(reqs.getByRole(BufferRole::INPUT).empty());
    EXPECT_TRUE(reqs.getByRole(BufferRole::OUTPUT).empty());
}

/**
 * @test RECV direction: tensor is output
 */
TEST_F(Test__GlobalPPTransferStage, BufferRequirementsRecv)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::RECV;
    params.tensor = test_tensor_.get();
    params.peer_rank = 0;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);
    auto reqs = stage->getBufferRequirements();

    EXPECT_TRUE(reqs.getByRole(BufferRole::INPUT).empty());
    EXPECT_FALSE(reqs.getByRole(BufferRole::OUTPUT).empty());
}

// =============================================================================
// Dump Info Tests
// =============================================================================

/**
 * @test Dump info includes transfer metadata
 */
TEST_F(Test__GlobalPPTransferStage, DumpInfoContainsMetadata)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;
    params.tag = 42;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);
    auto info = stage->buildDumpInfoImpl();

    // Should have scalars for direction, peer_rank, tag, count, my_rank, world_size
    EXPECT_GE(info.scalars.size(), 4u);

    // Should have input (for SEND)
    EXPECT_FALSE(info.inputs.empty());
    EXPECT_TRUE(info.outputs.empty());
}

/**
 * @test Dump info for RECV has output, not input
 */
TEST_F(Test__GlobalPPTransferStage, DumpInfoRecvHasOutput)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::RECV;
    params.tensor = test_tensor_.get();
    params.peer_rank = 0;
    params.tag = 99;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);
    auto info = stage->buildDumpInfoImpl();

    EXPECT_TRUE(info.inputs.empty());
    EXPECT_FALSE(info.outputs.empty());
}

// =============================================================================
// Parameter Update Tests
// =============================================================================

/**
 * @test setParams updates direction and name
 */
TEST_F(Test__GlobalPPTransferStage, SetParamsUpdatesName)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;
    params.tag = 10;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);
    EXPECT_EQ(stage->name(), "gpp_send_to_rank1_tag10");

    // Update to RECV
    GlobalPPTransferParams new_params = params;
    new_params.direction = GlobalPPTransferParams::Direction::RECV;
    new_params.peer_rank = 0;
    new_params.tag = 20;
    stage->setParams(new_params);

    EXPECT_EQ(stage->name(), "gpp_recv_from_rank0_tag20");
    EXPECT_EQ(stage->getDirection(), GlobalPPTransferParams::Direction::RECV);
}

// =============================================================================
// Parameter Validation Tests (execute() returns false for bad params)
// =============================================================================

/**
 * @test Execute fails with null tensor
 */
TEST_F(Test__GlobalPPTransferStage, ExecuteFailsWithNullTensor)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = nullptr;
    params.peer_rank = 1;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Execute fails with null MPI context
 */
TEST_F(Test__GlobalPPTransferStage, ExecuteFailsWithNullMPIContext)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = nullptr;
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 1;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Execute fails with invalid peer rank (-1)
 */
TEST_F(Test__GlobalPPTransferStage, ExecuteFailsWithInvalidPeerRank)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get();
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = -1;

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

/**
 * @test Execute fails when peer_rank equals own rank
 */
TEST_F(Test__GlobalPPTransferStage, ExecuteFailsWhenSendingToSelf)
{
    GlobalPPTransferParams params;
    params.mpi_ctx = mpi_ctx_.get(); // rank=0
    params.direction = GlobalPPTransferParams::Direction::SEND;
    params.tensor = test_tensor_.get();
    params.peer_rank = 0; // same as own rank

    auto stage = std::make_unique<GlobalPPTransferStage>(params);

    EXPECT_FALSE(stage->execute(nullptr));
}

// =============================================================================
// ComputeStageType Name Tests
// =============================================================================

/**
 * @test computeStageTypeName returns correct string
 */
TEST_F(Test__GlobalPPTransferStage, StageTypeNameString)
{
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::GLOBAL_PP_TRANSFER), "GLOBAL_PP_TRANSFER");
}
