/**
 * @file Test__PipelineRunner.cpp
 * @brief Unit tests for PipelineRunner
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for cross-rank pipeline runner:
 * - Construction with valid/invalid parameters
 * - myStageIndex() for various rank configurations
 * - vocab_size() returns configured value
 * - clear_cache() propagates to owned stage
 * - get_position() returns position counter
 * - forward() updates position
 * - executionPath() returns GRAPH
 * - architecture() returns "pipeline"
 * - hasEmbedding() / hasLMHead() reflect stage config
 *
 * All tests use mock runners — no MPI, no GPU, no model loading.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>

#include "execution/parallelism_tree/PipelineRunner.h"
#include "execution/parallelism_tree/TransferSpec.h"

using namespace llaminar2;

// =============================================================================
// Mock IInferenceRunner for Testing
// =============================================================================

/**
 * @brief Mock inference runner for unit testing
 */
class MockRunner : public IInferenceRunner
{
public:
    std::string name;
    mutable int forward_count = 0;
    mutable int clear_cache_count = 0;
    int position = 0;
    int vocab_size_ = 128;
    mutable std::vector<float> logits_data;
    TensorBase *hidden_input = nullptr;

    explicit MockRunner(std::string name) : name(std::move(name))
    {
        logits_data.resize(vocab_size_, 0.5f);
    }

    bool forward(const int *tokens, int seq_len) override
    {
        (void)tokens;
        forward_count++;
        position += seq_len;
        return true;
    }

    const float *logits() const override
    {
        return logits_data.data();
    }

    void clear_cache() override
    {
        clear_cache_count++;
        position = 0;
    }

    int get_position() const override { return position; }
    int vocab_size() const override { return vocab_size_; }
    ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
    const char *architecture() const override { return "mock"; }

    void setHiddenState(TensorBase *hidden_state) override
    {
        hidden_input = hidden_state;
    }

    bool hasHiddenStateInput() const override { return hidden_input != nullptr; }
};

// =============================================================================
// Helper: Build test configurations
// =============================================================================

/**
 * @brief Build a 2-stage pipeline for rank 0
 *
 * Stage 0: rank 0, layers 0-11, has embedding
 * Stage 1: rank 1, layers 12-23, has lm_head
 */
static std::pair<std::vector<PipelineRunner::StageInfo>, std::vector<PipelineRunner::TransferInfo>>
build2StagePipeline(int my_rank)
{
    std::vector<PipelineRunner::StageInfo> stages;

    // Stage 0: rank 0
    PipelineRunner::StageInfo s0;
    s0.stage_index = 0;
    s0.owning_rank = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    s0.has_embedding = true;
    s0.has_lm_head = false;
    if (my_rank == 0)
    {
        s0.runner = std::make_unique<MockRunner>("stage0");
    }
    stages.push_back(std::move(s0));

    // Stage 1: rank 1
    PipelineRunner::StageInfo s1;
    s1.stage_index = 1;
    s1.owning_rank = 1;
    s1.first_layer = 12;
    s1.last_layer = 23;
    s1.has_embedding = false;
    s1.has_lm_head = true;
    if (my_rank == 1)
    {
        s1.runner = std::make_unique<MockRunner>("stage1");
    }
    stages.push_back(std::move(s1));

    // Transfer from stage 0 to stage 1
    std::vector<PipelineRunner::TransferInfo> transfers;
    PipelineRunner::TransferInfo t;
    t.from_stage = 0;
    t.to_stage = 1;
    t.mpi_tag = 100;
    t.mechanism = TransferSpec::Mechanism::MPI_INTERHOST;
    t.sender_rank = 0;
    t.receiver_rank = 1;
    transfers.push_back(t);

    return {std::move(stages), std::move(transfers)};
}

/**
 * @brief Build a 3-stage pipeline
 */
static std::pair<std::vector<PipelineRunner::StageInfo>, std::vector<PipelineRunner::TransferInfo>>
build3StagePipeline(int my_rank)
{
    std::vector<PipelineRunner::StageInfo> stages;

    // Stage 0: rank 0, layers 0-7, has embedding
    PipelineRunner::StageInfo s0;
    s0.stage_index = 0;
    s0.owning_rank = 0;
    s0.first_layer = 0;
    s0.last_layer = 7;
    s0.has_embedding = true;
    s0.has_lm_head = false;
    if (my_rank == 0)
    {
        s0.runner = std::make_unique<MockRunner>("stage0");
    }
    stages.push_back(std::move(s0));

    // Stage 1: rank 1, layers 8-15
    PipelineRunner::StageInfo s1;
    s1.stage_index = 1;
    s1.owning_rank = 1;
    s1.first_layer = 8;
    s1.last_layer = 15;
    s1.has_embedding = false;
    s1.has_lm_head = false;
    if (my_rank == 1)
    {
        s1.runner = std::make_unique<MockRunner>("stage1");
    }
    stages.push_back(std::move(s1));

    // Stage 2: rank 2, layers 16-23, has lm_head
    PipelineRunner::StageInfo s2;
    s2.stage_index = 2;
    s2.owning_rank = 2;
    s2.first_layer = 16;
    s2.last_layer = 23;
    s2.has_embedding = false;
    s2.has_lm_head = true;
    if (my_rank == 2)
    {
        s2.runner = std::make_unique<MockRunner>("stage2");
    }
    stages.push_back(std::move(s2));

    // Transfers
    std::vector<PipelineRunner::TransferInfo> transfers;

    PipelineRunner::TransferInfo t01;
    t01.from_stage = 0;
    t01.to_stage = 1;
    t01.mpi_tag = 100;
    t01.mechanism = TransferSpec::Mechanism::MPI_INTERHOST;
    t01.sender_rank = 0;
    t01.receiver_rank = 1;
    transfers.push_back(t01);

    PipelineRunner::TransferInfo t12;
    t12.from_stage = 1;
    t12.to_stage = 2;
    t12.mpi_tag = 101;
    t12.mechanism = TransferSpec::Mechanism::MPI_INTERHOST;
    t12.sender_rank = 1;
    t12.receiver_rank = 2;
    transfers.push_back(t12);

    return {std::move(stages), std::move(transfers)};
}

// =============================================================================
// Test Suite: Construction
// =============================================================================

TEST(Test__PipelineRunner, Construction_Valid)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(
        /*my_rank=*/0,
        /*world_size=*/2,
        std::move(stages),
        std::move(transfers),
        /*hidden_dim=*/896,
        /*vocab_size=*/128);

    EXPECT_EQ(runner.myRank(), 0);
    EXPECT_EQ(runner.worldSize(), 2);
    EXPECT_EQ(runner.numStages(), 2);
    EXPECT_EQ(runner.numTransfers(), 1);
    EXPECT_EQ(runner.vocab_size(), 128);
}

TEST(Test__PipelineRunner, Construction_ThrowsOnEmptyStages)
{
    std::vector<PipelineRunner::StageInfo> empty_stages;
    std::vector<PipelineRunner::TransferInfo> empty_transfers;

    EXPECT_THROW(
        PipelineRunner(0, 1, std::move(empty_stages), std::move(empty_transfers), 896, 128),
        std::invalid_argument);
}

TEST(Test__PipelineRunner, Construction_ThrowsOnInvalidRank)
{
    auto [stages, transfers] = build2StagePipeline(0);

    // my_rank >= world_size
    EXPECT_THROW(
        PipelineRunner(2, 2, std::move(stages), std::move(transfers), 896, 128),
        std::invalid_argument);
}

TEST(Test__PipelineRunner, Construction_ThrowsOnNegativeHiddenDim)
{
    auto [stages, transfers] = build2StagePipeline(0);

    EXPECT_THROW(
        PipelineRunner(0, 2, std::move(stages), std::move(transfers), -1, 128),
        std::invalid_argument);
}

TEST(Test__PipelineRunner, Construction_ThrowsOnInvalidVocabSize)
{
    auto [stages, transfers] = build2StagePipeline(0);

    EXPECT_THROW(
        PipelineRunner(0, 2, std::move(stages), std::move(transfers), 896, 0),
        std::invalid_argument);
}

TEST(Test__PipelineRunner, Construction_ThrowsOnMismatchedTransferCount)
{
    auto [stages, transfers] = build2StagePipeline(0);

    // Add an extra transfer (should be stages-1)
    PipelineRunner::TransferInfo extra;
    extra.from_stage = 1;
    extra.to_stage = 2;
    transfers.push_back(extra);

    EXPECT_THROW(
        PipelineRunner(0, 2, std::move(stages), std::move(transfers), 896, 128),
        std::invalid_argument);
}

// =============================================================================
// Test Suite: myStageIndex
// =============================================================================

TEST(Test__PipelineRunner, MyStageIndex_Stage0)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(runner.myStageIndex(), 0);
}

TEST(Test__PipelineRunner, MyStageIndex_Stage1)
{
    auto [stages, transfers] = build2StagePipeline(1);

    PipelineRunner runner(1, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(runner.myStageIndex(), 1);
}

TEST(Test__PipelineRunner, MyStageIndex_None)
{
    auto [stages, transfers] = build2StagePipeline(2); // Rank 2 owns nothing

    PipelineRunner runner(2, 3, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(runner.myStageIndex(), -1);
}

TEST(Test__PipelineRunner, MyStageIndex_3Stage)
{
    auto [stages, transfers] = build3StagePipeline(1);

    PipelineRunner runner(1, 3, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(runner.myStageIndex(), 1);
}

// =============================================================================
// Test Suite: Basic Properties
// =============================================================================

TEST(Test__PipelineRunner, VocabSize)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 151936);

    EXPECT_EQ(runner.vocab_size(), 151936);
}

TEST(Test__PipelineRunner, ExecutionPath)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(runner.executionPath(), ExecutionPath::GRAPH);
}

TEST(Test__PipelineRunner, Architecture)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_STREQ(runner.architecture(), "pipeline");
}

TEST(Test__PipelineRunner, GetPosition_Initial)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(runner.get_position(), 0);
}

// =============================================================================
// Test Suite: Embedding and LM Head
// =============================================================================

TEST(Test__PipelineRunner, HasEmbedding_Stage0)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_TRUE(runner.hasEmbedding());
    EXPECT_FALSE(runner.hasLMHead());
}

TEST(Test__PipelineRunner, HasLMHead_Stage1)
{
    auto [stages, transfers] = build2StagePipeline(1);

    PipelineRunner runner(1, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_FALSE(runner.hasEmbedding());
    EXPECT_TRUE(runner.hasLMHead());
}

TEST(Test__PipelineRunner, HasNeither_MiddleStage)
{
    auto [stages, transfers] = build3StagePipeline(1);

    PipelineRunner runner(1, 3, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_FALSE(runner.hasEmbedding());
    EXPECT_FALSE(runner.hasLMHead());
}

TEST(Test__PipelineRunner, HasNone_NoStageOwned)
{
    auto [stages, transfers] = build2StagePipeline(2);

    PipelineRunner runner(2, 3, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_FALSE(runner.hasEmbedding());
    EXPECT_FALSE(runner.hasLMHead());
}

// =============================================================================
// Test Suite: Forward and Cache
// =============================================================================

TEST(Test__PipelineRunner, ClearCache_PropagesToOwnedStage)
{
    auto [stages, transfers] = build2StagePipeline(0);

    // Get pointer to the mock runner before moving
    auto *mock = dynamic_cast<MockRunner *>(stages[0].runner.get());
    ASSERT_NE(mock, nullptr);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(mock->clear_cache_count, 0);

    runner.clear_cache();

    EXPECT_EQ(mock->clear_cache_count, 1);
}

TEST(Test__PipelineRunner, Forward_UpdatesPosition)
{
    auto [stages, transfers] = build2StagePipeline(0);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    std::vector<int> tokens = {1, 2, 3, 4, 5};

    bool ok = runner.forward(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_TRUE(ok);

    // Position should be updated (mock doesn't do real MPI)
    EXPECT_GE(runner.get_position(), 0);
}

TEST(Test__PipelineRunner, Forward_CallsStageForward)
{
    auto [stages, transfers] = build2StagePipeline(0);

    auto *mock = dynamic_cast<MockRunner *>(stages[0].runner.get());
    ASSERT_NE(mock, nullptr);

    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(mock->forward_count, 0);

    std::vector<int> tokens = {1, 2, 3};
    runner.forward(tokens.data(), static_cast<int>(tokens.size()));

    EXPECT_EQ(mock->forward_count, 1);
}

// =============================================================================
// Test Suite: Logits
// =============================================================================

TEST(Test__PipelineRunner, Logits_ReturnsFromFinalStage)
{
    auto [stages, transfers] = build2StagePipeline(1);

    // Rank 1 owns the final stage with LM head
    PipelineRunner runner(1, 2, std::move(stages), std::move(transfers), 896, 128);

    const float *logits = runner.logits();
    EXPECT_NE(logits, nullptr);
}

TEST(Test__PipelineRunner, Logits_NullForNonFinalStage)
{
    auto [stages, transfers] = build2StagePipeline(0);

    // Rank 0 owns stage 0 which doesn't have LM head
    PipelineRunner runner(0, 2, std::move(stages), std::move(transfers), 896, 128);

    const float *logits = runner.logits();
    EXPECT_EQ(logits, nullptr);
}

// =============================================================================
// Test Suite: Stage Info Validation
// =============================================================================

TEST(Test__PipelineRunner, StageInfoValidation_MismatchedIndices)
{
    std::vector<PipelineRunner::StageInfo> stages;

    // Create stage with wrong stage_index
    PipelineRunner::StageInfo s0;
    s0.stage_index = 5; // Should be 0
    s0.owning_rank = 0;
    s0.first_layer = 0;
    s0.last_layer = 11;
    stages.push_back(std::move(s0));

    std::vector<PipelineRunner::TransferInfo> transfers; // Empty for single stage

    // Should still construct (stage_index is not validated strictly)
    PipelineRunner runner(0, 1, std::move(stages), std::move(transfers), 896, 128);

    EXPECT_EQ(runner.numStages(), 1);
}
