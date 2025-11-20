/**
 * @file Test__PipelineBase_LogitsOverride.cpp
 * @brief Unit tests for PipelineBase logits() method override
 *
 * Tests that pipeline implementations properly override logits() to avoid
 * null pointer dereferences when logits_ is not set.
 *
 * Background:
 * - PipelineBase::logits() dereferences logits_ member (base class field)
 * - Qwen2Pipeline uses logits_buffer_ instead, never sets logits_
 * - If logits() not overridden, segfault when dereferencing nullptr
 *
 * This test catches the regression where a pipeline implementation fails to
 * override logits() properly, leading to runtime segfaults during decode.
 *
 * @author David Sanftenberg
 * @date 2025-11-20
 */

#include "../../../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../../../src/v2/loaders/ModelContext.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;

/**
 * @brief Test fixture for pipeline logits() override validation
 *
 * Tests that all pipeline implementations properly override logits()
 * to avoid null pointer dereferences.
 */
class PipelineBaseLogitsOverrideTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test-only model context (no actual file needed)
        model_ctx_ = ModelContext::createForTesting(
            "/tmp/test_pipeline_logits.gguf",
            nullptr, // No MPI context
            24       // block_count
        );
        ASSERT_NE(model_ctx_, nullptr);
    }

    std::shared_ptr<ModelContext> model_ctx_;
};

/**
 * @brief Test that Qwen2Pipeline properly overrides logits() method
 *
 * This is a regression test for the segfault bug where:
 * 1. Main.cpp calls pipeline->logits() (base class method)
 * 2. PipelineBase::logits() tries to dereference logits_ (nullptr)
 * 3. Qwen2Pipeline uses logits_buffer_ instead, never sets logits_
 * 4. Result: segfault
 *
 * Fix: Qwen2Pipeline must override logits() to call getLogits(0)
 *
 * This test verifies the override exists and behaves correctly without
 * requiring actual model weights or a full forward pass.
 */
TEST_F(PipelineBaseLogitsOverrideTest, Qwen2Pipeline_LogitsOverride_NoSegfault)
{
    // Create Qwen2Pipeline (no MPI for unit test)
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_,
        nullptr, // null MPI context (single-rank)
        -1,      // CPU device
        nullptr, // No weight placement
        PipelineConfig{},
        1 // batch_size=1
    );

    ASSERT_NE(pipeline, nullptr);

    // Test 1: Calling logits() before forward() should NOT segfault
    // If not properly overridden, PipelineBase::logits() would dereference nullptr
    const float *logits_before = nullptr;
    EXPECT_NO_THROW({
        logits_before = pipeline->logits();
    }) << "logits() should not throw/segfault even before forward() - must be properly overridden";

    // Test 2: Should return nullptr before forward() (graceful handling)
    EXPECT_EQ(logits_before, nullptr) << "logits() should return nullptr before forward() called";

    // Test 3: Verify logits() and getLogits(0) return same result (delegation)
    const float *logits_get_before = pipeline->getLogits(0);
    EXPECT_EQ(logits_before, logits_get_before) << "logits() should delegate to getLogits(0)";
}

/**
 * @brief Test that the logits() override pattern is consistent
 *
 * Verifies the override delegates properly without needing actual weights.
 */
TEST_F(PipelineBaseLogitsOverrideTest, Qwen2Pipeline_LogitsGetLogitsEquivalence)
{
    auto pipeline = std::make_unique<Qwen2Pipeline>(
        model_ctx_,
        nullptr,
        -1,
        nullptr,
        PipelineConfig{},
        1
    );

    ASSERT_NE(pipeline, nullptr);

    // Both methods should return same pointer (delegation)
    const float *logits_base = pipeline->logits();
    const float *logits_qwen = pipeline->getLogits(0);

    EXPECT_EQ(logits_base, logits_qwen)
        << "logits() should delegate to getLogits(0) - same pointer expected";

    // Both should be nullptr before forward()
    EXPECT_EQ(logits_base, nullptr);
    EXPECT_EQ(logits_qwen, nullptr);
}
