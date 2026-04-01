/**
 * @file Test__CoherenceAuditLog.cpp
 * @brief Unit tests for CoherenceAuditLog ring buffer
 *
 * Tests:
 * - Recording and counting entries
 * - Ring buffer wrap-around
 * - hasTransition() query
 * - dump() formatting (does not crash, covers all branches)
 */

#include <gtest/gtest.h>
#include "v2/tensors/CoherenceAuditLog.h"
#include <sstream>

using namespace llaminar2;

TEST(Test__CoherenceAuditLog, InitiallyEmpty)
{
    CoherenceAuditLog log;
    EXPECT_EQ(log.count(), 0u);
}

TEST(Test__CoherenceAuditLog, RecordSingleEntry)
{
    CoherenceAuditLog log;
    log.record(TensorCoherenceState::HOST_ONLY,
               TensorCoherenceState::SYNCED,
               CoherenceOp::UPLOAD,
               "test_func",
               true);
    EXPECT_EQ(log.count(), 1u);
}

TEST(Test__CoherenceAuditLog, RecordMultipleEntries)
{
    CoherenceAuditLog log;
    log.record(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::SYNCED,
               CoherenceOp::UPLOAD, "upload", true);
    log.record(TensorCoherenceState::SYNCED, TensorCoherenceState::DEVICE_AUTHORITATIVE,
               CoherenceOp::MARK_DEVICE_DIRTY, "mark_dirty", true);
    log.record(TensorCoherenceState::DEVICE_AUTHORITATIVE, TensorCoherenceState::SYNCED,
               CoherenceOp::DOWNLOAD, "download", true);
    EXPECT_EQ(log.count(), 3u);
}

TEST(Test__CoherenceAuditLog, RingBufferWrapAround)
{
    CoherenceAuditLog log;
    // Fill beyond ring size (32)
    for (int i = 0; i < 50; ++i)
    {
        log.record(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::SYNCED,
                   CoherenceOp::UPLOAD, "loop", true);
    }
    // Count should saturate at RING_SIZE
    EXPECT_EQ(log.count(), 32u);
}

TEST(Test__CoherenceAuditLog, HasTransition_Found)
{
    CoherenceAuditLog log;
    log.record(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::SYNCED,
               CoherenceOp::UPLOAD, "upload", true);
    log.record(TensorCoherenceState::SYNCED, TensorCoherenceState::DEVICE_AUTHORITATIVE,
               CoherenceOp::MARK_DEVICE_DIRTY, "mark_dirty", true);

    EXPECT_TRUE(log.hasTransition(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::SYNCED));
    EXPECT_TRUE(log.hasTransition(TensorCoherenceState::SYNCED, TensorCoherenceState::DEVICE_AUTHORITATIVE));
}

TEST(Test__CoherenceAuditLog, HasTransition_NotFound)
{
    CoherenceAuditLog log;
    log.record(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::SYNCED,
               CoherenceOp::UPLOAD, "upload", true);

    EXPECT_FALSE(log.hasTransition(TensorCoherenceState::SYNCED, TensorCoherenceState::HOST_ONLY));
    EXPECT_FALSE(log.hasTransition(TensorCoherenceState::DEVICE_AUTHORITATIVE, TensorCoherenceState::SYNCED));
}

TEST(Test__CoherenceAuditLog, HasTransition_EmptyLog)
{
    CoherenceAuditLog log;
    EXPECT_FALSE(log.hasTransition(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::SYNCED));
}

TEST(Test__CoherenceAuditLog, DumpDoesNotCrash)
{
    CoherenceAuditLog log;
    log.record(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::SYNCED,
               CoherenceOp::UPLOAD, "upload", true);
    log.record(TensorCoherenceState::SYNCED, TensorCoherenceState::INVALID,
               CoherenceOp::MARK_DEVICE_DIRTY, "bad_op", false);

    std::ostringstream oss;
    log.dump(oss, "test_tensor", reinterpret_cast<const void *>(0xDEAD));

    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("test_tensor"), std::string::npos);
    EXPECT_NE(output.find("UPLOAD"), std::string::npos);
}

TEST(Test__CoherenceAuditLog, DumpOnEmptyLog)
{
    CoherenceAuditLog log;
    std::ostringstream oss;
    log.dump(oss, "empty", nullptr);

    std::string output = oss.str();
    EXPECT_NE(output.find("empty"), std::string::npos);
}

TEST(Test__CoherenceAuditLog, InvalidTransitionRecorded)
{
    CoherenceAuditLog log;
    log.record(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::INVALID,
               CoherenceOp::MARK_DEVICE_DIRTY, "bad_op", false);

    EXPECT_EQ(log.count(), 1u);
    EXPECT_TRUE(log.hasTransition(TensorCoherenceState::HOST_ONLY, TensorCoherenceState::INVALID));
}
