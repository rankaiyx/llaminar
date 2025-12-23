/**
 * @file Test__RegisterAllocation.cpp
 * @brief Unit tests for compile-time register allocation framework
 */

#include <gtest/gtest.h>
#include "kernels/cpu/jit/RegisterAllocation.h"

using namespace llaminar2::jit;

// ============================================================================
// Zone Definition Tests
// ============================================================================

TEST(Test__RegisterAllocation, ZoneRangesAreCorrect)
{
    // Accumulator zone: zmm0-zmm7
    EXPECT_EQ(AccumulatorZone::base_index, 0);
    EXPECT_EQ(AccumulatorZone::count, 8);

    // Q vector zone: zmm8-zmm15
    EXPECT_EQ(QVectorZone::base_index, 8);
    EXPECT_EQ(QVectorZone::count, 8);

    // State zone: zmm16-zmm19
    EXPECT_EQ(StateZone::base_index, 16);
    EXPECT_EQ(StateZone::count, 4);

    // Scratch zone: zmm20-zmm25
    EXPECT_EQ(ScratchZone::base_index, 20);
    EXPECT_EQ(ScratchZone::count, 6);

    // Score zone: xmm20-xmm23 (overlaps scratch)
    EXPECT_EQ(ScoreZone::base_index, 20);
    EXPECT_EQ(ScoreZone::count, 4);
}

TEST(Test__RegisterAllocation, ZoneOverlapDetection)
{
    // These should NOT overlap
    EXPECT_FALSE((zones_overlap_v<AccumulatorZone, QVectorZone>));
    EXPECT_FALSE((zones_overlap_v<QVectorZone, StateZone>));
    EXPECT_FALSE((zones_overlap_v<StateZone, ScratchZone>));
    EXPECT_FALSE((zones_overlap_v<AccumulatorZone, StateZone>));
    EXPECT_FALSE((zones_overlap_v<AccumulatorZone, ScratchZone>));

    // ScoreZone and ScratchZone DO overlap (by design - XMM/ZMM aliasing)
    EXPECT_TRUE((zones_overlap_v<ScoreZone, ScratchZone>));
}

// ============================================================================
// Typed Register Tests
// ============================================================================

TEST(Test__RegisterAllocation, TypedZmmAbsoluteIndices)
{
    // Accumulators: zmm0-zmm7
    EXPECT_EQ(Accum0::absolute_index, 0);
    EXPECT_EQ(Accum7::absolute_index, 7);

    // State registers: zmm16-zmm19
    EXPECT_EQ(StateMax::absolute_index, 16);
    EXPECT_EQ(StateSum::absolute_index, 17);
    EXPECT_EQ(StateWeight::absolute_index, 18);
    EXPECT_EQ(StateCorr::absolute_index, 19);

    // Scratch registers: zmm20-zmm25
    EXPECT_EQ(Scratch0::absolute_index, 20);
    EXPECT_EQ(Scratch4::absolute_index, 24);
    EXPECT_EQ(Scratch5::absolute_index, 25);

    // Score registers (XMM): xmm20-xmm23
    EXPECT_EQ(Score0::absolute_index, 20);
    EXPECT_EQ(Score3::absolute_index, 23);
}

TEST(Test__RegisterAllocation, TypedZmmConvertsToXbyak)
{
    // Verify that typed registers produce correct Xbyak registers
    Accum0 accum0;
    Xbyak::Zmm zmm0 = accum0.zmm();
    EXPECT_EQ(zmm0.getIdx(), 0);

    StateMax state_max;
    Xbyak::Zmm zmm16 = state_max.zmm();
    EXPECT_EQ(zmm16.getIdx(), 16);

    Scratch4 scratch4;
    Xbyak::Zmm zmm24 = scratch4.zmm();
    EXPECT_EQ(zmm24.getIdx(), 24);

    // Implicit conversion should also work
    Xbyak::Zmm implicit = Scratch5{};
    EXPECT_EQ(implicit.getIdx(), 25);
}

TEST(Test__RegisterAllocation, TypedXmmConvertsToXbyak)
{
    Score0 score0;
    Xbyak::Xmm xmm20 = score0.xmm();
    EXPECT_EQ(xmm20.getIdx(), 20);

    Score3 score3;
    Xbyak::Xmm xmm23 = score3.xmm();
    EXPECT_EQ(xmm23.getIdx(), 23);
}

TEST(Test__RegisterAllocation, XmmZmmAliasing)
{
    // Score0 (xmm20) aliases Scratch0 (zmm20)
    EXPECT_EQ(Score0::absolute_index, Scratch0::absolute_index);

    // Can get ZMM from XMM (but this is dangerous!)
    Score0 score0;
    Xbyak::Zmm aliased_zmm = score0.as_zmm();
    EXPECT_EQ(aliased_zmm.getIdx(), 20);
}

// ============================================================================
// Zone Type Trait Tests (C++17 compatible)
// ============================================================================

TEST(Test__RegisterAllocation, ZoneTypeTraitsWork)
{
    // Scratch registers should pass the scratch zone check
    EXPECT_TRUE((is_zone_v<Scratch0, ScratchZone>));
    EXPECT_TRUE((is_zone_v<Scratch4, ScratchZone>));

    // Non-scratch registers should fail
    EXPECT_FALSE((is_zone_v<Accum0, ScratchZone>));
    EXPECT_FALSE((is_zone_v<StateMax, ScratchZone>));

    // is_not_zone_v should be the inverse
    EXPECT_FALSE((is_not_zone_v<Scratch0, ScratchZone>));
    EXPECT_TRUE((is_not_zone_v<Accum0, ScratchZone>));

    // is_any_zone_v should work with multiple zones
    EXPECT_TRUE((is_any_zone_v<Scratch0, AccumulatorZone, ScratchZone>));
    EXPECT_TRUE((is_any_zone_v<Accum0, AccumulatorZone, ScratchZone>));
    EXPECT_FALSE((is_any_zone_v<StateMax, AccumulatorZone, ScratchZone>));
}

TEST(Test__RegisterAllocation, SafeScratchForFA2)
{
    // Scratch4 and Scratch5 are safe during FA2 (don't alias score registers)
    // local_index >= 4
    EXPECT_GE(Scratch4::local_index, 4);
    EXPECT_GE(Scratch5::local_index, 4);

    // Scratch0-3 are NOT safe (alias Score0-3)
    EXPECT_LT(Scratch0::local_index, 4);
    EXPECT_LT(Scratch1::local_index, 4);
    EXPECT_LT(Scratch2::local_index, 4);
    EXPECT_LT(Scratch3::local_index, 4);
}

// ============================================================================
// Register Set Conflict Detection
// ============================================================================

TEST(Test__RegisterAllocation, RegisterSetNoConflicts)
{
    // These should compile without conflict
    using SafeSet1 = UsesRegisters<Accum0, Accum1, StateMax>;
    EXPECT_EQ(SafeSet1::count, 3u);

    using SafeSet2 = UsesRegisters<Scratch4, Scratch5, StateCorr>;
    EXPECT_EQ(SafeSet2::count, 3u);
}

// ============================================================================
// Compile-Time Safety Demonstrations
// ============================================================================

// This test verifies that conflicting registers fail at compile time
// Uncomment to see the static_assert fire:
// TEST(Test__RegisterAllocation, RegisterSetDetectsConflicts) {
//     // This should NOT compile - Score0 and Scratch0 have same absolute index!
//     using ConflictSet = UsesRegisters<Score0, Scratch0>;
// }

// To verify the SFINAE constraint works, uncomment this - it should fail:
// template<typename Reg, require_safe_scratch_for_fa2<Reg> = true>
// void unsafe_test_fn(const Reg&) {}
// TEST(Test__RegisterAllocation, SFINAEBlocksUnsafeScratch) {
//     // This should NOT compile - Scratch0 aliases Score0 (local_index < 4)
//     unsafe_test_fn(Scratch0{});
// }

// ============================================================================
// Real-World Usage Patterns
// ============================================================================

TEST(Test__RegisterAllocation, FA2TileRegistersDocumented)
{
    // Document the FA2 tile register usage to catch future conflicts

    // During FA2 tile processing:
    // - Score0-Score3 (xmm20-23) hold the 4 dot product scores
    // - Scratch4 (zmm24) can be used for tile_max
    // - Scratch5 (zmm25) can be used for other scratch

    // This is the fix we made: tile_max uses Scratch4 (zmm24)
    // NOT Scratch0 (zmm20) which would clobber Score0

    EXPECT_NE(Scratch4::absolute_index, Score0::absolute_index);
    EXPECT_NE(Scratch4::absolute_index, Score1::absolute_index);
    EXPECT_NE(Scratch4::absolute_index, Score2::absolute_index);
    EXPECT_NE(Scratch4::absolute_index, Score3::absolute_index);

    // Scratch0-3 DO conflict with scores
    EXPECT_EQ(Scratch0::absolute_index, Score0::absolute_index);
    EXPECT_EQ(Scratch1::absolute_index, Score1::absolute_index);
    EXPECT_EQ(Scratch2::absolute_index, Score2::absolute_index);
    EXPECT_EQ(Scratch3::absolute_index, Score3::absolute_index);
}

// ============================================================================
// SFINAE-based constrained function example (C++17)
// ============================================================================

namespace example
{

    // Function that only accepts scratch registers using SFINAE
    template <typename Reg, require_zone<Reg, ScratchZone> = true>
    constexpr bool accepts_scratch(const Reg &)
    {
        return true;
    }

    // Function that only accepts safe scratch for FA2 (Scratch4, Scratch5)
    template <typename Reg, require_safe_scratch_for_fa2<Reg> = true>
    constexpr bool accepts_safe_fa2_scratch(const Reg &)
    {
        return true;
    }

} // namespace example

TEST(Test__RegisterAllocation, SFINAEConstrainedFunctions)
{
    // accepts_scratch works for any scratch register
    EXPECT_TRUE(example::accepts_scratch(Scratch0{}));
    EXPECT_TRUE(example::accepts_scratch(Scratch4{}));
    EXPECT_TRUE(example::accepts_scratch(Scratch5{}));

    // accepts_safe_fa2_scratch only works for Scratch4, Scratch5
    // (These would fail to compile for Scratch0-3)
    EXPECT_TRUE(example::accepts_safe_fa2_scratch(Scratch4{}));
    EXPECT_TRUE(example::accepts_safe_fa2_scratch(Scratch5{}));

    // These lines would NOT compile (uncomment to verify):
    // example::accepts_scratch(Accum0{});  // Not a scratch register
    // example::accepts_safe_fa2_scratch(Scratch0{});  // local_index < 4
}
