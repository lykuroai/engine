#include "core/generation/stop.h"

#include <gtest/gtest.h>

namespace lykuro::nie {
namespace {

TEST(StopValidationTest, EnforcesLimits) {
    StopLimits limits;
    EXPECT_TRUE(StopMatcher::Validate({"a", "bb"}, limits).ok());
    EXPECT_FALSE(StopMatcher::Validate({""}, limits).ok());
    EXPECT_FALSE(
        StopMatcher::Validate({std::string(100, 'x')}, limits).ok());
    std::vector<std::string> many(9, "s");
    EXPECT_FALSE(StopMatcher::Validate(many, limits).ok());
}

TEST(StopMatcherTest, StopsOnExactMatch) {
    StopMatcher m({"STOP"});
    EXPECT_EQ(m.Feed("hello STOP world"), "hello ");
    EXPECT_TRUE(m.stopped());
    EXPECT_EQ(m.Feed("more"), "");
    EXPECT_EQ(m.Flush(), "");
}

TEST(StopMatcherTest, StopSplitAcrossFeedsNeverLeaks) {
    StopMatcher m({"STOP"});
    std::string out;
    out += m.Feed("abc ST");
    EXPECT_EQ(out, "abc ");  // "ST" withheld as potential prefix
    out += m.Feed("OP xyz");
    EXPECT_TRUE(m.stopped());
    EXPECT_EQ(out, "abc ");
}

TEST(StopMatcherTest, FalsePrefixIsReleased) {
    StopMatcher m({"STOP"});
    std::string out;
    out += m.Feed("ST");
    out += m.Feed("ART");
    out += m.Flush();
    EXPECT_FALSE(m.stopped());
    EXPECT_EQ(out, "START");
}

TEST(StopMatcherTest, EarliestOfMultipleStopsWins) {
    StopMatcher m({"xx", "yy"});
    EXPECT_EQ(m.Feed("a yy b xx"), "a ");
    EXPECT_TRUE(m.stopped());
}

TEST(StopMatcherTest, NoStopsPassesThrough) {
    StopMatcher m({});
    EXPECT_EQ(m.Feed("all text "), "all text ");
    EXPECT_EQ(m.Flush(), "");
    EXPECT_FALSE(m.stopped());
}

}  // namespace
}  // namespace lykuro::nie
