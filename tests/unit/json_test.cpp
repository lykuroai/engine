#include "core/engine/json.h"

#include <gtest/gtest.h>

namespace lykuro::nie::json {
namespace {

TEST(JsonTest, ParsesScalars) {
    EXPECT_TRUE(Parse("null").value->is_null());
    EXPECT_EQ(Parse("true").value->as_bool(), true);
    EXPECT_EQ(Parse("false").value->as_bool(), false);
    EXPECT_EQ(Parse("42").value->as_int(), 42);
    EXPECT_EQ(Parse("-7").value->as_int(), -7);
    EXPECT_DOUBLE_EQ(Parse("1.5").value->as_double(), 1.5);
    EXPECT_DOUBLE_EQ(Parse("2e3").value->as_double(), 2000.0);
    EXPECT_EQ(Parse("\"abc\"").value->as_string(), "abc");
}

TEST(JsonTest, ParsesNested) {
    auto r = Parse(R"({"a": [1, {"b": "x"}], "c": null})");
    ASSERT_TRUE(r.ok());
    const Value* a = r.value->Find("a");
    ASSERT_NE(a, nullptr);
    ASSERT_TRUE(a->is_array());
    EXPECT_EQ(a->as_array()[0]->as_int(), 1);
    EXPECT_EQ(a->as_array()[1]->Find("b")->as_string(), "x");
}

TEST(JsonTest, RejectsDuplicateKeys) {
    EXPECT_FALSE(Parse(R"({"a": 1, "a": 2})").ok());
}

TEST(JsonTest, RejectsTrailingData) {
    EXPECT_FALSE(Parse("1 2").ok());
    EXPECT_FALSE(Parse("{} x").ok());
}

TEST(JsonTest, RejectsMalformed) {
    EXPECT_FALSE(Parse("").ok());
    EXPECT_FALSE(Parse("{").ok());
    EXPECT_FALSE(Parse("[1,]").ok());
    EXPECT_FALSE(Parse("{\"a\":}").ok());
    EXPECT_FALSE(Parse("01").ok());
    EXPECT_FALSE(Parse("+1").ok());
    EXPECT_FALSE(Parse("nul").ok());
    EXPECT_FALSE(Parse("\"unterminated").ok());
}

TEST(JsonTest, EnforcesDepthLimit) {
    std::string deep;
    for (int i = 0; i < 100; ++i) deep += "[";
    for (int i = 0; i < 100; ++i) deep += "]";
    EXPECT_FALSE(Parse(deep, /*max_depth=*/64).ok());
    EXPECT_TRUE(Parse(deep, /*max_depth=*/128).ok());
}

TEST(JsonTest, HandlesEscapes) {
    auto r = Parse(R"("a\n\t\"\\\u0041\u3042")");
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value->as_string(), "a\n\t\"\\A\xE3\x81\x82");
}

TEST(JsonTest, HandlesSurrogatePairs) {
    auto r = Parse(R"("\uD83D\uDE00")");
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value->as_string(), "\xF0\x9F\x98\x80");
    EXPECT_FALSE(Parse(R"("\uD83D")").ok());
    EXPECT_FALSE(Parse(R"("\uDE00")").ok());
}

TEST(JsonTest, RejectsInvalidUtf8) {
    std::string bad = "\"\xC3\x28\"";  // invalid continuation
    EXPECT_FALSE(Parse(bad).ok());
    std::string overlong = "\"\xC0\xAF\"";  // overlong '/'
    EXPECT_FALSE(Parse(overlong).ok());
}

TEST(JsonTest, RejectsControlCharacters) {
    std::string ctrl = "\"a\x01b\"";
    EXPECT_FALSE(Parse(ctrl).ok());
}

TEST(JsonTest, BigIntegersFallBackToDouble) {
    auto r = Parse("123456789012345678901234567890");
    ASSERT_TRUE(r.ok());
    EXPECT_TRUE(r.value->is_number());
    EXPECT_FALSE(r.value->is_int());
}

}  // namespace
}  // namespace lykuro::nie::json
