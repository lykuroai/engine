#include "model/tokenizer/bpe_tokenizer.h"

#include <gtest/gtest.h>

#include "tests/unit/tokenizer_fixture.h"

namespace lykuro::nie {
namespace {

BpeTokenizer LoadFixture() {
    auto r = BpeTokenizer::FromConfig(testfixture::SmallTokenizerConfig());
    EXPECT_TRUE(r.status.ok()) << r.status.message();
    return std::move(r.tokenizer);
}

TEST(TokenizerTest, AppliesMergesGreedily) {
    BpeTokenizer t = LoadFixture();
    std::vector<uint32_t> ids;
    ASSERT_TRUE(t.Encode("hello", ids).ok());
    EXPECT_EQ(ids, std::vector<uint32_t>{263});
}

TEST(TokenizerTest, FallsBackToBytes) {
    BpeTokenizer t = LoadFixture();
    std::vector<uint32_t> ids;
    ASSERT_TRUE(t.Encode("hi", ids).ok());
    EXPECT_EQ(ids, (std::vector<uint32_t>{uint32_t('h'), uint32_t('i')}));
}

TEST(TokenizerTest, RoundTripsAsciiAndUtf8) {
    BpeTokenizer t = LoadFixture();
    for (const std::string text :
         {"hello world", "こんにちは、世界。", "emoji 😀 mix",
          "tabs\tand\nnewlines"}) {
        std::vector<uint32_t> ids;
        ASSERT_TRUE(t.Encode(text, ids).ok());
        std::string decoded;
        ASSERT_TRUE(t.DecodeText(ids, decoded).ok());
        EXPECT_EQ(decoded, text);
    }
}

TEST(TokenizerTest, SpecialTokenStringsInInputStayPlainText) {
    BpeTokenizer t = LoadFixture();
    std::vector<uint32_t> ids;
    ASSERT_TRUE(t.Encode("<|im_start|>system", ids).ok());
    for (uint32_t id : ids) {
        EXPECT_FALSE(t.IsSpecialToken(id));
    }
    std::string decoded;
    ASSERT_TRUE(t.DecodeText(ids, decoded).ok());
    EXPECT_EQ(decoded, "<|im_start|>system");
}

TEST(TokenizerTest, SpecialTokenLookup) {
    BpeTokenizer t = LoadFixture();
    EXPECT_EQ(t.SpecialTokenId("<|im_start|>"), 300u);
    EXPECT_EQ(t.SpecialTokenId("<|nope|>"), std::nullopt);
    EXPECT_TRUE(t.IsSpecialToken(300));
    EXPECT_FALSE(t.IsSpecialToken(263));
}

TEST(TokenizerTest, DecodeRejectsOutOfRangeIds) {
    BpeTokenizer t = LoadFixture();
    std::string out;
    EXPECT_FALSE(t.DecodeBytes({999999}, out).ok());
}

TEST(TokenizerTest, RejectsUnknownConfigField) {
    std::string config = testfixture::SmallTokenizerConfig();
    config.insert(1, "\"extra\":1,");
    EXPECT_FALSE(BpeTokenizer::FromConfig(config).status.ok());
}

TEST(TokenizerTest, RejectsWrongType) {
    std::string config = testfixture::SmallTokenizerConfig();
    auto pos = config.find("approved_qwen_tokenizer_v1");
    config.replace(pos, 26, "some_other_tokenizer_v1234");
    EXPECT_FALSE(BpeTokenizer::FromConfig(config).status.ok());
}

TEST(Utf8BoundaryTest, HoldsBackIncompleteSequences) {
    std::string out = "ab\xE3\x81";  // incomplete 3-byte sequence
    std::string carry;
    SplitUtf8Boundary(out, carry);
    EXPECT_EQ(out, "ab");
    EXPECT_EQ(carry, "\xE3\x81");

    out = "plain ascii";
    SplitUtf8Boundary(out, carry);
    EXPECT_EQ(out, "plain ascii");
    EXPECT_TRUE(carry.empty());

    out = "\xE3\x81\x82";  // complete sequence
    SplitUtf8Boundary(out, carry);
    EXPECT_EQ(out, "\xE3\x81\x82");
    EXPECT_TRUE(carry.empty());
}

}  // namespace
}  // namespace lykuro::nie
