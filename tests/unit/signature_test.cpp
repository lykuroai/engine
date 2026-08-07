#include "security/signature.h"

#include <gtest/gtest.h>

#include <cstring>

namespace lykuro::nie {
namespace {

class SignatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(GenerateKeypair(pub_, priv_).ok());
        trusted_.keys.push_back(pub_);
    }

    std::array<uint8_t, 32> pub_;
    std::array<uint8_t, 64> priv_;
    TrustedKeys trusted_;
};

TEST_F(SignatureTest, SignAndVerifyRoundTrip) {
    const std::string message = "{\"schema_version\":\"1\"}";
    std::array<uint8_t, 64> sig;
    ASSERT_TRUE(SignMessage(message, priv_, sig).ok());
    std::string sig_hex = BytesToHex(sig.data(), sig.size());
    EXPECT_TRUE(VerifyManifestSignature(message, sig_hex, trusted_).ok());
}

TEST_F(SignatureTest, RejectsTamperedMessage) {
    const std::string message = "payload";
    std::array<uint8_t, 64> sig;
    ASSERT_TRUE(SignMessage(message, priv_, sig).ok());
    std::string sig_hex = BytesToHex(sig.data(), sig.size());
    Status s = VerifyManifestSignature("payloae", sig_hex, trusted_);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kArtifactVerificationFailed);
}

TEST_F(SignatureTest, RejectsWrongKey) {
    std::array<uint8_t, 32> other_pub;
    std::array<uint8_t, 64> other_priv;
    ASSERT_TRUE(GenerateKeypair(other_pub, other_priv).ok());
    const std::string message = "payload";
    std::array<uint8_t, 64> sig;
    ASSERT_TRUE(SignMessage(message, other_priv, sig).ok());
    std::string sig_hex = BytesToHex(sig.data(), sig.size());
    EXPECT_FALSE(VerifyManifestSignature(message, sig_hex, trusted_).ok());
}

TEST_F(SignatureTest, AcceptsAnyTrustedKey) {
    std::array<uint8_t, 32> second_pub;
    std::array<uint8_t, 64> second_priv;
    ASSERT_TRUE(GenerateKeypair(second_pub, second_priv).ok());
    trusted_.keys.push_back(second_pub);

    const std::string message = "payload";
    std::array<uint8_t, 64> sig;
    ASSERT_TRUE(SignMessage(message, second_priv, sig).ok());
    std::string sig_hex = BytesToHex(sig.data(), sig.size());
    EXPECT_TRUE(VerifyManifestSignature(message, sig_hex, trusted_).ok());
}

TEST_F(SignatureTest, RejectsMalformedInputs) {
    EXPECT_FALSE(VerifyManifestSignature("m", "zz", trusted_).ok());
    EXPECT_FALSE(VerifyManifestSignature("m", "", trusted_).ok());
    EXPECT_FALSE(
        VerifyManifestSignature("m", std::string(128, 'g'), trusted_).ok());
    TrustedKeys empty;
    EXPECT_FALSE(
        VerifyManifestSignature("m", std::string(128, 'a'), empty).ok());
}

TEST(HexTest, RoundTrip) {
    uint8_t bytes[4] = {0x00, 0xff, 0x12, 0xab};
    std::string hex = BytesToHex(bytes, 4);
    EXPECT_EQ(hex, "00ff12ab");
    uint8_t back[4];
    ASSERT_TRUE(HexToBytes(hex, back, 4).ok());
    EXPECT_EQ(0, std::memcmp(bytes, back, 4));
    EXPECT_FALSE(HexToBytes("00ff12", back, 4).ok());
}

}  // namespace
}  // namespace lykuro::nie
