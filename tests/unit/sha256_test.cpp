#include "security/sha256.h"

#include <gtest/gtest.h>

#include <string>

namespace lykuro::nie {
namespace {

// FIPS 180-4 / NIST test vectors.
TEST(Sha256Test, EmptyInput) {
    EXPECT_EQ(
        Sha256::HexDigest("", 0),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(Sha256Test, Abc) {
    EXPECT_EQ(
        Sha256::HexDigest("abc", 3),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(Sha256Test, TwoBlockMessage) {
    const std::string msg =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    EXPECT_EQ(
        Sha256::HexDigest(msg.data(), msg.size()),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST(Sha256Test, MillionA) {
    Sha256 h;
    std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) h.Update(chunk.data(), chunk.size());
    EXPECT_EQ(
        Sha256::ToHex(h.Finish()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, IncrementalMatchesOneShot) {
    const std::string msg = "The quick brown fox jumps over the lazy dog";
    Sha256 h;
    for (char c : msg) h.Update(&c, 1);
    EXPECT_EQ(Sha256::ToHex(h.Finish()),
              Sha256::HexDigest(msg.data(), msg.size()));
}

TEST(Sha256Test, DigestEqualsIsConstantTimeCompatible) {
    EXPECT_TRUE(DigestEquals("abcd", "abcd"));
    EXPECT_FALSE(DigestEquals("abcd", "abce"));
    EXPECT_FALSE(DigestEquals("abcd", "abc"));
}

}  // namespace
}  // namespace lykuro::nie
