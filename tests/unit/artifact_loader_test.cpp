#include "model/loader/artifact_loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "security/sha256.h"
#include "tests/testutil/tiny_model.h"
#include "tests/unit/tokenizer_fixture.h"

namespace lykuro::nie {
namespace {

namespace fs = std::filesystem;

std::string FileSha256(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    return Sha256::HexDigest(data.data(), data.size());
}

class ArtifactLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::path(testing::TempDir()) / "loader_artifact";
        fs::remove_all(dir_);
        fs::create_directories(dir_ / "config");
        fs::create_directories(dir_ / "weights");

        testutil::TinyModelSpec spec;
        spec.vocab_size = 303;
        spec.eos_token_id = 302;
        spec.max_context_tokens = 128;
        std::string tmp =
            testutil::WriteTinyWeights(spec, "loader_tiny.safetensors");
        fs::rename(tmp, dir_ / "weights" / "model.safetensors");
        {
            std::ofstream tok(dir_ / "config" / "tokenizer.json");
            tok << testfixture::SmallTokenizerConfig();
        }
        WriteManifest();
        ASSERT_TRUE(GenerateKeypair(pub_, priv_).ok());
    }

    void TearDown() override { fs::remove_all(dir_); }

    void WriteManifest() {
        auto entry = [&](const std::string& rel) {
            fs::path p = dir_ / rel;
            return "{\"path\": \"" + rel + "\", \"sha256\": \"" +
                   FileSha256(p) + "\", \"size_bytes\": " +
                   std::to_string(fs::file_size(p)) + "}";
        };
        std::ofstream m(dir_ / "manifest.json");
        m << R"({
          "schema_version": "1", "artifact_id": "ma_qwen_loader",
          "model_family": "qwen",
          "architecture": "approved_qwen_decoder_v1",
          "model_version": "tiny", "weight_format": "safetensors",
          "precision": "fp32", "vocab_size": 303, "hidden_size": 16,
          "num_layers": 2, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 4,
          "max_context_tokens": 128, "intermediate_size": 32,
          "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
          "tie_word_embeddings": true, "eos_token_ids": [302],
          "tokenizer_type": "approved_qwen_tokenizer_v1",
          "chat_template_id": "qwen_chat_v1",
          "files": [)"
          << entry("weights/model.safetensors") << ","
          << entry("config/tokenizer.json") << R"(],
          "license_review_id": "lic_test_01",
          "certified_profiles": [],
          "created_at": "2026-08-07T00:00:00Z"
        })";
    }

    void SignManifest(const std::array<uint8_t, 64>& key) {
        std::ifstream f(dir_ / "manifest.json", std::ios::binary);
        std::string manifest((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        std::array<uint8_t, 64> sig;
        ASSERT_TRUE(SignMessage(manifest, key, sig).ok());
        std::ofstream out(dir_ / "manifest.sig");
        out << BytesToHex(sig.data(), sig.size()) << "\n";
    }

    fs::path dir_;
    std::array<uint8_t, 32> pub_;
    std::array<uint8_t, 64> priv_;
};

TEST_F(ArtifactLoaderTest, RefusesLoadWithoutTrustAnchor) {
    ArtifactLoadOptions options;  // no keys, no dev flag
    auto r = LoadArtifact(dir_.string(), options);
    EXPECT_FALSE(r.status.ok());
    EXPECT_EQ(r.status.code(), ErrorCode::kArtifactVerificationFailed);
}

TEST_F(ArtifactLoaderTest, DevFlagAllowsUnsignedLoad) {
    ArtifactLoadOptions options;
    options.allow_unsigned_dev = true;
    auto r = LoadArtifact(dir_.string(), options);
    ASSERT_TRUE(r.status.ok()) << r.status.message();
    EXPECT_EQ(r.artifact.manifest.artifact_id, "ma_qwen_loader");
    EXPECT_NE(r.artifact.model, nullptr);
    EXPECT_NE(r.artifact.tokenizer, nullptr);
}

TEST_F(ArtifactLoaderTest, SignedLoadWithTrustedKeySucceeds) {
    SignManifest(priv_);
    ArtifactLoadOptions options;
    options.trusted_keys.keys.push_back(pub_);
    auto r = LoadArtifact(dir_.string(), options);
    ASSERT_TRUE(r.status.ok()) << r.status.message();
}

TEST_F(ArtifactLoaderTest, MissingSignatureRejectedWhenKeysConfigured) {
    ArtifactLoadOptions options;
    options.trusted_keys.keys.push_back(pub_);
    auto r = LoadArtifact(dir_.string(), options);
    EXPECT_FALSE(r.status.ok());
}

TEST_F(ArtifactLoaderTest, UntrustedSignatureRejected) {
    std::array<uint8_t, 32> other_pub;
    std::array<uint8_t, 64> other_priv;
    ASSERT_TRUE(GenerateKeypair(other_pub, other_priv).ok());
    SignManifest(other_priv);  // signed, but not by a trusted key

    ArtifactLoadOptions options;
    options.trusted_keys.keys.push_back(pub_);
    auto r = LoadArtifact(dir_.string(), options);
    EXPECT_FALSE(r.status.ok());
    EXPECT_EQ(r.status.code(), ErrorCode::kArtifactVerificationFailed);
}

TEST_F(ArtifactLoaderTest, ManifestTamperAfterSigningRejected) {
    SignManifest(priv_);
    // Re-write the manifest (same content shape, different bytes).
    {
        std::ofstream m(dir_ / "manifest.json", std::ios::app);
        m << "\n";
    }
    ArtifactLoadOptions options;
    options.trusted_keys.keys.push_back(pub_);
    auto r = LoadArtifact(dir_.string(), options);
    EXPECT_FALSE(r.status.ok());
}

TEST_F(ArtifactLoaderTest, TamperedWeightsRejectedEvenWhenSigned) {
    // Signature covers the manifest; the digest list covers the weights.
    SignManifest(priv_);
    {
        std::fstream f(dir_ / "weights" / "model.safetensors",
                       std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(-1, std::ios::end);
        f.put('\x5a');
    }
    ArtifactLoadOptions options;
    options.trusted_keys.keys.push_back(pub_);
    auto r = LoadArtifact(dir_.string(), options);
    EXPECT_FALSE(r.status.ok());
}

}  // namespace
}  // namespace lykuro::nie
