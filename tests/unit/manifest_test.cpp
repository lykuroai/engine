#include "model/manifest/manifest.h"

#include <gtest/gtest.h>

#include <string>

namespace lykuro::nie {
namespace {

std::string ValidManifestJson() {
    return R"({
      "schema_version": "1",
      "artifact_id": "ma_qwen_test",
      "model_family": "qwen",
      "architecture": "approved_qwen_decoder_v1",
      "model_version": "test-0.1",
      "weight_format": "safetensors",
      "precision": "bf16",
      "vocab_size": 1024,
      "hidden_size": 64,
      "num_layers": 2,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 16,
      "max_context_tokens": 512,
      "intermediate_size": 128,
      "rms_norm_eps": 1e-6,
      "rope_theta": 10000.0,
      "tie_word_embeddings": true,
      "eos_token_ids": [1000],
      "tokenizer_type": "approved_qwen_tokenizer_v1",
      "chat_template_id": "qwen_chat_v1",
      "files": [
        {"path": "weights/model.safetensors",
         "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
         "size_bytes": 123}
      ],
      "license_review_id": "lic_test_01",
      "certified_profiles": ["cp_test_01"],
      "created_at": "2026-08-07T00:00:00Z"
    })";
}

// Naive single-occurrence replace helper for building negative fixtures.
std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
    auto pos = text.find(from);
    EXPECT_NE(pos, std::string::npos);
    return text.replace(pos, from.size(), to);
}

TEST(ManifestTest, AcceptsValidManifest) {
    auto r = ParseManifest(ValidManifestJson());
    ASSERT_TRUE(r.status.ok()) << r.status.message();
    EXPECT_EQ(r.manifest.artifact_id, "ma_qwen_test");
    EXPECT_EQ(r.manifest.num_layers, 2u);
    EXPECT_EQ(r.manifest.files.size(), 1u);
    EXPECT_TRUE(r.manifest.tie_word_embeddings);
    EXPECT_EQ(r.manifest.eos_token_ids, std::vector<uint32_t>{1000});
}

TEST(ManifestTest, RejectsUnknownField) {
    auto json = Replace(ValidManifestJson(), "\"schema_version\": \"1\",",
                        "\"schema_version\": \"1\", \"mystery\": 1,");
    auto r = ParseManifest(json);
    EXPECT_FALSE(r.status.ok());
    EXPECT_EQ(r.status.code(), ErrorCode::kArtifactVerificationFailed);
}

TEST(ManifestTest, RejectsMissingRequiredField) {
    auto json = Replace(ValidManifestJson(),
                        "\"license_review_id\": \"lic_test_01\",", "");
    EXPECT_FALSE(ParseManifest(json).status.ok());
}

TEST(ManifestTest, RejectsWrongSchemaVersion) {
    auto json = Replace(ValidManifestJson(), "\"schema_version\": \"1\"",
                        "\"schema_version\": \"2\"");
    EXPECT_FALSE(ParseManifest(json).status.ok());
}

TEST(ManifestTest, RejectsUnapprovedFamily) {
    auto json = Replace(ValidManifestJson(), "\"model_family\": \"qwen\"",
                        "\"model_family\": \"other\"");
    auto r = ParseManifest(json);
    EXPECT_FALSE(r.status.ok());
    EXPECT_EQ(r.status.code(), ErrorCode::kUnsupportedModel);
}

TEST(ManifestTest, RejectsPathTraversal) {
    auto json = Replace(ValidManifestJson(), "weights/model.safetensors",
                        "../../etc/passwd");
    EXPECT_FALSE(ParseManifest(json).status.ok());
}

TEST(ManifestTest, RejectsAbsolutePath) {
    auto json = Replace(ValidManifestJson(), "weights/model.safetensors",
                        "/etc/passwd");
    EXPECT_FALSE(ParseManifest(json).status.ok());
}

TEST(ManifestTest, RejectsBadDigestFormat) {
    auto json = Replace(
        ValidManifestJson(),
        "0000000000000000000000000000000000000000000000000000000000000000",
        "ZZ");
    EXPECT_FALSE(ParseManifest(json).status.ok());
}

TEST(ManifestTest, RejectsInconsistentHeads) {
    auto json = Replace(ValidManifestJson(), "\"num_key_value_heads\": 2",
                        "\"num_key_value_heads\": 3");
    EXPECT_FALSE(ParseManifest(json).status.ok());
}

TEST(ManifestTest, RejectsOversizedInput) {
    EXPECT_FALSE(
        ParseManifest(ValidManifestJson(), /*max_bytes=*/16).status.ok());
}

TEST(ManifestTest, RejectsZeroDimensions) {
    auto json = Replace(ValidManifestJson(), "\"hidden_size\": 64",
                        "\"hidden_size\": 0");
    EXPECT_FALSE(ParseManifest(json).status.ok());
}

TEST(SafePathTest, Cases) {
    EXPECT_TRUE(IsSafeArtifactPath("weights/model.safetensors"));
    EXPECT_TRUE(IsSafeArtifactPath("config/model.json"));
    EXPECT_FALSE(IsSafeArtifactPath(""));
    EXPECT_FALSE(IsSafeArtifactPath("/abs/path"));
    EXPECT_FALSE(IsSafeArtifactPath("a/../b"));
    EXPECT_FALSE(IsSafeArtifactPath(".."));
    EXPECT_FALSE(IsSafeArtifactPath("a//b"));
    EXPECT_FALSE(IsSafeArtifactPath("a\\b"));
    EXPECT_FALSE(IsSafeArtifactPath("a/./b"));
    EXPECT_FALSE(IsSafeArtifactPath("sp ace"));
    EXPECT_FALSE(IsSafeArtifactPath("-flag"));
}

}  // namespace
}  // namespace lykuro::nie
