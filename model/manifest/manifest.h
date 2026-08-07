#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// Parsed, validated model manifest (spec §10.2, schema
// api/schema/model-manifest.schema.json). Instances only exist after full
// validation; there is no partially-valid manifest.

struct ManifestFile {
    std::string path;       // relative, normalized, no traversal
    std::string sha256;     // lowercase hex
    uint64_t size_bytes = 0;
};

struct ModelManifest {
    std::string schema_version;
    std::string artifact_id;
    std::string model_family;
    std::string architecture;
    std::string model_version;
    std::string weight_format;
    std::string precision;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t num_layers = 0;
    uint32_t num_attention_heads = 0;
    uint32_t num_key_value_heads = 0;
    uint32_t head_dim = 0;
    uint32_t max_context_tokens = 0;
    uint32_t intermediate_size = 0;      // optional (0 = unset)
    double rms_norm_eps = 0.0;           // optional
    double rope_theta = 0.0;             // optional
    bool tie_word_embeddings = false;
    std::vector<uint32_t> eos_token_ids; // optional
    std::string tokenizer_type;
    std::string chat_template_id;
    std::vector<ManifestFile> files;
    std::string license_review_id;
    std::vector<std::string> certified_profiles;
    std::string created_at;
};

struct ManifestParseResult {
    Status status;
    ModelManifest manifest;  // valid only when status.ok()
};

// Parses and validates manifest.json content. Enforces:
//  - schema_version == "1"
//  - all required fields present with valid types/ranges
//  - unknown fields rejected (spec §10.4)
//  - identifier patterns (artifact/license/profile ids)
//  - file paths: relative, no "..", no leading '/', no backslash
//  - per-file sha256 format and non-duplicate paths
// max_bytes bounds the accepted input size (DoS control).
ManifestParseResult ParseManifest(std::string_view json_text,
                                  size_t max_bytes = 1 << 20);

// True when `path` is a safe relative path inside an artifact.
bool IsSafeArtifactPath(std::string_view path);

}  // namespace lykuro::nie
