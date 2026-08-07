#pragma once

#include <memory>
#include <string>

#include "core/engine/error.h"
#include "model/architectures/qwen/qwen_model.h"
#include "model/manifest/manifest.h"
#include "model/tokenizer/bpe_tokenizer.h"
#include "security/signature.h"

namespace lykuro::nie {

struct ArtifactLoadOptions {
    // Ed25519 public keys trusted to sign manifests. When set, the
    // artifact's manifest.sig must verify over the exact manifest.json
    // bytes with one of these keys.
    TrustedKeys trusted_keys;
    // Development escape hatch: skips signature verification entirely.
    // Must never be set in production; with neither trusted keys nor this
    // flag, every load is refused (fail-closed, spec §0.2).
    bool allow_unsigned_dev = false;
    uint64_t max_manifest_bytes = 1 << 20;
    uint64_t max_tokenizer_bytes = 64ull << 20;
};

struct LoadedArtifact {
    ModelManifest manifest;
    std::unique_ptr<QwenModel> model;
    std::unique_ptr<BpeTokenizer> tokenizer;
    // Keeps the weight mapping alive for the model's lifetime.
    std::unique_ptr<SafetensorsFile> weights;
};

struct ArtifactLoadResult {
    Status status;
    LoadedArtifact artifact;  // valid only when status.ok()
};

// Full load flow (spec §12.1): manifest verify -> digest check for every
// listed file -> architecture resolve -> weight load with shape checks ->
// tokenizer + template validation -> smoke inference. Any failure rejects
// the load and leaves nothing resident.
//
// Layout expected under `artifact_dir` (spec §10.1):
//   manifest.json, config/tokenizer.json, weights/*.safetensors
ArtifactLoadResult LoadArtifact(const std::string& artifact_dir,
                                const ArtifactLoadOptions& options);

}  // namespace lykuro::nie
