#include "model/loader/artifact_loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>

#include "core/generation/sampler.h"
#include "model/tokenizer/prompt_template.h"
#include "security/sha256.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "artifact_loader";

Status VerifyFailed(const std::string& msg) {
    return Status(ErrorCode::kArtifactVerificationFailed, msg, kComponent);
}

// Reads a regular file (rejecting symlinks) with a size cap.
Status ReadFileCapped(const std::string& path, uint64_t max_bytes,
                      std::string& out) {
    int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return VerifyFailed("artifact file cannot be opened");
    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        return VerifyFailed("artifact file is not a regular file");
    }
    if (uint64_t(st.st_size) > max_bytes) {
        ::close(fd);
        return VerifyFailed("artifact file exceeds size limit");
    }
    out.resize(size_t(st.st_size));
    size_t off = 0;
    while (off < out.size()) {
        ssize_t n = ::read(fd, out.data() + off, out.size() - off);
        if (n <= 0) {
            ::close(fd);
            return VerifyFailed("artifact file read failed");
        }
        off += size_t(n);
    }
    ::close(fd);
    return Status::Ok();
}

// Streams a file through SHA-256 and compares against the manifest digest.
Status VerifyFileDigest(const std::string& path, const std::string& expected,
                        uint64_t expected_size) {
    int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return VerifyFailed("artifact file cannot be opened");
    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        return VerifyFailed("artifact file is not a regular file");
    }
    if (uint64_t(st.st_size) != expected_size) {
        ::close(fd);
        return VerifyFailed("artifact file size mismatch");
    }
    Sha256 hasher;
    std::vector<char> buf(1 << 20);
    ssize_t n;
    while ((n = ::read(fd, buf.data(), buf.size())) > 0) {
        hasher.Update(buf.data(), size_t(n));
    }
    ::close(fd);
    if (n < 0) return VerifyFailed("artifact file read failed");
    if (!DigestEquals(Sha256::ToHex(hasher.Finish()), expected)) {
        return VerifyFailed("artifact file digest mismatch");
    }
    return Status::Ok();
}

}  // namespace

ArtifactLoadResult LoadArtifact(const std::string& artifact_dir,
                                const ArtifactLoadOptions& options) {
    ArtifactLoadResult result;

    // Phase 3 will verify manifest.sig against the trusted key set; until
    // that lands, refuse to load unless explicitly in dev mode.
    if (!options.allow_unsigned_dev) {
        result.status = VerifyFailed(
            "signature verification unavailable; refusing unsigned load");
        return result;
    }

    std::string manifest_text;
    result.status = ReadFileCapped(artifact_dir + "/manifest.json",
                                   options.max_manifest_bytes, manifest_text);
    if (!result.status.ok()) return result;

    ManifestParseResult manifest =
        ParseManifest(manifest_text, options.max_manifest_bytes);
    if (!manifest.status.ok()) {
        result.status = manifest.status;
        return result;
    }
    result.artifact.manifest = std::move(manifest.manifest);
    const ModelManifest& m = result.artifact.manifest;

    // Digest check for every listed file. Paths were already validated
    // as traversal-safe by the manifest parser.
    std::string weights_path;
    std::string tokenizer_path;
    for (const ManifestFile& f : m.files) {
        const std::string full = artifact_dir + "/" + f.path;
        result.status = VerifyFileDigest(full, f.sha256, f.size_bytes);
        if (!result.status.ok()) return result;
        if (f.path.size() > 12 &&
            f.path.compare(f.path.size() - 12, 12, ".safetensors") == 0) {
            if (!weights_path.empty()) {
                result.status =
                    VerifyFailed("sharded weights not supported in MVP");
                return result;
            }
            weights_path = full;
        }
        if (f.path == "config/tokenizer.json") {
            tokenizer_path = full;
        }
    }
    if (weights_path.empty() || tokenizer_path.empty()) {
        result.status =
            VerifyFailed("manifest must list weights and tokenizer files");
        return result;
    }

    // Tokenizer + template.
    std::string tokenizer_text;
    result.status = ReadFileCapped(tokenizer_path,
                                   options.max_tokenizer_bytes,
                                   tokenizer_text);
    if (!result.status.ok()) return result;
    auto tokenizer = BpeTokenizer::FromConfig(tokenizer_text);
    if (!tokenizer.status.ok()) {
        result.status = tokenizer.status;
        return result;
    }
    result.artifact.tokenizer =
        std::make_unique<BpeTokenizer>(std::move(tokenizer.tokenizer));

    if (m.chat_template_id != QwenChatTemplate::kTemplateId) {
        result.status = VerifyFailed("chat_template_id is not approved");
        return result;
    }
    result.status = QwenChatTemplate::Validate(*result.artifact.tokenizer);
    if (!result.status.ok()) return result;

    // Tokenizer vocab must fit the model's output space.
    if (result.artifact.tokenizer->vocab_size() > m.vocab_size) {
        result.status =
            VerifyFailed("tokenizer vocab exceeds model vocab_size");
        return result;
    }

    // Weights.
    result.artifact.weights = std::make_unique<SafetensorsFile>();
    result.status = result.artifact.weights->Open(weights_path);
    if (!result.status.ok()) return result;

    auto model = QwenModel::Load(m, *result.artifact.weights);
    if (!model.status.ok()) {
        result.status = model.status;
        return result;
    }
    result.artifact.model = std::move(model.model);

    // Smoke inference (spec §12.4): one greedy token from the chat
    // template's own special tokens; ready is never reported without it.
    {
        std::vector<uint32_t> probe;
        Status s = QwenChatTemplate::BuildPrompt(
            *result.artifact.tokenizer, {{Role::kUser, "ping"}}, probe);
        if (s.ok()) {
            QwenKvCache cache(result.artifact.model->config(),
                              uint32_t(probe.size()) + 1);
            std::vector<float> logits;
            s = result.artifact.model->Prefill(probe, cache, logits);
            if (s.ok()) {
                SamplingParams greedy;
                greedy.temperature = 0.0f;
                Sampler sampler(greedy);
                uint32_t token;
                s = sampler.Sample(logits, token);
            }
        }
        if (!s.ok()) {
            result.status = Status(ErrorCode::kArtifactVerificationFailed,
                                   "smoke inference failed", kComponent);
            result.artifact = LoadedArtifact{};
            return result;
        }
    }

    result.status = Status::Ok();
    return result;
}

}  // namespace lykuro::nie
