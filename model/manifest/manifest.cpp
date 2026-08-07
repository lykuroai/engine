#include "model/manifest/manifest.h"

#include <algorithm>
#include <set>

#include "core/engine/json.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "manifest";

Status Invalid(const std::string& msg) {
    return Status(ErrorCode::kArtifactVerificationFailed, msg, kComponent);
}

bool MatchesIdPattern(std::string_view value, std::string_view prefix,
                      size_t max_len = 68) {
    if (value.size() <= prefix.size() || value.size() > max_len) return false;
    if (value.substr(0, prefix.size()) != prefix) return false;
    return std::all_of(value.begin() + long(prefix.size()), value.end(),
                       [](char c) {
                           return (c >= 'a' && c <= 'z') ||
                                  (c >= '0' && c <= '9') || c == '_';
                       });
}

bool IsLowerHex64(std::string_view s) {
    if (s.size() != 64) return false;
    return std::all_of(s.begin(), s.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

// Helper for pulling typed fields out of the manifest object while
// tracking which keys have been consumed (unknown keys are rejected).
class FieldReader {
public:
    explicit FieldReader(const json::Value& obj) : obj_(obj) {}

    const json::Value* Get(std::string_view key, bool required,
                           Status& status) {
        seen_.insert(std::string(key));
        const json::Value* v = obj_.Find(key);
        if (v == nullptr && required && status.ok()) {
            status = Invalid("manifest missing required field: " +
                             std::string(key));
        }
        return v;
    }

    bool GetString(std::string_view key, bool required, std::string& out,
                   Status& status) {
        const json::Value* v = Get(key, required, status);
        if (v == nullptr) return false;
        if (!v->is_string()) {
            if (status.ok()) {
                status = Invalid("manifest field has wrong type: " +
                                 std::string(key));
            }
            return false;
        }
        out = v->as_string();
        return true;
    }

    bool GetU32(std::string_view key, bool required, uint32_t& out,
                Status& status, int64_t min_value = 1) {
        const json::Value* v = Get(key, required, status);
        if (v == nullptr) return false;
        if (!v->is_int() || v->as_int() < min_value ||
            v->as_int() > int64_t(UINT32_MAX)) {
            if (status.ok()) {
                status = Invalid("manifest field out of range: " +
                                 std::string(key));
            }
            return false;
        }
        out = uint32_t(v->as_int());
        return true;
    }

    bool GetPositiveDouble(std::string_view key, double& out, Status& status) {
        const json::Value* v = Get(key, /*required=*/false, status);
        if (v == nullptr) return false;
        if (!v->is_number() || v->as_double() <= 0.0) {
            if (status.ok()) {
                status = Invalid("manifest field out of range: " +
                                 std::string(key));
            }
            return false;
        }
        out = v->as_double();
        return true;
    }

    Status CheckNoUnknownKeys() const {
        for (const auto& [key, value] : obj_.as_object()) {
            (void)value;
            if (!seen_.count(key)) {
                // Unknown config fields are never silently accepted
                // (spec §10.4).
                return Invalid("manifest contains unknown field: " + key);
            }
        }
        return Status::Ok();
    }

private:
    const json::Value& obj_;
    std::set<std::string> seen_;
};

}  // namespace

bool IsSafeArtifactPath(std::string_view path) {
    if (path.empty() || path.size() > 512) return false;
    if (path.front() == '/' || path.front() == '-') return false;
    if (path.find('\\') != std::string_view::npos) return false;
    if (path.find('\0') != std::string_view::npos) return false;
    // Reject any ".." segment and empty segments ("//").
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find('/', start);
        if (end == std::string_view::npos) end = path.size();
        std::string_view seg = path.substr(start, end - start);
        if (seg.empty() || seg == "." || seg == "..") return false;
        // Allow conservative filename characters only.
        for (char c : seg) {
            bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '.' || c == '_' ||
                      c == '-';
            if (!ok) return false;
        }
        if (end == path.size()) break;
        start = end + 1;
    }
    return true;
}

ManifestParseResult ParseManifest(std::string_view json_text,
                                  size_t max_bytes) {
    ManifestParseResult result;

    if (json_text.size() > max_bytes) {
        result.status = Invalid("manifest exceeds size limit");
        return result;
    }

    json::ParseResult parsed = json::Parse(json_text);
    if (!parsed.ok()) {
        result.status = Invalid("manifest is not valid JSON: " + parsed.error);
        return result;
    }
    if (!parsed.value->is_object()) {
        result.status = Invalid("manifest root must be an object");
        return result;
    }

    const json::Value& root = *parsed.value;
    ModelManifest& m = result.manifest;
    Status status;
    FieldReader r(root);

    r.GetString("schema_version", true, m.schema_version, status);
    r.GetString("artifact_id", true, m.artifact_id, status);
    r.GetString("model_family", true, m.model_family, status);
    r.GetString("architecture", true, m.architecture, status);
    r.GetString("model_version", true, m.model_version, status);
    r.GetString("weight_format", true, m.weight_format, status);
    r.GetString("precision", true, m.precision, status);
    r.GetU32("vocab_size", true, m.vocab_size, status);
    r.GetU32("hidden_size", true, m.hidden_size, status);
    r.GetU32("num_layers", true, m.num_layers, status);
    r.GetU32("num_attention_heads", true, m.num_attention_heads, status);
    r.GetU32("num_key_value_heads", true, m.num_key_value_heads, status);
    r.GetU32("head_dim", true, m.head_dim, status);
    r.GetU32("max_context_tokens", true, m.max_context_tokens, status);
    r.GetU32("intermediate_size", false, m.intermediate_size, status);
    r.GetPositiveDouble("rms_norm_eps", m.rms_norm_eps, status);
    r.GetPositiveDouble("rope_theta", m.rope_theta, status);
    r.GetString("tokenizer_type", true, m.tokenizer_type, status);
    r.GetString("chat_template_id", true, m.chat_template_id, status);
    r.GetString("license_review_id", true, m.license_review_id, status);
    r.GetString("created_at", true, m.created_at, status);

    if (const json::Value* v =
            r.Get("tie_word_embeddings", /*required=*/false, status)) {
        if (!v->is_bool()) {
            if (status.ok()) {
                status = Invalid(
                    "manifest field has wrong type: tie_word_embeddings");
            }
        } else {
            m.tie_word_embeddings = v->as_bool();
        }
    }

    if (const json::Value* v =
            r.Get("eos_token_ids", /*required=*/false, status)) {
        if (!v->is_array() || v->as_array().empty() ||
            v->as_array().size() > 8) {
            if (status.ok()) {
                status = Invalid("manifest field out of range: eos_token_ids");
            }
        } else {
            for (const auto& item : v->as_array()) {
                if (!item->is_int() || item->as_int() < 0 ||
                    item->as_int() > int64_t(UINT32_MAX)) {
                    if (status.ok()) {
                        status = Invalid(
                            "manifest field out of range: eos_token_ids");
                    }
                    break;
                }
                m.eos_token_ids.push_back(uint32_t(item->as_int()));
            }
        }
    }

    if (const json::Value* v = r.Get("files", /*required=*/true, status)) {
        if (!v->is_array() || v->as_array().empty() ||
            v->as_array().size() > 4096) {
            if (status.ok()) {
                status = Invalid("manifest files list invalid");
            }
        } else {
            std::set<std::string> seen_paths;
            for (const auto& item : v->as_array()) {
                if (!item->is_object()) {
                    if (status.ok()) status = Invalid("manifest file entry invalid");
                    break;
                }
                ManifestFile f;
                FieldReader fr(*item);
                fr.GetString("path", true, f.path, status);
                fr.GetString("sha256", true, f.sha256, status);
                const json::Value* size =
                    fr.Get("size_bytes", /*required=*/true, status);
                if (status.ok() && size != nullptr) {
                    if (!size->is_int() || size->as_int() < 0) {
                        status = Invalid("manifest file size invalid");
                    } else {
                        f.size_bytes = uint64_t(size->as_int());
                    }
                }
                if (status.ok()) status = fr.CheckNoUnknownKeys();
                if (!status.ok()) break;
                if (!IsSafeArtifactPath(f.path)) {
                    status = Invalid("manifest file path is unsafe");
                    break;
                }
                if (!IsLowerHex64(f.sha256)) {
                    status = Invalid("manifest file sha256 invalid");
                    break;
                }
                if (!seen_paths.insert(f.path).second) {
                    status = Invalid("manifest file path duplicated");
                    break;
                }
                m.files.push_back(std::move(f));
            }
        }
    }

    if (const json::Value* v =
            r.Get("certified_profiles", /*required=*/true, status)) {
        if (!v->is_array()) {
            if (status.ok()) {
                status = Invalid("manifest certified_profiles invalid");
            }
        } else {
            for (const auto& item : v->as_array()) {
                if (!item->is_string() ||
                    !MatchesIdPattern(item->as_string(), "cp_")) {
                    if (status.ok()) {
                        status = Invalid("manifest certified profile id invalid");
                    }
                    break;
                }
                m.certified_profiles.push_back(item->as_string());
            }
        }
    }

    if (status.ok()) status = r.CheckNoUnknownKeys();

    // Semantic validation.
    if (status.ok() && m.schema_version != "1") {
        status = Invalid("unsupported manifest schema_version");
    }
    if (status.ok() && !MatchesIdPattern(m.artifact_id, "ma_")) {
        status = Invalid("manifest artifact_id invalid");
    }
    if (status.ok() && m.model_family != "qwen") {
        status = Status(ErrorCode::kUnsupportedModel,
                        "model family is not approved", kComponent);
    }
    if (status.ok() && !MatchesIdPattern(m.architecture, "approved_")) {
        status = Status(ErrorCode::kUnsupportedModel,
                        "architecture id is not approved", kComponent);
    }
    if (status.ok() && m.weight_format != "safetensors") {
        status = Invalid("unsupported weight_format");
    }
    if (status.ok() && m.precision != "bf16" && m.precision != "fp16" &&
        m.precision != "fp32") {
        status = Invalid("unsupported precision");
    }
    if (status.ok() && !MatchesIdPattern(m.tokenizer_type, "approved_")) {
        status = Invalid("tokenizer_type is not approved");
    }
    if (status.ok() && !MatchesIdPattern(m.license_review_id, "lic_")) {
        status = Invalid("license_review_id invalid");
    }
    if (status.ok() &&
        m.num_attention_heads % m.num_key_value_heads != 0) {
        status = Invalid("attention head configuration inconsistent");
    }

    result.status = status;
    if (!status.ok()) {
        result.manifest = ModelManifest{};
    }
    return result;
}

}  // namespace lykuro::nie
