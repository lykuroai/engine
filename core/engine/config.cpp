#include "core/engine/config.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <set>

#include "core/engine/json.h"

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "config";

Status Invalid(const std::string& msg) {
    return Status(ErrorCode::kInvalidRequest, msg, kComponent);
}

// Validates that an object contains no keys outside `allowed`.
Status CheckKeys(const json::Value& obj, const char* section,
                 const std::set<std::string>& allowed) {
    for (const auto& [key, value] : obj.as_object()) {
        (void)value;
        if (!allowed.count(key)) {
            return Invalid(std::string("config contains unknown key in ") +
                           section + ": " + key);
        }
    }
    return Status::Ok();
}

Status GetString(const json::Value& obj, const char* key, std::string& out) {
    const json::Value* v = obj.Find(key);
    if (v == nullptr) return Status::Ok();
    if (!v->is_string()) {
        return Invalid(std::string("config key has wrong type: ") + key);
    }
    out = v->as_string();
    return Status::Ok();
}

Status GetBool(const json::Value& obj, const char* key, bool& out) {
    const json::Value* v = obj.Find(key);
    if (v == nullptr) return Status::Ok();
    if (!v->is_bool()) {
        return Invalid(std::string("config key has wrong type: ") + key);
    }
    out = v->as_bool();
    return Status::Ok();
}

template <typename T>
Status GetUint(const json::Value& obj, const char* key, T& out,
               int64_t min_value, int64_t max_value) {
    const json::Value* v = obj.Find(key);
    if (v == nullptr) return Status::Ok();
    if (!v->is_int() || v->as_int() < min_value || v->as_int() > max_value) {
        return Invalid(std::string("config key out of range: ") + key);
    }
    out = T(v->as_int());
    return Status::Ok();
}

Status GetStringList(const json::Value& obj, const char* key,
                     std::vector<std::string>& out) {
    const json::Value* v = obj.Find(key);
    if (v == nullptr) return Status::Ok();
    if (!v->is_array()) {
        return Invalid(std::string("config key has wrong type: ") + key);
    }
    for (const auto& item : v->as_array()) {
        if (!item->is_string()) {
            return Invalid(std::string("config key has wrong type: ") + key);
        }
        out.push_back(item->as_string());
    }
    return Status::Ok();
}

}  // namespace

FileConfigResult ParseFileConfig(std::string_view json_text,
                                 size_t max_bytes) {
    FileConfigResult result;
    if (json_text.size() > max_bytes) {
        result.status = Invalid("config exceeds size limit");
        return result;
    }
    json::ParseResult parsed = json::Parse(json_text);
    if (!parsed.ok()) {
        result.status = Invalid("config is not valid JSON: " + parsed.error);
        return result;
    }
    if (!parsed.value->is_object()) {
        result.status = Invalid("config root must be an object");
        return result;
    }
    const json::Value& root = *parsed.value;
    FileConfig& c = result.config;
    Status s = CheckKeys(root, "root",
                         {"engine", "security", "model", "hardware",
                          "scheduler", "generation", "observability"});

    if (s.ok()) {
        if (const json::Value* v = root.Find("engine")) {
            if (!v->is_object()) {
                s = Invalid("config section must be an object: engine");
            } else {
                s = CheckKeys(*v, "engine",
                              {"id", "listen_address", "grpc_port",
                               "log_level"});
                if (s.ok()) s = GetString(*v, "id", c.engine_id);
                if (s.ok()) s = GetString(*v, "listen_address",
                                          c.listen_address);
                if (s.ok()) s = GetUint(*v, "grpc_port", c.grpc_port, 0,
                                        65535);
                if (s.ok()) s = GetString(*v, "log_level", c.log_level);
            }
        }
    }
    if (s.ok()) {
        if (const json::Value* v = root.Find("security")) {
            if (!v->is_object()) {
                s = Invalid("config section must be an object: security");
            } else {
                s = CheckKeys(*v, "security",
                              {"mtls_required", "server_cert_path",
                               "server_key_path", "client_ca_path",
                               "control_identities", "data_identities",
                               "trusted_signing_keys", "allow_unsigned_dev"});
                if (s.ok()) s = GetBool(*v, "mtls_required", c.mtls_required);
                if (s.ok()) s = GetString(*v, "server_cert_path",
                                          c.server_cert_path);
                if (s.ok()) s = GetString(*v, "server_key_path",
                                          c.server_key_path);
                if (s.ok()) s = GetString(*v, "client_ca_path",
                                          c.client_ca_path);
                if (s.ok()) s = GetStringList(*v, "control_identities",
                                              c.control_identities);
                if (s.ok()) s = GetStringList(*v, "data_identities",
                                              c.data_identities);
                if (s.ok()) s = GetStringList(*v, "trusted_signing_keys",
                                              c.trusted_signing_keys_hex);
                if (s.ok()) s = GetBool(*v, "allow_unsigned_dev",
                                        c.allow_unsigned_dev);
            }
        }
    }
    if (s.ok()) {
        if (const json::Value* v = root.Find("model")) {
            if (!v->is_object()) {
                s = Invalid("config section must be an object: model");
            } else {
                s = CheckKeys(*v, "model", {"artifact_path"});
                if (s.ok()) s = GetString(*v, "artifact_path",
                                          c.artifact_path);
            }
        }
    }
    if (s.ok()) {
        if (const json::Value* v = root.Find("hardware")) {
            if (!v->is_object()) {
                s = Invalid("config section must be an object: hardware");
            } else {
                s = CheckKeys(*v, "hardware", {"backend", "device_id"});
                if (s.ok()) s = GetString(*v, "backend", c.hardware_backend);
                if (s.ok()) {
                    uint32_t device = 0;
                    s = GetUint(*v, "device_id", device, 0, 63);
                    c.device_id = int(device);
                }
            }
        }
    }
    if (s.ok()) {
        if (const json::Value* v = root.Find("scheduler")) {
            if (!v->is_object()) {
                s = Invalid("config section must be an object: scheduler");
            } else {
                s = CheckKeys(*v, "scheduler",
                              {"max_queue", "max_sequences"});
                if (s.ok()) s = GetUint(*v, "max_queue", c.max_queue, 1,
                                        1 << 20);
                if (s.ok()) s = GetUint(*v, "max_sequences",
                                        c.max_sequences, 1, 4096);
            }
        }
    }
    if (s.ok()) {
        if (const json::Value* v = root.Find("generation")) {
            if (!v->is_object()) {
                s = Invalid("config section must be an object: generation");
            } else {
                s = CheckKeys(*v, "generation",
                              {"max_output_tokens", "max_input_bytes"});
                if (s.ok()) s = GetUint(*v, "max_output_tokens",
                                        c.max_output_tokens, 1, 1 << 20);
                if (s.ok()) s = GetUint(*v, "max_input_bytes",
                                        c.max_input_bytes, 1,
                                        int64_t(1) << 32);
            }
        }
    }
    if (s.ok()) {
        if (const json::Value* v = root.Find("observability")) {
            if (!v->is_object()) {
                s = Invalid(
                    "config section must be an object: observability");
            } else {
                s = CheckKeys(*v, "observability",
                              {"metrics_enabled", "metrics_port"});
                if (s.ok()) s = GetBool(*v, "metrics_enabled",
                                        c.metrics_enabled);
                if (s.ok()) s = GetUint(*v, "metrics_port", c.metrics_port,
                                        0, 65535);
            }
        }
    }

    // Cross-field validation (spec §26.3 incompatible combinations).
    if (s.ok() && c.hardware_backend != "cpu" &&
        c.hardware_backend != "cuda") {
        s = Invalid("hardware backend must be cpu or cuda");
    }
    if (s.ok() && c.log_level != "debug" && c.log_level != "info" &&
        c.log_level != "warn" && c.log_level != "error") {
        s = Invalid("config log_level invalid");
    }
    if (s.ok() && c.mtls_required &&
        (c.server_cert_path.empty() || c.server_key_path.empty() ||
         c.client_ca_path.empty())) {
        s = Invalid("mtls_required needs cert, key, and client CA paths");
    }
    if (s.ok() && c.trusted_signing_keys_hex.empty() &&
        !c.allow_unsigned_dev) {
        s = Invalid(
            "either trusted_signing_keys or allow_unsigned_dev must be set");
    }
    for (const std::string& key : c.trusted_signing_keys_hex) {
        if (!s.ok()) break;
        if (key.size() != 64) {
            s = Invalid("trusted signing key must be 64 hex chars");
        }
    }

    result.status = s;
    if (!s.ok()) result.config = FileConfig{};
    return result;
}

FileConfigResult LoadFileConfig(const std::string& path) {
    FileConfigResult result;
    int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        result.status = Invalid("config file cannot be opened");
        return result;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
        uint64_t(st.st_size) > (1 << 20)) {
        ::close(fd);
        result.status = Invalid("config file invalid");
        return result;
    }
    std::string text(size_t(st.st_size), '\0');
    size_t off = 0;
    while (off < text.size()) {
        ssize_t n = ::read(fd, text.data() + off, text.size() - off);
        if (n <= 0) {
            ::close(fd);
            result.status = Invalid("config file read failed");
            return result;
        }
        off += size_t(n);
    }
    ::close(fd);
    return ParseFileConfig(text);
}

}  // namespace lykuro::nie
