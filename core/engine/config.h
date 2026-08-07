#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// Engine configuration file (spec §26, JSON encoding). Strict parsing:
// unknown keys are rejected at every level, ranges are validated, and
// secret material is referenced by path, never inlined (spec §26.2).
struct FileConfig {
    // [engine]
    std::string engine_id = "nie-node-01";
    std::string listen_address = "127.0.0.1";
    uint16_t grpc_port = 19443;
    std::string log_level = "info";  // debug|info|warn|error

    // [security]
    bool mtls_required = true;
    std::string server_cert_path;  // required when mtls_required
    std::string server_key_path;
    std::string client_ca_path;
    std::vector<std::string> control_identities;
    std::vector<std::string> data_identities;
    std::vector<std::string> trusted_signing_keys_hex;  // Ed25519, 64 hex
    bool allow_unsigned_dev = false;

    // [model]
    std::string artifact_path;  // optional; loaded at startup when set

    // [scheduler]
    uint32_t max_queue = 256;
    uint32_t max_sequences = 8;

    // [generation]
    uint32_t max_output_tokens = 4096;
    uint64_t max_input_bytes = 1 << 20;

    // [observability]
    bool metrics_enabled = false;
    uint16_t metrics_port = 19090;
};

struct FileConfigResult {
    Status status;
    FileConfig config;  // valid only when status.ok()
};

// Parses and validates the JSON config text.
FileConfigResult ParseFileConfig(std::string_view json_text,
                                 size_t max_bytes = 1 << 20);

// Reads the file (rejecting symlinks) and parses it.
FileConfigResult LoadFileConfig(const std::string& path);

}  // namespace lykuro::nie
