#include <csignal>
#include <cstdio>
#include <cstring>

#include "core/engine/config.h"
#include "observability/log.h"

#ifdef LYKURO_HAVE_GRPC
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "api/server/grpc_server.h"
#endif

namespace {

constexpr const char kVersion[] = "1.0.0";

void PrintUsage() {
    std::printf(
        "lykuro-native-engine %s\n"
        "\n"
        "Usage: native-engine --config <path> [--version] [--help]\n"
        "\n"
        "Runs the Lykuro Native Inference Engine with the given JSON\n"
        "configuration (spec LYK-NIE-SD-001 §26).\n",
        kVersion);
}

#ifdef LYKURO_HAVE_GRPC

std::atomic<bool> g_stop{false};

void HandleSignal(int) { g_stop = true; }

int RunServer(const lykuro::nie::FileConfig& config) {
    using namespace lykuro::nie;

    Logger::Get().SetLevel(config.log_level == "debug"  ? LogLevel::kDebug
                           : config.log_level == "warn" ? LogLevel::kWarn
                           : config.log_level == "error"
                               ? LogLevel::kError
                               : LogLevel::kInfo);

    ServerConfig server_config;
    server_config.listen_address =
        config.listen_address + ":" + std::to_string(config.grpc_port);
    server_config.control_identities = config.control_identities;
    server_config.data_identities = config.data_identities;
    server_config.engine.max_sequences = config.max_sequences;
    server_config.engine.scheduler.max_queue = config.max_queue;
    server_config.engine.max_output_tokens_limit = config.max_output_tokens;
    server_config.engine.max_input_bytes = config.max_input_bytes;
    server_config.metrics_enabled = config.metrics_enabled;
    server_config.metrics_port = config.metrics_port;
    server_config.load_options.allow_unsigned_dev = config.allow_unsigned_dev;
    for (const std::string& hex : config.trusted_signing_keys_hex) {
        std::array<uint8_t, 32> key;
        Status s = ParsePublicKeyHex(hex, key);
        if (!s.ok()) {
            std::fprintf(stderr, "invalid trusted signing key in config\n");
            return 1;
        }
        server_config.load_options.trusted_keys.keys.push_back(key);
    }

    if (config.mtls_required) {
        Status s = MakeMtlsServerCredentials(
            config.server_cert_path, config.server_key_path,
            config.client_ca_path, server_config.credentials);
        if (!s.ok()) {
            std::fprintf(stderr, "mtls setup failed: %s\n",
                         s.message().c_str());
            return 1;
        }
    } else {
        // Development only; production configs must keep mtls_required.
        server_config.credentials = grpc::InsecureServerCredentials();
        Logger::Get().Log(LogLevel::kWarn, "main", "mtls_disabled");
    }

    EngineServer server(std::move(server_config));
    int port = 0;
    uint16_t metrics_port = 0;
    Status s = server.Start(&port, &metrics_port);
    if (!s.ok()) {
        std::fprintf(stderr, "server start failed: %s\n",
                     s.message().c_str());
        return 1;
    }
    Logger::Get().Log(LogLevel::kInfo, "main", "server_started",
                      {{"engine_id", config.engine_id}},
                      {{"grpc_port", port}, {"metrics_port", metrics_port}});

    if (!config.artifact_path.empty()) {
        s = server.LoadModel(config.artifact_path);
        if (!s.ok()) {
            // Refuse to run without the configured model (fail fast for
            // the Model Manager to observe).
            std::fprintf(stderr, "model load failed: %s\n",
                         s.message().c_str());
            server.Shutdown();
            return 1;
        }
        Logger::Get().Log(LogLevel::kInfo, "main", "model_loaded");
    }

    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    Logger::Get().Log(LogLevel::kInfo, "main", "shutdown_requested");
    server.Shutdown();
    return 0;
}

#endif  // LYKURO_HAVE_GRPC

}  // namespace

int main(int argc, char** argv) {
    const char* config_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            std::printf("%s\n", kVersion);
            return 0;
        }
        if (std::strcmp(argv[i], "--help") == 0) {
            PrintUsage();
            return 0;
        }
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
            continue;
        }
        std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
        return 2;
    }
    if (config_path == nullptr) {
        PrintUsage();
        return 2;
    }

    lykuro::nie::FileConfigResult config =
        lykuro::nie::LoadFileConfig(config_path);
    if (!config.status.ok()) {
        std::fprintf(stderr, "config error: %s\n",
                     config.status.message().c_str());
        return 1;
    }

#ifdef LYKURO_HAVE_GRPC
    return RunServer(config.config);
#else
    std::fprintf(stderr,
                 "this build has no gRPC transport; rebuild with "
                 "-DLYKURO_ENABLE_GRPC=ON\n");
    return 1;
#endif
}
