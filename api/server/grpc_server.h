#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>

#include "core/engine/engine.h"
#include "model/loader/artifact_loader.h"

namespace lykuro::nie {

struct ServerConfig {
    // Binds loopback by default; production fronts this with mTLS creds
    // (spec §9.1). Never a public interface.
    std::string listen_address = "127.0.0.1:0";
    std::shared_ptr<grpc::ServerCredentials> credentials;  // required

    // Service identity allowlists, matched against the peer certificate
    // common name (spec §8.2: Control API is Model Manager only; Gateway
    // identities are rejected there). Empty list = no identity check
    // (development / insecure credentials only).
    std::vector<std::string> control_identities;
    std::vector<std::string> data_identities;

    EngineConfig engine;
    ArtifactLoadOptions load_options;

    // Serving backend: "cpu" (reference) or "cuda" (requires a build with
    // LYKURO_ENABLE_CUDA and an explicit device id, spec §17.3).
    std::string hardware_backend = "cpu";
    int device_id = 0;

    // Loopback metrics endpoint (GET /metrics). Disabled when false.
    bool metrics_enabled = false;
    uint16_t metrics_port = 0;  // 0 = ephemeral
};

// Builds mTLS server credentials: server cert/key plus the client CA,
// with client certificate REQUIRED and verified.
Status MakeMtlsServerCredentials(
    const std::string& server_cert_pem_path,
    const std::string& server_key_pem_path,
    const std::string& client_ca_pem_path,
    std::shared_ptr<grpc::ServerCredentials>& out);

// Hosts one InferenceEngine behind the Data/Control gRPC services and
// drives it from a single worker thread (spec §7.1 GPU worker analogue
// for the CPU reference backend).
class EngineServer {
public:
    explicit EngineServer(ServerConfig config);
    ~EngineServer();

    // Starts the gRPC server. Returns the bound port via `port_out`, and
    // the metrics port via `metrics_port_out` when metrics are enabled.
    Status Start(int* port_out = nullptr,
                 uint16_t* metrics_port_out = nullptr);
    void Shutdown();

    // Direct (in-process) access used by the Control service and tests.
    Status LoadModel(const std::string& artifact_dir);
    Status UnloadModel();
    InferenceEngine* engine();
    const ModelManifest* manifest() const;

private:
    class DataServiceImpl;
    class ControlServiceImpl;

    void WorkerLoop();

    ServerConfig config_;
    std::unique_ptr<DataServiceImpl> data_service_;
    std::unique_ptr<ControlServiceImpl> control_service_;
    std::unique_ptr<grpc::Server> server_;

    mutable std::mutex model_mutex_;
    std::unique_ptr<InferenceEngine> engine_;
    ModelManifest manifest_;
    bool model_loaded_ = false;
    // Weight mapping + tokenizer storage kept alive alongside the engine.
    std::unique_ptr<SafetensorsFile> weights_;

    MetricsRegistry metrics_;
    std::unique_ptr<MetricsHttpServer> metrics_http_;

    std::atomic<bool> running_{false};
    std::thread worker_;
};

}  // namespace lykuro::nie
