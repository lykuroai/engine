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
    // MVP binds loopback only; production fronts this with mTLS creds
    // (spec §9.1). Never a public interface.
    std::string listen_address = "127.0.0.1:0";
    std::shared_ptr<grpc::ServerCredentials> credentials;  // required
    EngineConfig engine;
    ArtifactLoadOptions load_options;
};

// Hosts one InferenceEngine behind the Data/Control gRPC services and
// drives it from a single worker thread (spec §7.1 GPU worker analogue
// for the CPU reference backend).
class EngineServer {
public:
    explicit EngineServer(ServerConfig config);
    ~EngineServer();

    // Starts the gRPC server. Returns the bound port via `port_out`.
    Status Start(int* port_out = nullptr);
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

    std::atomic<bool> running_{false};
    std::thread worker_;
};

}  // namespace lykuro::nie
