#include "api/server/grpc_server.h"

#include <chrono>
#include <fstream>
#include <sstream>

#include "lykuro/nie/v1/control.grpc.pb.h"
#include "lykuro/nie/v1/data.grpc.pb.h"

#ifdef LYKURO_HAVE_CUDA
#include "backends/cuda/qwen_cuda_model.h"
#endif
#ifdef LYKURO_HAVE_METAL
#include "backends/metal/qwen_metal_model.h"
#endif

namespace lykuro::nie {

namespace pb = ::lykuro::nie::v1;

namespace {

pb::ErrorCode ToProtoCode(ErrorCode code) {
    switch (code) {
        case ErrorCode::kOk: return pb::ERROR_CODE_UNSPECIFIED;
        case ErrorCode::kInvalidRequest: return pb::ERROR_CODE_INVALID_REQUEST;
        case ErrorCode::kAuthenticationFailed:
            return pb::ERROR_CODE_AUTHENTICATION_FAILED;
        case ErrorCode::kModelNotLoaded:
            return pb::ERROR_CODE_MODEL_NOT_LOADED;
        case ErrorCode::kUnsupportedModel:
            return pb::ERROR_CODE_UNSUPPORTED_MODEL;
        case ErrorCode::kArtifactVerificationFailed:
            return pb::ERROR_CODE_ARTIFACT_VERIFICATION_FAILED;
        case ErrorCode::kContextLengthExceeded:
            return pb::ERROR_CODE_CONTEXT_LENGTH_EXCEEDED;
        case ErrorCode::kResourceExhausted:
            return pb::ERROR_CODE_RESOURCE_EXHAUSTED;
        case ErrorCode::kCapacityExhausted:
            return pb::ERROR_CODE_CAPACITY_EXHAUSTED;
        case ErrorCode::kDeadlineRejected:
            return pb::ERROR_CODE_DEADLINE_REJECTED;
        case ErrorCode::kDeadlineExceeded:
            return pb::ERROR_CODE_DEADLINE_EXCEEDED;
        case ErrorCode::kRequestCancelled:
            return pb::ERROR_CODE_REQUEST_CANCELLED;
        case ErrorCode::kEngineDraining:
            return pb::ERROR_CODE_ENGINE_DRAINING;
        case ErrorCode::kGpuOom: return pb::ERROR_CODE_GPU_OOM;
        case ErrorCode::kGpuUnhealthy: return pb::ERROR_CODE_GPU_UNHEALTHY;
        case ErrorCode::kInferenceFailed:
            return pb::ERROR_CODE_INFERENCE_FAILED;
        case ErrorCode::kStreamConsumerSlow:
            return pb::ERROR_CODE_STREAM_CONSUMER_SLOW;
        case ErrorCode::kInternalError: return pb::ERROR_CODE_INTERNAL_ERROR;
    }
    return pb::ERROR_CODE_INTERNAL_ERROR;
}

void FillProtoError(const Status& status, const std::string& request_id,
                    pb::EngineError* out) {
    out->set_request_id(request_id);
    out->set_code(ToProtoCode(status.code()));
    out->set_message(status.message());
    out->set_retryable(status.retryable());
    out->set_component(status.component());
    for (const auto& [key, value] : status.details()) {
        (*out->mutable_numeric_details())[key] = value;
    }
}

pb::FinishReason ToProtoFinish(FinishReason reason) {
    switch (reason) {
        case FinishReason::kStop: return pb::FINISH_REASON_STOP;
        case FinishReason::kLength: return pb::FINISH_REASON_LENGTH;
        case FinishReason::kCancelled: return pb::FINISH_REASON_CANCELLED;
        case FinishReason::kDeadline: return pb::FINISH_REASON_DEADLINE;
        case FinishReason::kError: return pb::FINISH_REASON_ERROR;
    }
    return pb::FINISH_REASON_ERROR;
}

Status FromProtoRequest(const pb::GenerateRequest& proto,
                        InferenceRequest& out) {
    if (proto.request_id().empty()) {
        return Status(ErrorCode::kInvalidRequest, "request_id required",
                      "api");
    }
    out.request_id = proto.request_id();
    out.tenant_scope = proto.tenant_scope();
    out.project_scope = proto.project_scope();
    out.priority = proto.scheduling().priority();
    out.deadline_unix_ms = proto.scheduling().deadline_unix_ms();

    const pb::GenerateInput& input = proto.input();
    const bool has_messages = input.messages_size() > 0;
    const bool has_prompt = !input.prompt().empty();
    if (has_messages == has_prompt) {
        return Status(ErrorCode::kInvalidRequest,
                      "exactly one of messages or prompt required", "api");
    }
    if (has_prompt) {
        out.messages.push_back({Role::kUser, input.prompt()});
    } else {
        for (const pb::Message& m : input.messages()) {
            Role role;
            switch (m.role()) {
                case pb::ROLE_SYSTEM: role = Role::kSystem; break;
                case pb::ROLE_DEVELOPER: role = Role::kDeveloper; break;
                case pb::ROLE_USER: role = Role::kUser; break;
                case pb::ROLE_ASSISTANT: role = Role::kAssistant; break;
                case pb::ROLE_TOOL: role = Role::kTool; break;
                default:
                    return Status(ErrorCode::kInvalidRequest,
                                  "message role not in allowlist", "api");
            }
            out.messages.push_back({role, m.content()});
        }
    }

    const pb::GenerationParams& gen = proto.generation();
    out.max_output_tokens = gen.max_output_tokens();
    out.sampling.temperature = gen.temperature();
    out.sampling.top_p = gen.top_p() > 0 ? gen.top_p() : 1.0f;
    out.sampling.top_k = gen.top_k();
    out.sampling.seed = uint64_t(gen.seed());
    for (const std::string& s : gen.stop()) {
        out.stop_sequences.push_back(s);
    }
    out.allow_prefix_cache = proto.cache().allow_prefix_cache();
    return Status::Ok();
}

// Peer identity check against an allowlist of certificate common names.
// An empty allowlist disables the check (development only).
grpc::Status Authorize(grpc::ServerContext* context,
                       const std::vector<std::string>& allowed) {
    if (allowed.empty()) return grpc::Status::OK;
    auto auth = context->auth_context();
    if (auth) {
        for (const grpc::string_ref& value :
             auth->FindPropertyValues("x509_common_name")) {
            std::string cn(value.data(), value.size());
            for (const std::string& id : allowed) {
                if (cn == id) return grpc::Status::OK;
            }
        }
    }
    return grpc::Status(grpc::StatusCode::PERMISSION_DENIED,
                        "authentication_failed");
}

Status ReadPemFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return Status(ErrorCode::kInternalError,
                      "credential file cannot be opened", "api");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return Status::Ok();
}

}  // namespace

Status MakeMtlsServerCredentials(
    const std::string& server_cert_pem_path,
    const std::string& server_key_pem_path,
    const std::string& client_ca_pem_path,
    std::shared_ptr<grpc::ServerCredentials>& out) {
    std::string cert, key, ca;
    Status s = ReadPemFile(server_cert_pem_path, cert);
    if (s.ok()) s = ReadPemFile(server_key_pem_path, key);
    if (s.ok()) s = ReadPemFile(client_ca_pem_path, ca);
    if (!s.ok()) return s;

    grpc::SslServerCredentialsOptions options(
        GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY);
    options.pem_root_certs = ca;
    options.pem_key_cert_pairs.push_back({key, cert});
    out = grpc::SslServerCredentials(options);
    if (!out) {
        return Status(ErrorCode::kInternalError,
                      "mtls credentials rejected", "api");
    }
    return Status::Ok();
}

// ---------------------------------------------------------------- Data API

class EngineServer::DataServiceImpl final : public pb::DataService::Service {
public:
    explicit DataServiceImpl(EngineServer* server) : server_(server) {}

    grpc::Status Generate(grpc::ServerContext* context,
                          const pb::GenerateRequest* request,
                          pb::GenerateResponse* response) override {
        grpc::Status auth =
            Authorize(context, server_->config_.data_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        if (engine == nullptr) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "model_not_loaded");
        }
        InferenceRequest req;
        Status s = FromProtoRequest(*request, req);
        if (!s.ok()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                std::string(ErrorCodeName(s.code())));
        }
        auto submit = engine->Submit(req);
        if (!submit.status.ok()) {
            return grpc::Status(
                grpc::StatusCode::FAILED_PRECONDITION,
                std::string(ErrorCodeName(submit.status.code())));
        }

        response->set_request_id(req.request_id);
        std::string text;
        while (auto event = submit.events->Pop()) {
            switch (event->kind) {
                case StreamEvent::Kind::kOutputTextDelta:
                    text += event->text;
                    break;
                case StreamEvent::Kind::kCompleted: {
                    response->set_output_text(text);
                    response->set_finish_reason(
                        ToProtoFinish(event->finish_reason));
                    auto* usage = response->mutable_usage();
                    usage->set_input_tokens(event->usage.input_tokens);
                    usage->set_output_tokens(event->usage.output_tokens);
                    usage->set_total_tokens(event->usage.total_tokens());
                    break;
                }
                case StreamEvent::Kind::kError:
                    return grpc::Status(
                        grpc::StatusCode::ABORTED,
                        std::string(ErrorCodeName(event->error.code())));
                default:
                    break;
            }
        }
        return grpc::Status::OK;
    }

    grpc::Status GenerateStream(
        grpc::ServerContext* context, const pb::GenerateRequest* request,
        grpc::ServerWriter<pb::StreamEvent>* writer) override {
        grpc::Status auth =
            Authorize(context, server_->config_.data_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        if (engine == nullptr) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "model_not_loaded");
        }
        InferenceRequest req;
        Status s = FromProtoRequest(*request, req);
        if (!s.ok()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                std::string(ErrorCodeName(s.code())));
        }
        auto submit = engine->Submit(req);
        if (!submit.status.ok()) {
            return grpc::Status(
                grpc::StatusCode::FAILED_PRECONDITION,
                std::string(ErrorCodeName(submit.status.code())));
        }

        while (auto event = submit.events->Pop()) {
            if (context->IsCancelled()) {
                engine->Cancel(req.request_id);
                return grpc::Status::CANCELLED;
            }
            pb::StreamEvent proto;
            proto.set_request_id(event->request_id);
            proto.set_sequence(event->sequence);
            proto.set_timestamp_unix_ms(event->timestamp_unix_ms);
            switch (event->kind) {
                case StreamEvent::Kind::kStarted:
                    proto.mutable_started();
                    break;
                case StreamEvent::Kind::kOutputTextDelta:
                    proto.mutable_output_text_delta()->set_text(event->text);
                    break;
                case StreamEvent::Kind::kUsage: {
                    auto* usage = proto.mutable_usage()->mutable_usage();
                    usage->set_input_tokens(event->usage.input_tokens);
                    usage->set_output_tokens(event->usage.output_tokens);
                    usage->set_total_tokens(event->usage.total_tokens());
                    break;
                }
                case StreamEvent::Kind::kCompleted: {
                    auto* completed = proto.mutable_completed();
                    completed->set_finish_reason(
                        ToProtoFinish(event->finish_reason));
                    auto* usage = completed->mutable_usage();
                    usage->set_input_tokens(event->usage.input_tokens);
                    usage->set_output_tokens(event->usage.output_tokens);
                    usage->set_total_tokens(event->usage.total_tokens());
                    break;
                }
                case StreamEvent::Kind::kError:
                    FillProtoError(event->error, event->request_id,
                                   proto.mutable_error());
                    break;
            }
            if (!writer->Write(proto)) {
                engine->Cancel(req.request_id);
                return grpc::Status::CANCELLED;
            }
        }
        return grpc::Status::OK;
    }

    grpc::Status Cancel(grpc::ServerContext* context,
                        const pb::CancelRequest* request,
                        pb::CancelResponse* response) override {
        grpc::Status auth =
            Authorize(context, server_->config_.data_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        response->set_found(engine != nullptr &&
                            engine->Cancel(request->request_id()));
        return grpc::Status::OK;
    }

    grpc::Status CountTokens(grpc::ServerContext* context,
                             const pb::CountTokensRequest* request,
                             pb::CountTokensResponse* response) override {
        grpc::Status auth =
            Authorize(context, server_->config_.data_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        if (engine == nullptr) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "model_not_loaded");
        }
        pb::GenerateRequest wrapper;
        wrapper.set_request_id("count_tokens");
        *wrapper.mutable_input() = request->input();
        wrapper.mutable_generation()->set_max_output_tokens(1);
        InferenceRequest req;
        Status s = FromProtoRequest(wrapper, req);
        uint32_t tokens = 0;
        if (s.ok()) s = engine->CountTokens(req.messages, tokens);
        if (!s.ok()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                std::string(ErrorCodeName(s.code())));
        }
        response->set_input_tokens(tokens);
        return grpc::Status::OK;
    }

    grpc::Status GetCapabilities(grpc::ServerContext* context,
                                 const pb::GetCapabilitiesRequest*,
                                 pb::GetCapabilitiesResponse* out) override {
        grpc::Status auth =
            Authorize(context, server_->config_.data_identities);
        if (!auth.ok()) return auth;
        out->set_engine_version("0.1.0");
        out->set_engine_abi("nie_abi_1");
        out->set_streaming(true);
        out->set_logprobs(false);
        out->set_prefix_cache(false);
        out->set_multi_gpu(false);
        InferenceEngine* engine = server_->engine();
        const ModelManifest* manifest = server_->manifest();
        if (engine != nullptr && manifest != nullptr) {
            out->set_model_instance_id("mi_" + manifest->artifact_id);
            out->set_model_family(manifest->model_family);
            out->set_architecture(manifest->architecture);
            out->set_max_context_tokens(manifest->max_context_tokens);
            out->add_supported_precisions(manifest->precision);
        }
        return grpc::Status::OK;
    }

private:
    EngineServer* server_;
};

// -------------------------------------------------------------- Control API

class EngineServer::ControlServiceImpl final
    : public pb::ControlService::Service {
public:
    explicit ControlServiceImpl(EngineServer* server) : server_(server) {}

    grpc::Status LoadModel(grpc::ServerContext* context,
                           const pb::LoadModelRequest* request,
                           pb::LoadModelResponse* response) override {
        grpc::Status auth =
            Authorize(context, server_->config_.control_identities);
        if (!auth.ok()) return auth;
        Status s = server_->LoadModel(request->artifact_path());
        if (!s.ok()) {
            response->set_state(pb::MODEL_STATE_FAILED);
            FillProtoError(s, "", response->mutable_error());
            return grpc::Status::OK;
        }
        const ModelManifest* manifest = server_->manifest();
        if (!request->expected_artifact_id().empty() && manifest != nullptr &&
            manifest->artifact_id != request->expected_artifact_id()) {
            server_->UnloadModel();
            response->set_state(pb::MODEL_STATE_FAILED);
            FillProtoError(Status(ErrorCode::kArtifactVerificationFailed,
                                  "artifact id mismatch", "control"),
                           "", response->mutable_error());
            return grpc::Status::OK;
        }
        if (manifest != nullptr) {
            response->set_model_instance_id("mi_" + manifest->artifact_id);
        }
        response->set_state(pb::MODEL_STATE_READY);
        return grpc::Status::OK;
    }

    grpc::Status UnloadModel(grpc::ServerContext* context,
                             const pb::UnloadModelRequest*,
                             pb::UnloadModelResponse* response) override {
        grpc::Status auth =
            Authorize(context, server_->config_.control_identities);
        if (!auth.ok()) return auth;
        server_->UnloadModel();
        response->set_state(pb::MODEL_STATE_UNLOADED);
        return grpc::Status::OK;
    }

    grpc::Status Drain(grpc::ServerContext* context, const pb::DrainRequest*,
                       pb::DrainResponse* response) override {
        grpc::Status auth =
            Authorize(context, server_->config_.control_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        if (engine != nullptr) {
            engine->StartDrain();
            response->set_active_requests(engine->active_sequences());
            response->set_queued_requests(uint32_t(engine->queue_depth()));
        }
        return grpc::Status::OK;
    }

    grpc::Status Resume(grpc::ServerContext* context, const pb::ResumeRequest*,
                        pb::ResumeResponse* response) override {
        grpc::Status auth =
            Authorize(context, server_->config_.control_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        if (engine != nullptr) engine->Resume();
        response->set_accepting(engine != nullptr);
        return grpc::Status::OK;
    }

    grpc::Status GetModelStatus(grpc::ServerContext* context,
                                const pb::GetModelStatusRequest*,
                                pb::GetModelStatusResponse* out) override {
        grpc::Status auth =
            Authorize(context, server_->config_.control_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        const ModelManifest* manifest = server_->manifest();
        if (engine == nullptr || manifest == nullptr) {
            out->set_state(pb::MODEL_STATE_UNLOADED);
            out->set_engine_state(pb::ENGINE_STATE_IDLE);
            return grpc::Status::OK;
        }
        out->set_model_instance_id("mi_" + manifest->artifact_id);
        out->set_state(pb::MODEL_STATE_READY);
        out->set_engine_state(pb::ENGINE_STATE_READY);
        out->set_artifact_id(manifest->artifact_id);
        out->set_architecture(manifest->architecture);
        return grpc::Status::OK;
    }

    grpc::Status GetCapacity(grpc::ServerContext* context,
                             const pb::GetCapacityRequest*,
                             pb::GetCapacityResponse* out) override {
        grpc::Status auth =
            Authorize(context, server_->config_.control_identities);
        if (!auth.ok()) return auth;
        InferenceEngine* engine = server_->engine();
        const ModelManifest* manifest = server_->manifest();
        if (engine != nullptr && manifest != nullptr) {
            out->set_active_sequences(engine->active_sequences());
            out->set_queued_requests(uint32_t(engine->queue_depth()));
            out->set_loaded_model_instance_id("mi_" + manifest->artifact_id);
            out->set_context_limit_tokens(manifest->max_context_tokens);
            out->set_gpu_healthy(true);
        }
        return grpc::Status::OK;
    }

    grpc::Status GetManifest(grpc::ServerContext* context,
                             const pb::GetManifestRequest*,
                             pb::GetManifestResponse* out) override {
        grpc::Status auth =
            Authorize(context, server_->config_.control_identities);
        if (!auth.ok()) return auth;
        const ModelManifest* manifest = server_->manifest();
        if (manifest == nullptr) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "model_not_loaded");
        }
        out->set_artifact_id(manifest->artifact_id);
        out->set_model_family(manifest->model_family);
        out->set_architecture(manifest->architecture);
        out->set_model_version(manifest->model_version);
        out->set_weight_format(manifest->weight_format);
        out->set_precision(manifest->precision);
        out->set_max_context_tokens(manifest->max_context_tokens);
        out->set_tokenizer_type(manifest->tokenizer_type);
        out->set_chat_template_id(manifest->chat_template_id);
        out->set_license_review_id(manifest->license_review_id);
        for (const std::string& cp : manifest->certified_profiles) {
            out->add_certified_profile_ids(cp);
        }
        return grpc::Status::OK;
    }

private:
    EngineServer* server_;
};

// ------------------------------------------------------------- EngineServer

EngineServer::EngineServer(ServerConfig config)
    : config_(std::move(config)),
      data_service_(std::make_unique<DataServiceImpl>(this)),
      control_service_(std::make_unique<ControlServiceImpl>(this)) {}

EngineServer::~EngineServer() { Shutdown(); }

Status EngineServer::Start(int* port_out, uint16_t* metrics_port_out) {
    if (config_.metrics_enabled) {
        metrics_.RegisterGauge("nie_queue_depth",
                               "Requests waiting for admission", [this] {
                                   InferenceEngine* e = engine();
                                   return e ? int64_t(e->queue_depth()) : 0;
                               });
        metrics_.RegisterGauge("nie_active_sequences",
                               "Sequences currently decoding", [this] {
                                   InferenceEngine* e = engine();
                                   return e ? int64_t(e->active_sequences())
                                            : 0;
                               });
        metrics_http_ = std::make_unique<MetricsHttpServer>(metrics_);
        uint16_t bound = 0;
        if (!metrics_http_->Start(config_.metrics_port, &bound)) {
            return Status(ErrorCode::kInternalError,
                          "metrics endpoint failed to bind", "api");
        }
        if (metrics_port_out != nullptr) *metrics_port_out = bound;
    }

    if (!config_.credentials) {
        return Status(ErrorCode::kInternalError,
                      "server credentials are required", "api");
    }
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort(config_.listen_address, config_.credentials,
                             &port);
    builder.RegisterService(data_service_.get());
    builder.RegisterService(control_service_.get());
    server_ = builder.BuildAndStart();
    if (!server_) {
        return Status(ErrorCode::kInternalError, "server failed to start",
                      "api");
    }
    if (port_out != nullptr) *port_out = port;
    running_ = true;
    worker_ = std::thread([this] { WorkerLoop(); });
    return Status::Ok();
}

void EngineServer::WorkerLoop() {
    while (running_) {
        InferenceEngine* engine = nullptr;
        {
            std::lock_guard<std::mutex> lock(model_mutex_);
            engine = engine_.get();
        }
        bool worked = engine != nullptr && engine->Step();
        if (!worked) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void EngineServer::Shutdown() {
    if (metrics_http_) {
        metrics_http_->Stop();
        metrics_http_.reset();
    }
    if (server_) {
        server_->Shutdown();
    }
    running_ = false;
    if (worker_.joinable()) worker_.join();
    if (server_) {
        server_->Wait();
        server_.reset();
    }
    UnloadModel();
}

Status EngineServer::LoadModel(const std::string& artifact_dir) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (model_loaded_) {
        return Status(ErrorCode::kInvalidRequest,
                      "a model is already loaded; unload first", "control");
    }
    ArtifactLoadResult loaded =
        LoadArtifact(artifact_dir, config_.load_options);
    if (!loaded.status.ok()) return loaded.status;

    // Pick the serving backend. The CPU reference model built during
    // artifact validation doubles as the smoke-test oracle; for CUDA it
    // is replaced by the device model before serving.
    std::unique_ptr<GenerativeModel> serving_model =
        std::move(loaded.artifact.model);
    if (config_.hardware_backend == "cuda") {
#ifdef LYKURO_HAVE_CUDA
        auto cuda = QwenCudaModel::Load(loaded.artifact.manifest,
                                        *loaded.artifact.weights,
                                        config_.device_id);
        if (!cuda.status.ok()) return cuda.status;
        serving_model = std::move(cuda.model);
#else
        return Status(ErrorCode::kUnsupportedModel,
                      "this build has no CUDA backend", "control");
#endif
    } else if (config_.hardware_backend == "metal") {
#ifdef LYKURO_HAVE_METAL
        auto metal = QwenMetalModel::Load(loaded.artifact.manifest,
                                          *loaded.artifact.weights);
        if (!metal.status.ok()) return metal.status;
        serving_model = std::move(metal.model);
#else
        return Status(ErrorCode::kUnsupportedModel,
                      "this build has no Metal backend", "control");
#endif
    } else if (config_.hardware_backend != "cpu") {
        return Status(ErrorCode::kInvalidRequest,
                      "unknown hardware backend", "control");
    }

    manifest_ = std::move(loaded.artifact.manifest);
    weights_ = std::move(loaded.artifact.weights);
    EngineConfig engine_config = config_.engine;
    if (config_.metrics_enabled) {
        engine_config.metrics = &metrics_;
    }
    engine_ = std::make_unique<InferenceEngine>(
        std::move(serving_model), std::move(loaded.artifact.tokenizer),
        engine_config);
    model_loaded_ = true;
    return Status::Ok();
}

Status EngineServer::UnloadModel() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    // Unload ordering (spec §12.5): the engine destructor cancels active
    // sequences and releases KV before the weights unmap below.
    engine_.reset();
    weights_.reset();
    manifest_ = ModelManifest{};
    model_loaded_ = false;
    return Status::Ok();
}

InferenceEngine* EngineServer::engine() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return engine_.get();
}

const ModelManifest* EngineServer::manifest() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    return model_loaded_ ? &manifest_ : nullptr;
}

}  // namespace lykuro::nie
