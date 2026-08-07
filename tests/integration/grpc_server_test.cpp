#include "api/server/grpc_server.h"

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "lykuro/nie/v1/control.grpc.pb.h"
#include "lykuro/nie/v1/data.grpc.pb.h"
#include "security/sha256.h"
#include "tests/testutil/tiny_model.h"
#include "tests/unit/tokenizer_fixture.h"

namespace lykuro::nie {
namespace {

namespace pb = ::lykuro::nie::v1;
namespace fs = std::filesystem;

std::string FileSha256(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::string data((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    return Sha256::HexDigest(data.data(), data.size());
}

// Builds a complete on-disk artifact: manifest.json with real digests,
// config/tokenizer.json, weights/model.safetensors.
class GrpcServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        artifact_dir_ = fs::path(testing::TempDir()) / "nie_artifact";
        fs::remove_all(artifact_dir_);
        fs::create_directories(artifact_dir_ / "config");
        fs::create_directories(artifact_dir_ / "weights");

        testutil::TinyModelSpec spec;
        spec.vocab_size = 303;
        spec.eos_token_id = 302;
        spec.max_context_tokens = 128;
        std::string tmp_weights =
            testutil::WriteTinyWeights(spec, "grpc_tiny.safetensors");
        fs::rename(tmp_weights, artifact_dir_ / "weights" /
                                    "model.safetensors");

        {
            std::ofstream tok(artifact_dir_ / "config" / "tokenizer.json");
            tok << testfixture::SmallTokenizerConfig();
        }

        auto file_entry = [&](const std::string& rel) {
            fs::path p = artifact_dir_ / rel;
            return "{\"path\": \"" + rel + "\", \"sha256\": \"" +
                   FileSha256(p) + "\", \"size_bytes\": " +
                   std::to_string(fs::file_size(p)) + "}";
        };
        std::ofstream manifest(artifact_dir_ / "manifest.json");
        manifest << R"({
          "schema_version": "1",
          "artifact_id": "ma_qwen_grpc",
          "model_family": "qwen",
          "architecture": "approved_qwen_decoder_v1",
          "model_version": "tiny-grpc",
          "weight_format": "safetensors",
          "precision": "fp32",
          "vocab_size": 303,
          "hidden_size": 16,
          "num_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 4,
          "max_context_tokens": 128,
          "intermediate_size": 32,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "tie_word_embeddings": true,
          "eos_token_ids": [302],
          "tokenizer_type": "approved_qwen_tokenizer_v1",
          "chat_template_id": "qwen_chat_v1",
          "files": [)"
                 << file_entry("weights/model.safetensors") << ",\n"
                 << file_entry("config/tokenizer.json") << R"(],
          "license_review_id": "lic_test_01",
          "certified_profiles": [],
          "created_at": "2026-08-07T00:00:00Z"
        })";
        manifest.close();

        ServerConfig config;
        config.credentials = grpc::InsecureServerCredentials();
        config.load_options.allow_unsigned_dev = true;
        server_ = std::make_unique<EngineServer>(config);
        int port = 0;
        ASSERT_TRUE(server_->Start(&port).ok());
        channel_ = grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                       grpc::InsecureChannelCredentials());
        data_stub_ = pb::DataService::NewStub(channel_);
        control_stub_ = pb::ControlService::NewStub(channel_);
    }

    void TearDown() override {
        server_->Shutdown();
        fs::remove_all(artifact_dir_);
    }

    grpc::Status LoadViaControl(pb::LoadModelResponse* response) {
        grpc::ClientContext ctx;
        pb::LoadModelRequest req;
        req.set_artifact_path(artifact_dir_.string());
        req.set_expected_artifact_id("ma_qwen_grpc");
        return control_stub_->LoadModel(&ctx, req, response);
    }

    static pb::GenerateRequest MakeGenerateRequest(const std::string& id) {
        pb::GenerateRequest req;
        req.set_request_id(id);
        req.set_tenant_scope("tn_test");
        auto* msg = req.mutable_input()->add_messages();
        msg->set_role(pb::ROLE_USER);
        msg->set_content("hello");
        req.mutable_generation()->set_max_output_tokens(5);
        req.mutable_generation()->set_temperature(0.0f);
        return req;
    }

    fs::path artifact_dir_;
    std::unique_ptr<EngineServer> server_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<pb::DataService::Stub> data_stub_;
    std::unique_ptr<pb::ControlService::Stub> control_stub_;
};

TEST_F(GrpcServerTest, GenerateFailsBeforeModelLoad) {
    grpc::ClientContext ctx;
    pb::GenerateResponse response;
    grpc::Status s =
        data_stub_->Generate(&ctx, MakeGenerateRequest("req_early"),
                             &response);
    EXPECT_EQ(s.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_F(GrpcServerTest, LoadModelThenGenerate) {
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());
    ASSERT_EQ(load.state(), pb::MODEL_STATE_READY)
        << load.error().message();
    EXPECT_EQ(load.model_instance_id(), "mi_ma_qwen_grpc");

    grpc::ClientContext ctx;
    pb::GenerateResponse response;
    grpc::Status s =
        data_stub_->Generate(&ctx, MakeGenerateRequest("req_1"), &response);
    ASSERT_TRUE(s.ok()) << s.error_message();
    EXPECT_EQ(response.request_id(), "req_1");
    EXPECT_EQ(response.finish_reason(), pb::FINISH_REASON_LENGTH);
    EXPECT_EQ(response.usage().output_tokens(), 5u);
    EXPECT_GT(response.usage().input_tokens(), 0u);
}

TEST_F(GrpcServerTest, GenerateStreamDeliversOrderedEvents) {
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());
    ASSERT_EQ(load.state(), pb::MODEL_STATE_READY);

    grpc::ClientContext ctx;
    auto reader =
        data_stub_->GenerateStream(&ctx, MakeGenerateRequest("req_s"));
    std::vector<pb::StreamEvent> events;
    pb::StreamEvent event;
    while (reader->Read(&event)) events.push_back(event);
    ASSERT_TRUE(reader->Finish().ok());

    ASSERT_GE(events.size(), 3u);
    EXPECT_TRUE(events.front().has_started());
    EXPECT_TRUE(events.back().has_completed());
    EXPECT_EQ(events.back().completed().finish_reason(),
              pb::FINISH_REASON_LENGTH);
    for (size_t i = 0; i < events.size(); ++i) {
        EXPECT_EQ(events[i].sequence(), i);
        EXPECT_EQ(events[i].request_id(), "req_s");
    }
}

TEST_F(GrpcServerTest, CancelIsIdempotentOverRpc) {
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());

    grpc::ClientContext ctx;
    pb::CancelRequest req;
    req.set_request_id("req_unknown");
    pb::CancelResponse response;
    ASSERT_TRUE(data_stub_->Cancel(&ctx, req, &response).ok());
    EXPECT_FALSE(response.found());
}

TEST_F(GrpcServerTest, LoadRejectsWrongExpectedArtifactId) {
    grpc::ClientContext ctx;
    pb::LoadModelRequest req;
    req.set_artifact_path(artifact_dir_.string());
    req.set_expected_artifact_id("ma_other");
    pb::LoadModelResponse response;
    ASSERT_TRUE(control_stub_->LoadModel(&ctx, req, &response).ok());
    EXPECT_EQ(response.state(), pb::MODEL_STATE_FAILED);
}

TEST_F(GrpcServerTest, TamperedWeightsRejected) {
    // Flip one byte in the weights: digest verification must reject.
    fs::path weights = artifact_dir_ / "weights" / "model.safetensors";
    {
        std::fstream f(weights,
                       std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(-1, std::ios::end);
        char c;
        f.seekg(-1, std::ios::end);
        f.get(c);
        f.seekp(-1, std::ios::end);
        f.put(char(c ^ 0x01));
    }
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());
    EXPECT_EQ(load.state(), pb::MODEL_STATE_FAILED);
    EXPECT_EQ(load.error().code(),
              pb::ERROR_CODE_ARTIFACT_VERIFICATION_FAILED);
}

TEST_F(GrpcServerTest, CapabilitiesAndManifestMetadata) {
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());

    {
        grpc::ClientContext ctx;
        pb::GetCapabilitiesResponse caps;
        ASSERT_TRUE(
            data_stub_->GetCapabilities(&ctx, {}, &caps).ok());
        EXPECT_EQ(caps.model_family(), "qwen");
        EXPECT_TRUE(caps.streaming());
        EXPECT_FALSE(caps.multi_gpu());
        EXPECT_EQ(caps.max_context_tokens(), 128u);
    }
    {
        grpc::ClientContext ctx;
        pb::GetManifestResponse manifest;
        ASSERT_TRUE(control_stub_->GetManifest(&ctx, {}, &manifest).ok());
        EXPECT_EQ(manifest.artifact_id(), "ma_qwen_grpc");
        EXPECT_EQ(manifest.chat_template_id(), "qwen_chat_v1");
    }
}

TEST_F(GrpcServerTest, DrainRejectsThenResumeAccepts) {
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());

    {
        grpc::ClientContext ctx;
        pb::DrainResponse response;
        ASSERT_TRUE(control_stub_->Drain(&ctx, {}, &response).ok());
    }
    {
        grpc::ClientContext ctx;
        pb::GenerateResponse response;
        grpc::Status s = data_stub_->Generate(
            &ctx, MakeGenerateRequest("req_drained"), &response);
        EXPECT_EQ(s.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
        EXPECT_EQ(s.error_message(), "engine_draining");
    }
    {
        grpc::ClientContext ctx;
        pb::ResumeResponse response;
        ASSERT_TRUE(control_stub_->Resume(&ctx, {}, &response).ok());
        EXPECT_TRUE(response.accepting());
    }
    {
        grpc::ClientContext ctx;
        pb::GenerateResponse response;
        EXPECT_TRUE(data_stub_
                        ->Generate(&ctx, MakeGenerateRequest("req_after"),
                                   &response)
                        .ok());
    }
}

TEST_F(GrpcServerTest, CountTokensMatchesGenerateUsage) {
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());

    pb::GenerateRequest gen = MakeGenerateRequest("req_count");
    grpc::ClientContext gen_ctx;
    pb::GenerateResponse gen_response;
    ASSERT_TRUE(data_stub_->Generate(&gen_ctx, gen, &gen_response).ok());

    grpc::ClientContext ctx;
    pb::CountTokensRequest req;
    *req.mutable_input() = gen.input();
    pb::CountTokensResponse response;
    ASSERT_TRUE(data_stub_->CountTokens(&ctx, req, &response).ok());
    EXPECT_EQ(response.input_tokens(), gen_response.usage().input_tokens());
    EXPECT_GT(response.input_tokens(), 0u);
}

TEST_F(GrpcServerTest, MetricsEndpointReportsRequestCounters) {
    ServerConfig config;
    config.credentials = grpc::InsecureServerCredentials();
    config.load_options.allow_unsigned_dev = true;
    config.metrics_enabled = true;
    EngineServer metrics_server(config);
    int port = 0;
    uint16_t metrics_port = 0;
    ASSERT_TRUE(metrics_server.Start(&port, &metrics_port).ok());
    ASSERT_GT(metrics_port, 0);
    ASSERT_TRUE(metrics_server.LoadModel(artifact_dir_.string()).ok());

    auto channel =
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                            grpc::InsecureChannelCredentials());
    auto stub = pb::DataService::NewStub(channel);
    grpc::ClientContext ctx;
    pb::GenerateResponse response;
    ASSERT_TRUE(
        stub->Generate(&ctx, MakeGenerateRequest("req_m"), &response).ok());

    // Raw HTTP GET against the loopback metrics port.
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(metrics_port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(
        ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    const char kReq[] = "GET /metrics HTTP/1.1\r\nHost: l\r\n\r\n";
    ::send(fd, kReq, sizeof(kReq) - 1, 0);
    std::string body;
    char buf[8192];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) body.append(buf, n);
    ::close(fd);

    EXPECT_NE(body.find("nie_requests_received_total 1"), std::string::npos)
        << body;
    EXPECT_NE(body.find("nie_requests_completed_total 1"),
              std::string::npos);
    EXPECT_NE(body.find("nie_output_tokens_total 5"), std::string::npos);
    // Content-free check: no prompt text in the exposition.
    EXPECT_EQ(body.find("hello"), std::string::npos);
    metrics_server.Shutdown();
}

TEST_F(GrpcServerTest, UnloadThenReloadWorks) {
    pb::LoadModelResponse load;
    ASSERT_TRUE(LoadViaControl(&load).ok());
    {
        grpc::ClientContext ctx;
        pb::UnloadModelRequest req;
        pb::UnloadModelResponse response;
        ASSERT_TRUE(control_stub_->UnloadModel(&ctx, req, &response).ok());
        EXPECT_EQ(response.state(), pb::MODEL_STATE_UNLOADED);
    }
    pb::LoadModelResponse reload;
    ASSERT_TRUE(LoadViaControl(&reload).ok());
    EXPECT_EQ(reload.state(), pb::MODEL_STATE_READY);
}

}  // namespace
}  // namespace lykuro::nie
