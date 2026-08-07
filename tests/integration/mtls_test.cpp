#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "api/server/grpc_server.h"
#include "lykuro/nie/v1/control.grpc.pb.h"
#include "lykuro/nie/v1/data.grpc.pb.h"

// mTLS + service identity tests (spec §8.2, §22.1, §30.7):
//  - Model Manager identity may call the Control API
//  - Gateway identity is rejected on Control but allowed on Data
//  - clients without a certificate cannot connect at all
//
// Fixture certificates live in tests/fixtures/certs (test-only CA,
// 10-year validity, generated with OpenSSL; never used outside tests).

namespace lykuro::nie {
namespace {

namespace pb = ::lykuro::nie::v1;
namespace fs = std::filesystem;

const char* CertDir() { return LYKURO_TEST_CERT_DIR; }

std::string ReadFile(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

class MtlsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ServerConfig config;
        Status s = MakeMtlsServerCredentials(
            fs::path(CertDir()) / "server.crt",
            fs::path(CertDir()) / "server.key",
            fs::path(CertDir()) / "ca.crt", config.credentials);
        ASSERT_TRUE(s.ok()) << s.message();
        config.control_identities = {"lykuro-model-manager"};
        config.data_identities = {"lykuro-model-manager", "lykuro-gateway"};
        config.listen_address = "localhost:0";

        server_ = std::make_unique<EngineServer>(config);
        ASSERT_TRUE(server_->Start(&port_).ok());
    }

    void TearDown() override { server_->Shutdown(); }

    std::shared_ptr<grpc::Channel> ClientChannel(const std::string& name) {
        grpc::SslCredentialsOptions options;
        options.pem_root_certs = ReadFile(fs::path(CertDir()) / "ca.crt");
        if (!name.empty()) {
            options.pem_private_key =
                ReadFile(fs::path(CertDir()) / (name + ".key"));
            options.pem_cert_chain =
                ReadFile(fs::path(CertDir()) / (name + ".crt"));
        }
        return grpc::CreateChannel("localhost:" + std::to_string(port_),
                                   grpc::SslCredentials(options));
    }

    static grpc::ClientContext* WithDeadline(grpc::ClientContext* ctx) {
        ctx->set_deadline(std::chrono::system_clock::now() +
                          std::chrono::seconds(5));
        return ctx;
    }

    std::unique_ptr<EngineServer> server_;
    int port_ = 0;
};

TEST_F(MtlsTest, ManagerIdentityMayUseControlApi) {
    auto stub = pb::ControlService::NewStub(ClientChannel("manager"));
    grpc::ClientContext ctx;
    pb::GetCapacityResponse response;
    grpc::Status s =
        stub->GetCapacity(WithDeadline(&ctx), {}, &response);
    EXPECT_TRUE(s.ok()) << s.error_message();
}

TEST_F(MtlsTest, GatewayIdentityRejectedOnControlApi) {
    auto stub = pb::ControlService::NewStub(ClientChannel("gateway"));
    grpc::ClientContext ctx;
    pb::GetCapacityResponse response;
    grpc::Status s =
        stub->GetCapacity(WithDeadline(&ctx), {}, &response);
    EXPECT_EQ(s.error_code(), grpc::StatusCode::PERMISSION_DENIED);
}

TEST_F(MtlsTest, GatewayIdentityAllowedOnDataApi) {
    auto stub = pb::DataService::NewStub(ClientChannel("gateway"));
    grpc::ClientContext ctx;
    pb::GetCapabilitiesResponse response;
    grpc::Status s =
        stub->GetCapabilities(WithDeadline(&ctx), {}, &response);
    EXPECT_TRUE(s.ok()) << s.error_message();
}

TEST_F(MtlsTest, NoClientCertificateCannotConnect) {
    auto stub = pb::DataService::NewStub(ClientChannel(""));
    grpc::ClientContext ctx;
    pb::GetCapabilitiesResponse response;
    grpc::Status s =
        stub->GetCapabilities(WithDeadline(&ctx), {}, &response);
    EXPECT_FALSE(s.ok());  // TLS handshake fails without a client cert
}

}  // namespace
}  // namespace lykuro::nie
