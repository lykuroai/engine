#include "core/engine/config.h"

#include <gtest/gtest.h>

namespace lykuro::nie {
namespace {

const char kValidConfig[] = R"({
  "engine": {"id": "nie-node-01", "listen_address": "127.0.0.1",
             "grpc_port": 19443, "log_level": "info"},
  "security": {"mtls_required": true,
               "server_cert_path": "/run/secrets/server.crt",
               "server_key_path": "/run/secrets/server.key",
               "client_ca_path": "/run/secrets/client-ca.crt",
               "control_identities": ["lykuro-model-manager"],
               "data_identities": ["lykuro-gateway"],
               "trusted_signing_keys": [
                 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
               ]},
  "model": {"artifact_path": "/models/current"},
  "scheduler": {"max_queue": 256, "max_sequences": 8},
  "generation": {"max_output_tokens": 4096},
  "observability": {"metrics_enabled": true, "metrics_port": 19090}
})";

TEST(ConfigTest, ParsesValidConfig) {
    auto r = ParseFileConfig(kValidConfig);
    ASSERT_TRUE(r.status.ok()) << r.status.message();
    EXPECT_EQ(r.config.engine_id, "nie-node-01");
    EXPECT_EQ(r.config.grpc_port, 19443);
    EXPECT_TRUE(r.config.mtls_required);
    EXPECT_EQ(r.config.control_identities.size(), 1u);
    EXPECT_EQ(r.config.trusted_signing_keys_hex.size(), 1u);
    EXPECT_EQ(r.config.max_sequences, 8u);
    EXPECT_TRUE(r.config.metrics_enabled);
}

TEST(ConfigTest, DefaultsApplyForOmittedSections) {
    auto r = ParseFileConfig(
        R"({"security": {"allow_unsigned_dev": true, "mtls_required": false}})");
    ASSERT_TRUE(r.status.ok()) << r.status.message();
    EXPECT_EQ(r.config.grpc_port, 19443);
    EXPECT_EQ(r.config.max_queue, 256u);
}

TEST(ConfigTest, RejectsUnknownKeys) {
    std::string bad = kValidConfig;
    bad.insert(1, "\"mystery\": 1,");
    EXPECT_FALSE(ParseFileConfig(bad).status.ok());

    auto r = ParseFileConfig(
        R"({"engine": {"id": "x", "unknown_flag": true},
            "security": {"allow_unsigned_dev": true,
                         "mtls_required": false}})");
    EXPECT_FALSE(r.status.ok());
}

TEST(ConfigTest, RejectsOutOfRangeValues) {
    auto r = ParseFileConfig(
        R"({"engine": {"grpc_port": 99999},
            "security": {"allow_unsigned_dev": true,
                         "mtls_required": false}})");
    EXPECT_FALSE(r.status.ok());

    r = ParseFileConfig(
        R"({"scheduler": {"max_queue": 0},
            "security": {"allow_unsigned_dev": true,
                         "mtls_required": false}})");
    EXPECT_FALSE(r.status.ok());
}

TEST(ConfigTest, RejectsInvalidLogLevel) {
    auto r = ParseFileConfig(
        R"({"engine": {"log_level": "verbose"},
            "security": {"allow_unsigned_dev": true,
                         "mtls_required": false}})");
    EXPECT_FALSE(r.status.ok());
}

TEST(ConfigTest, MtlsRequiresCredentialPaths) {
    auto r = ParseFileConfig(
        R"({"security": {"mtls_required": true,
                         "allow_unsigned_dev": true}})");
    EXPECT_FALSE(r.status.ok());
}

TEST(ConfigTest, RequiresTrustAnchorOrDevFlag) {
    // Neither trusted keys nor dev flag: refused (fail-closed).
    auto r = ParseFileConfig(
        R"({"security": {"mtls_required": false}})");
    EXPECT_FALSE(r.status.ok());
}

TEST(ConfigTest, RejectsMalformedSigningKey) {
    auto r = ParseFileConfig(
        R"({"security": {"mtls_required": false,
                         "trusted_signing_keys": ["abcd"]}})");
    EXPECT_FALSE(r.status.ok());
}

TEST(ConfigTest, RejectsNonJson) {
    EXPECT_FALSE(ParseFileConfig("max_queue: 10").status.ok());
    EXPECT_FALSE(ParseFileConfig("").status.ok());
}

}  // namespace
}  // namespace lykuro::nie
