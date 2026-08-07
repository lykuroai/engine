#include "security/signature.h"

#include <openssl/evp.h>

#include <memory>

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "signature";

Status VerifyFailed(const std::string& msg) {
    return Status(ErrorCode::kArtifactVerificationFailed, msg, kComponent);
}

using EvpKeyPtr = std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)>;
using EvpCtxPtr = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;

}  // namespace

std::string BytesToHex(const uint8_t* data, size_t len) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(kHex[data[i] >> 4]);
        out.push_back(kHex[data[i] & 0xF]);
    }
    return out;
}

Status HexToBytes(std::string_view hex, uint8_t* out, size_t expected_len) {
    if (hex.size() != expected_len * 2) {
        return VerifyFailed("hex value has wrong length");
    }
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < expected_len; ++i) {
        int hi = nibble(hex[i * 2]);
        int lo = nibble(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return VerifyFailed("hex value is malformed");
        out[i] = uint8_t((hi << 4) | lo);
    }
    return Status::Ok();
}

Status ParsePublicKeyHex(std::string_view hex, std::array<uint8_t, 32>& out) {
    return HexToBytes(hex, out.data(), out.size());
}

Status VerifyManifestSignature(std::string_view message,
                               std::string_view signature_hex,
                               const TrustedKeys& trusted) {
    if (trusted.keys.empty()) {
        return VerifyFailed("no trusted signing keys configured");
    }
    std::array<uint8_t, 64> signature;
    Status s = HexToBytes(signature_hex, signature.data(), signature.size());
    if (!s.ok()) return VerifyFailed("manifest signature is malformed");

    for (const auto& key : trusted.keys) {
        EvpKeyPtr pkey(
            EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr,
                                        key.data(), key.size()),
            EVP_PKEY_free);
        if (!pkey) continue;
        EvpCtxPtr ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
        if (!ctx) continue;
        if (EVP_DigestVerifyInit(ctx.get(), nullptr, nullptr, nullptr,
                                 pkey.get()) != 1) {
            continue;
        }
        if (EVP_DigestVerify(
                ctx.get(), signature.data(), signature.size(),
                reinterpret_cast<const unsigned char*>(message.data()),
                message.size()) == 1) {
            return Status::Ok();
        }
    }
    return VerifyFailed("manifest signature does not match any trusted key");
}

Status GenerateKeypair(std::array<uint8_t, 32>& public_key,
                       std::array<uint8_t, 64>& private_key) {
    EVP_PKEY* raw = nullptr;
    std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)> ctx(
        EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr), EVP_PKEY_CTX_free);
    if (!ctx || EVP_PKEY_keygen_init(ctx.get()) != 1 ||
        EVP_PKEY_keygen(ctx.get(), &raw) != 1) {
        return Status(ErrorCode::kInternalError, "keygen failed", kComponent);
    }
    EvpKeyPtr pkey(raw, EVP_PKEY_free);

    size_t pub_len = public_key.size();
    size_t priv_len = 32;
    uint8_t priv_seed[32];
    if (EVP_PKEY_get_raw_public_key(pkey.get(), public_key.data(),
                                    &pub_len) != 1 ||
        EVP_PKEY_get_raw_private_key(pkey.get(), priv_seed, &priv_len) != 1 ||
        pub_len != 32 || priv_len != 32) {
        return Status(ErrorCode::kInternalError, "key export failed",
                      kComponent);
    }
    // Store seed || public so SignMessage can rebuild the key.
    std::copy(priv_seed, priv_seed + 32, private_key.begin());
    std::copy(public_key.begin(), public_key.end(),
              private_key.begin() + 32);
    return Status::Ok();
}

Status SignMessage(std::string_view message,
                   const std::array<uint8_t, 64>& private_key,
                   std::array<uint8_t, 64>& signature) {
    EvpKeyPtr pkey(
        EVP_PKEY_new_raw_private_key(EVP_PKEY_ED25519, nullptr,
                                     private_key.data(), 32),
        EVP_PKEY_free);
    if (!pkey) {
        return Status(ErrorCode::kInternalError, "invalid private key",
                      kComponent);
    }
    EvpCtxPtr ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    size_t sig_len = signature.size();
    if (!ctx ||
        EVP_DigestSignInit(ctx.get(), nullptr, nullptr, nullptr,
                           pkey.get()) != 1 ||
        EVP_DigestSign(ctx.get(), signature.data(), &sig_len,
                       reinterpret_cast<const unsigned char*>(message.data()),
                       message.size()) != 1 ||
        sig_len != 64) {
        return Status(ErrorCode::kInternalError, "signing failed",
                      kComponent);
    }
    return Status::Ok();
}

}  // namespace lykuro::nie
