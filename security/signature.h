#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// Ed25519 manifest signature verification (spec §10.3 signature chain,
// §22.3). Uses OpenSSL libcrypto (approved low-level dependency, recorded
// in the SBOM). Keys and signatures are raw bytes hex-encoded on disk:
//   manifest.sig        64-byte signature over the exact bytes of
//                       manifest.json, hex-encoded (128 hex chars)
//   trusted key         32-byte Ed25519 public key, hex-encoded
//
// Trust policy: the engine accepts one or more trusted public keys from
// configuration; a manifest is accepted when any trusted key verifies.

struct TrustedKeys {
    // Each entry: 32-byte public key.
    std::vector<std::array<uint8_t, 32>> keys;
};

// Parses a hex-encoded 32-byte public key.
Status ParsePublicKeyHex(std::string_view hex, std::array<uint8_t, 32>& out);

// Verifies `signature_hex` (128 hex chars) over `message` with any of the
// trusted keys. Returns artifact_verification_failed on any mismatch.
Status VerifyManifestSignature(std::string_view message,
                               std::string_view signature_hex,
                               const TrustedKeys& trusted);

// Development/test helper: generates a keypair and signs a message.
// Not part of the production engine path (the signer lives in the release
// pipeline); used by tools/sign_artifact and tests.
Status GenerateKeypair(std::array<uint8_t, 32>& public_key,
                       std::array<uint8_t, 64>& private_key);
Status SignMessage(std::string_view message,
                   const std::array<uint8_t, 64>& private_key,
                   std::array<uint8_t, 64>& signature);

std::string BytesToHex(const uint8_t* data, size_t len);
Status HexToBytes(std::string_view hex, uint8_t* out, size_t expected_len);

}  // namespace lykuro::nie
