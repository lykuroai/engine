#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace lykuro::nie {

// Self-contained SHA-256 (FIPS 180-4) used for artifact digest verification.
// Implemented in-tree to keep the production dependency surface minimal.
class Sha256 {
public:
    Sha256();

    void Update(const void* data, size_t len);
    // Finalizes and returns the 32-byte digest. The object must not be
    // updated after Finish().
    std::array<uint8_t, 32> Finish();

    static std::array<uint8_t, 32> Hash(const void* data, size_t len);
    static std::string HexDigest(const void* data, size_t len);
    static std::string ToHex(const std::array<uint8_t, 32>& digest);

private:
    void ProcessBlock(const uint8_t* block);

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> buffer_;
    size_t buffer_len_ = 0;
    uint64_t total_len_ = 0;
    bool finished_ = false;
};

// Constant-time hex digest comparison (both lowercase hex, 64 chars).
bool DigestEquals(std::string_view a, std::string_view b);

}  // namespace lykuro::nie
