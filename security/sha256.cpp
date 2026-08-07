#include "security/sha256.h"

#include <cassert>
#include <cstring>

namespace lykuro::nie {

namespace {

constexpr std::array<uint32_t, 64> kK = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t Rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
      buffer_{} {}

void Sha256::ProcessBlock(const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[i * 4]) << 24) |
               (uint32_t(block[i * 4 + 1]) << 16) |
               (uint32_t(block[i * 4 + 2]) << 8) |
               uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + kK[i] + w[i];
        uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
}

void Sha256::Update(const void* data, size_t len) {
    assert(!finished_);
    const uint8_t* p = static_cast<const uint8_t*>(data);
    total_len_ += len;

    if (buffer_len_ > 0) {
        size_t take = std::min(len, buffer_.size() - buffer_len_);
        std::memcpy(buffer_.data() + buffer_len_, p, take);
        buffer_len_ += take;
        p += take;
        len -= take;
        if (buffer_len_ == buffer_.size()) {
            ProcessBlock(buffer_.data());
            buffer_len_ = 0;
        }
    }
    while (len >= 64) {
        ProcessBlock(p);
        p += 64;
        len -= 64;
    }
    if (len > 0) {
        std::memcpy(buffer_.data(), p, len);
        buffer_len_ = len;
    }
}

std::array<uint8_t, 32> Sha256::Finish() {
    assert(!finished_);

    // Padding: 0x80, zeros to a 56-byte boundary, then the 64-bit
    // big-endian message length in bits.
    const uint64_t bit_len = total_len_ * 8;
    const uint8_t pad = 0x80;
    const uint8_t zero = 0;
    Update(&pad, 1);
    while (buffer_len_ != 56) {
        Update(&zero, 1);
    }
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) {
        len_be[i] = uint8_t(bit_len >> (56 - i * 8));
    }
    Update(len_be, 8);
    finished_ = true;

    std::array<uint8_t, 32> digest;
    for (int i = 0; i < 8; ++i) {
        digest[i * 4] = uint8_t(state_[i] >> 24);
        digest[i * 4 + 1] = uint8_t(state_[i] >> 16);
        digest[i * 4 + 2] = uint8_t(state_[i] >> 8);
        digest[i * 4 + 3] = uint8_t(state_[i]);
    }
    return digest;
}

std::array<uint8_t, 32> Sha256::Hash(const void* data, size_t len) {
    Sha256 h;
    h.Update(data, len);
    return h.Finish();
}

std::string Sha256::ToHex(const std::array<uint8_t, 32>& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t b : digest) {
        out.push_back(kHex[b >> 4]);
        out.push_back(kHex[b & 0xf]);
    }
    return out;
}

std::string Sha256::HexDigest(const void* data, size_t len) {
    return ToHex(Hash(data, len));
}

bool DigestEquals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    unsigned diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        diff |= unsigned(a[i]) ^ unsigned(b[i]);
    }
    return diff == 0;
}

}  // namespace lykuro::nie
