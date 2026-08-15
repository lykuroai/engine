#include "security/sha256.h"

#include <cassert>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SHA2)
#include <arm_neon.h>
#define LYKURO_SHA256_ARM 1
#endif
#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#define LYKURO_SHA256_X86 1
#endif

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

#ifdef LYKURO_SHA256_ARM
// FIPS 180-4 compression via the ARMv8 SHA2 instructions. Same in-tree
// implementation policy as the portable rounds — only the CPU's crypto
// unit does the arithmetic. ~5x the portable throughput, which matters
// because every model load digests multi-GB weight files.
static void ProcessBlocksArm(std::array<uint32_t, 8>& state,
                             const uint8_t* p, size_t n) {
    uint32x4_t s0 = vld1q_u32(state.data());
    uint32x4_t s1 = vld1q_u32(state.data() + 4);
    while (n-- > 0) {
        const uint32x4_t save0 = s0;
        const uint32x4_t save1 = s1;
        uint32x4_t m[4];
        for (int i = 0; i < 4; ++i) {
            m[i] = vreinterpretq_u32_u8(
                vrev32q_u8(vld1q_u8(p + 16 * i)));
        }
        p += 64;
        for (int q = 0; q < 16; ++q) {
            const uint32x4_t wk =
                vaddq_u32(m[q & 3], vld1q_u32(&kK[4 * q]));
            const uint32x4_t e0 = s0;
            s0 = vsha256hq_u32(s0, s1, wk);
            s1 = vsha256h2q_u32(s1, e0, wk);
            if (q < 12) {
                m[q & 3] = vsha256su1q_u32(
                    vsha256su0q_u32(m[q & 3], m[(q + 1) & 3]),
                    m[(q + 2) & 3], m[(q + 3) & 3]);
            }
        }
        s0 = vaddq_u32(s0, save0);
        s1 = vaddq_u32(s1, save1);
    }
    vst1q_u32(state.data(), s0);
    vst1q_u32(state.data() + 4, s1);
}
#endif  // LYKURO_SHA256_ARM

#ifdef LYKURO_SHA256_X86
static bool HasShaNi() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    return (b & (1u << 29)) != 0;  // EBX bit 29: SHA
}

__attribute__((target("sha,sse4.1")))
static void ProcessBlocksX86(std::array<uint32_t, 8>& state,
                             const uint8_t* p, size_t n) {
    // State layout for the SHA-NI instructions: ABEF / CDGH.
    __m128i abcd = _mm_shuffle_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(state.data())),
        0x1B);  // D C B A -> A B C D reversed
    __m128i efgh = _mm_shuffle_epi32(
        _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(state.data() + 4)),
        0x1B);
    __m128i abef = _mm_alignr_epi8(abcd, efgh, 8);
    __m128i cdgh = _mm_blend_epi16(efgh, abcd, 0xF0);
    const __m128i mask =
        _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    while (n-- > 0) {
        const __m128i save_abef = abef;
        const __m128i save_cdgh = cdgh;
        __m128i m[4];
        for (int i = 0; i < 4; ++i) {
            m[i] = _mm_shuffle_epi8(
                _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(p + 16 * i)),
                mask);
        }
        p += 64;
        for (int q = 0; q < 16; ++q) {
            __m128i wk = _mm_add_epi32(
                m[q & 3], _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                              &kK[4 * q])));
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
            wk = _mm_shuffle_epi32(wk, 0x0E);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, wk);
            if (q < 12) {
                const __m128i tmp =
                    _mm_alignr_epi8(m[(q + 3) & 3], m[(q + 2) & 3], 4);
                m[q & 3] = _mm_sha256msg2_epu32(
                    _mm_add_epi32(
                        _mm_sha256msg1_epu32(m[q & 3], m[(q + 1) & 3]),
                        tmp),
                    m[(q + 3) & 3]);
            }
        }
        abef = _mm_add_epi32(abef, save_abef);
        cdgh = _mm_add_epi32(cdgh, save_cdgh);
    }
    __m128i abcd_out = _mm_shuffle_epi32(
        _mm_alignr_epi8(abef, cdgh, 8), 0x1B);
    __m128i efgh_out = _mm_shuffle_epi32(
        _mm_blend_epi16(cdgh, abef, 0xF0), 0x1B);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state.data()), abcd_out);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state.data() + 4),
                     efgh_out);
}
#endif  // LYKURO_SHA256_X86

void Sha256::ProcessBlocks(const uint8_t* p, size_t nblocks) {
#ifdef LYKURO_SHA256_ARM
    ProcessBlocksArm(state_, p, nblocks);
    return;
#endif
#ifdef LYKURO_SHA256_X86
    static const bool has_sha = HasShaNi();
    if (has_sha) {
        ProcessBlocksX86(state_, p, nblocks);
        return;
    }
#endif
    for (size_t i = 0; i < nblocks; ++i) ProcessBlock(p + 64 * i);
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
            ProcessBlocks(buffer_.data(), 1);
            buffer_len_ = 0;
        }
    }
    if (len >= 64) {
        const size_t nblocks = len / 64;
        ProcessBlocks(p, nblocks);
        p += nblocks * 64;
        len -= nblocks * 64;
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
