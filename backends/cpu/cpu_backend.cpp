#include "backends/cpu/cpu_backend.h"

#include <cmath>
#include <cstring>

namespace lykuro::nie {

float Bf16ToFloat(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t FloatToBf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    if (std::isnan(v)) return uint16_t((bits >> 16) | 0x0040);
    // Round to nearest even on the truncated 16 bits.
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    return uint16_t((bits + rounding) >> 16);
}

float Fp16ToFloat(uint16_t v) {
    const uint32_t sign = uint32_t(v & 0x8000) << 16;
    const uint32_t exp = (v >> 10) & 0x1F;
    const uint32_t mant = v & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +-0
        } else {
            // Subnormal: normalize.
            int e = -1;
            uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400) == 0);
            bits = sign | uint32_t(127 - 15 - e) << 23 | ((m & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);  // inf/NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t FloatToFp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
    int32_t exp = int32_t((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0xFF) {
        // Inf/NaN
        return uint16_t(sign | 0x7C00 | (mant != 0 ? 0x200 : 0));
    }
    if (exp >= 0x1F) return uint16_t(sign | 0x7C00);  // overflow -> inf
    if (exp <= 0) {
        if (exp < -10) return sign;  // underflow -> 0
        // Subnormal half.
        mant |= 0x800000;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t half_mant = mant >> shift;
        // Round to nearest even.
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1))) {
            ++half_mant;
        }
        return uint16_t(sign | half_mant);
    }
    uint32_t half_mant = mant >> 13;
    uint32_t rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (half_mant & 1))) {
        ++half_mant;
        if (half_mant == 0x400) {
            half_mant = 0;
            ++exp;
            if (exp >= 0x1F) return uint16_t(sign | 0x7C00);
        }
    }
    return uint16_t(sign | uint32_t(exp << 10) | half_mant);
}

void Bf16ToFloatArray(const uint16_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = Bf16ToFloat(in[i]);
}

void Fp16ToFloatArray(const uint16_t* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = Fp16ToFloat(in[i]);
}

}  // namespace lykuro::nie
