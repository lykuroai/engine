#include "backends/cpu/cpu_backend.h"

#include <gtest/gtest.h>

#include <cmath>

namespace lykuro::nie {
namespace {

TEST(Bf16Test, RoundTripsExactValues) {
    for (float v : {0.0f, 1.0f, -2.0f, 0.5f, 256.0f, -0.09375f}) {
        EXPECT_FLOAT_EQ(Bf16ToFloat(FloatToBf16(v)), v);
    }
}

TEST(Bf16Test, HandlesSpecials) {
    EXPECT_TRUE(std::isinf(Bf16ToFloat(FloatToBf16(INFINITY))));
    EXPECT_TRUE(std::isnan(Bf16ToFloat(FloatToBf16(NAN))));
    EXPECT_EQ(FloatToBf16(0.0f), 0);
}

TEST(Bf16Test, RoundsToNearestEven) {
    // 1.0 + 2^-9 is exactly halfway between two bf16 values.
    float halfway = 1.0f + std::ldexp(1.0f, -9);
    float rounded = Bf16ToFloat(FloatToBf16(halfway));
    EXPECT_TRUE(rounded == 1.0f || rounded == 1.0f + std::ldexp(1.0f, -8));
    EXPECT_FLOAT_EQ(rounded, 1.0f);  // even mantissa wins
}

TEST(Fp16Test, RoundTripsExactValues) {
    for (float v : {0.0f, 1.0f, -1.5f, 0.25f, 1024.0f, 65504.0f}) {
        EXPECT_FLOAT_EQ(Fp16ToFloat(FloatToFp16(v)), v);
    }
}

TEST(Fp16Test, HandlesSubnormals) {
    float smallest = std::ldexp(1.0f, -24);  // smallest fp16 subnormal
    EXPECT_FLOAT_EQ(Fp16ToFloat(FloatToFp16(smallest)), smallest);
}

TEST(Fp16Test, HandlesSpecials) {
    EXPECT_TRUE(std::isinf(Fp16ToFloat(FloatToFp16(INFINITY))));
    EXPECT_TRUE(std::isnan(Fp16ToFloat(FloatToFp16(NAN))));
    EXPECT_TRUE(std::isinf(Fp16ToFloat(FloatToFp16(1e6f))));  // overflow
}

TEST(ArrayConversionTest, ConvertsArrays) {
    uint16_t bf16[3] = {FloatToBf16(1.0f), FloatToBf16(-2.0f),
                        FloatToBf16(0.5f)};
    float out[3];
    Bf16ToFloatArray(bf16, out, 3);
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], -2.0f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
}

}  // namespace
}  // namespace lykuro::nie
