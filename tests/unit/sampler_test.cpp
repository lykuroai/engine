#include "core/generation/sampler.h"

#include <gtest/gtest.h>

#include <cmath>

namespace lykuro::nie {
namespace {

TEST(SamplerValidationTest, AcceptsDefaults) {
    EXPECT_TRUE(ValidateSamplingParams({}, {}).ok());
}

TEST(SamplerValidationTest, RejectsOutOfRange) {
    SamplingLimits limits;
    SamplingParams p;
    p.temperature = -0.1f;
    EXPECT_FALSE(ValidateSamplingParams(p, limits).ok());
    p = {};
    p.temperature = 100.0f;
    EXPECT_FALSE(ValidateSamplingParams(p, limits).ok());
    p = {};
    p.top_p = 0.0f;
    EXPECT_FALSE(ValidateSamplingParams(p, limits).ok());
    p = {};
    p.top_p = 1.5f;
    EXPECT_FALSE(ValidateSamplingParams(p, limits).ok());
    p = {};
    p.top_k = 100000;
    EXPECT_FALSE(ValidateSamplingParams(p, limits).ok());
    p = {};
    p.temperature = NAN;
    EXPECT_FALSE(ValidateSamplingParams(p, limits).ok());
}

TEST(SamplerTest, GreedyPicksArgmax) {
    SamplingParams p;
    p.temperature = 0.0f;
    Sampler s(p);
    uint32_t token = 0;
    ASSERT_TRUE(s.Sample({0.1f, 2.0f, -1.0f, 1.9f}, token).ok());
    EXPECT_EQ(token, 1u);
}

TEST(SamplerTest, GreedyBreaksTiesToLowestIndex) {
    SamplingParams p;
    p.temperature = 0.0f;
    Sampler s(p);
    uint32_t token = 0;
    ASSERT_TRUE(s.Sample({1.0f, 5.0f, 5.0f}, token).ok());
    EXPECT_EQ(token, 1u);
}

TEST(SamplerTest, SeededSamplingIsReproducible) {
    std::vector<float> logits = {1.0f, 0.5f, 0.2f, 3.0f, -1.0f};
    SamplingParams p;
    p.temperature = 0.8f;
    p.seed = 1234;

    std::vector<uint32_t> run1, run2;
    {
        Sampler s(p);
        for (int i = 0; i < 20; ++i) {
            uint32_t t;
            ASSERT_TRUE(s.Sample(logits, t).ok());
            run1.push_back(t);
        }
    }
    {
        Sampler s(p);
        for (int i = 0; i < 20; ++i) {
            uint32_t t;
            ASSERT_TRUE(s.Sample(logits, t).ok());
            run2.push_back(t);
        }
    }
    EXPECT_EQ(run1, run2);
}

TEST(SamplerTest, TopKRestrictsCandidates) {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_k = 2;
    p.seed = 7;
    Sampler s(p);
    // Only the two highest logits (indices 3 and 1) may ever appear.
    std::vector<float> logits = {0.0f, 5.0f, 1.0f, 6.0f};
    for (int i = 0; i < 100; ++i) {
        uint32_t t;
        ASSERT_TRUE(s.Sample(logits, t).ok());
        EXPECT_TRUE(t == 1u || t == 3u);
    }
}

TEST(SamplerTest, SmallTopPPicksTopToken) {
    SamplingParams p;
    p.temperature = 1.0f;
    p.top_p = 0.01f;
    p.seed = 7;
    Sampler s(p);
    std::vector<float> logits = {0.0f, 10.0f, 1.0f};
    for (int i = 0; i < 50; ++i) {
        uint32_t t;
        ASSERT_TRUE(s.Sample(logits, t).ok());
        EXPECT_EQ(t, 1u);
    }
}

TEST(SamplerTest, RejectsNonFiniteLogits) {
    Sampler s({});
    uint32_t t;
    EXPECT_FALSE(s.Sample({1.0f, INFINITY}, t).ok());
    EXPECT_FALSE(s.Sample({1.0f, NAN}, t).ok());
    EXPECT_FALSE(s.Sample({}, t).ok());
}

}  // namespace
}  // namespace lykuro::nie
