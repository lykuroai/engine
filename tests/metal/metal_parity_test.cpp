// Metal backend parity tests (LYK-NIE-ADD-METAL-001 §29.2/§29.3).
// The MPSGraph implementation must reproduce the CPU reference within
// tolerance on real Apple Silicon hardware.

#include <gtest/gtest.h>

#include <cstdio>

#include "backends/metal/metal_backend.h"
#include "backends/metal/qwen_metal_model.h"
#include "core/generation/sampler.h"
#include "model/architectures/qwen/qwen_model.h"
#include "tests/testutil/tiny_model.h"

namespace lykuro::nie {
namespace {

TEST(MetalBackendTest, InspectsDevice) {
    MetalDeviceInfo info;
    ASSERT_TRUE(InspectMetalDevice(info).ok());
    std::printf("metal device=%s unified=%d working_set=%.1fGB\n",
                info.device_name.c_str(), info.unified_memory,
                double(info.recommended_working_set_bytes) / 1e9);
    EXPECT_TRUE(info.unified_memory);
    EXPECT_GT(info.recommended_working_set_bytes, 0u);
}

class MetalParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        testutil::TinyModelSpec spec;
        spec.vocab_size = 303;
        spec.eos_token_id = 302;
        spec.max_context_tokens = 128;
        manifest_ = testutil::MakeTinyManifest(spec);
        path_ = testutil::WriteTinyWeights(spec, "metal_tiny.safetensors");
        ASSERT_TRUE(file_.Open(path_).ok());

        auto cpu = QwenModel::Load(manifest_, file_);
        ASSERT_TRUE(cpu.status.ok()) << cpu.status.message();
        cpu_ = std::move(cpu.model);

        auto metal = QwenMetalModel::Load(manifest_, file_);
        ASSERT_TRUE(metal.status.ok()) << metal.status.message();
        metal_ = std::move(metal.model);
    }

    void TearDown() override { std::remove(path_.c_str()); }

    static void ExpectClose(const std::vector<float>& a,
                            const std::vector<float>& b, float tol,
                            const char* what) {
        ASSERT_EQ(a.size(), b.size());
        float max_diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        EXPECT_LE(max_diff, tol) << what << " max_diff=" << max_diff;
    }

    ModelManifest manifest_;
    std::string path_;
    SafetensorsFile file_;
    std::unique_ptr<QwenModel> cpu_;
    std::unique_ptr<QwenMetalModel> metal_;
};

TEST_F(MetalParityTest, PrefillLogitsMatchCpuReference) {
    const std::vector<uint32_t> prompt = {1, 5, 9, 2, 17, 250};
    std::unique_ptr<SequenceState> cpu_state, metal_state;
    ASSERT_TRUE(cpu_->CreateSequence(128, cpu_state).ok());
    ASSERT_TRUE(metal_->CreateSequence(128, metal_state).ok());

    std::vector<float> cpu_logits, metal_logits;
    ASSERT_TRUE(cpu_->Prefill(*cpu_state, prompt, cpu_logits).ok());
    ASSERT_TRUE(metal_->Prefill(*metal_state, prompt, metal_logits).ok());
    ExpectClose(cpu_logits, metal_logits, 1e-3f, "metal prefill logits");
}

TEST_F(MetalParityTest, DecodeTrajectoryMatchesCpuReference) {
    const std::vector<uint32_t> prompt = {3, 7, 11};
    std::unique_ptr<SequenceState> cpu_state, metal_state;
    ASSERT_TRUE(cpu_->CreateSequence(128, cpu_state).ok());
    ASSERT_TRUE(metal_->CreateSequence(128, metal_state).ok());

    std::vector<float> cpu_logits, metal_logits;
    ASSERT_TRUE(cpu_->Prefill(*cpu_state, prompt, cpu_logits).ok());
    ASSERT_TRUE(metal_->Prefill(*metal_state, prompt, metal_logits).ok());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler s1(greedy), s2(greedy);
    for (int step = 0; step < 20; ++step) {
        ExpectClose(cpu_logits, metal_logits, 1e-3f, "metal decode logits");
        uint32_t t1 = 0, t2 = 0;
        ASSERT_TRUE(s1.Sample(cpu_logits, t1).ok());
        ASSERT_TRUE(s2.Sample(metal_logits, t2).ok());
        ASSERT_EQ(t1, t2) << "metal greedy divergence at step " << step;
        ASSERT_TRUE(cpu_->Decode(*cpu_state, t1, cpu_logits).ok());
        ASSERT_TRUE(metal_->Decode(*metal_state, t2, metal_logits).ok());
    }
}

TEST_F(MetalParityTest, DeterministicAcrossRuns) {
    auto run = [&](std::vector<float>& out) {
        std::unique_ptr<SequenceState> state;
        ASSERT_TRUE(metal_->CreateSequence(128, state).ok());
        std::vector<float> logits;
        ASSERT_TRUE(metal_->Prefill(*state, {1, 2, 3}, logits).ok());
        for (int i = 0; i < 6; ++i) {
            ASSERT_TRUE(
                metal_->Decode(*state, uint32_t(i * 7 % 300), logits).ok());
        }
        out = logits;
    };
    std::vector<float> a, b;
    run(a);
    run(b);
    EXPECT_EQ(a, b);
}

TEST_F(MetalParityTest, RejectsCapacityOverrunAndBadTokens) {
    std::unique_ptr<SequenceState> state;
    ASSERT_TRUE(metal_->CreateSequence(4, state).ok());
    std::vector<float> logits;
    std::vector<uint32_t> long_prompt(5, 1);
    EXPECT_EQ(metal_->Prefill(*state, long_prompt, logits).code(),
              ErrorCode::kContextLengthExceeded);
    EXPECT_EQ(metal_->Prefill(*state, {9999}, logits).code(),
              ErrorCode::kInvalidRequest);
}

}  // namespace
}  // namespace lykuro::nie
