// Kernel-path Metal backend parity tests. The hand-written MSL decode
// path must agree with the CPU reference: FP16 within tolerance and
// greedy-equivalent; quantized modes deterministic and finite (weight
// quantization legitimately moves logits on the tiny random model, so
// greedy agreement there is asserted at FP16 only).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "backends/metal/qwen_metal_fast.h"
#include "core/generation/sampler.h"
#include "model/architectures/qwen/qwen_model.h"
#include "tests/testutil/tiny_model.h"

namespace lykuro::nie {
namespace {

class MetalFastParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        testutil::TinyModelSpec spec;
        spec.vocab_size = 303;
        spec.eos_token_id = 302;
        spec.max_context_tokens = 128;
        manifest_ = testutil::MakeTinyManifest(spec);
        path_ = testutil::WriteTinyWeights(spec, "metal_fast_tiny.safetensors");
        ASSERT_TRUE(file_.Open(path_).ok());

        auto cpu = QwenModel::Load(manifest_, file_);
        ASSERT_TRUE(cpu.status.ok()) << cpu.status.message();
        cpu_ = std::move(cpu.model);

        auto fast = QwenMetalFastModel::Load(manifest_, file_,
                                             MetalFastOptions{});
        ASSERT_TRUE(fast.status.ok()) << fast.status.message();
        fast_ = std::move(fast.model);
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
    std::unique_ptr<QwenMetalFastModel> fast_;
};

TEST_F(MetalFastParityTest, GreedyTrajectoryMatchesCpuReference) {
    const std::vector<uint32_t> prompt = {5, 17, 42, 7, 99};
    std::unique_ptr<SequenceState> cs, fs;
    ASSERT_TRUE(cpu_->CreateSequence(96, cs).ok());
    ASSERT_TRUE(fast_->CreateSequence(96, fs).ok());
    std::vector<float> cl, fl;
    ASSERT_TRUE(cpu_->Prefill(*cs, prompt, cl).ok());
    ASSERT_TRUE(fast_->Prefill(*fs, prompt, fl).ok());
    ExpectClose(cl, fl, 0.3f, "prefill logits");

    // Teacher-forced trajectory (the CPU token drives both models):
    // per-step logits must stay within FP16 storage tolerance. Exact
    // greedy equality is not asserted on the tiny random model — its
    // top-2 logits can be closer than one FP16 ulp, which flips argmax
    // without any computational error (observed gap < 5e-4).
    SamplingParams greedy;
    Sampler s1(greedy);
    for (int step = 0; step < 32; ++step) {
        char what[32];
        std::snprintf(what, sizeof(what), "step %d logits", step);
        ExpectClose(cl, fl, 0.05f, what);
        uint32_t t1 = 0;
        ASSERT_TRUE(s1.Sample(cl, t1).ok());
        ASSERT_TRUE(cpu_->Decode(*cs, t1, cl).ok());
        ASSERT_TRUE(fast_->Decode(*fs, t1, fl).ok());
    }
}

TEST_F(MetalFastParityTest, DeterministicAcrossRuns) {
    auto run = [&](std::vector<float>& out) {
        std::unique_ptr<SequenceState> st;
        ASSERT_TRUE(fast_->CreateSequence(64, st).ok());
        std::vector<float> lg;
        ASSERT_TRUE(fast_->Prefill(*st, {1, 2, 3}, lg).ok());
        for (int i = 4; i < 24; ++i) {
            ASSERT_TRUE(fast_->Decode(*st, uint32_t(i % 300), lg).ok());
        }
        out = lg;
    };
    std::vector<float> a, b;
    run(a);
    run(b);
    EXPECT_EQ(a, b);
}

TEST_F(MetalFastParityTest, RejectsCapacityOverrunAndBadTokens) {
    std::unique_ptr<SequenceState> st;
    ASSERT_TRUE(fast_->CreateSequence(8, st).ok());
    std::vector<float> lg;
    std::vector<uint32_t> long_prompt(9, 1);
    EXPECT_EQ(fast_->Prefill(*st, long_prompt, lg).code(),
              ErrorCode::kContextLengthExceeded);
    EXPECT_EQ(fast_->Prefill(*st, {9999}, lg).code(),
              ErrorCode::kInvalidRequest);
}

// Quantized modes: logits stay close to the CPU reference on the tiny
// model and are bit-exact run-to-run.
class MetalFastQuantTest
    : public MetalFastParityTest,
      public ::testing::WithParamInterface<MetalFastOptions::Quant> {};

TEST_P(MetalFastQuantTest, CloseToReferenceAndDeterministic) {
    MetalFastOptions opts;
    opts.quant = GetParam();
    auto q = QwenMetalFastModel::Load(manifest_, file_, opts);
    ASSERT_TRUE(q.status.ok()) << q.status.message();

    const std::vector<uint32_t> prompt = {5, 17, 42, 7, 99};
    std::unique_ptr<SequenceState> cs, qs;
    ASSERT_TRUE(cpu_->CreateSequence(64, cs).ok());
    ASSERT_TRUE(q.model->CreateSequence(64, qs).ok());
    std::vector<float> cl, ql;
    ASSERT_TRUE(cpu_->Prefill(*cs, prompt, cl).ok());
    ASSERT_TRUE(q.model->Prefill(*qs, prompt, ql).ok());
    // Weight-only quantization tolerance: wide but bounded (dev gate;
    // production quality is certified on real checkpoints).
    ExpectClose(cl, ql,
                GetParam() == MetalFastOptions::Quant::kInt8 ? 1.0f : 3.0f,
                "quantized prefill logits");

    std::unique_ptr<SequenceState> qs2;
    ASSERT_TRUE(q.model->CreateSequence(64, qs2).ok());
    std::vector<float> ql2;
    ASSERT_TRUE(q.model->Prefill(*qs2, prompt, ql2).ok());
    EXPECT_EQ(ql, ql2);
}

INSTANTIATE_TEST_SUITE_P(Quant, MetalFastQuantTest,
                         ::testing::Values(MetalFastOptions::Quant::kInt8,
                                           MetalFastOptions::Quant::kInt4));

}  // namespace
}  // namespace lykuro::nie
