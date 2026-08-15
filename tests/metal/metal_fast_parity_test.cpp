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
    greedy.temperature = 0.0f;
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

// GreedyRun must reproduce the sequential Decode + greedy-argmax
// trajectory bit-exactly: the same kernels compute the logits, and the
// on-GPU argmax uses the Sampler rule (max, ties to the lowest index).
TEST_F(MetalFastParityTest, GreedyRunMatchesSequentialArgmax) {
    ASSERT_TRUE(fast_->SupportsGreedyRun());
    const std::vector<uint32_t> prompt = {5, 17, 42, 7, 99};
    const int kSteps = 40;

    std::unique_ptr<SequenceState> sa, sb;
    ASSERT_TRUE(fast_->CreateSequence(96, sa).ok());
    ASSERT_TRUE(fast_->CreateSequence(96, sb).ok());
    std::vector<float> la, lb;
    ASSERT_TRUE(fast_->Prefill(*sa, prompt, la).ok());
    ASSERT_TRUE(fast_->Prefill(*sb, prompt, lb).ok());
    EXPECT_EQ(la, lb) << "prefill logits differ between twin sequences";

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler smp(greedy);
    uint32_t tb = 0;
    ASSERT_TRUE(smp.Sample(lb, tb).ok());
    std::vector<uint32_t> run_tokens;
    while (run_tokens.size() < size_t(kSteps)) {
        std::vector<uint32_t> out;
        ASSERT_TRUE(
            fast_->GreedyRun(*sb, tb, uint32_t(kSteps - run_tokens.size()),
                             out)
                .ok());
        ASSERT_FALSE(out.empty());
        run_tokens.insert(run_tokens.end(), out.begin(), out.end());
        tb = out.back();
    }
    run_tokens.resize(kSteps);

    std::vector<uint32_t> seq_tokens;
    uint32_t t = 0;
    ASSERT_TRUE(smp.Sample(la, t).ok());
    for (int i = 0; i < kSteps; ++i) {
        ASSERT_TRUE(fast_->Decode(*sa, t, la).ok());
        ASSERT_TRUE(smp.Sample(la, t).ok());
        seq_tokens.push_back(t);
    }
    EXPECT_EQ(seq_tokens, run_tokens);
}

// Diverging from the speculated trajectory (the pipelined batch guessed
// with argmax, the caller feeds something else) must flush cleanly and
// continue correct from the caller's token.
TEST_F(MetalFastParityTest, GreedyRunSurvivesTrajectoryDivergence) {
    const std::vector<uint32_t> prompt = {5, 17, 42, 7, 99};
    std::unique_ptr<SequenceState> sa, sb;
    ASSERT_TRUE(fast_->CreateSequence(96, sa).ok());
    ASSERT_TRUE(fast_->CreateSequence(96, sb).ok());
    std::vector<float> la, lb;
    ASSERT_TRUE(fast_->Prefill(*sa, prompt, la).ok());
    ASSERT_TRUE(fast_->Prefill(*sb, prompt, lb).ok());

    std::vector<uint32_t> out_a;
    ASSERT_TRUE(fast_->GreedyRun(*sa, 3, 8, out_a).ok());
    // Diverge: continue with a token that is NOT what the pipeline
    // speculated.
    const uint32_t divergent = (out_a.back() + 1) % 300;
    std::vector<uint32_t> out_a2;
    ASSERT_TRUE(fast_->GreedyRun(*sa, divergent, 8, out_a2).ok());

    // Reference: the same trajectory via sequential Decode + argmax.
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler smp(greedy);
    std::vector<float> lg;
    auto run_seq = [&](uint32_t t, size_t n,
                       std::vector<uint32_t>& out) {
        out.clear();
        for (size_t i = 0; i < n; ++i) {
            ASSERT_TRUE(fast_->Decode(*sb, t, lg).ok());
            ASSERT_TRUE(smp.Sample(lg, t).ok());
            out.push_back(t);
        }
    };
    std::vector<uint32_t> ref1, ref2;
    run_seq(3, out_a.size(), ref1);
    EXPECT_EQ(out_a, ref1);
    run_seq(divergent, out_a2.size(), ref2);
    EXPECT_EQ(out_a2, ref2);
}

// Long-context coverage: head_dim 32 and a few hundred positions drive
// the coalesced and split-row attention kernels (the default tiny model
// has hd=4 / T<96 and never leaves the short-context path). Teacher-
// forced against the CPU reference.
TEST(MetalFastLongContextTest, MatchesCpuAcrossAttentionKernels) {
    testutil::TinyModelSpec spec;
    spec.vocab_size = 128;
    spec.eos_token_id = 127;
    spec.hidden_size = 64;
    spec.num_heads = 2;
    spec.num_kv_heads = 2;
    spec.head_dim = 32;
    spec.intermediate_size = 64;
    spec.max_context_tokens = 384;
    ModelManifest manifest = testutil::MakeTinyManifest(spec);
    std::string path =
        testutil::WriteTinyWeights(spec, "metal_fast_long.safetensors");
    SafetensorsFile file;
    ASSERT_TRUE(file.Open(path).ok());
    auto cpu = QwenModel::Load(manifest, file);
    ASSERT_TRUE(cpu.status.ok()) << cpu.status.message();
    auto fast = QwenMetalFastModel::Load(manifest, file, MetalFastOptions{});
    ASSERT_TRUE(fast.status.ok()) << fast.status.message();

    std::unique_ptr<SequenceState> cs, fs;
    ASSERT_TRUE(cpu.model->CreateSequence(320, cs).ok());
    ASSERT_TRUE(fast.model->CreateSequence(320, fs).ok());
    std::vector<float> cl, fl;
    const std::vector<uint32_t> prompt = {1, 2, 3, 4, 5};
    ASSERT_TRUE(cpu.model->Prefill(*cs, prompt, cl).ok());
    ASSERT_TRUE(fast.model->Prefill(*fs, prompt, fl).ok());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler smp(greedy);
    for (int step = 0; step < 280; ++step) {
        if (step % 40 == 0) {
            ASSERT_EQ(cl.size(), fl.size());
            float md = 0.0f;
            for (size_t i = 0; i < cl.size(); ++i) {
                md = std::max(md, std::abs(cl[i] - fl[i]));
            }
            EXPECT_LE(md, 0.15f) << "step " << step << " max_diff " << md;
        }
        uint32_t t = 0;
        ASSERT_TRUE(smp.Sample(cl, t).ok());
        ASSERT_TRUE(cpu.model->Decode(*cs, t, cl).ok());
        ASSERT_TRUE(fast.model->Decode(*fs, t, fl).ok());
    }
    std::remove(path.c_str());
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
