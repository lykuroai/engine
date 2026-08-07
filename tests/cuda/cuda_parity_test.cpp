// CPU-reference vs CUDA parity tests. These are the certification anchor
// for the CUDA backend (spec §30.2, AT-04): the GPU implementation must
// reproduce the correctness oracle within tolerance on real hardware.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>

#include "backends/cuda/cuda_backend.h"
#include "backends/cuda/qwen_cuda_model.h"
#include "core/engine/engine.h"
#include "core/generation/sampler.h"
#include "model/architectures/qwen/qwen_model.h"
#include "tests/testutil/tiny_model.h"
#include "tests/unit/tokenizer_fixture.h"

namespace lykuro::nie {
namespace {

int TestDevice() {
    const char* env = std::getenv("LYKURO_TEST_DEVICE");
    return env != nullptr ? std::atoi(env) : 0;
}

TEST(CudaBackendTest, DiscoversDevices) {
    std::vector<CudaDeviceInfo> devices;
    ASSERT_TRUE(DiscoverCudaDevices(devices).ok());
    ASSERT_FALSE(devices.empty()) << "no CUDA device visible";
    for (const auto& d : devices) {
        std::printf("device %d: %s cc%d.%d vram=%llu MiB free=%llu MiB\n",
                    d.device_id, d.name.c_str(),
                    d.compute_capability_major, d.compute_capability_minor,
                    (unsigned long long)(d.total_vram_bytes >> 20),
                    (unsigned long long)(d.free_vram_bytes >> 20));
        EXPECT_GT(d.total_vram_bytes, 0u);
    }
    CudaDeviceInfo info;
    EXPECT_TRUE(CheckCudaDevice(TestDevice(), info).ok());
    EXPECT_FALSE(CheckCudaDevice(99, info).ok());
}

class CudaParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        testutil::TinyModelSpec spec;
        spec.vocab_size = 303;
        spec.eos_token_id = 302;
        spec.max_context_tokens = 128;
        manifest_ = testutil::MakeTinyManifest(spec);
        path_ = testutil::WriteTinyWeights(spec, "cuda_tiny.safetensors");
        ASSERT_TRUE(file_.Open(path_).ok());

        auto cpu = QwenModel::Load(manifest_, file_);
        ASSERT_TRUE(cpu.status.ok()) << cpu.status.message();
        cpu_ = std::move(cpu.model);

        auto cuda = QwenCudaModel::Load(manifest_, file_, TestDevice());
        ASSERT_TRUE(cuda.status.ok()) << cuda.status.message();
        cuda_ = std::move(cuda.model);
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
    std::unique_ptr<QwenCudaModel> cuda_;
};

TEST_F(CudaParityTest, PrefillLogitsMatchCpuReference) {
    const std::vector<uint32_t> prompt = {1, 5, 9, 2, 17, 250};
    std::unique_ptr<SequenceState> cpu_state, cuda_state;
    ASSERT_TRUE(cpu_->CreateSequence(128, cpu_state).ok());
    ASSERT_TRUE(cuda_->CreateSequence(128, cuda_state).ok());

    std::vector<float> cpu_logits, cuda_logits;
    ASSERT_TRUE(cpu_->Prefill(*cpu_state, prompt, cpu_logits).ok());
    ASSERT_TRUE(cuda_->Prefill(*cuda_state, prompt, cuda_logits).ok());
    ExpectClose(cpu_logits, cuda_logits, 1e-3f, "prefill logits");
}

TEST_F(CudaParityTest, DecodeTrajectoryMatchesCpuReference) {
    const std::vector<uint32_t> prompt = {3, 7, 11};
    std::unique_ptr<SequenceState> cpu_state, cuda_state;
    ASSERT_TRUE(cpu_->CreateSequence(128, cpu_state).ok());
    ASSERT_TRUE(cuda_->CreateSequence(128, cuda_state).ok());

    std::vector<float> cpu_logits, cuda_logits;
    ASSERT_TRUE(cpu_->Prefill(*cpu_state, prompt, cpu_logits).ok());
    ASSERT_TRUE(cuda_->Prefill(*cuda_state, prompt, cuda_logits).ok());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler cpu_sampler(greedy), cuda_sampler(greedy);
    for (int step = 0; step < 30; ++step) {
        ExpectClose(cpu_logits, cuda_logits, 1e-3f, "decode logits");
        uint32_t cpu_token = 0, cuda_token = 0;
        ASSERT_TRUE(cpu_sampler.Sample(cpu_logits, cpu_token).ok());
        ASSERT_TRUE(cuda_sampler.Sample(cuda_logits, cuda_token).ok());
        ASSERT_EQ(cpu_token, cuda_token) << "greedy divergence at step "
                                         << step;
        ASSERT_TRUE(cpu_->Decode(*cpu_state, cpu_token, cpu_logits).ok());
        ASSERT_TRUE(
            cuda_->Decode(*cuda_state, cuda_token, cuda_logits).ok());
    }
}

TEST_F(CudaParityTest, CudaOutputIsDeterministicAcrossRuns) {
    auto run = [&](std::vector<float>& logits_out) {
        std::unique_ptr<SequenceState> state;
        ASSERT_TRUE(cuda_->CreateSequence(128, state).ok());
        std::vector<float> logits;
        ASSERT_TRUE(cuda_->Prefill(*state, {1, 2, 3}, logits).ok());
        for (int i = 0; i < 10; ++i) {
            ASSERT_TRUE(cuda_->Decode(*state, uint32_t(i * 7 % 300), logits)
                            .ok());
        }
        logits_out = logits;
    };
    std::vector<float> a, b;
    run(a);
    run(b);
    EXPECT_EQ(a, b);  // bit-exact run-to-run on the same device
}

TEST_F(CudaParityTest, RejectsCapacityOverrunAndBadTokens) {
    std::unique_ptr<SequenceState> state;
    ASSERT_TRUE(cuda_->CreateSequence(4, state).ok());
    std::vector<float> logits;
    std::vector<uint32_t> long_prompt(5, 1);
    EXPECT_EQ(cuda_->Prefill(*state, long_prompt, logits).code(),
              ErrorCode::kContextLengthExceeded);
    EXPECT_EQ(cuda_->Prefill(*state, {9999}, logits).code(),
              ErrorCode::kInvalidRequest);
}

// End-to-end: the engine running on the CUDA backend must produce the
// same text as the engine on the CPU reference.
TEST_F(CudaParityTest, EngineOutputMatchesCpuEngine) {
    auto tokenizer1 =
        BpeTokenizer::FromConfig(testfixture::SmallTokenizerConfig());
    auto tokenizer2 =
        BpeTokenizer::FromConfig(testfixture::SmallTokenizerConfig());
    ASSERT_TRUE(tokenizer1.status.ok());

    auto run = [&](std::unique_ptr<GenerativeModel> model,
                   BpeTokenizer::LoadResult& tok) {
        InferenceEngine engine(
            std::move(model),
            std::make_unique<BpeTokenizer>(std::move(tok.tokenizer)), {});
        InferenceRequest req;
        req.request_id = "req_parity";
        req.messages = {{Role::kUser, "hello"}};
        req.max_output_tokens = 16;
        req.sampling.temperature = 0.0f;
        auto submit = engine.Submit(req);
        EXPECT_TRUE(submit.status.ok());
        engine.RunUntilIdle();
        std::string text;
        while (auto e = submit.events->Pop()) {
            if (e->kind == StreamEvent::Kind::kOutputTextDelta) {
                text += e->text;
            }
        }
        return text;
    };

    std::string cpu_text = run(std::move(cpu_), tokenizer1);
    std::string cuda_text = run(std::move(cuda_), tokenizer2);
    EXPECT_EQ(cpu_text, cuda_text);
    EXPECT_FALSE(cuda_text.empty());
}

// Not a certified benchmark (tiny random model): records rough
// tokens/second so the pipeline exists for real certified profiles.
TEST_F(CudaParityTest, DecodeThroughputSmoke) {
    std::unique_ptr<SequenceState> state;
    ASSERT_TRUE(cuda_->CreateSequence(128, state).ok());
    std::vector<float> logits;
    ASSERT_TRUE(cuda_->Prefill(*state, {1, 2, 3}, logits).ok());

    const int steps = 100;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) {
        ASSERT_TRUE(
            cuda_->Decode(*state, uint32_t(i % 300), logits).ok());
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count();
    std::printf(
        "tiny-model decode: %d steps in %lld us (%.1f tokens/s) — NOT a "
        "certified figure\n",
        steps, (long long)elapsed, steps * 1e6 / double(elapsed));
}

}  // namespace
}  // namespace lykuro::nie
