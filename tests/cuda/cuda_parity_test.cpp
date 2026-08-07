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

// Long prompt (> the batched-decode staging width): regression anchor for
// the chunked-prefill KV staging overflow found during the paged-KV work.
TEST_F(CudaParityTest, LongPromptPrefillMatchesCpuReference) {
    std::vector<uint32_t> prompt;
    for (uint32_t i = 0; i < 60; ++i) prompt.push_back(1 + i * 5 % 300);
    std::unique_ptr<SequenceState> cpu_state, cuda_state;
    ASSERT_TRUE(cpu_->CreateSequence(128, cpu_state).ok());
    ASSERT_TRUE(cuda_->CreateSequence(128, cuda_state).ok());

    std::vector<float> cpu_logits, cuda_logits;
    ASSERT_TRUE(cpu_->Prefill(*cpu_state, prompt, cpu_logits).ok());
    ASSERT_TRUE(cuda_->Prefill(*cuda_state, prompt, cuda_logits).ok());
    ExpectClose(cpu_logits, cuda_logits, 1e-3f, "long prompt logits");
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

// Batched decode must match per-sequence decode: same greedy tokens and
// logits within reordering tolerance (batch GEMM sums in tiled order).
TEST_F(CudaParityTest, BatchedDecodeMatchesSequentialDecode) {
    const std::vector<std::vector<uint32_t>> prompts = {
        {1, 5, 9}, {17, 4, 22, 8, 30}, {7, 7, 25}};
    const size_t n = prompts.size();

    // Sequential baseline.
    std::vector<std::unique_ptr<SequenceState>> seq_states(n);
    std::vector<std::vector<float>> seq_logits(n);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(cuda_->CreateSequence(64, seq_states[i]).ok());
        ASSERT_TRUE(
            cuda_->Prefill(*seq_states[i], prompts[i], seq_logits[i]).ok());
    }

    // Batched run.
    std::vector<std::unique_ptr<SequenceState>> bat_states(n);
    std::vector<std::vector<float>> bat_logits(n);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(cuda_->CreateSequence(64, bat_states[i]).ok());
        ASSERT_TRUE(
            cuda_->Prefill(*bat_states[i], prompts[i], bat_logits[i]).ok());
    }

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    for (int step = 0; step < 12; ++step) {
        std::vector<uint32_t> seq_tokens(n), bat_tokens(n);
        for (size_t i = 0; i < n; ++i) {
            Sampler s1(greedy), s2(greedy);
            ASSERT_TRUE(s1.Sample(seq_logits[i], seq_tokens[i]).ok());
            ASSERT_TRUE(s2.Sample(bat_logits[i], bat_tokens[i]).ok());
            ASSERT_EQ(seq_tokens[i], bat_tokens[i])
                << "greedy divergence seq " << i << " step " << step;
            ExpectClose(seq_logits[i], bat_logits[i], 2e-3f,
                        "batch vs sequential logits");
        }
        for (size_t i = 0; i < n; ++i) {
            ASSERT_TRUE(cuda_->Decode(*seq_states[i], seq_tokens[i],
                                      seq_logits[i])
                            .ok());
        }
        std::vector<GenerativeModel::DecodeBatchItem> items;
        for (size_t i = 0; i < n; ++i) {
            items.push_back(
                {bat_states[i].get(), bat_tokens[i], &bat_logits[i]});
        }
        std::vector<Status> per_item;
        ASSERT_TRUE(cuda_->DecodeBatch(items, per_item).ok());
        for (const Status& s : per_item) ASSERT_TRUE(s.ok());
    }
}

TEST_F(CudaParityTest, BatchedDecodeIsolatesInvalidItems) {
    std::unique_ptr<SequenceState> good, full;
    ASSERT_TRUE(cuda_->CreateSequence(64, good).ok());
    ASSERT_TRUE(cuda_->CreateSequence(3, full).ok());
    std::vector<float> good_logits, full_logits;
    ASSERT_TRUE(cuda_->Prefill(*good, {1, 2}, good_logits).ok());
    ASSERT_TRUE(cuda_->Prefill(*full, {1, 2, 3}, full_logits).ok());

    std::vector<GenerativeModel::DecodeBatchItem> items = {
        {good.get(), 5, &good_logits},
        {full.get(), 6, &full_logits},   // capacity exhausted
        {good.get(), 99999, &good_logits},  // invalid token
    };
    std::vector<Status> per_item;
    ASSERT_TRUE(cuda_->DecodeBatch(items, per_item).ok());
    EXPECT_TRUE(per_item[0].ok());
    EXPECT_EQ(per_item[1].code(), ErrorCode::kContextLengthExceeded);
    EXPECT_EQ(per_item[2].code(), ErrorCode::kInvalidRequest);
}

// ---- paged KV / prefix cache (spec §16.3) ----

TEST_F(CudaParityTest, PagedPoolExhaustionReportsCapacity) {
    testutil::TinyModelSpec spec;
    spec.vocab_size = 303;
    spec.eos_token_id = 302;
    spec.max_context_tokens = 128;
    CudaModelOptions options;
    options.device_id = TestDevice();
    options.kv_pool_tokens = 32;  // 2 blocks of 16
    options.kv_block_tokens = 16;
    auto small = QwenCudaModel::Load(manifest_, file_, options);
    ASSERT_TRUE(small.status.ok()) << small.status.message();

    std::unique_ptr<SequenceState> state;
    ASSERT_TRUE(small.model->CreateSequence(128, state).ok());
    std::vector<float> logits;
    // 40 tokens need 3 blocks; the pool only has 2.
    std::vector<uint32_t> prompt(40, 1);
    Status s = small.model->Prefill(*state, prompt, logits);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kCapacityExhausted);

    // Releasing the sequence returns its blocks to the pool.
    state.reset();
    EXPECT_EQ(small.model->kv_blocks_free(),
              small.model->kv_blocks_total());
}

TEST_F(CudaParityTest, PrefixCacheReusesBlocksWithinScope) {
    testutil::TinyModelSpec spec;
    spec.vocab_size = 303;
    spec.eos_token_id = 302;
    spec.max_context_tokens = 128;
    CudaModelOptions options;
    options.device_id = TestDevice();
    options.kv_pool_tokens = 512;
    options.kv_block_tokens = 16;
    auto model = QwenCudaModel::Load(manifest_, file_, options);
    ASSERT_TRUE(model.status.ok());

    // 40-token prompt: blocks 0,1 (32 tokens) are cacheable full blocks.
    std::vector<uint32_t> prompt;
    for (uint32_t i = 0; i < 40; ++i) prompt.push_back(1 + i % 250);
    SequenceOptions seq_options;
    seq_options.allow_prefix_cache = true;
    seq_options.scope_key = "tn_a/prj_1";

    std::unique_ptr<SequenceState> first;
    ASSERT_TRUE(model.model->CreateSequenceEx(128, seq_options, first).ok());
    std::vector<float> logits_first;
    ASSERT_TRUE(model.model->Prefill(*first, prompt, logits_first).ok());
    EXPECT_EQ(model.model->prefix_cache_hit_tokens(), 0u);

    // Same prompt, same scope: two blocks (32 tokens) come from cache and
    // the logits are identical (deterministic kernels).
    std::unique_ptr<SequenceState> second;
    ASSERT_TRUE(
        model.model->CreateSequenceEx(128, seq_options, second).ok());
    std::vector<float> logits_second;
    ASSERT_TRUE(model.model->Prefill(*second, prompt, logits_second).ok());
    EXPECT_EQ(model.model->prefix_cache_hit_tokens(), 32u);
    EXPECT_EQ(logits_first, logits_second);
    const std::vector<float> pristine_prefill_logits = logits_first;

    // Decode continues identically after a cached prefill.
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    Sampler s1(greedy), s2(greedy);
    for (int step = 0; step < 8; ++step) {
        uint32_t t1 = 0, t2 = 0;
        ASSERT_TRUE(s1.Sample(logits_first, t1).ok());
        ASSERT_TRUE(s2.Sample(logits_second, t2).ok());
        ASSERT_EQ(t1, t2);
        ASSERT_TRUE(model.model->Decode(*first, t1, logits_first).ok());
        ASSERT_TRUE(model.model->Decode(*second, t2, logits_second).ok());
    }

    // Different scope: no reuse (spec §16.1 isolation).
    SequenceOptions other_scope;
    other_scope.allow_prefix_cache = true;
    other_scope.scope_key = "tn_b/prj_9";
    std::unique_ptr<SequenceState> third;
    ASSERT_TRUE(
        model.model->CreateSequenceEx(128, other_scope, third).ok());
    std::vector<float> logits_third;
    ASSERT_TRUE(model.model->Prefill(*third, prompt, logits_third).ok());
    EXPECT_EQ(model.model->prefix_cache_hit_tokens(), 32u);  // unchanged
    EXPECT_EQ(pristine_prefill_logits, logits_third);  // same math, no sharing
}

TEST_F(CudaParityTest, PrefixCacheDisabledByDefault) {
    testutil::TinyModelSpec spec;
    CudaModelOptions options;
    options.device_id = TestDevice();
    options.kv_pool_tokens = 512;
    options.kv_block_tokens = 16;
    auto model = QwenCudaModel::Load(manifest_, file_, options);
    ASSERT_TRUE(model.status.ok());

    std::vector<uint32_t> prompt(40, 7);
    for (int round = 0; round < 2; ++round) {
        std::unique_ptr<SequenceState> state;
        ASSERT_TRUE(model.model->CreateSequence(128, state).ok());
        std::vector<float> logits;
        ASSERT_TRUE(model.model->Prefill(*state, prompt, logits).ok());
    }
    EXPECT_EQ(model.model->prefix_cache_hit_tokens(), 0u);
}

TEST_F(CudaParityTest, PrefixCacheEvictionReclaimsUnreferencedBlocks) {
    testutil::TinyModelSpec spec;
    CudaModelOptions options;
    options.device_id = TestDevice();
    options.kv_pool_tokens = 64;  // 4 blocks of 16
    options.kv_block_tokens = 16;
    auto model = QwenCudaModel::Load(manifest_, file_, options);
    ASSERT_TRUE(model.status.ok());

    SequenceOptions seq_options;
    seq_options.allow_prefix_cache = true;
    seq_options.scope_key = "tn_a";

    // Prompt A fills + registers 2 full blocks, then the sequence ends
    // (blocks stay cached with refcount 0).
    {
        std::vector<uint32_t> prompt_a(40, 3);
        std::unique_ptr<SequenceState> a;
        ASSERT_TRUE(model.model->CreateSequenceEx(128, seq_options, a).ok());
        std::vector<float> logits;
        ASSERT_TRUE(model.model->Prefill(*a, prompt_a, logits).ok());
    }
    EXPECT_EQ(model.model->kv_blocks_free(), model.model->kv_blocks_total());

    // Prompt B needs 3 blocks; eviction reclaims A's cached blocks.
    {
        std::vector<uint32_t> prompt_b(48, 9);
        std::unique_ptr<SequenceState> b;
        ASSERT_TRUE(model.model->CreateSequenceEx(128, seq_options, b).ok());
        std::vector<float> logits;
        ASSERT_TRUE(model.model->Prefill(*b, prompt_b, logits).ok());
    }
}

// ---- weight-only quantization (Phase 4) ----

TEST_F(CudaParityTest, QuantizedModelsRunAndStayClose) {
    for (WeightQuant quant : {WeightQuant::kInt8, WeightQuant::kInt4}) {
        CudaModelOptions options;
        options.device_id = TestDevice();
        options.quantization = quant;
        auto qmodel = QwenCudaModel::Load(manifest_, file_, options);
        ASSERT_TRUE(qmodel.status.ok()) << qmodel.status.message();

        const std::vector<uint32_t> prompt = {1, 5, 9, 2, 17, 250};
        std::unique_ptr<SequenceState> fp_state, q_state;
        ASSERT_TRUE(cuda_->CreateSequence(128, fp_state).ok());
        ASSERT_TRUE(qmodel.model->CreateSequence(128, q_state).ok());
        std::vector<float> fp_logits, q_logits;
        ASSERT_TRUE(cuda_->Prefill(*fp_state, prompt, fp_logits).ok());
        ASSERT_TRUE(qmodel.model->Prefill(*q_state, prompt, q_logits).ok());

        // Loose closeness bound: quantization error, not a bug detector —
        // certified quality gates live in the offline evaluation pipeline.
        const float tol = quant == WeightQuant::kInt8 ? 0.05f : 0.4f;
        ExpectClose(fp_logits, q_logits, tol, "quantized logits");

        // Determinism holds for quantized weights too.
        std::unique_ptr<SequenceState> rerun_state;
        ASSERT_TRUE(qmodel.model->CreateSequence(128, rerun_state).ok());
        std::vector<float> rerun_logits;
        ASSERT_TRUE(
            qmodel.model->Prefill(*rerun_state, prompt, rerun_logits).ok());
        EXPECT_EQ(q_logits, rerun_logits);
    }
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
