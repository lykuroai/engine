#include "core/engine/engine.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <vector>

#include "tests/testutil/tiny_model.h"
#include "tests/unit/tokenizer_fixture.h"

namespace lykuro::nie {
namespace {

// The engine fixture pairs the tiny random model (vocab 303 to cover every
// tokenizer id) with the small BPE tokenizer fixture. EOS is
// <|endoftext|> (302), which the random model effectively never emits, so
// finish reasons are driven by max_output_tokens / stops / cancel.
class EngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        testutil::TinyModelSpec spec;
        spec.vocab_size = 303;
        spec.eos_token_id = 302;
        spec.max_context_tokens = 128;
        manifest_ = testutil::MakeTinyManifest(spec);
        path_ = testutil::WriteTinyWeights(spec, "engine_tiny.safetensors");
        ASSERT_TRUE(file_.Open(path_).ok());
    }

    void TearDown() override { std::remove(path_.c_str()); }

    std::unique_ptr<InferenceEngine> MakeEngine(
        EngineConfig config, InferenceEngine::Clock clock = nullptr) {
        auto model_result = QwenModel::Load(manifest_, file_);
        EXPECT_TRUE(model_result.status.ok())
            << model_result.status.message();
        auto tok_result =
            BpeTokenizer::FromConfig(testfixture::SmallTokenizerConfig());
        EXPECT_TRUE(tok_result.status.ok());
        return std::make_unique<InferenceEngine>(
            std::move(model_result.model),
            std::make_unique<BpeTokenizer>(std::move(tok_result.tokenizer)),
            config, std::move(clock));
    }

    static InferenceRequest MakeRequest(const std::string& id,
                                        uint32_t max_tokens = 5) {
        InferenceRequest r;
        r.request_id = id;
        r.tenant_scope = "tn_a";
        r.messages = {{Role::kUser, "hello"}};
        r.max_output_tokens = max_tokens;
        r.sampling.temperature = 0.0f;
        return r;
    }

    static std::vector<StreamEvent> Drain(EventChannel& ch) {
        std::vector<StreamEvent> events;
        while (auto e = ch.Pop()) events.push_back(std::move(*e));
        return events;
    }

    ModelManifest manifest_;
    std::string path_;
    SafetensorsFile file_;
};

TEST_F(EngineTest, CompletesRequestWithOrderedEvents) {
    auto engine = MakeEngine({});
    auto submit = engine->Submit(MakeRequest("req_1"));
    ASSERT_TRUE(submit.status.ok()) << submit.status.message();
    engine->RunUntilIdle();

    auto events = Drain(*submit.events);
    ASSERT_GE(events.size(), 3u);
    EXPECT_EQ(events.front().kind, StreamEvent::Kind::kStarted);
    EXPECT_EQ(events[events.size() - 2].kind, StreamEvent::Kind::kUsage);
    EXPECT_EQ(events.back().kind, StreamEvent::Kind::kCompleted);
    EXPECT_EQ(events.back().finish_reason, FinishReason::kLength);
    EXPECT_EQ(events.back().usage.output_tokens, 5u);
    EXPECT_GT(events.back().usage.input_tokens, 0u);

    // Monotonic per-request sequence numbers (spec §19.1).
    for (size_t i = 0; i < events.size(); ++i) {
        EXPECT_EQ(events[i].sequence, i);
        EXPECT_EQ(events[i].request_id, "req_1");
    }
}

TEST_F(EngineTest, OutputIsDeterministicAcrossRuns) {
    auto run = [&]() {
        auto engine = MakeEngine({});
        auto submit = engine->Submit(MakeRequest("req_d", 10));
        EXPECT_TRUE(submit.status.ok());
        engine->RunUntilIdle();
        std::string text;
        for (auto& e : Drain(*submit.events)) {
            if (e.kind == StreamEvent::Kind::kOutputTextDelta) text += e.text;
        }
        return text;
    };
    EXPECT_EQ(run(), run());
}

TEST_F(EngineTest, RunsConcurrentSequences) {
    EngineConfig cfg;
    cfg.max_sequences = 4;
    auto engine = MakeEngine(cfg);
    std::vector<std::shared_ptr<EventChannel>> channels;
    for (int i = 0; i < 4; ++i) {
        auto submit =
            engine->Submit(MakeRequest("req_" + std::to_string(i), 4));
        ASSERT_TRUE(submit.status.ok());
        channels.push_back(submit.events);
    }
    engine->Step();
    EXPECT_EQ(engine->active_sequences(), 4u);
    engine->RunUntilIdle();
    for (auto& ch : channels) {
        auto events = Drain(*ch);
        EXPECT_EQ(events.back().kind, StreamEvent::Kind::kCompleted);
        EXPECT_EQ(events.back().finish_reason, FinishReason::kLength);
    }
}

TEST_F(EngineTest, CancelQueuedEmitsCancelledError) {
    EngineConfig cfg;
    cfg.max_sequences = 1;
    auto engine = MakeEngine(cfg);
    auto first = engine->Submit(MakeRequest("req_a", 3));
    auto second = engine->Submit(MakeRequest("req_b", 3));
    ASSERT_TRUE(first.status.ok());
    ASSERT_TRUE(second.status.ok());

    engine->Cancel("req_b");  // still queued
    engine->RunUntilIdle();

    auto events = Drain(*second.events);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].kind, StreamEvent::Kind::kError);
    EXPECT_EQ(events[0].error.code(), ErrorCode::kRequestCancelled);

    EXPECT_EQ(Drain(*first.events).back().kind,
              StreamEvent::Kind::kCompleted);
}

TEST_F(EngineTest, CancelActiveFinishesWithCancelledReason) {
    auto engine = MakeEngine({});
    auto submit = engine->Submit(MakeRequest("req_c", 100));
    ASSERT_TRUE(submit.status.ok());
    engine->Step();  // admit + first decode
    engine->Step();
    engine->Cancel("req_c");
    engine->RunUntilIdle();

    auto events = Drain(*submit.events);
    EXPECT_EQ(events.back().kind, StreamEvent::Kind::kCompleted);
    EXPECT_EQ(events.back().finish_reason, FinishReason::kCancelled);
    EXPECT_LT(events.back().usage.output_tokens, 100u);
}

TEST_F(EngineTest, DeadlinePassedInQueueFails) {
    int64_t fake_now = 1000;
    EngineConfig cfg;
    cfg.max_sequences = 1;
    auto engine = MakeEngine(cfg, [&] { return fake_now; });

    auto blocker = engine->Submit(MakeRequest("req_block", 50));
    InferenceRequest late = MakeRequest("req_late", 3);
    late.deadline_unix_ms = 1500;
    auto late_submit = engine->Submit(late);
    ASSERT_TRUE(blocker.status.ok());
    ASSERT_TRUE(late_submit.status.ok());

    engine->Step();      // admits blocker only
    fake_now = 2000;     // late's deadline passes while queued
    engine->RunUntilIdle();

    auto events = Drain(*late_submit.events);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].kind, StreamEvent::Kind::kError);
    EXPECT_EQ(events[0].error.code(), ErrorCode::kDeadlineExceeded);
}

TEST_F(EngineTest, DeadlineDuringDecodeFinishesWithDeadlineReason) {
    int64_t fake_now = 1000;
    auto engine = MakeEngine({}, [&] { return fake_now; });
    InferenceRequest r = MakeRequest("req_dl", 100);
    r.deadline_unix_ms = 1500;
    auto submit = engine->Submit(r);
    ASSERT_TRUE(submit.status.ok());

    engine->Step();
    engine->Step();
    fake_now = 2000;
    engine->RunUntilIdle();

    auto events = Drain(*submit.events);
    EXPECT_EQ(events.back().kind, StreamEvent::Kind::kCompleted);
    EXPECT_EQ(events.back().finish_reason, FinishReason::kDeadline);
}

TEST_F(EngineTest, InfeasibleDeadlineRejectedAtSubmit) {
    int64_t fake_now = 5000;
    auto engine = MakeEngine({}, [&] { return fake_now; });
    InferenceRequest r = MakeRequest("req_past", 3);
    r.deadline_unix_ms = 1000;
    auto submit = engine->Submit(r);
    EXPECT_FALSE(submit.status.ok());
    EXPECT_EQ(submit.status.code(), ErrorCode::kDeadlineRejected);
}

TEST_F(EngineTest, TokenBudgetRejectedAtSubmit) {
    auto engine = MakeEngine({});
    InferenceRequest r = MakeRequest("req_big", 4096);
    auto submit = engine->Submit(r);
    EXPECT_FALSE(submit.status.ok());
    EXPECT_EQ(submit.status.code(), ErrorCode::kContextLengthExceeded);
}

TEST_F(EngineTest, QueueFullRejectsWithResourceExhausted) {
    EngineConfig cfg;
    cfg.scheduler.max_queue = 1;
    auto engine = MakeEngine(cfg);
    ASSERT_TRUE(engine->Submit(MakeRequest("req_1", 3)).status.ok());
    auto second = engine->Submit(MakeRequest("req_2", 3));
    EXPECT_FALSE(second.status.ok());
    EXPECT_EQ(second.status.code(), ErrorCode::kResourceExhausted);
}

TEST_F(EngineTest, DrainRejectsNewRequests) {
    auto engine = MakeEngine({});
    engine->StartDrain();
    auto submit = engine->Submit(MakeRequest("req_drained", 3));
    EXPECT_FALSE(submit.status.ok());
    EXPECT_EQ(submit.status.code(), ErrorCode::kEngineDraining);
    engine->Resume();
    EXPECT_TRUE(engine->Submit(MakeRequest("req_ok", 3)).status.ok());
    engine->RunUntilIdle();
}

TEST_F(EngineTest, SlowConsumerIsCancelledWithTerminalError) {
    EngineConfig cfg;
    cfg.event_channel_capacity = 2;  // started + 1 delta, then overflow
    auto engine = MakeEngine(cfg);
    auto submit = engine->Submit(MakeRequest("req_slow", 100));
    ASSERT_TRUE(submit.status.ok());
    engine->RunUntilIdle();  // consumer never pops

    auto events = Drain(*submit.events);
    ASSERT_FALSE(events.empty());
    EXPECT_EQ(events.back().kind, StreamEvent::Kind::kError);
    EXPECT_EQ(events.back().error.code(), ErrorCode::kStreamConsumerSlow);
    EXPECT_FALSE(events.back().error.retryable());
}

TEST_F(EngineTest, StopSequenceEndsGenerationWithoutLeaking) {
    auto engine = MakeEngine({});
    InferenceRequest r = MakeRequest("req_stop", 50);
    // The tiny model echoes the last prompt token, so pick the first
    // generated character as the stop sequence: generation must stop
    // immediately with empty output text.
    {
        auto probe = MakeEngine({});
        auto ps = probe->Submit(MakeRequest("probe", 2));
        ASSERT_TRUE(ps.status.ok());
        probe->RunUntilIdle();
        std::string text;
        for (auto& e : Drain(*ps.events)) {
            if (e.kind == StreamEvent::Kind::kOutputTextDelta) text += e.text;
        }
        ASSERT_FALSE(text.empty());
        r.stop_sequences = {text.substr(0, 1)};
    }
    auto submit = engine->Submit(r);
    ASSERT_TRUE(submit.status.ok());
    engine->RunUntilIdle();

    std::string out;
    auto events = Drain(*submit.events);
    for (auto& e : events) {
        if (e.kind == StreamEvent::Kind::kOutputTextDelta) out += e.text;
    }
    EXPECT_EQ(events.back().kind, StreamEvent::Kind::kCompleted);
    EXPECT_EQ(events.back().finish_reason, FinishReason::kStop);
    EXPECT_TRUE(out.empty());
    EXPECT_LT(events.back().usage.output_tokens, 50u);
}

TEST_F(EngineTest, DuplicateRequestIdRejected) {
    auto engine = MakeEngine({});
    ASSERT_TRUE(engine->Submit(MakeRequest("req_dup", 3)).status.ok());
    auto second = engine->Submit(MakeRequest("req_dup", 3));
    EXPECT_FALSE(second.status.ok());
    engine->RunUntilIdle();
}

}  // namespace
}  // namespace lykuro::nie
