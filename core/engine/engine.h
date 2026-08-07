#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "core/engine/error.h"
#include "core/generation/sampler.h"
#include "core/generation/stop.h"
#include "core/scheduler/scheduler.h"
#include "core/streaming/event_channel.h"
#include "model/architectures/qwen/qwen_model.h"
#include "model/tokenizer/bpe_tokenizer.h"
#include "model/tokenizer/prompt_template.h"

namespace lykuro::nie {

struct EngineConfig {
    uint32_t max_sequences = 4;        // active decode slots
    size_t event_channel_capacity = 1024;
    uint32_t max_output_tokens_limit = 4096;
    size_t max_input_bytes = 1 << 20;
    SchedulerConfig scheduler;
    SamplingLimits sampling_limits;
    StopLimits stop_limits;
};

// Normalized inference request (Data API Generate/GenerateStream payload
// after transport decoding).
struct InferenceRequest {
    std::string request_id;
    std::string tenant_scope;
    std::string project_scope;
    uint32_t priority = 50;
    int64_t deadline_unix_ms = 0;

    std::vector<ChatMessage> messages;

    uint32_t max_output_tokens = 0;
    SamplingParams sampling;
    std::vector<std::string> stop_sequences;
};

enum class RequestState {
    kQueued,
    kActive,
    kCompleted,
    kCancelled,
    kFailed,
};

// Single-model inference engine (spec §5.2: one process, one active model).
// MVP execution model: callers drive Step() from one worker thread;
// Submit/Cancel are thread-safe. Continuous batching: new requests are
// admitted between decode iterations, finished sequences leave the batch
// immediately (spec §15.5).
class InferenceEngine {
public:
    using Clock = std::function<int64_t()>;  // unix ms

    InferenceEngine(std::unique_ptr<QwenModel> model,
                    std::unique_ptr<BpeTokenizer> tokenizer,
                    EngineConfig config, Clock clock = nullptr);
    ~InferenceEngine();

    struct SubmitResult {
        Status status;
        std::shared_ptr<EventChannel> events;  // set when status.ok()
    };

    // Validates, tokenizes, budget-checks, and enqueues the request.
    SubmitResult Submit(const InferenceRequest& request);

    // Idempotent cancel across queued and active stages (spec §19.2).
    void Cancel(const std::string& request_id);

    // Runs one scheduler+decode iteration. Returns true while work remains.
    bool Step();
    void RunUntilIdle();

    // Drain: stop admitting new requests (engine_draining), finish active.
    void StartDrain();
    void Resume();

    uint32_t active_sequences() const;
    size_t queue_depth() const { return scheduler_.queue_depth(); }

private:
    struct ActiveSequence {
        std::unique_ptr<ScheduledRequest> request;
        std::shared_ptr<EventChannel> channel;
        std::unique_ptr<QwenKvCache> cache;
        std::vector<float> logits;
        std::unique_ptr<Sampler> sampler;
        std::unique_ptr<StopMatcher> stop_matcher;
        std::string utf8_carry;
        uint64_t event_sequence = 0;
        uint32_t output_tokens = 0;
        int64_t started_unix_ms = 0;
    };

    void Emit(ActiveSequence& seq, StreamEvent event);
    void FinishSequence(size_t index, FinishReason reason);
    void FailSequence(size_t index, Status status);
    void RejectExpired(int64_t now);
    void AdmitPending(int64_t now);
    void DecodeIteration(int64_t now);
    void SetState(const std::string& request_id, RequestState state);
    bool IsCancelled(const std::string& request_id) const;

    std::unique_ptr<QwenModel> model_;
    std::unique_ptr<BpeTokenizer> tokenizer_;
    EngineConfig config_;
    Clock clock_;
    Scheduler scheduler_;

    mutable std::mutex state_mutex_;
    std::map<std::string, RequestState> registry_;
    std::map<std::string, bool> cancel_flags_;
    std::map<std::string, std::shared_ptr<EventChannel>> channels_;
    std::set<std::string> slow_consumers_;
    bool draining_ = false;

    // Touched only by the Step() thread.
    std::vector<ActiveSequence> active_;
};

}  // namespace lykuro::nie
