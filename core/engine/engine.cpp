#include "core/engine/engine.h"

#include <algorithm>
#include <chrono>

namespace lykuro::nie {

namespace {

constexpr const char kComponent[] = "engine";

int64_t SystemNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

std::string_view FinishReasonName(FinishReason reason) {
    switch (reason) {
        case FinishReason::kStop: return "stop";
        case FinishReason::kLength: return "length";
        case FinishReason::kCancelled: return "cancelled";
        case FinishReason::kDeadline: return "deadline";
        case FinishReason::kError: return "error";
    }
    return "error";
}

InferenceEngine::InferenceEngine(std::unique_ptr<QwenModel> model,
                                 std::unique_ptr<BpeTokenizer> tokenizer,
                                 EngineConfig config, Clock clock)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      config_(config),
      clock_(clock ? std::move(clock) : SystemNowMs),
      scheduler_(config.scheduler) {
    if (config_.metrics != nullptr) {
        MetricsRegistry& m = *config_.metrics;
        m_received_ = m.GetCounter("nie_requests_received_total",
                                   "Requests received by the engine");
        m_rejected_ = m.GetCounter("nie_requests_rejected_total",
                                   "Requests rejected at admission");
        m_admitted_ = m.GetCounter("nie_requests_admitted_total",
                                   "Requests admitted to a decode slot");
        m_completed_ = m.GetCounter("nie_requests_completed_total",
                                    "Requests finished successfully");
        m_failed_ = m.GetCounter("nie_requests_failed_total",
                                 "Requests finished with an error");
        m_cancelled_ = m.GetCounter("nie_requests_cancelled_total",
                                    "Requests cancelled");
        m_output_tokens_ = m.GetCounter("nie_output_tokens_total",
                                        "Output tokens generated");
    }
}

InferenceEngine::~InferenceEngine() {
    // Safe-shutdown ordering (spec §29.2): cancel queued and active work,
    // then release sequences (KV goes with them).
    for (size_t i = active_.size(); i > 0; --i) {
        FinishSequence(i - 1, FinishReason::kCancelled);
    }
}

void InferenceEngine::SetState(const std::string& request_id,
                               RequestState state) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    registry_[request_id] = state;
}

bool InferenceEngine::IsCancelled(const std::string& request_id) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = cancel_flags_.find(request_id);
    return it != cancel_flags_.end() && it->second;
}

InferenceEngine::SubmitResult InferenceEngine::Submit(
    const InferenceRequest& request) {
    if (m_received_ != nullptr) m_received_->Increment();
    SubmitResult result = SubmitImpl(request);
    if (!result.status.ok() && m_rejected_ != nullptr) {
        m_rejected_->Increment();
    }
    return result;
}

Status InferenceEngine::CountTokens(
    const std::vector<ChatMessage>& messages, uint32_t& tokens_out) const {
    std::vector<uint32_t> tokens;
    Status s = QwenChatTemplate::BuildPrompt(*tokenizer_, messages, tokens);
    if (!s.ok()) return s;
    tokens_out = uint32_t(tokens.size());
    return Status::Ok();
}

InferenceEngine::SubmitResult InferenceEngine::SubmitImpl(
    const InferenceRequest& request) {
    SubmitResult result;
    const int64_t now = clock_();

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (draining_) {
            result.status = Status(ErrorCode::kEngineDraining,
                                   "engine is draining", kComponent);
            return result;
        }
        if (registry_.count(request.request_id)) {
            result.status = Status(ErrorCode::kInvalidRequest,
                                   "request_id already known", kComponent);
            return result;
        }
    }

    if (request.max_output_tokens == 0 ||
        request.max_output_tokens > config_.max_output_tokens_limit) {
        result.status = Status(ErrorCode::kInvalidRequest,
                               "max_output_tokens out of range", kComponent);
        return result;
    }
    result.status =
        ValidateSamplingParams(request.sampling, config_.sampling_limits);
    if (!result.status.ok()) return result;
    result.status =
        StopMatcher::Validate(request.stop_sequences, config_.stop_limits);
    if (!result.status.ok()) return result;

    size_t input_bytes = 0;
    for (const ChatMessage& m : request.messages) {
        input_bytes += m.content.size();
    }
    if (input_bytes > config_.max_input_bytes) {
        result.status = Status(ErrorCode::kInvalidRequest,
                               "input exceeds byte limit", kComponent);
        return result;
    }

    ScheduledRequest scheduled;
    scheduled.request_id = request.request_id;
    scheduled.tenant_scope = request.tenant_scope;
    scheduled.project_scope = request.project_scope;
    scheduled.priority = request.priority;
    scheduled.deadline_unix_ms = request.deadline_unix_ms;
    scheduled.max_output_tokens = request.max_output_tokens;
    scheduled.sampling = request.sampling;
    scheduled.stop_sequences = request.stop_sequences;

    result.status = QwenChatTemplate::BuildPrompt(
        *tokenizer_, request.messages, scheduled.prompt_tokens);
    if (!result.status.ok()) return result;

    result.status = CheckTokenBudget(scheduled.prompt_tokens.size(),
                                     request.max_output_tokens,
                                     model_->config().max_context_tokens);
    if (!result.status.ok()) return result;

    auto channel =
        std::make_shared<EventChannel>(config_.event_channel_capacity);

    result.status = scheduler_.Enqueue(std::move(scheduled), now);
    if (!result.status.ok()) return result;

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        registry_[request.request_id] = RequestState::kQueued;
        cancel_flags_[request.request_id] = false;
        channels_.emplace(request.request_id, channel);
    }
    result.events = std::move(channel);
    return result;
}

bool InferenceEngine::Cancel(const std::string& request_id) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto it = cancel_flags_.find(request_id);
        if (it == cancel_flags_.end()) return false;  // idempotent
        it->second = true;
    }
    // Queued requests can be removed immediately; active ones are
    // collected at the next decode iteration.
    if (scheduler_.Cancel(request_id)) {
        std::shared_ptr<EventChannel> channel;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            registry_[request_id] = RequestState::kCancelled;
            auto ch = channels_.find(request_id);
            if (ch != channels_.end()) {
                channel = ch->second;
                channels_.erase(ch);
            }
        }
        if (channel) {
            StreamEvent e;
            e.kind = StreamEvent::Kind::kError;
            e.request_id = request_id;
            e.timestamp_unix_ms = clock_();
            e.error = Status(ErrorCode::kRequestCancelled,
                             "request cancelled while queued", kComponent);
            channel->TryPush(std::move(e));
            channel->Close();
        }
    }
    return true;
}

void InferenceEngine::StartDrain() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    draining_ = true;
}

void InferenceEngine::Resume() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    draining_ = false;
}

uint32_t InferenceEngine::active_sequences() const {
    return uint32_t(active_.size());
}

void InferenceEngine::Emit(ActiveSequence& seq, StreamEvent event) {
    event.request_id = seq.request->request_id;
    event.sequence = seq.event_sequence++;
    event.timestamp_unix_ms = clock_();
    if (!seq.channel->TryPush(std::move(event))) {
        // Slow consumer: stop generating for this request (spec §19.3).
        std::lock_guard<std::mutex> lock(state_mutex_);
        cancel_flags_[seq.request->request_id] = true;
        slow_consumers_.insert(seq.request->request_id);
    }
}

void InferenceEngine::FinishSequence(size_t index, FinishReason reason) {
    ActiveSequence& seq = active_[index];
    const std::string request_id = seq.request->request_id;

    // Flush withheld text (UTF-8 carry is dropped: incomplete sequences
    // are never emitted; stop-matcher tail is flushed unless stopped).
    if (reason == FinishReason::kStop || reason == FinishReason::kLength) {
        std::string tail = seq.stop_matcher->Flush();
        if (!tail.empty()) {
            std::string carry;
            SplitUtf8Boundary(tail, carry);
            if (!tail.empty()) {
                StreamEvent delta;
                delta.kind = StreamEvent::Kind::kOutputTextDelta;
                delta.text = std::move(tail);
                Emit(seq, std::move(delta));
            }
        }
    }

    UsageInfo usage;
    usage.input_tokens = uint32_t(seq.request->prompt_tokens.size());
    usage.output_tokens = seq.output_tokens;

    StreamEvent usage_event;
    usage_event.kind = StreamEvent::Kind::kUsage;
    usage_event.usage = usage;
    Emit(seq, std::move(usage_event));

    StreamEvent completed;
    completed.kind = StreamEvent::Kind::kCompleted;
    completed.request_id = request_id;
    completed.sequence = seq.event_sequence++;
    completed.timestamp_unix_ms = clock_();
    completed.usage = usage;
    completed.finish_reason = reason;
    seq.channel->PushTerminal(std::move(completed));
    seq.channel->Close();

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        registry_[request_id] = reason == FinishReason::kCancelled
                                    ? RequestState::kCancelled
                                    : RequestState::kCompleted;
        channels_.erase(request_id);
    }
    if (reason == FinishReason::kCancelled) {
        if (m_cancelled_ != nullptr) m_cancelled_->Increment();
    } else if (m_completed_ != nullptr) {
        m_completed_->Increment();
    }
    if (m_output_tokens_ != nullptr) {
        m_output_tokens_->Increment(usage.output_tokens);
    }
    // Releases KV cache and all per-sequence state (spec §16.1).
    active_.erase(active_.begin() + long(index));
}

void InferenceEngine::FailSequence(size_t index, Status status) {
    ActiveSequence& seq = active_[index];
    const std::string request_id = seq.request->request_id;

    StreamEvent error;
    error.kind = StreamEvent::Kind::kError;
    error.request_id = request_id;
    error.sequence = seq.event_sequence++;
    error.timestamp_unix_ms = clock_();
    error.error = std::move(status);
    seq.channel->PushTerminal(std::move(error));
    seq.channel->Close();

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        registry_[request_id] = RequestState::kFailed;
        channels_.erase(request_id);
    }
    if (m_failed_ != nullptr) m_failed_->Increment();
    active_.erase(active_.begin() + long(index));
}

void InferenceEngine::RejectExpired(int64_t now) {
    for (auto& expired : scheduler_.PopExpired(now)) {
        std::shared_ptr<EventChannel> channel;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            registry_[expired->request_id] = RequestState::kFailed;
            auto ch = channels_.find(expired->request_id);
            if (ch != channels_.end()) {
                channel = ch->second;
                channels_.erase(ch);
            }
        }
        if (channel) {
            StreamEvent e;
            e.kind = StreamEvent::Kind::kError;
            e.request_id = expired->request_id;
            e.timestamp_unix_ms = now;
            e.error = Status(ErrorCode::kDeadlineExceeded,
                             "deadline passed while queued", kComponent);
            channel->TryPush(std::move(e));
            channel->Close();
        }
        if (m_failed_ != nullptr) m_failed_->Increment();
    }
}

void InferenceEngine::AdmitPending(int64_t now) {
    while (active_.size() < config_.max_sequences) {
        std::map<std::string, uint32_t> active_per_tenant;
        for (const auto& seq : active_) {
            ++active_per_tenant[seq.request->tenant_scope];
        }
        auto next = scheduler_.PopNext(now, active_per_tenant,
                                       config_.max_sequences);
        if (!next) break;

        const std::string request_id = next->request_id;
        std::shared_ptr<EventChannel> channel;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            auto ch = channels_.find(request_id);
            channel = ch != channels_.end() ? ch->second : nullptr;
        }
        if (!channel) continue;  // cancelled between queue and admission

        if (IsCancelled(request_id)) {
            StreamEvent e;
            e.kind = StreamEvent::Kind::kError;
            e.request_id = request_id;
            e.timestamp_unix_ms = now;
            e.error = Status(ErrorCode::kRequestCancelled,
                             "request cancelled while queued", kComponent);
            channel->TryPush(std::move(e));
            channel->Close();
            std::lock_guard<std::mutex> lock(state_mutex_);
            registry_[request_id] = RequestState::kCancelled;
            channels_.erase(request_id);
            continue;
        }

        ActiveSequence seq;
        seq.request = std::move(next);
        seq.channel = channel;
        seq.cache = std::make_unique<QwenKvCache>(
            model_->config(), model_->config().max_context_tokens);
        seq.sampler = std::make_unique<Sampler>(seq.request->sampling);
        seq.stop_matcher =
            std::make_unique<StopMatcher>(seq.request->stop_sequences);
        seq.started_unix_ms = now;

        SetState(request_id, RequestState::kActive);
        if (m_admitted_ != nullptr) m_admitted_->Increment();

        StreamEvent started;
        started.kind = StreamEvent::Kind::kStarted;
        Emit(seq, std::move(started));

        Status s = model_->Prefill(seq.request->prompt_tokens, *seq.cache,
                                   seq.logits);
        active_.push_back(std::move(seq));
        if (!s.ok()) {
            FailSequence(active_.size() - 1, std::move(s));
        }
    }
}

void InferenceEngine::DecodeIteration(int64_t now) {
    for (size_t i = active_.size(); i > 0; --i) {
        const size_t index = i - 1;
        ActiveSequence& seq = active_[index];
        const ScheduledRequest& req = *seq.request;

        bool slow = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            slow = slow_consumers_.count(req.request_id) > 0;
        }
        if (slow) {
            FailSequence(index,
                         Status(ErrorCode::kStreamConsumerSlow,
                                "stream consumer too slow", kComponent));
            continue;
        }
        if (IsCancelled(req.request_id)) {
            FinishSequence(index, FinishReason::kCancelled);
            continue;
        }
        if (req.deadline_unix_ms != 0 && req.deadline_unix_ms <= now) {
            FinishSequence(index, FinishReason::kDeadline);
            continue;
        }

        uint32_t token = 0;
        Status s = seq.sampler->Sample(seq.logits, token);
        if (!s.ok()) {
            FailSequence(index, std::move(s));
            continue;
        }
        ++seq.output_tokens;

        const auto& eos = model_->config().eos_token_ids;
        if (std::find(eos.begin(), eos.end(), token) != eos.end()) {
            FinishSequence(index, FinishReason::kStop);
            continue;
        }

        std::string bytes;
        s = tokenizer_->DecodeBytes({token}, bytes);
        if (!s.ok()) {
            FailSequence(index, std::move(s));
            continue;
        }
        std::string text = seq.utf8_carry + bytes;
        SplitUtf8Boundary(text, seq.utf8_carry);
        std::string emit_text = seq.stop_matcher->Feed(text);
        if (!emit_text.empty()) {
            StreamEvent delta;
            delta.kind = StreamEvent::Kind::kOutputTextDelta;
            delta.text = std::move(emit_text);
            Emit(seq, std::move(delta));
        }
        if (seq.stop_matcher->stopped()) {
            FinishSequence(index, FinishReason::kStop);
            continue;
        }
        if (seq.output_tokens >= req.max_output_tokens) {
            FinishSequence(index, FinishReason::kLength);
            continue;
        }

        s = model_->Decode(token, *seq.cache, seq.logits);
        if (!s.ok()) {
            FailSequence(index, std::move(s));
            continue;
        }
    }
}

bool InferenceEngine::Step() {
    const int64_t now = clock_();
    RejectExpired(now);
    AdmitPending(now);
    DecodeIteration(now);
    return !active_.empty() || scheduler_.queue_depth() > 0;
}

void InferenceEngine::RunUntilIdle() {
    while (Step()) {
    }
}

}  // namespace lykuro::nie
