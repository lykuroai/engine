#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/engine/error.h"
#include "core/generation/sampler.h"

namespace lykuro::nie {

// A queued generation request after tokenization. Content-free except for
// the token IDs the engine itself needs.
struct ScheduledRequest {
    std::string request_id;
    std::string tenant_scope;   // opaque; used for fairness only
    std::string project_scope;
    uint32_t priority = 50;     // 0..100 after clamping
    int64_t deadline_unix_ms = 0;  // 0 = none
    int64_t enqueued_unix_ms = 0;

    std::vector<uint32_t> prompt_tokens;
    uint32_t max_output_tokens = 0;
    SamplingParams sampling;
    std::vector<std::string> stop_sequences;
    bool allow_prefix_cache = false;
};

struct SchedulerConfig {
    size_t max_queue = 256;
    uint32_t priority_ceiling = 100;   // platform-imposed clamp
    // Aging: +1 effective priority per this many ms waited (bounded).
    int64_t aging_ms_per_point = 1000;
    uint32_t aging_max_bonus = 20;
    // Fairness: a tenant may hold at most this fraction of active slots
    // while other tenants are waiting (weighted fair share, spec §15.4).
    double max_tenant_active_share = 0.5;
};

// Bounded priority queue with aging, per-tenant fairness, deadline
// screening, and idempotent cancel (spec §15). Thread-safe.
class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config) : config_(config) {}

    // Admission-checks and enqueues. Fails with resource_exhausted when
    // the queue is full, deadline_rejected when the deadline already
    // passed. Priority is clamped to the configured ceiling.
    Status Enqueue(ScheduledRequest request, int64_t now_unix_ms);

    // Selects the next request to run: highest effective priority
    // (priority + age bonus), FIFO within equal effective priority.
    // Tenants at or above their active-share cap are skipped while other
    // tenants wait. Returns nullptr when nothing is eligible.
    std::unique_ptr<ScheduledRequest> PopNext(
        int64_t now_unix_ms,
        const std::map<std::string, uint32_t>& active_per_tenant,
        uint32_t max_active_sequences);

    // Removes a queued request. Idempotent: returns false when the id is
    // not queued (already popped, finished, or unknown).
    bool Cancel(const std::string& request_id);

    // Pops every queued request whose deadline has passed.
    std::vector<std::unique_ptr<ScheduledRequest>> PopExpired(
        int64_t now_unix_ms);

    size_t queue_depth() const;

private:
    int64_t EffectivePriority(const ScheduledRequest& r,
                              int64_t now_unix_ms) const;

    SchedulerConfig config_;
    mutable std::mutex mutex_;
    std::deque<std::unique_ptr<ScheduledRequest>> queue_;
};

}  // namespace lykuro::nie
