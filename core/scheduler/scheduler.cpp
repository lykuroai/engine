#include "core/scheduler/scheduler.h"

#include <algorithm>
#include <cmath>

namespace lykuro::nie {

namespace {
constexpr const char kComponent[] = "scheduler";
}

Status Scheduler::Enqueue(ScheduledRequest request, int64_t now_unix_ms) {
    if (request.request_id.empty()) {
        return Status(ErrorCode::kInvalidRequest, "request_id required",
                      kComponent);
    }
    if (request.deadline_unix_ms != 0 &&
        request.deadline_unix_ms <= now_unix_ms) {
        return Status(ErrorCode::kDeadlineRejected,
                      "deadline is not feasible", kComponent);
    }
    request.priority = std::min(request.priority, config_.priority_ceiling);
    request.enqueued_unix_ms = now_unix_ms;

    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= config_.max_queue) {
        return Status(ErrorCode::kResourceExhausted, "admission queue full",
                      kComponent)
            .WithDetail("max_queue", int64_t(config_.max_queue));
    }
    for (const auto& queued : queue_) {
        if (queued->request_id == request.request_id) {
            return Status(ErrorCode::kInvalidRequest,
                          "request_id already queued", kComponent);
        }
    }
    queue_.push_back(
        std::make_unique<ScheduledRequest>(std::move(request)));
    return Status::Ok();
}

int64_t Scheduler::EffectivePriority(const ScheduledRequest& r,
                                     int64_t now_unix_ms) const {
    int64_t waited = std::max<int64_t>(0, now_unix_ms - r.enqueued_unix_ms);
    int64_t bonus = config_.aging_ms_per_point > 0
                        ? waited / config_.aging_ms_per_point
                        : 0;
    bonus = std::min<int64_t>(bonus, config_.aging_max_bonus);
    return int64_t(r.priority) + bonus;
}

std::unique_ptr<ScheduledRequest> Scheduler::PopNext(
    int64_t now_unix_ms,
    const std::map<std::string, uint32_t>& active_per_tenant,
    uint32_t max_active_sequences) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return nullptr;

    const uint32_t tenant_cap = std::max<uint32_t>(
        1, uint32_t(std::floor(double(max_active_sequences) *
                               config_.max_tenant_active_share)));

    auto at_cap = [&](const std::string& tenant) {
        auto it = active_per_tenant.find(tenant);
        return it != active_per_tenant.end() && it->second >= tenant_cap;
    };

    // Two passes: first respecting the fairness cap, then (when every
    // waiting tenant is capped or slots are otherwise idle) ignoring it so
    // capacity never sits unused.
    for (int pass = 0; pass < 2; ++pass) {
        auto best = queue_.end();
        int64_t best_priority = -1;
        for (auto it = queue_.begin(); it != queue_.end(); ++it) {
            if (pass == 0 && at_cap((*it)->tenant_scope)) continue;
            int64_t p = EffectivePriority(**it, now_unix_ms);
            // FIFO within equal effective priority: strict '>' keeps the
            // earliest-enqueued candidate.
            if (p > best_priority) {
                best_priority = p;
                best = it;
            }
        }
        if (best != queue_.end()) {
            std::unique_ptr<ScheduledRequest> out = std::move(*best);
            queue_.erase(best);
            return out;
        }
    }
    return nullptr;
}

bool Scheduler::Cancel(const std::string& request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if ((*it)->request_id == request_id) {
            queue_.erase(it);
            return true;
        }
    }
    return false;
}

std::vector<std::unique_ptr<ScheduledRequest>> Scheduler::PopExpired(
    int64_t now_unix_ms) {
    std::vector<std::unique_ptr<ScheduledRequest>> expired;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = queue_.begin(); it != queue_.end();) {
        if ((*it)->deadline_unix_ms != 0 &&
            (*it)->deadline_unix_ms <= now_unix_ms) {
            expired.push_back(std::move(*it));
            it = queue_.erase(it);
        } else {
            ++it;
        }
    }
    return expired;
}

size_t Scheduler::queue_depth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

}  // namespace lykuro::nie
