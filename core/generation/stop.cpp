#include "core/generation/stop.h"

#include <algorithm>

namespace lykuro::nie {

Status StopMatcher::Validate(const std::vector<std::string>& stops,
                             const StopLimits& limits) {
    if (stops.size() > limits.max_sequences) {
        return Status(ErrorCode::kInvalidRequest,
                      "too many stop sequences", "admission");
    }
    for (const std::string& s : stops) {
        if (s.empty() || s.size() > limits.max_sequence_bytes) {
            return Status(ErrorCode::kInvalidRequest,
                          "stop sequence length out of range", "admission");
        }
    }
    return Status::Ok();
}

StopMatcher::StopMatcher(std::vector<std::string> stops)
    : stops_(std::move(stops)) {}

std::string StopMatcher::Feed(std::string_view text) {
    if (stopped_) return {};
    pending_.append(text);

    // Check for a completed stop sequence.
    size_t stop_at = std::string::npos;
    for (const std::string& s : stops_) {
        size_t pos = pending_.find(s);
        if (pos != std::string::npos) {
            stop_at = std::min(stop_at, pos);
        }
    }
    if (stop_at != std::string::npos) {
        stopped_ = true;
        std::string out = pending_.substr(0, stop_at);
        pending_.clear();
        return out;
    }

    // Withhold the longest tail that is a proper prefix of any stop
    // sequence; everything before it is safe to emit.
    size_t hold = 0;
    for (const std::string& s : stops_) {
        size_t max_check = std::min(pending_.size(), s.size() - 1);
        for (size_t len = max_check; len > hold; --len) {
            if (pending_.compare(pending_.size() - len, len, s, 0, len) == 0) {
                hold = len;
                break;
            }
        }
    }
    std::string out = pending_.substr(0, pending_.size() - hold);
    pending_.erase(0, pending_.size() - hold);
    return out;
}

std::string StopMatcher::Flush() {
    if (stopped_) return {};
    std::string out = std::move(pending_);
    pending_.clear();
    return out;
}

}  // namespace lykuro::nie
