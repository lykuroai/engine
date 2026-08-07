#pragma once

#include <cstdint>
#include <string>

#include "core/engine/error.h"

namespace lykuro::nie {

enum class FinishReason {
    kStop,        // EOS or stop sequence
    kLength,      // max_output_tokens reached
    kCancelled,   // client/platform cancel
    kDeadline,    // deadline exceeded mid-generation
    kError,       // engine failure
};

std::string_view FinishReasonName(FinishReason reason);

struct UsageInfo {
    uint32_t input_tokens = 0;
    uint32_t output_tokens = 0;
    uint32_t total_tokens() const { return input_tokens + output_tokens; }
};

// Streaming events (spec §9.7). `sequence` is monotonic per request
// starting at 0; `timestamp_unix_ms` is stamped at emission.
struct StreamEvent {
    enum class Kind { kStarted, kOutputTextDelta, kUsage, kCompleted, kError };

    Kind kind = Kind::kStarted;
    std::string request_id;
    uint64_t sequence = 0;
    int64_t timestamp_unix_ms = 0;

    std::string text;            // kOutputTextDelta
    UsageInfo usage;             // kUsage / kCompleted
    FinishReason finish_reason = FinishReason::kStop;  // kCompleted
    Status error;                // kError
};

}  // namespace lykuro::nie
