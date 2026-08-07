#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

struct StopLimits {
    size_t max_sequences = 8;
    size_t max_sequence_bytes = 64;
};

// Streaming stop-sequence matcher. Feed decoded text incrementally; it
// emits only text that can no longer be part of a stop sequence, so a stop
// string is never leaked to the client even when it arrives split across
// tokens.
class StopMatcher {
public:
    static Status Validate(const std::vector<std::string>& stops,
                           const StopLimits& limits);

    explicit StopMatcher(std::vector<std::string> stops);

    // Appends `text`; returns the bytes safe to emit now. Once a stop
    // sequence matches, `stopped()` turns true and nothing further is
    // emitted.
    std::string Feed(std::string_view text);

    // Flushes withheld text at end of generation when no stop matched.
    std::string Flush();

    bool stopped() const { return stopped_; }

private:
    std::vector<std::string> stops_;
    std::string pending_;
    bool stopped_ = false;
};

}  // namespace lykuro::nie
