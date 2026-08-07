#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace lykuro::nie {

enum class LogLevel { kDebug, kInfo, kWarn, kError };

// Content-free structured logger (spec §24.3). Values are restricted to
// identifiers and numbers by design: there is no API that accepts free-form
// request content, and callers must never pass prompt/response text,
// tokens, or credentials.
class Logger {
public:
    static Logger& Get();

    void SetLevel(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }

    void Log(LogLevel level, std::string_view component,
             std::string_view event,
             const std::map<std::string, std::string>& string_fields = {},
             const std::map<std::string, int64_t>& numeric_fields = {});

private:
    Logger() = default;
    LogLevel level_ = LogLevel::kInfo;
};

}  // namespace lykuro::nie
