#include "observability/log.h"

#include <chrono>
#include <cstdio>
#include <mutex>

namespace lykuro::nie {

namespace {

std::mutex g_log_mutex;

std::string_view LevelName(LogLevel level) {
    switch (level) {
        case LogLevel::kDebug: return "DEBUG";
        case LogLevel::kInfo: return "INFO";
        case LogLevel::kWarn: return "WARN";
        case LogLevel::kError: return "ERROR";
    }
    return "INFO";
}

void AppendJsonString(std::string& out, std::string_view value) {
    out.push_back('"');
    for (char c : value) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

}  // namespace

Logger& Logger::Get() {
    static Logger logger;
    return logger;
}

void Logger::Log(LogLevel level, std::string_view component,
                 std::string_view event,
                 const std::map<std::string, std::string>& string_fields,
                 const std::map<std::string, int64_t>& numeric_fields) {
    if (level < level_) return;

    const auto now = std::chrono::system_clock::now();
    const int64_t unix_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch())
            .count();

    std::string line = "{\"unix_ms\":" + std::to_string(unix_ms);
    line += ",\"level\":";
    AppendJsonString(line, LevelName(level));
    line += ",\"component\":";
    AppendJsonString(line, component);
    line += ",\"event\":";
    AppendJsonString(line, event);
    for (const auto& [key, value] : string_fields) {
        line += ",";
        AppendJsonString(line, key);
        line += ":";
        AppendJsonString(line, value);
    }
    for (const auto& [key, value] : numeric_fields) {
        line += ",";
        AppendJsonString(line, key);
        line += ":" + std::to_string(value);
    }
    line += "}\n";

    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::fwrite(line.data(), 1, line.size(), stderr);
}

}  // namespace lykuro::nie
