#pragma once

#include <string>
#include <string_view>
#include <map>
#include <cstdint>

namespace lykuro::nie {

// Stable engine error codes (spec §21.2). Must stay in sync with
// api/proto/lykuro/nie/v1/common.proto.
enum class ErrorCode : int32_t {
    kOk = 0,
    kInvalidRequest,
    kAuthenticationFailed,
    kModelNotLoaded,
    kUnsupportedModel,
    kArtifactVerificationFailed,
    kContextLengthExceeded,
    kResourceExhausted,
    kCapacityExhausted,
    kDeadlineRejected,
    kDeadlineExceeded,
    kRequestCancelled,
    kEngineDraining,
    kGpuOom,
    kGpuUnhealthy,
    kInferenceFailed,
    kStreamConsumerSlow,
    kInternalError,
};

std::string_view ErrorCodeName(ErrorCode code);
bool IsRetryable(ErrorCode code);

// Content-free status object. `message` must be a static description and
// must never contain prompt/response text, file contents, paths, or
// credentials (spec §21.1). Numeric context goes into `details`.
class Status {
public:
    Status() = default;
    Status(ErrorCode code, std::string message, std::string component = {})
        : code_(code), message_(std::move(message)),
          component_(std::move(component)) {}

    static Status Ok() { return Status(); }

    bool ok() const { return code_ == ErrorCode::kOk; }
    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }
    const std::string& component() const { return component_; }
    bool retryable() const { return IsRetryable(code_); }

    Status& WithDetail(std::string_view key, int64_t value) {
        details_[std::string(key)] = value;
        return *this;
    }
    const std::map<std::string, int64_t>& details() const { return details_; }

private:
    ErrorCode code_ = ErrorCode::kOk;
    std::string message_;
    std::string component_;
    std::map<std::string, int64_t> details_;
};

}  // namespace lykuro::nie
