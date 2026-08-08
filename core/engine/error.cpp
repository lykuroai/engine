#include "core/engine/error.h"

namespace lykuro::nie {

std::string_view ErrorCodeName(ErrorCode code) {
    switch (code) {
        case ErrorCode::kOk: return "ok";
        case ErrorCode::kInvalidRequest: return "invalid_request";
        case ErrorCode::kAuthenticationFailed: return "authentication_failed";
        case ErrorCode::kModelNotLoaded: return "model_not_loaded";
        case ErrorCode::kUnsupportedModel: return "unsupported_model";
        case ErrorCode::kArtifactVerificationFailed:
            return "artifact_verification_failed";
        case ErrorCode::kContextLengthExceeded:
            return "context_length_exceeded";
        case ErrorCode::kResourceExhausted: return "resource_exhausted";
        case ErrorCode::kCapacityExhausted: return "capacity_exhausted";
        case ErrorCode::kDeadlineRejected: return "deadline_rejected";
        case ErrorCode::kDeadlineExceeded: return "deadline_exceeded";
        case ErrorCode::kRequestCancelled: return "request_cancelled";
        case ErrorCode::kEngineDraining: return "engine_draining";
        case ErrorCode::kGpuOom: return "gpu_oom";
        case ErrorCode::kGpuUnhealthy: return "gpu_unhealthy";
        case ErrorCode::kInferenceFailed: return "inference_failed";
        case ErrorCode::kStreamConsumerSlow: return "stream_consumer_slow";
        case ErrorCode::kInternalError: return "internal_error";
        case ErrorCode::kMetalBackendUnavailable:
            return "metal_backend_unavailable";
        case ErrorCode::kMetalDeviceUnsupported:
            return "metal_device_unsupported";
        case ErrorCode::kMetalLibraryInvalid:
            return "metal_library_invalid";
        case ErrorCode::kMetalPipelineCreationFailed:
            return "metal_pipeline_creation_failed";
        case ErrorCode::kMetalCommandFailed: return "metal_command_failed";
        case ErrorCode::kMetalMemoryPressure:
            return "metal_memory_pressure";
        case ErrorCode::kMetalOutOfMemory: return "metal_out_of_memory";
        case ErrorCode::kMetalExecutionTimeout:
            return "metal_execution_timeout";
        case ErrorCode::kMacProfileNotCertified:
            return "mac_profile_not_certified";
        case ErrorCode::kBackendAbiMismatch: return "backend_abi_mismatch";
    }
    return "internal_error";
}

// Retryability per spec §21.2. Conditional codes default to non-retryable;
// the caller may upgrade based on context (e.g. deadline_exceeded with a
// fresh deadline).
bool IsRetryable(ErrorCode code) {
    switch (code) {
        case ErrorCode::kModelNotLoaded:
        case ErrorCode::kResourceExhausted:
        case ErrorCode::kCapacityExhausted:
        case ErrorCode::kEngineDraining:
        case ErrorCode::kGpuUnhealthy:
        case ErrorCode::kMetalMemoryPressure:  // §21: retryable
            return true;
        default:
            return false;
    }
}

}  // namespace lykuro::nie
