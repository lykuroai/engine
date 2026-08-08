#pragma once

#include <cstdint>
#include <string>

#include "core/engine/error.h"

namespace lykuro::nie {

// Metal device discovery / capability (LYK-NIE-ADD-METAL-001 §9).
// MVP targets the system default device on Apple Silicon with unified
// memory; anything else is refused.

struct MetalDeviceInfo {
    std::string device_name;
    bool unified_memory = false;
    uint64_t recommended_working_set_bytes = 0;
    uint64_t current_allocated_bytes = 0;
};

// Inspects the system default MTLDevice. Fails with
// metal_backend_unavailable (mapped onto gpu_unhealthy) when no device
// exists, and metal_device_unsupported (unsupported_model) when unified
// memory is absent.
Status InspectMetalDevice(MetalDeviceInfo& out);

}  // namespace lykuro::nie
