#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/engine/error.h"

namespace lykuro::nie {

// CUDA device discovery and health (spec §17.2). Thin wrapper over the
// CUDA runtime API; the engine consumes this at load time and for the
// Capacity API.

struct CudaDeviceInfo {
    int device_id = 0;
    std::string name;
    uint64_t total_vram_bytes = 0;
    uint64_t free_vram_bytes = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int driver_version = 0;   // e.g. 12080
    int runtime_version = 0;
};

// Enumerates visible CUDA devices. Empty vector + ok status = no devices.
Status DiscoverCudaDevices(std::vector<CudaDeviceInfo>& out);

// Validates that `device_id` exists and meets the minimum compute
// capability (7.0 for BF16-convertible FP32 math in this backend).
Status CheckCudaDevice(int device_id, CudaDeviceInfo& info_out);

}  // namespace lykuro::nie
