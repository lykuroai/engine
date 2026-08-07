#include "backends/cuda/cuda_backend.h"

#include <cuda_runtime.h>

namespace lykuro::nie {

namespace {
constexpr const char kComponent[] = "cuda_backend";
}

Status DiscoverCudaDevices(std::vector<CudaDeviceInfo>& out) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice) return Status::Ok();
    if (err != cudaSuccess) {
        return Status(ErrorCode::kGpuUnhealthy,
                      "cuda device enumeration failed", kComponent);
    }
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            return Status(ErrorCode::kGpuUnhealthy,
                          "cuda device query failed", kComponent);
        }
        CudaDeviceInfo info;
        info.device_id = i;
        info.name = prop.name;
        info.total_vram_bytes = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.driver_version = driver;
        info.runtime_version = runtime;

        // Free memory requires a context on the device.
        int prev = 0;
        cudaGetDevice(&prev);
        if (cudaSetDevice(i) == cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                info.free_vram_bytes = free_bytes;
            }
        }
        cudaSetDevice(prev);
        out.push_back(std::move(info));
    }
    return Status::Ok();
}

Status CheckCudaDevice(int device_id, CudaDeviceInfo& info_out) {
    std::vector<CudaDeviceInfo> devices;
    Status s = DiscoverCudaDevices(devices);
    if (!s.ok()) return s;
    for (CudaDeviceInfo& d : devices) {
        if (d.device_id == device_id) {
            if (d.compute_capability_major < 7) {
                return Status(ErrorCode::kUnsupportedModel,
                              "device compute capability below certified "
                              "minimum",
                              kComponent);
            }
            info_out = std::move(d);
            return Status::Ok();
        }
    }
    return Status(ErrorCode::kGpuUnhealthy, "requested device not present",
                  kComponent);
}

}  // namespace lykuro::nie
