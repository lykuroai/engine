#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backends/metal/metal_backend.h"

namespace lykuro::nie {

Status InspectMetalDevice(MetalDeviceInfo& out) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return Status(ErrorCode::kGpuUnhealthy,
                          "metal device unavailable", "metal_backend");
        }
        if (!device.hasUnifiedMemory) {
            return Status(ErrorCode::kUnsupportedModel,
                          "device lacks unified memory", "metal_backend");
        }
        out.device_name = device.name.UTF8String;
        out.unified_memory = true;
        out.recommended_working_set_bytes =
            device.recommendedMaxWorkingSetSize;
        out.current_allocated_bytes = device.currentAllocatedSize;
        return Status::Ok();
    }
}

}  // namespace lykuro::nie
