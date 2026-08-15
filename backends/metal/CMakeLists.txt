enable_language(OBJCXX)

add_library(lykuro_backend_metal STATIC
    metal_backend.mm
    qwen_metal_model.mm
    qwen_metal_fast.mm
)
set_target_properties(lykuro_backend_metal PROPERTIES
    OBJCXX_STANDARD 20)
target_compile_options(lykuro_backend_metal PRIVATE -fobjc-arc)
target_include_directories(lykuro_backend_metal PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_backend_metal
    PUBLIC lykuro_core lykuro_model lykuro_backend_cpu
           "-framework Foundation"
           "-framework Metal"
           "-framework MetalPerformanceShaders"
           "-framework MetalPerformanceShadersGraph")
