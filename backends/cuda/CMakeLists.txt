find_package(CUDAToolkit REQUIRED)

add_library(lykuro_backend_cuda STATIC
    cuda_backend.cu
    qwen_cuda_model.cu
    qwen_tp_model.cu
)
# Certified targets: sm_75 (Turing) and sm_86 (Ampere).
set_target_properties(lykuro_backend_cuda PROPERTIES
    CUDA_ARCHITECTURES "75;86"
    CUDA_SEPARABLE_COMPILATION OFF)
target_include_directories(lykuro_backend_cuda PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_backend_cuda
    PUBLIC lykuro_core lykuro_model lykuro_backend_cpu CUDA::cudart)
