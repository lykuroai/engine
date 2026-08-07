add_executable(sign_artifact sign_artifact.cpp)
target_link_libraries(sign_artifact
    PRIVATE lykuro_security lykuro_core lykuro_options)

add_executable(bench_decode bench_decode.cpp)
target_link_libraries(bench_decode
    PRIVATE lykuro_model lykuro_core lykuro_security lykuro_backend_cpu
            lykuro_options)
if(LYKURO_ENABLE_CUDA)
    target_compile_definitions(bench_decode PRIVATE LYKURO_HAVE_CUDA)
    target_link_libraries(bench_decode PRIVATE lykuro_backend_cuda)
endif()

add_executable(verify_reference verify_reference.cpp)
target_link_libraries(verify_reference
    PRIVATE lykuro_model lykuro_core lykuro_security lykuro_backend_cpu
            lykuro_options)
if(LYKURO_ENABLE_CUDA)
    target_compile_definitions(verify_reference PRIVATE LYKURO_HAVE_CUDA)
    target_link_libraries(verify_reference PRIVATE lykuro_backend_cuda)
endif()
