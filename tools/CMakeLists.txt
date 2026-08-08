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
if(LYKURO_ENABLE_METAL)
    target_compile_definitions(bench_decode PRIVATE LYKURO_HAVE_METAL)
    target_link_libraries(bench_decode PRIVATE lykuro_backend_metal)
endif()

add_executable(soak_engine soak_engine.cpp)
target_link_libraries(soak_engine
    PRIVATE lykuro_engine lykuro_model lykuro_core lykuro_security
            lykuro_backend_cpu lykuro_options)
if(LYKURO_ENABLE_CUDA)
    target_compile_definitions(soak_engine PRIVATE LYKURO_HAVE_CUDA)
    target_link_libraries(soak_engine PRIVATE lykuro_backend_cuda)
endif()
if(LYKURO_ENABLE_METAL)
    target_compile_definitions(soak_engine PRIVATE LYKURO_HAVE_METAL)
    target_link_libraries(soak_engine PRIVATE lykuro_backend_metal)
endif()

add_executable(verify_reference verify_reference.cpp)
target_link_libraries(verify_reference
    PRIVATE lykuro_model lykuro_core lykuro_security lykuro_backend_cpu
            lykuro_options)
if(LYKURO_ENABLE_CUDA)
    target_compile_definitions(verify_reference PRIVATE LYKURO_HAVE_CUDA)
    target_link_libraries(verify_reference PRIVATE lykuro_backend_cuda)
endif()
if(LYKURO_ENABLE_METAL)
    target_compile_definitions(verify_reference PRIVATE LYKURO_HAVE_METAL)
    target_link_libraries(verify_reference PRIVATE lykuro_backend_metal)
endif()
