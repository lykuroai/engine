add_executable(native-engine
    main.cpp
    generator.cpp
    http_api.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(native-engine
    PRIVATE lykuro_core lykuro_model lykuro_security lykuro_observability
            lykuro_backend_cpu lykuro_options Threads::Threads)
if(LYKURO_ENABLE_GRPC)
    target_compile_definitions(native-engine PRIVATE LYKURO_HAVE_GRPC)
    target_link_libraries(native-engine PRIVATE lykuro_api)
endif()
# The `run` subcommand (config-less standalone inference) can drive the GPU
# backends directly, so link whichever are built.
if(LYKURO_ENABLE_METAL)
    target_compile_definitions(native-engine PRIVATE LYKURO_HAVE_METAL)
    target_link_libraries(native-engine PRIVATE lykuro_backend_metal)
endif()
if(LYKURO_ENABLE_CUDA)
    target_compile_definitions(native-engine PRIVATE LYKURO_HAVE_CUDA)
    target_link_libraries(native-engine PRIVATE lykuro_backend_cuda)
endif()
