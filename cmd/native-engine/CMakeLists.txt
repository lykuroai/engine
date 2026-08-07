add_executable(native-engine
    main.cpp
)
target_link_libraries(native-engine
    PRIVATE lykuro_core lykuro_model lykuro_security lykuro_observability
            lykuro_backend_cpu lykuro_options)
if(LYKURO_ENABLE_GRPC)
    target_compile_definitions(native-engine PRIVATE LYKURO_HAVE_GRPC)
    target_link_libraries(native-engine PRIVATE lykuro_api)
endif()
