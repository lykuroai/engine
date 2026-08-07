add_executable(native-engine
    main.cpp
)
target_link_libraries(native-engine
    PRIVATE lykuro_core lykuro_model lykuro_security lykuro_observability
            lykuro_backend_cpu lykuro_options)
