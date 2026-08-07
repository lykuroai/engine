add_executable(sign_artifact sign_artifact.cpp)
target_link_libraries(sign_artifact
    PRIVATE lykuro_security lykuro_core lykuro_options)
