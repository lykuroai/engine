add_library(lykuro_backend_cpu STATIC
    cpu/cpu_backend.cpp
)
target_include_directories(lykuro_backend_cpu PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_backend_cpu
    PUBLIC lykuro_core
    PRIVATE lykuro_options)

if(LYKURO_ENABLE_CUDA)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "The CUDA backend is only supported on Linux.")
    endif()
    enable_language(CUDA)
    add_subdirectory(cuda)
endif()

if(LYKURO_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "The Metal backend is only supported on macOS.")
    endif()
    add_subdirectory(metal)
endif()
