add_library(lykuro_observability STATIC
    log.cpp
    metrics.cpp
)
target_include_directories(lykuro_observability PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_observability
    PUBLIC lykuro_core
    PRIVATE lykuro_options)
