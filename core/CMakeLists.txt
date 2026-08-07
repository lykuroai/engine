add_library(lykuro_core STATIC
    engine/error.cpp
    engine/json.cpp
    generation/sampler.cpp
    generation/stop.cpp
)
target_include_directories(lykuro_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_core PRIVATE lykuro_options)
