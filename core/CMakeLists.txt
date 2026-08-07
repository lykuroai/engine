add_library(lykuro_core STATIC
    engine/error.cpp
    engine/json.cpp
    generation/sampler.cpp
    generation/stop.cpp
    scheduler/scheduler.cpp
)
target_include_directories(lykuro_core PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_core PRIVATE lykuro_options)

# The engine orchestrator sits above both the model-agnostic core and the
# model layer (it owns the loaded model + tokenizer), so it is a separate
# target to keep the dependency graph acyclic.
add_library(lykuro_engine STATIC
    engine/engine.cpp
)
target_include_directories(lykuro_engine PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_engine
    PUBLIC lykuro_core lykuro_model lykuro_observability
    PRIVATE lykuro_options)
