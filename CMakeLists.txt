cmake_minimum_required(VERSION 3.24)

project(lykuro-native-inference-engine
        VERSION 0.1.0
        DESCRIPTION "Lykuro Native Inference Engine"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LYKURO_ENABLE_CUDA "Build the CUDA backend" OFF)
option(LYKURO_ENABLE_GRPC "Build the gRPC API server" OFF)
option(LYKURO_BUILD_TESTS "Build tests" ON)
option(LYKURO_ENABLE_SANITIZERS "Enable ASan/UBSan in Debug builds" ON)

add_library(lykuro_options INTERFACE)
target_compile_options(lykuro_options INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(LYKURO_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(lykuro_options INTERFACE -fsanitize=address,undefined)
    target_link_options(lykuro_options INTERFACE -fsanitize=address,undefined)
endif()

add_subdirectory(core)
add_subdirectory(model)
add_subdirectory(backends)
add_subdirectory(security)
add_subdirectory(observability)
add_subdirectory(cmd/native-engine)

if(LYKURO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
