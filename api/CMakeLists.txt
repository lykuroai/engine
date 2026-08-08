find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

set(PROTO_FILES
    proto/lykuro/nie/v1/common.proto
    proto/lykuro/nie/v1/data.proto
    proto/lykuro/nie/v1/control.proto
)

add_library(lykuro_api_proto STATIC ${PROTO_FILES})
target_link_libraries(lykuro_api_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_OUT_DIR})
target_include_directories(lykuro_api_proto PUBLIC ${PROTO_OUT_DIR})

protobuf_generate(
    TARGET lykuro_api_proto
    LANGUAGE cpp
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PROTOC_OUT_DIR ${PROTO_OUT_DIR})
protobuf_generate(
    TARGET lykuro_api_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
    IMPORT_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PROTOC_OUT_DIR ${PROTO_OUT_DIR})

add_library(lykuro_api STATIC
    server/grpc_server.cpp
)
target_include_directories(lykuro_api PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_api
    PUBLIC lykuro_api_proto lykuro_engine lykuro_model
    PRIVATE lykuro_options)
if(LYKURO_ENABLE_CUDA)
    target_compile_definitions(lykuro_api PUBLIC LYKURO_HAVE_CUDA)
    target_link_libraries(lykuro_api PUBLIC lykuro_backend_cuda)
endif()
if(LYKURO_ENABLE_METAL)
    target_compile_definitions(lykuro_api PUBLIC LYKURO_HAVE_METAL)
    target_link_libraries(lykuro_api PUBLIC lykuro_backend_metal)
endif()
