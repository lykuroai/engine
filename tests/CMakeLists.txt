find_package(GTest REQUIRED)

add_executable(unit_tests
    unit/config_test.cpp
    unit/json_test.cpp
    unit/sha256_test.cpp
    unit/signature_test.cpp
    unit/manifest_test.cpp
    unit/safetensors_test.cpp
    unit/tokenizer_fixture.cpp
    unit/tokenizer_test.cpp
    unit/prompt_template_test.cpp
    unit/dtype_test.cpp
    unit/sampler_test.cpp
    unit/stop_test.cpp
    unit/qwen_model_test.cpp
    golden/golden_test.cpp
    unit/artifact_loader_test.cpp
    unit/scheduler_test.cpp
    unit/event_channel_test.cpp
    unit/metrics_test.cpp
    testutil/tiny_model.cpp
    integration/engine_test.cpp
)
target_link_libraries(unit_tests
    PRIVATE lykuro_core lykuro_engine lykuro_model lykuro_security
            lykuro_backend_cpu lykuro_options GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)

if(LYKURO_ENABLE_CUDA)
    add_executable(cuda_tests
        cuda/cuda_parity_test.cpp
        testutil/tiny_model.cpp
        unit/tokenizer_fixture.cpp
    )
    target_link_libraries(cuda_tests
        PRIVATE lykuro_backend_cuda lykuro_engine lykuro_model
                lykuro_security lykuro_backend_cpu lykuro_options
                GTest::gtest GTest::gtest_main)
    gtest_discover_tests(cuda_tests)
endif()

if(LYKURO_ENABLE_GRPC)
    add_executable(grpc_tests
        integration/grpc_server_test.cpp
        integration/mtls_test.cpp
        testutil/tiny_model.cpp
        unit/tokenizer_fixture.cpp
    )
    target_compile_definitions(grpc_tests PRIVATE
        LYKURO_TEST_CERT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/certs")
    target_link_libraries(grpc_tests
        PRIVATE lykuro_api lykuro_engine lykuro_model lykuro_security
                lykuro_backend_cpu lykuro_options
                GTest::gtest GTest::gtest_main)
    gtest_discover_tests(grpc_tests)
endif()
