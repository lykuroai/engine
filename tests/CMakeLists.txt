find_package(GTest REQUIRED)

add_executable(unit_tests
    unit/json_test.cpp
    unit/sha256_test.cpp
    unit/manifest_test.cpp
    unit/safetensors_test.cpp
    unit/tokenizer_fixture.cpp
    unit/tokenizer_test.cpp
    unit/prompt_template_test.cpp
    unit/dtype_test.cpp
)
target_link_libraries(unit_tests
    PRIVATE lykuro_core lykuro_model lykuro_security lykuro_backend_cpu
            lykuro_options GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests)
