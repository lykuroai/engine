add_library(lykuro_model STATIC
    manifest/manifest.cpp
    loader/safetensors.cpp
    tokenizer/bpe_tokenizer.cpp
    tokenizer/prompt_template.cpp
)
target_include_directories(lykuro_model PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_model
    PUBLIC lykuro_core lykuro_security
    PRIVATE lykuro_options)
