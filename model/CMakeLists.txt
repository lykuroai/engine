add_library(lykuro_model STATIC
    manifest/manifest.cpp
    loader/safetensors.cpp
    loader/artifact_loader.cpp
    tokenizer/bpe_tokenizer.cpp
    tokenizer/prompt_template.cpp
    architectures/qwen/qwen_model.cpp
    convert/hf_convert.cpp
)
target_include_directories(lykuro_model PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_model
    PUBLIC lykuro_core lykuro_security lykuro_backend_cpu
    PRIVATE lykuro_options)
