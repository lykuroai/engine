add_library(lykuro_security STATIC
    sha256.cpp
)
target_include_directories(lykuro_security PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(lykuro_security PRIVATE lykuro_options)
