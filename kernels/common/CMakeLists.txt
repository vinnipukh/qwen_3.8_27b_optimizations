add_library(kernels_common INTERFACE)
target_include_directories(kernels_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(kernels_common INTERFACE hip::device)
