cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(kernels_playground LANGUAGES HIP CXX)

set(CMAKE_HIP_ARCHITECTURES "gfx1100" CACHE STRING "gfx1100 only - no amdgpu-arch auto-detect")
set(GPU_TARGETS "gfx1100" CACHE STRING "Target GPU architectures")
set(AMDGPU_TARGETS "gfx1100" CACHE STRING "Target GPU architectures")

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}" "/opt/rocm")
# Windows-native gate REQ-WIN-07: HIP SDK via HIP_PATH (Windows: C:/Program Files/AMD/ROCm/6.4) - with /opt/rocm fallback
find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip" "/opt/rocm/lib/cmake/hip")

# Ensure HIP offload arch is passed for HIP language files
add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1100>)

add_subdirectory(common)
add_subdirectory(template)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demo_iq4xs_dequant/CMakeLists.txt")
    add_subdirectory(demo_iq4xs_dequant)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/matmul_iq4xs/CMakeLists.txt")
    add_subdirectory(matmul_iq4xs)
endif()
