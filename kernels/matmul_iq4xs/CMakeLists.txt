cmake_minimum_required(VERSION 3.21)

# kernels/matmul_iq4xs — standalone IQ4_XS GEMV/GEMM playground for gfx1100
# Provides ref_cpu, stock comparator, custom GEMV/GEMM, and benchmarks.

# Common interface library to collapse repeated include and link dependencies
add_library(matmul_common_iface INTERFACE)
target_include_directories(matmul_common_iface INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
target_link_libraries(matmul_common_iface INTERFACE kernels_common)

# --- CPU reference library (no HIP) ---
add_library(matmul_ref_cpu STATIC
    ref_cpu.cpp
)
target_link_libraries(matmul_ref_cpu PUBLIC matmul_common_iface)
set_target_properties(matmul_ref_cpu PROPERTIES LINKER_LANGUAGE CXX)

# --- Stock HIP comparator object ---
add_library(matmul_stock_hip OBJECT
    stock_hip_comparator.hip
)
target_link_libraries(matmul_stock_hip PRIVATE matmul_common_iface)

# --- Baseline validation executable ---
add_executable(matmul_test_baseline
    test_stock_compare.cpp
    $<TARGET_OBJECTS:matmul_stock_hip>
)
target_link_libraries(matmul_test_baseline PRIVATE matmul_ref_cpu matmul_common_iface)
add_custom_target(matmul_stock_test DEPENDS matmul_test_baseline)

# --- Custom GEMV (gfx1100) ---
add_library(matmul_gemv_hip OBJECT impl_gemv_gfx1100.hip)
target_link_libraries(matmul_gemv_hip PRIVATE matmul_common_iface)

add_executable(test_gemv_compare test_gemv_compare.cpp $<TARGET_OBJECTS:matmul_gemv_hip> $<TARGET_OBJECTS:matmul_stock_hip>)
target_link_libraries(test_gemv_compare PRIVATE matmul_ref_cpu matmul_common_iface)

add_executable(bench_gemv bench_gemv.cpp $<TARGET_OBJECTS:matmul_gemv_hip> $<TARGET_OBJECTS:matmul_stock_hip>)
target_link_libraries(bench_gemv PRIVATE matmul_ref_cpu matmul_common_iface)

# --- Custom GEMM (WMMA) ---
add_library(matmul_gemm_hip OBJECT impl_gemm_wmma.hip)
target_link_libraries(matmul_gemm_hip PRIVATE matmul_common_iface)

add_executable(test_gemm_compare test_gemm_compare.cpp $<TARGET_OBJECTS:matmul_gemm_hip> $<TARGET_OBJECTS:matmul_stock_hip>)
target_link_libraries(test_gemm_compare PRIVATE matmul_ref_cpu matmul_common_iface)

add_executable(bench_gemm bench_gemm.cpp $<TARGET_OBJECTS:matmul_gemm_hip> $<TARGET_OBJECTS:matmul_stock_hip>)
target_link_libraries(bench_gemm PRIVATE matmul_ref_cpu matmul_common_iface)

# --- Unified sweep benchmark ---
add_executable(bench_matmul
    bench_matmul.cpp
    $<TARGET_OBJECTS:matmul_gemv_hip>
    $<TARGET_OBJECTS:matmul_gemm_hip>
    $<TARGET_OBJECTS:matmul_stock_hip>
)
target_link_libraries(bench_matmul PRIVATE matmul_ref_cpu matmul_common_iface)
