# Correct demo op test executable
add_executable(demo_test
    test_compare.cpp
    ref_cpu.cpp
    impl.hip
)
target_link_libraries(demo_test PRIVATE kernels_common)

# Broken demo op test executable (deliberate bug for negative testing)
add_executable(demo_test_broken
    test_compare.cpp
    ref_cpu.cpp
    impl_broken.hip
)
target_compile_definitions(demo_test_broken PRIVATE TEST_BROKEN=1)
target_link_libraries(demo_test_broken PRIVATE kernels_common)

# Benchmark sweep executable
add_executable(demo_bench
    bench_sweep.cpp
    impl.hip
)
target_link_libraries(demo_bench PRIVATE kernels_common)
