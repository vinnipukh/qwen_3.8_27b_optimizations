add_executable(template_test
    test_compare.cpp
    ref_cpu.cpp
    impl.hip
)
target_link_libraries(template_test PRIVATE kernels_common)

add_executable(template_bench
    bench_sweep.cpp
    impl.hip
)
target_link_libraries(template_bench PRIVATE kernels_common)
