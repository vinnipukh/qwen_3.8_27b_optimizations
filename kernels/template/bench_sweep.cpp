#include "block_iq4_xs.h"
#include "hip_helpers.h"
#include "bench.h"
#include <vector>
#include <cstdio>
#include <string>

// Forward declaration
hipError_t template_stub_gpu(const block_iq4_xs* d_src, float* d_dst, int64_t n_blocks, int warp_size, hipStream_t stream = 0);

int main(int argc, char** argv) {
    const std::vector<int64_t> block_counts = {1, 8, 64, 512};
    const std::vector<int> warp_sizes = {32, 64};

    printf("[\n");
    bool first = true;

    for (int64_t n_blocks : block_counts) {
        int64_t total_elements = n_blocks * QK_K;
        size_t src_bytes = n_blocks * sizeof(block_iq4_xs);
        size_t dst_bytes = total_elements * sizeof(float);
        size_t total_bytes = src_bytes + dst_bytes;

        block_iq4_xs* d_src = nullptr;
        float* d_dst = nullptr;
        HIP_CHECK(hipMalloc(&d_src, src_bytes));
        HIP_CHECK(hipMalloc(&d_dst, dst_bytes));
        HIP_CHECK(hipMemset(d_src, 0, src_bytes));

        for (int ws : warp_sizes) {
            auto launcher = [d_src, d_dst, n_blocks, ws](hipStream_t s) {
                HIP_CHECK(template_stub_gpu(d_src, d_dst, n_blocks, ws, s));
            };

            BenchStats stats = bench_hip_event(launcher, 0, 50, 200, total_bytes);

            if (!first) {
                printf(",\n");
            }
            first = false;

            printf("  {\n");
            printf("    \"op\": \"template_stub\",\n");
            printf("    \"n_blocks\": %lld,\n", (long long)n_blocks);\
            printf("    \"n_elements\": %lld,\n", (long long)total_elements);\
            printf("    \"warp_size\": %d,\n", ws);\
            printf("    \"median_us\": %.3f,\n", stats.median_us);\
            printf("    \"p95_us\": %.3f,\n", stats.p95_us);\
            printf("    \"min_us\": %.3f,\n", stats.min_us);\
            printf("    \"max_us\": %.3f,\n", stats.max_us);\
            printf("    \"mean_us\": %.3f,\n", stats.mean_us);\
            printf("    \"stdev_us\": %.3f,\n", stats.stdev_us);\
            printf("    \"count\": %d,\n", stats.count);\
            printf("    \"warmup\": %d,\n", stats.warmup);\
            printf("    \"gb_s\": %.2f\n", stats.gb_s);\
            printf("  }");
        }

        HIP_CHECK(hipFree(d_src));
        HIP_CHECK(hipFree(d_dst));
    }

    printf("\n]\n");
    return 0;
}
