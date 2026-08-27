// bench_gemv.cpp — Microbenchmark for GEMV M=1 across canonical shapes and wave32
// Emits JSON array to stdout (parsed by benchmarks/tools/run_kernel_bench.py)
// Reports median_us, gb_s, and speedup vs stock comparator.

#include "ref_cpu.h"
#include "bench.h"
#include "hip_helpers.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemv_iq4xs_gfx1100_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemv_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);

int main() {
    std::vector<std::tuple<std::string, int64_t, int64_t>> shapes;
    for (int i = 0; i < NUM_CANONICAL_SHAPES; ++i) {
        shapes.emplace_back(CANONICAL_SHAPES[i].name, CANONICAL_SHAPES[i].K, CANONICAL_SHAPES[i].N);
    }

    printf("[\n");
    bool first = true;
    for (auto &sh : shapes) {
        std::string name = std::get<0>(sh);
        int64_t K = std::get<1>(sh);
        int64_t N = std::get<2>(sh);
        int64_t blocks_per_row = K / QK_K;
        int64_t total_blocks = N * blocks_per_row;
        size_t W_bytes = total_blocks * sizeof(block_iq4_xs);
        size_t X_bytes = K * sizeof(float);
        size_t Y_bytes = N * sizeof(float);
        size_t total_bytes = W_bytes + X_bytes + Y_bytes;

        std::vector<block_iq4_xs> h_W;
        gen_iq4xs_weights(h_W, K, N, 12345);
        std::vector<float> h_x(K);
        std::normal_distribution<float> g(0, 1);
        std::mt19937 rng2(42);
        for (int64_t i = 0; i < K; ++i) h_x[i] = g(rng2);

        block_iq4_xs* dW = nullptr;
        float *dx = nullptr, *dy = nullptr;
        HIP_CHECK(hipMalloc(&dW, W_bytes));
        HIP_CHECK(hipMalloc(&dx, X_bytes));
        HIP_CHECK(hipMalloc(&dy, Y_bytes));
        HIP_CHECK(hipMemcpy(dW, h_W.data(), W_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dx, h_x.data(), X_bytes, hipMemcpyHostToDevice));

        // Bench stock
        auto stock_launch = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_stock_gpu(dW, dx, dy, K, N, s)); };
        BenchStats stock = bench_hip_event(stock_launch, 0, 50, 200, total_bytes);
        // Bench custom gfx1100
        auto gfx_launch = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_gfx1100_gpu(dW, dx, dy, K, N, s)); };
        BenchStats gfx = bench_hip_event(gfx_launch, 0, 50, 200, total_bytes);

        double speedup = stock.median_us / gfx.median_us;
        double bw_stock = stock.gb_s;
        double bw_gfx = gfx.gb_s;

        if (!first) printf(",\n");
        first = false;
        printf("  {\n");
        printf("    \"op\": \"gemv_iq4xs\",\n");
        printf("    \"shape\": \"%s\",\n", name.c_str());
        printf("    \"K\": %lld,\n", (long long)K);
        printf("    \"N\": %lld,\n", (long long)N);
        printf("    \"M\": 1,\n");
        printf("    \"bytes\": %zu,\n", total_bytes);
        printf("    \"stock_median_us\": %.3f,\n", stock.median_us);
        printf("    \"stock_p95_us\": %.3f,\n", stock.p95_us);
        printf("    \"stock_gb_s\": %.2f,\n", bw_stock);
        printf("    \"gfx1100_median_us\": %.3f,\n", gfx.median_us);
        printf("    \"gfx1100_p95_us\": %.3f,\n", gfx.p95_us);
        printf("    \"gfx1100_gb_s\": %.2f,\n", bw_gfx);
        printf("    \"speedup\": %.3f,\n", speedup);
        printf("    \"winner\": \"%s\"\n", speedup > 1.0 ? "gfx1100" : "stock");
        printf("  }");

        HIP_CHECK(hipFree(dW));
        HIP_CHECK(hipFree(dx));
        HIP_CHECK(hipFree(dy));
    }
    printf("\n]\n");
    return 0;
}
