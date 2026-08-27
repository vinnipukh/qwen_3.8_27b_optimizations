// bench_gemv_dp4a.cpp — Direct speedup benchmark of cooperative DP4A GEMV vs real stock DP4A
// Across all 8 canonical shapes. Emits JSON array to stdout with speedup field.
// Compares real_stock_dp4a (quantize+vec_dot) vs custom cooperative 8-thread DP4A GEMV.

#include "ref_cpu.h"
#include "bench.h"
#include "hip_helpers.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemv_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemv_iq4xs_dp4a_gfx1100_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("[\n");
    bool first = true;
    std::mt19937 rng(42);

    for (int i = 0; i < NUM_CANONICAL_SHAPES; ++i) {
        auto &sh = CANONICAL_SHAPES[i];
        int64_t K = sh.K, N = sh.N;
        const char* name = sh.name;
        int64_t blocks_per_row = K / QK_K;
        size_t W_bytes = N * blocks_per_row * sizeof(block_iq4_xs);
        size_t X_bytes = K * sizeof(float);
        size_t Y_bytes = N * sizeof(float);
        size_t total_bytes = W_bytes + X_bytes + Y_bytes;

        std::vector<block_iq4_xs> h_W;
        gen_iq4xs_weights(h_W, K, N, 12345 + i * 100);
        std::vector<float> h_x(K);
        std::normal_distribution<float> g(0,1);
        for (int64_t k = 0; k < K; ++k) h_x[k] = g(rng);

        block_iq4_xs *dW = nullptr;
        float *dx = nullptr, *dy = nullptr;
        HIP_CHECK(hipMalloc(&dW, W_bytes));
        HIP_CHECK(hipMalloc(&dx, X_bytes));
        HIP_CHECK(hipMalloc(&dy, Y_bytes));
        HIP_CHECK(hipMemcpy(dW, h_W.data(), W_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dx, h_x.data(), X_bytes, hipMemcpyHostToDevice));

        auto stock = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_stock_dp4a_gpu(dW, dx, dy, K, N, s)); };
        auto coop  = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_dp4a_gfx1100_gpu(dW, dx, dy, K, N, s)); };

        BenchStats s_stock = bench_hip_event(stock, 0, 50, 200, total_bytes);
        BenchStats s_coop  = bench_hip_event(coop, 0, 50, 200, total_bytes);

        double speedup = s_stock.median_us > 0 ? s_stock.median_us / s_coop.median_us : 0.0;

        if (!first) printf(",\n");
        first = false;
        printf("  {\n");
        printf("    \"op\": \"gemv_iq4xs_dp4a_coop\",\n");
        printf("    \"shape\": \"%s\",\n", name);
        printf("    \"K\": %lld,\n", (long long)K);
        printf("    \"N\": %lld,\n", (long long)N);
        printf("    \"M\": 1,\n");
        printf("    \"bytes\": %zu,\n", total_bytes);
        printf("    \"real_dp4a_median_us\": %.3f,\n", s_stock.median_us);
        printf("    \"real_dp4a_p95_us\": %.3f,\n", s_stock.p95_us);
        printf("    \"real_dp4a_gb_s\": %.2f,\n", s_stock.gb_s);
        printf("    \"coop_dp4a_median_us\": %.3f,\n", s_coop.median_us);
        printf("    \"coop_dp4a_p95_us\": %.3f,\n", s_coop.p95_us);
        printf("    \"coop_dp4a_gb_s\": %.2f,\n", s_coop.gb_s);
        printf("    \"speedup\": %.3f,\n", speedup);
        printf("    \"speedup_vs_real_dp4a\": %.3f,\n", speedup);
        printf("    \"winner\": \"%s\",\n", speedup > 1.0 ? "coop_dp4a" : "real_dp4a_stock");
        printf("    \"note\": \"coop 8-thread per 256SB, 32 rows/block, DP4A v_dot4, LDS[32][33] padded, launch_bounds(256,4)\"\n");
        printf("  }");

        HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy));
    }
    printf("\n]\n");
    fprintf(stderr, "\n=== GEMV DP4A Coop vs Real Stock DP4A — target >1.2x speedup (decode 40-45 t/s) ===\n");
    fprintf(stderr, "Kernel: gemv_iq4xs_dp4a_coop_kernel (256 threads/block, 8 lanes per row, WARP_SIZE=32, 128-bit qs via ulong2)\n");
    fprintf(stderr, "Occupancy: __launch_bounds__(256,4) + amdgpu_flat_work_group_size(256,256) => <=64 VGPRs, 16 waves/SIMD\n");
    return 0;
}
