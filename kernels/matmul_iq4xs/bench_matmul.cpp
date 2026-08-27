// bench_matmul.cpp — Unified sweep for GEMV (M=1) + GEMM (M>>1) across all canonical shapes
// Produces JSON array for KERNEL-BENCH-DIFF.md and RunStore archival.

#include "ref_cpu.h"
#include "bench.h"
#include "hip_helpers.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemv_iq4xs_gfx1100_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemv_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemm_iq4xs_wmma_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);

int main() {
    int64_t Ms[] = {1, 16, 128, 512};
    printf("[\n");
    bool first = true;
    for (int si = 0; si < NUM_CANONICAL_SHAPES; ++si) {
        const char* name = CANONICAL_SHAPES[si].name;
        int64_t K = CANONICAL_SHAPES[si].K;
        int64_t N = CANONICAL_SHAPES[si].N;
        int64_t blocks_per_row = K / QK_K;
        int64_t total_blocks = N * blocks_per_row;
        size_t W_bytes = total_blocks * sizeof(block_iq4_xs);

        std::vector<block_iq4_xs> h_W;
        gen_iq4xs_weights(h_W, K, N, 9999 + si);

        block_iq4_xs* dW = nullptr;
        HIP_CHECK(hipMalloc(&dW, W_bytes));
        HIP_CHECK(hipMemcpy(dW, h_W.data(), W_bytes, hipMemcpyHostToDevice));

        for (int64_t M : Ms) {
            size_t X_bytes = K * M * sizeof(float);
            size_t Y_bytes = N * M * sizeof(float);
            size_t total_bytes = W_bytes + X_bytes + Y_bytes;
            double flops = 2.0 * (double)N * (double)M * (double)K;

            std::vector<float> h_X(K * M);
            std::normal_distribution<float> g(0, 1);
            std::mt19937 rng2(42 + M);
            for (int64_t i = 0; i < K * M; ++i) h_X[i] = g(rng2);

            float *dX = nullptr, *dY = nullptr;
            HIP_CHECK(hipMalloc(&dX, X_bytes));
            HIP_CHECK(hipMalloc(&dY, Y_bytes));
            HIP_CHECK(hipMemcpy(dX, h_X.data(), X_bytes, hipMemcpyHostToDevice));

            BenchStats stock, gfx;
            if (M == 1) {
                auto sl = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_stock_gpu(dW, dX, dY, K, N, s)); };
                stock = bench_hip_event(sl, 0, 5, 20, total_bytes);
                auto gl = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_gfx1100_gpu(dW, dX, dY, K, N, s)); };
                gfx = bench_hip_event(gl, 0, 5, 20, total_bytes);
            } else {
                auto sl = [&](hipStream_t s){ HIP_CHECK(gemm_iq4xs_stock_gpu(dW, dX, dY, K, N, M, s)); };
                stock = bench_hip_event(sl, 0, 5, 20, total_bytes);
                auto gl = [&](hipStream_t s){ HIP_CHECK(gemm_iq4xs_wmma_gpu(dW, dX, dY, K, N, M, s)); };
                gfx = bench_hip_event(gl, 0, 5, 20, total_bytes);
            }
            double tflops_stock = flops / (stock.median_us * 1e-6) / 1e12;
            double tflops_gfx = flops / (gfx.median_us * 1e-6) / 1e12;
            double speedup = stock.median_us / gfx.median_us;

            if (!first) printf(",\n");
            first = false;
            printf("  {\n");
            printf("    \"op\": \"mul_mat_iq4xs\",\n");
            printf("    \"shape\": \"%s\",\n", name);
            printf("    \"K\": %lld,\n", (long long)K);
            printf("    \"N\": %lld,\n", (long long)N);
            printf("    \"M\": %lld,\n", (long long)M);
            printf("    \"bytes\": %zu,\n", total_bytes);
            printf("    \"flops\": %.0f,\n", flops);
            printf("    \"stock_median_us\": %.3f,\n", stock.median_us);
            printf("    \"stock_p95_us\": %.3f,\n", stock.p95_us);
            printf("    \"stock_gb_s\": %.2f,\n", stock.gb_s);
            printf("    \"stock_tflops\": %.3f,\n", tflops_stock);
            printf("    \"gfx1100_median_us\": %.3f,\n", gfx.median_us);
            printf("    \"gfx1100_p95_us\": %.3f,\n", gfx.p95_us);
            printf("    \"gfx1100_gb_s\": %.2f,\n", gfx.gb_s);
            printf("    \"gfx1100_tflops\": %.3f,\n", tflops_gfx);
            printf("    \"speedup\": %.3f,\n", speedup);
            printf("    \"winner\": \"%s\"\n", speedup > 1.0 ? "gfx1100" : "stock");
            printf("  }");

            HIP_CHECK(hipFree(dX));
            HIP_CHECK(hipFree(dY));
        }
        HIP_CHECK(hipFree(dW));
    }
    printf("\n]\n");
    return 0;
}
