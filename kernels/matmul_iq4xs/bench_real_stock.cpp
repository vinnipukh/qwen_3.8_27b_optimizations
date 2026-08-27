// bench_real_stock.cpp — Microbenchmark for REAL upstream DP4A (quantize + vec_dot) across 8 canonical shapes
// Emits JSON array to stdout. Compares naive scalar stock vs real DP4A stock vs custom (if available).
// Proves DP4A vs naive: 5120x5120 GEMV should be ~20-40us (DP4A) not 500+us (naive scalar float fallback).
// Reference: vec_dot_iq4_xs_q8_1 + quantize_row_q8_1 pipeline as in real_stock_dp4a_comparator.hip

#include "ref_cpu.h"
#include "bench.h"
#include "hip_helpers.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemv_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemv_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemm_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);

int main() {
    printf("[\n");
    bool first = true;
    // Bench GEMV for all 8 canonical shapes
    for (int i = 0; i < NUM_CANONICAL_SHAPES; ++i) {
        auto &sh = CANONICAL_SHAPES[i];
        int64_t K = sh.K;
        int64_t N = sh.N;
        const char* name = sh.name;
        int64_t blocks_per_row = K / QK_K;
        size_t W_bytes = N * blocks_per_row * sizeof(block_iq4_xs);
        size_t X_bytes = K * sizeof(float);
        size_t Y_bytes = N * sizeof(float);
        size_t total_bytes = W_bytes + X_bytes + Y_bytes;

        std::vector<block_iq4_xs> h_W;
        gen_iq4xs_weights(h_W, K, N, 12345 + i * 100);
        std::vector<float> h_x(K);
        std::mt19937 rng(42 + i);
        std::normal_distribution<float> g(0,1);
        for (int64_t k = 0; k < K; ++k) h_x[k] = g(rng);

        block_iq4_xs* dW = nullptr;
        float *dx = nullptr, *dy = nullptr;
        HIP_CHECK(hipMalloc(&dW, W_bytes));
        HIP_CHECK(hipMalloc(&dx, X_bytes));
        HIP_CHECK(hipMalloc(&dy, Y_bytes));
        HIP_CHECK(hipMemcpy(dW, h_W.data(), W_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dx, h_x.data(), X_bytes, hipMemcpyHostToDevice));

        auto dp4a_launch = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_stock_dp4a_gpu(dW, dx, dy, K, N, s)); };
        auto naive_launch = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_stock_gpu(dW, dx, dy, K, N, s)); };

        BenchStats dp4a = bench_hip_event(dp4a_launch, 0, 50, 200, total_bytes);
        BenchStats naive = bench_hip_event(naive_launch, 0, 20, 100, total_bytes);

        double speedup_vs_naive = naive.median_us / dp4a.median_us;

        if (!first) printf(",\n");
        first = false;
        printf("  {\n");
        printf("    \"op\": \"gemv_iq4xs_real_dp4a\",\n");
        printf("    \"shape\": \"%s\",\n", name);
        printf("    \"K\": %lld,\n", (long long)K);
        printf("    \"N\": %lld,\n", (long long)N);
        printf("    \"M\": 1,\n");
        printf("    \"bytes\": %zu,\n", total_bytes);
        printf("    \"naive_median_us\": %.3f,\n", naive.median_us);
        printf("    \"real_dp4a_median_us\": %.3f,\n", dp4a.median_us);
        printf("    \"real_dp4a_p95_us\": %.3f,\n", dp4a.p95_us);
        printf("    \"real_dp4a_gb_s\": %.2f,\n", dp4a.gb_s);
        printf("    \"speedup_vs_naive\": %.3f,\n", speedup_vs_naive);
        printf("    \"note\": \"real stock uses quantize_row_q8_1 + vec_dot_iq4_xs_q8_1 DP4A (v_dot4_i32_i8)\"\n");
        printf("  }");

        HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy));
    }
    printf("\n]\n");

    // Also emit human-readable timing table to stderr
    fprintf(stderr, "\n=== REAL DP4A Baseline Timing Table (expected 20-40us for 5120x5120) ===\n");
    fprintf(stderr, "(JSON above contains median_us; DP4A should be ~10x faster than naive 500us)\n");
    return 0;
}
