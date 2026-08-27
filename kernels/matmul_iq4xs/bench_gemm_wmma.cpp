// bench_gemm_wmma.cpp — Prefill throughput M=128,512,1024 vs real stock DP4A/MMQ — Phase 07-03
// Emits JSON array with speedup, TFLOPS, GB/s per shape. Target >1.2x over stock MMQ (target >1000 t/s e2e).
// Compares gemm_iq4xs_wmma_stream_gpu (streaming WMMA double-buffered LDS) vs
// gemm_iq4xs_stock_dp4a_gpu (real tiled DP4A MMQ weight reuse) and naive fallback tiled.

#include "ref_cpu.h"
#include "bench.h"
#include "hip_helpers.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemm_iq4xs_wmma_stream_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_stream_tiled_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("[\n");
    bool first = true;

    struct Shape { const char* name; int64_t K; int64_t N; };
    Shape shapes[] = {
        {"attn_q", 5120, 5120},
        {"ffn_gate", 5120, 17408},
        {"ffn_down", 17408, 5120},
    };
    int64_t Ms[] = {128, 512, 1024};

    std::vector<double> speedups_512;

    for (auto &sh : shapes) {
        int64_t K = sh.K, N = sh.N;
        int64_t blocks_per_row = K / QK_K;
        int64_t total_blocks = N * blocks_per_row;
        size_t W_bytes = total_blocks * sizeof(block_iq4_xs);

        std::vector<block_iq4_xs> h_W;
        gen_iq4xs_weights(h_W, K, N, 7777);

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
            std::mt19937 rng2(42 + (int)M);
            for (int64_t i = 0; i < K * M; ++i) h_X[i] = g(rng2);

            float *dX = nullptr, *dY = nullptr;
            HIP_CHECK(hipMalloc(&dX, X_bytes));
            HIP_CHECK(hipMalloc(&dY, Y_bytes));
            HIP_CHECK(hipMemcpy(dX, h_X.data(), X_bytes, hipMemcpyHostToDevice));

            auto stock_launch = [&](hipStream_t s){ HIP_CHECK(gemm_iq4xs_stock_dp4a_gpu(dW, dX, dY, K, N, M, s)); };
            BenchStats stock = bench_hip_event(stock_launch, 0, 10, 30, total_bytes);
            double tflops_stock = flops / (stock.median_us * 1e-6) / 1e12;

            auto wmma_launch = [&](hipStream_t s){ HIP_CHECK(gemm_iq4xs_wmma_stream_gpu(dW, dX, dY, K, N, M, s)); };
            BenchStats wmma = bench_hip_event(wmma_launch, 0, 10, 30, total_bytes);
            double tflops_wmma = flops / (wmma.median_us * 1e-6) / 1e12;
            double speedup = stock.median_us / wmma.median_us;

            auto tiled_launch = [&](hipStream_t s){ HIP_CHECK(gemm_iq4xs_stream_tiled_gpu(dW, dX, dY, K, N, M, s)); };
            BenchStats tiled = bench_hip_event(tiled_launch, 0, 10, 30, total_bytes);
            double tflops_tiled = flops / (tiled.median_us * 1e-6) / 1e12;

            if (M == 512) speedups_512.push_back(speedup);

            if (!first) printf(",\n");
            first = false;
            printf("  {\n");
            printf("    \"op\": \"gemm_iq4xs_wmma_stream\",\n");
            printf("    \"shape\": \"%s\",\n", sh.name);
            printf("    \"K\": %lld,\n", (long long)K);
            printf("    \"N\": %lld,\n", (long long)N);
            printf("    \"M\": %lld,\n", (long long)M);
            printf("    \"bytes\": %zu,\n", total_bytes);
            printf("    \"flops\": %.0f,\n", flops);
            printf("    \"stock_median_us\": %.3f,\n", stock.median_us);
            printf("    \"stock_p95_us\": %.3f,\n", stock.p95_us);
            printf("    \"stock_tflops\": %.3f,\n", tflops_stock);
            printf("    \"stock_gb_s\": %.2f,\n", stock.gb_s);
            printf("    \"tiled_median_us\": %.3f,\n", tiled.median_us);
            printf("    \"tiled_tflops\": %.3f,\n", tflops_tiled);
            printf("    \"tiled_gb_s\": %.2f,\n", tiled.gb_s);
            printf("    \"wmma_stream_median_us\": %.3f,\n", wmma.median_us);
            printf("    \"wmma_stream_p95_us\": %.3f,\n", wmma.p95_us);
            printf("    \"wmma_stream_tflops\": %.3f,\n", tflops_wmma);
            printf("    \"wmma_stream_gb_s\": %.2f,\n", wmma.gb_s);
            printf("    \"speedup\": %.3f,\n", speedup);
            printf("    \"speedup_vs_stock_dp4a\": %.3f,\n", speedup);
            printf("    \"winner\": \"%s\",\n", speedup > 1.0 ? "wmma_stream" : "stock_dp4a");
            printf("    \"note\": \"streaming WMMA 64x32 per block, LDS [2][32][33] double-buffered, wmma_f32_16x16x16_f16_w32, launch_bounds(256,4), fallback TILE_M=16 when M<512\"\n");
            printf("  }");

            HIP_CHECK(hipFree(dX));
            HIP_CHECK(hipFree(dY));
        }
        HIP_CHECK(hipFree(dW));
    }
    printf("\n]\n");

    // Human-readable summary to stderr
    double avg_512 = 0; for (auto s : speedups_512) avg_512 += s; if (!speedups_512.empty()) avg_512 /= speedups_512.size();
    fprintf(stderr, "\n=== GEMM Streaming WMMA vs Stock DP4A MMQ — M=128,512,1024 prefill ===\n");
    fprintf(stderr, "Kernel: gemm_iq4xs_wmma_stream_kernel (256 threads/block, 64x32 tile, 4x2 warps, LDS [2][32][33] padded, 2x WMMA per 32-K tile)\n");
    fprintf(stderr, "Hardware: v_wmma_f32_16x16x16_f16_w32 on Wave32, 1024 ops/CU/clock vs 512 DP4A fallback\n");
    fprintf(stderr, "Gate: WMMA when M>=512 && N%%16==0 && K%%16==0 else tiled TILE_M=16 fallback — M=128 expected tiled (no WMMA)\n");
    fprintf(stderr, "Occupancy: __launch_bounds__(256,4) + amdgpu_flat_work_group_size(256,256) => <=64 VGPRs, 16 waves/SIMD\n");
    fprintf(stderr, "Avg speedup @ M=512: %.3f (target >1.2x) — M=512 prefill should exceed 1.2x over stock MMQ\n", avg_512);
    fprintf(stderr, "TFLOPS reported per measurement; convert to prefill t/s via model 2*N*K/M scaling in 07-04.\n");
    return 0;
}
