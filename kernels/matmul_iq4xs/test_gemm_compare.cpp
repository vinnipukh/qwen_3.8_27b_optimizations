// test_gemm_compare.cpp — Validate custom WMMA GEMM vs CPU oracle
#include "ref_cpu.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemm_iq4xs_wmma_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t stream);
hipError_t gemm_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t stream);

bool test_one(const char* name, int64_t K, int64_t N, int64_t M, std::mt19937& rng) {
    printf("[GEMM-CUSTOM] %-28s K=%5lld N=%5lld M=%4lld ... ", name, (long long)K, (long long)N, (long long)M);
    std::vector<block_iq4_xs> W;
    gen_iq4xs_weights(W, K, N, 2000 + (uint32_t)M * 11 + (uint32_t)K);
    std::vector<float> X(K * M), Y_ref(N * M), Y_gfx(N * M), Y_stock(N * M);
    std::normal_distribution<float> g(0, 1);
    for (int64_t i = 0; i < K * M; ++i) X[i] = g(rng);
    gemm_iq4xs_cpu_ref(W.data(), X.data(), Y_ref.data(), K, N, M);

    block_iq4_xs* dW = nullptr;
    float *dX = nullptr, *dY = nullptr;
    HIP_CHECK(hipMalloc(&dW, W.size() * sizeof(block_iq4_xs)));
    HIP_CHECK(hipMalloc(&dX, K * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&dY, N * M * sizeof(float)));
    HIP_CHECK(hipMemcpy(dW, W.data(), W.size() * sizeof(block_iq4_xs), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX, X.data(), K * M * sizeof(float), hipMemcpyHostToDevice));

    // Stock
    HIP_CHECK(gemm_iq4xs_stock_gpu(dW, dX, dY, K, N, M, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Y_stock.data(), dY, N * M * sizeof(float), hipMemcpyDeviceToHost));

    // Custom wmma/tiled
    HIP_CHECK(gemm_iq4xs_wmma_gpu(dW, dX, dY, K, N, M, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Y_gfx.data(), dY, N * M * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dY));

    Metrics ms = compute_metrics(Y_ref, Y_stock);
    Metrics mg = compute_metrics(Y_ref, Y_gfx);
    printf("stock cos=%.6f %s | gfx cos=%.6f max_rel=%.2e %s\n",
           ms.cosine, ms.pass ? "PASS" : "FAIL",
           mg.cosine, mg.max_rel, mg.pass ? "PASS" : "FAIL");
    if (mg.bad) printf("  gfx NaN/Inf!\n");
    return mg.pass;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== GEMM Custom WMMA vs CPU Oracle ===\n");
    std::mt19937 rng(42);
    bool ok = true;
    struct Case { const char* n; int64_t K, N, M; };
    Case cases[] = {
        {"small_512_512_16", 512, 512, 16},
        {"small_512_512_128", 512, 512, 128},
        {"med_1024_1024_16", 1024, 1024, 16},
        {"med_1024_1024_64", 1024, 1024, 64},
        {"ffn_gate_trunc_5120_1024_16", 5120, 1024, 16},
        {"ffn_gate_trunc_5120_1024_64", 5120, 1024, 64},
        {"ffn_down_trunc_17408_512_16", 17408, 512, 16},
        {"attn_q_trunc_5120_1024_128", 5120, 1024, 128},
        {"wmma_5120_5120_64", 5120, 512, 64},
        {"wmma_5120_1024_32", 5120, 1024, 32},
        {"wmma_gate_pass_5120_1024_512", 5120, 1024, 512},
    };
    for (auto &c : cases) {
        if (!test_one(c.n, c.K, c.N, c.M, rng)) ok = false;
    }
    printf("=== FINAL GEMM-CUSTOM: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
