// test_gemv_compare.cpp — Validate custom gfx1100 GEMV vs CPU oracle
#include "ref_cpu.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemv_iq4xs_gfx1100_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t stream);
hipError_t gemv_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t stream);

bool test_one(const char* name, int64_t K, int64_t N, std::mt19937& rng) {
    printf("[GEMV-CUSTOM] %-16s K=%5lld N=%5lld ... ", name, (long long)K, (long long)N);
    std::vector<block_iq4_xs> W;
    gen_iq4xs_weights(W, K, N, 1000 + (uint32_t)K * 17 + (uint32_t)N);
    std::vector<float> x(K), y_ref(N), y_gfx(N), y_stock(N);
    std::normal_distribution<float> g(0, 1);
    for (int64_t i = 0; i < K; ++i) x[i] = g(rng);
    gemv_iq4xs_cpu_ref(W.data(), x.data(), y_ref.data(), K, N);

    block_iq4_xs* dW = nullptr;
    float *dx = nullptr, *dy = nullptr;
    HIP_CHECK(hipMalloc(&dW, W.size() * sizeof(block_iq4_xs)));
    HIP_CHECK(hipMalloc(&dx, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dW, W.data(), W.size() * sizeof(block_iq4_xs), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), K * sizeof(float), hipMemcpyHostToDevice));

    // Stock
    HIP_CHECK(gemv_iq4xs_stock_gpu(dW, dx, dy, K, N, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(y_stock.data(), dy, N * sizeof(float), hipMemcpyDeviceToHost));

    // Custom
    HIP_CHECK(gemv_iq4xs_gfx1100_gpu(dW, dx, dy, K, N, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(y_gfx.data(), dy, N * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));

    Metrics m_stock = compute_metrics(y_ref, y_stock);
    Metrics m_gfx = compute_metrics(y_ref, y_gfx);
    printf("stock cosine=%.6f %s | gfx cosine=%.6f max_rel=%.2e %s\n",
           m_stock.cosine, m_stock.pass ? "PASS" : "FAIL",
           m_gfx.cosine, m_gfx.max_rel, m_gfx.pass ? "PASS" : "FAIL");
    if (m_gfx.bad) printf("  gfx NaN/Inf!\n");
    return m_gfx.pass;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== GEMV Custom gfx1100 vs CPU Oracle ===\n");
    std::mt19937 rng(42);
    bool ok = true;
    for (int i = 0; i < NUM_CANONICAL_SHAPES; ++i) {
        if (!test_one(CANONICAL_SHAPES[i].name, CANONICAL_SHAPES[i].K, CANONICAL_SHAPES[i].N, rng)) ok = false;
    }
    if (!test_one("small_512", 512, 512, rng)) ok = false;
    if (!test_one("small_1024", 1024, 2048, rng)) ok = false;
    printf("=== FINAL GEMV-CUSTOM: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
