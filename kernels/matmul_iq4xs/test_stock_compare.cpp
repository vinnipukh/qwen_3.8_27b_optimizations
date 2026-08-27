// test_stock_compare.cpp — Validate stock HIP comparator vs CPU oracle
// Covers both GEMV (M=1) and GEMM (M=16,128) for canonical and synthetic shapes.
// Pass criteria: cosine >=0.999, max_rel <=1e-3, no NaN/Inf.

#include "ref_cpu.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>

hipError_t gemv_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t stream);
hipError_t gemm_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t stream);

bool test_gemv_shape(const char* name, int64_t K, int64_t N, std::mt19937& rng_act) {
    printf("[GEMV] Testing shape %-12s K=%5lld N=%5lld ... ", name, (long long)K, (long long)N);
    std::vector<block_iq4_xs> W;
    gen_iq4xs_weights(W, K, N, 12345 + (uint32_t)K * 7919u + (uint32_t)N);
    std::vector<float> x(K), y_ref(N), y_gpu(N);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (int64_t i = 0; i < K; ++i) x[i] = gauss(rng_act);

    gemv_iq4xs_cpu_ref(W.data(), x.data(), y_ref.data(), K, N);

    block_iq4_xs* d_W = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_W, W.size() * sizeof(block_iq4_xs)));
    HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_W, W.data(), W.size() * sizeof(block_iq4_xs), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, x.data(), K * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(gemv_iq4xs_stock_gpu(d_W, d_x, d_y, K, N, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(y_gpu.data(), d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    Metrics m = compute_metrics(y_ref, y_gpu);
    printf("cos=%.6f max_rel=%.2e %s\n", m.cosine, m.max_rel, m.pass ? "PASS" : "FAIL");
    return m.pass;
}

bool test_gemm_shape(const char* name, int64_t K, int64_t N, int64_t M, std::mt19937& rng_act) {
    printf("[GEMM] Testing shape %-12s K=%5lld N=%5lld M=%4lld ... ", name, (long long)K, (long long)N, (long long)M);
    std::vector<block_iq4_xs> W;
    gen_iq4xs_weights(W, K, N, 54321 + (uint32_t)K * 31u + (uint32_t)N);
    std::vector<float> X(K * M), Y_ref(N * M), Y_gpu(N * M);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (int64_t i = 0; i < K * M; ++i) X[i] = gauss(rng_act);

    gemm_iq4xs_cpu_ref(W.data(), X.data(), Y_ref.data(), K, N, M);

    block_iq4_xs* d_W = nullptr;
    float* d_X = nullptr;
    float* d_Y = nullptr;
    HIP_CHECK(hipMalloc(&d_W, W.size() * sizeof(block_iq4_xs)));
    HIP_CHECK(hipMalloc(&d_X, K * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Y, N * M * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_W, W.data(), W.size() * sizeof(block_iq4_xs), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, X.data(), K * M * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(gemm_iq4xs_stock_gpu(d_W, d_X, d_Y, K, N, M, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Y_gpu.data(), d_Y, N * M * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_Y));

    Metrics m = compute_metrics(Y_ref, Y_gpu);
    printf("cos=%.6f max_rel=%.2e %s\n", m.cosine, m.max_rel, m.pass ? "PASS" : "FAIL");
    return m.pass;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== Stock HIP Comparator vs CPU Reference Oracle ===\n");
    std::mt19937 rng(42);
    bool all_pass = true;

    printf("\n--- GEMV (M=1) Canonical and Corner Cases ---\n");
    for (int i = 0; i < NUM_CANONICAL_SHAPES; ++i) {
        if (!test_gemv_shape(CANONICAL_SHAPES[i].name, CANONICAL_SHAPES[i].K, CANONICAL_SHAPES[i].N, rng)) all_pass = false;
    }
    if (!test_gemv_shape("synthetic_small", 512, 512, rng)) all_pass = false;
    if (!test_gemv_shape("synthetic_rect", 5120, 256, rng)) all_pass = false;

    printf("\n--- GEMM (M>1) Canonical Shapes ---\n");
    if (!test_gemm_shape("attn_q_m16", 5120, 5120, 16, rng)) all_pass = false;
    if (!test_gemm_shape("attn_q_m128", 5120, 5120, 128, rng)) all_pass = false;
    if (!test_gemm_shape("ffn_gate_m16", 5120, 17408, 16, rng)) all_pass = false;
    if (!test_gemm_shape("ffn_gate_m128", 5120, 17408, 128, rng)) all_pass = false;
    if (!test_gemm_shape("ffn_down_m16", 17408, 5120, 16, rng)) all_pass = false;
    if (!test_gemm_shape("ffn_down_m128", 17408, 5120, 128, rng)) all_pass = false;

    printf("\n=== Overall Validation: %s ===\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
