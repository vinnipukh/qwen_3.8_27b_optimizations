// test_gemm_wmma_compare.cpp — Validate streaming WMMA GEMM vs CPU FP64 oracle — Phase 07-03
// Covers canonical prefill shapes M=16,32,64,128,512,1024 plus tail cases. Cosine gate >=0.999.
// Also validates WMMA vs tiled parity when both applicable. Reports ATOMIC PASS/FAIL.

#include "ref_cpu.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>

hipError_t gemm_iq4xs_wmma_stream_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t stream);
hipError_t gemm_iq4xs_stream_tiled_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t stream);

struct GemmCase { const char* name; int64_t K; int64_t N; int64_t M; };

static bool test_one(const GemmCase& c, int64_t seed_base) {
    int64_t K = c.K, N = c.N, M = c.M;
    printf("[GEMM-WMMA-STREAM] %-32s K=%5lld N=%5lld M=%4lld ... ", c.name, (long long)K, (long long)N, (long long)M);

    std::mt19937 rng((uint32_t)(seed_base + M * 1009 + K * 917 + N));
    std::vector<block_iq4_xs> W;
    gen_iq4xs_weights(W, K, N, (uint32_t)seed_base);
    std::vector<float> X(K * M), Y_ref(N * M), Y_gpu(N * M), Y_tiled(N * M);
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

    // Streaming WMMA (or fallback tiled when gate fails)
    HIP_CHECK(gemm_iq4xs_wmma_stream_gpu(dW, dX, dY, K, N, M, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Y_gpu.data(), dY, N * M * sizeof(float), hipMemcpyDeviceToHost));

    // Explicit tiled for parity check
    HIP_CHECK(gemm_iq4xs_stream_tiled_gpu(dW, dX, dY, K, N, M, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Y_tiled.data(), dY, N * M * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dY));

    Metrics m_gpu = compute_metrics(Y_ref, Y_gpu);
    Metrics m_tiled = compute_metrics(Y_ref, Y_tiled);
    Metrics m_gpu_vs_tiled = compute_metrics(Y_tiled, Y_gpu);

    printf("gpu cos=%.6f max_abs=%.2e %s | tiled cos=%.6f %s | gpu/tiled cos=%.6f %s\n",
           m_gpu.cosine, m_gpu.max_abs, m_gpu.pass ? "PASS" : "FAIL",
           m_tiled.cosine, m_tiled.pass ? "PASS" : "FAIL",
           m_gpu_vs_tiled.cosine, m_gpu_vs_tiled.cosine >= 0.999 ? "OK" : "MISMATCH");
    if (m_gpu.bad) printf("  [WARN] gpu produced NaN/Inf\n");
    // Gating: cosine >=0.999 vs oracle; on WMMA path we allow float->half rounding noise (still >0.999)
    // Tail shapes may exercise fallback tiled which is more precise (cos ~0.99998).
    if (!m_gpu.pass) {
        printf("    max_rel=%.2e mean_abs=%.2e\n", m_gpu.max_rel, m_gpu.mean_abs);
    }
    // Require tiled also passes (sanity)
    bool ok = m_gpu.pass && m_tiled.pass;
    // When WMMA path was taken, gpu vs tiled should also be >=0.999 (same math within half rounding)
    if (M >= 512 && N % 16 == 0 && K % 16 == 0) {
        if (m_gpu_vs_tiled.cosine < 0.999) ok = false;
    }
    return ok;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== GEMM Streaming WMMA vs CPU Oracle (cosine >=0.999) ===\n");
    bool ok = true;
    GemmCase cases[] = {
        {"small_512_512_16", 512, 512, 16},
        {"prefill_512x512_128", 512, 512, 128},
        {"prefill_512x512_512", 512, 512, 512},
        {"prefill_512x512_1024", 512, 512, 1024},
        {"med_1024_1024_128", 1024, 1024, 128},
        {"med_1024_1024_512", 1024, 1024, 512},
        {"ffn_gate_trunc_5120_1024_128", 5120, 1024, 128},
        {"ffn_gate_trunc_5120_1024_512", 5120, 1024, 512},
        {"ffn_gate_trunc_5120_1024_1024", 5120, 1024, 1024},
        {"ffn_down_trunc_17408_512_128", 17408, 512, 128},
        {"attn_q_trunc_5120_1024_512", 5120, 1024, 512},
        {"attn_q_trunc_5120_1024_1024", 5120, 1024, 1024},
        {"wmma_gate_5120_1024_512", 5120, 1024, 512},
        {"tail_5120_5120_64", 5120, 512, 64},
        {"tail_512_512_32", 512, 512, 32},
    };
    for (auto &c : cases) {
        if (!test_one(c, 7000 + c.M)) ok = false;
    }
    printf("=== FINAL GEMM-WMMA-STREAM: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
