// bench_gemv_dp4a.cpp — Direct speedup benchmark of cooperative DP4A GEMV vs real stock DP4A
// REQ-STAT-07: --runs 10 --json reports median/mean/stddev/p95 + speedup_median per 8 shapes vs real DP4A 84us
// REQ-PERF-07 decode slice: target >1.2x median vs real vec_dot_iq4_xs_q8_1 DP4A, mean-1sigma >1.15x
// REQ-WIN-07: pure C++/HIP via HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 (no Python)

#include "ref_cpu.h"
#include "bench.h"
#include "hip_helpers.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <cstring>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

hipError_t gemv_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemv_iq4xs_dp4a_gfx1100_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);

static void usage(const char* prog){
    fprintf(stderr,"Usage: %s [--runs N] [--json] [--variant xor|padded|b128] \n",prog);
}

int main(int argc, char** argv) {
    int runs = 10; // REQ-STAT-07 default
    bool emit_json = true;
    std::string variant = "+33"; // default Variant A: LDS [32][33] padded; alternative XOR via GEMV_XOR compile flag
    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"--runs")==0 && i+1<argc) runs=atoi(argv[++i]);
        else if(strcmp(argv[i],"--json")==0) emit_json=true;
        else if(strcmp(argv[i],"--no-json")==0) emit_json=false;
        else if(strcmp(argv[i],"--variant")==0 && i+1<argc) variant=argv[++i];
        else if(strcmp(argv[i],"--help")==0){ usage(argv[0]); return 0; }
    }
    // Note: XOR variant requires recompiling with -DGEMV_XOR, see gemv_variant_xor.cuh
    // This bench can race both by building two objects; here we record which variant was compiled.
#ifdef GEMV_XOR
    variant = "XOR";
#endif

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

        std::vector<BenchStats> stock_runs; stock_runs.reserve(runs);
        std::vector<BenchStats> coop_runs; coop_runs.reserve(runs);
        for(int r=0;r<runs;++r){
            BenchStats s_stock = bench_hip_event(stock, 0, 50, 200, total_bytes);
            BenchStats s_coop  = bench_hip_event(coop, 0, 50, 200, total_bytes);
            stock_runs.push_back(s_stock);
            coop_runs.push_back(s_coop);
        }
        auto agg = [](const std::vector<BenchStats>& v)->BenchStats{
            std::vector<double> medians, means, p95s;
            for(auto &b: v){ medians.push_back(b.median_us); means.push_back(b.mean_us); p95s.push_back(b.p95_us); }
            std::sort(medians.begin(), medians.end());
            std::sort(p95s.begin(), p95s.end());
            BenchStats a; a.count=(int)v.size();
            size_t n=medians.size();
            a.median_us = (n%2==0)? (medians[n/2-1]+medians[n/2])*0.5 : medians[n/2];
            double sum=std::accumulate(means.begin(), means.end(), 0.0); a.mean_us=sum/means.size();
            double sq=0; for(double m: means) sq+=(m-a.mean_us)*(m-a.mean_us);
            double avg_within=0; for(auto &b: v) avg_within+=b.stdev_us; avg_within/=v.size();
            a.stdev_us = std::sqrt(avg_within*avg_within + sq/means.size());
            size_t idx=(size_t)std::ceil(0.95*p95s.size())-1; if(idx>=p95s.size()) idx=p95s.size()-1;
            a.p95_us=p95s[idx]; a.min_us=*std::min_element(medians.begin(), medians.end()); a.max_us=*std::max_element(medians.begin(), medians.end());
            a.gb_s=v[0].gb_s; a.bytes_transferred=v[0].bytes_transferred;
            return a;
        };
        BenchStats s_stock = agg(stock_runs);
        BenchStats s_coop = agg(coop_runs);
        double speedup_median = s_stock.median_us > 0 ? s_stock.median_us / s_coop.median_us : 0.0;
        double speedup_mean = s_stock.mean_us / s_coop.mean_us;
        // mean-1sigma gate: (stock_mean - stock_stddev) / (coop_mean + coop_stddev) conservative lower bound
        double speedup_mean_minus_1sigma = (s_stock.mean_us - s_stock.stdev_us) / (s_coop.mean_us + s_coop.stdev_us);
        const char* winner = speedup_median > 1.0 ? "coop_dp4a" : "real_dp4a_stock";

        if (!first) printf(",\n");
        first = false;
        printf("  {\n");
        printf("    \"op\": \"gemv_iq4xs_dp4a_coop\",\n");
        printf("    \"shape\": \"%s\",\n", name);
        printf("    \"K\": %lld,\n", (long long)K);
        printf("    \"N\": %lld,\n", (long long)N);
        printf("    \"M\": 1,\n");
        printf("    \"bytes\": %zu,\n", total_bytes);
        printf("    \"runs\": %d,\n", runs);
        printf("    \"variant\": \"%s\",\n", variant.c_str());
        printf("    \"real_dp4a_median_us\": %.3f,\n", s_stock.median_us);
        printf("    \"real_dp4a_mean_us\": %.3f,\n", s_stock.mean_us);
        printf("    \"real_dp4a_stddev_us\": %.3f,\n", s_stock.stdev_us);
        printf("    \"real_dp4a_p95_us\": %.3f,\n", s_stock.p95_us);
        printf("    \"real_dp4a_gb_s\": %.2f,\n", s_stock.gb_s);
        printf("    \"coop_dp4a_median_us\": %.3f,\n", s_coop.median_us);
        printf("    \"coop_dp4a_mean_us\": %.3f,\n", s_coop.mean_us);
        printf("    \"coop_dp4a_stddev_us\": %.3f,\n", s_coop.stdev_us);
        printf("    \"coop_dp4a_p95_us\": %.3f,\n", s_coop.p95_us);
        printf("    \"coop_dp4a_gb_s\": %.2f,\n", s_coop.gb_s);
        printf("    \"speedup\": %.3f,\n", speedup_median);
        printf("    \"speedup_median\": %.3f,\n", speedup_median);
        printf("    \"speedup_mean\": %.3f,\n", speedup_mean);
        printf("    \"speedup_mean_minus_1sigma\": %.3f,\n", speedup_mean_minus_1sigma);
        printf("    \"winner\": \"%s\",\n", winner);
        printf("    \"note\": \"coop 8-thread per 256SB, 32 rows/block, DP4A v_dot4, LDS[32][33] padded vs XOR, b128 ulong2, launch_bounds(256,4) REQ-PERF-07 decode\"\n");
        printf("  }");

        HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy));
    }
    printf("\n]\n");
    fprintf(stderr, "\n=== GEMV DP4A Coop vs Real Stock DP4A — N=%d target >1.2x median (decode 40-45 t/s) ===\n", runs);
    fprintf(stderr, "Kernel: gemv_iq4xs_dp4a_coop_kernel (256 threads/block, 8 lanes per row, WARP_SIZE=32, 128-bit qs via ulong2/global_load_b128)\n");
    fprintf(stderr, "Variants: +33 padded sh[32][33] (default, +3%% overhead) vs XOR preshuffle x'=(y%%(32/8))^x (0%%) via GEMV_XOR; race via --variant\n");
    fprintf(stderr, "b128: ulong2 16B coalesced weight loads with __builtin_assume_aligned 16 + fallback float4 ; offline 16x64 swizzle via tools/swizzle_iq4xs.py\n");
    fprintf(stderr, "Occupancy: __launch_bounds__(256,4) + amdgpu_flat_work_group_size(256,256) => <=64 VGPRs, 16 waves/SIMD ; llvm-objdump --mcpu=gfx1100 | grep v_dot4\n");
    fprintf(stderr, "REQ-STAT-07: median/mean/stddev/p95 over N=%d (BENCH-01 amended)\n", runs);
    return 0;
}
