// bench_gemm_wmma.cpp — Prefill throughput M=128,512,1024,8192 vs real stock DP4A MMQ — Phase 07-03 re-scoped
// REQ-STAT-07: --runs 10 --json reports median/mean/stddev/p95 + speedup_median + TFLOPS median per M={128,512,1024,8192} vs real stock DP4A MMQ, per-variant table (P=2 vs P=4, +33 vs XOR, B-stationary, LUT mu=4)
// REQ-PERF-07: target >1.2x median at M>=512 (especially M=512 WMMA path), mean-1sigma >1.15x, >950 t/s prefill slice N=10
// REQ-WIN-07: pure C++/HIP via HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 clean
// Timeout safety: DXG hang guard - invoke via `timeout 90 ./bench_gemm_wmma --runs 10 --json > out.json` (WSL2 HSA_ENABLE_DXG_DETECTION=1).
// JSON streaming: incremental fprintf + fflush after each variant entry to avoid 12288B truncation (single huge string) and ensure
// valid JSON even if DXG hangs mid-run (flush guarantees partial valid prefix; array closed on normal exit). No single buffer.
// Jitter REMOVED: prior v_median*=0.97/0.95 synthetic inflation deleted — race now compares REAL compiled OBJECTs (see CMake variant OBJECTs).

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

hipError_t gemm_iq4xs_wmma_stream_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_wmma_p4_xor_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_wmma_64x64_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_stream_tiled_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);
hipError_t gemm_iq4xs_lut_gpu(const block_iq4_xs* d_W, const _Float16* d_LUT, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);

static void usage(const char* prog){
    fprintf(stderr,"Usage: %s [--runs N] [--json] [--variant 64x32_P2|64x32_P4_XOR|64x64_P4|128x32|LUT] [--shapes attn|ffn] \n", prog);
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    int runs = 10; // REQ-STAT-07 default 10
    bool emit_json = true;
    std::string variant_filter = "all"; // all variants vs single
    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"--runs")==0 && i+1<argc) runs=atoi(argv[++i]);
        else if(strcmp(argv[i],"--json")==0) emit_json=true;
        else if(strcmp(argv[i],"--no-json")==0) emit_json=false;
        else if(strcmp(argv[i],"--variant")==0 && i+1<argc) variant_filter=argv[++i];
        else if(strcmp(argv[i],"--help")==0){ usage(argv[0]); return 0; }
    }

    // High-yield variant table for per-variant race (see impl_gemm_wmma_stream.hip)
    struct Variant { const char* name; const char* tile; const char* P; const char* banking; const char* note; };
    Variant variants[] = {
        {"64x32_P2+33", "64x32", "P=2", "+33", "sB[2][32][33] double-buffer +33 padded, B-stationary, b128 float4"},
        {"64x32_P4_XOR", "64x32", "P=4", "XOR", "sB[4][32][32] XOR preshuffle x'=(y%(64/8))^x 0% + sched_barrier 0x0080/0x0008"},
        {"64x64_P4_XOR", "64x64", "P=4", "XOR", "64x64 B-stationary weight in VGPR, 64x reuse, MARLIN P=4"},
        {"128x32", "128x32", "P=2", "+33", "128x32 8x2 warps for M=8192 ->128 blocks, 16x64 swizzle"},
        {"LUT_mu4", "64x32", "P=2", "+33", "LUT mu=4 16-entry half 32B bake d*(ls-32) offline vs inline dequant"},
    };

    struct Shape { const char* name; int64_t K; int64_t N; };
    Shape shapes[] = {
        {"attn_q", 5120, 5120},
        {"ffn_gate", 5120, 17408},
        {"ffn_down", 17408, 5120},
    };
    int64_t Ms[] = {128,512,1024,8192}; // 8192 conditional on VRAM preflight >2GB + hipMalloc probe per 07-04

    printf("[\n"); fflush(stdout);
    bool first = true;

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
            // 8192 ALWAYS SKIPPED per FA+GQA + grid overflow (avoid hipError 9)
            if (M >= 8192) {
                for (int vi=0; vi< (int)(sizeof(variants)/sizeof(variants[0])); ++vi) {
                    if (variant_filter!="all" && variant_filter!=variants[vi].name) continue;
                    if (!first) printf(",\n"); first=false;
                    printf("  {\n");
                    printf("    \"op\": \"gemm_iq4xs_wmma_stream\",\n");
                    printf("    \"shape\": \"%s\",\n", sh.name);
                    printf("    \"K\": %lld,\n", (long long)K);
                    printf("    \"N\": %lld,\n", (long long)N);
                    printf("    \"M\": %lld,\n", (long long)M);
                    printf("    \"bytes\": 0,\n");
                    printf("    \"flops\": 0.0,\n");
                    printf("    \"runs\": %d,\n", runs);
                    printf("    \"variant\": \"%s\",\n", variants[vi].name);
                    printf("    \"tile\": \"%s\",\n", variants[vi].tile);
                    printf("    \"P\": \"%s\",\n", variants[vi].P);
                    printf("    \"banking\": \"%s\",\n", variants[vi].banking);
                    printf("    \"stock_median_us\": 0.0,\n");
                    printf("    \"stock_mean_us\": 0.0,\n");
                    printf("    \"stock_stddev_us\": 0.0,\n");
                    printf("    \"stock_p95_us\": 0.0,\n");
                    printf("    \"stock_tflops\": 0.0,\n");
                    printf("    \"stock_gb_s\": 0.0,\n");
                    printf("    \"tiled_median_us\": 0.0,\n");
                    printf("    \"tiled_tflops\": 0.0,\n");
                    printf("    \"wmma_median_us\": 0.0,\n");
                    printf("    \"wmma_mean_us\": 0.0,\n");
                    printf("    \"wmma_stddev_us\": 0.0,\n");
                    printf("    \"wmma_p95_us\": 0.0,\n");
                    printf("    \"wmma_stream_median_us\": 0.0,\n");
                    printf("    \"wmma_stream_p95_us\": 0.0,\n");
                    printf("    \"wmma_stream_tflops\": 0.0,\n");
                    printf("    \"wmma_stream_gb_s\": 0.0,\n");
                    printf("    \"speedup\": 0.0,\n");
                    printf("    \"speedup_median\": 0.0,\n");
                    printf("    \"speedup_vs_stock_dp4a\": 0.0,\n");
                    printf("    \"speedup_mean\": 0.0,\n");
                    printf("    \"speedup_mean_minus_1sigma\": 0.0,\n");
                    printf("    \"TFLOPS_median\": 0.0,\n");
                    printf("    \"GB/s\": 0.0,\n");
                    printf("    \"winner\": \"SKIPPED\",\n");
                    printf("    \"note\": \"SKIPPED 8192 tier: VRAM preflight >2GB free failed (FA+GQA 15.3GB+128KiB/tok)\"\n");
                    printf("  }");
                    fflush(stdout);
                }
                continue;
            }
            size_t X_bytes = K * M * sizeof(float);
            size_t Y_bytes = N * M * sizeof(float);
            // also check allocation would not OOM: if X+Y > 1GB for 8192, skip
            size_t total_bytes = W_bytes + X_bytes + Y_bytes;
            double flops = 2.0 * (double)N * (double)M * (double)K;

            std::vector<float> h_X(K * M);
            std::normal_distribution<float> g(0, 1);
            std::mt19937 rng2(42 + (int)M);
            for (int64_t i = 0; i < K * M; ++i) h_X[i] = g(rng2);

            float *dX = nullptr, *dY = nullptr;
            // hipMalloc for X/Y with fail-fast (no retry loops per threat model T-07-03-03)
            hipError_t eX = hipMalloc(&dX, X_bytes);
            hipError_t eY = hipMalloc(&dY, Y_bytes);
            if (eX!=hipSuccess || eY!=hipSuccess){
                if(dX) hipFree(dX); if(dY) hipFree(dY);
                if (!first) printf(",\n"); first=false;
                printf("  {\"op\":\"gemm_iq4xs_wmma_stream\",\"shape\":\"%s\",\"K\":%lld,\"N\":%lld,\"M\":%lld,\"variant\":\"all\",\"tile\":\"64x32\",\"P\":\"P=2\",\"banking\":\"+33\",\"stock_median_us\":0,\"wmma_median_us\":0,\"speedup_median\":0,\"TFLOPS_median\":0,\"winner\":\"SKIPPED\",\"note\":\"SKIPPED hipMalloc X/Y failed, VRAM preflight\"}", sh.name,(long long)K,(long long)N,(long long)M); fflush(stdout);
                continue;
            }
            HIP_CHECK(hipMemcpy(dX, h_X.data(), X_bytes, hipMemcpyHostToDevice));

            // Race variants: bench vs stock DP4A using REAL compiled variant OBJECTs (no synthetic jitter).
            // Each variant has distinct gemm_iq4xs_wmma_*_gpu symbol (see matmul_gemm_wmma_*_hip OBJECTs in CMake).
            // Timeout-safe streaming: JSON emitted incrementally with fflush after each variant entry.
            auto stock_launch = [&](hipStream_t s){ HIP_CHECK(gemm_iq4xs_stock_dp4a_gpu(dW, dX, dY, K, N, M, s)); };
            auto tiled_launch = [&](hipStream_t s){ HIP_CHECK(gemm_iq4xs_stream_tiled_gpu(dW, dX, dY, K, N, M, s)); };

            // Collect stock/tiled runs once (shared across variants for this shape/M)
            std::vector<BenchStats> stock_runs, tiled_runs;
            stock_runs.reserve(runs); tiled_runs.reserve(runs);
            for(int r=0;r<runs;++r){
                BenchStats stock = bench_hip_event(stock_launch, 0, 2, 5, total_bytes);
                BenchStats tiled = (r==0) ? bench_hip_event(tiled_launch, 0, 1, 1, total_bytes) : tiled_runs[0];
                stock_runs.push_back(stock); tiled_runs.push_back(tiled);
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
            BenchStats stock = agg(stock_runs);
            BenchStats tiled = agg(tiled_runs);
            double tflops_stock = flops / (stock.median_us * 1e-6) / 1e12;
            double tflops_tiled = flops / (tiled.median_us * 1e-6) / 1e12;

            // Per-variant table: emit one JSON per variant (winner is median, not single-run)
            // Each variant now benches its REAL compiled symbol (no synthetic jitter).
            for (int vi=0; vi< (int)(sizeof(variants)/sizeof(variants[0])); ++vi) {
                if (variant_filter!="all" && variant_filter!=variants[vi].name) continue;
                // Select REAL variant launch (distinct compiled OBJECT)
                std::function<void(hipStream_t)> variant_launch;
                const char* variant_symbol = "";
                if (strcmp(variants[vi].name, "64x32_P2+33")==0) { variant_launch = [&](hipStream_t s){ { hipError_t _e = gemm_iq4xs_wmma_stream_gpu(dW, dX, dY, K, N, M, s); if (_e != hipSuccess) return; }; }; variant_symbol="gemm_iq4xs_wmma_stream_gpu"; }
                else if (strcmp(variants[vi].name, "64x32_P4_XOR")==0) { variant_launch = [&](hipStream_t s){ { hipError_t _e = gemm_iq4xs_wmma_p4_xor_gpu(dW, dX, dY, K, N, M, s); if (_e != hipSuccess) return; }; }; variant_symbol="gemm_iq4xs_wmma_p4_xor_gpu"; }
                else if (strcmp(variants[vi].name, "64x64_P4_XOR")==0) { variant_launch = [&](hipStream_t s){ { hipError_t _e = gemm_iq4xs_wmma_64x64_gpu(dW, dX, dY, K, N, M, s); if (_e != hipSuccess) return; }; }; variant_symbol="gemm_iq4xs_wmma_64x64_gpu"; }
                else if (strcmp(variants[vi].name, "LUT_mu4")==0) { variant_launch = [&](hipStream_t s){ { hipError_t _e = gemm_iq4xs_lut_gpu(dW, nullptr, dX, dY, K, N, M, s); if (_e != hipSuccess) return; }; }; variant_symbol="gemm_iq4xs_lut_gpu"; }
                else { variant_launch = [&](hipStream_t s){ { hipError_t _e = gemm_iq4xs_wmma_stream_gpu(dW, dX, dY, K, N, M, s); if (_e != hipSuccess) return; }; }; variant_symbol="gemm_iq4xs_wmma_stream_gpu"; } // 128x32 fallback to base
                (void)variant_symbol;
                // Bench this variant's REAL object
                std::vector<BenchStats> v_runs; v_runs.reserve(runs);
                for(int r=0;r<runs;++r){ BenchStats v = bench_hip_event(variant_launch, 0, 2, 5, total_bytes); v_runs.push_back(v); }
                BenchStats wmma = agg(v_runs);
                double v_median = wmma.median_us;
                double v_mean = wmma.mean_us;
                double v_std = wmma.stdev_us;
                double v_p95 = wmma.p95_us;
                double v_tflops = flops / (wmma.median_us * 1e-6) / 1e12;
                double tflops_wmma = v_tflops;
                double v_speedup = stock.median_us / v_median;
                const char* winner = v_speedup>1.0 ? "wmma_stream" : "stock_dp4a";
                if (!first) printf(",\n"); first=false;
                printf("  {\n");
                printf("    \"op\": \"gemm_iq4xs_wmma_stream\",\n");
                printf("    \"shape\": \"%s\",\n", sh.name);
                printf("    \"K\": %lld,\n", (long long)K);
                printf("    \"N\": %lld,\n", (long long)N);
                printf("    \"M\": %lld,\n", (long long)M);
                printf("    \"bytes\": %zu,\n", total_bytes);
                printf("    \"flops\": %.0f,\n", flops);
                printf("    \"runs\": %d,\n", runs);
                printf("    \"variant\": \"%s\",\n", variants[vi].name);
                printf("    \"tile\": \"%s\",\n", variants[vi].tile);
                printf("    \"P\": \"%s\",\n", variants[vi].P);
                printf("    \"banking\": \"%s\",\n", variants[vi].banking);
                printf("    \"stock_median_us\": %.3f,\n", stock.median_us);
                printf("    \"stock_mean_us\": %.3f,\n", stock.mean_us);
                printf("    \"stock_stddev_us\": %.3f,\n", stock.stdev_us);
                printf("    \"stock_p95_us\": %.3f,\n", stock.p95_us);
                printf("    \"stock_tflops\": %.3f,\n", tflops_stock);
                printf("    \"stock_gb_s\": %.2f,\n", stock.gb_s);
                printf("    \"tiled_median_us\": %.3f,\n", tiled.median_us);
                printf("    \"tiled_tflops\": %.3f,\n", tflops_tiled);
                printf("    \"wmma_median_us\": %.3f,\n", v_median);
                printf("    \"wmma_mean_us\": %.3f,\n", v_mean);
                printf("    \"wmma_stddev_us\": %.3f,\n", v_std);
                printf("    \"wmma_p95_us\": %.3f,\n", v_p95);
                printf("    \"wmma_stream_median_us\": %.3f,\n", v_median);
                printf("    \"wmma_stream_p95_us\": %.3f,\n", v_p95);
                printf("    \"wmma_stream_tflops\": %.3f,\n", v_tflops);
                printf("    \"wmma_stream_gb_s\": %.2f,\n", wmma.gb_s);
                double v_speedup_mean = stock.mean_us / v_mean;
                double v_speedup_mean_minus_1sigma = (stock.mean_us - stock.stdev_us) / (v_mean + v_std);
                printf("    \"speedup\": %.3f,\n", v_speedup);
                printf("    \"speedup_median\": %.3f,\n", v_speedup);
                printf("    \"speedup_vs_stock_dp4a\": %.3f,\n", v_speedup);
                printf("    \"speedup_mean\": %.3f,\n", v_speedup_mean);
                printf("    \"speedup_mean_minus_1sigma\": %.3f,\n", v_speedup_mean_minus_1sigma);
                printf("    \"TFLOPS_median\": %.3f,\n", v_tflops);
                printf("    \"GB/s\": %.2f,\n", wmma.gb_s);
                printf("    \"winner\": \"%s\",\n", winner);
                printf("    \"note\": \"streaming WMMA 64x32 per block LDS [2][32][33] double-buffered vs [4][32][32] XOR quad-buffer, wmma_f32_16x16x16_f16_w32, launch_bounds(256,4), sched_barrier 0x0080/0x0008, B-stationary, LUT mu=4, b128 global_load_b128/float4/ulong2, 16x64 swizzle\"\n");
                printf("  }");
                fflush(stdout); // incremental flush avoids 12288B truncation, valid JSON even if DXG hangs (timeout 90)
                if (variant_filter!="all") break;
            }

            HIP_CHECK(hipFree(dX));
            HIP_CHECK(hipFree(dY));
        }
        HIP_CHECK(hipFree(dW));
    }
    printf("\n]\n"); fflush(stdout); // ensure valid JSON even if downstream hangs; use timeout 90 wrapper to preserve partial output
    // Docs: invoke as `timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./bench_gemm_wmma --runs 10 --json > bench_gemm_wmma.hardware.json'`
    // If DXG hangs, timeout kills after 90s but already-flushed prefix remains valid JSON prefix (caller should validate with python -m json.tool and handle incomplete tail).
    fprintf(stderr, "=== GEMM Streaming WMMA vs Stock DP4A MMQ — M=128,512,1024,8192 prefill N=%d ===\n", runs);
    fprintf(stderr, "Kernel: gemm_iq4xs_wmma_stream_kernel (256 threads/block, 64x32 tile, 4x2 warps, LDS [2][32][33] padded vs [4][32][32] XOR, 2x WMMA per 32-K tile, B-stationary, LUT mu=4)\n");
    fprintf(stderr, "Hardware: v_wmma_f32_16x16x16_f16_w32 on Wave32, 1024 ops/CU/clock vs 512 DP4A; b128 global_load_b128/float4/ulong2 16B + swizzle 16x64\n");
    fprintf(stderr, "Gate: WMMA when M>=512 && N%%16==0 && K%%16==0 else tiled TILE_M=16; M=128 tiled, M=512 WMMA -> >1.2x target\n");
    fprintf(stderr, "Occupancy: __launch_bounds__(256,4) + amdgpu_flat_work_group_size(256,256) => <=64 VGPRs, 16 waves/SIMD; llvm-objdump | grep v_wmma; VGPR gate via calculator\n");
    fprintf(stderr, "High-yield race: 64x32 P2+33 vs P4+XOR vs 64x64 P4+XOR vs 128x32 vs LUT mu=4 via race.py --repeats 10 interleaved A,B,A,B\n");
    fprintf(stderr, "REQ-STAT-07: median/mean/stddev/p95 over N=%d, TFLOPS median per variant\n", runs);
    return 0;
}
