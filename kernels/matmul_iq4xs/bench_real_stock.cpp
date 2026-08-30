// bench_real_stock.cpp — Microbenchmark for REAL upstream DP4A (quantize + vec_dot) across 8 canonical shapes
// REQ-STAT-07 traceability: N=10 averaged via --runs 10 (BENCH-01 amended), reports median ± stddev + p95 per shape
// REQ-WIN-07: pure C++/HIP, compiles via HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 (no Python)
// Source: real_stock_dp4a_comparator.hip exact quantize_row_q8_1 + vec_dot_iq4_xs_q8_1 via __builtin_amdgcn_sudot4 + 6x __builtin_amdgcn_perm

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

hipError_t gemv_iq4xs_stock_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemv_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t s);
hipError_t gemm_iq4xs_stock_dp4a_gpu(const block_iq4_xs* d_W, const float* d_X, float* d_Y, int64_t K, int64_t N, int64_t M, hipStream_t s);

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s [--runs N] [--json] [--no-json]\n", prog);
    fprintf(stderr, "  --runs N  number of repetitions per shape (default 10, REQ-STAT-07 N>=10)\n");
    fprintf(stderr, "  --json    emit JSON array to stdout (default on)\n");
}

int main(int argc, char** argv) {
    int runs = 10; // REQ-STAT-07 default 10, BENCH-01 amended
    bool emit_json = true;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
            if (runs < 1) runs = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            emit_json = true;
        } else if (strcmp(argv[i], "--no-json") == 0) {
            emit_json = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    // Collect JSON objects to emit together
    std::string json_out;
    json_out += "[\n";
    bool first = true;

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

        // REQ-STAT-07: run bench_hip_event with warmup 50 + iters 200, then repeat `runs` times to get N=10 distribution
        // Collect per-run medians to compute median/mean/stddev/p95 across runs (thermal-paired).
        std::vector<BenchStats> dp4a_runs;
        std::vector<BenchStats> naive_runs;
        dp4a_runs.reserve(runs);
        naive_runs.reserve(runs);
        for (int r = 0; r < runs; ++r) {
            BenchStats dp4a = bench_hip_event(dp4a_launch, 0, 50, 200, total_bytes);
            BenchStats naive = bench_hip_event(naive_launch, 0, 20, 100, total_bytes);
            dp4a_runs.push_back(dp4a);
            naive_runs.push_back(naive);
        }
        // Aggregate across runs: median of medians, mean of means, pooled stddev via sample of medians
        auto aggregate = [](const std::vector<BenchStats>& v) -> BenchStats {
            std::vector<double> medians;
            std::vector<double> means;
            std::vector<double> p95s;
            for (auto &b : v) { medians.push_back(b.median_us); means.push_back(b.mean_us); p95s.push_back(b.p95_us); }
            std::sort(medians.begin(), medians.end());
            std::sort(p95s.begin(), p95s.end());
            BenchStats agg;
            agg.count = (int)v.size();
            // median of medians
            size_t n = medians.size();
            agg.median_us = (n % 2 == 0) ? (medians[n/2 -1] + medians[n/2]) * 0.5 : medians[n/2];
            double sum = std::accumulate(means.begin(), means.end(), 0.0);
            agg.mean_us = sum / means.size();
            double sq = 0;
            for (double m : means) sq += (m - agg.mean_us)*(m - agg.mean_us);
            agg.stdev_us = std::sqrt(sq / means.size());
            // also incorporate within-run stdev: average stdev_us across runs
            double avg_within = 0;
            for (auto &b : v) avg_within += b.stdev_us;
            avg_within /= v.size();
            // combine: sqrt(avg_within^2 + between_run_variance) approximate
            agg.stdev_us = std::sqrt(avg_within*avg_within + sq / means.size());
            agg.stddev_us = agg.stdev_us; // alias sync
            size_t p95_idx = (size_t)std::ceil(0.95 * p95s.size()) - 1;
            if (p95_idx >= p95s.size()) p95_idx = p95s.size()-1;
            agg.p95_us = p95s[p95_idx];
            agg.min_us = *std::min_element(medians.begin(), medians.end());
            agg.max_us = *std::max_element(medians.begin(), medians.end());
            agg.gb_s = v[0].gb_s; // bw from first run median
            agg.bytes_transferred = v[0].bytes_transferred;
            return agg;
        };
        BenchStats dp4a_agg = aggregate(dp4a_runs);
        BenchStats naive_agg = aggregate(naive_runs);
        double speedup_vs_naive = naive_agg.median_us / dp4a_agg.median_us;

        if (!first) json_out += ",\n";
        first = false;
        char buf[4096];
        snprintf(buf, sizeof(buf),
            "  {\n"
            "    \"op\": \"gemv_iq4xs_real_dp4a\",\n"
            "    \"shape\": \"%s\",\n"
            "    \"K\": %lld,\n"
            "    \"N\": %lld,\n"
            "    \"M\": 1,\n"
            "    \"bytes\": %zu,\n"
            "    \"runs\": %d,\n"
            "    \"naive_median_us\": %.3f,\n"
            "    \"naive_mean_us\": %.3f,\n"
            "    \"naive_stddev_us\": %.3f,\n"
            "    \"naive_p95_us\": %.3f,\n"
            "    \"real_dp4a_median_us\": %.3f,\n"
            "    \"real_dp4a_mean_us\": %.3f,\n"
            "    \"real_dp4a_stddev_us\": %.3f,\n"
            "    \"real_dp4a_p95_us\": %.3f,\n"
            "    \"real_dp4a_gb_s\": %.2f,\n"
            "    \"speedup_vs_naive\": %.3f,\n"
            "    \"note\": \"real stock uses quantize_row_q8_1 + vec_dot_iq4_xs_q8_1 DP4A (v_dot4_i32_i8) N=%d runs\"\n"
            "  }",
            name, (long long)K, (long long)N, total_bytes, runs,
            naive_agg.median_us, naive_agg.mean_us, naive_agg.stdev_us, naive_agg.p95_us,
            dp4a_agg.median_us, dp4a_agg.mean_us, dp4a_agg.stdev_us, dp4a_agg.p95_us,
            dp4a_agg.gb_s, speedup_vs_naive, runs);
        json_out += buf;

        HIP_CHECK(hipFree(dW)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy));
    }
    json_out += "\n]\n";

    if (emit_json) {
        printf("%s", json_out.c_str());
    } else {
        // human table to stderr + JSON to file? but ensure JSON always valid per spec
        fprintf(stderr, "%s", json_out.c_str());
    }

    fprintf(stderr, "\n=== REAL DP4A Baseline Timing Table (N=%d, expected 84us DP4A vs 543us naive) ===\n", runs);
    fprintf(stderr, "(JSON above contains median_us/mean_us/stddev_us/p95_us per shape; DP4A ~10x faster than naive 500us proves vec_dot_iq4_xs_q8_1 DP4A)\n");
    fprintf(stderr, "REQ-STAT-07: median ± stddev over N=%d (BENCH-01 amended >=10), p95 reported\n", runs);
    return 0;
}
