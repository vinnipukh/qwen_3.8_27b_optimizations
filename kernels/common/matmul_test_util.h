#pragma once
// matmul_test_util.h — Shared testing utilities and metric computations
// Deduplicated harness utilities for IQ4_XS GEMV and GEMM tests & benchmarks.

#include "block_iq4_xs.h"
#include "hip_helpers.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstdint>

struct Metrics {
    double max_abs;
    double mean_abs;
    double max_rel;
    double cosine;
    bool bad;
    bool pass;
};

inline Metrics compute_metrics(const std::vector<float>& ref, const std::vector<float>& gpu) {
    size_t n = ref.size();
    double max_abs = 0.0, sum_abs = 0.0, max_rel = 0.0, dot = 0.0, nr = 0.0, ng = 0.0;
    bool bad = false;
    for (size_t i = 0; i < n; ++i) {
        float r = ref[i], g = gpu[i];
        if (std::isnan(r) || std::isnan(g) || std::isinf(r) || std::isinf(g)) {
            bad = true;
        }
        double d = std::abs((double)r - (double)g);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
        if (std::abs(r) > 1e-3) {
            double rel = d / std::abs((double)r);
            max_rel = std::max(max_rel, rel);
        }
        dot += (double)r * (double)g;
        nr += (double)r * (double)r;
        ng += (double)g * (double)g;
    }
    double mean_abs = n ? sum_abs / n : 0.0;
    double denom = std::sqrt(nr) * std::sqrt(ng);
    double cosine = denom > 1e-12 ? dot / denom : ((nr < 1e-12 && ng < 1e-12) ? 1.0 : 0.0);
    bool pass = !bad && cosine >= 0.999;
    return {max_abs, mean_abs, max_rel, cosine, bad, pass};
}

inline void gen_iq4xs_weights(std::vector<block_iq4_xs>& W, int64_t K, int64_t N, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d_dist(0.008f, 0.6f);
    std::uniform_int_distribution<int> sh_dist(0, 65535), b_dist(0, 255);
    int64_t blocks_per_row = K / QK_K;
    int64_t total = N * blocks_per_row;
    W.resize(total);
    for (int64_t i = 0; i < total; ++i) {
        float dv = d_dist(rng);
        W[i].d = fp32_to_fp16(dv);
        W[i].scales_h = (uint16_t)sh_dist(rng);
        for (int k = 0; k < 4; ++k) {
            W[i].scales_l[k] = (uint8_t)b_dist(rng);
        }
        for (int k = 0; k < 128; ++k) {
            W[i].qs[k] = (uint8_t)b_dist(rng);
        }
    }
}
