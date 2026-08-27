#pragma once

#include "hip_helpers.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <string>
#include <cstdio>

struct BenchStats {
    double median_us = 0.0;
    double p95_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double mean_us = 0.0;
    double stdev_us = 0.0;
    int count = 0;
    int warmup = 0;
    size_t bytes_transferred = 0;
    double gb_s = 0.0;
    int vgpr = 0;
    int lds = 0;
};

inline BenchStats bench_hip_event(
    std::function<void(hipStream_t)> launch,
    hipStream_t stream = 0,
    int warmup = 50,
    int iters = 200,
    size_t bytes_transferred = 0
) {
    // 1. Warmup loops
    for (int i = 0; i < warmup; ++i) {
        launch(stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    // 2. Measure individual iterations using hipEvent pairs
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    std::vector<double> times_us;
    times_us.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(start, stream));
        launch(stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times_us.push_back(static_cast<double>(ms) * 1000.0); // convert ms to us
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    std::sort(times_us.begin(), times_us.end());

    BenchStats stats;
    stats.count = iters;
    stats.warmup = warmup;
    stats.bytes_transferred = bytes_transferred;

    if (iters > 0) {
        stats.min_us = times_us.front();
        stats.max_us = times_us.back();
        stats.median_us = (iters % 2 == 0)
            ? (times_us[iters / 2 - 1] + times_us[iters / 2]) * 0.5
            : times_us[iters / 2];

        size_t p95_idx = static_cast<size_t>(std::ceil(0.95 * iters)) - 1;
        if (p95_idx >= static_cast<size_t>(iters)) p95_idx = iters - 1;
        stats.p95_us = times_us[p95_idx];

        double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
        stats.mean_us = sum / iters;

        double sq_sum = 0.0;
        for (double t : times_us) {
            sq_sum += (t - stats.mean_us) * (t - stats.mean_us);
        }
        stats.stdev_us = std::sqrt(sq_sum / iters);

        if (bytes_transferred > 0 && stats.median_us > 0.0) {
            // bytes / (median_us * 1e-6) / 1e9 = bytes / (median_us * 1e3) = GB/s
            stats.gb_s = static_cast<double>(bytes_transferred) / (stats.median_us * 1000.0);
        }
    }

    return stats;
}
