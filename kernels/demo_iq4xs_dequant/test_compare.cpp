// test_compare.cpp — Numerical comparison harness for block_iq4_xs dequantization.
// Zero external llama.cpp / ggml header dependencies.
#include "block_iq4_xs.h"
#include "hip_helpers.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>

// Declarations
void dequantize_row_iq4_xs_cpu(const block_iq4_xs* src, float* dst, int64_t n_blocks);

#ifdef TEST_BROKEN
hipError_t dequant_iq4xs_broken_gpu(const block_iq4_xs* d_src, float* d_dst, int64_t n_blocks, int warp_size, hipStream_t stream = 0);
#define RUN_GPU dequant_iq4xs_broken_gpu
#else
hipError_t dequant_iq4xs_gpu(const block_iq4_xs* d_src, float* d_dst, int64_t n_blocks, int warp_size, hipStream_t stream = 0);
#define RUN_GPU dequant_iq4xs_gpu
#endif

struct CompareResult {
    std::string name;
    int64_t n_blocks;
    double max_abs;
    double mean_abs;
    double max_rel;
    double cosine;
    bool has_nan_or_inf;
    bool pass;
};

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    return {};
}

CompareResult run_comparison_on_blocks(
    const std::string& name,
    const block_iq4_xs* host_src,
    int64_t n_blocks,
    int warp_size
) {
    int64_t total_elements = n_blocks * QK_K;
    std::vector<float> host_ref(total_elements, 0.0f);
    std::vector<float> host_gpu(total_elements, 0.0f);

    // 1. CPU reference
    dequantize_row_iq4_xs_cpu(host_src, host_ref.data(), n_blocks);

    // 2. GPU execution
    block_iq4_xs* d_src = nullptr;
    float* d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, n_blocks * sizeof(block_iq4_xs)));
    HIP_CHECK(hipMalloc(&d_dst, total_elements * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_src, host_src, n_blocks * sizeof(block_iq4_xs), hipMemcpyHostToDevice));
    HIP_CHECK(RUN_GPU(d_src, d_dst, n_blocks, warp_size, 0));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host_gpu.data(), d_dst, total_elements * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    // 3. Compare metrics
    double max_abs = 0.0;
    double sum_abs = 0.0;
    double max_rel = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_gpu = 0.0;
    bool has_nan_or_inf = false;

    for (int64_t i = 0; i < total_elements; ++i) {
        float r = host_ref[i];
        float g = host_gpu[i];

        if (std::isnan(r) || std::isnan(g) || std::isinf(r) || std::isinf(g)) {
            has_nan_or_inf = true;
        }

        double diff = std::abs(static_cast<double>(r) - static_cast<double>(g));
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;

        if (std::abs(r) > 1e-3f) {
            double rel = diff / std::abs(static_cast<double>(r));
            if (rel > max_rel) max_rel = rel;
        }

        dot += static_cast<double>(r) * static_cast<double>(g);
        norm_ref += static_cast<double>(r) * static_cast<double>(r);
        norm_gpu += static_cast<double>(g) * static_cast<double>(g);
    }

    double mean_abs = total_elements > 0 ? (sum_abs / total_elements) : 0.0;
    double denom = std::sqrt(norm_ref) * std::sqrt(norm_gpu);
    double cosine = (denom > 1e-12) ? (dot / denom) : ((norm_ref < 1e-12 && norm_gpu < 1e-12) ? 1.0 : 0.0);

    // Strict gate requirements: max_abs < 1e-5, mean_abs < 1e-6, cosine > 0.99999
    bool pass = (!has_nan_or_inf) && (max_abs < 1e-5) && (mean_abs < 1e-6) && (cosine >= 0.99999);

    CompareResult res;
    res.name = name;
    res.n_blocks = n_blocks;
    res.max_abs = max_abs;
    res.mean_abs = mean_abs;
    res.max_rel = max_rel;
    res.cosine = cosine;
    res.has_nan_or_inf = has_nan_or_inf;
    res.pass = pass;

    return res;
}

int main(int argc, char** argv) {
    std::string custom_fixture = "";
    int warp_size = 32;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fixture") == 0 && i + 1 < argc) {
            custom_fixture = argv[++i];
        } else if (std::strcmp(argv[i], "--warp") == 0 && i + 1 < argc) {
            warp_size = std::atoi(argv[++i]);
        }
    }

    std::vector<std::pair<std::string, std::string>> fixture_paths;
    if (!custom_fixture.empty()) {
        fixture_paths.push_back({"custom_fixture", custom_fixture});
    } else {
        fixture_paths = {
            {"synthetic_zero", "kernels/fixtures/synthetic_zero.bin"},
            {"synthetic_max_scale", "kernels/fixtures/synthetic_max_scale.bin"},
            {"synthetic_split_half", "kernels/fixtures/synthetic_split_half.bin"},
            {"synthetic_nibble_extremes", "kernels/fixtures/synthetic_nibble_extremes.bin"},
            {"synthetic_subblock_isolated", "kernels/fixtures/synthetic_subblock_isolated.bin"},
            {"blk_0_ffn_down_weight", "kernels/fixtures/blk_0_ffn_down_weight.bin"},
            {"blk_0_attn_gate_weight", "kernels/fixtures/blk_0_attn_gate_weight.bin"},
            {"token_embd_weight", "kernels/fixtures/token_embd_weight.bin"}
        };
    }

    bool all_passed = true;
    int tested_count = 0;

#ifdef TEST_BROKEN
    printf("=== RUNNING DEMO DEQUANT COMPARISON (BROKEN KERNEL TARGET) [warp=%d] ===\n", warp_size);
#else
    printf("=== RUNNING DEMO DEQUANT COMPARISON (CORRECT KERNEL TARGET) [warp=%d] ===\n", warp_size);
#endif

    for (const auto& kv : fixture_paths) {
        std::vector<uint8_t> raw = read_file_bytes(kv.second);
        if (raw.empty() || raw.size() % sizeof(block_iq4_xs) != 0) {
            fprintf(stderr, "[WARNING] Cannot read fixture '%s' at '%s' (size=%zu bytes)\n",
                    kv.first.c_str(), kv.second.c_str(), raw.size());
            continue;
        }

        int64_t n_blocks = raw.size() / sizeof(block_iq4_xs);
        const block_iq4_xs* blocks = reinterpret_cast<const block_iq4_xs*>(raw.data());

        CompareResult res = run_comparison_on_blocks(kv.first, blocks, n_blocks, warp_size);
        tested_count++;

        printf("[FIXTURE %-28s] n_blocks=%2ld | max_abs=%.2e mean_abs=%.2e max_rel=%.2e cosine=%.8f | %s\n",
               res.name.c_str(), (long)res.n_blocks, res.max_abs, res.mean_abs, res.max_rel, res.cosine,
               res.pass ? "PASS" : "FAIL");

        if (!res.pass) {
            all_passed = false;
        }
    }

    if (tested_count == 0) {
        fprintf(stderr, "[ERROR] No fixtures could be loaded and tested!\n");
        return 2;
    }

    printf("=== FINAL RESULT: %s (%d/%d passed) ===\n",
           all_passed ? "PASS" : "FAIL", all_passed ? tested_count : 0, tested_count);

    return all_passed ? 0 : 1;
}
