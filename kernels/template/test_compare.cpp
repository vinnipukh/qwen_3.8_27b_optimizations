#include "block_iq4_xs.h"
#include "hip_helpers.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Forward declarations
void template_ref_cpu(const block_iq4_xs* src, float* dst, int64_t n_blocks);
hipError_t template_stub_gpu(const block_iq4_xs* d_src, float* d_dst, int64_t n_blocks, int warp_size, hipStream_t stream = 0);

int main(int argc, char** argv) {
    const int64_t n_blocks = 8;
    const int64_t total_elements = n_blocks * QK_K;

    std::vector<block_iq4_xs> host_src(n_blocks);
    for (int64_t b = 0; b < n_blocks; ++b) {
        host_src[b].d = fp32_to_fp16(1.5f + static_cast<float>(b) * 0.25f);
        host_src[b].scales_h = static_cast<uint16_t>(b * 17);
        for (int k = 0; k < 4; ++k) host_src[b].scales_l[k] = static_cast<uint8_t>(k * 31);
        for (int k = 0; k < 128; ++k) host_src[b].qs[k] = static_cast<uint8_t>((k * 7) & 0xFF);
    }

    std::vector<float> host_ref(total_elements, 0.0f);
    std::vector<float> host_gpu(total_elements, 0.0f);

    // Run CPU reference
    template_ref_cpu(host_src.data(), host_ref.data(), n_blocks);

    // Allocate GPU buffers
    block_iq4_xs* d_src = nullptr;
    float* d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, n_blocks * sizeof(block_iq4_xs)));
    HIP_CHECK(hipMalloc(&d_dst, total_elements * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_src, host_src.data(), n_blocks * sizeof(block_iq4_xs), hipMemcpyHostToDevice));

    // Launch GPU stub (testing warp 32)
    HIP_CHECK(template_stub_gpu(d_src, d_dst, n_blocks, 32));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(host_gpu.data(), d_dst, total_elements * sizeof(float), hipMemcpyDeviceToHost));

    // Numerical validation
    double max_abs = 0.0;
    double sum_abs = 0.0;
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
        dot += static_cast<double>(r) * static_cast<double>(g);
        norm_ref += static_cast<double>(r) * static_cast<double>(r);
        norm_gpu += static_cast<double>(g) * static_cast<double>(g);
    }

    double mean_abs = sum_abs / total_elements;
    double denom = std::sqrt(norm_ref) * std::sqrt(norm_gpu);
    double cosine = (denom > 1e-12) ? (dot / denom) : 1.0;

    printf("[template_test] max_abs=%.2e mean_abs=%.2e cosine=%.8f has_nan=%d\n",
           max_abs, mean_abs, cosine, has_nan_or_inf ? 1 : 0);

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    bool pass = (!has_nan_or_inf) && (max_abs < 1e-5) && (mean_abs < 1e-6) && (cosine > 0.99999);
    printf("[template_test] Result: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
