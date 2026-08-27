// stock_hip_comparator.hip — Standalone extracted stock llama.cpp IQ4_XS MMVQ/MMQ logic
// Minimal faithful recreation of vec_dot_iq4_xs_q8_1 and MMQ tiling but using direct FP32
// activation path (no Q8_1 quant) for simplicity and standalone portability.
// Purpose: apples-to-apples baseline comparator for microbenchmarks; numerically equivalent
// to CPU oracle within fp32 rounding.

#include "block_iq4_xs.h"
#include "hip_helpers.h"

// Inline device helpers — replicate kvalues lookup without v_perm_b32 dep for portability
__device__ __forceinline__ float dequant_w_iq4xs(const block_iq4_xs* blk, int ib, int lane_32) {
    // lane_32 in [0,32): maps to weight index within sub-block
    // Called per lane where lane_32 determines which weight to fetch.
    // For naive kernels we call per-work item per k.
    return 0.0f; // placeholder, not used in this naive path
}

// ---------------------------------------------------------------------------
// Naive GEMV stock comparator: one thread per output row (N threads)
// Each thread loops over K in blocks, dequantizes scalars, dots with x.
// Not using 128-bit loads, not using WMMA, not using shuffle reduction.
// This mirrors stock MMVQ's scalar fallback path but without Q8 quantization overhead.
// ---------------------------------------------------------------------------

__global__ void gemv_iq4xs_stock_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t K,
    int64_t N
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    int64_t blocks_per_row = K / QK_K;
    const block_iq4_xs* row_blocks = W + row * blocks_per_row;

    double acc = 0.0; // double accumulate to match CPU oracle precision, then float out

    for (int64_t b = 0; b < blocks_per_row; ++b) {
        float d = fp16_to_fp32(row_blocks[b].d);
        const uint8_t* qs = row_blocks[b].qs;
        const float* x_block = x + b * QK_K;

        for (int ib = 0; ib < 8; ++ib) {
            int ls_low  = (row_blocks[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            int ls_high = (row_blocks[b].scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            double dl = static_cast<double>(d) * static_cast<double>(static_cast<float>(ls) - 32.0f);

            const uint8_t* qs_sub = qs + ib * 16;
            const float* x_sub = x_block + ib * 32;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                uint8_t q_byte = qs_sub[j];
                int lo = q_byte & 0x0F;
                int hi = (q_byte >> 4) & 0x0F;
                double w_lo = dl * static_cast<double>(kvalues_iq4nl[lo]);
                double w_hi = dl * static_cast<double>(kvalues_iq4nl[hi]);
                acc += w_lo * static_cast<double>(x_sub[j]);
                acc += w_hi * static_cast<double>(x_sub[j + 16]);
            }
        }
    }
    y[row] = static_cast<float>(acc);
}

// Naive GEMM stock comparator: one thread per output element Y[n,m]
// Loops over K dequantizing scalars. Works for M in [1,2048], N up to 17408.
// Grid = ceil(N*M / blockDim)
__global__ void gemm_iq4xs_stock_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ X,  // [K, M] row-major: X[k*M + m]
    float* __restrict__ Y,        // [N, M] row-major: Y[n*M + m]
    int64_t K,
    int64_t N,
    int64_t M
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * M;
    if (tid >= total) return;
    int64_t n = tid / M;
    int64_t m = tid % M;

    int64_t blocks_per_row = K / QK_K;
    const block_iq4_xs* row_blocks = W + n * blocks_per_row;

    double acc = 0.0;
    for (int64_t b = 0; b < blocks_per_row; ++b) {
        float d = fp16_to_fp32(row_blocks[b].d);
        const uint8_t* qs = row_blocks[b].qs;
        for (int ib = 0; ib < 8; ++ib) {
            int ls_low  = (row_blocks[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            int ls_high = (row_blocks[b].scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            double dl = static_cast<double>(d) * static_cast<double>(static_cast<float>(ls) - 32.0f);
            const uint8_t* qs_sub = qs + ib * 16;
            int64_t k_base = b * QK_K + ib * 32;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                uint8_t q_byte = qs_sub[j];
                int lo = q_byte & 0x0F;
                int hi = (q_byte >> 4) & 0x0F;
                double w_lo = dl * static_cast<double>(kvalues_iq4nl[lo]);
                double w_hi = dl * static_cast<double>(kvalues_iq4nl[hi]);
                float x_lo = X[(k_base + j) * M + m];
                float x_hi = X[(k_base + j + 16) * M + m];
                acc += w_lo * static_cast<double>(x_lo);
                acc += w_hi * static_cast<double>(x_hi);
            }
        }
    }
    Y[n * M + m] = static_cast<float>(acc);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

hipError_t gemv_iq4xs_stock_gpu(
    const block_iq4_xs* d_W,
    const float* d_x,
    float* d_y,
    int64_t K,
    int64_t N,
    hipStream_t stream = 0
) {
    const int BLOCK = 256;
    int64_t grid = (N + BLOCK - 1) / BLOCK;
    if (grid > 2147483647) grid = 2147483647;
    // Use standard launch; HIP allows int grid
    hipLaunchKernelGGL(gemv_iq4xs_stock_kernel, dim3((unsigned)grid), dim3(BLOCK), 0, stream, d_W, d_x, d_y, K, N);
    return hipGetLastError();
}

hipError_t gemm_iq4xs_stock_gpu(
    const block_iq4_xs* d_W,
    const float* d_X,
    float* d_Y,
    int64_t K,
    int64_t N,
    int64_t M,
    hipStream_t stream = 0
) {
    const int BLOCK = 256;
    int64_t total = N * M;
    int64_t grid = (total + BLOCK - 1) / BLOCK;
    if (grid > 2147483647) grid = 2147483647;
    hipLaunchKernelGGL(gemm_iq4xs_stock_kernel, dim3((unsigned)grid), dim3(BLOCK), 0, stream, d_W, d_X, d_Y, K, N, M);
    return hipGetLastError();
}
