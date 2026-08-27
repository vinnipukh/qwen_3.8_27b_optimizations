// impl_broken.hip — Deliberately broken HIP kernel variant for KERN-01 discrimination testing.
// Contains intentional scale mutation and nibble swap bugs.
// Zero external llama.cpp / ggml header dependencies.
#include "block_iq4_xs.h"
#include "hip_helpers.h"

__constant__ int8_t c_kvalues_iq4nl_broken[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

template<int WarpSize = WARP_SIZE>
__global__ __launch_bounds__(256, 4)
void dequant_iq4xs_broken_kernel(
    const block_iq4_xs* __restrict__ src,
    float* __restrict__ dst,
    int64_t n_blocks
) {
    int64_t global_subblock = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_subblocks = n_blocks * 8;

    if (global_subblock >= total_subblocks) {
        return;
    }

    int64_t b = global_subblock / 8;
    int ib = static_cast<int>(global_subblock % 8);

    const block_iq4_xs* blk = &src[b];
    float d = fp16_to_fp32(blk->d);

    int ls_low  = (blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
    int ls_high = (blk->scales_h >> (2 * ib)) & 0x03;

    // BUG 1: Intentionally shifted by 3 instead of 4 (scale corruption)
    int ls = ls_low | (ls_high << 3);
    float dl = d * (static_cast<float>(ls) - 32.0f);

    const uint8_t* qs = blk->qs + ib * 16;
    float* y = dst + b * QK_K + ib * 32;

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        uint8_t q_val = qs[j];
        // BUG 2: Intentionally swapped low and high nibble assignments
        int lo = (q_val >> 4) & 0x0F;
        int hi = q_val & 0x0F;

        y[j]      = dl * static_cast<float>(c_kvalues_iq4nl_broken[lo]);
        y[j + 16] = dl * static_cast<float>(c_kvalues_iq4nl_broken[hi]);
    }
}

// Host launcher for broken kernel
hipError_t dequant_iq4xs_broken_gpu(
    const block_iq4_xs* d_src,
    float* d_dst,
    int64_t n_blocks,
    int warp_size,
    hipStream_t stream
) {
    int64_t total_subblocks = n_blocks * 8;
    int block_dim = 256;
    int grid_dim = static_cast<int>((total_subblocks + block_dim - 1) / block_dim);
    if (grid_dim < 1) grid_dim = 1;

    if (warp_size == 64) {
        hipLaunchKernelGGL((dequant_iq4xs_broken_kernel<64>), dim3(grid_dim), dim3(block_dim), 0, stream, d_src, d_dst, n_blocks);
    } else {
        hipLaunchKernelGGL((dequant_iq4xs_broken_kernel<32>), dim3(grid_dim), dim3(block_dim), 0, stream, d_src, d_dst, n_blocks);
    }
    return hipGetLastError();
}
