#include "block_iq4_xs.h"
#include "hip_helpers.h"

// Templated stub kernel on WarpSize (32 or 64) with launch bounds
template<int WarpSize = WARP_SIZE>
__global__ __launch_bounds__(256, 4)
void template_stub_kernel(const block_iq4_xs* __restrict__ src, float* __restrict__ dst, int64_t n_blocks) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_elements = n_blocks * QK_K;

    if (tid < total_elements) {
        int64_t b = tid / QK_K;
        int i = tid % QK_K;
        float d = fp16_to_fp32(src[b].d);
        dst[tid] = d * static_cast<float>(i % 16);
    }
}

// Host launcher
hipError_t template_stub_gpu(const block_iq4_xs* d_src, float* d_dst, int64_t n_blocks, int warp_size, hipStream_t stream) {
    int64_t total_elements = n_blocks * QK_K;
    int block_dim = 256;
    int grid_dim = static_cast<int>((total_elements + block_dim - 1) / block_dim);

    if (warp_size == 64) {
        hipLaunchKernelGGL((template_stub_kernel<64>), dim3(grid_dim), dim3(block_dim), 0, stream, d_src, d_dst, n_blocks);
    } else {
        hipLaunchKernelGGL((template_stub_kernel<32>), dim3(grid_dim), dim3(block_dim), 0, stream, d_src, d_dst, n_blocks);
    }
    return hipGetLastError();
}
