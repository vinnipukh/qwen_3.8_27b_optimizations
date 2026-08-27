// Vendored from kernels/matmul_iq4xs/impl_gemv_gfx1100.hip -- gfx1100 Wave32 GEMV M=1
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "ggml-common.h"

#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)

static __device__ __constant__ const int8_t kvalues_iq4nl_dev[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static inline __device__ float custom_fp16_to_fp32(ggml_fp16_t h) {
    union { uint16_t u; __half h; } val;
    val.u = h;
    return __half2float(val.h);
}

__global__ __launch_bounds__(256, 4) __attribute__((amdgpu_flat_work_group_size(256,256)))
static void gemv_iq4xs_gfx1100_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t K,
    int64_t N
) {
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ROWS_PER_BLOCK = 256 / THREADS_PER_ROW;

    int tid = threadIdx.x;
    int group_id = tid / THREADS_PER_ROW;
    int lane_in_group = tid & (THREADS_PER_ROW - 1);

    int64_t row = (int64_t)blockIdx.x * ROWS_PER_BLOCK + group_id;
    double thread_sum = 0.0;

    if (row < N) {
        int64_t blocks_per_row = K / 256;
        const block_iq4_xs* row_blocks = W + row * blocks_per_row;

        for (int64_t b = 0; b < blocks_per_row; ++b) {
            const block_iq4_xs* blk = &row_blocks[b];
            float d = custom_fp16_to_fp32(blk->d);

            int ib = lane_in_group;
            int ls_low  = (blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            int ls_high = (blk->scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            double dl = static_cast<double>(d) * static_cast<double>(static_cast<float>(ls) - 32.0f);

            struct Aligned16 { uint64_t lo; uint64_t hi; };
            Aligned16 qs_val;
            qs_val.lo = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16);
            qs_val.hi = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16 + 8);
            const uint8_t* qs_bytes = reinterpret_cast<const uint8_t*>(&qs_val);

            const float* x_base = x + b * 256 + ib * 32;

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                uint8_t q_byte = qs_bytes[j];
                int lo = q_byte & 0x0F;
                int hi = (q_byte >> 4) & 0x0F;

                double w_lo = dl * static_cast<double>(kvalues_iq4nl_dev[lo]);
                double w_hi = dl * static_cast<double>(kvalues_iq4nl_dev[hi]);

                float x_lo = x_base[j];
                float x_hi = x_base[j + 16];
                thread_sum += w_lo * static_cast<double>(x_lo);
                thread_sum += w_hi * static_cast<double>(x_hi);
            }
        }
    }

    __shared__ double sh[256];
    sh[tid] = thread_sum;
    __syncthreads();

    if (row < N && lane_in_group == 0) {
        double acc = 0.0;
        #pragma unroll
        for (int t = 0; t < THREADS_PER_ROW; ++t) {
            acc += sh[group_id * THREADS_PER_ROW + t];
        }
        y[row] = static_cast<float>(acc);
    }
}

inline bool custom_gemv_iq4xs_can_handle(int64_t K, int64_t N, int64_t M, ggml_type type) {
    if (type != GGML_TYPE_IQ4_XS) return false;
    if (M != 1) return false;
    if (K <= 0 || N <= 0 || K % 256 != 0) return false;
    if (K != 5120 && K != 17408) return false;
    if (N != 5120 && N != 6144 && N != 17408) return false;
    return true;
}

inline hipError_t custom_gemv_iq4xs_dispatch(
    const block_iq4_xs* W,
    const float* x,
    float* y,
    int64_t K,
    int64_t N,
    hipStream_t stream
) {
    constexpr int BLOCK = 256;
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ROWS_PER_BLOCK = BLOCK / THREADS_PER_ROW;
    int64_t grid = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (grid > 2147483647) return hipErrorInvalidConfiguration;

    hipLaunchKernelGGL(gemv_iq4xs_gfx1100_kernel, dim3((unsigned)grid), dim3(BLOCK), 0, stream, W, x, y, K, N);
    return hipGetLastError();
}

#endif
