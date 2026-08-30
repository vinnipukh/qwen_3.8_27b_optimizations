// Vendored from kernels/matmul_iq4xs/impl_gemm_wmma.hip -- gfx1100 Wave32 GEMM M>>1
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "ggml-common.h"

#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)

typedef float v8f32 __attribute__((ext_vector_type(8)));
typedef _Float16 v16f16 __attribute__((ext_vector_type(16)));

static __device__ __constant__ const int8_t kvalues_iq4nl_dev_gemm[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static inline __device__ float custom_fp16_to_fp32(ggml_fp16_t h) {
    union { uint16_t u; __half h; } val;
    val.u = h;
    return __half2float(val.h);
}

template<int TILE_M = 16>
__global__ __launch_bounds__(256, 4) __attribute__((amdgpu_flat_work_group_size(256,256)))
static void gemm_iq4xs_tiled_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int64_t K,
    int64_t N,
    int64_t M
) {
    int64_t total_tiles = N * ((M + TILE_M - 1) / TILE_M);
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_tiles) return;
    int64_t tiles_per_row = (M + TILE_M - 1) / TILE_M;
    int64_t n = tid / tiles_per_row;
    int64_t tile_id = tid % tiles_per_row;
    int64_t m0 = tile_id * TILE_M;
    int64_t m1 = m0 + TILE_M;
    if (m1 > M) m1 = M;
    int tile_m = (int)(m1 - m0);

    int64_t blocks_per_row = K / 256;
    const block_iq4_xs* row_blocks = W + n * blocks_per_row;

    double acc[16] = {0.0};

    for (int64_t b = 0; b < blocks_per_row; ++b) {
        const block_iq4_xs* blk = &row_blocks[b];
        float d = custom_fp16_to_fp32(blk->d);
        for (int ib = 0; ib < 8; ++ib) {
            int ls_low  = (blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            int ls_high = (blk->scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            double dl = static_cast<double>(d) * static_cast<double>(static_cast<float>(ls) - 32.0f);

            struct Aligned16 { uint64_t lo; uint64_t hi; };
            Aligned16 qs_val;
            qs_val.lo = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16);
            qs_val.hi = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16 + 8);
            const uint8_t* qs_bytes = reinterpret_cast<const uint8_t*>(&qs_val);
            int64_t k_base = b * 256 + ib * 32;

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                uint8_t q_byte = qs_bytes[j];
                int lo = q_byte & 0x0F;
                int hi = (q_byte >> 4) & 0x0F;
                double w_lo = dl * static_cast<double>(kvalues_iq4nl_dev_gemm[lo]);
                double w_hi = dl * static_cast<double>(kvalues_iq4nl_dev_gemm[hi]);

                #pragma unroll
                for (int tm = 0; tm < 16; ++tm) {
                    if (tm >= tile_m) break;
                    int64_t m = m0 + tm;
                    double x_lo = static_cast<double>(X[m * K + (k_base + j)]);
                    double x_hi = static_cast<double>(X[m * K + (k_base + j + 16)]);
                    acc[tm] += w_lo * x_lo;
                    acc[tm] += w_hi * x_hi;
                }
            }
        }
    }
    #pragma unroll
    for (int tm = 0; tm < 16; ++tm) {
        if (tm >= tile_m) break;
        int64_t m = m0 + tm;
        Y[m * N + n] = static_cast<float>(acc[tm]);
    }
}

__global__ __launch_bounds__(256, 4) __attribute__((amdgpu_flat_work_group_size(256,256)))
static void gemm_iq4xs_wmma_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int64_t K,
    int64_t N,
    int64_t M
) {
    const int BLOCK_N = 64;
    const int BLOCK_M = 32;
    int block_n = blockIdx.x * BLOCK_N;
    int block_m = blockIdx.y * BLOCK_M;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lIdx = tid % 32;
    int lane = lIdx % 16;
    int half_wave = lIdx / 16;

    int warp_n = warp_id / 2;
    int warp_m = warp_id % 2;
    int tile_n = block_n + warp_n * 16;
    int tile_m = block_m + warp_m * 16;

    if (tile_n >= N || tile_m >= M) return;

    v8f32 c_frag = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int K_tiles = (int)(K / 16);
    int64_t blocks_per_row = K / 256;

    for (int kt = 0; kt < K_tiles; ++kt) {
        int k0 = kt * 16;

        v16f16 a_frag;
        v16f16 b_frag;

        int64_t g_n = tile_n + lane;
        #pragma unroll
        for (int ele = 0; ele < 16; ++ele) {
            int64_t g_k = k0 + ele;
            if (g_n < N && g_k < K) {
                int64_t b = g_k / 256;
                int ib = (g_k % 256) / 32;
                int j = (g_k % 32) % 16;
                bool hi = (g_k % 32) >= 16;
                const block_iq4_xs* blk = W + g_n * blocks_per_row + b;
                float d = custom_fp16_to_fp32(blk->d);
                int ls_low  = (blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
                int ls_high = (blk->scales_h >> (2 * ib)) & 0x03;
                int ls = ls_low | (ls_high << 4);
                float dl = d * (static_cast<float>(ls) - 32.0f);
                uint8_t q_byte = blk->qs[ib * 16 + j];
                int q = hi ? ((q_byte >> 4) & 0x0F) : (q_byte & 0x0F);
                float w = dl * static_cast<float>(kvalues_iq4nl_dev_gemm[q]);
                a_frag[ele] = (_Float16)w;
            } else {
                a_frag[ele] = (_Float16)0.0f;
            }
        }

        int64_t g_m = tile_m + lane;
        #pragma unroll
        for (int ele = 0; ele < 16; ++ele) {
            int64_t g_k = k0 + ele;
            if (g_k < K && g_m < M) {
                b_frag[ele] = (_Float16)X[g_m * K + g_k];
            } else {
                b_frag[ele] = (_Float16)0.0f;
            }
        }

        c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
    }

    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        int r = ele * 2 + half_wave;
        int64_t out_n = tile_n + r;
        int64_t out_m = tile_m + lane;
        if (out_n < N && out_m < M) {
            Y[out_m * N + out_n] = c_frag[ele];
        }
    }
}

inline bool custom_gemm_iq4xs_can_handle(int64_t K, int64_t N, int64_t M, ggml_type type) {
    // Real gate: type==IQ4_XS && M>=16 && K%256==0 && N%16==0 (plus K>0,N>0)
    if (type != GGML_TYPE_IQ4_XS) return false;
    if (M < 16) return false;
    if (K <= 0 || N <= 0) return false;
    if (K % 256 != 0) return false;
    if (N % 16 != 0) return false;
    return true;
}

inline hipError_t custom_gemm_iq4xs_dispatch(
    const block_iq4_xs* W,
    const float* X,
    float* Y,
    int64_t K,
    int64_t N,
    int64_t M,
    hipStream_t stream
) {
    bool wmma_ok = (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0) && (M >= 512) && (N >= 1024);
    if (wmma_ok) {
        const int BLOCK_N = 64;
        const int BLOCK_M = 32;
        int grid_x = (int)((N + BLOCK_N - 1) / BLOCK_N);
        int grid_y = (int)((M + BLOCK_M - 1) / BLOCK_M);
        hipLaunchKernelGGL(gemm_iq4xs_wmma_kernel, dim3(grid_x, grid_y), dim3(256), 0, stream, W, X, Y, K, N, M);
        hipError_t err = hipGetLastError();
        if (err == hipSuccess) return err;
    }
    const int TILE_M = 16;
    int64_t tiles_per_row = (M + TILE_M - 1) / TILE_M;
    int64_t total_tiles = N * tiles_per_row;
    const int BLOCK = 256;
    int64_t grid = (total_tiles + BLOCK - 1) / BLOCK;
    if (grid > 2147483647) return hipErrorInvalidConfiguration;
    hipLaunchKernelGGL((gemm_iq4xs_tiled_kernel<16>), dim3((unsigned)grid), dim3(BLOCK), 0, stream, W, X, Y, K, N, M);
    return hipGetLastError();
}

#endif
