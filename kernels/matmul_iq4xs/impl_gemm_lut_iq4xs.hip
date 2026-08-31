// impl_gemm_lut_iq4xs.hip — LUT variant μ=4 for GEMM dequant (secondary high-yield kernel, Phase 07-03)
// B-stationary + LUT-GEMM BCQ Σαb μ=4 16-entry half LUT eliminates inline dequant d*(ls-32)*kvalues_iq4nl
// Architecture: Same tiling 64x32 per block (4x2 warps, 8 warps =256 thr) of 16x16x16 WMMA tiles as impl_gemm_wmma_stream.hip
// But dequant baked offline: LUT per tile 16-entry half (32B/LUT) via tools/swizzle_iq4xs.py bake step, same tiling
// See output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md LUT-GEMM μ=4
// Offline bake: d*(ls-32) baked into LUT entries, runtime just LUT lookup + WMMA, no inline dl*kvalues
// VGPR gate: LUT variant saves VGPR on dequant (8 vs inline), same __launch_bounds__(256,4)+amdgpu_flat_work_group_size(256,256) <=64 VGPR
// B-stationary: IQ4_XS weight tile in VGPR frag (A_frag 8 VGPR, B_frag 8 VGPR wave32) while streaming Q8_1 activation via LDS ping-pong

#include "block_iq4_xs.h"
#include "hip_helpers.h"

typedef float v8f32 __attribute__((ext_vector_type(8)));
typedef _Float16 v16f16 __attribute__((ext_vector_type(16)));

// LUT μ=4: 16-entry half LUT per 32-weight subblock, 32B per LUT, baked offline d*(ls-32)*kvalues_iq4nl
struct lut_entry {
    _Float16 lut[16]; // 16-entry half LUT, 32B
};

// High-yield variant notes: LUT μ=4 16-entry half bake vs inline dequant, B-stationary weight in VGPR, sB[2][32][33] +33 vs sB[4][32][32] XOR
// P=2 [2][32][33] double-buffer vs P=4 [4][32][32] quad-buffer with __builtin_amdgcn_sched_barrier(0x0080 DS / 0x0008 WMMA)
// b128 coalescing via __builtin_amdgcn_global_load_b128 / float4 / ulong2 16B + __builtin_assume_aligned(ptr,16)
// Offline 16x64 swizzle to 128B cache lines via tools/swizzle_iq4xs.py (MARLIN 16x64 swizzle B in registers)
// Calculator: amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d predicts A 8 VGPR / D 8 VGPR wave32 <=64

__global__ __launch_bounds__(256,4) __attribute__((amdgpu_flat_work_group_size(256,256)))
void gemm_iq4xs_lut_kernel(
    const block_iq4_xs* __restrict__ W,     // original IQ4_XS (or LUT-baked weight buffer)
    const _Float16* __restrict__ LUT,      // baked LUT: [N * blocks_per_row * 8][16] half entries (32B each)
    const float* __restrict__ X,           // [K, M]
    float* __restrict__ Y,                 // [N, M]
    int64_t K, int64_t N, int64_t M)
{
    const int BLOCK_N = 64;
    const int BLOCK_M = 32;
    const int K_TILE = 32;
    int block_n = blockIdx.x * BLOCK_N;
    int block_m = blockIdx.y * BLOCK_M;
    int tid = threadIdx.x;
    int lIdx = tid % 32;
    int warp_id = tid / 32;
    int lane = lIdx % 16;
    int half_wave = lIdx / 16;
    int warp_n = warp_id / 2;
    int warp_m = warp_id % 2;
    int tile_n = block_n + warp_n * 16;
    int tile_m = block_m + warp_m * 16;

    // LDS [2][32][33] double-buffered vs P=4 [4][32][32] XOR variant; here P=2 +33 default for LUT variant
    __shared__ _Float16 sB[2][32][33]; // sB[2][32][33] padded +33 (+3%) vs XOR preshuffle x'=(y%(64/8))^x 0%
    // __shared__ _Float16 sB_P4[4][32][32]; // P=4 XOR alternative, pinned with sched_barrier 0x0080/0x0008

    v8f32 c_frag = {0};
    int64_t blocks_per_row = K / QK_K;
    int num_k_tiles = (int)(K / K_TILE);
    int buf = 0;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k0 = kt * K_TILE;
        __builtin_amdgcn_sched_barrier(0x0080);
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            int linear = tid + iter * 256;
            int kr = linear / 32;
            int cm = linear % 32;
            int64_t gk = (int64_t)k0 + kr;
            int64_t gm = (int64_t)block_m + cm;
            _Float16 v = (_Float16)0.0f;
            if (gk < K && gm < M) v = (_Float16)X[gk * M + gm];
            sB[buf][kr][cm] = v;
        }
        __builtin_amdgcn_sched_barrier(0x0008);
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k_sub = sub * 16;
            int k_base = k0 + k_sub;

            v16f16 a_frag;
            #pragma unroll
            for (int ele = 0; ele < 16; ++ele) {
                int64_t g_n = (int64_t)tile_n + lane;
                int64_t g_k = (int64_t)k_base + ele;
                if (g_n < N && g_k < K) {
                    // LUT path: lookup baked half instead of inline d*(ls-32)*kvalues
                    // LUT index: block of this g_k within row + subblock
                    int64_t b = g_k / QK_K;
                    int ib = (int)((g_k % QK_K) / 32);
                    int j = (int)((g_k % 32) % 16);
                    bool hi = (g_k % 32) >= 16;
                    const block_iq4_xs* blk = W + g_n * blocks_per_row + b;
                    uint8_t qb = blk->qs[ib*16 + j];
                    int q = hi ? ((qb>>4)&0xF) : (qb & 0xF);
                    // If LUT provided, use it; else fallback inline (demonstrates both paths)
                    if (LUT != nullptr) {
                        int64_t lut_idx = (g_n * blocks_per_row + b) * 8 + ib;
                        a_frag[ele] = LUT[lut_idx*16 + q];
                    } else {
                        float d = fp16_to_fp32(blk->d);
                        int ls_low = (blk->scales_l[ib/2] >> (4*(ib&1))) & 0x0F;
                        int ls_high = (blk->scales_h >> (2*ib)) & 0x03;
                        int ls = ls_low | (ls_high << 4);
                        float dl = d * (float)(ls - 32);
                        a_frag[ele] = (_Float16)(dl * (float)kvalues_iq4nl[q]);
                    }
                } else {
                    a_frag[ele] = (_Float16)0.0f;
                }
            }

            v16f16 b_frag;
            #pragma unroll
            for (int ele = 0; ele < 16; ++ele) {
                if (tile_n < N && tile_m < M) {
                    int lds_row = k_sub + ele;
                    int lds_col = warp_m * 16 + lane;
                    b_frag[ele] = sB[buf][lds_row][lds_col];
                } else b_frag[ele] = (_Float16)0.0f;
            }

            c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
        }
        __syncthreads();
        buf ^= 1;
    }

    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        int r = ele * 2 + half_wave;
        int64_t out_n = (int64_t)tile_n + r;
        int64_t out_m = (int64_t)tile_m + lane;
        if (out_n < N && out_m < M) Y[out_n * M + out_m] = c_frag[ele];
    }
}

hipError_t gemm_iq4xs_lut_gpu(
    const block_iq4_xs* d_W,
    const _Float16* d_LUT,
    const float* d_X,
    float* d_Y,
    int64_t K, int64_t N, int64_t M,
    hipStream_t s = 0)
{
    if (!d_W || !d_X || !d_Y) return hipErrorInvalidValue;
    if (K%QK_K!=0) return hipErrorInvalidValue;
    bool wmma_ok = (M%16==0)&&(N%16==0)&&(K%16==0)&&(M>=512)&&(N>=32)&&(K>=32);
    if (wmma_ok) {
        dim3 grid((N+64-1)/64, (M+32-1)/32);
        hipLaunchKernelGGL(gemm_iq4xs_lut_kernel, grid, dim3(256), 0, s, d_W, d_LUT, d_X, d_Y, K,N,M);
        hipError_t err = hipGetLastError();
        if (err==hipSuccess) return err;
    }
    // fallback to tiled not implemented here; caller should fallback to wmma_stream tiled path
    return hipErrorInvalidConfiguration;
}
