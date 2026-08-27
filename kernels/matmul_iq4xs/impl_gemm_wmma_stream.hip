// impl_gemm_wmma_stream.hip — Streaming RDNA3 WMMA GEMM for prefill (M >= 128) — Phase 07-03
// Goal: Beat stock MMQ GEMM via hardware WMMA matrix cores at 1024 ops/CU/clock.
// Architecture: 2D tiled 64x32 output per 256-thread block (4x2 warps of 16x16 WMMA tiles)
//               Padded double-buffered LDS [2][32][33] for B tiles (float16), bank-conflict-free
//               On-the-fly IQ4_XS dequant into v16f16 a_frag, B staged via LDS -> v16f16 b_frag
//               __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 per K-tile, v8f32 accumulator.
// Fallback: TILE_M=16 tiled path when WMMA not applicable (M<512 or N/K not aligned, K%16!=0).
// Guardrails: __launch_bounds__(256,4) + amdgpu_flat_work_group_size(256,256) on every kernel (<=64 VGPRs, 16 waves/SIMD)
//             LDS stride 33 prevents 32-way Wave32 bank conflicts. Keep GGML_CUDA_ENABLE_CUSTOM_GFX1100 switch intact (standalone file).
// Dequant matches upstream IQ4_XS layout: d (fp16), scales_h (16b), scales_l[4] (4B), qs[128] (128B) -> f16 via dl*(kvalues-32)

#include "block_iq4_xs.h"
#include "hip_helpers.h"

typedef float v8f32 __attribute__((ext_vector_type(8)));
typedef _Float16 v16f16 __attribute__((ext_vector_type(16)));

// ---------------------------------------------------------------------------
// Fallback tiled GEMM — weight-reuse TILE_M=16 path (identical math to Phase06 but with guardrails)
// Each thread computes TILE_M columns for one N row — 16x reuse of dequantized weight traffic.
// Guardrail: __launch_bounds__(256,4) + amdgpu_flat_work_group_size(256,256)
// ---------------------------------------------------------------------------
template<int TILE_M = 16>
__global__ __launch_bounds__(256, 4) __attribute__((amdgpu_flat_work_group_size(256,256)))
void gemm_iq4xs_stream_tiled_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ X, // [K, M] row-major: X[k*M + m]
    float* __restrict__ Y,       // [N, M] row-major
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

    int64_t blocks_per_row = K / QK_K;
    const block_iq4_xs* row_blocks = W + n * blocks_per_row;

    double acc[16] = {0.0};

    for (int64_t b = 0; b < blocks_per_row; ++b) {
        const block_iq4_xs* blk = &row_blocks[b];
        float d = fp16_to_fp32(blk->d);
        for (int ib = 0; ib < 8; ++ib) {
            int ls_low  = (blk->scales_l[ib/2] >> (4*(ib&1))) & 0x0F;
            int ls_high = (blk->scales_h >> (2*ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            double dl = static_cast<double>(d) * static_cast<double>(static_cast<float>(ls) - 32.0f);

            // Aligned 8-byte loads for qs sub-block (8-byte aligned pair, satisfies 8B alignment guard)
            struct Aligned16 { uint64_t lo; uint64_t hi; };
            Aligned16 qs_val;
            qs_val.lo = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16);
            qs_val.hi = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16 + 8);
            const uint8_t* qs_bytes = reinterpret_cast<const uint8_t*>(&qs_val);
            int64_t k_base = b * QK_K + ib * 32;

            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                uint8_t q_byte = qs_bytes[j];
                int lo = q_byte & 0x0F;
                int hi = (q_byte >> 4) & 0x0F;
                double w_lo = dl * static_cast<double>(kvalues_iq4nl[lo]);
                double w_hi = dl * static_cast<double>(kvalues_iq4nl[hi]);

                #pragma unroll
                for (int tm = 0; tm < 16; ++tm) {
                    if (tm >= tile_m) break;
                    int64_t m = m0 + tm;
                    double x_lo = static_cast<double>(X[(k_base + j) * M + m]);
                    double x_hi = static_cast<double>(X[(k_base + j + 16) * M + m]);
                    acc[tm] += w_lo * x_lo;
                    acc[tm] += w_hi * x_hi;
                }
            }
        }
    }
    #pragma unroll
    for (int tm = 0; tm < 16; ++tm) {
        if (tm >= tile_m) break;
        Y[n * M + m0 + tm] = static_cast<float>(acc[tm]);
    }
}

// ---------------------------------------------------------------------------
// Streaming WMMA kernel — 64x32 per block, double-buffered LDS [2][32][33] (_Float16)
// Each warp (32 threads Wave32) computes 16x16 output tile via 16x16x16 wmma.
// Block 256 threads = 8 warps: 4 along N (64/16) x 2 along M (32/16).
// B tiles are streamed through padded double-buffered LDS: B[2][32][33] (K_TILE=32, BLOCK_M=32, pad 33)
//   — cooperatively loaded from global X (float->half), then each warp loads b_frag from LDS.
// A tiles are on-the-fly dequantized directly into v16f16 a_frag registers (no LDS).
// Per K_TILE=32 iteration we issue 2 WMMA instructions (sub=0: k+0..15, sub=1: k+16..31).
// Correct lane/half_wave mapping: lane = lIdx%16, half_wave = lIdx/16, C[ele*2+half_wave, lane].
// Hardware path: 1024 ops/CU/clock via v_wmma_f32_16x16x16_f16_w32.
// VGPR target <=64 via __launch_bounds__(256,4) -> 16 waves/SIMD on gfx1100.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4) __attribute__((amdgpu_flat_work_group_size(256,256)))
void gemm_iq4xs_wmma_stream_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ X, // [K, M]
    float* __restrict__ Y,       // [N, M]
    int64_t K,
    int64_t N,
    int64_t M
) {
    const int BLOCK_N = 64;
    const int BLOCK_M = 32;
    const int K_TILE = 32; // LDS tile K dimension; 2 WMMA steps per tile

    int block_n = blockIdx.x * BLOCK_N;
    int block_m = blockIdx.y * BLOCK_M;

    int tid = threadIdx.x; // 0..255
    int lIdx = tid % 32;
    int warp_id = tid / 32; // 0..7
    int lane = lIdx % 16;
    int half_wave = lIdx / 16;

    int warp_n = warp_id / 2; // 0..3
    int warp_m = warp_id % 2; // 0..1
    int tile_n = block_n + warp_n * 16;
    int tile_m = block_m + warp_m * 16;

    // Padded double-buffered LDS for B tiles: [2][32][33] half
    // Stride 33 eliminates 32-way Wave32 bank conflicts (33*2B=66B per row, banks rotate).
    // Double-buffered ping-pong: buf ^1 while computing, but we sync per tile for correctness.
    __shared__ _Float16 sB[2][32][33];

    // Early exit: if this warp's tile is fully out-of-bounds, still need to participate in LDS loads and barriers.
    // So we don't return early before the loop — we predicate only computes and stores, but participate in cooperative loads.
    v8f32 c_frag = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    int64_t blocks_per_row = K / QK_K;
    int num_k_tiles = (int)(K / K_TILE); // K is multiple of 256 -> divisible by 32

    int buf = 0;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k0 = kt * K_TILE;

        // Cooperative load of B tile [K_TILE=32][BLOCK_M=32] from global X into sB[buf]
        // Total elements 32*32=1024, 256 threads => 4 elements per thread.
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            int linear = tid + iter * 256; // 0..1023
            int kr = linear / 32; // 0..31 K offset
            int cm = linear % 32; // 0..31 M offset
            int64_t gk = (int64_t)k0 + kr;
            int64_t gm = (int64_t)block_m + cm;
            _Float16 v = (_Float16)0.0f;
            if (gk < K && gm < M) {
                v = (_Float16)X[gk * M + gm];
            }
            sB[buf][kr][cm] = v;
        }
        __syncthreads();

        // Two WMMA steps per K_TILE (16 each)
        #pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            int k_sub = sub * 16;
            int k_base = k0 + k_sub;

            // Build A fragment on-the-fly: A[lane, ele] = dequant(W[tile_n+lane, k_base+ele])
            v16f16 a_frag;
            #pragma unroll
            for (int ele = 0; ele < 16; ++ele) {
                int64_t g_n = (int64_t)tile_n + lane;
                int64_t g_k = (int64_t)k_base + ele;
                if (g_n < N && g_k < K) {
                    int64_t b = g_k / QK_K;
                    int ib = (int)((g_k % QK_K) / 32);
                    int j = (int)((g_k % 32) % 16);
                    bool hi_nibble = (g_k % 32) >= 16;
                    const block_iq4_xs* blk = W + g_n * blocks_per_row + b;
                    float d = fp16_to_fp32(blk->d);
                    int ls_low  = (blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
                    int ls_high = (blk->scales_h >> (2 * ib)) & 0x03;
                    int ls = ls_low | (ls_high << 4);
                    float dl = d * (static_cast<float>(ls) - 32.0f);
                    uint8_t q_byte = blk->qs[ib * 16 + j];
                    int q = hi_nibble ? ((q_byte >> 4) & 0x0F) : (q_byte & 0x0F);
                    float w = dl * static_cast<float>(kvalues_iq4nl[q]);
                    a_frag[ele] = (_Float16)w;
                } else {
                    a_frag[ele] = (_Float16)0.0f;
                }
            }

            // Build B fragment from LDS: B[ele, lane] = sB[buf][k_sub+ele][warp_m*16+lane]
            v16f16 b_frag;
            #pragma unroll
            for (int ele = 0; ele < 16; ++ele) {
                // Guard: tile_n/m bounds already predicate store; for correctness we zero out-of-bounds fragments
                if (tile_n < N && tile_m < M) {
                    int lds_row = k_sub + ele;
                    int lds_col = warp_m * 16 + lane;
                    b_frag[ele] = sB[buf][lds_row][lds_col];
                } else {
                    b_frag[ele] = (_Float16)0.0f;
                }
            }

            // Issue hardware WMMA: 16x16x16 FP16->FP32, Wave32
            // Only warp-level active threads should contribute; but all lanes participate in wmma builtin.
            // Predicate NaN: if warp tile out-of-bounds, still need to issue wmma to keep all lanes uniform? We gate accum.
            // We issue for all warps uniformly to avoid divergence; out-of-bounds warps still execute but their C will be discarded on store.
            c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
        }

        __syncthreads();
        buf ^= 1;
    }

    // Store output: C[ele*2+half_wave, lane] per RDNA3 Wave32 layout -> Y[(tile_n+r)*M + (tile_m+lane)]
    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        int r = ele * 2 + half_wave;
        int64_t out_n = (int64_t)tile_n + r;
        int64_t out_m = (int64_t)tile_m + lane;
        if (out_n < N && out_m < M) {
            Y[out_n * M + out_m] = c_frag[ele];
        }
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
static inline hipError_t launch_stream_tiled_gemm(
    const block_iq4_xs* d_W,
    const float* d_X,
    float* d_Y,
    int64_t K,
    int64_t N,
    int64_t M,
    hipStream_t stream
) {
    const int TILE_M = 16;
    int64_t tiles_per_row = (M + TILE_M - 1) / TILE_M;
    int64_t total_tiles = N * tiles_per_row;
    const int BLOCK = 256;
    int64_t grid = (total_tiles + BLOCK - 1) / BLOCK;
    if (grid > 2147483647) return hipErrorInvalidConfiguration;
    hipLaunchKernelGGL((gemm_iq4xs_stream_tiled_kernel<16>), dim3((unsigned)grid), dim3(BLOCK), 0, stream, d_W, d_X, d_Y, K, N, M);
    return hipGetLastError();
}

hipError_t gemm_iq4xs_wmma_stream_gpu(
    const block_iq4_xs* d_W,
    const float* d_X,
    float* d_Y,
    int64_t K,
    int64_t N,
    int64_t M,
    hipStream_t stream = 0
) {
    if (K <= 0 || N <= 0 || M <= 0 || K % QK_K != 0) return hipErrorInvalidValue;
    // WMMA path requires: M >= 512, N >= 32, K >= 32, and 16-alignment for hardware fragments.
    // M threshold matches task fallback spec (M<512 -> tiled). N/K alignment mirrors existing gate but slightly relaxed to 16.
    bool wmma_ok = (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0) && (M >= 512) && (N >= 32) && (K >= 32);
    // Additionally require BLOCK-level alignment for double-buffered LDS path efficiency: we handle remainder via predication,
    // so we keep gate as above. For full 64x32 block coverage when N/M not multiples of 64/32, tail warps predicate correctly.
    if (wmma_ok) {
        const int BLOCK_N = 64;
        const int BLOCK_M = 32;
        int grid_x = (int)((N + BLOCK_N - 1) / BLOCK_N);
        int grid_y = (int)((M + BLOCK_M - 1) / BLOCK_M);
        dim3 grid(grid_x, grid_y);
        dim3 block(256);
        hipLaunchKernelGGL(gemm_iq4xs_wmma_stream_kernel, grid, block, 0, stream, d_W, d_X, d_Y, K, N, M);
        hipError_t err = hipGetLastError();
        if (err == hipSuccess) return err;
        // Fall through to tiled on launch error
    }
    return launch_stream_tiled_gemm(d_W, d_X, d_Y, K, N, M, stream);
}

// Explicit tiled launch for benchmarking (bypass WMMA)
hipError_t gemm_iq4xs_stream_tiled_gpu(
    const block_iq4_xs* d_W,
    const float* d_X,
    float* d_Y,
    int64_t K,
    int64_t N,
    int64_t M,
    hipStream_t stream = 0
) {
    if (K <= 0 || N <= 0 || M <= 0 || K % QK_K != 0) return hipErrorInvalidValue;
    return launch_stream_tiled_gemm(d_W, d_X, d_Y, K, N, M, stream);
}
