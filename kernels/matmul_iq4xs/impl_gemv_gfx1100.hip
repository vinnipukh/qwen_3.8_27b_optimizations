// impl_gemv_gfx1100.hip — Custom gfx1100 GEMV kernel for decode (M=1)
// Target: Beat stock HIP via 128-bit vector loads, zero-LDS collaborative dequant,
//         and Wave32 cooperative reduction. Wave32 exclusive, launch_bounds 256,4.
// Design: 8 threads cooperatively process one super-block (256 weights) = 1 output row slice.
//         Each Wave32 (32 threads) handles 4 super-blocks; 256-thread block = 32 rows per block (amortized).
// Performance goal: Saturate ~800 GB/s VRAM.

#include "block_iq4_xs.h"
#include "hip_helpers.h"

template<int WarpSize = WARP_SIZE>
__global__ __launch_bounds__(256, 4) __attribute__((amdgpu_flat_work_group_size(256,256)))
void gemv_iq4xs_gfx1100_kernel(
    const block_iq4_xs* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t K,
    int64_t N
) {
    // Cooperative 8-thread per row decomposition
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ROWS_PER_BLOCK = 256 / THREADS_PER_ROW; // 32

    int tid = threadIdx.x;
    int group_id = tid / THREADS_PER_ROW;      // 0..31 row within block
    int lane_in_group = tid & (THREADS_PER_ROW - 1); // 0..7 sub-block index

    int64_t row = (int64_t)blockIdx.x * ROWS_PER_BLOCK + group_id;

    // Per-thread partial sum for its assigned sub-blocks across all super-blocks
    double thread_sum = 0.0;

    if (row < N) {
        int64_t blocks_per_row = K / QK_K;
        const block_iq4_xs* row_blocks = W + row * blocks_per_row;

        // Loop over super-blocks (each 256 weights = 136 bytes)
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            const block_iq4_xs* blk = &row_blocks[b];
            float d = fp16_to_fp32(blk->d);

            // Each lane handles exactly one sub-block ib = lane_in_group within this super-block
            int ib = lane_in_group;
            int ls_low  = (blk->scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            int ls_high = (blk->scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            double dl = static_cast<double>(d) * static_cast<double>(static_cast<float>(ls) - 32.0f);

            // Aligned 8-byte loads for qs sub-block (16 bytes = 2x uint64_t)
            // Offset in block is 8 + ib * 16 (always 8-byte aligned)
            struct Aligned16 { uint64_t lo; uint64_t hi; };
            Aligned16 qs_val;
            qs_val.lo = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16);
            qs_val.hi = *reinterpret_cast<const uint64_t*>(blk->qs + ib * 16 + 8);
            const uint8_t* qs_bytes = reinterpret_cast<const uint8_t*>(&qs_val);

            // Load activation slice: 32 floats = ib*32 offset within super-block
            const float* x_base = x + b * QK_K + ib * 32;

            // Unrolled MAC over 16 q bytes => 32 weights
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                uint8_t q_byte = qs_bytes[j];
                int lo = q_byte & 0x0F;
                int hi = (q_byte >> 4) & 0x0F;

                double w_lo = dl * static_cast<double>(kvalues_iq4nl[lo]);
                double w_hi = dl * static_cast<double>(kvalues_iq4nl[hi]);

                float x_lo = x_base[j];
                float x_hi = x_base[j + 16];
                thread_sum += w_lo * static_cast<double>(x_lo);
                thread_sum += w_hi * static_cast<double>(x_hi);
            }
        }
    }

    // Reduction across 8 threads in group using shared memory
    // Uniform barrier participation: all 256 threads execute __syncthreads()
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

// Alternate entry with Wave64 template (emulated 2x32)
template __global__ void gemv_iq4xs_gfx1100_kernel<32>(const block_iq4_xs*, const float*, float*, int64_t, int64_t);
template __global__ void gemv_iq4xs_gfx1100_kernel<64>(const block_iq4_xs*, const float*, float*, int64_t, int64_t);

// Host launcher
hipError_t gemv_iq4xs_gfx1100_gpu(
    const block_iq4_xs* d_W,
    const float* d_x,
    float* d_y,
    int64_t K,
    int64_t N,
    hipStream_t stream = 0
) {
    if (K <= 0 || N <= 0 || K % QK_K != 0) {
        return hipErrorInvalidValue;
    }
    constexpr int BLOCK = 256;
    constexpr int THREADS_PER_ROW = 8;
    constexpr int ROWS_PER_BLOCK = BLOCK / THREADS_PER_ROW;
    int64_t grid = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (grid > 2147483647) {
        return hipErrorInvalidConfiguration;
    }
    // Wave32 path only — gfx1100 native
    hipLaunchKernelGGL((gemv_iq4xs_gfx1100_kernel<32>), dim3((unsigned)grid), dim3(BLOCK), 0, stream, d_W, d_x, d_y, K, N);
    return hipGetLastError();
}
