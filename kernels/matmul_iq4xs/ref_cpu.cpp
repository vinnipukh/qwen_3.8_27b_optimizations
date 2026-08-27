// ref_cpu.cpp — Pure C++17 FP64 golden oracle for IQ4_XS GEMV/GEMM
// Zero HIP dependencies; compile with -std=c++17.
// Accuracy: double accumulators; fp16 scale via fp16_to_fp32() -> float -> double multiply.

#include "ref_cpu.h"
#include <cmath>
#include <cstring>

// Internal: decode one full row's 256*blocks_per_row weights into temporary buffer
// Used for tiling variants. But main GEMV loops directly dequantize per subblock to avoid large temps.

void gemv_iq4xs_cpu_ref(
    const block_iq4_xs* W,
    const float* x,
    float* y,
    int64_t K,
    int64_t N
) {
    const int64_t blocks_per_row = K / QK_K;
    for (int64_t r = 0; r < N; ++r) {
        double row_sum = 0.0;
        const block_iq4_xs* row_blocks = W + r * blocks_per_row;
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            float d = fp16_to_fp32(row_blocks[b].d);
            const uint8_t* qs = row_blocks[b].qs;
            const float* x_block = x + b * QK_K;

            // 8 sub-blocks of 32 weights each (16 bytes qs per sub-block)
            for (int ib = 0; ib < 8; ++ib) {
                int ls_low  = (row_blocks[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
                int ls_high = (row_blocks[b].scales_h >> (2 * ib)) & 0x03;
                int ls = ls_low | (ls_high << 4);
                double dl = static_cast<double>(d) * static_cast<double>(static_cast<float>(ls) - 32.0f);

                // qs layout: low nibble -> weight at offset j, high nibble -> weight at j+16
                // Each sub-block's 16 bytes hold 32 nibbles = 32 weights.
                const uint8_t* qs_sub = qs + ib * 16;
                const float* x_sub = x_block + ib * 32;
                for (int j = 0; j < 16; ++j) {
                    uint8_t q_byte = qs_sub[j];
                    int lo = q_byte & 0x0F;
                    int hi = (q_byte >> 4) & 0x0F;
                    double w_lo = dl * static_cast<double>(kvalues_iq4nl[lo]);
                    double w_hi = dl * static_cast<double>(kvalues_iq4nl[hi]);
                    row_sum += w_lo * static_cast<double>(x_sub[j]);
                    row_sum += w_hi * static_cast<double>(x_sub[j + 16]);
                }
            }
        }
        y[r] = static_cast<float>(row_sum);
    }
}

void gemm_iq4xs_cpu_ref(
    const block_iq4_xs* W,
    const float* X,
    float* Y,
    int64_t K,
    int64_t N,
    int64_t M
) {
    // X: [K, M] row-major => X[k*M + m]
    // Y: [N, M] row-major => Y[n*M + m] = sum_k W[n,k] * X[k,m]
    const int64_t blocks_per_row = K / QK_K;
    // For cache friendliness, we iterate over M outermost? But for golden reference correctness
    // we do naive triple loop with double accumulation per output element.
    // Optimized version still bit-exact because addition order differs slightly;
    // but double accumulation makes ordering effect <1e-12 relative.

    // Zero initialize
    for (int64_t i = 0; i < N * M; ++i) Y[i] = 0.0f;

    // We iterate per output column m to improve X access pattern
    for (int64_t r = 0; r < N; ++r) {
        const block_iq4_xs* row_blocks = W + r * blocks_per_row;
        for (int64_t m = 0; m < M; ++m) {
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
                    // X base for this K block and sub-block
                    // K offset = b*256 + ib*32
                    int64_t k_base = b * QK_K + ib * 32;
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
            Y[r * M + m] = static_cast<float>(acc);
        }
    }
}

void dequant_mat_iq4xs_cpu(
    const block_iq4_xs* W,
    float* out,
    int64_t K,
    int64_t N
) {
    const int64_t blocks_per_row = K / QK_K;
    for (int64_t r = 0; r < N; ++r) {
        const block_iq4_xs* row_blocks = W + r * blocks_per_row;
        float* out_row = out + r * K;
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            float d = fp16_to_fp32(row_blocks[b].d);
            const uint8_t* qs = row_blocks[b].qs;
            float* y = out_row + b * QK_K;
            for (int ib = 0; ib < 8; ++ib) {
                int ls_low  = (row_blocks[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
                int ls_high = (row_blocks[b].scales_h >> (2 * ib)) & 0x03;
                int ls = ls_low | (ls_high << 4);
                float dl = d * (static_cast<float>(ls) - 32.0f);
                const uint8_t* qs_sub = qs + ib * 16;
                float* y_sub = y + ib * 32;
                for (int j = 0; j < 16; ++j) {
                    uint8_t q_byte = qs_sub[j];
                    int lo = q_byte & 0x0F;
                    int hi = (q_byte >> 4) & 0x0F;
                    y_sub[j]      = dl * static_cast<float>(kvalues_iq4nl[lo]);
                    y_sub[j + 16] = dl * static_cast<float>(kvalues_iq4nl[hi]);
                }
            }
        }
    }
}
