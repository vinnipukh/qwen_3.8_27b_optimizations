// ref_cpu.cpp — Pure C++17 CPU oracle for block_iq4_xs dequantization.
// Zero external llama.cpp / ggml header dependencies.
#include "block_iq4_xs.h"
#include <cstdint>

void dequantize_row_iq4_xs_cpu(const block_iq4_xs* src, float* dst, int64_t n_blocks) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        float d = fp16_to_fp32(src[b].d);
        const uint8_t* qs = src[b].qs;
        float* y = dst + b * QK_K;

        for (int ib = 0; ib < 8; ++ib) {
            int ls_low  = (src[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
            int ls_high = (src[b].scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            float dl = d * (static_cast<float>(ls) - 32.0f);

            for (int j = 0; j < 16; ++j) {
                int lo = qs[j] & 0x0F;
                int hi = (qs[j] >> 4) & 0x0F;
                y[j + 0]  = dl * static_cast<float>(kvalues_iq4nl[lo]);
                y[j + 16] = dl * static_cast<float>(kvalues_iq4nl[hi]);
            }
            y += 32;
            qs += 16;
        }
    }
}
