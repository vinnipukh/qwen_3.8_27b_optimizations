#include "block_iq4_xs.h"
#include <cstdint>

// Pure C++17 stub reference implementation for the template quartet.
// Unpacks a dummy pattern or zero/scale into float destination.
void template_ref_cpu(const block_iq4_xs* src, float* dst, int64_t n_blocks) {
    for (int64_t b = 0; b < n_blocks; ++b) {
        float d = fp16_to_fp32(src[b].d);
        for (int i = 0; i < QK_K; ++i) {
            dst[b * QK_K + i] = d * static_cast<float>(i % 16);
        }
    }
}
