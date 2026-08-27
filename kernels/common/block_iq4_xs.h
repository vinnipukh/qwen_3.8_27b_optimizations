// Vendored from ggml/src/ggml-common.h @ bb4caa75 — Apache-2.0
// Zero external llama.cpp / ggml header dependencies.
#pragma once

#include <stdint.h>
#include <cstring>

#define QK_K 256

#pragma pack(push, 1)
typedef struct {
    uint16_t d;           // fp16 scale (stored as uint16_t, little-endian)
    uint16_t scales_h;    // 2-bit high scale bits for 8 sub-blocks (2 bits * 8 = 16 bits)
    uint8_t  scales_l[4]; // 4-bit low scale bits for 8 sub-blocks (2 sub-blocks per byte = 4 bytes)
    uint8_t  qs[128];     // 4-bit quant indices for 256 values (128 bytes)
} block_iq4_xs;
#pragma pack(pop)

static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be exactly 136 bytes");

// Non-linear 4-bit codebook table (kvalues_iq4nl[16])
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Pure C++17 software fp16 (IEEE 754 half-precision) to fp32 (single-precision) converter
// Works on both host and device without requiring external fp16 headers.
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

HOST_DEVICE static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // subnormal
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // inf/nan
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        // normalized
        f = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
    }
#if defined(__HIP_DEVICE_COMPILE__)
    union { uint32_t u; float flt; } pun;
    pun.u = f;
    return pun.flt;
#else
    float res;
    std::memcpy(&res, &f, sizeof(res));
    return res;
#endif
}

// Helper: Convert float to uint16_t fp16 (for fixture generators / test mocks)
static inline uint16_t fp32_to_fp16(float val) {
    uint32_t x;
    std::memcpy(&x, &val, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x400) >> (1 - exp);
        return (uint16_t)(sign | mant);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (exp << 10) | mant);
}
