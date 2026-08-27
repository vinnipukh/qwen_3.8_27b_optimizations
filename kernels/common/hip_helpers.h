#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define HIP_CHECK(ans) do { \
    hipError_t err = (ans); \
    if (err != hipSuccess) { \
        fprintf(stderr, "[HIP ERROR] %s:%d: %s (code %d)\n", __FILE__, __LINE__, hipGetErrorString(err), (int)err); \
        std::abort(); \
    } \
} while (0)

#define HIP_EVENT_CHECK(ans) HIP_CHECK(ans)

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Wavefront lane mask type depending on wave size
template<int W>
struct wave_mask;

template<>
struct wave_mask<32> {
    using type = uint32_t;
    static constexpr uint32_t all_active = 0xFFFFFFFFu;
};

template<>
struct wave_mask<64> {
    using type = uint64_t;
    static constexpr uint64_t all_active = 0xFFFFFFFFFFFFFFFFull;
};

template<int W>
using wave_mask_t = typename wave_mask<W>::type;
