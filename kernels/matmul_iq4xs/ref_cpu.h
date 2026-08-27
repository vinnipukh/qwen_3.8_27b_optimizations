#pragma once
// ref_cpu.h — FP64 golden oracle for IQ4_XS GEMV/GEMM
// Pure C++17, no HIP dependencies.

#include "block_iq4_xs.h"
#include <cstdint>

struct MatmulShape {
    const char* name;
    int64_t K;  // in-features (columns of W)
    int64_t N;  // out-features (rows of W)
};

// 8 canonical Qwen3.8-27B projection shapes (K = columns, N = rows of W)
static const MatmulShape CANONICAL_SHAPES[] = {
    {"attn_q",    5120,  5120},
    {"attn_k",    5120,  5120},
    {"attn_v",    5120,  5120},
    {"attn_gate", 5120,  6144},
    {"attn_out",  5120,  5120},
    {"ffn_gate",  5120, 17408},
    {"ffn_up",    5120, 17408},
    {"ffn_down", 17408,  5120},
};
static constexpr int NUM_CANONICAL_SHAPES = sizeof(CANONICAL_SHAPES) / sizeof(CANONICAL_SHAPES[0]);

// GEMV: y = W * x, W:[N,K] quantized IQ4_XS, x:[K] f32, y:[N] f32
// W layout: N rows, each row has K/QK_K blocks (136 B each)
void gemv_iq4xs_cpu_ref(
    const block_iq4_xs* W,
    const float* x,
    float* y,
    int64_t K,
    int64_t N
);

// GEMM: Y = W * X, W:[N,K], X:[K,M], Y:[N,M] all row-major contiguous
// Y[n*M + m] = sum_k W[n,K,k] * X[k*M + m]
// For IQ4_XS, W is quantized, X/Y are f32.
void gemm_iq4xs_cpu_ref(
    const block_iq4_xs* W,
    const float* X,
    float* Y,
    int64_t K,
    int64_t N,
    int64_t M
);

// Utility: dequantize entire matrix W (N x K) to f32 for debugging/validation
void dequant_mat_iq4xs_cpu(
    const block_iq4_xs* W,
    float* f32_out,
    int64_t K,
    int64_t N
);
