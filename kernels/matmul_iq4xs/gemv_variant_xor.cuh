// gemv_variant_xor.cuh — XOR preshuffle helper for LDS banking (0% overhead alternative to +33 padding)
// CK TileWindow pattern: x' = (y % (KPerBlock/KPack)) ^ x prevents 32-way bank conflict without padding waste
// Used by impl_gemv_dp4a_gfx1100.hip Variant B vs Variant A sh[32][33] +33 padded (+3% via CK Tile +33)
// See output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md LDS 32x4B 8-phase ds_write_b128 lane0~7...56~63
// REQ-PERF-07 high-yield variant: XOR vs +33 race determines 1.10x gate winner
#pragma once

// XOR preshuffle for GEMV cooperative LDS [32][33] vs XOR [32][32]
// y = row index within block (0..31), x = lane/col (0..7 for GEMV 8-thread coop, but general)
// KPerBlock=32 (rows), KPack=8 (threads per row) => y % (32/8)= y%4 gives 2-bit hash to xor with x
// For GEMV: sh_xor[y][xor_preshuffle(y,x)] achieves 0% overhead banking vs [32][33] +3%

__device__ __forceinline__ int xor_preshuffle_32x33(int y, int x) {
    // GEMV_XOR variant: 0% padding overhead, CK Tile lds_bank_conflicts.html
    // y in [0,32), x in [0,32) but for GEMV x in [0,8)
    return (y % (32 / 8)) ^ x;
}

__device__ __forceinline__ int xor_preshuffle_32x32(int y, int x) {
    // General GEMM variant: template<xor_KPerBlock=64, KPack=8> idx = (row % (KPerBlock/KPack)) ^ col
    // For 64x32 GEMM tile: y%8 ^ x
    return (y % 8) ^ x;
}

// Example usage for Variant B sh_xor[32][32]:
// __shared__ float sh_xor[32][32];
// int x_xor = xor_preshuffle_32x33(group_id, lane);
// sh_xor[group_id][x_xor] = local; __syncthreads(); if(lane==0) acc = sh_xor[group_id][0]+... (need to xor again on read)
// Note: read must xor same: sh_xor[group_id][xor_preshuffle(group_id, t)]

// Compile-time switch: define GEMV_XOR to enable XOR path in impl_gemv_dp4a_gfx1100.hip, else +33 padding
#ifdef GEMV_XOR
#define GEMV_LDS_VARIANT "XOR preshuffle x'=(y%(32/8))^x 0% overhead"
#else
#define GEMV_LDS_VARIANT "+33 padded sh[32][33] +3% overhead"
#endif

// VGPR note: XOR adds 1 VGPR (xor calc) but saves LDS 32 floats (128B) and +33 padding overhead; tradeoff gated via race.py --repeats 10
