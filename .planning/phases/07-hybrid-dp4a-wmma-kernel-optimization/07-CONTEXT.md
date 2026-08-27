# Phase 7: Hybrid DP4A & Matrix Core Optimization — Context

**Date:** 2026-08-25
**Goal:** Outperform real production `llama.cpp` stock kernels on AMD Radeon RX 7900 XT (`gfx1100`) end-to-end by fusing `Q8_1` integer activation quantization and RDNA3 hardware matrix cores (`v_dot4_i32_i8` / `v_wmma`) with our Wave32 cooperative workgroup architecture.

## Background & Post-Mortem Findings

In Phases 4–6, custom kernels were benchmarked and proven against `stock_hip_comparator.hip` (a naive scalar float reference loop) achieving 2.13× GEMV and 9.27× GEMM microbenchmark speedups. However, real production `llama.cpp` uses:
1. **On-the-fly `Q8_1` Quantization:** Converts float activations to 8-bit integers, cutting activation memory traffic by $4\times$.
2. **Hardware DP4A SIMD (`v_dot4_i32_i8`):** Computes 4-way INT8 dot products in a single clock cycle.
3. **Multi-Warp Register Blocking in MMQ:** 4-way unrolled DP4A matrix multiplication.

Our initial Phase 5/6 kernel operated directly on unquantized 32-bit floats with double-precision accumulation to guarantee numerical parity (`cosine = 1.0`), which incurred a $4\times$ memory bandwidth penalty and lower arithmetic density than stock DP4A.

## Architectural Opportunities on gfx1100

1. **Decode Opportunity (GEMV $M=1$):**
   - Stock `mmvq.cu` restricts RDNA3 (`MMVQ_PARAMETERS_RDNA3_0`) to a single warp per row for IQ4_XS (`calc_nwarps` returns 1).
   - Our 8-thread cooperative decomposition (256 threads $\to$ 32 output rows per block) combined with DP4A instructions (`__dp4a` / `v_dot4_i32_i8`) can achieve higher occupancy and full memory coalescing over stock.
   - Expected Target: **>40 t/s** (vs Stock 34.8 t/s).

2. **Prefill Opportunity (GEMM $M \ge 128$):**
   - Stock `mmq.cu` runs integer DP4A on general shader ALUs (peak 512 ops/CU/clock).
   - RDNA3 Wave32 hardware WMMA matrix cores (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` or `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`) execute matrix multiplication directly in tensor registers at **1024 ops/CU/clock** ($2\times$ higher compute density).
   - Expected Target: **>1000 t/s** (vs Stock 880 t/s).

## Plans in Phase 7

- **07-01:** Real Upstream Microbenchmark Comparator (Wire true `vec_dot_iq4_xs_q8_1` and `quantize_row_q8_1` into `kernels/matmul_iq4xs/`).
- **07-02:** Custom Cooperative Wave32 DP4A GEMV Kernel (`impl_gemv_dp4a_gfx1100.hip`).
- **07-03:** Custom Hardware WMMA Matrix Core GEMM Kernel (`impl_gemm_wmma_q8.hip`).
- **07-04:** Quilt Patch Integration & Paired End-to-End A/B Benchmark Verification (`llama-bench`).
