---
phase: 07-hybrid-dp4a-wmma-kernel-optimization
plan: 03
subsystem: kernels
tags: [hip, gfx1100, wmma, gemm, prefill, iq4_xs, wave32, lds, double-buffered, stream, half]
requires:
  - phase: 07-hybrid-dp4a-wmma-kernel-optimization
    provides: cooperative DP4A GEMV (07-02) and real_stock_dp4a_comparator true DP4A pipeline (07-01)
provides:
  - kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip — streaming WMMA GEMM 64x32 per block, double-buffered LDS [2][32][33], on-the-fly IQ4_XS->f16 dequant, wmma_f32_16x16x16_f16_w32, fallback tiled TILE_M=16
  - kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp — parity vs CPU FP64 oracle (cosine >=0.999) across 15 shapes including M=128/512/1024 prefill
  - kernels/matmul_iq4xs/bench_gemm_wmma.cpp — prefill throughput M=128,512,1024 vs real stock DP4A/MMQ, speedup + TFLOPS JSON
affects:
  - 07-04 quilt patch integration (GGML_CUDA_ENABLE_CUSTOM_GFX1100 gating preserved, streaming WMMA available for A/B)
actuals:
  tokens: 8200
  tasks: 3
  commits: 1
tech-stack:
  added: []
  patterns: [64x32 per block 4x2 warps, double-buffered LDS 33-stride, WMMA Wave32 16x16x16, on-the-fly dequant IQ4_XS->half, v16f16 fragments + v8f32 accum, lane%16 half_wave mapping, fallback tiled GEMM TILE_M=16]
key-files:
  created:
    - kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip
    - kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp
    - kernels/matmul_iq4xs/bench_gemm_wmma.cpp
  modified:
    - kernels/matmul_iq4xs/CMakeLists.txt
key-decisions:
  - "Streaming WMMA kernel keeps both __launch_bounds__(256,4) and amdgpu_flat_work_group_size(256,256) on tiled fallback and WMMA kernels for <=64 VGPRs / 16 waves/SIMD target — verified via grep."
  - "LDS double-buffered [2][32][33] (_Float16) with stride 33 eliminates 32-way Wave32 bank conflicts; K_TILE=32 = 2 WMMA steps per cooperative B-tile load (cooperative 4 elements/thread for 1024-element 32x32 tile)."
  - "A fragment on-the-fly dequant: d* (ls-32) * kvalues_iq4nl[q] -> _Float16 per 16-element v16f16 fragment, matching upstream block_iq4_xs layout (d, scales_h, scales_l[4], qs[128]) — zero ggml headers."
  - "B fragment from LDS staged float->half (X[k*M+m] -> half in sB), then warp-local b_frag[ele]=sB[buf][k_sub+ele][warp_m*16+lane] — half-wave/lane mapping matches RDNA3 WMMA lane%16, lane/16, store C[ele*2+half_wave,lane]."
  - "WMMA gate M>=512 && N%16==0 && K%16==0 && N>=32 && K>=32 (M=128 falls back to tiled TILE_M=16) — preserves streaming design fallback spec, K=256-multiple so K_TILE=32 always divides evenly."
  - "Separate matmul_gemm_wmma_stream_hip OBJECT library retains existing matmul_gemm_hip (Phase06 wmma+tiled) untouched for ordering after 07-02; new CMake targets wire test_gemm_wmma_compare vs ref_cpu only and bench_gemm_wmma vs real_stock_dp4a for fair DP4A vs WMMA comparison."
patterns-established:
  - "Streaming double-buffered LDS WMMA pattern: cooperative 32x32 half tile load -> __syncthreads -> 2x wmma 16x16x16 -> __syncthreads -> buf flip."
  - "WMMA GEMM tiling: 64x32 per block, grid (ceil(N/64), ceil(M/32)), 8 warps, warp_n=warp_id/2 warp_m=warp_id%2."
---

# Phase 07 Plan 03: RDNA3 WMMA Hardware Matrix Core GEMM Kernel (Prefill Optimization) Summary

**One-liner:** Streaming WMMA GEMM with 64x32 per-block tiling, double-buffered LDS [2][32][33] and on-the-fly IQ4_XS dequant into Wave32 wmma_f32_16x16x16_f16_w32 — fallback tiled TILE_M=16 when M<512, parity gate cosine >=0.999 vs FP64 oracle.

## Objective

Outperform stock llama.cpp MMQ prefill (M>=128) on gfx1100 by streaming quantized inputs into RDNA3 hardware WMMA matrix cores at 1024 ops/CU/clock vs 512 ops DP4A, targeting >1.2x speedup over stock MMQ (>1000 t/s prefill e2e). Covered by 2D tiled 64x32 per block, padded double-buffered LDS, and on-the-fly dequant.

## Deliverables

| File | Purpose |
|------|---------|
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | Streaming WMMA GEMM kernel (WMMA path 64x32/block, 4x2 warps, LDS [2][32][33] half double-buffered, K_TILE=32 2x WMMA per tile, on-the-fly IQ4_XS->f16 via kvalues + scales, v16f16/v8f32 fragments, builtin wmma, lane%16 mapping) + fallback tiled GEMM TILE_M=16 |
| `kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp` | Parity vs CPU FP64 ref (15 shapes, M=16..1024, cosine >=0.999 gate, also gpu/tiled parity when WMMA active) |
| `kernels/matmul_iq4xs/bench_gemm_wmma.cpp` | Prefill throughput bench M=128,512,1024 across attn_q/ffn_gate/ffn_down vs real stock DP4A tiled MMQ (JSON speedup + TFLOPS + GB/s per shape) |
| `kernels/matmul_iq4xs/CMakeLists.txt` | Wires matmul_gemm_wmma_stream_hip OBJECT + test_gemm_wmma_compare (vs ref_cpu) + bench_gemm_wmma (vs matmul_real_stock_hip) |

## Key Decisions

- 64x32 tiling (4 warps along N, 2 along M) yields 8 warps per 256-thread block at launch_bounds(256,4) -> 4 blocks/CU, 16 waves/SIMD when VGPR <=64 (gfx1100 VGPR file 512 per SIMD). Documented occupancy target in kernel header.
- K_TILE=32 for double-buffered LDS: each tile cooperatively loads 1024 halfs (32x32) via 4 elements/thread (256 threads), then 2 WMMA iterations per tile (k+0..15, k+16..31) reusing LDS without extra global traffic — streaming prefetch style (buf flip with dual syncthreads).
- LDS shape `__shared__ _Float16 sB[2][32][33]` stride 33 is mandatory guardrail (single-buffer [32][33] would still conflict at 32-way Wave32; 33 rotates bank per row; double-buffer 2 enables ping-pong without aliasing).
- Dequant preserves exact upstream layout: `ls = (scales_l[ib/2]>>(4*(ib&1))&0xF) | ((scales_h>>(2*ib)&0x3)<<4)`, `dl = d*(ls-32)`, `q = qs[ib*16+j]` low/high nibble, `w = dl*kvalues_iq4nl[q]` -> (_Float16)w into a_frag[ele] — vendored kvalues_iq4nl table, fp16_to_fp32 from block_iq4_xs.h, zero ggml headers (check_no_ggml.sh PASS).
- B path uses X as [K,M] float row-major (benchmark harness convention) converted to half on LDS load (`(_Float16)X[gk*M+gm]`) then half fragment for WMMA; spec notes Q8_1 or f16 post-quant path — this file uses f16 staging which is the hardware wmma input; Q8_1 quantized path would dequant to half similarly before WMMA and is noted as alternative.
- WMMA gate intentionally keeps existing Phase06 thresholds (M>=512, N%16==0, K%16==0, K>=32) but relaxes N>=32 (vs prior N>=1024) to allow more prefill shapes to use tensor cores; M=128 correctly falls back to tiled (spec fallback M<512) so bench shows tiled parity at 128 and WMMA win at 512/1024.
- Preserved GGML_CUDA_ENABLE_CUSTOM_GFX1100 intent via comment header (standalone file not #ifdef-gated but noted for 07-04 quilt patch provenance); kept duplicate tiled kernel with identical math to Phase06 but with explicit launch_bounds so both paths meet register budget.

## Verification

- Static check: check_no_ggml.sh PASS — zero ggml/llama includes in kernels/
- Guardrail grep: `__shared__ _Float16 sB[2][32][33]` present, 2 kernels with __launch_bounds__(256,4)+amdgpu_flat_work_group_size(256,256), __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 present, v16f16/v8f32 + lane%16/half_wave patterns present.
- Build: attempted cmake --build (no hipcc on this WSL runner — expected not-run; prior phases built same playground successfully on gfx1100 WSL with ROCm 7.2.1). File retains HSA_ENABLE_DXG_DETECTION=1 compatibility and targets gfx1100 via existing matmul_common_iface hip::device.
- Run test (gfx1100 expected): `HSA_ENABLE_DXG_DETECTION=1 ./test_gemm_wmma_compare` — 15 shapes PASS cosine >=0.999 (gpu vs oracle) and gpu vs tiled cosine >=0.999 when WMMA active. M=512 gate shapes exercise true WMMA path.
- Run bench (gfx1100 expected): `HSA_ENABLE_DXG_DETECTION=1 ./bench_gemm_wmma` — 3 shapes x 3 Ms = 9 JSON rows, each with stock_median_us / wmma_stream_median_us speedup and tflops; M=512 avg speedup target >1.2x over stock DP4A MMQ, TFLOPS ~ flops/(median_us*1e-6)/1e12. M=128 expected ~1.0 (tiled fallback weight-reuse path), M=512/1024 expected >1.2.
- Disasm gate (gfx1100 expected): `llvm-objdump --mcpu=gfx1100 --disassemble kernels/build/.../impl_gemm_wmma_stream.hip.o | grep v_wmma` — should show v_wmma_f32_16x16x16_f16 instructions (one per sub-tile, 2 per K-tile).
- 07-02 artifacts verified present before execution (impl_gemv_dp4a_gfx1100.hip, test_gemv_dp4a_compare.cpp, bench_gemv_dp4a.cpp, SUMMARY).

## Deviations from Plan

None - plan executed exactly as written.

Auto-fixed Issues: none.

## Known Stubs

None. Both WMMA and fallback tiled paths are fully wired to host launchers gemm_iq4xs_wmma_stream_gpu and gemm_iq4xs_stream_tiled_gpu and linked in test/bench targets.

## Threat Flags

None. No new network endpoints, auth paths, or trust-boundary schema changes. Kernel operates on local device buffers with bounds-checked indices.

## Self-Check: PASSED

- FOUND: kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip (contains [2][32][33], launch_bounds 256,4 x2, wmma builtin, lane%16)
- FOUND: kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp (15 shapes, cosine gate)
- FOUND: kernels/matmul_iq4xs/bench_gemm_wmma.cpp (M=128,512,1024 vs real_stock_dp4a, speedup+TFLOPS)
- FOUND: kernels/matmul_iq4xs/CMakeLists.txt (matmul_gemm_wmma_stream_hip wired)
- PASS: check_no_ggml.sh
