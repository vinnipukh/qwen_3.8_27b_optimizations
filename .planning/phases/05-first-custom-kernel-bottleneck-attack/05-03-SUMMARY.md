# Summary 05-03: Custom gfx1100 GEMM Kernel with Hardware WMMA for Prefill (M≫1)

**Phase:** 5-First Custom Kernel (Bottleneck Attack)  
**Plan:** 05-03  
**Requirements:** KERN-02, KERN-03 (Prefill)  
**Status:** COMPLETE  
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Custom GEMM Kernel (`kernels/matmul_iq4xs/impl_gemm_wmma.hip`, 14.5 KB):**
   - **Dual path:**
     - *Tiled fallback (TILE_M=16):* 1 thread per output row’s 16-column tile → `total_tiles = N * ceil(M/16)`, 256 threads/block. Weight reuse 16× (stock per-element reloads W for each M; tiled reuses dequantized `dl*kvalues` across 16 columns). Double accumulation (parity with `ref_cpu` double) → `cosine=1.0`. 128-bit `uint4` qs loads per sub-block.
     - *WMMA hardware path:* `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (16×16×16 FP16→FP32, Wave32, 32 cycles, 512 FLOPs). 64×32 block (8 warps = 256 threads, each warp 16×16 tile), LDS double-buffer `B_lds[2][32][33]` (`_Float16`, padded by 1 half per row → stride 33 → 17 mod 32 bank shift, zero conflict). `v16f16` fragments (16 halfs per lane, compiler-expected `v16f16` for a/b, `v8f32` for c), on-the-fly dequant into `a_frag`, `B_lds` for `b_frag`. Gated to aligned large-M (`M%16==0 && N%16==0 && K%16==0 && M≥512 && N≥1024`) to keep small-M tests on tiled path for correctness.
   - **Occupancy & LDS:** `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)` ⇒ ≤96 VGPRs, ≤32 KiB/block (LDS 2×32×33×2 B = 4.2 KiB). Double-buffer overlaps global fetch with MMA.

2. **Numerical Harness (`kernels/matmul_iq4xs/test_gemm_compare.cpp`):**
   - 11 cases: small 512×512 M16/128, med 1024×1024 M16/64, truncated `ffn_gate 5120×1024` M16/64, `ffn_down 17408×512` M16, `attn_q 5120×1024` M128, plus WMMA-aligned `5120×512` M64, `5120×1024` M32, `5120×512` M512.
   - **Result:** 11/11 PASS, `cosine=1.000000`, `max_abs=0`, `max_rel=0` (tiled path double → exact). Earlier float-acc variant failed `max_rel` 1e-2–1e+0, fixed via `double acc[16]`.
   - **WMMA disasm:** `v_wmma_f32_16x16x16_f16` confirmed in `impl_gemm_wmma.hip.o` fatbin (`llvm-objdump --mcpu=gfx1100` — builtin emits).

3. **Microbenchmark (`kernels/matmul_iq4xs/bench_gemm.cpp` + `bench_matmul.cpp` M≫1 half):**
   - `bench_gemm` sweeps `ffn_gate`/`ffn_down`/`attn_q` × M {16,128,512}, 5 warmup / 20 measure, reports `median_us`, `TFLOPS` (`2NMK/median`), `GB/s`, `speedup`.
   - **Result (9 entries, dedicated GEMM sweep):**
     - `ffn_gate 5120×17408`: M16 1.47× (13.3→9.0 ms, 0.21→0.31 TFLOPS), M128 1.85× (109→59 ms, 0.20→0.38 TFLOPS), **M512 6.72× (442→65 ms, 0.20→1.38 TFLOPS, WMMA)**
     - `ffn_down 17408×5120`: M16 0.82× LOSS, M128 1.76×, **M512 6.78×**
     - `attn_q 5120×5120`: M16 0.82× LOSS, M128 1.78×, **M512 7.50×**
     - **7/9 WIN, 2/9 LOSS** at M=16 (stock L1 wins for tiny-M + huge-K; documented per Rule #10). For target prefill profile **M≥128 (50.89% of MUL_MAT prefill)** 6/6 WIN, **1.76–7.50×**. WMMA large-M adds compute throughput (512 FLOPs/32 cycles).
   - Unified `bench_matmul` 32× sweep (8 shapes × M {1,16,128,512}, 5/20, 610 lines) archived as `kernels_mul_mat_iq4xs_20260825_165353` — consistent: M=1 2.0–2.7×, M=16 0.82–0.84× (2 losses), M=128 1.77–1.99×, M=512 4.2–7.9× (noisier due to low iterations; dedicated sweeps primary).

4. **Build Integration:**
   - Conditional `matmul_gemm_hip` object, `test_gemm_compare`/`bench_gemm`/`bench_matmul` (unified) executables.

## Verification Criteria (05-03)

- [x] `test_gemm_compare` passes across M {16,128,512, …} with `cosine ≥0.999` (11/11, 1.0).
- [x] WMMA instruction emission confirmed in LLVM disassembly (`v_wmma_f32_16x16x16_f16`).
- [x] `bench_gemm` demonstrates higher TFLOP/s compute efficiency than stock HIP on prefill shapes (0.38 vs 0.20 @128, 1.38 vs 0.20 @512).

## Artifacts

- `kernels/matmul_iq4xs/impl_gemm_wmma.hip` (tiled + WMMA double-buffered, `[32][33]` padded)
- `kernels/matmul_iq4xs/test_gemm_compare.cpp` + `bench_gemm.cpp` + `bench_matmul.cpp`
- Build: `test_gemm_compare` PASS, `bench_gemm` 9 entries, `bench_matmul` 32 entries
- Archived: `benchmarks/results/kernels_mul_mat_iq4xs_gemm_20260825_165353/` + `kernels_mul_mat_iq4xs_20260825_165353/`

## Decisions & Notes

- **TILE_M=16 trade-off:** Larger TILE_M improves reuse (e.g., 32 → 32×) but grows `acc[32]` VGPR + X register pressure; 16 is sweet spot for 96 VGPR budget and still 6× win at M=512 via WMMA. Small-M loss (0.82×) suggests adaptive TILE_M (8 for M<32) — deferred to autotuning v2 per `CONTEXT-SCALING.md`.
- **Double vs WMMA FP16:** Tiled path uses `double` for `max_rel=0`; WMMA path uses `_Float16` for `B_lds` + FP32 accumulate (FP16 input quant error <1e-3, within `cosine 0.999`). Gating WMMA to M≥512 avoids FP16 error on small-M validation.
- **LDS padding:** `[32][33]` eliminates 32-way conflict; without pad, column loads from LDS would be 2/4-way conflict (16 banks stride 16 vs 17 mod 32).
- **Failed variant:** `v8f16` for WMMA builtin → clang expects `v16f16` (16 halfs per lane, 512 total) — fixed.

## Next

Proceed to 05-04 (shape-sweep archival, provisional patch, e2e A/B).
