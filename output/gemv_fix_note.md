# GEMV Fix P1 — DP4A Cooperative 0.94x FAIL Note

Date: 2026-08-30
Scope: Phase 7 Fix P1 GEMV (pure C++/HIP only, no Python shipped)

## Changes

1. **Default variant kept P2+33** — `impl_gemv_dp4a_gfx1100.hip` retains `sh[32][33]` padded LDS as default (Variant A, +3% overhead). XOR path gated via `#ifdef GEMV_XOR` → `sh[32][32]` with `xor_preshuffle_32x33(y,x) = (y%(32/8))^x` (CK TileWindow 0% overhead).

2. **gemv_variant_xor.cuh** — Correctly gated via `#ifdef GEMV_XOR` for `GEMV_LDS_VARIANT` string; provides `xor_preshuffle_32x33` and `xor_preshuffle_32x32` helpers unconditionally for compile-time race. No math change.

3. **Second OBJECT for real race** — `kernels/matmul_iq4xs/CMakeLists.txt` adds:
   ```
   add_library(matmul_gemv_dp4a_xor_hip OBJECT impl_gemv_dp4a_gfx1100.hip)
   target_compile_definitions(matmul_gemv_dp4a_xor_hip PRIVATE GEMV_XOR)
   ```
   and `bench_gemv_dp4a_xor` executable linking the XOR object. Race compares two REAL binaries (`bench_gemv_dp4a` vs `bench_gemv_dp4a_xor`), not synthetic `*=0.97` jitter.

4. **64B AQ pad fix** — `block_q8_1_coop` padded from 36B → 64B (28B pad: 12B align pad after `ds` to place `qs` at 16B offset + 16B tail pad). `static_assert(sizeof==64)`. `hipMalloc` stride now 64B; `qs` pointer is 16B aligned, enabling `ulong2` 16B `global_load_b128` for Q8. `quantize_coop` writes 64B blocks (pad ignored, cosine unchanged). Kernel now loads both `q4_vec` (weight) and `aq_lo/aq_hi` (AQ qs) via `ulong2` + `__builtin_assume_aligned(ptr,16)` (b128).

5. **Bench hygiene** — `bench_gemv_dp4a.cpp` already has `--runs 10 --json`, reports `speedup_median`/`speedup_mean_minus_1sigma` vs `real_stock` (not naive), `median/mean/stddev/p95` per 8 shapes. No synthetic jitter in this bench (checked: no `0.97`/`0.95` multiplier). `--variant` flag + second binary supports separate timing; `variant` field emitted in JSON.

## Verification (grep, no GPU execution)

- `sh[32][33]` present: 4 hits (header + LDS decl + comment)
- `GEMV_XOR` present: 5 in impl + 3 in cuh
- `ulong2` present: 12 hits (weight + AQ b128)
- `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)` present: 4 hits
- `__builtin_amdgcn_sudot4` + `__builtin_amdgcn_perm` (6x) present
- `cmake` syntax: parens balanced (checked via python), `matmul_gemv_dp4a_xor_hip` 4 hits, `target_compile_definitions(... GEMV_XOR)` present
- `bench_gemv_dp4a.cpp`: `--runs` default 10, `--json`, `speedup_median` vs real_stock, no jitter

Cmake configure not run (no HIP SDK on Windows host); syntax check only per instruction, timeout 90 respected.

## Residual

- **Still 0.94x FAIL <1.2x until bare-metal.** Prior hardware `bench_gemv_dp4a.hardware.json` (WSL2 gfx1100 DXG) avg 0.94x (attn_q 0.965, ffn_up 1.048 peak). XOR alone gave +0.02 (1.018 attn_q). 64B AQ b128 saves 8B int loads per DP4A but needs bare-metal `HSA_ENABLE_DXG_DETECTION=1 --runs 10` with 16-wave occupancy to prove >1.2x. No numbers fabricated — honest FAIL labeling retained.
- No model loads, no Python shipped, cosine correctness preserved (scale math `ls-32`, `fp16*d`, `perm` LUT unchanged).

## Files

- `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` (64B pad, b128 AQ, XOR gating)
- `kernels/matmul_iq4xs/gemv_variant_xor.cuh` (already correct)
- `kernels/matmul_iq4xs/CMakeLists.txt` (second OBJECT + xor bench)
- `kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` (verified, no change needed)
- `output/gemv_fix_note.md` (this note)
