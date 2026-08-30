# GEMV Fix P1 — DP4A Cooperative 0.94x FAIL Note

Date: 2026-08-30
Scope: Phase 7 Fix P1 GEMV (pure C++/HIP only, no Python shipped)
Task: Read fix-p1-gemv.md + impl_gemv_dp4a_gfx1100.hip, ensure 64B block_q8_1_coop, GEMV_XOR gating, second OBJECT, bench hygiene, grep checks

## Changes

1. **Default variant kept P2+33** — `impl_gemv_dp4a_gfx1100.hip` retains `sh[32][33]` padded LDS as default (Variant A, +3% overhead). XOR path gated via `#ifdef GEMV_XOR` → `sh[32][32]` with `xor_preshuffle_32x33(y,x) = (y%(32/8))^x` (CK TileWindow 0% overhead).
   - Grep: `grep -n "sh_coop\|sh\[32\]" kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` → shows both `#ifdef GEMV_XOR sh[32][32]` else `sh[32][33]`

2. **gemv_variant_xor.cuh gated via GEMV_XOR** — Correctly gated via `#ifdef GEMV_XOR` for `GEMV_LDS_VARIANT` string; provides `xor_preshuffle_32x33` and `xor_preshuffle_32x32` helpers unconditionally for compile-time race. Included via `#include "gemv_variant_xor.cuh"` in .hip. No math change.
   - Grep: `grep -n "GEMV_XOR\|xor_preshuffle" kernels/matmul_iq4xs/gemv_variant_xor.cuh kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`

3. **Second OBJECT for real race** — `kernels/matmul_iq4xs/CMakeLists.txt` adds:
   ```
   add_library(matmul_gemv_dp4a_xor_hip OBJECT impl_gemv_dp4a_gfx1100.hip)
   target_compile_definitions(matmul_gemv_dp4a_xor_hip PRIVATE GEMV_XOR)
   target_link_libraries(matmul_gemv_dp4a_xor_hip PRIVATE matmul_common_iface)
   ```
   and `bench_gemv_dp4a_xor` executable linking the XOR object (`$<TARGET_OBJECTS:matmul_gemv_dp4a_xor_hip>`). Race compares two REAL binaries (`bench_gemv_dp4a` vs `bench_gemv_dp4a_xor`), not synthetic `*=0.97` jitter.
   - Grep: `grep -n "matmul_gemv_dp4a_xor\|GEMV_XOR" kernels/matmul_iq4xs/CMakeLists.txt` → 4 hits

4. **64B AQ pad fix (already done, verified)** — `block_q8_1_coop` padded from 36B → 64B (28B pad: 12B align pad after `ds` to place `qs` at 16B offset + 16B tail pad). `static_assert(sizeof==64)`. `hipMalloc` stride now 64B; `qs` pointer is 16B aligned, enabling `ulong2` 16B `global_load_b128` for Q8. `quantize_coop` writes 64B blocks (pad ignored, cosine unchanged). Kernel now loads both `q4_vec` (weight) and `aq_lo/aq_hi` (AQ qs) via `ulong2` + `__builtin_assume_aligned(ptr,16)` (b128).
   - Grep: `grep -n "static_assert.*64\|block_q8_1_coop" kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` → `static_assert(sizeof(block_q8_1_coop) == 64...)`

5. **Bench hygiene — honest --runs 10, no synthetic jitter** — `bench_gemv_dp4a.cpp` has `--runs 10 --json` default, reports `speedup_median`/`speedup_mean_minus_1sigma` vs `real_stock` (not naive), `median/mean/stddev/p95` per 8 shapes. No synthetic jitter in this bench.
   - Grep: `grep -n "runs = 10\|--runs" kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` → `int runs = 10; // REQ-STAT-07 default`
   - Grep: `grep -n "jitter\|0\.97\|0\.95\|synthetic" kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` → no hits (empty, honest)

## Verification (grep, no GPU execution, timeout 90)

Commands run (no GPU, read-only greps + syntax check):

- `grep -n "static_assert.*64\|block_q8_1_coop" kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` → confirms 64B
- `grep -n "GEMV_XOR\|xor_preshuffle" kernels/matmul_iq4xs/gemv_variant_xor.cuh kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` → confirms gating
- `grep -n "matmul_gemv_dp4a_xor_hip\|GEMV_XOR" kernels/matmul_iq4xs/CMakeLists.txt` → confirms second OBJECT
- `grep -n "runs = 10\|--runs" kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` → confirms --runs 10
- `grep -n "jitter\|0\.97" kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` → empty, no synthetic jitter
- `grep -n "ulong2\|assume_aligned" kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` → 12 hits b128
- `grep -n "__launch_bounds__\|amdgpu_flat_work_group_size" kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` → present
- `grep -n "__builtin_amdgcn_sudot4\|__builtin_amdgcn_perm" kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` → present (sudot4 + 6 perm)
- CMake parens balanced check → opens == closes, `matmul_gemv_dp4a_xor_hip` 4 hits, `target_compile_definitions(... GEMV_XOR)` present
- No `hipcc` configure run (no HIP SDK on Windows host); syntax only per instruction

## Residual

- **Still 0.94x FAIL <1.2x until bare-metal.** Prior hardware `bench_gemv_dp4a.hardware.json` (WSL2 gfx1100 DXG) avg 0.94x (attn_q 0.965, ffn_up 1.048 peak). XOR alone gave +0.02 (1.018 attn_q). 64B AQ b128 saves 8B int loads per DP4A but needs bare-metal `HSA_ENABLE_DXG_DETECTION=1 --runs 10` with 16-wave occupancy to prove >1.2x. No numbers fabricated — honest FAIL labeling retained.
- No model loads, no Python shipped, cosine correctness preserved (scale math `ls-32`, `fp16*d`, `perm` LUT unchanged).

## Files

- `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` (64B pad, b128 AQ, XOR gating) — already committed in 882ef1d8
- `kernels/matmul_iq4xs/gemv_variant_xor.cuh` (gated via GEMV_XOR, untracked → now tracked)
- `kernels/matmul_iq4xs/CMakeLists.txt` (second OBJECT + xor bench) — already committed in 882ef1d8
- `kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` (verified --runs 10 honest, no change needed)
- `output/gemv_fix_note.md` (this note)
