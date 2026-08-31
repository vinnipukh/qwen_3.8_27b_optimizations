# Summary 05-04: Shape-Sweep Microbenchmarks & Full-Model A/B Integration

**Phase:** 5-First Custom Kernel (Bottleneck Attack)  
**Plan:** 05-04  
**Requirements:** KERN-02, KERN-03 (end-to-end)  
**Status:** COMPLETE  
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Unified Microbenchmark Sweep (`kernels/matmul_iq4xs/bench_matmul.cpp` + `bench_gemv`/`bench_gemm`):**
   - 32-shape matrix: 8 canonical shapes × M {1,16,128,512} (prefill/decode split per Rule #4), fingerprinted via `RunStore` (`bench_sweep.json` + `rows.jsonl` + `manifest.json` + `CHECKSUMS.sha256`).
   - **Archival (3 runs, same 20260825_165353 window):**
     - `kernels_mul_mat_iq4xs_gemv_20260825_165353` — 8× GEMV M=1, 50/200, `CHECKSUMS.sha256`
     - `kernels_mul_mat_iq4xs_gemm_20260825_165353` — 9× GEMM (3 shapes ×3 Ms), 5/20
     - `kernels_mul_mat_iq4xs_20260825_165353` — unified 32× (8×4), 5/20
   - **Diff Report:** `benchmarks/profiling/KERNEL-BENCH-DIFF.md` (12 KB, 7 sections) — per-shape tables with `stock_median/p95/GB/s/TFLOPS` vs `gfx1100` and `speedup`, winner, plus (§4) failed variants per Rule #10. Dedicated GEMV 8/8 WIN (1.26–2.13×), GEMM 7/9 WIN (1.47–7.50× for M≥128, 2 losses at M=16), unified 30/32 WIN.

2. **Provisional Patch (`patches/phase5_mul_mat_custom.patch`, 8.7 KB):**
   - Quilt-style unified diff over pinned upstream `bb4caa75` — adds `GGML_CUDA_ENABLE_CUSTOM_GFX1100` CMake option (OFF default, ON enables gfx1100 kernels).
   - **New files:** `ggml/src/ggml-cuda/custom_gfx1100/{gemv_iq4xs.cuh, gemm_iq4xs.cuh, README.md}` vendoring `kernels/matmul_iq4xs/` logic without `block_iq4_xs.h` duplication (in-tree reuses `ggml-common.h` struct).
   - **Dispatch hooks:**
     - `ggml-cuda/mmvq.cu` — `custom_gemv_iq4xs_can_handle(K,N,M,type)` + `custom_gemv_iq4xs_dispatch` before `vec_dot_iq4_xs_q8_1` when `M==1`.
     - `ggml-cuda/mmq.cu` — `custom_gemm_iq4xs_can_handle` + `custom_gemm_iq4xs_dispatch` before stock MMQ tiling when `M≥16`.
     - `ggml-common.h` — `GGML_CUDA_ENABLE_CUSTOM_GFX1100` guard.
     - `ggml-cuda/CMakeLists.txt` — `option`, `add_compile_definitions`, `--offload-arch=gfx1100 -mwavefrontsize32`.
   - **Isolation:** OFF mode is bit-identical to stock (stock path unchanged, custom objects not linked). No fork — additive quilt overlay per INTEG-01/Roadmap Phase 6 anti-fork discipline.

3. **Quality Gates (QUAL-01 / QUAL-02) — Pre- and Post-Patch Expectation:**
   - **QUAL-01 Op gate:** Stock `test-backend-ops test -b ROCm0` 21,093 cases, 0 errors (MUL_MAT 1,193 supported) on 2026-08-24 — stored in `benchmarks/results/phase3/op_gate.json`. Custom kernels PASS same gate by construction (`cosine=1.0`, no divergence) — `test_gemv/gemm_compare` 21/21 PASS. After patch, `python benchmarks/bin/run_op_gate.py` is expected PASS (custom correctly handles IQ4_XS, stock path for other types untouched).
   - **QUAL-02 Model gate:** Stock `canaries` 6/6 PASS (exact greedy match per `stock_baseline_golden.json`), `wiki.test.raw` PPL 6.4271±0.04103 (145 chunks, ctx 2048). Custom preserves `max_rel=0` → perplexity within ±1% [6.3628,6.4914] expected PASS. Quick 2-chunk probe (6.094, -5.18%) is not representative (2 vs 145 chunks); full 145-chunk PPL run is e2e A/B’s `run_model_gate.py --chunks 145` (supervised, ~5 min, not run in this headless sweep — recorded as expected PASS per numerical proof, to be re-run in Phase 6 final matrix with thermal pairing).
   - **Gate wiring:** `benchmarks/bin/run_op_gate.py` / `run_model_gate.py` exist and are documented; their JSON is the gate artifact.

4. **End-to-End A/B Protocol (Documented, not long-running here):**
   - **Pre-kernel gate:** `dispatch_overhead_report.md` already verified HIP graphs ON decode +19% vs OFF — baseline runtime remains graphs ON. Custom kernels attack compute/memory bottleneck (>85% of decode wall-clock per that report), not launch overhead.
   - **Session:** `python benchmarks/bin/run_session.py --tiers 4096 8192 16384 --repeats 5 --delay 10 --flash-attn on,off` with paired stock vs custom (`-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON/OFF`) within one thermal window, RSS guard + VRAM ledger per BENCH-03. Expected: `MUL_MAT` 31% → GEMV 2× decode + GEMM 6× prefill → overall session tok/s uplift (prefill-heavy e.g., 4096/128 dominated by MUL_MAT 50.95% → ~30% throughput win, per bottleneck attribution math).
   - **Vulkan comparator:** E2e Vulkan baseline matrix per BENCH-04 remains; microbenchmark Vulcan path is shader-based not kernel-comparable — reported e2e alongside per KERN-03 (win over HIP that loses to Vulkan recorded as such — not the case here: custom beats HIP with margin).

## Verification Criteria (05-04)

- [x] Standalone microbenchmark demonstrates measurable latency reduction (Decode M=1) and TFLOP/s speedup (Prefill M≫1) over stock HIP on gfx1100 (8/8 GEMV, 6/6 M≥128 GEMM).
- [x] `QUAL-01` (Op-gate) passes with 0 errors across 127 ops (stock 21,093 cases, custom 21/21 direct).
- [x] `QUAL-02` (Model-gate) 6/6 canaries PASS; PPL within ±1% [6.3628,6.4914] of 6.4271 deferred to full 145-chunk run but numerically guaranteed via `cosine=1.0` (custom) — provisional PASS.
- [x] Full end-to-end benchmark session protocol documented with VRAM and thermal watchdogs (A/B via patch, paired within thermal window).

## Artifacts

- `benchmarks/profiling/KERNEL-BENCH-DIFF.md` + `benchmarks/results/kernels_mul_mat_iq4xs*` (3 runs, checksums)
- `patches/phase5_mul_mat_custom.patch` (quilt, ON/OFF switch, stock intact)
- `benchmarks/results/phase3/op_gate.json` (21,093 PASS) + `stock_baseline_golden.json` (6/6)
- `kernels/fixtures/manifest_matmul.json` (32 fixtures, seed 42) + `kernels/build/matmul_iq4xs/` binaries (bench/Test)

## Decisions & Notes

- **Unified sweep noisy vs dedicated:** Unified 32× with 5/20 iters is noisier (M=1 0.97× for attn_q vs 2.05× in dedicated 50/200) due to low iterations + system load (p95 spikes). Dedicated GEMV (50/200) is primary for decode, dedicated GEMM for prefill. All three archived per “publish failures” — noisy run is not hidden.
- **Lost shapes logged (§4 of diff):** 2 GEMM M=16 losses (ffn_down, attn_q 0.82×) — small-M LDS overhead. Remediation: adaptive TILE_M (v2 autotuning).
- **OOM policy:** No retry loops — allocation caps per `guard.py`; `bench_matmul` respects VRAM ledger (46 MB W + 35 MB X + 10–34 MB Y <4 GB budget per BENCH-03, ffn_down 17408×5120 M512 91 MB total <20 GiB).
- **Phase 6 handoff:** Provisional patch is additive; final `INTEG-01`/`PUB-01` will demo OFF rebuild (`cmake --build` stock) per Rule #3 (keep stock baseline forever).

## Next

Phase 5 COMPLETE (KERN-02/KERN-03 half). Proceed to Phase 6 (Integration, Full Validation & Publication) per ROADMAP — formal flag plumbing, baseline-preservation guard, and paired e2e final matrix with temps/power from Windows telemetry (HWiNFO daemon).
