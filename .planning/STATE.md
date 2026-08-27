---
gsd_state_version: 1.0
current_phase: 7
current_phase_name: Hybrid DP4A & WMMA Matrix Core Optimization
status: gaps_found
stopped_at: Phase 7 artifacts complete 07-01..07-04 (DP4A comparator 84us vs 543us, GEMV peak 1.178x, WMMA [2][32][33] + wmma builtin, quilt patch 355 lines verified) — verifier 2/5 must-haves, 3 gaps pending WSL2 gfx1100 bare-metal re-bench (07-VERIFICATION.md).
last_updated: "2026-08-27T19:50:00.000Z"
last_activity: 2026-08-27
last_activity_desc: Phase 7 full auto 07-01->07-04 + gsd-verify complete; thermal monitor no 90C aborts (fallback polling WinError5, no HWiNFO daemon).
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 28
  completed_plans: 28
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Beat stock llama.cpp HIP on at least one important Qwen3.8-27B workload on the RX 7900 XT with a custom gfx1100 kernel, within agreed numerical tolerance — measured, reproducible, bisectable.

**Current focus:** Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization (beating real production stock end-to-end).

## Current Position

Phase: 7 (Hybrid DP4A & WMMA Matrix Core Optimization) — ARTIFACTS COMPLETE, VERIFICATION gaps_found (2/5)
Status: All 4 plans executed (45m+60m+45m+2h) — guardrails 1-4 PASS via grep, 3 gaps require WSL2 gfx1100 hardware (see 07-VERIFICATION.md)

Progress: [█████████▉] 100% artifacts (6 of 7 phases verified, 28/28 plans) — verifier says 2/5 truths, 3 missing bare-metal benches

## What Phase 6 Delivered

- **Thermos Remediation & Kernel Guard Hardening (06-05):**
  - Eliminated barrier divergence in GEMV and WMMA GEMM workgroups.
  - Replaced unaligned 16-byte `uint4*` pointer casting with 8-byte aligned `uint64_t` pairs.
  - Implemented correct RDNA3 Wave32 WMMA 16x16x16 lane and fragment layout (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`), eliminating uninitialized LDS reads.
  - Added WMMA gate-passing test case (`wmma_gate_pass_5120_1024_512`) -> PASS `cosine=1.000000`.
  - Extracted common test/bench harness utilities (`kernels/common/matmul_test_util.h`).
  - Standardized fixture manifests (`manifest_dequant.json` and `manifest_matmul.json`) with deterministic SHA256 seeds.
- **Empty-Flag Switch Plumbing Proof (06-01, INTEG-01):**
  - Added `GGML_CUDA_ENABLE_CUSTOM_GFX1100` CMake option (default `OFF`).
  - Proved both `build-stock` (OFF) and `build-custom-empty` (ON + stub) compile cleanly targeting gfx1100 and pass op-gate (1193/1193 PASS).
- **Winner In-Tree Integration & Patch Provenance (06-02, INTEG-01):**
  - Vendored custom `custom_gfx1100/{gemv_iq4xs.cuh, gemm_iq4xs.cuh}` with exact GGML tensor convention layouts (`[K, M]` and `[N, M]`).
  - Generated quilt patch `patches/0001-gfx1100-mul-mat-custom.patch` via `git diff` against pinned upstream `bb4caa75`, verified with `git apply --check`.
  - Guarded all dispatch code inside `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)`.
- **Baseline-Preservation Guard (06-03, INTEG-01, Rule #3):**
  - Verified `build-stock` rebuilds cleanly from same tree with 0 errors across 4,243 supported ops (`benchmarks/results/phase6/op_gate_stock_20260827.json`).
  - Verified `baseline/binaries/v0.2.0-bb4caa75/` binaries are preserved with intact sha256 checksums.
- **Publication Package & Release Hygiene (06-04, PUB-01):**
  - `docs/PUBLICATION.md` published covering exact build commands, hardware and environment versions, model provenance, methodology, raw data, kernel sources, and known limitations.
  - `CHANGELOG.md` created with full release history.
  - `NOTICE` attribution and `LICENSE` (Apache 2.0) delivered.
  - Tagged `v1.0.0-gfx1100`.

## What Phase 7 Delivered (07-01..07-04 full auto)

- **07-01 True Upstream DP4A Comparator:** `real_stock_dp4a_comparator.hip` (25156B) vendors exact `vec_dot_iq4_xs_q8_1` + `quantize_row_q8_1` via `ggml_cuda_dp4a`/`__builtin_amdgcn_sudot4` + 6× `__builtin_amdgcn_perm` LUT, `ls decode + d=half2float*low2float`, GEMV single-warp MMVQ (calc_nwarps=1,VDR=4) + GEMM tiled TILE_M=16, `test_real_stock_compare` 15/15 PASS cosine 0.999985, `bench_real_stock` 84.39us DP4A vs 542.97us naive 6.43× for attn_q, 8-shape table in `BASELINE_DP4A.md` + `baseline_dp4a.json`, CMake `matmul_real_stock_hip` PASS
- **07-02 Cooperative Wave32 DP4A GEMV:** `impl_gemv_dp4a_gfx1100.hip` (15186B) 8-thread/row coop (256→32 rows/block, grid ceil(N/32)), ulong2 128-bit qs, LDS `sh[32][33]` padded, `__launch_bounds__(256,4)`+`amdgpu_flat_work_group_size(256,256)` Wave32 templated, `coop_dp4a` via `__builtin_amdgcn_sudot4` + `perm` LUT, `quantize_coop` Q8_1, `test_gemv_dp4a_compare` 10/10 PASS cos 0.999985 vs ref & 1.000 vs stock, `bench_gemv_dp4a` peak 1.178x (111.47→94.67us attn_q) avg 1.00 under WSL DXG jitter (bare-metal target >1.2x/40-45 t/s)
- **07-03 Streaming WMMA GEMM:** `impl_gemm_wmma_stream.hip` (13610B) 64x32 per block (4×2 warps, 8 warps = 256 thr), double-buffered `_Float16 sB[2][32][33]` stride33, K_TILE=32 =2×WMMA per tile, on-the-fly IQ4_XS→half `d*(ls-32)*kvalues_iq4nl`, `v16f16`/`v8f32` + `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, lane%16/half_wave, fallback tiled TILE_M=16 gated M≥512, `test_gemm_wmma_compare` 15 shapes cosine ≥0.999 gate, `bench_gemm_wmma` M=128/512/1024 vs real stock DP4A (not run on Windows host, needs metal)
- **07-04 Quilt Overlay & A/B Protocol:** `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv_iq4xs.cuh,gemm_iq4xs.cuh}` vendored winners compact with GGML layout fix `X[gm*K+gk]`/`Y[m*N+n]` (was `X[gk*M+gm]` bug), kept LDS [32][33] + launch_bounds, patch `patches/0001-gfx1100-mul-mat-custom.patch` 355 lines/276 insertions via `git -C llama.cpp diff HEAD` over `bb4caa75`, `git apply --check` PASS, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF default + `#if defined` guards in `mmq.cu`/`mmvq.cu`/`CMakeLists`, `empty.cuh` fallback preserved, `KERNEL-BENCH-DIFF.md §8` + `docs/PUBLICATION.md` Phase7 update + `CHANGELOG.md` unreleased (no fabricated tok/s)
- **Full-auto thermal guard:** `thermal_monitor.py` (90C threshold, 2s poll, timeout-guarded bash) ran 1500s fallback polling due WinError5 HWiNFO access denied (no daemon, no hwmon in WSL) → no kills, correctly not aborted; `logs/thermal_monitor.log` captured

## Verifier Gaps (07-VERIFICATION.md 2026-08-27T19:50Z score 2/5)

- GAP1: GEMV >1.2x + >38 t/s decode — microbench peak 1.178 avg 1.00 under virtualization, no llama-bench JSON phase7/ab_* decode tok/s
- GAP2: WMMA >950 t/s prefill — kernel substantive but no bench_gemm_wmma JSON vs real DP4A at M≥128/512 nor prefill tok/s JSON (prior 6-7x was vs naive, not vs DP4A)
- GAP3: QUAL-01/02 green on custom + hwinfo thermal trace — stock op_gate 4243 PASS exists, custom gates not executed (no hipcc/model on Windows host), no hwinfo_daemon 1Hz trace
- Artifacts PASS guardrails 1-4 via grep (DP4A/perm, LDS33, launch_bounds, patch gating)

## Next Step (bare-metal WSL2 gfx1100 required)

- `wsl -d Ubuntu-24.04 -- bash -c 'HSA_ENABLE_DXG_DETECTION=1 cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 && cmake --build kernels/build'` (timeout 120s)
- `./kernels/build/matmul_iq4xs/bench_real_stock`, `./bench_gemv_dp4a`, `./bench_gemm_wmma` (timeout 90s each) → capture median_us + speedup JSON proving >1.2x vs real DP4A
- `cmake -S llama.cpp -B build-stock -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF` and `build-custom -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` (timeout 300s) + `git -C llama.cpp apply --check ../patches/0001*`
- `HSA_ENABLE_DXG_DETECTION=1 python benchmarks/bin/run_op_gate.py` (0 errors) and `run_model_gate.py` (PPL 6.4271) on build-custom (timeout 90s/300s)
- `llvm-objdump --mcpu=gfx1100 ... | grep v_wmma` + `v_dot4` for disasm, `hipcc --save-temps` VGPR ≤64 check
- Paired `llama-bench` --single-turn --simple-io --load-mode none -ngl 99 -b 2048 across {512,1024,2048,4096} stock vs custom in ONE thermal window with `python benchmarks/host/hwinfo_daemon.py --watch --pid-file /tmp/bench.pid --out-dir benchmarks/results/phase7/ab_*` + `thermal_watchdog.py --threshold-c 90` (timeout 300s) → assert decode >38 & >stock, prefill >950 & >stock, update `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` + `docs/PUBLICATION.md` + `benchmarks/results/phase7/CHECKSUMS`
- Then `/gsd-plan-phase --gaps` to close 07-VERIFICATION.md gaps
