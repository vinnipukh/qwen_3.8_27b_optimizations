---
gsd_state_version: 1.0
current_phase: 7
current_phase_name: Hybrid DP4A & WMMA Matrix Core Optimization — RE-SCOPED 2026-08-28 (≥10% + Windows + 10×/15×)
status: gaps_found
stopped_at: Phase 7 artifacts complete 07-01..07-04 (DP4A 84us vs 543us, GEMV peak 1.178x avg 1.00 under WSL DXG jitter, WMMA [2][32][33]+wmma builtin, quilt 355 lines) — verifier 2/5 on OLD 5-truth set; RE-SCOPED 2026-08-28 to 7 truths (added REQ-WIN-07 Windows ≤2 langs, REQ-PERF-07 ≥1.10× pp+tg at {512,1024,2048,4096,8192}, REQ-STAT-07 N≥10 / LLM QA N≥15) per deep-research report + owner 3 wishes — 5 gaps now pending (Windows build + 10% gate + 10× rigour) before Phase 7 can close (see ROADMAP.md Phase 7 2026-08-28).
last_updated: "2026-08-29T00:00:00.000Z"
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

**Core value:** Beat **stock llama.cpp HIP by ≥10%** on at least one important Qwen3.8-27B workload on the RX 7900 XT with a **Windows-native (≤2 langs, no Python/JS servers)** custom gfx1100 kernel, within agreed numerical tolerance — **measured N≥10 (LLM QA N≥15) averaged, reproducible, bisectable** (re-scoped 2026-08-28).

**Current focus:** Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization — **RE-SCOPED 2026-08-28** (≥10% + Windows + 10×/15×) per deep-research `output/deep-research/1000t-s-at-8k-gfx1100.md` + owner 3 wishes. Awaiting owner articles before re-planning.

## Current Position

Phase: 7 (Hybrid DP4A & WMMA Matrix Core Optimization — RE-SCOPED 2026-08-28) — ARTIFACTS COMPLETE on OLD scope, **RE-SCOPED to 7 truths (2/7 passed)**
Status: All 4 plans executed (45m+60m+45m+2h) — guardrails 1-4 PASS via grep on OLD 5-truth set; **3 NEW must-have gaps** added 2026-08-28: REQ-WIN-07 Windows-native (`build_windows.bat` + `llama-server.exe` @ `localhost:8000`, `≤2` langs), REQ-PERF-07 `≥1.10× pp+tg` at `{512,1024,2048,4096,8192}` (current `808→849` pp4096 = +5.1% **fails**), REQ-STAT-07 `N≥10` (`15×` LLM QA) rigour — all 5 gaps require **Windows bare-metal HIP SDK + WSL2 gfx1100 re-bench** (see ROADMAP.md Phase 7 2026-08-28 + `output/deep-research/1000t-s-at-8k-gfx1100.md`).

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

## Verifier Gaps (07-VERIFICATION.md 2026-08-27T19:50Z score 2/5 on OLD 5-truth set — **RE-SCOPED 2026-08-28 to 7 truths, score now 2/7**)

- GAP1 (old Truth 2): GEMV >1.2x + >38 t/s decode — microbench peak 1.178 avg 1.00 under virtualization, no llama-bench JSON phase7/ab_* decode tok/s (now must be **`N=10` averaged** per REQ-STAT-07)
- GAP2 (old Truth 3): WMMA >950 t/s prefill — kernel substantive but no bench_gemm_wmma JSON vs real DP4A at M≥128/512 nor prefill tok/s JSON (prior 6-7x was vs naive, not vs DP4A) (now **`N=10` averaged**)
- GAP3 (old Truth 5): QUAL-01/02 green on custom + hwinfo thermal trace — stock op_gate 4243 PASS exists, custom gates not executed (no hipcc/model on Windows host), no hwinfo_daemon 1Hz trace (now **`N=10` per gate**)
- **GAP4 (NEW must-have #1, REQ-WIN-07)**: No Windows-native build — `build_windows.bat` missing / not proven `hipcc --offload-arch=gfx1100` via `HIP_PATH` + `Ninja` + `clang++.exe`, no `build-windows/bin/llama-server.exe` `curl :8000 → 200` smoke, tree still has `benchmarks/` Python + `.mjs` (`≤2` langs not met)
- **GAP5 (NEW must-have #2, REQ-PERF-07)**: `≥1.10× pp+tg` at `{512,1024,2048,4096,8192}` not proven — current `808→849` pp4096 = **+5.1% FAILS** the `≥10%` gate (`mean−1σ` must be `≥1.10×`, `N=10` thermal-paired)
- **GAP6 (NEW must-have #3, REQ-STAT-07)**: No `10×` / `15×` rigour — prior benches were `N=1` or `N=3`; need `median`+`mean`+`stddev`+`p95` over `N=10` (microbench + `llama-bench`) and `N≥15` LLM QA (`avg tok/s` + `per-run` table) — single-run claims banned
- Artifacts PASS guardrails 1-4 via grep (DP4A/perm, LDS33, launch_bounds, patch gating) — still pass, but now gated on `N=10` re-bench

> **Deep-research implication (2026-08-28):** `output/deep-research/1000t-s-at-8k-gfx1100.md` shows `8k` is quadratic cliff (`800 GB/s` roof, `KV≈128 KiB/tok`, `L3 96MB≪134MB` tile, WSL2 `800 GiB` lying + `BSOD` risk, `rocprofv3` blind). `8192` tier is now conditional on VRAM pre-flight (`FA` + `GQA`); `≥1.10×` must hold even if `8192` is gated FAIL.

## Next Step — READY FOR EXECUTION (re-planned 2026-08-28, NOT STARTED — awaiting execution agent)

**State:** Re-scoped to 7 truths (2/7 passed, 3 new must-haves REQ-WIN-07/REQ-PERF-07/REQ-STAT-07). Execution has **NOT started** — awaiting another agent (`/gsd-execute-phase 7` or equivalent plan runner). Do not re-plan — plans 07-01..07-04 are amended and ready. Bare-metal WSL2 gfx1100 required; all benches `N=10`, LLM QA `N=15`.

**Executable wave order:**

- **wave1: 07-01 bench_real_stock N=10** — `bench_real_stock --runs 10 --json` vs real `vec_dot_iq4_xs_q8_1` DP4A (84us vs naive 543us) → `kernels/matmul_iq4xs/baseline_dp4a.json` + `BASELINE_DP4A.md` median/mean/stddev/p95 per 8 shapes, `test_real_stock_compare` 15/15 PASS; `hipcc --offload-arch=gfx1100` clean and Windows `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100` compile probe.
- **wave2 (parallel): 07-02 GEMV variants vs 07-03 GEMM tile/XOR/P/b128/LUT variants (both N=10)**
  - **07-02:** `impl_gemv_dp4a_gfx1100.hip` cooperative 8-thread/row (256→32 rows/block) LDS `[32][33]` vs XOR preshuffle `x'=(y%(32/8))^x` + b128 `ulong2`/`global_load_b128`/`float4` + offline 16×64 swizzle (`tools/swizzle_iq4xs.py` offline-only) + `__launch_bounds__(256,4)`/`amdgpu_flat_work_group_size(256,256)` + `__builtin_amdgcn_sudot4`/`perm`; `bench_gemv_dp4a --runs 10 --json` median/mean/stddev/p95 + `speedup_median` vs real DP4A + variant winner table; `llvm-objdump --mcpu=gfx1100 | grep v_dot4` + `hipcc --save-temps` VGPR ≤64 gate.
  - **07-03:** `impl_gemm_wmma_stream.hip` streaming WMMA `64x32` base + tile sweeps `64x32`/`64x64`/`128x32` + `P=2` `sB[2][32][33]` vs `P=4` `sB[4][32][32]` + `__builtin_amdgcn_sched_barrier(0x0080 DS / 0x0008 WMMA)` + B-stationary + LUT mu=4 (`impl_gemm_lut_iq4xs.hip`) + b128 + XOR — `bench_gemm_wmma --runs 10 --json` per-variant median/mean/stddev/p95 + TFLOPS + `speedup_median` vs real DP4A MMQ at `M={128,512,1024,8192}`; `llvm-objdump | grep v_wmma` + VGPR ≤64 + `rocprof lds_bank_conflict 0` (bare-metal, WSL2 blind noted).
- **wave3: 07-04 quilt + build_windows.bat + race.py --repeats 10 interleaved 5-tier llama-bench N=10 + 15× LLM QA + rocprof lds_bank_conflict 0 + calculator VGPR ≤64 + llvm-objdump gates** — vendor winners to `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv,gemm}_iq4xs.cuh` and regenerate `patches/0001-gfx1100-mul-mat-custom.patch` via `git -C llama.cpp diff HEAD` over `bb4caa75` (`git apply --check` PASS WSL2 `/opt/rocm` + Windows `HIP_PATH`, `core.autocrlf=false`); author `build_windows.bat` (`HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja` not `cl`, `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`, builds `build-windows/bin/llama-server.exe` serving `curl http://127.0.0.1:8000/v1/chat/completions → 200` on gfx1100, `≤2` langs); run `race.py --repeats 10` interleaved `A,B,A,B` (not `AAAA BBBB`) across 5-tier `llama-bench` `{512,1024,2048,4096,8192}` (`8192` conditional VRAM preflight `>2GB` + `hipMalloc` probe, `FA`+`GQA` rationale) `N=10` thermal-paired (`hwinfo_daemon 1Hz` + `thermal_watchdog 90C`, `RunStore rows.jsonl` + `CHECKSUMS.sha256`) proving `≥1.10× pp+tg` median + `mean−1σ ≥1.10×` per tier (prior `808→849 pp4096 +5.1% FAILS`); `15×` LLM QA `temp=0` fixed prompt `avg tok/s` + per-run 15-row table; plus `rocprof lds_bank_conflict 0` + calculator `amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16` VGPR ≤64 + `llvm-objdump` `v_wmma`/`v_dot4` gates and `QUAL-01/02 N=10` green; then update `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` + `docs/PUBLICATION.md` + `benchmarks/results/phase7/README.md`.
