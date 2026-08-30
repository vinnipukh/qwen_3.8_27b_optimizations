<!-- refreshed: 2026-08-30 -->
# Codebase Concerns

**Analysis Date:** 2026-08-30

Phase 7 (Hybrid DP4A & WMMA, RE-SCOPED 2026-08-28) honest status: the 2026-08-30 bare-metal WSL2 gfx1100 re-bench fulfilled **0/3 must-haves** (REQ-WIN-07, REQ-PERF-07, REQ-STAT-07). One hardware win exists (real DP4A comparator 87.8 µs vs 548 µs naive = 6.25×) and one microbench pass (GEMM 64×64 P4+XOR M1024 peak 1.89×, median 1.929×) — but every end-to-end gate still FAILs. Nothing below is fabricated; all numbers are quoted from captured hardware JSONs (`kernels/matmul_iq4xs/*.hardware.json`, `benchmarks/results/phase7/llama_bench_*_4tier_N10.json`, docs).

## Tech Debt

**[State documentation drift — STATE.md stale vs 2026-08-30 results]:**
- Issue: `.planning/STATE.md` reports `last_updated: 2026-08-29`, `status: gaps_found`, "Execution has NOT started — awaiting another agent", while `docs/ARCHITECTURE.md` and `docs/PUBLICATION.md` already document the 2026-08-30 hardware N=10 re-bench (0/3 must-haves, GEMV 0.968× FAIL, GEMM 64×64 M1024 1.929×, llama-bench 4-tier all FAIL). `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-VERIFICATION.md` is dated `2026-08-29T14:45:00Z` and its tables (truncated JSON, stub `return false`, "patch 355 lines") describe the pre-fix state; the working tree now has the fixed 356-line patch, the real `can_handle` gate, and a valid 18 880-byte GEMM JSON.
- Files: `.planning/STATE.md`, `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-VERIFICATION.md`, `.planning/ROADMAP.md`
- Impact: A planner/executor reading STATE.md or 07-VERIFICATION.md will re-execute already-completed waves or chase already-fixed blockers (truncation, stub).
- Fix approach: Refresh STATE.md/07-VERIFICATION.md to 2026-08-30 facts before any `/gsd-plan-phase` or `/gsd-execute-phase` run; keep the honest FAIL framing.

**[CHANGELOG "Unreleased" entry is stale]:**
- Issue: `CHANGELOG.md` line 42 (`## [Unreleased] - 2026-08-27 - Phase 7 ...`) still says "Paired `llama-bench` sweep ... documented as simulation on this Windows host ... real hardware execution pending WSL2 gfx1100". Hardware execution happened 2026-08-30 (`benchmarks/results/phase7/llama_bench_*_4tier_N10.json`, `test_time 2026-08-30T17:37-17:40Z`) — all tiers FAIL.
- Files: `CHANGELOG.md`
- Impact: Release notes contradict evidence archive; a reader may believe the benchmark matrix is still unrun.
- Fix approach: Add a Phase 7 `[Unreleased]` entry dated 2026-08-30 with the honest 4-tier FAIL table and 0/3 must-have summary.

**[≤2-language gate FAIL — Python/JS not yet pruned]:**
- Issue: `find . -name "*.py" ! -path "./llama.cpp/*"` returns **40** and `*.mjs` returns **1** (`freetoken-rocm-probe/src/qstar.mjs`). REQ-WIN-07 requires `==0`. Explicitly deferred to Phase 8 (`08-refactor-windows-native/`, not started).
- Files: `benchmarks/` harness, `benchmarks/results/phase7/race.py`, `tools/swizzle_iq4xs.py`, `tools/{ask_model,dump_gguf_fixtures,dump_matmul_fixtures}.py`, `benchmarks/bin/{run_op_gate,run_model_gate}.py`, `output/` python, `freetoken-rocm-probe/src/qstar.mjs`
- Impact: REQ-WIN-07 cannot close; shipped-tree definition (kernels/ + patches/ + build_windows.bat) not yet realized.
- Fix approach: Phase 8 prune per `07-01-PLAN.md` allowlist; archive offline-only helpers (swizzle logic as comment in gemm header) rather than deleting knowledge.

**[Repo hygiene — untracked artifacts and stray files]:**
- Issue: `impl_gemv_dp4a_gfx1100-hip-amdgcn-amd-amdhsa-gfx1100.out` (11 KB) is **not** covered by `.gitignore` (only `*.out.resolution.txt` is); `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip.bak` is a stray backup in-tree; root-level `bench_*.bare.json` (4 files), `.playwright-cli/`, `temp_readme.txt`, `tools/__pycache__/` are untracked; `freetoken-rocm-probe/tools/` vendor is 474 MB on disk (gitignored).
- Files: `.gitignore`, repo root, `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip.bak`
- Impact: `git status` noise; risk of accidentally committing compiler outputs or a `.bak` file; repo bloat on disk.
- Fix approach: Add `*.out`, `*.bare.json`, `*.bak`, `.playwright-cli/` to `.gitignore`; delete the `.bak`.

**[llama.cpp overlay provenance lives only in a gitignored clone]:**
- Issue: `.gitignore` excludes `/llama.cpp`, so the entire vendored tree — including `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv,gemm}_iq4xs.cuh` and the local commit `5c6b397` — is invisible to the repo. The quilt patch `patches/0001-gfx1100-mul-mat-custom.patch` is the only tracked overlay artifact, and it was generated from the **working tree**, not from HEAD: `git -C llama.cpp diff HEAD` shows 2 modified files (`gemm_iq4xs.cuh` 5+/4−, `gemv_iq4xs.cuh` 3+/2−) — the `can_handle` real-gate fix is uncommitted in the clone.
- Files: `patches/0001-gfx1100-mul-mat-custom.patch`, `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/*.cuh`, `.gitignore`
- Impact: If the clone is re-cloned from `bb4caa75` and the patch applied, behavior matches the working tree (good), but the local HEAD commit diverges from what was verified; a future `git -C llama.cpp diff bb4caa75` would silently resync the patch. The patch itself is correct (356 lines, real `can_handle` gate at patch lines 148-150; `empty.cuh` stubs at lines 38/48 are the intended OFF-mode fallback).
- Fix approach: Commit the two `.cuh` changes in the llama.cpp clone (`git -C llama.cpp add ... && git -C llama.cpp commit`) or regenerate the patch from a fresh pinned checkout and record the clone's HEAD sha in `docs/PUBLICATION.md`.

## Known Bugs

**[LUT μ=4 M128 "62.405× win" is an anomalous measurement artifact in official evidence]:**
- Symptoms: `kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json` entry `LUT_mu4 M=128 speedup_median 62.405, winner "wmma_stream"` while every other variant at M128 measures 0.040–0.042×. The 12 µs LUT entry was flagged in docs as "anomalous partial capture, not claimed" (`docs/ARCHITECTURE.md` §7.02), but the JSON itself — the machine-readable evidence — carries a 62× win with no anomaly flag.
- Files: `kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json` (LUT_mu4/M128 entry)
- Trigger: Any consumer parsing the JSON for the M128 winner.
- Workaround: Docs correctly exclude the claim. Fix: add an `"anomalous": true, "note": "partial capture"` field or regenerate that entry; do not silently drop.

**[Synthetic `rows.jsonl` masquerades as RunStore evidence]:**
- Symptoms: `benchmarks/results/phase7/rows.jsonl` has 250 lines with co-millisecond future timestamps (`1788106278.3768xx` cluster = 2026-08-30T17:31Z, identical ms) and uniform random jitter — batch-generated by `race.py`, median 1.03–1.08× FAIL. Correctly labeled synthetic in prose docs, but there is no machine-readable marker in the file, and `CHECKSUMS.sha256` hashes rows.jsonl only (single 8d6a943a… entry) — not the llama-bench or hardware JSONs that the "RunStore + CHECKSUMS over all rows" protocol requires.
- Files: `benchmarks/results/phase7/rows.jsonl`, `benchmarks/results/phase7/CHECKSUMS.sha256`, `benchmarks/results/phase7/race.py`
- Trigger: `sha256sum -c CHECKSUMS.sha256` passes while the underlying rows are not hardware.
- Workaround: Docs disclaim it. Fix: (1) replace rows.jsonl with real `llama-bench -r 10` rows from the Aug 30 run, (2) rewrite CHECKSUMS.sha256 over rows.jsonl **and** `llama_bench_*_4tier_N10.json` **and** `kernels/matmul_iq4xs/*.hardware.json`, (3) add `"synthetic": true` gating in race.py output.

**[Thermal pairing is not evidenced — hwinfo.log has no sensor readings]:**
- Symptoms: `benchmarks/results/phase7/hwinfo.log` contains exactly 4 lines of `{"hwinfo_daemon": "1Hz", "thermal": "polling"}` with **no temperature/power fields** (HWiNFO SHM access denied: WinError5, same as Phase 6); `benchmarks/results/phase7/telemetry/` is empty; `llama_bench_{stock,custom}.log` contain 0 temperature mentions.
- Files: `benchmarks/results/phase7/hwinfo.log`, `benchmarks/results/phase7/telemetry/`
- Trigger: Any claim that the 2026-08-30 4-tier N=10 run was "thermal-paired in one window".
- Workaround: PUBLICATION wording is technically accurate ("daemon polling confirmed") but overstates pairing. Fix: capture real thermal CSV on bare-metal Windows/HWiNFO or state plainly that thermal data was not acquired.

**[`logs/gate.log` shows a failed/aborted gate run]:**
- Symptoms: `logs/gate.log` ends with `RUNTIME-GATE-EXIT=130` (128+SIGINT) and an empty offload line; `logs/build.log` ends with `BUILD-EXIT=2` on `build-ci` targets.
- Files: `logs/gate.log`, `logs/build.log`
- Trigger: Re-running the CI gate targets unguarded will fail the same way; exit 130 means the harness was interrupted, exit 2 is a build failure.
- Workaround: None recorded. Fix: document or delete these logs; re-run gates properly per `07-03-PLAN.md` QUAL-01/QUAL-02 N=10.

**[Patch/link correctness — 8192 tier never recorded as SKIPPED in llama-bench evidence]:**
- Symptoms: the plan (`07-02-PLAN.md`) requires 8192 entries recorded as `"SKIPPED"` with FA+GQA rationale. The Aug 30 `llama_bench_*_4tier_N10.json` files simply omit 8192 (5 entries: 512/1024/2048/4096 pp + tg128) with no SKIPPED row. The microbench path (`bench_gemm_wmma.cpp` lines 66-118) does emit SKIPPED JSON with preflight notes when VRAM preflight/hipMalloc fails.
- Files: `benchmarks/results/phase7/llama_bench_stock_4tier_N10.json`, `kernels/matmul_iq4xs/bench_gemm_wmma.cpp`
- Fix approach: Emit an explicit 8192 SKIPPED record (with `hipMemGetInfo`/probe result) into the RunStore so the conditional-skip clause is evidenced, not just absent.

## Security Considerations

**[WSL2 root execution + OOM→BSOD host stability risk]:**
- Risk: All benches run as `-u root` in WSL2 (`docs/PUBLICATION.md` line 37) with HSA DXG passthrough; 3–5 OOMs at 8192 VRAM pressure cause host **BSOD** (microsoft/WSL#40732; `fix-p10-thermal.md`). WSL2 lies about free memory (reports ~800 GiB vs 3.48 GiB contiguous).
- Files: `fix-p10-thermal.md`, `benchmarks/results/phase7/race.py`, `kernels/matmul_iq4xs/bench_gemm_wmma.cpp` (VRAM preflight lines 66-118)
- Current mitigation: VRAM preflight `hipMemGetInfo >2GB free` + 10 MiB `hipMalloc` probe before any 8192 alloc, fail-fast no retry (`hip_helpers.h` soft-HIP_CHECK).
- Recommendations: Keep 8192 hard-gated as planned; consider `timeout 90` + `hipDeviceReset` between tiers; document the BSOD recovery procedure in the RUNBOOK.

**[External surface is minimal but the smoke endpoint is a live model server]:**
- Risk: `build_windows.bat` (when executed) starts `llama-server.exe` on `127.0.0.1:8000` serving `/v1/chat/completions`. Localhost-only binding is correct, but the batch file places no auth on the endpoint; any local process can query the model.
- Files: `build_windows.bat`
- Current mitigation: `--host 127.0.0.1` binding.
- Recommendations: Document that the server is for smoke testing only; no secrets/credentials were found in the repo (no `.env` files detected; `models/*.gguf` gitignored).

**[Vendored toolchain supply chain]:**
- Risk: `freetoken-rocm-probe/tools/` vendors a zig 0.16.0 toolchain (~474 MB) in the working tree (gitignored). No pinned checksum recorded.
- Files: `freetoken-rocm-probe/tools/`
- Recommendation: Delete the vendor tree or pin+checksum it; it is unrelated to the shipped kernels.

## Performance Bottlenecks

**[REQ-PERF-07 end-to-end gate FAILs at every tier (HONEST, N=10 hardware 2026-08-30)]:**
- Problem: Paired `llama-bench` A/B, stock OFF vs custom ON, `-r 10`, one window (`benchmarks/results/phase7/llama_bench_{stock,custom}_4tier_N10.json`):
  - pp512: 838.27±185.69 → 904.50±36.92 = **1.079× FAIL** (mean−1σ 0.847)
  - pp1024: 918.49±51.16 → 914.75±46.74 = **0.996× FAIL** (0.895)
  - pp2048: 878.62±106.13 → 880.90±33.93 = **1.003× FAIL** (0.860)
  - pp4096: 871.06±68.76 → 851.93±69.33 = **0.978× FAIL** (0.833)
  - tg128: 34.80±2.80 → 34.56±2.55 = **0.993× FAIL** (0.851)
- Files: `benchmarks/results/phase7/llama_bench_stock_4tier_N10.json`, `llama_bench_custom_4tier_N10.json`, `docs/PUBLICATION.md` §8
- Cause: GEMV avg 0.968 (+33) / 0.976 (XOR) FAIL <1.2×; GEMM M512 avg 0.70, M1024 avg 1.08 (only 64×64 P4+XOR median 1.929× at M1024, 1.208× at M512); decode path drags tg.
- Improvement path: 64×64 B-stationary sweep showed the winning direction; P=4 XOR + offline 16×64 swizzle + b128 + LUT μ=4 are the documented levers (`07-02-PLAN.md`); prior best synthetic 1.08 <1.10 means 64×32 P2 alone is insufficient — needs the 64×64/128×32 family hardened and GEMM dispatch on.

**[GEMV decode uplift flattened by WSL2 DXG jitter]:**
- Problem: `bench_gemv_dp4a.hardware.json` speedup_median per shape 0.850–1.148, avg **0.968**; `bench_gemv_xor.hardware.json` avg **0.976**, peak 1.161. Per-shape mean−1σ 0.42–0.58. DXG 15–30 µs jitter (p95 148–343 µs) flattens the earlier 1.178× peak to ~1.00.
- Files: `kernels/matmul_iq4xs/bench_gemv_dp4a.hardware.json`, `bench_gemv_xor.hardware.json`, `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`
- Cause: Virtualization jitter + 256→32 row cooperative layout (16 waves) under WSL2; bare-metal 16-waves and XOR as a **compiled second OBJECT** (currently only the `+33` object is built; XOR is a `#ifdef GEMV_XOR` helper `gemv_variant_xor.cuh`) are pending.
- Improvement path: Interleave A,B,A,B `race.py --repeats 10` on bare-metal; compile XOR as `matmul_gemv_dp4a_xor_hip` OBJECT; VGPR ≤64 audit (`--save-temps -Rpass-analysis`).

**[Small-M GEMM is catastrophically slow AND the real `can_handle` gate now routes M≥16 into it]:**
- Problem: WMMA family at M=128 measures **0.040–0.042×** (17 619 µs vs stock 736 µs — 24× slower); the tiled fallback `launch_stream_tiled_gemm_cuh` is the M<512 path. The restored real gate in `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` is `type==IQ4_XS && M>=16 && K%256==0 && N%16==0` — i.e. any prefill batch M∈[16,511] that passes shape checks now **dispatches to the ~24×-slower custom path** in production llama.cpp ON mode.

  **This is a latent end-to-end regression risk for small-batch prefill** (e.g., M=128 prompts) once the patch is enabled, exactly the region `bench_gemm_wmma.hardware.json` measures as 0.04×.
- Files: `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` (lines 87-90 / patch lines 148-150), `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` (`wmma_ok` M≥512 gate), `kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json`
- Cause: `can_handle` gate is broader than the fast path's `wmma_ok (M%16==0 && M>=512)` condition; docs' "tiled proven 1.47–7.39× vs naive" is vs the **naive** comparator, not vs real stock DP4A.
- Fix approach: Tighten `custom_gemm_iq4xs_can_handle` to require `M>=512` (or make the tiled M<512 path competitive vs real stock first), and re-bench M=128/256 in llama-bench A/B before enabling the patch.

**[Thermal throttling contaminates N=10 samples]:**
- Problem: pp512 stock sample contains a 317.7 t/s outlier vs 840–945 for the other 9 (σ=185.7); pp2048 stock range 584–947. No thermal data captured (see Known Bugs) so throttling cannot be separated from DXG jitter.
- Files: `benchmarks/results/phase7/llama_bench_stock_4tier_N10.json`
- Fix approach: Real HWiNFO 1 Hz CSV + watchdog kills (planned `hwinfo_daemon`/`thermal_watchdog`), plus interleaving; report trimmed stats if an outlier is thermal.

## Fragile Areas

**[Five-variant single-TU weak-ODR pattern]:**
- Files: `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` (variant macro + `__attribute__((weak)) hipError_t gemm_iq4xs_stream_tiled_gpu(...)` at line 353), CMake OBJECT targets `matmul_gemm_wmma_{stream,p4_xor,64x64,lut}_hip` + 128x32 via M-switch
- Why fragile: 5 OBJECTs each compile the same TU with a variant macro; each emits a **weak** `gemm_iq4xs_stream_tiled_gpu` plus variant-specific strong `gemm_iq4xs_wmma_*_gpu` symbols. If a bench binary links several OBJECTs, all weak definitions collapse to whichever the linker picks first — silent aliasing. The pattern is called "ODR-safe" in `docs/ARCHITECTURE.md`, but weak linkage *masks* ODR collisions rather than preventing them; any future divergence between the 5 copies (tuning, banking) silently changes which code actually runs. No `nm`/link-map audit is recorded to prove single-definition resolution per binary.
- Safe modification: Keep the tiled helper byte-identical across variants, or replace weak with `inline`/`static` per TU and (better) rename per variant with `WMMA_GPU_NAME`-style suffixing; add a link-map or `nm` gate to CI.
- Test coverage: `kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp` tests one compiled variant; no per-variant correctness gate for 64×64/LUT/128x32 in the tree.

**[Evidence JSON pipeline (streaming + timeout)]:**
- Files: `kernels/matmul_iq4xs/bench_gemm_wmma.cpp` (incremental `fprintf`+`fflush` per entry, lines 66-235; jitter multiplication removed — line 8 comment confirms)
- Why fragile: The 12 288 B truncation bug (Aug 29 capture) was fixed by streaming, but the 271 s DXG deadlock still requires a `timeout 90` wrapper; a hung hipEvent still yields a partial (valid) JSON whose tail may be incomplete; consumers must validate `python -m json.tool` and treat incomplete tails as SKIPPED, per `07-02-PLAN.md` T-07-02-02.

**[WSL2-only development loop]:**
- Files: `build_windows.bat`, `kernels/CMakeLists.txt` (line 17 HIP_PATH search), `.wslconfig` requirements (`memory=28GB`, DXG ENOMEM below 20 GB)
- Why fragile: All hardware benches run in WSL2 (root); rocprofv3 is Instinct-only/blind on WSL2 (404), HWiNFO SHM is access-denied (WinError5), DrvFs git locks deadlock builds (llama.cpp must live on ext4 `/root/...`). Windows-native evidence (REQ-WIN-07) is entirely unproven — no HIP SDK at `C:/Program Files/AMD/ROCm/6.4`, no `build-windows/`, no `llama-server.exe`, no `curl :8000 → 200`.

## Scaling Limits

**[8192-token context is at a hard VRAM cliff]:**
- Current capacity: 20 GiB VRAM; fully-resident model 15.31 GB (`JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, sha256 `53adc4bb…`) leaves ~4.7 GB for KV. GQA KV ≈ 128 KiB/token → 8192 tokens ≈ 18.5 GB total, over budget once fragmentation counts.
- Limit: 8192 tier is **almost certainly unbenchable** on this card; FA + GQA required even to approach it.
- Scaling path: FA + GQA enabled llama.cpp upsteam kernels (not custom), `-c 8192` conditional on VRAM preflight >2 GB free + 10 MiB `hipMalloc` probe; record explicit SKIPPED rows (currently only documented, not recorded in RunStore). Custom kernels are not the gating factor at 8192 — VRAM is.

**[N=10/N=15 statistical rigour is harness-complete but hardware-incomplete]:**
- Current: microbench N=10 exists (`bench_real_stock` 87.8±? / `bench_gemv*` / `bench_gemm*` all `runs:10`); llama-bench N=10 exists for 4 tiers+tg. **N=15 LLM QA (`llama-cli --temp 0 -n 128` ×15, per-run table) has never been run.** Benchmark samples confirm real variance (pp512 σ≈186), so single-run claims are genuinely banned.
- Files: `benchmarks/results/phase7/race.py`, `KERNEL-BENCH-DIFF.md` §8, `07-03-PLAN.md`
- Scaling path: One bare-metal window (~45–60 min) per the RUNBOOK to close REQ-STAT-07.

## Dependencies at Risk

**[Windows HIP SDK — the load-bearing uninstalled dependency]:**
- Risk: REQ-WIN-07 rests entirely on `build_windows.bat` using `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja` (MSVC `cl` cannot compile `__builtin_amdgcn_*`). The host has **no HIP SDK** (`C:/Program Files/AMD/ROCm/6.4` absent), so the batch file has never executed and `build-windows/bin/llama-server.exe` does not exist. Code-level review passes (HIP_PATH, clang++ probe, `where ninja` guard, `find_package(hip PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`, curl 200 smoke with MODEL_PATH guard) — nothing at runtime is proven.
- Impact: 1 of 3 must-haves cannot close until hardware+SDK is provisioned.
- Migration plan: Install AMD HIP SDK 6.4 + VS Build Tools + Ninja on Windows 11; run the bat; record `curl_code.txt`/`curl_out.json`. Keep the `--load-mode none`/`--single-turn` llama-cli flags in mind for headless runs (PTY stall issue noted in the phase-6 baseline).

**[ROCm/adrenalin/llama.cpp frozen-config fragility]:**
- Risk: Frozen lock (Adrenalin 26.2.2, ROCm 7.2.1 librocdxg 1.2.2, llama.cpp `bb4caa75`, GGUF sha256 `53adc4bb…`) means any environment drift (driver update, ROCm bump, model swap) invalidates the whole evidence chain; the overlay depends on exact line numbers in `mmq.cu`/`mmvq.cu` guards.
- Impact: Re-baselining after any upgrade requires repeating 6 phases of gates.
- Migration plan: Reproduce from `baseline/binaries/v0.2.0-bb4caa75/` + patch; keep the `.wslconfig` snapshot `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar`.

**[HWiNFO SHM + rocprofv3 — degraded telemetry tooling]:**
- Risk: HWiNFO access denied (WinError5) on the WSL2 host → the 1 Hz thermal daemon logs only "polling" placeholders; rocprofv3 is Instinct-only under librocdxg → `lds_bank_conflict 0` gate cannot be measured on this hardware; proxy audits (+33 vs XOR, llvm-objdump `v_wmma`/`v_dot4`, VGPR ≤64 calculator) are the only available substitutes.
- Impact: Thermal-pairing and bank-conflict claims stay unverifiable on WSL2; bare-metal Linux or Windows HWiNFO needed.

## Missing Critical Features

**[Windows-native build & smoke (REQ-WIN-07) — unexecuted]:**
- Problem: `build_windows.bat` exists (5857 B, code-correct) but was never run; no `llama-server.exe`, no `curl http://127.0.0.1:8000/v1/chat/completions → 200`.
- Blocks: 1/3 must-have; the entire "beats stock on Windows-native ≤2 langs" project value proposition.
- Path: Phase-8 or 07-01 wave on a Windows 11 + HIP SDK box.

**[QUAL-01/QUAL-02 gates not executed on custom build at N=10]:**
- Problem: stock op-gate 4243 PASS exists from Phase 6; custom-ON op/model gates (`run_op_gate.py --runs 10`, `run_model_gate.py --runs 10`, PPL 6.4271±1% + 6/6 canaries) were never run on hardware; `logs/gate.log` shows exit 130.
- Blocks: Trust that the integrated patch does not corrupt numerics end-to-end (microbench cosine gates alone don't cover llama.cpp integration).

**[Explicit 8192 SKIPPED evidence + N=15 LLM QA (REQ-STAT-07)]:**
- Problem: No recorded 8192 SKIPPED row in RunStore; no 15-row `temp=0` QA table; no thermal CSV.
- Blocks: 2/3 + 3/3 must-haves.

## Test Coverage Gaps

**[Untested: per-variant GEMM correctness (64×64, LUT μ=4, 128×32)]:**
- What's not tested: `test_gemm_wmma_compare.cpp` exercises one WMMA variant; the 64×64 P4+XOR winner (median 1.929× at M1024) and LUT μ=4 have no cosine gate in the tree. The LUT M128 62.4× anomaly means the LUT path deserves a correctness gate most of all.
- Files: `kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp`, `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip`, `kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip`
- Risk: A numerically wrong variant could be picked as race winner (LUT already produces a suspicious measurement).
- Priority: High

**[Untested: `M∈[16,511]` tiled fallback vs real stock in production dispatch]:**
- What's not tested: the `can_handle` gate now accepts M≥16 and routes small prefill to a path measured 0.04× at M128. No llama-bench A/B at n_prompt 128/256 with the patch ON.
- Files: `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh`, `bench_gemm_wmma.hardware.json` (M128 rows)
- Risk: End-to-end regression on small-batch prefill once the patch is enabled.
- Priority: High

**[Untested: XOR GEMV as a compiled variant object]:**
- What's not tested: `gemv_variant_xor.cuh` helpers are `#ifdef GEMV_XOR`-gated, not a second OBJECT (`matmul_gemv_dp4a_xor_hip` missing per 07-VERIFICATION); the XOR bench JSON (`bench_gemv_xor.hardware.json`) exists but no correctness test for the XOR layout.
- Files: `kernels/matmul_iq4xs/gemv_variant_xor.cuh`, `kernels/matmul_iq4xs/CMakeLists.txt`
- Risk: Winner selection incomplete; XOR could be the bare-metal winner but is not independently verifiable.
- Priority: Medium

**[Untested: Windows-native toolchain path]:**
- What's not tested: `clang++.exe --offload-arch=gfx1100` on Windows HIP SDK, Ninja generator, `find_package(hip)` via `HIP_PATH`, `llama-server.exe` smoke. Nothing runs until SDK is installed.
- Files: `build_windows.bat`
- Risk: REQ-WIN-07 unverifiable; unknown clang++-on-Windows behavior for `__builtin_amdgcn_*`.
- Priority: High (blocking must-have)

**[Stale claims that read as current]:**
- What's at risk: `CHANGELOG.md` v1.0.0 entries ("1.26–2.13× speedup (8/8 wins) over stock HIP", "1.76–7.50× over naive scalar baseline") and `KERNEL-BENCH-DIFF.md` §2 table were measured vs the **naive** comparator on 2026-08-25; Phase 7 vs real DP4A shows GEMV 0.968× avg. §8 correctly reframes with HONEST FAIL, but §2/§3 headers still headline the old wins.
- Files: `CHANGELOG.md`, `benchmarks/profiling/KERNEL-BENCH-DIFF.md` §2-3 vs §8
- Risk: A reader quotes 2× GEMV or 7.5× GEMM as current; contradicts the honest-evidence culture the repo otherwise maintains.
- Priority: Medium — add a "superseded by Phase 7 §8 (vs real DP4A)" banner on §2-3.

---

*Concerns audit: 2026-08-30*