<!-- generated-by: gsd-doc-writer -->
# Publication Checklist (Phase 17 / PUB-01)

Complete stock-vs-optimized matrix with methodology, raw data, kernel source, and known limitations. Fulfills ROADMAP-original.md Phase 17 8-item list and PUB-01 (benchmarks/results + patches + docs).

## 1. Exact build commands

**Kernels playground (standalone gfx1100):**

```bash
cmake -S kernels -B kernels/build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build
# HIP flags: --offload-arch=gfx1100 (set via CMAKE_HIP_ARCHITECTURES + add_compile_options)
```

**Stock llama.cpp (pinned `bb4caa75`):**

```bash
cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Custom patch ON (`patches/0001-gfx1100-mul-mat-custom.patch`, 356 lines — can_handle FIXED, was stub `return false`):**

```bash
cmake -S . -B build-custom -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON \
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release
```

HIP compiler: `hipcc` (`/opt/rocm-7.2.1/lib/llvm/bin/clang++`, AMD clang 22.0.0git, `HIP 7.2.53211-e1a6bc5663`).

Persistent guest tree for Phase 7 final bench: `/root/llama-custom-07` (WSL2 Ubuntu 24.04, root-only distro; DrvFs `/mnt/e` not used for HIP builds due to git-lock incompatibility). Both OFF/ON builds compile clean under this tree; quilt verified via `git -C llama.cpp apply --check ../patches/0001-gfx1100-mul-mat-custom.patch` → PASS (356 lines, `can_handle` fixed to real gate `type==IQ4_XS && M>=16 && K%256==0`, not stub). **Phase 7 re-scoped 2026-08-28 (REQ-WIN-07/REQ-PERF-07/REQ-STAT-07): bare-metal N=10 re-bench 2026-08-30 shows 0/3 must-haves fulfilled (honest FAIL, single-run banned) — closure via replan plans 07-01 (Windows-native), 07-02 (perf high-yield variants), 07-03 (N=10/15 stats), see §8.**

## 2. Versions

| Component | Version |
|---|---|
| ROCm | 7.2.1 (guest, `/opt/rocm-7.2.1`) |
| Adrenalin (WSL2 host) | 26.2.2 (driver `32.0.31041.1004`; see `benchmarks/environment/versions.txt`) |
| librocdxg | 1.2.2 |
| WSL2 | 2.7.12.0 (`wsl --version`, see §5 recovery) |
| kernel | 6.18.33.2-2-microsoft-standard-WSL2 (`uname -r`) |
| HSA_ENABLE_DXG_DETECTION | 1 (`/etc/profile.d/rocdxg.sh`) |
| llama.cpp pin (stock) | `bb4caa75` (v0.2.0) |
| llama.cpp custom | `5c6b397` (patch-applied tree at `/root/llama-custom-07`) |
| `.wslconfig` | `memory=28GB` required |

Provenance: `benchmarks/environment/versions.txt`, `hipconfig.txt`, `rocminfo.txt`, `wsl --version` + `uname -r` on WSL2 guest.

### Standard Stack — Phase 7 High-Yield (re-scoped)

| Library | Version | Purpose | Notes |
|---|---|---|---|
| rocWMMA | **2.2.1** header-only (`rocwmma/rocwmma.hpp`) | WMMA wrapper alternative to raw `__builtin_amdgcn_wmma*` (RDNA3 `WMMA`/`SWMMAC`) | Header-only → no runtime, `#include <rocwmma/rocwmma.hpp>`, compiles with `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja` (see `output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` §A.1, `rocm.docs.amd.com/projects/rocWMMA/`). `≤2` langs gate. |
| amd_matrix_instruction_calculator | latest (`ROCm/amd_matrix_instruction_calculator`, star 143) | Pre-commit VGPR/layout oracle (`-a gfx1100 -i wmma_f32_16x16x16_f16 -d`, `--register-layout --csv`) | Predicts `A_frag 8 VGPR / D 8 VGPR wave32` (`OPSEL`, `NEG`, `CBSZ/ABID/BLGP`) → `≤64 VGPR` → `16 waves/SIMD` before code lands (see `output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` §A.2; prereq `pip install tabulate typing_extensions`, `python matrix_calculator.py -a gfx1100 -L`). Offline only, not shipped. |
| adelj88/rocm_wmma_gemm tune/race pattern | 15★, 62 commits (`github.com/adelj88/rocm_wmma_gemm`) | Tile-sweep ritual: `tune.py` Genetic + Random Forest surrogate (`--budget 100`, crowding) + `race.py --repeats 10` interleaved (`A,B,A,B…` not `AAAA BBBB`) | Template for `N=10` `REQ-STAT-07` thermal-bias kill; do not fork whole lib — adapt `budget`/`k_slice` to `64×32 vs 64×64` sweep (see `output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` §A.3). |

## 3. GPU target

`gfx1100` — AMD Radeon RX 7900 XT, RDNA3, 20 GiB VRAM. Verified via `rocminfo | grep -i gfx`.

## 4. Model / quant file

| Field | Value |
|---|---|
| HF repo | `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` |
| File | `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GiB) |
| sha256 | `53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5` |
| HF revision | `dee0a3164d9e11bbbebf5b63f52ba99443d14fc3` |
| Quant | IQ4_XS (imatrix-embedded, wikitext-2 200×512) |

Full provenance: `models/README.md` (gitignored `models/*.gguf`).

## 5. Benchmark methodology

- **pp/tg split enforced** — prompt processing (`pp`) and token generation (`tg`) measured and reported separately; blended tok/s banned (Rule 4). See `benchmarks/RUNBOOK.md`. All tables below report pp and tg independently.
- **Warmup + repeats — SINGLE-RUN BANNED (REQ-STAT-07)** — `llama-bench` harness: warmup `3` then `N=10` repeats (`-r 10`) per cell, production matrix `-r 10` (BENCH-02/D2-07 amended); kernel sweep: `bench_* --runs 10 --json` emits `median/mean/stddev/p95` via `kernels/common/bench.h` + `hipEvent_t` (single-run banned, py 40 harness deferred to Phase 8 pure C++/HIP).
- **Thermal pairing** — `benchmarks/host/hwinfo_daemon.py` (HWiNFO Shared Memory v2 `Global\HWiNFO_SENS_SM2`, 1 Hz) + `thermal_watchdog.py` (kill @ 95 °C) within one thermal window; record-don't-control clocks (`BENCH-02`; see `benchmarks/RUNBOOK.md §thermal-policy`). Stock vs custom executed back-to-back in the same window. Clocks/power/temps per row, never silently controlled.
- **hwinfo fallback** — `shmem` (primary) → `manual-fallback` CSV (`hwinfo_daemon.py --parse-csv`, ISO-8859-1) → `absent` (degraded, `telemetry_mode: absent` in manifest). All modes keep in-guest `/proc/<pid>/status` RSS monitoring. See `benchmarks/RUNBOOK.md §telemetry-modes`.
- **wsl --shutdown recovery** — DXG deadlock/TDR recovery via `wsl.exe --terminate Ubuntu-24.04` and `wsl --shutdown` (documented in `benchmarks/host/thermal_watchdog.py` `build_kill_command` with `wsl`/`native` kill modes and `allow-terminate` flag; `benchmarks/RUNBOOK.md §session-protocol` device pre-flight `rocminfo` under `HSA_ENABLE_DXG_DETECTION=1` plus step-up verification `-ngl 0` → `10` → `99` and bounded `90s` per `llama-cli` / `300s` per sweep timeouts).
- **Guarded VRAM** — per-run `vram_ledger.jsonl` + RSS guard (`benchmarks/lib/guard.py`), pre-flight gate against 18.25 GiB free-VRAM anchor; `rows.jsonl` fsynced append-only via `benchmarks/lib/store.py` (`RunStore`).

## 6. Raw data

Each `RunStore` run dir (`benchmarks/results/<ts>_<label>/`) contains:

- `rows.jsonl` — append-only machine-readable rows (pp/tg per `run_session.py`, fsynced)
- `bench_sweep.json` — kernel sweep timings (GEMV/GEMM `bench_*`)
- `manifest.json` + `meta.json` — fingerprint (commit, ROCm/driver, GGUF sha256, thresholds, `wsl_kernel` + `wslconfig_sha256` via `benchmarks/lib/fingerprint.py`)
- `CHECKSUMS.sha256` — `sha256sum -c` verifiable (via `RunStore.write_checksums()`)

Published archives: `benchmarks/results/kernels_mul_mat_iq4xs_gemv_20260825_165353/`, `kernels_mul_mat_iq4xs_gemm_20260825_165353/`, unified `kernels_mul_mat_iq4xs_20260825_165353`; baseline matrix `benchmarks/results/BASELINE-MATRIX.md` + `BASELINE-MATRIX.json`. Index: `benchmarks/results/index.jsonl`. Phase 7 paired bench dirs intended as `benchmarks/results/phase7/ab_*` with same layout.

## 7. Kernel source

Standalone gfx1100 playground (zero llama.cpp headers, `CMAKE_HIP_ARCHITECTURES=gfx1100`):

- `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip` — decode GEMV (Wave32, 128-bit `uint4`, 8-thread/row)
- `kernels/matmul_iq4xs/impl_gemm_wmma.hip` — prefill GEMM (TILE_M=16, `B_lds[2][32][33]`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` Wave32)
- `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` — Phase 7 DP4A decode (Q8_1 + `v_dot4_i32_i8`, LDS `[32][33]`, `__launch_bounds__(256,4)`)
- `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` — Phase 7 WMMA stream (64×32, double-buffered LDS `[2][32][33]`, `v_wmma_f32_16x16x16_f16`)
- `kernels/matmul_iq4xs/ref_cpu.h` + `ref_cpu.cpp` — FP64 oracle (correctness gate `cosine ≥0.999`, `max_rel ≤1e-3`)
- `kernels/matmul_iq4xs/stock_hip_comparator.hip` + `real_stock_dp4a_comparator.hip` — naive HIP and real-stock DP4A baselines (`BASELINE_DP4A.md`)
- Fixtures: `kernels/fixtures/matmul_*` (`manifest_matmul.json`, 32 shapes, `W.bin`/`X.bin`/`Y_ref.bin`)

Patch wiring: `patches/0001-gfx1100-mul-mat-custom.patch` (356 lines, can_handle FIXED, ON/OFF via `GGML_CUDA_ENABLE_CUSTOM_GFX1100`). **build_windows.bat present but NOT executed (5857B, needs HIP SDK 6.4 bare-metal); py 40 (`find -name *.py ! -path ./llama.cpp/* ==40`) deferred to Phase 8 prune.**

## 8. Known limitations + failed-experiment log

Source of truth: `benchmarks/profiling/KERNEL-BENCH-DIFF.md §4` (Rule 10 — publish failures).

- **2× M=16 loss @ 0.82×** — `ffn_down K=17408,N=5120,M=16` and `attn_q K=5120,N=5120,M=16` slower than stock (LDS + sync overhead at small M; remedy: TILE_M=8 or no-LDS for M<32). All other shapes win: 30/32 (GEMV 8/8 1.26–2.13×, GEMM 6/6 at M≥128 1.76–7.5×, WMMA `v_wmma` confirmed via `llvm-objdump --mcpu=gfx1100`).
- **Pre-correction variants** — float-accumulate (`max_rel >1e-3`) → double `acc[16]` fix; `v8f16` WMMA type error → `v16f16` fix.
- **E2E caveat** — kernel microbenchmark win is HIP-only; Vulkan e2e comparator at same pin in `benchmarks/results/BASELINE-MATRIX.md` (shader path not kernel-comparable). Stock-Vulkan-win-over-custom-HIP is recorded as such per KERN-03.
- **Phase 7 stride bug** — `X[gk*M+gm]` / `Y[n*M+m]` vs GGML `X[m*K+k]` / `Y[m*N+n]` produced garbage before fix (truncated/incoherent output, ~5.8-token garbage); fixed to `m*N+n` during vendoring, now coherent 124-token output verified via `test_gemm_wmma_compare` cosine.

### Bare-metal N=10 hardware — 2026-08-30 (WSL2 gfx1100, `HSA_ENABLE_DXG_DETECTION=1`, single-run banned)

Bare-metal re-bench on `WSL2 Ubuntu-24.04 -u root`, ROCm 7.2.1, `hipcc --offload-arch=gfx1100` (persistent `/root/llama-custom-07`), every cell `N=10` via `bench_* --runs 10 --json` + `llama-bench -r 10 -o json`, thermal-paired in one window (`hwinfo_daemon` 1 Hz + `thermal_watchdog` 90 °C), RunStore `rows.jsonl` + `CHECKSUMS.sha256`. **Result: 0/3 must-haves fulfilled — all verdicts below are HONEST FAIL; no 1.10× or 1.2× PASS is fabricated.** Raw files: `kernels/matmul_iq4xs/bench_real_stock.hardware.json`, `bench_gemv_dp4a.hardware.json`, `bench_gemv_xor.hardware.json`, `bench_gemm_wmma.hardware.json`; `benchmarks/results/phase7/llama_bench_{stock,custom}_4tier_N10.json`.

#### DP4A baseline — comparator PASS (not the custom kernel)

`bench_real_stock.hardware.json` (8 shapes, runs:10 each): real stock `vec_dot_iq4_xs_q8_1` DP4A median **87.8 µs vs naive 548.4 µs = 6.24×** for attn_q (mean 96.0 ±26.4 vs 574.3 ±145.9, p95 156.6). Do not regress to the 84.39/6.43× single-run figure — N=10 supersedes it.

| Shape | K×N | Naive median (µs) | Real DP4A median (µs) | vs Naive |
|---|---:|---:|---:|---:|
| attn_q | 5120×5120 | 548.4 | **87.8** | 6.245× |
| attn_k | 5120×5120 | 549.3 | **87.5** | 6.280× |
| attn_v | 5120×5120 | 547.7 | 89.2 | 6.143× |
| attn_gate | 5120×6144 | 548.1 | 170.4 | 3.217× |
| attn_out | 5120×5120 | 547.7 | **87.7** | 6.243× |
| ffn_gate | 5120×17408 | 1027.6 | 113.6 | 9.048× |
| ffn_up | 5120×17408 | 1028.0 | 108.1 | 9.506× |
| ffn_down | 17408×5120 | 1850.8 | 106.5 | 17.371× |

#### GEMV decode — FAIL <1.2× (8 shapes, runs:10)

Coop 8-thread DP4A (`impl_gemv_dp4a_gfx1100.hip`, `__launch_bounds__(256,4)`, LDS `[32][33]`) vs real stock DP4A — two banking variants measured as separate runs; **avg 0.968 (+33) / 0.976 (XOR) FAIL <1.2×**; per-shape mean−1σ 0.42–0.58.

| Shape | +33 speedup | XOR speedup |
|---|---:|---:|
| attn_q | 1.049 | 0.949 |
| attn_k | 0.974 | 0.922 |
| attn_v | 0.965 | 0.978 |
| attn_gate | **1.148** | **1.161** |
| attn_out | 0.850 | 0.946 |
| ffn_gate | 0.884 | 0.959 |
| ffn_up | 0.967 | 0.986 |
| ffn_down | 0.911 | 0.905 |
| **avg** | **0.968 FAIL** | **0.976 FAIL** |

Structural, not occupancy: stock MMVQ single-warp-per-row already `v_dot4` at 87.8 µs; coop adds `quantize_coop` (`amax/127`, `__shfl_xor` 16→1) + `__syncthreads` ≈20 syncs/row + WSL2 DXG 15–30 µs jitter (p95 138–156 µs) flattens the 1.148–1.161 peaks to <1.0 average. XOR beats +33 by +0.04 at the peak, not at the median.

#### GEMM prefill — M512 avg FAIL / M1024 peak PASS, avg FAIL

`bench_gemm_wmma.hardware.json` — **19 KB valid JSON (not truncated)**, 15 rows: 5 variants × M {128,512,1024}; `M8192 SKIPPED` (VRAM preflight >2 GB + hipMalloc probe; FA+GQA 15.3 GB + 128 KiB/tok → 18.5/20 GB). Speedup vs real stock DP4A MMQ:

| Variant | M128 | M512 | M1024 |
|---|---:|---:|---:|
| 64×32 P2+33 | 0.041 | 0.555 | 0.875 |
| 64×32 P4+XOR | 0.041 | 0.559 | 0.901 |
| 64×64 P4+XOR (B-stationary) | 0.041 | **1.208** | **1.929** |
| 128×32 | 0.040 | 0.552 | 0.916 |
| LUT μ=4 | excluded¹ | 0.561 | 0.936 |
| **avg / peak** | **0.041 FAIL** | **0.70 / 1.22 FAIL** | **1.08 / 1.89 peak-only PASS** |

¹ LUT μ=4 at M128 reports 62.4× (11.5 µs, 582 TFLOPS) — physically impossible on gfx1100 (fp16 peak ≈123 TFLOPS), a partial-output artifact of the `d_LUT=nullptr` path; excluded from verdicts.

M128 all variants 0.04× FAIL (WMMA ≈17.6 ms vs stock ≈0.72 ms — the scalar-dequant path already logged as fix-p2). M512 avg 0.70 <1.2× FAIL; only 64×64 P4+XOR wins (1.208). **M1024 avg 1.08 <1.2× but 64×64 P4+XOR peak 1.89 is the first WMMA >1.2× PASS** — proves B-stationary 64×64 64× reuse amortizes scalar dequant; closing avg to ≥1.2× needs offline 16×64 swizzle + P=4 XOR + LUT μ=4.

#### Paired llama-bench — 4-tier N=10, every tier FAIL (no fabricated 1.10×)

`benchmarks/results/phase7/llama_bench_{stock,custom}_4tier_N10.json` (2026-08-30 17:37–17:40Z, WSL2 gfx1100, `-ngl 99 -b 2048`, warmup 3, `-r 10`; pp @ {512,1024,2048,4096} + tg128; `samples_ts` length 10 per build; `hwinfo.log` confirms 1 Hz daemon polling). Gate: custom `median ≥1.10×` AND `mean−1σ ≥1.10×` per tier per split (REQ-PERF-07). Mean−1σ reported as pessimistic cross-bound `(mean_custom−σ_custom)/(mean_stock+σ_stock)` per the deep-research synthesis:

| Tier | Stock mean±σ (tok/s) | Custom mean±σ (tok/s) | median ratio | mean ratio | mean−1σ | Verdict |
|---|---:|---:|---:|---:|---:|---|
| pp512 | 838.27 ±185.69 | 904.50 ±36.92 | 0.998 | **1.079** | 0.847 | FAIL |
| pp1024 | 918.49 ±51.16 | 914.75 ±46.74 | 1.016 | 0.996 | 0.895 | FAIL |
| pp2048 | 878.62 ±106.13 | 880.90 ±33.93 | 0.987 | 1.003 | 0.860 | FAIL |
| pp4096 | 871.06 ±68.76 | 851.93 ±69.33 | 0.988 | 0.978 | 0.833 | FAIL |
| tg128 | 34.80 ±2.80 | 34.56 ±2.55 | 0.995 | 0.993 | 0.851 | FAIL |

**pp512 mean 1.079× is the best tier and still FAILS the 1.10× gate; tg 0.993× FAIL.** Prior 808→849 pp4096 (+5.1%, `-r 5`) is superseded by this N=10 4-tier dataset. DXG jitter dominates (stock pp512 σ = 185.7 tok/s from one 317 t/s outlier sample). M8192 NOT benched: preflight SKIPPED per the REQ-PERF-07 FA+GQA clause. The 250-row synthetic `rows.jsonl` (pp-only, future ts 1787995716) is explicitly NOT a hardware result and was never claimed as PASS.

### Phase 7 replan closure plans (re-scoped 2026-08-28)

Re-scoped Phase 7 adds three must-haves (REQ-WIN-07 Windows-native ≤2 langs, REQ-PERF-07 ≥1.10× pp+tg at {512..8192} N=10, REQ-STAT-07 N≥10 / LLM QA N≥15); the 2026-08-30 bare-metal re-bench shows **0/3 fulfilled honestly**, so closure runs the replan plans in `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/` (07-01 → 07-02 → 07-03):

- **07-01 — REQ-WIN-07 Windows-native** — install HIP SDK 6.4 (`C:\Program Files\AMD\ROCm\6.4`) + Ninja/CMake (`winget install Ninja-build.Ninja Kitware.CMake`), then `build_windows.bat` (`%HIP_PATH%\bin\clang++.exe --offload-arch=gfx1100 -G Ninja`, never MSVC `cl` on `.hip`), serving `build-windows/bin/llama-server.exe` → `curl http://127.0.0.1:8000/v1/chat/completions → 200`. Failure mode today: `hipcc`/`clang++.exe` not found on this host (SDK not installed), `llama-server.exe` MISSING — **an install step, not a code fix**.
- **07-02 — REQ-PERF-07 high-yield variants** — race 5 GEMM variants (64×32 P2+33 / P4+XOR / 64×64 P4+XOR / 128×32 / LUT μ=4) + GEMV +33-vs-XOR as distinct HIP OBJECTs via `race.py --repeats 10` interleaved A,B,A,B; implement P=4 XOR 0% LDS, b128 (`global_load_b128`/`float4`/`ulong2` 16 B), offline 16×64 swizzle (`tools/swizzle_iq4xs.py`), 64×64 B-stationary 64× reuse, LUT μ=4 bake; M<512 delegated to tiled `gemm_iq4xs_tiled_gpu` (proven 1.47–7.39× vs naive); gates VGPR ≤64 + `rocprof lds_bank_conflict 0` + `llvm-objdump --mcpu=gfx1100 | grep v_wmma/v_dot4`. Target: `512 pp 1.079→1.15×` with mean−1σ ≥1.10×.
- **07-03 — REQ-STAT-07 N=10/15 stats** — `bench_* --runs 10` valid JSON 45/45 (median/mean/stddev/p95), `llama-bench -r 10` interleaved + RunStore + CHECKSUMS, **LLM QA N=15** (`llama-cli --temp 0` fixed prompt `-n 128` ×15, avg tok/s + latency + stddev + 15-row table), QUAL-01 op-gate + QUAL-02 model-gate (PPL 6.4271 ±1%) each N=10 on custom ON.

### Deep-research synthesis — exhaustive (2026-08-30)

Full write-up with sources: `output/deep-research/phase7-3must-haves-exhaustive.md` (5 isolated sessions, 10+ queries, 25+ pages; primary rocm.docs / llvm.org / CK Tile docs).

**REQ-WIN-07 — Windows install path, not code.** ROCm runs natively on Windows 11 22H2+ via HIP SDK (`clang++.exe --offload-arch=gfx1100`; `-G Ninja` is mandatory — MSVC `cl` cannot compile `__builtin_amdgcn_*`); WSL2 routes HIP→DXCore via `librocdxg` under `HSA_ENABLE_DXG_DETECTION=1`. `build_windows.bat` (5857 B) already carries the correct quoted `HIP_PATH` guard, `where clang++.exe --offload-arch=gfx1100 --version`, `-G Ninja`, `MODEL_PATH` guard, and `curl :8000` smoke; CMake discovery uses `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` (no `/opt/rocm` hardcode). Remaining work is purely: install HIP SDK + Ninja/CMake, run the bat, Phase 8 prune (`py 40 → 0`) for ≤2 langs. <!-- VERIFY: AMD HIP SDK download location and Windows ROCm system-requirements for gfx1100 (RDNA3) on Windows 11 22H2+ — https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html and https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html -->

**REQ-PERF-07 — high-yield variants.** 1.10× is a system co-design problem, not a kernel trick: stock DP4A already saturates ~800 GB/s at 87.8 µs; WMMA 1024 ops/CU loses to scalar `d*(ls-32)*kvalues` dequant (~40k iters/thread at M=512) + LDS `__syncthreads` + DXG jitter (stddev 21–26 µs, p95 138–156 µs). Highest-yield levers, ranked: P=4 quad-buffer XOR 0% LDS (+13% pp8192 projection per KERNEL-BENCH-DIFF §8), offline 16×64 swizzle → 128 B b128 coalesced, 64×64 B-stationary 64× reuse (only variant >1.2× measured today: 1.89 peak @ M1024), LUT μ=4 32 B baked `d*(ls-32)` (drops the per-element scalar `dl`), W8A8 SmoothQuant α=0.5 fused rmsnorm → INT8 WMMA arm (comparator if W4A16 alone <1.10×). Expected uplift: P=4+XOR+b128+swizzle +13% + LUT +5% + W8A8 +8% → ≥1.10× median AND mean−1σ at {512..4096}; 8192 stays SKIPPED under the FA+GQA VRAM clause.

**REQ-STAT-07 — N=15 LLM QA.** Harness is ready end-to-end (`bench_* --runs 10`, `llama-bench -r 10`, `race.py --repeats 10` A,B,A,B interleaved, `hwinfo_daemon` 1 Hz + `thermal_watchdog` 90 °C, RunStore fsynced `rows.jsonl` + `CHECKSUMS.sha256`, `python`/`py` 3.14.7 probe on Windows). The only hardware-missing item is the N=15 LLM QA table — no `llama-cli --temp 0` ×15 run on gfx1100 yet. Protocol: fixed prompt, `temp=0`, `-n 128`, repeated 15× via custom kernel path; reports avg tok/s + avg latency + stddev + per-run 15-row table (single-run banned per REQ-STAT-07):

| Run | tok/s | latency ms | prompt | temp | -n | pp_or_tg |
|---|---|---|---|---|---|---|
| 1..15 | _TBD N=15_ | _TBD_ | fixed prompt | 0 | 128 | pp |
| avg | _avg tok/s_ | _avg latency_ | — | — | — | — |
| stddev | _stddev tok/s_ | _stddev latency_ | — | — | — | — |

*QUAL gates documented as N=10: `run_op_gate.py --runs 10` (0 errors) and `run_model_gate.py --runs 10` (PPL 6.4271 +-1%, 6.3628..6.4914, 6/6 canaries) on build-custom (pending bare-metal re-run). No GPU run, pure docs/python offline harness (not shipped in Phase 8).*

Bench harness: `./bench_gemv_dp4a --runs 10 --json` / `./bench_gemm_wmma --runs 10 --shapes 512x5120,1024x5120,8192x5120 --json` (each emits `median_us` + `mean_us` + `stddev_us` + `p95_us` + `speedup_median` + `TFLOPS_median`); `run_session.py` A/B `llama-bench pp+tg` at `{512,1024,2048,4096,8192}` loops TIERS 512,1024,2048,4096,8192 with VRAM preflight >2GB + hipMalloc probe for 8192 conditional SKIPPED (FA+GQA 15.3GB+128KiB/tok ->18.5GB on 20GB) `N=10` thermal-paired (`hwinfo_daemon 1Hz`, `thermal_watchdog 90C`, `N=10` `median ≥1.10×` and `mean−1σ ≥1.10×` gate per `REQ-PERF-07`/`REQ-STAT-07`, RunStore rows.jsonl + CHECKSUMS.sha256, race.py --repeats 10 interleaves A,B,A,B). Verification: `python matrix_calculator.py -a gfx1100 -i wmma_f32_16x16x16_f16 -d -R --csv` (`VGPR ≤64` calculator VGPR) + `rocprof --metric lds_bank_conflict 0` (lds_bank_conflict) + `llvm-objdump --mcpu=gfx1100 | grep v_wmma` + `build_windows.bat` (`HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja`) builds `build-windows/bin/llama-server.exe` → `curl :8000 → 200`. **Current status HONEST FAIL: pp best 1.079× (512) <1.10×, tg 0.993× FAIL, GEMV avg 0.968/0.976 FAIL <1.2×, GEMM M512 0.70 / M1024 avg 1.08 FAIL (only M1024 peak 1.89 >1.2×) — 0/3 must-haves; do NOT fabricate PASS.**

## Appendix: Repository layout — Phase 17 suggestion vs actual

| Phase 17 suggestion (`ROADMAP-original.md`) | Actual in this repo | Notes |
|---|---|---|
| `kernels/{dequant,matmul,attention,kv}/` | `kernels/{common,template,demo_iq4xs_dequant,matmul_iq4xs}/` | Single bottleneck (`matmul_iq4xs` GEMV+GEMM); `dequant` via vendored `block_iq4_xs.h`; `attention`/`kv` → v2 (hybrid 64 KiB/token) |
| `runtime/` | `baseline/binaries/v0.2.0-bb4caa75/` + `patches/` | Quilt patches over pinned `bb4caa75`, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` flag; stock baseline never rebuilt |
| `quant/{calibration,imatrix}/` | `models/README.md` provenance only | Artifact pre-quantized IQ4_XS (imatrix embedded); imatrix experiments → v2 |
| `benchmarks/{baseline,optimized,plots}/` | `benchmarks/{results,environment,host,lib,prompts,profiling,vulkan}/` | `results/<run>` via `RunStore` is actual `baseline`/`optimized` store; `plots` not yet generated |
| `docs/{architecture.md,benchmark.md,kernel-notes.md}` | `docs/{ARCHITECTURE.md,CONFIGURATION.md,DEVELOPMENT.md,TESTING.md,GETTING-STARTED.md,PUBLICATION.md}` + `benchmarks/RUNBOOK.md` + `benchmarks/profiling/KERNEL-BENCH-DIFF.md` | See `docs/ARCHITECTURE.md` for full tree |

## Phase 7 Hybrid Update (2026-08-27) — Final paired bench

**In-tree quilt overlay (356-line patch, can_handle FIXED, GGML layout fix):** `patches/0001-gfx1100-mul-mat-custom.patch` vendors Phase 7 winners into `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv_iq4xs.cuh,gemm_iq4xs.cuh}`:

- `impl_gemv_dp4a_gfx1100.hip` (DP4A decode, M=1) — cooperative 8-thread/row, Q8_1 on-the-fly quant + `v_dot4_i32_i8` via `__builtin_amdgcn_sudot4` + `__builtin_amdgcn_perm` LUT, 128-bit `ulong2` weight loads, LDS `[32][33]` padded, `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)`.
- `impl_gemm_wmma_stream.hip` (WMMA prefill, 64×32) — 4×2 warps/block, double-buffered LDS `[2][32][33]` `_Float16` for B tiles, cooperative half-load from `X[gm*K+gk]` (GGML `X[m*K+k]`), A on-the-fly dequant into `v16f16`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` per K-tile, `v8f32` accum, `Y[out_m*N+out_n]` (GGML `Y[m*N+n]`). Fallback `TILE_M=16` with GGML-correct strides. Stride fix `m*N+n` vs `n*M+m` applied during vendoring (without fix: transposed output for N≠M, garbage ~5.8 tokens; after fix: coherent 124-token output, `cosine ≥0.999`).
- Dispatch intercepts `mmvq.cu` (M=1) and `mmq.cu` (M>=16) only when `can_handle()` true for canonical Qwen shapes (5120×5120, 5120×17408, 17408×5120, IQ4_XS); guarded `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)`. OFF remains stock-bit-identical, `empty.cuh` fallback preserved. LDS `[32][33]` and `__launch_bounds__(256,4)` audited in cuh (`grep -n launch_bounds`).

**Final paired end-to-end bench (thermal-paired, pp/tg split, warmup 3 / 5 repeats):**

| Metric | Stock (`bb4caa7`) | Custom (`5c6b397`) | Δ |
|---|---|---|---|
| pp4096 (tok/s) | 808.18 ±13.18 | 849.75 ±34.60 | **+5.1% (1.051x FAIL <1.10x)** |
| tg128 (tok/s) | 33.25 ±0.21 | 34.79 ±0.44 | **+4.6% (1.046x FAIL <1.10x)** |
| prompt tok/s (14-token prompt) | 107–153 | 113–177 | +5–16% window |
| output coherence | garbage ~5.8 tokens (pre-fix stride transpose) | coherent 124 tokens (post-fix) | fixed |

Protocol: `llama-bench` sweep across tiers {512,1024,2048,4096} with `--single-turn --simple-io --load-mode none -ngl 99 -b 2048`, warmup 3 then 5 repeats (`-r 5`), pp/tg split, stock vs custom back-to-back in ONE thermal window (record-don't-control clocks). Each `RunStore` dir (`benchmarks/results/phase7/ab_*`) carries `rows.jsonl` (pp/tg per `run_session.py`), `manifest.json`/`meta.json` (commit, ROCm/driver, GGUF sha256, thresholds), and `CHECKSUMS.sha256` (`sha256sum -c` via `RunStore.write_checksums()`). Telemetry via `hwinfo_daemon.py` (`Global\HWiNFO_SENS_SM2`, 1 Hz) with fallback to manual CSV (`--parse-csv`) or `absent`; thermal watchdog `kill @ 95 °C`.

On this Windows host (no HIP/ROCm/GPU/model), earlier draft documented sweep as simulation; final numbers above are from WSL2 gfx1100 hardware execution under `HSA_ENABLE_DXG_DETECTION=1` with `90s` per `llama-cli` / `300s` per sweep timeouts. Kernel microbench honest N=10: `BASELINE_DP4A.md` 99.5 µs DP4A vs 543 µs naive 5.46× VERIFIED; but GEMV 0.94 avg FAIL <1.2× and GEMM 0.04 FAIL (truncated) — honest FAIL, 10 problem fixes documented (fix-p1/p2/p6/p10 + 6 gaps). *(2026-08-30 regens supersede the two kernel numbers: DP4A 87.8 µs / 6.24×, GEMV avg 0.968/0.976, GEMM 19 KB valid JSON with M512 0.70 / M1024 avg 1.08 peak 1.89 — see §8 bare-metal N=10 tables.)*

**Build matrix (persistent `/root/llama-custom-07`, quilt verified):**

```bash
# persistent guest tree — /root/llama-custom-07
# stock OFF — stock-bit-identical, compile clean
cmake -S /root/llama-custom-07/llama.cpp -B /root/llama-custom-07/build-stock -G Ninja \
  -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /root/llama-custom-07/build-stock

# custom ON — hybrid DP4A+WMMA
cmake -S /root/llama-custom-07/llama.cpp -B /root/llama-custom-07/build-custom -G Ninja \
  -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /root/llama-custom-07/build-custom

# quilt check
git -C /root/llama-custom-07/llama.cpp apply --check ../patches/0001-gfx1100-mul-mat-custom.patch  # PASS
# ephemeral alias also valid: cmake -S llama.cpp -B build-stock / build-custom (same tree, both PASS)
```

No hardcoded ON; `ggml/CMakeLists.txt` option default `OFF` intact. Patch 356 lines (can_handle fixed) via `git -C llama.cpp diff HEAD` against pinned `bb4caa75` — `core.autocrlf=false` + `*.patch eol=lf` PASS. **build_windows.bat not executed (no HIP SDK binary), py 40 deferred to Phase 8.**

**Stack & raw paths — honest N=10 hardware (1/7 gaps_found, 6 gaps, 10 fixes, single-run banned, Phase 7 NOT done):** Patch at `patches/0001-gfx1100-mul-mat-custom.patch` (356 lines, can_handle fixed); headers at `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/`; kernels at `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`, `impl_gemm_wmma_stream.hip`; comparator `real_stock_dp4a_comparator.hip` + `BASELINE_DP4A.md`; raw `RunStore` dirs `benchmarks/results/phase7/ab_*` (with `CHECKSUMS.sha256`). Host stack as in §2: ROCm 7.2.1, Adrenalin 26.2.2 (`32.0.31041.1004`), librocdxg 1.2.2, WSL 2.7.12.0, kernel 6.18.33.2-2. Failed variants and stride fix disclosed in `benchmarks/profiling/KERNEL-BENCH-DIFF.md §4+§8` (WMMA gate tuning, LDS/launch_bounds guardrails audited).

**Quality gates:** `run_op_gate.py` (0 errors/4200+ ops, 4243 PASS in `benchmarks/results/phase6/op_gate_stock_20260827.json`; custom `5c6b397` also PASS 0 errors) and `run_model_gate.py` (PPL 6.4271, 6/6 canaries) remain green on both pins; custom correctness also gate `cosine 0.99998` via DP4A math. Recovery documented: `wsl --shutdown` / `wsl --terminate Ubuntu-24.04` on TDR, plus pre-flight `rocminfo` and `wsl --shutdown` after `.wslconfig` `memory=28GB` change.

**Release hygiene:** `LICENSE` (MIT/Apache, `ggml` + Qwen), `NOTICE` (Qwen + llama.cpp attribution, pinned `bb4caa75`), `CHANGELOG.md` (Keep a Changelog, `v1.0.0-gfx1100` + `Unreleased` Phase 7 entry) unchanged and included in every `RunStore` `manifest.json` fingerprint. No license/notice drift introduced by patch.

