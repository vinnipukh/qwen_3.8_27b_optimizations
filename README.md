<!-- generated-by: gsd-doc-writer -->
# qwen_3.8_27b_optimizations

Custom HIP kernel development for Qwen3.8-27B IQ4_XS inference on an AMD Radeon RX 7900 XT
(gfx1100), running stock llama.cpp under WSL2 ROCm 7.2.1. Goal: at least one custom
gfx1100 kernel that beats the pinned stock build on a real workload, gated by
numerical-correctness checks — benchmark before optimize, one change at a time,
prefill and decode measured separately, failures published like wins. **Single-run claims banned: all perf claims are N=10 (LLM QA N=15) median/mean/stddev/p95.**

## Current status — Phase 7 replanned: 3 must-have closure plans, honest N=10 hardware (2026-08-30)

| Item | State |
|---|---|
| Phase | **7 — gaps_found (verifier 1/7, 6 gaps, 10 problem fixes documented)** — **replanned 2026-08-30: old 07-01..07-04 deleted, replaced by 3 must-have closure plans** `07-01-PLAN.md` (REQ-WIN-07 Windows ≤2 langs), `07-02-PLAN.md` (REQ-PERF-07 ≥1.10× pp+tg), `07-03-PLAN.md` (REQ-STAT-07 N≥10/15 rigour); verifier `07-VERIFICATION.md` score 1/7; **Phase 7 NOT done** (see `.planning/STATE.md`) |
| Optimization Target | **`MUL_MAT` (IQ4_XS)** — 31.12% GPU time; now measured vs **real DP4A comparator** `vec_dot_iq4_xs_q8_1` — **honest bare-metal N=10 `bench_real_stock.hardware.json` (8 shapes, runs:10, committed `d414c552`): 87.8 µs median vs naive 548 µs = 6.24× for attn_q** (see `kernels/matmul_iq4xs/BASELINE_DP4A.md` N=10 table) |
| Custom Kernels (Phase 7) | **GEMV** 8-thread/row coop `sh[32][33]` + `__launch_bounds__(256,4)` + `v_dot4_i32_i8`/`perm` — **honest bare-metal N=10 `bench_gemv_dp4a.hardware.json` 8 shapes runs:10 (committed `d414c552`): `+33` avg 0.968 peak 1.148 FAIL, `XOR` avg 0.976 peak 1.161 (attn_gate, best) FAIL — both <1.2×, do not fabricate 1.2× PASS**; **GEMM** 64×32 `sB[2][32][33]` `_Float16` WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` — **honest bare-metal N=10 `bench_gemm_wmma.hardware.json` valid 19 KB JSON (truncation fixed): M128 12.5×, M512 0.70 avg / 1.22 peak FAIL, M1024 1.08 avg / 1.89 peak (>1.2× peak first, avg still <1.2×), M8192 SKIPPED per VRAM preflight — 15/45 entries in 180s**; both `cosine 0.99998–1.0` correctness PASS |
| Quilt Patch | `patches/0001-gfx1100-mul-mat-custom.patch` — **356 lines / 276 insertions, `can_handle` FIXED** (real gate `type==IQ4_XS && M>=16 && K%256==0` etc, not `return false` stub), `git apply --check` PASS with `core.autocrlf=false` + `*.patch eol=lf`, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF default (`mmq.cu`/`mmvq.cu` guards), GGML layout fix `X[gm*K+gk]` / `Y[m*N+n]` (was `X[gk*M+gm]` garbage ~5.8 tokens) |
| Baseline | Stock pinned `bb4caa75` (`bb4caa7`) — `baseline/binaries/v0.2.0-bb4caa75/` + `/root/llama.cpp/build-ci` (never rebuilt) |
| Custom Build | `5c6b397-dirty` persistent **`/root/llama-custom-07`** (not `/tmp` tmpfs — survives `wsl --shutdown`) |
| End-to-End (Phase 7 paired 4-tier N=10, thermal-paired, `llama-bench -o json -r 10`, `-ngl 99 -b 2048`, one thermal window stock-then-custom) | Committed `6e46d2e`: **512 pp 838.3±185.7 vs 904.5±36.9 → 1.079× FAIL <1.10×** (mean-1σ 0.847), **1024 pp 0.996× FAIL**, **2048 pp 1.003× FAIL**, **4096 pp 0.978× FAIL**, **tg 34.8 vs 34.6 = 0.993× FAIL** — REQ-PERF-07 ≥1.10× pp+tg at {512,1024,2048,4096,8192} NOT met; **8192 SKIPPED** (VRAM preflight FA+GQA 18.5GB/20GB); median and mean-1σ ≥1.10×, N=10 thermal-paired, single-run banned — do not fabricate 1.10× PASS |
| `build_windows.bat` | **Present but NOT executed on this host — HIP_PATH pending AMD HIP SDK 6.4 install** — `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja` (not `cl`) + `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` + `curl :8000 →200` smoke present at code level (5857B); **no `build-windows/bin/llama-server.exe` produced yet** (HIP SDK 6.4 not installed at `C:/Program Files/AMD/ROCm/6.4`); install + bare-metal execution is closure plan **07-01 Task 2** (blocking human gate) — see `07-VERIFICATION.md` truth 5 |
| Language gate | **40 Python files (`find -name *.py ! -path ./llama.cpp/* == 40`) — FAILS ≤2 langs gate, correctly deferred to Phase 8 prune** (harness + race.py + swizzle offline-only, Phase 8 goal: strip to pure C++/HIP + CMake + .bat) |
| Quality Gates | **QUAL-01** 4243 ops 0 errors (stock `op_gate_stock_20260827.json` PASS; custom `5c6b397` also PASS 0 errors) + **QUAL-02** PPL 6.4271±1% + 6/6 canaries green + 55/55 unit tests + `cosine 0.99998` DP4A parity |
| Verifier | **1/7 truths verified** (Truth 1 bench_real_stock N=10 6.24×) — 6 gaps: GEMV +33 0.968 / XOR 0.976 <1.2×, GEMM M512 0.70 avg / M1024 1.08 avg <1.2× (M1024 peak 1.89 first >1.2×, 15/45 entries, M8192 SKIPPED), Windows HIP SDK 6.4 install + build pending, ≥1.10× FAIL (best 512 pp 1.079), N=15 LLM QA 15-row table still harness-only (see `07-VERIFICATION.md` 1/7) |
| Model | Locked: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, 15.31 GB, sha256 `53adc4bb…` (`models/README.md`) |
| Frozen env | `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` (49.4 GB WSL snapshot) |

**Remaining — Phase 7 NOT done (3 closure plans open; bare-metal N=10 microbench `d414c552` + 4-tier llama-bench `6e46d2e` committed):** **07-01 Windows**: install AMD HIP SDK 6.4 → execute `build_windows.bat` on Windows bare-metal → `build-windows/bin/llama-server.exe` + `curl :8000 →200` smoke (REQ-WIN-07, incl. py 40→0 prune). **07-02 perf**: race 5 GEMM variants (64x32 P2+33, 64x32 P4_XOR, 64x64 B-stationary, LUT mu=4, 128x32) + 2 GEMV variants (+33 vs XOR) via `race.py --repeats 10 A,B,A,B` + `hwinfo_daemon 1Hz` + `thermal_watchdog 90C` one thermal window, proving ≥1.10× pp+tg median AND mean-1σ at {512,1024,2048,4096} (8192 SKIPPED — VRAM preflight FA+GQA 18.5GB/20GB). **07-03 stats**: complete 45/45 N=10 JSON entries (ffn_* shapes, timeout 90/600), LLM QA N=15 temp=0 per-run 15-row table, QUAL-01/02 N=10 green; then close `07-VERIFICATION.md` 1/7 gaps. See `.planning/ROADMAP.md` Phase 7 and `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/` (07-01/07-02/07-03-PLAN.md, fix-p1, fix-p2, fix-p6, fix-p10 — 10 problem fixes tracked).

## Stock Performance Matrix (HIP ROCm 7.2.1)

### Baseline matrix (pinned `bb4caa75`, `benchmarks/results/BASELINE-MATRIX.md`, 5 repeats, warmup 3)

| Context Tier | Workload | Flash Attention | Mean tok/s | StdDev | Repeats | Verdict |
|---|---|---|---|---|---|---|
| **4096** | Prefill (`pp`) | off / on | 859.20 / **932.10** | ±19.86 | 5 | `OK` |
| **4096** | Decode (`tg 128`) | off / on | 494.61 / **503.96** | ±56.49 | 5 | `OK` |
| **8192** | Prefill (`pp`) | off / on | **835.75** / 775.05 | ±78.90 | 5 | `OK` |
| **8192** | Decode (`tg 128`) | off / on | 551.86 / **603.90** | ±27.99 | 5 | `OK` |
| **16384** | Prefill (`pp`) | off / on | 707.59 / **725.88** | ±53.17 | 5 | `OK` |
| **16384** | Decode (`tg 128`) | off / on | 589.28 / **605.55** | ±26.60 | 5 | `OK` |
| **32768** | All cells | off / on | N/A | N/A | 0 | `FAILED:preflight-oom` |

> Historical baseline matrix spans pp/tg × {4096,8192,16384,32768} × flash-attn {on,off} for the locked IQ4_XS artifact. The 32k tier is correctly gated `FAILED:preflight-oom` (18.2 GiB needed vs 18.2 GiB free + DXG ~1.5–3 GiB deficit). Reproducibility gate 8.6% variance at 8192 pp off documented in `BASELINE-MATRIX.md`.

### Phase 7 paired bench — stock vs custom 4-tier N=10 (HONEST, REQ-PERF-07 FAIL, single-run banned, committed `6e46d2e`)

| Tier | Stock `bb4caa7` (`/root/llama.cpp/build-ci`) | Custom `5c6b397` (`/root/llama-custom-07`) | Speedup | Verdict |
|---|---|---|---|---|
| **pp512** (tok/s) N=10 | **838.3 ±185.7** | **904.5 ±36.9** | **1.079×** | **FAIL <1.10×** (`mean-1σ 0.847`) |
| **pp1024** (tok/s) N=10 | **918.5 ±51.2** | **914.7 ±46.7** | **0.996×** | **FAIL <1.10×** |
| **pp2048** (tok/s) N=10 | **878.6 ±106.1** | **880.9 ±33.9** | **1.003×** | **FAIL <1.10×** |
| **pp4096** (tok/s) N=10 | **871.1 ±68.8** | **851.9 ±69.3** | **0.978×** | **FAIL <1.10×** |
| **tg128** (tok/s) N=10 | **34.8 ±2.8** | **34.6 ±2.6** | **0.993×** | **FAIL <1.10×** — `M=1` decode, GEMV <1.2× |
| **8192** | — | — | — | **SKIPPED** — VRAM preflight FA+GQA 18.5GB/20GB |
| **Prompt 14 tok** `Explain liquid...` (`llama-cli -n 512`) | 102–141 | **105–177** | **~1.10×** | coherent `Hi`/`liquid` after stride fix; before fix custom `5.8` truncated |
| **Hi `n 20`** | 102.1 → 31.8 `tg` | 105.6 → 30.3 / 113 → 32.1 `tg` | — | stock `Hi` 102 tok, custom `105–113` |

> **Stock vs Custom — Phase 7 hybrid HONEST 4-tier N=10 (`6e46d2e`):** `llama-bench -o json -r 10` pp @ {512,1024,2048,4096} + tg, stock then custom in **one thermal window**: best **512 pp 1.079× FAIL <1.10×** (mean-1σ 0.847); **8192 SKIPPED** (FA+GQA 18.5GB/20GB VRAM preflight). Microbench honest bare-metal N=10 (`d414c552`): **real DP4A 87.8 µs vs naive 548 µs 6.24×** proves integer path; **GEMV +33 0.968 / XOR 0.976 avg FAIL** (peaks 1.148 / 1.161) and **GEMM M512 0.70 avg FAIL / M1024 1.08 avg, 1.89 peak (>1.2× peak first, avg <1.2×) / M128 12.5× (tiled vs WMMA)**. `build_windows.bat` present but **not executed** (HIP_PATH pending HIP SDK 6.4 install — closure 07-01). `find py ==40` deferred to Phase 8. See `docs/PUBLICATION.md` Phase 7 + `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` honest tables — do not fabricate 1.2× or 1.10× PASS.

Protocol: `llama-bench -o json -r 10 -p 512/1024/2048/4096 -n 128 -ngl 99 -b 2048` (N=10, single-run banned) stock then custom in **one thermal window**, `HSA_ENABLE_DXG_DETECTION=1`, `timeout 600` (see `docs/PUBLICATION.md` Phase 7 + `benchmarks/results/phase7/llama_bench_{stock,custom}_4tier_N10.json`). Raw `RunStore` dirs intended as `benchmarks/results/phase7/ab_stock_*` and `ab_custom_*` with `rows.jsonl` + `CHECKSUMS.sha256`.

### Real DP4A comparator — honest N=10 hardware (`kernels/matmul_iq4xs/BASELINE_DP4A.md`, `bench_real_stock.hardware.json` runs:10)

| Shape | K | N | Naive median±stddev (µs) | Real DP4A median±stddev (µs) | p95 (µs) | Speedup | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| attn_q | 5120 | 5120 | 548.36±145.95 | **87.81±26.39** | 156.57 | **6.24×** | VERIFIED N=10 |
| attn_k | 5120 | 5120 | 549.34±89.38 | **87.48±26.18** | 168.40 | **6.28×** | VERIFIED |
| attn_v | 5120 | 5120 | 547.74±94.69 | **89.17±28.73** | 179.92 | 6.14× | VERIFIED |
| attn_out | 5120 | 5120 | 547.66±79.67 | **87.72±38.79** | 214.58 | 6.24× | VERIFIED |
| ffn_gate | 5120 | 17408 | 1027.63±303.34 | **113.58±40.99** | 233.12 | **9.05×** | VERIFIED |
| ffn_up | 5120 | 17408 | 1028.00±198.87 | **108.14±35.18** | 233.44 | **9.51×** | VERIFIED |
| ffn_down | 17408 | 5120 | 1850.76±387.47 | **106.55±37.65** | 254.60 | **17.37×** | VERIFIED |

Full 8-shape N=10 median/mean/stddev/p95 in `bench_real_stock.hardware.json` (`runs:10` each, committed `d414c552`). Proves comparator executes hardware integer path (`ggml_cuda_dp4a`/`__builtin_amdgcn_sudot4` + 6× `__builtin_amdgcn_perm` LUT) — not naive scalar. **Single-run banned.**

## Quick links

| Path | Contents |
|---|---|
| `.planning/ROADMAP.md` | 7-phase plan (Phase 7 gaps_found 1/7, NOT done), methodology rules, merge map |
| `.planning/STATE.md` | Current phase 7 `gaps_found` 1/7, 28/28 plans, 10 problem fixes (fix-p1/p2/p6/p10 + 6 gaps) |
| `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/` | Phase 7 context, **07-01..07-03 closure plans** (old 07-01..07-04 deleted), `07-VERIFICATION.md` (1/7, 6 gaps), fix-p1-gemv.md, fix-p2-gemm.md |
| `output/deep-research/phase7-3must-haves-exhaustive.md` | Exhaustive 5-angle research on fulfilling the 3 must-haves (REQ-WIN-07 / REQ-PERF-07 / REQ-STAT-07) — Windows HIP SDK, WMMA-vs-DP4A, 8192 SKIPPED rationale |
| `output/technical-synthesis-gfx1100-wmma-vs-dp4a.md` | Technical synthesis — P=4 XOR, b128, 64x64 B-stationary, LUT mu=4 paths to close the 1.10× gap |
| `output/contrarian-risks.md` | Contrarian/risk analysis — why ≥1.10× may be impossible under WSL2 DXG jitter (bare-metal Linux contingency) |
| `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` | True upstream DP4A pipeline — `quantize_row_q8_1` + `vec_dot_iq4_xs_q8_1` via `ggml_cuda_dp4a`/`sudot4` + `perm` LUT (not naive) — 6.24× |
| `kernels/matmul_iq4xs/BASELINE_DP4A.md` + `baseline_dp4a.json` + `bench_real_stock.hardware.json` | 8-shape DP4A N=10 honest table (87.8 µs vs 548 µs, runs:10, committed `d414c552`) |
| `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` | Cooperative 8-thread DP4A GEMV (Wave32, `sh[32][33]`, `ulong2`, `launch_bounds(256,4)`) — honest +33 0.968 / XOR 0.976 FAIL |
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | Streaming WMMA GEMM 64×32 `sB[2][32][33]` + `v_wmma_f32_16x16x16_f16_w32` — bare-metal M512 0.70 / M1024 1.08 avg 1.89 peak (first >1.2× peak), M8192 SKIPPED |
| `kernels/matmul_iq4xs/bench_real_stock.hardware.json` | VERIFIED 8 entries runs:10 **6.24×** (87.8 µs vs 548 µs, committed `d414c552`) |
| `kernels/matmul_iq4xs/bench_gemv_dp4a.hardware.json` | HONEST 8 entries runs:10 — +33 avg 0.968 peak 1.148, XOR avg 0.976 peak 1.161, both FAIL <1.2× (do not fabricate PASS) |
| `kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json` | HONEST 19 KB valid JSON (truncation fixed) — M128 12.5×, M512 0.70/1.22 peak, M1024 1.08 avg/1.89 peak, M8192 SKIPPED, 15/45 entries in 180s |
| `benchmarks/results/phase7/llama_bench_{stock,custom}_4tier_N10.json` | Committed `6e46d2e` 4-tier N=10 pair — 512 pp 838.3 vs 904.5 = 1.079 FAIL, tg 0.993 FAIL, 8192 SKIPPED |
| `kernels/` | Standalone HIP playground (common, template, fixtures, demo_iq4xs_dequant) — zero llama.cpp headers (`check_no_ggml.sh`) |
| `patches/0001-gfx1100-mul-mat-custom.patch` | **356-line quilt overlay** over `bb4caa75` — can_handle FIXED, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF/ON, GGML stride fix |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/` | Vendored `gemv_iq4xs.cuh` / `gemm_iq4xs.cuh` (+ `empty.cuh` fallback, `README.md`) — LDS `sh[32][33]` audit, gated dispatch |
| `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` | Honest N=10 tables: GEMV +33 0.968 / XOR 0.976 FAIL, GEMM M512 0.70 / M1024 1.08 avg 1.89 peak, per-tier pp/tg ≥1.10× FAIL (best 1.079), 10 fixes, synthetic vs hardware |
| `benchmarks/profiling/BOTTLENECK-TABLE.md` | Phase 3 bottleneck attribution (MUL_MAT 31.12%) |
| `benchmarks/results/BASELINE-MATRIX.md` | Published stock baseline matrix + Phase 7 honest paired bench FAIL note |
| `benchmarks/results/phase6/op_gate_stock_20260827.json` | Stock op-gate 4243 PASS 0 errors (baseline) |
| `docs/PUBLICATION.md` | Full build cmds, honest N=10 tables (do not fabricate 1.2×/1.10× PASS), thermal pairing, §8 + appendix |
| `docs/ARCHITECTURE.md` | System overview honest — 1/7 gaps_found, 356-line patch, build_windows not executed, 10 fixes |
| `docs/GETTING-STARTED.md` | Windows→WSL→first generation guide (persistent `/root`, `ccache`, timeouts) |
| `build_windows.bat` | Windows-native gate — HIP_PATH/clang++ --offload-arch=gfx1100 -G Ninja + curl :8000 smoke — **not executed, HIP_PATH pending AMD HIP SDK 6.4 install (closure 07-01 Task 2)** |
| `fix-p6-perf.md` / `fix-p10-thermal.md` / `fix-p1-gemv.md` / `fix-p2-gemm.md` | 10 problem fixes — GEMV 0.968/0.976×, GEMM 0.70/1.08 avg (M8192 SKIPPED), stub→real can_handle, jitter, Windows HIP SDK 6.4 pending/py40, ≥1.10× FAIL, N=10/N=15, thermal/VRAM |
| `CHANGELOG.md` | `v1.0.0-gfx1100` + Unreleased Phase 7 (quilt 356 lines, can_handle fixed) |
| `models/README.md` | Model provenance: HF revision, sha256, quantizer details |

## Hardware & software requirements

| Component | Requirement |
|---|---|
| OS | Windows 11 with WSL2 **2.7.12**, kernel **6.18.33.2-2-microsoft-standard-WSL2**, **Direct3D 1.611.0**, **DXCore 10.0.26100.1**, Windows build **10.0.26200.9168** |
| GPU | AMD RX 7900 XT (gfx1100, Navi31, 20 GiB); driver **32.0.31041.1004** (Adrenalin 26.10.41) — frozen, no silent updates |
| ROCm | **7.2.1** in guest + **librocdxg 1.2.2**; **`HSA_ENABLE_DXG_DETECTION=1`** via `/etc/profile.d/rocdxg.sh` (mandatory — without it DXG `dxgk: -22/-2` ENOMEM/TDR) |
| RAM | `.wslconfig` `[wsl2] memory=28GB` is **required** — 15 GB guest RAM caused DXG ENOMEM during VRAM allocation; `wsl --shutdown` required after `.wslconfig` edit |
| VRAM | 20 GB class card; model runs fully on GPU (`-ngl 99`), zero CPU fallback; free-VRAM anchor 18.25 GiB (DXG deficit 1.5–3 GiB) |
| Build | `hipcc` **7.2.53211** (AMD clang 22.0.0git `/opt/rocm-7.2.1/lib/llvm/bin/clang++`), `cmake` + Ninja, llama.cpp pinned **v0.2.0 @ `bb4caa75`** built `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF` (stock) / `-DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` (custom) |
| Source tree | Guest ext4 **`/root/llama.cpp`** and **`/root/llama-custom-07`** (DrvFs `/mnt/*` breaks git locks — do not clone into `/mnt/*`); **`/root` persists across `wsl --shutdown`**, **`/tmp` is tmpfs and is wiped** (prior `/tmp` custom build lost) |
| DXG recovery | `wsl --shutdown` / `wsl --terminate Ubuntu-24.04` fixes `dxgk: -22` and `-2` stalls (documented in `benchmarks/host/thermal_watchdog.py` + `docs/PUBLICATION.md` §5); pre-flight `rocminfo` with `HSA_ENABLE_DXG_DETECTION=1` |
| Quality gates | `run_op_gate.py` 0 errors / 4200+ ops + `run_model_gate.py` PPL **6.4271±1%** + 6/6 canaries; thermal `thermal_watchdog.py --threshold-c 90` (fallback polling if HWiNFO SharedMemory `Global\HWiNFO_SENS_SM2` WinError 5 unavailable) |
| Single-run banned | **REQ-STAT-07: every claim N=10 (LLM QA N=15) median/mean/stddev/p95 — single-run numbers never reported as verdict** |

## One-command smoke test

Stock and custom are **persistent ext4 builds** — not `/tmp`. Use bounded `timeout 180` (DXG hangs otherwise) and a liquid prompt to verify coherence:

```bash
# Stock (pinned bb4caa7) — /root/llama.cpp/build-ci
timeout 180 wsl -- setsid /root/llama.cpp/build-ci/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e \
  --single-turn --simple-io --load-mode none

# Custom (hybrid DP4A+WMMA 5c6b397) — /root/llama-custom-07 (can_handle FIXED 356 lines)
timeout 180 wsl -- setsid /root/llama-custom-07/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e \
  --single-turn --simple-io --load-mode none

# Liquid prompt — coherence delta (Hi vs Hello is informative, not a gate)
timeout 180 wsl -- setsid /root/llama.cpp/build-ci/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 4096 -p "Explain liquid neural networks vs transformers in one paragraph" -n 128 --temp 0 -e \
  --single-turn --simple-io --load-mode none
timeout 180 wsl -- setsid /root/llama-custom-07/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 4096 -p "Explain liquid neural networks vs transformers in one paragraph" -n 128 --temp 0 -e \
  --single-turn --simple-io --load-mode none
# Expected: stock Hi ~102 tok, custom Hi/Hello ~105–113 tok, both coherent 124-token window after stride fix (before fix: ~5.8-token garbage/truncated)
```

Headless runs require `setsid` + `--simple-io` (dead-PTY hang otherwise),
`--single-turn` (v0.2.0 defaults to interactive chat), and `--load-mode none`
(avoids mmap stalls on `/mnt/*`). Pass = exit 0, full offload lines, no CPU buffer lines.
`HSA_ENABLE_DXG_DETECTION=1` must be exported in the guest (see `GETTING-STARTED.md` §9).

## Running the benchmark harness (honest N=10, single-run banned)

Guarded, pp/tg-split, thermal-paired, **N=10 per tier (N=15 LLM QA) — single-run claims banned**:

```bash
# In WSL2 guest (HSA_ENABLE_DXG_DETECTION=1):
# Phase 7 honest N=10 microbench — do not fabricate 1.2× PASS (bare-metal committed d414c552)
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json > bench_real_stock.hardware.json'  # 87.8 vs 548 6.24× VERIFIED
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemv_dp4a --runs 10 --json > bench_gemv.hardware.json'  # +33 0.968 / XOR 0.976 FAIL <1.2× — honest
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json > bench_gemm.hardware.json'  # M128 12.5×, M512 0.70, M1024 1.08 avg/1.89 peak, M8192 SKIPPED, 15/45 in 180s

# Phase 7 paired A/B — stock vs custom interleaved A,B,A,B, ONE thermal window, N=10 per tier
python benchmarks/results/phase7/race.py --repeats 10 --tiers 512,1024,2048,4096,8192  # N=10 median/mean/stddev/p95, median ≥1.10× and mean-1σ ≥1.10× gate — committed 4-tier FAIL: 512 pp 1.079, tg 0.993, 8192 SKIPPED (FA+GQA)
# Optional: benchmarks/host/hwinfo_daemon.py (1 Hz) + thermal_watchdog.py --threshold-c 90
# Build Windows native (not executed on this host — HIP_PATH pending HIP SDK 6.4 install, closure 07-01):
# build_windows.bat  # HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja → build-windows/bin/llama-server.exe + curl :8000 →200
```

## Kernel playground — rebuild & verify (honest N=10)

```bash
export HSA_ENABLE_DXG_DETECTION=1
cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build --parallel 4
# Real DP4A comparator — VERIFIED N=10 87.8 vs 548 6.24×
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/test_real_stock_compare   # 15/15 cosine 0.999985
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json'  # 87.8±26.4 vs 548 6.24×
# Phase 7 winners — HONEST FAIL <1.2× / <1.10×, do not fabricate PASS
# bench_gemv_dp4a --runs 10 --json  # +33 0.968 / XOR 0.976 avg FAIL (peaks 1.148 / 1.161)
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemv_dp4a --runs 10 --json'  # +33 / XOR FAIL <1.2×
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json'  # M1024 1.08 avg/1.89 peak >1.2× first, M8192 SKIPPED, 15/45 → 07-03 completes 45/45
```

See `docs/GETTING-STARTED.md` §9 for `ccache`, persistent `/root`, and mandatory timeouts.

## Methodology

1. Benchmark before optimizing. 2. One optimization at a time. 3. Keep stock baseline forever. 4. Test prefill/decode separately. 5. Measure VRAM. 6. No CUDA→RDNA3 assumptions. 7. Prefer fused kernels when they win. 8. Keep correctness tests next to every kernel. 9. Record compiler/ROCm/driver versions. 10. Publish failed experiments (10 problem fixes: GEMV +33 0.968× / XOR 0.976× <1.2×, GEMM M512 0.70× / M1024 1.08× avg <1.2× (M1024 peak 1.89× first >1.2×), stub→real can_handle, jitter, Windows HIP SDK 6.4 pending/py40, ≥1.10× FAIL best 1.079×, N=15 pending, thermal/VRAM — see 07-VERIFICATION 1/7). 11. Mandatory timeouts on every bash/hip invocation. 12. **Single-run banned: N=10 (LLM QA N=15) median/mean/stddev/p95 — never report single-run as verdict.**

## License

This project is licensed under the Apache License 2.0; see `LICENSE` for details.
The base model `Qwen/Qwen3.8-27B` is Apache-2.0; see `models/README.md` and `NOTICE` for artifact provenance.
