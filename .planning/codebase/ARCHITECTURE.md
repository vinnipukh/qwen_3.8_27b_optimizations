<!-- refreshed: 2026-08-30 -->
# Architecture

**Analysis Date:** 2026-08-30 (Phase 7 replan closure: 07-01 Windows toolchain, 07-02 variant race, 07-03 statistical rigour)

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│              Windows Host (driver + telemetry, measurement arm)                │
│  Adrenalin 26.2.2, .wslconfig memory=28GB, HWiNFO64 SM2 bridge 1Hz,           │
│  thermal watchdog 90–95°C, build_windows.bat (HIP_PATH, -G Ninja)             │
│  `benchmarks/host/hwinfo_daemon.py` `benchmarks/host/thermal_watchdog.py`     │
└──────────────────────────────┬───────────────────────────────────────────────┘
                              │ /dev/dxg passthrough (librocdxg 1.2.2, HSA_ENABLE_DXG_DETECTION=1)
┌─────────────────────────────▼───────────────────────────────────────────────┐
│          WSL2 Guest: Ubuntu 24.04 + ROCm 7.2.1 (pinned, gfx1100 only)         │
│                                                                              │
│  ┌──────────────────────────┐   ┌─────────────────────────────────────────┐  │
│  │ Benchmark Harness (Py)   │   │ Kernel Playground (HIP/C++17, standalone)│  │
│  │ `benchmarks/bin/*`       │   │ `kernels/` — zero llama.cpp headers      │  │
│  │  run_session.py          │   │  quartet: ref_cpu → impl*.hip →          │  │
│  │  run_op_gate / model     │   │  test_*_compare → bench_* --runs 10      │  │
│  │  race.py (benchmarks/    │   │  real_stock_dp4a_comparator.hip = honest │  │
│  │    results/phase7/)      │   │  N=10 denominator                        │  │
│  │  RunStore append-only    │   │  `kernels/matmul_iq4xs/CMakeLists.txt`   │  │
│  │ `benchmarks/lib/*`       │   │  (7 variant OBJECT libs)                 │  │
│  └──────────┬───────────────┘   └──────────────────┬──────────────────────┘  │
│             │ subprocess                            │ vendoring winners        │
│  ┌──────────▼───────────────────────────────────────▼──────────────────────┐  │
│  │ llama.cpp v0.2.0 @ bb4caa75 + Quilt Patch Overlay                        │  │
│  │ `patches/0001-gfx1100-mul-mat-custom.patch` (356 lines, 276 insertions)  │  │
│  │ OFF: `build-stock`     -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF (bit-ident.)│  │
│  │ ON:  `build-custom` /root/llama-custom-07 -D...=ON                       │  │
│  │ intercepts: mmq.cu:114 (prefill M>=16), mmvq.cu:1280 (decode M=1)        │  │
│  │ `ggml/src/ggml-cuda/custom_gfx1100/{gemv,gemm}_iq4xs.cuh`                │  │
│  └──────────────────────────────┬──────────────────────────────────────────┘  │
└─────────────────────────────────┼────────────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  AMD RX 7900 XT (gfx1100, RDNA3) — 15.31 GB IQ4_XS GGUF fully resident        │
│  Evidence: append-only `benchmarks/results/<ts>_<label>/rows.jsonl` +          │
│  CHECKSUMS.sha256 + kernels/matmul_iq4xs/*.hardware.json (all N=10)            │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Session orchestrator | Fingerprinted, guarded, pre-flighted benchmark sessions across ascending context tiers; spawns host daemons | `benchmarks/bin/run_session.py` |
| llama-bench wrapper | Explicit cell matrix construction (pure prefill `-p C`, decode `-pg C,128`); rejects default-cell contamination | `benchmarks/lib/llabench.py` |
| Guard / Preflight | Three-signal VRAM-spill/RSS/throughput-deviation detection; 18.25 GiB VRAM gate; verdict vocabulary (`OK`, `FAILED:suspected-spill`, `FAILED:preflight-oom`, `REVIEW:repeat-deviation`, `FAILED:thermal-abort`) | `benchmarks/lib/guard.py`, `benchmarks/lib/preflight.py` |
| Run store | Crash-resilient append-only result journaling: fsynced `rows.jsonl`, `CHECKSUMS.sha256`, `manifest.json` | `benchmarks/lib/store.py` |
| Fingerprinting | System/binary/model sha256 manifests for reproducibility | `benchmarks/lib/fingerprint.py` |
| Quality gates | Op-level gate (test-backend-ops QUAL-01, 4,243 ops) and model-level PPL + canary gate (QUAL-02, PPL 6.4271) | `benchmarks/bin/run_op_gate.py`, `benchmarks/bin/run_model_gate.py` |
| Variant race | Interleaved A,B,A,B thermal-paired racing of 5 GEMM + 2 GEMV variants, `--repeats 10`, winner = median AND mean−1σ ≥ 1.10× per tier; 8192 tier SKIPPED via VRAM preflight (offline-only, pruned in Phase 8) | `benchmarks/results/phase7/race.py` |
| Kernel playground | Standalone gfx1100 HIP kernels, op quartets, 32-shape matmul fixtures; zero ggml/llama headers | `kernels/CMakeLists.txt`, `kernels/matmul_iq4xs/`, `scripts/check_no_ggml.sh` |
| Real-stock DP4A comparator | Vendored exact upstream `quantize_row_q8_1` + `vec_dot_iq4_xs_q8_1` (DP4A `/__builtin_amdgcn_sudot4` + `perm` LUT); honest N=10 denominator (attn_q 87.8 µs vs naive 548.4 µs = 6.25×) | `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` (552 lines) |
| Coop DP4A GEMV (decode) | 8-thread/row cooperative, 32 rows/block, `sh[32][33]` (+33) vs XOR preshuffle (0%); two distinct OBJECTs; N=10 result: all <1.2× FAIL (XOR peak 1.161) | `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`, `kernels/matmul_iq4xs/gemv_variant_xor.cuh`, `bench_gemv_dp4a.cpp` |
| Streaming WMMA GEMM (prefill) | 5-variant OBJECT family: 64×32 P2+33 / 64×32 P4_XOR / 64×64 P4+XOR / 128×32 / LUT μ=4; `sched_barrier 0x0080/0x0008`, b128 16B loads, weak tiled helper; N=10: 64x64_P4_XOR M1024 1.929× PASS, M512 1.208× PASS, M128 0.041× FAIL | `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` (364 lines), `impl_gemm_lut_iq4xs.hip`, `bench_gemm_wmma.cpp` |
| In-tree dispatch | `can_handle` shape gates + guarded early-return intercept in GGML MUL_MAT paths; `empty.cuh` OFF fallback stub | `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv,gemm}_iq4xs.cuh` (95/120 lines), hooks in `llama.cpp/ggml/src/ggml-cuda/mmq.cu:114`, `mmvq.cu:1280` |
| Quilt patch integration | Switch-gated patch over pristine `bb4caa75` (`git -C llama.cpp diff bb4caa75`), `core.autocrlf=false`, `*.patch eol=lf` | `patches/0001-gfx1100-mul-mat-custom.patch` |
| Baseline archive | Frozen stock binaries, never rebuilt or overwritten | `baseline/binaries/v0.2.0-bb4caa75/` |
| Fixture extraction | GGUF tensor dumpers producing binary/npz fixtures with manifests + offline 16×64 swizzle/LUT bake | `tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py`, `tools/swizzle_iq4xs.py` |
| Windows-native build | HIP_PATH-quoted `clang++.exe --offload-arch=gfx1100` + `-G Ninja`; `curl :8000 → 200 choices` smoke; gated on HIP SDK (not executed on this host) | `build_windows.bat` |

## Pattern Overview

**Overall:** Measurement-first optimization harness (frozen-baseline discipline) feeding a standalone kernel-development pipeline whose winners are vendored into llama.cpp behind a compile-time switch via quilt patch overlays.

**Key Characteristics:**
- **Quilt overlay, not fork:** all in-tree change lives as `patches/0001-gfx1100-mul-mat-custom.patch` regenerated from `git -C llama.cpp diff bb4caa75` — reviewable, revertible, zero drift. Source of truth for vendored kernels is `kernels/matmul_iq4xs/`; `custom_gfx1100/*.cuh` copies carry provenance headers.
- **OFF/ON switch discipline:** `option(GGML_CUDA_ENABLE_CUSTOM_GFX1100 ... OFF)` in `llama.cpp/ggml/CMakeLists.txt:221` + `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)` guards in `mmq.cu`/`mmvq.cu`. OFF = bit-identical stock; ON fires only when `custom_{gemv,gemm}_iq4xs_can_handle()` shape gates pass. Never hardcode ON.
- **Real-stock comparator, not strawman:** Phase 7 replaced the naive scalar float comparator with vendored production DP4A (`real_stock_dp4a_comparator.hip`) so every speedup claim is vs the true upstream integer pipeline.
- **Compiled variant OBJECTs, not synthetic jitter:** 5 GEMM + 2 GEMV variants compile as distinct HIP OBJECT libraries (`matmul_gemm_wmma_{stream,p4_xor,64x64,lut}_hip`, `matmul_gemv_dp4a_{,xor}_hip`) with distinct exported symbols; `bench --variant <name>` dispatches to the real object (see `kernels/matmul_iq4xs/CMakeLists.txt` and `output/P8_VARIANT_COMPILE_PROPOSAL.md` — the earlier bench mapped 5 documented variants to the same compiled kernel; fixed).
- **Gate-before-claim + N=10/15 rigour:** `test_*_compare` cosine ≥0.999 vs FP64 CPU oracle before bench; every perf number is N=10 median/mean/stddev/p95 (LLM QA N=15) written to hardware JSONs; single-run claims banned (REQ-STAT-07).
- **Hard isolation:** `scripts/check_no_ggml.sh` enforces zero ggml/llama includes in `kernels/`; only vendored `kernels/common/block_iq4_xs.h` (136-byte IQ4_XS layout) is shared.
- **Append-only evidence:** results are timestamped, fsynced, checksummed journals; failures (`FAIL`, `SKIPPED`) are published exactly like wins — no fabricated PASS (REQ-PERF-07 honestly FAIL at llama-bench scale: 0.978–1.079×).

## Layers

**Host Telemetry Layer (Windows):**
- Purpose: GPU sensor capture and thermal protection outside the guest, plus the Windows-native build gate.
- Location: `benchmarks/host/`, `build_windows.bat`
- Contains: HWiNFO SM2 memory-mapped reader daemon (1 Hz), manual CSV fallback decoder, cross-boundary process-kill watchdog, HIP_PATH-aware CMake/Ninja build script.
- Depends on: Windows shared memory (`Global\HWiNFO_SENS_SM2`), `wsl.exe` for kills, `%HIP_PATH%\bin\clang++.exe` + `ninja` for the native build.
- Used by: session orchestrator (`run_session.py` spawns both daemons); operator on a Windows 11 HIP SDK host for REQ-WIN-07.

**Harness Orchestration Layer (Python CLIs):**
- Purpose: end-to-end guarded sessions, gates, profiling, matrix publication, variant racing.
- Location: `benchmarks/bin/`, `benchmarks/results/phase7/race.py`
- Contains: argparse-driven entry points; each CLI imports from `benchmarks.lib`.
- Depends on: harness libraries, pinned llama.cpp binaries, model at `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, benchmarks/results/phase7 for the race.
- Used by: operator/agent per `benchmarks/RUNBOOK.md`. Deferred for Python prune (40→0) in Phase 8 landing.

**Harness Library Layer (reusable modules):**
- Purpose: single-responsibility modules with pure-function cores (testable without GPU).
- Location: `benchmarks/lib/`
- Contains: `llabench.py`, `guard.py`, `preflight.py`, `store.py`, `fingerprint.py`, `parse_profile.py`, `toast.py`.
- Depends on: stdlib only plus thresholds config `benchmarks/config/thresholds.json`.
- Used by: everything in `benchmarks/bin/`, `benchmarks/vulkan/run_session_vulkan.py`, `benchmarks/tools/run_kernel_bench.py`.

**Kernel Playground Layer (standalone HIP/C++):**
- Purpose: develop custom gfx1100 kernels completely decoupled from llama.cpp; gate them numerically; microbench them with N=10.
- Location: `kernels/`
- Contains: shared headers (`kernels/common/`), op quartets in `kernels/template/`, `kernels/demo_iq4xs_dequant/`, `kernels/matmul_iq4xs/` (ref_cpu oracle + stock + real-DP4A comparators + GEMV/GEMM impl families + test/bench binaries per kernel), fixtures in `kernels/fixtures/` with manifests.
- Depends on: HIP runtime only (`hip::device` via `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`); vendored `block_iq4_xs.h`.
- Used by: Phase 5/6/7 integration path via quilt patches; benchmarks via `benchmarks/tools/run_kernel_bench.py`.

**In-Tree Integration Layer (llama.cpp Overlay):**
- Purpose: hook custom gfx1100 kernels into GGML CUDA/HIP execution graph behind the compile switch.
- Location: `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/` (`gemv_iq4xs.cuh`, `gemm_iq4xs.cuh`, `empty.cuh`, `README.md`)
- Intercepts: `mmvq.cu:1280` (decode M=1, `M==1` + canonical shapes) and `mmq.cu:114` (prefill M≥16, real gate `type==IQ4_XS && M>=16 && K%256==0 && N%16==0`; WMMA path additionally needs `M>=512 && N>=32 && K>=32 && 16-alignment`, else falls back to tiled kernel).
- Switch: `GGML_CUDA_ENABLE_CUSTOM_GFX1100` (default OFF; ON fires only when `can_handle` true).
- Used by: GGML HIP build only when ON; `build-custom` coexists with `build-stock` from one tree.

**Evidence Layer (append-only journals + hardware JSONs):**
- Purpose: reproducible gapless record of every run.
- Location: `benchmarks/results/`, `kernels/matmul_iq4xs/*.hardware.json`, `benchmarks/profiling/`
- Contains: `rows.jsonl`, `CHECKSUMS.sha256`, `manifest.json`, N=10 hardware JSONs (`bench_real_stock.hardware.json` 8 entries, `bench_gemv_dp4a.hardware.json` 8, `bench_gemv_xor.hardware.json` 8, `bench_gemm_wmma.hardware.json` 15), N=10 paired llama-bench JSONs, honest FAIL tables in `KERNEL-BENCH-DIFF.md §8` and `docs/PUBLICATION.md §8`.

## Data Flow

### Primary Request Path (llama-bench A/B session)

1. `benchmarks/bin/run_session.py` selects tier {512,1024,2048,4096} with `-ngl 99 -b 2048 --single-turn --simple-io --load-mode none`, sets `HSA_ENABLE_DXG_DETECTION=1`, acquires `benchmarks/results/.session.lock`, starts `hwinfo_daemon.py` (1 Hz) + `thermal_watchdog.py` (90 °C race / 95 °C hard).
2. Preflight checks free VRAM ≥ 18.25 GiB (`benchmarks/lib/preflight.py`); fail → `FAILED:preflight` abort (no retry loop). The 8192 tier additionally fails the `hipMalloc` probe and is recorded `SKIPPED` — FA+GQA 15.3 GB model + 128 KiB/tok KV ≈ 18.5 GB on 20 GB, WSL2 `800 GiB` VRAM lie + BSOD risk (microsoft/WSL#40732).
3. `llama-bench -r 10` runs N=10 per cell; stock and custom interleaved `A,B,A,B` via `race.py --repeats 10` (adelj88 thermal-bias-kill pattern) inside one thermal window.
4. GGML dispatches `MUL_MAT`: `mmvq.cu` (M=1 decode) / `mmq.cu` (M≥16 prefill). With `GGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` and `can_handle()` true, the early-return intercept at `mmvq.cu:1280` / `mmq.cu:114` runs the vendored `custom_gfx1100/{gemv,gemm}_iq4xs.cuh`; otherwise stock `vec_dot_iq4_xs_q8_1` DP4A runs.
5. Result rows stream to `benchmarks/results/<ts>_<label>/rows.jsonl` via `RunStore.append_row` (fsynced, append-only) with fingerprint (commit, ROCm/driver, GGUF sha256, clocks/temps per row); run closes with `CHECKSUMS.sha256`.
6. `publish_matrix.py` aggregates; verdict asserted only after QUAL-01 0 errors and QUAL-02 within 1% gates.

### Phase 7 High-Yield Inner Pipeline (07-02, inside the WMMA kernel)

`Q8_1` activation quant (`amax/127` → `half2 ds`) → LDS `[2..4][32][33]` double-buffer (`P=2` stride-33 `+3%` or `P=4` XOR preshuffle `x'=(y%(K_TILE/8))⊕x` 0%) with `__builtin_amdgcn_sched_barrier(0x0080)` (DS) / `0x0008` (WMMA) pinning GMEM→VGPR→LDS→VGPR→WMMA 4-stage overlap → B-stationary weight frag `v16f16` in VGPRs (64× reuse at `T=64`) / activations streamed via LDS → `b128` 16B coalesced loads (`ulong2`, `float4`, `__builtin_assume_aligned(ptr,16)`) → WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (wave32, 1024 ops/CU/clock); variant winner picked by `speedup_median` then `speedup_mean_minus_1sigma`. Offline companion: `tools/swizzle_iq4xs.py` 16×64 swizzle to 128B lines + LUT bake (host-only, not shipped).

### Kernel Playground Pipeline (per op)

`ref_cpu.cpp` (FP64 oracle) → `impl*.hip` (gfx1100) → `test_*_compare.cpp` (cosine ≥ 0.999 gate; failures recorded like successes) → `bench_*.cpp --runs 10 --json` (N=10 median/mean/stddev/p95, vs `real_stock_dp4a_comparator.hip`).

**State Management:** stateless kernels (all state in params + `__shared__` LDS + const device tables); harness state persists only in append-only RunStore journals. No server, no long-lived process state.

## Key Abstractions

**`block_iq4_xs` (136 B) + `block_q8_1_coop` (64 B):**
- Purpose: weight + activation quantization layouts; 64 B pad puts `qs` at 16B-aligned offset for `ulong2` b128 loads.
- Examples: `kernels/common/block_iq4_xs.h` (vendored), `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` (`struct block_q8_1_coop` — distinct name to avoid ODR clash with the comparator).
- Pattern: plain structs; dequant tables (`kvalues_iq4nl`, `kvalues_iq4nl_dev_gemm`) are `__device__ __constant__`.

**`coop_dp4a` / `vec_dot_iq4_xs_q8_1_device`:**
- Purpose: the production integer dot — `ls=(scales_l…)|…; scale=ls-32; sumi=ggml_cuda_dp4a(v,u)` via `__builtin_amdgcn_sudot4` + `__builtin_amdgcn_perm` LUT (`get_int_from_table_16`).
- Examples: `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip:145`, `impl_gemv_dp4a_gfx1100.hip:91`.
- Pattern: exact port of upstream `vecdotq.cuh`/`mmvq.cu` logic.

**`gemv_iq4xs_dp4a_coop_kernel<WARP_SIZE>`:**
- Purpose: template decode kernel — 8-thread/row, 32 rows/block, `sh[32][33]` (+33) or XOR (via `gemv_variant_xor.cuh`, second OBJECT), `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)` → ≤64 VGPR / 16 waves/SIMD.
- Examples: `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip:180`.

**`gemm_iq4xs_wmma_stream_gpu` (5-variant family):**
- Purpose: prefill WMMA kernel family compiled per-variant via `#define`-driven macros (`WMMA_BLOCK_N/M`, `USE_XOR`, `P`, LUT), distinct exported symbols per OBJECT.
- Examples: `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` (launch `:330`, `__attribute__((weak))` tiled helper `:353`).
- Pattern: `hipLaunchKernelGGL` with WMMA gate `M>=512 && N>=32 && K>=32 && 16-align`, fallback to weak tiled helper on launch error.

**`custom_{gemv,gemm}_iq4xs_can_handle` + `dispatch`:**
- Purpose: guarded dispatch intercept — real shape gates (not stubs): gemv `type==IQ4_XS` + `M==1` + canonical shapes; gemm `type==IQ4_XS && M>=16 && K%256==0 && N%16==0`; early-return on `err==hipSuccess` keeps stock path otherwise.
- Examples: `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv,gemm}_iq4xs.cuh` (gemv:113, gemm:87), intercepts `mmvq.cu:1280` / `mmq.cu:114`.

**`RunStore` + `CHECKSUMS.sha256`:**
- Purpose: append-only fingerprinted journal; `create` → `append_row` (fsync) → `write_checksums`; never overwrite rows.
- Examples: `benchmarks/lib/store.py`.

**`race.py --repeats 10`:**
- Purpose: interleaved A,B,A,B variant racing + paired llama-bench driver across tiers {512,1024,2048,4096,8192}; 8192 conditionally SKIPPED; winner gate median AND mean−1σ ≥ 1.10× per tier per split.
- Examples: `benchmarks/results/phase7/race.py` (offline-only; pruned in Phase 8).

## Entry Points

**`benchmarks/bin/run_session.py`:**
- Location: `benchmarks/bin/run_session.py`
- Triggers: operator/agent per `benchmarks/RUNBOOK.md`
- Responsibilities: tier selection, preflight gate, daemon orchestration, guarded llama-bench execution, RunStore journaling.

**`benchmarks/results/phase7/race.py`:**
- Location: `benchmarks/results/phase7/race.py`
- Triggers: `python benchmarks/results/phase7/race.py --repeats 10 --tiers 512,...,8192`
- Responsibilities: interleaved variant race (5 GEMM + 2 GEMV, optional W8A8 `--include-w8a8`), VRAM preflight for 8192, thermal pairing, winner pick, RunStore rows + CHECKSUMS; exit 0 only if median ≥ 1.10× at all non-8192 tiers.

**`kernels/matmul_iq4xs/bench_gemm_wmma` / `bench_gemv_dp4a`:**
- Location: `kernels/matmul_iq4xs/bench_gemm_wmma.cpp`, `bench_gemv_dp4a.cpp` (built in `kernels/build/matmul_iq4xs/`)
- Triggers: `./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json [--variant <name>]`
- Responsibilities: per-variant N=10 microbench vs real-stock DP4A; emits incremental JSON lines per entry (avoids 12,288 B truncation); 8192 tier emits `winner:"SKIPPED"` with VRAM-preflight note; soft HIP_CHECK path on OOM probe (no abort).

**`test_gemv_dp4a_compare` / `test_gemm_wmma_compare`:**
- Location: `kernels/matmul_iq4xs/test_gemv_dp4a_compare.cpp`, `test_gemm_wmma_compare.cpp`
- Triggers: `ctest` in `kernels/build/` or direct `cmake --build`
- Responsibilities: numerical correctness gate (cosine ≥ 0.999 vs FP64 CPU oracle) before any bench claim.

**`build_windows.bat`:**
- Location: `build_windows.bat`
- Triggers: operator on a Windows 11 host with HIP SDK (`%HIP_PATH%\bin\clang++.exe`)
- Responsibilities: REQ-WIN-07 native gate — `cmake -S . -B build-windows -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON …, llama-server.exe :8000 → 200` smoke; not executed on this host (no HIP SDK).

**`kernels/CMakeLists.txt` (top-level playground build):**
- Location: `kernels/CMakeLists.txt`
- Triggers: `cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100`
- Responsibilities: standalone HIP build — `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` with `/opt/rocm` fallback; targets for template/demo/matmul_iq4xs.

## Architectural Constraints

- **Threading:** HIP single-stream model. Kernels are Wave32 (`amdgpu_flat_work_group_size(256,256)`, `__launch_bounds__(256,4)` → ≤64 VGPR, 16 waves/SIMD target). Host side is a single main thread per CLI (daemons run as separate OS processes/threads). Grid shapes: GEMV `ceil(N/32)`, GEMM `grid_x=N/64, grid_y=M/32`.
- **Global state:** device `__constant__` tables only (`kvalues_iq4nl` in `real_stock_dp4a_comparator.hip` and `kvalues_iq4nl_dev_gemm` in `impl_gemm_wmma_stream.hip` / `gemm_iq4xs.cuh`). No host-side mutable singletons except RunStore paths configured per-run.
- **ODR across variant OBJECTs:** the tiled fallback helper is declared `__attribute__((weak))` (`impl_gemm_wmma_stream.hip:353`) so 5 GEMM OBJECTs link into one binary without duplicate-symbol errors. GEMV avoids the issue by distinct struct names (`block_q8_1_coop`) and a dedicated XOR header (`gemv_variant_xor.cuh`).
- **Soft HIP_CHECK:** the hard `HIP_CHECK` macro (`kernels/common/hip_helpers.h:8`, abort-on-error) is used for setup paths; bench paths use a log-and-skip soft path — on `hipErrorOutOfMemory`/hipError 9 (grid/large alloc overflow) the bench emits `winner:"SKIPPED"` JSON with a VRAM-preflight note and continues, never aborts, never retries (threat model T-07-02-01).
- **OFF/ON switch:** custom dispatch compiles out entirely when `GGML_CUDA_ENABLE_CUSTOM_GFX1100` is OFF (guards in `mmq.cu`/`mmvq.cu`); `empty.cuh` stub keeps the ON-but-empty build bit-identical.
- **8192 tier always SKIPPED:** VRAM preflight >2 GB free + `hipMalloc` probe; FA+GQA 15.3 GB + 128 KiB/tok KV → ~18.5 GB on 20 GB; WSL2 lies about VRAM + BSOD risk → recorded, never forced.
- **Circular import / vendoring chain:** `kernels/matmul_iq4xs/*.hip` → vendored verbatim into `custom_gfx1100/*.cuh` → quilt patch generated from `git diff bb4caa75` → patch applies back onto `llama.cpp`. The playground must stay source-of-truth; edits to the `.cuh` copies alone drift the pipeline (the 07-VERIFICATION attestation keeps them in sync by grep).
- **Bash discipline:** every harness/bash subprocess must carry an explicit bounded timeout (rule 11) — `bench_gemm_wmma` originally truncated output at 12,288 B on `timeout 90`; incremental `fprintf+fflush` per entry fixes capture.

## Anti-Patterns

### WMMA on-the-fly dequant at small M (M=128)
**What happens:** `impl_gemm_wmma_stream.hip` computes scalar `dl = d*(ls-32)*kvalues` per element on-the-fly into `a_frag`; at M=128 the dequant + LDS overhead dominates — N=10 result ~0.041× (24× slower than real DP4A).
**Why it's wrong:** WMMA's 1024 ops/CU/clock only pays off when B-stationary reuse amortizes the decode; at small M the integer DP4A 9 TFLOPS path wins.
**Do this instead:** keep the `wmma_ok` gate (`M>=512`) with tiled fallback below it; verify new variants against `real_stock_dp4a_comparator.hip` (never the naive float loop) at every M tier. The 64×64 B-stationary variant (64× reuse) proves the fix direction — M512 1.208×, M1024 1.929× PASS.

### Documented variants that all compiled to the same kernel
**What happens:** earlier `bench_gemm_wmma.cpp` documented 5 variants (`64x32_P2+33` … `LUT_mu4`) and `--variant race`, but all mapped to the single P=2 64×32 compiled kernel — racing compared identical code (see `output/P8_VARIANT_COMPILE_PROPOSAL.md`).
**Why it's wrong:** fabricated differentiation produced synthetic jitter instead of measured deltas.
**Do this instead:** one OBJECT library per variant with distinct `target_compile_definitions` (`GEMV_XOR`, `GEMM_P4_XOR`, `TILE_64x64`) in `kernels/matmul_iq4xs/CMakeLists.txt`; `--variant` dispatches to the real object; verify distinct symbols via `nm -D`.

### `can_handle` stub returning false
**What happens:** `gemm_iq4xs.cuh` shipped a `custom_gemm_iq4xs_can_handle` stub `return false`, silently disabling the WMMA dispatch even when the kernel was vendored.
**Why it's wrong:** OFF-behavior disguised as ON; a perf regressions hides behind an unreachable branch.
**Do this instead:** real shape gates (`M>=16 && K%256==0 && N%16==0`), mirror the bench-side `wmma_ok` predicates, and gate the patch with `grep -q "can_handle"` verification (07-VERIFICATION.md).

### Truncated JSON capture
**What happens:** `bench_gemm_wmma --runs 10` output truncated at 12,288 B under `timeout 90` produced `bench_gemm_direct.json` with a JSON parse error — flagged needs-regen, never claimed.
**Why it's wrong:** a corrupted evidence artifact can masquerade as a run.
**Do this instead:** incremental `fprintf(stdout, ...) + fflush` per metric entry (already in `bench_gemm_wmma.cpp`); `validationOutput`-style parse checks (`python3 -m json.tool`) before any number is used.

## Error Handling

**Strategy:** fail-fast at correctness gates, fail-soft at resource boundaries, record-don't-fabricate at the verdict layer.

**Patterns:**
- Hard `HIP_CHECK` macro (abort + stderr) for setup/alloc/event code paths: `kernels/common/hip_helpers.h`, `kernels/common/bench.h`.
- Soft HIP_CHECK for OOM/grid-overflow at bench time: log-and-skip, emit `winner:"SKIPPED"` + note, continue to next tier (no retry loops): `kernels/matmul_iq4xs/bench_gemm_wmma.cpp:83-118`.
- Kernel launch fallback: WMMA launch error → return code → fall through to weak tiled helper: `impl_gemm_wmma_stream.hip:330-348`.
- Guard/preflight verdict vocabulary: `OK`, `FAILED:suspected-spill`, `FAILED:preflight-oom`, `REVIEW:repeat-deviation`, `FAILED:thermal-abort`, `KILLED:thermal@90C`, `SKIPPED` (8192) — checked by `benchmarks/lib/guard.py` + `preflight.py`.
- Statistical honesty: `median >= 1.10×` AND `mean−1σ >= 1.10×` both required for REQ-PERF-07; `NaN`/`SKIPPED` tiers excluded, never counted as PASS (race.py + 07-VERIFICATION.md).

## Cross-Cutting Concerns

**Logging:** harness — `benchmarks/host/hwinfo_daemon.py` (1 Hz SHM poll, record-don't-control) + `benchmarks/host/thermal_watchdog.py`; kernel benches — incremental JSON to stdout with `fflush`; llama-bench `.log` files in `benchmarks/results/phase7/`.
**Validation:** two-tier gates — QUAL-01 op-gate (test-backend-ops, 0 errors) and QUAL-02 model-gate (PPL 6.4271 ±1%, 6/6 canaries) at `benchmarks/bin/run_op_gate.py` / `run_model_gate.py`; kernel-level cosine ≥ 0.999 vs FP64 CPU oracle in `kernels/matmul_iq4xs/test_*_compare.cpp`; candidate gates arm before any bench claim.
**Authentication:** not applicable (local tooling; no external services; `tools/ask_model.py` optional local-model CLI).
**Reproducibility:** fingerprint manifests (`benchmarks/lib/fingerprint.py`), frozen binaries (`baseline/binaries/v0.2.0-bb4caa75/`), pinned upstream `bb4caa75`, pinned ROCm 7.2.1, `.gitattributes` `*.patch eol=lf` + `core.autocrlf=false`.

---

*Architecture analysis: 2026-08-30*