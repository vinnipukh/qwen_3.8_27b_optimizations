# Phase 8: Refactor to Windows Native — Context

**Date:** 2026-08-28
**Mode:** standard (horizontal layers)
**Goal:** Repository builds and runs natively on Windows 11 for RX 7900 XT gfx1100 via AMD HIP SDK + VS Build Tools, reduced to pure C++/HIP + minimal CMake, no Python/JS, serving llama-server.exe on localhost:8000.

## Decisions (binding)

### D-01 — Bloat elimination (OBJ-1)
Remove all auxiliary dirs, multi-phase benchmark suites, Python monitoring/thermal daemons, FreeToken/JavaScript, planning logs, complex JSON/JSONL harnesses. Reduce to pure C++/HIP + minimal CMake. No Python/JS remains in tree or git history tracked files.
- Source: user objectives list item 1, acceptance contract review-findings.
- Rationale: WSL2 ROCm probe period is over; production path is Windows HIP SDK native.

### D-02 — Core kernels retained (OBJ-2)
Keep only IQ4_XS essentials for gfx1100: DP4A GEMV (decode) and WMMA GEMM (prefill), minimal headers, clean patch.
- Retained: `kernels/common/block_iq4_xs.h`, `kernels/common/hip_helpers.h`, `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`, `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` (or `impl_gemm_wmma.hip` — canonical WMMA kernel), `patches/0001-gfx1100-mul-mat-custom.patch`.
- Conditionally retained if needed for validation: `kernels/common/bench.h`, `kernels/common/matmul_test_util.h`, `kernels/matmul_iq4xs/ref_cpu.h/.cpp`.
- Everything else in `kernels/` is deleted (demo_iq4xs_dequant, template, fixtures, synthetic binaries, leftover naive `impl_gemv_gfx1100.hip`/`impl_gemm_wmma.hip` duplicates unless canonical).

### D-03 — Windows build pipeline (OBJ-3)
Provide foolproof `build_windows.bat` using cmake + ninja + Clang from AMD HIP SDK (HIP_PATH) to compile `llama-server.exe` targeting gfx1100. AMDGCN intrinsics (`__builtin_amdgcn_sudot4`, `__builtin_amdgcn_perm`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) must compile on Windows without MSVC errors (use hipcc as compiler, not cl).
- Toolchain: Visual Studio C++ Build Tools (cl for host, but HIP files via hipcc → clang++ at `%HIP_PATH%\bin\clang++.exe`), cmake 3.21+, ninja.
- Target: `-DCMAKE_HIP_ARCHITECTURES=gfx1100` / `--offload-arch=gfx1100`, no `amdgpu-arch` auto-detect.

### D-04 — Output & usage (OBJ-4)
Clean minimal tree: `kernels/`, `patches/`, `CMakeLists.txt` (root), `build_windows.bat`. Compiled `llama-server.exe` runs directly on Windows to serve OpenAI API at `localhost:8000`.
- No WSL dependency for build or run.
- Model path handling uses Windows native paths (`E:\models\...` or `models\Qwen...gguf`), not `/root/models` or `/mnt/*`.

### D-05 — Llama.cpp provenance
`llama.cpp/` pinned at `bb4caa75` remains as the only large external source tree (or fetched via CMake FetchContent / git submodule if size-constrained). It is NOT bloat — it is the build input. Patch applies via `git apply` / `patch` on Windows. Keep `.gitmodules` or `CMakeLists.txt` FetchContent pointing to `https://github.com/ggml-org/llama.cpp@bb4caa75`.

### D-06 — Mode standard, horizontal layers
Phase 8 is refactor, not new kernel research. No tracer prototype; plans are horizontal slices (inventory → prune → build → verify).

## Canonical References

- Current repo at `c7f63db8` on `main`, README stock vs custom table (stock 808 vs custom 849 pp4096 +5.1%).
- Phase 7 state: hybrid DP4A GEMV + WMMA GEMM, 28/28 plans, verifier 2/5 gaps, stock vs custom 849 pp4096, quilt patch 30 lines GEMM disabled for correctness (`return false` in `custom_gemm_iq4xs_can_handle`), persistent `/root/llama-custom-07`, WSL2 ROCm 7.2.1.
- Kernels essentials: `kernels/common/block_iq4_xs.h` (136B, vendored from ggml/src/ggml-common.h@bb4caa75), `kernels/common/hip_helpers.h`, `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` (15186B, 8-thread/row coop, `sh[32][33]`, `__launch_bounds__(256,4)`, `__builtin_amdgcn_sudot4`+perm), `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` (13610B, 64x32 per block, `sB[2][32][33]`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`).
- Patches: `patches/0001-gfx1100-mul-mat-custom.patch` (clean 355 lines, 276 insertions, `git apply --check` PASS, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF default).
- Root build entry: no current `CMakeLists.txt` at repo root; `kernels/CMakeLists.txt` uses `find_package(hip REQUIRED)` + `--offload-arch=gfx1100` + `CMAKE_HIP_ARCHITECTURES=gfx1100`.
- Recipe references: `docs/ARCHITECTURE.md` (quilt overlay, OFF/ON switch, real-stock DP4A comparator), `.planning/codebase/ARCHITECTURE.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/REQUIREMENTS.md`.
- Windows HIP SDK: HIP_PATH env (e.g. `C:\Program Files\AMD\ROCm\6.2`), `clang++.exe` at `%HIP_PATH%\bin\clang++.exe`, `hipcc.bat` wrapper, driver 32.0.31041.1004, VRAM 20GB.

## Existing Code Insights

- `kernels/CMakeLists.txt` already correctly sets `CMAKE_HIP_ARCHITECTURES=gfx1100` cache, `add_compile_options(--offload-arch=gfx1100)`, `find_package(hip CONFIG PATHS /opt/rocm/lib/cmake/hip)`. On Windows this must change to `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` or rely on `HIP_PATH` env, and `clang++` must be the HIP compiler, not `cl`.
- `kernels/matmul_iq4xs/CMakeLists.txt` defines interface `matmul_common_iface`, object libs for `matmul_stock_hip`, `matmul_real_stock_hip`, `matmul_gemv_dp4a_hip`, `matmul_gemm_wmma_stream_hip`, and executables `test_real_stock_compare`, `bench_real_stock`, `test_gemv_dp4a_compare`, `bench_gemv_dp4a`, `test_gemm_wmma_compare`, `bench_gemm_wmma`. For minimal Windows tree, only the two winner kernels + `bench.h`/`matmul_test_util.h` + `ref_cpu` are needed; the naive comparators and demo can be deleted but `ref_cpu` is cheap to keep for cosine validation.
- `patches/0001...` currently disables GEMM (`return false`) to avoid corruption — Windows plan must document re-enable path: audit `TILE_M=16` fallback + `M>=512` guard before returning true.
- `.gitignore` currently ignores `kernels/build/`, `models/*.gguf`, `baseline/binaries/`, etc. After bloat removal it should ignore `build/`, `build-windows/`, `*.exe`, `*.pdb`, `*.obj`, but NOT ignore `kernels/` essentials.
- `block_iq4_xs.h` uses `#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)` → portable to Windows clang; `hip_helpers.h` includes `<hip/hip_runtime.h>` which on Windows resolves via HIP SDK include path.
- Intrinsic portability: `__builtin_amdgcn_sudot4` and `__builtin_amdgcn_perm` are clang builtins, not MSVC. They compile only when HIP file is compiled with `hipcc`/`clang++ --offload-arch=gfx1100`. If `cl` tries to compile `.hip` as C++, it will error. Build must set `CMAKE_HIP_COMPILER` to HIP clang.

## Allowlist vs Blocklist (explicit, exhaustive scan 2026-08-28)

### Allowlist — KEEP (must justify gfx1100 IQ4_XS acceleration or Windows build)

| Path | Justification |
|------|---------------|
| `README.md` | Minimal project readme + stock vs custom table (update for Windows native) |
| `LICENSE` | Apache 2.0 |
| `NOTICE` | Qwen/base attribution |
| `CHANGELOG.md` | Release history |
| `.gitignore` | Minimal: `build*/`, `*.exe`, `*.obj`, `kernels/build/` |
| `CMakeLists.txt` (root, NEW) | Top-level: `project(qwen-gfx1100-hip)`, `find_package(hip)`, `add_subdirectory(kernels)`, `add_subdirectory(llama.cpp)` or FetchContent, option `GGML_CUDA_ENABLE_CUSTOM_GFX1100` |
| `build_windows.bat` (NEW) | Foolproof Windows build: checks HIP_PATH, VS env, cmake+ninja, builds `llama-server.exe` gfx1100 |
| `kernels/CMakeLists.txt` | Reduced: only `common` + `matmul_iq4xs` (pruned) |
| `kernels/common/CMakeLists.txt` | Interface lib `kernels_common` |
| `kernels/common/block_iq4_xs.h` | Vendored 136B IQ4_XS layout + `kvalues_iq4nl[16]` + fp16 helpers (per D-02) |
| `kernels/common/hip_helpers.h` | HIP error macros + WARP_SIZE templating |
| `kernels/common/bench.h` | `bench_hip_event` microbenchmark helper (needed for cosine/bench smoke) |
| `kernels/common/matmul_test_util.h` | `compute_metrics` + `gen_iq4xs_weights` (keep for validation; delete if redundant) |
| `kernels/matmul_iq4xs/CMakeLists.txt` | Pruned: only `matmul_ref_cpu`, `matmul_gemv_dp4a_hip`, `matmul_gemm_wmma_stream_hip`, tests/benches |
| `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` | WINNER decode kernel per D-02 |
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | WINNER prefill WMMA kernel per D-02 (rename to `impl_gemm_wmma.hip` if preferred) |
| `kernels/matmul_iq4xs/ref_cpu.h` + `ref_cpu.cpp` | FP64 oracle for cosine check (keep minimal) |
| `kernels/matmul_iq4xs/test_gemv_dp4a_compare.cpp` + `bench_gemv_dp4a.cpp` | GEMV correctness + bench (optional but validates >1.2x claim) |
| `kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp` + `bench_gemm_wmma.cpp` | WMMA correctness + bench |
| `patches/0001-gfx1100-mul-mat-custom.patch` | Clean quilt overlay over bb4caa75 per D-02 |
| `llama.cpp/` (submodule or FetchContent) | Pinned upstream `bb4caa75` — build input for `llama-server.exe` (or clone via CMake if not vendored) |
| `models/README.md` (minimal) | Model provenance: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` sha256 `53adc4bb…` + Windows path note |
| `.planning/phases/08-refactor-windows-native/` | This phase plans only (ephemeral; not shipped) — excluded from final bloat-free tree via `.gitignore` or deleted after tag |

### Blocklist — DELETE (bloat, per D-01 / user mandate)

| Path | Reason |
|------|--------|
| `.planning/` (phases 01-07, research, codebase, reference) | Planning logs, 7-phase history |
| `benchmarks/` (`bin/run_session.py`, `lib/guard.py`, `lib/preflight.py`, `lib/store.py`, `lib/fingerprint.py`, `lib/llabench.py`, `host/hwinfo_daemon.py`, `host/thermal_watchdog.py`, `profiling/`, `results/`, `golden/`, `prompts/`, `vulkan/`, `config/`, `environment/`, `data/`, `tests/`, `tools/`) | Multi-phase benchmark suites, Python monitoring/thermal daemons, complex JSON/JSONL harnesses |
| `freetoken-rocm-probe/` (`bin/bench_*.exe`, `src/bench_*.cpp`, `src/qstar.mjs`, `tools/`, `zigidx.json`) | FreeToken/JavaScript probe |
| `scrape_out/` | Scraped web output |
| `.rocprofv3/` | Profiling scratch |
| `tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py`, `tools/gemm_iq4xs.cuh`, `tools/gemv_iq4xs.cuh` | Fixture dumpers + vendored cuh copies |
| `kernels/demo_iq4xs_dequant/` | Demo op (D4-00-1 dequant-only demo — superseded by matmul winners) |
| `kernels/template/` | Op quartet skeleton |
| `kernels/fixtures/` (130+ `.bin`/`.npz`/`.f32.bin`) | GGUF-extracted fixtures (regeneratable; not needed for Windows build, keep one canary if desired else delete) |
| `kernels/build/` | Build artifacts |
| `kernels/results -> ../benchmarks/results` symlink | Stale symlink |
| `kernels/matmul_iq4xs/stock_hip_comparator.hip`, `real_stock_dp4a_comparator.hip` (25KB), `impl_gemv_gfx1100.hip`, `impl_gemm_wmma.hip` (naive), `bench_gemv.cpp`, `bench_gemm.cpp`, `bench_gemm_wmma.cpp` (old), `bench_matmul.cpp`, `bench_real_stock.cpp`, `test_stock_compare.cpp`, `test_real_stock_compare.cpp`, `test_gemv_compare.cpp`, `test_gemm_compare.cpp` | Naive/duplicate comparators and non-canonical kernels (keep only the two winners + their tests) |
| `benchmarks/results/` (30+ dated run dirs, `BASELINE-MATRIX.*`, `kernels_*`, `phase3/`, `phase6/`, `smoke_tracer*`) | JSON/JSONL RunStore + CHECKSUMS history |
| `baseline/` | Frozen stock binaries (Windows rebuilds fresh) |
| `docs/` (`ARCHITECTURE.md`, `CONFIGURATION.md`, `DEVELOPMENT.md`, `GETTING-STARTED.md`, `PUBLICATION.md`, `QWEN-GRAPH.md`, `TESTING.md`) | Published methodology — archive elsewhere; minimal tree keeps only `README.md` |
| `src/` | Placeholder empty dir |
| `scripts/check_no_ggml.sh` | Isolation gate (not needed post-cleanup; HIP headers are gone) |
| `.agents/` (`skills/magpie-kernel-evaluator/`, `skills/rocm-doctor/`, `AGENTS.md`) | Agent skills |
| `logs/` (`build.log`, `gate.log`, `thermal_monitor.log`) | Runtime logs |
| `models/*.gguf` (actual 15GB weight) | Gitignored anyway; `models/README.md` keeps provenance only |
| `scrape_sahibinden.sh`, `skills-lock.json` (if not needed), `qstar.mjs` | Misc JS/shell |
| Root-level `AGENTS.md`, `CONTRIBUTING.md` (optional) | Methodology rules (11 rules) — keep only if desired; per bloat-free principle delete unless needed for OSS hygiene |

> **Rule:** Every file kept must justify gfx1100 IQ4_XS acceleration or Windows build. If neither, it is deleted.

## Specifics

- Windows HIP SDK paths: `HIP_PATH` env var (installer sets `C:\Program Files\AMD\ROCm\6.2` or `C:\hip`). `hipcc` is `%HIP_PATH%\bin\hipcc.bat`, Clang is `%HIP_PATH%\bin\clang++.exe` (amdclang 17/18). Do NOT use `cl.exe` for `.hip` files.
- `build_windows.bat` contract:
  ```bat
  @echo off
  setlocal
  if not defined HIP_PATH set HIP_PATH=C:\Program Files\AMD\ROCm\6.2
  where cmake >nul 2>&1 || (echo cmake not found & exit /b 1)
  where ninja >nul 2>&1 || (echo ninja not found & exit /b 1)
  if not exist "%HIP_PATH%\bin\clang++.exe" (echo HIP clang not found at %HIP_PATH% & exit /b 1)
  cmake -S . -B build-windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang++.exe" -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DLLAMA_BUILD_SERVER=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON
  cmake --build build-windows --config Release
  where llama-server.exe  (expect build-windows\bin\llama-server.exe or build-windows\llama-server.exe)
  ```
  Use `%~dp0` for repo root, quote paths with spaces, fail fast with `exit /b 1` on missing deps.
- Intrinsic portability: `__builtin_amdgcn_sudot4` requires `__HIP_DEVICE_COMPILE__` + `__gfx1100__` target; guard with `#ifdef __HIP_DEVICE_COMPILE__` if needed. `__builtin_amdgcn_perm` is byte-perm, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` requires Wave32 (`-mwavefrontsize32` or `__launch_bounds__` + `amdgpu_flat_work_group_size(256,256)` already in kernel).
- Root `CMakeLists.txt` minimal skeleton:
  ```cmake
  cmake_minimum_required(VERSION 3.21)
  project(qwen-gfx1100-hip LANGUAGES HIP CXX)
  set(CMAKE_HIP_ARCHITECTURES gfx1100)
  set(CMAKE_HIP_STANDARD 17)
  add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1100>)
  option(GGML_CUDA_ENABLE_CUSTOM_GFX1100 "Enable gfx1100 custom IQ4_XS kernels" ON)
  add_subdirectory(kernels)  # optional: only for standalone tests
  add_subdirectory(llama.cpp)  # provides llama-server, llama-cli, ggml
  ```
- Model path on Windows: `models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf` relative or `E:\models\...` absolute. Batch/script must not hardcode `/root/models` or `/mnt/e/...`. Use `.\models\` or env `MODEL_PATH`.
- Server smoke: `build-windows\bin\llama-server.exe -m models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf --host 127.0.0.1 --port 8000 --n-gpu-layers 99` → `curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"` → `200` with `choices[0].message.content`.

## Risks

| # | Risk | Severity | Mitigation in plans |
|---|------|----------|---------------------|
| R-01 | HIP SDK on Windows path divergence (`/opt/rocm` vs `HIP_PATH`, `find_package(hip)` location, `clang++.exe` vs `hipcc.bat`) | High | 08-03 forces `CMAKE_HIP_COMPILER=$HIP_PATH/bin/clang++.exe`, `CMAKE_PREFIX_PATH=$HIP_PATH/lib/cmake/hip`, explicit `where` checks, no `amdgpu-arch` |
| R-02 | Ninja vs MSBuild generator — MSBuild tries to compile `.hip` with `cl` and fails on `__builtin_amdgcn_*` | High | 08-03 mandates `-G Ninja`, documents MSBuild trap, `build_windows.bat` fails fast if generator missing |
| R-03 | Intrinsic portability — `__builtin_amdgcn_sudot4`/`perm`/`wmma` not found if compiled with `cl` or without `--offload-arch=gfx1100` | High | 08-03 validates intrinsics via standalone `check_intrinsics.hip` probe, 08-02 keeps `__HIPCC__` guards |
| R-04 | `gfx1100` vs `amdgpu-arch` — Windows HIP SDK may default to different arch | Medium | 08-03 pins `CMAKE_HIP_ARCHITECTURES=gfx1100` and `--offload-arch=gfx1100` everywhere, removes auto-detect |
| R-05 | `llama-server` vs `llama-cli` naming — some llama.cpp tags build `llama-server` as `llama-server.exe` vs `server.exe` | Medium | 08-04 checks both names, uses `LLAMA_BUILD_SERVER=ON`, verifies binary exists before smoke |
| R-06 | Model path handling on Windows (spaces, backslashes, 15GB file) | Medium | 08-04 uses quoted `%MODEL_PATH%` or `models\...` relative, tests with `dir` existence check, documents `E:\models` alternative |
| R-07 | VRAM 20GB ceiling — `llama-server --n-gpu-layers 99` may OOM if KV grows on Windows driver overhead | Medium | 08-04 smoke uses `-c 2048` small context first, then explains `-c 4096`, documents `HSA_ENABLE_DXG_DETECTION` not needed on Windows |
| R-08 | Driver 32.0.31041.1004 pinning — Windows Update may silently upgrade | Low | 08-CONTEXT documents freeze + `benchmarks/environment/versions.txt` moved to `README.md` version table |
| R-09 | Patch drift — `patches/0001...` generated against `bb4caa75` may not apply clean on fresh clone if line endings CRLF | Medium | 08-04 runs `git apply --check` with `--whitespace=fix`, normalizes `core.autocrlf=false`, verifies `ggml/src/ggml-cuda/custom_gfx1100/` exists post-patch |
| R-10 | Deleting `.planning/` loses audit trail | Low | Plans propose `git tag archive/pre-windows-08` before deletion, or move to `archive/` branch |
| R-11 | Cosine regression after pruning — deleting `ref_cpu`/`matmul_test_util` loses validation | Low | 08-02 keeps minimal `ref_cpu` + one test per kernel, 08-04 gates on `cosine >=0.999` |

## New Requirements (Phase 8 refactor, trace to objectives)

- **REQ-WIN-01** Eliminate bloat & unnecessary languages (OBJ-1): no Python/JS, no benchmark harnesses, no planning logs remain. (Maps to OBJ-1)
- **REQ-WIN-02** Retain core acceleration kernels only (OBJ-2): only IQ4_XS winners + minimal headers + clean patch remain in `kernels/`. (Maps to OBJ-2, extends KERN-04/05)
- **REQ-WIN-03** Simplify Windows build pipeline (OBJ-3): `build_windows.bat` compiles clean on Windows 11 + VS Build Tools + HIP SDK targeting gfx1100, intrinsics compile via hipcc not cl. (Maps to OBJ-3)
- **REQ-WIN-04** Output & usage (OBJ-4): clean minimal tree (`kernels/`, `patches/`, `CMakeLists.txt`, `build_windows.bat`), `llama-server.exe` serves OpenAI API at `localhost:8000`. (Maps to OBJ-4, extends INTEG-02)

## Thermos Review Amendments (2026-08-28 — unified verdict REQUEST CHANGES, quality review)

Applied before execution to eliminate plan-structure drift and over-pruning (see 07-VERIFICATION.md gaps):

1. **Single source of truth `allowlist.yaml`** — replaces 5 prose copies of keep/delete lists (CONTEXT table + 08-01 Task1 + 08-02 keep list + 08-04 `find` allowlist). Format:
   ```yaml
   keep: [README.md, LICENSE, CMakeLists.txt, build_windows.bat, kernels/common/block_iq4_xs.h, ...]
   delete: [benchmarks/, freetoken-rocm-probe/, .planning/phases/0[1-7]/, ...]
   vars: { gfx_arch: gfx1100, hip_path_default: "C:/Program Files/AMD/ROCm/6.2" }
   ```
   `scripts/prune_bloat.sh` becomes `yq '.delete[]' allowlist.yaml | xargs git rm -rf`; `CMakeLists.txt` + `build_windows.bat` read `vars`; verification is `diff <(find ... | sort) <(yq .keep[])` — no hand-maintained `grep -c`.

2. **Unify HIP toolchain to one file** — `cmake/toolchain-hip-gfx1100.cmake` (or root `CMakeLists.txt` only) owns `HIP_PATH` quoted, `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`, `--offload-arch=gfx1100`, `-G Ninja` mandate, `CMAKE_HIP_COMPILER`. `build_windows.bat` only passes `-DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-hip-gfx1100.cmake`. Deletes duplication between `08-02 Task2` (CMake) and `08-03 Task1` (bat).

3. **Keep minimal verification harness** — do NOT delete entire `benchmarks/lib/store.py` (117L) + `guard.py` (244L) with no replacement. Keep a 30-line `benchmarks/lib/store_minimal.py` (or `kernels/common/bench.h` CSV logger) as append-only `RunStore` + `CHECKSUMS` replacement. Gate `08-04` on `cosine >=0.999` + `curl 200` + this logger, not `find` + `curl` alone. Prevents harness-less regression (Phase 6/7 `849 vs 808` had ledger).

4. **Ephemeral planning artifacts + `.gitignore` hygiene** — `INVENTORY.md` and `scripts/prune_bloat.*` are **not** committed; they are generated under `build-*/` from `allowlist.yaml` (`yq`). `/.planning/` is gitignored after `archive/pre-windows-08` tag (or moved to `archive/` branch). Collapse `prune_bloat.{sh,ps1}` dual into one manifest-driven `prune_bloat.py` + thin wrappers. Removes 3 sources of truth for one `rm -rf`.

5. **Patch line endings + single helper** — add `.gitattributes: *.patch text eol=lf` once. Collapse `08-02 Task3` + `08-03 Task2` + `08-04 Task1` `core.autocrlf=false`/`--whitespace=fix`/`patch -p1` three-way branch into one `scripts/apply_patch.sh` (`git -C llama.cpp apply --check ../patches/0001...` with `core.autocrlf false`), called from all plans. Delete per-plan CRLF prose.

KERN-04/05 and INTEG-02 proven on WSL remain provenance; Phase 8 re-validates via cosine + server smoke on Windows.

## Traceability

| Requirement | Plans |
|-------------|-------|
| REQ-WIN-01 | 08-01 |
| REQ-WIN-02 | 08-02 |
| REQ-WIN-03 | 08-03 |
| REQ-WIN-04 | 08-04 |

No orphan requirements. CTX-01..05 remain deferred v2 (not in Phase 8 scope).

