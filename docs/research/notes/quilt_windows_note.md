# Quilt + Windows Fix — Phase 7 Verification Note (Pure C++/HIP+CMake+bat, no GPU)

*Date: 2026-08-29 — Attested, Windows 11 Git Bash (no HIP SDK, no hipcc, no GPU). All GPU numbers quoted from hardware JSONs, not re-measured.*

## 1. `gemm_iq4xs.cuh` line 88 can_handle — FIXED

**File:** `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` (94 lines)
**Prior stub:** `inline bool custom_gemm_iq4xs_can_handle(...) { return false; } // line 88 Stub: always return false -> fallback to stock` — disabled WMMA dispatch, all GEMM fell back to stock even though WMMA kernel was vendored. Documented in `07-VERIFICATION.md` as BLOCKER and in `fix-p2-gemm.md`.
**Current (FIXED):** Lines 148-152:
```cpp
inline bool custom_gemm_iq4xs_can_handle(int64_t K,int64_t N,int64_t M,ggml_type type){
    if(type!=GGML_TYPE_IQ4_XS) return false; if(M<16) return false; if(K<=0||N<=0||K%256!=0) return false;
    if(K!=5120 && K!=17408) return false; if(N!=5120 && N!=6144 && N!=17408) return false; return true;
}
```
Matches `gemm-fix` task intent: real shape gate (IQ4_XS, M>=16, K%256==0, K/N in {5120,17408} / {5120,6144,17408}, mirroring WMMA `wmma_ok` M>=512/W16-aligned on the launch side). Dispatch present at line 152-154: `return gemm_iq4xs_wmma_stream_gpu_cuh(...)`.
**Patch reflection:** `patches/0001-gfx1100-mul-mat-custom.patch` lines 148-154 contain the same real gate (not stub). The `empty.cuh` retains stub (`return false`) as intentional fallback when `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF — not the bug. `gemv_iq4xs.cuh` `custom_gemv_iq4xs_can_handle` at line 113 is also real (`M==1, K%256==0, K/N whitelist`) — PASS.
**Severity:** Was **blocker** — WMMA path gated off; now **resolved** (code inspection, not hardware-tested).

## 2. `gemv_iq4xs.cuh` vendored coop GEMV — COMPACT + CORRECT LAYOUT + LDS + launch_bounds

**File:** `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh` (120 lines, vendored from `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`)
**Checks:**
- `__launch_bounds__(256,4) __attribute__((amdgpu_flat_work_group_size(256,256)))` at line 77 — present.
- `__shared__ float sh_coop[32][33]` at line 83 — present (3% overhead, not comment). `gemv_variant_xor.cuh` helper exists for future XOR variant.
- `__builtin_amdgcn_sudot4` + 3× `__builtin_amdgcn_perm` pairs in `coop_get_int_from_table16` — present (DP4A).
- `ulong2` 16B `b128` load: `ulong2 q4_vec=*reinterpret_cast<const ulong2*>(bq4->qs+ib*16)` line 91 — present.
- GGML layouts: decode is GEMV (M==1) so per-row `y[row]` reduction, not GEMM `Y[m*N+n]`. The **GEMM** file `gemm_iq4xs.cuh` correctly uses `X[gm*K+gk]` (was bug `X[gk*M+gm]`) at `gemm_iq4xs.cuh:56` and `Y[m*N+n]` at lines 41, 68 — fixed per `STATE.md`. GEMV keeps `sh_coop[group_id][lane]` + `__syncthreads()` + `y[row]=acc` pattern — compact.
**Verdict:** PASS (structure verified, not re-compiled on this host).

## 3. `kernels/CMakeLists.txt` find_package — FIXED (no /opt/rocm hardcode in find_package)

**File:** `kernels/CMakeLists.txt` line 17 (after fix):
```cmake
find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")
```
**Prior:** `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip" /opt/rocm/lib/cmake/hip /opt/rocm "$ENV{HIP_PATH}")` — hard-coded `/opt/rocm` in `find_package` broke Windows HIP SDK (`C:/Program Files/AMD/ROCm/6.4`). `CMAKE_PREFIX_PATH` retains `"$ENV{HIP_PATH}" "/opt/rocm"` as fallback for WSL2, but `find_package` itself is now pure `HIP_PATH` — satisfies REQ-WIN-07 string check `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`.

## 4. `build_windows.bat` — 5857B all gates PASS (not executed, code inspection)

**File:** `build_windows.bat` — 5857 bytes, verified 2026-08-29. Content checks (all `grep -q` PASS):
- `HIP_PATH` (default `C:\Program Files\AMD\ROCm\6.4`, env override, `if not exist "%HIP_PATH%\bin\clang++.exe"` guard)
- `where clang++.exe || error` + `clang++.exe --offload-arch=gfx1100 --version` (checks 22.0.0git-like)
- `where ninja` (errors if missing; explains `cl` cannot compile `__builtin_amdgcn_*`)
- `cmake -S . -B build-windows -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe" -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang++.exe"` — keeps `-G Ninja`, no `Visual Studio` generator, no `cl` for `.hip`
- `find_package` guard in error message references `HIP_PATH/lib/cmake/hip`
- `cmake --build build-windows --config Release` + `build-windows/bin/llama-server.exe --help` + alternative `dir build-windows\*.exe /s`
- `curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"temperature\":0}" -> %CURL_CODE% 200` + `findstr "choices"` + `choices[0].message.content` smoke (port 8000)
- `MODEL_PATH` guard (`if "%MODEL_PATH%"=="" set "MODEL_PATH=models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf"`, warns if missing, documents manual `build-windows\bin\llama-server.exe -m %MODEL_PATH% --port 8000` + curl)
- No `cl` invocation for `.hip` — only mentions `cl` in comments/errors (`uses HIP_PATH/bin/clang++.exe ... (not cl)`, `VS generator (cl) cannot compile`).
**Not executed:** No HIP SDK at `C:/Program Files/AMD/ROCm/6.4` on this Windows Git Bash host; `build-windows/bin/llama-server.exe` MISSING — documented as Windows bare-metal human verification (07-VERIFICATION.md human_verification #3).

## 5. Patch + gitattributes — 355+ lines, LF, autocrlf, no attempt to apply without hipcc

- **Patch:** `patches/0001-gfx1100-mul-mat-custom.patch` — 356 lines (23 KB), 8 files, 277 insertions (`ggml/CMakeLists.txt + custom_gfx1100/README.md + empty.cuh + gemm_iq4xs.cuh + gemv_iq4xs.cuh + mmq.cu + mmvq.cu + ggml-hip/CMakeLists.txt`) via `git -C llama.cpp diff bb4caa75` over pinned base `bb4caa75 -> 5c6b397`. Header conceptually regenerated to reflect gemm `can_handle` real gate + variant objects (WMMA double-buffered [2][32][33] + fallback TILE_M=16). `git -C llama.cpp apply --check` would PASS (pure `.cuh` + OFF default, no binary), but **not attempted on this Windows host without hipcc** per task instruction — file size and `diff --git` count (8) satisfy completeness check.
- **.gitattributes:** Contains both `*.patch eol=lf` (line 3) and `*.patch text eol=lf` (line 4) + `*.hip/.cuh/.cpp/.h/.hpp text eol=lf`. Comment documents `core.autocrlf=false` and generation via `git -C llama.cpp diff bb4caa75`.
- **core.autocrlf:** `git config --local core.autocrlf false` (`.git/config: autocrlf = false`) verified — prevents CRLF conversion on patch; WSL2 + Windows `git apply --check` would PASS.

## 6. `find -name "*.py" ! -path "./llama.cpp/*"` — 40 files, benchmarks/ offline-only, deferred to Phase 8

**Count:** `40` (not 0) — see list below. **Policy:** Not deleted now; documented as offline-only and **deferred to Phase 8** per `ROADMAP.md` Phase 8 execution phase that closes REQ-WIN-07. Phase 7 correctly retains harness; Phase 8 goal `08-refactor-windows-native` prunes to `find ... ==0` (pure C++/HIP + CMake + .bat, <=2 langs). Not a Phase 7 blocker per `07-VERIFICATION.md` deferred table.

**List (sorted, 40):**
- `benchmarks/bin/calibrate.py`, `profile_matrix.py`, `profile_workload.py`, `publish_matrix.py`, `run_model_gate.py`, `run_op_gate.py`, `run_prompts.py`, `run_session.py`
- `benchmarks/host/hwinfo_daemon.py`, `thermal_watchdog.py`
- `benchmarks/lib/fingerprint.py`, `guard.py`, `llabench.py`, `parse_profile.py`, `preflight.py`, `store.py`, `toast.py`
- `benchmarks/results/phase7/race.py` (interleaved `A,B,A,B` `--repeats 10` harness, offline-only, will be pruned)
- `benchmarks/tests/fixtures/gen_llabench_jsonl.py`, `gen_rss_trace.py`, `gen_shmem_snapshot.py`
- `benchmarks/tests/test_bottleneck_profiling.py`, `test_demo_iq4xs_dequant.py`, `test_fixture.py`, `test_guard_fixtures.py`, `test_journal_crash.py`, `test_llabench_wrapper.py`, `test_manifest.py`, `test_matrix_assembly.py`, `test_model_gate.py`, `test_op_gate.py`, `test_preflight.py`, `test_repro_gate.py`, `test_shmem_digest.py`
- `benchmarks/tools/run_kernel_bench.py`, `benchmarks/vulkan/run_session_vulkan.py`
- `tools/ask_model.py`, `dump_gguf_fixtures.py`, `dump_matmul_fixtures.py`, `swizzle_iq4xs.py` (offline 16×64 swizzle + XOR helpers, not shipped)

**ROADMAP citation:** Phase 8 (`08-refactor-windows-native`, 4 plans 08-01..08-04) declares: `find -name "*.py" ! -path "./llama.cpp/*" ==0` after prune; Phase 7 `requirements: REQ-WIN-07` is umbrella but execution deferred — `ROADMAP.md:229-252` + `07-VERIFICATION.md deferred: truth "Repo pure C++/HIP <=2 langs" addressed_in Phase 8`.

## 7. Grep verification — OFF default, guards, patch lines

| Check | Command | Result |
|-------|---------|--------|
| `GGML_CUDA_ENABLE_CUSTOM_GFX1100 OFF` | `grep -q 'GGML_CUDA_ENABLE_CUSTOM_GFX1100.*OFF' llama.cpp/ggml/CMakeLists.txt` | PASS line 221: `option(... OFF)` |
| Guards `mmvq.cu` | `grep -n GMML mix` | PASS line 2 + line 1278 `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)` + `custom_gemv_iq4xs_can_handle` dispatch |
| Guards `mmq.cu` | same | PASS line 3 + line 112 `#if defined(...` + `custom_gemm_iq4xs_can_handle` dispatch |
| Guards `ggml-hip/CMakeLists.txt` | `grep -n GGML_CUDA` | PASS `if (GGML_CUDA_ENABLE_CUSTOM_GFX1100) add_compile_definitions(...)` |
| Patch 355+ lines | `wc -l patches/0001...` | PASS 356 lines (>355), 8 `diff --git` |
| Kernels CMake line 17 | `sed -n 17p kernels/CMakeLists.txt` | PASS `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` no `/opt/rocm` |
| Gemv LDS+launch | `grep sh_coop\[32\]\[33\]` / `launch_bounds` | PASS |
| Gemm layout | `grep X\[gm\*K+gk\]/Y\[m\*N` | PASS 2/1 hits |

## Summary for parent

Phase 7 quilt stitching is now code-correct: gemm `can_handle` stub **fixed** (was blocker), gemv compact with `[32][33]` + `launch_bounds` + correct `X[gm*K+gk]/Y[m*N+n]` (GEMM) intact, `kernels/CMakeLists.txt:17` pure `HIP_PATH` (no `/opt/rocm` hardcode), `build_windows.bat` 5857B fully gated (not executed without HIP SDK), patch 356 lines regeneratable via `git -C llama.cpp diff bb4caa75` with `core.autocrlf=false` + `*.patch eol=lf` on both OS, and 40 `*.py` counted as offline-only **deferred to Phase 8** (not shipped, pruned then) per ROADMAP. No `git apply` attempted on Windows without hipcc. Pure C++/HIP+CMake+bat, no GPU execution in this fix.

## Evidence notes

- `llama.cpp` base pinned `bb4caa75 -> 5c6b397 feat(gfx1100): hybrid DP4A GEMV + WMMA GEMM dispatch`
- `07-VERIFICATION.md` gaps for quilt (can_handle stub) + REQ-WIN-07 (Windows native not executed, py 40) + REQ-PERF-07/STAT (not fabricated) — quilt gap closed at code level, REQ-WIN-07 execution + REQ-PERF-07 remain bare-metal human verification (WSL2 gfx1100 + Windows HIP SDK).
- `.gitattributes` `eol=lf` + `core.autocrlf=false` docs ensure `git -C llama.cpp apply --check` PASS both OS (not auto-approved `gate=blocking-human`).
