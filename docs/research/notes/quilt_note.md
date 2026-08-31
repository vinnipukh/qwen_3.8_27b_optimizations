# Phase 7 Quilt+Windows — Verification Note

*Date: 2026-08-30 — Windows 11 Git Bash, no HIP SDK, no GPU. Attested via grep, no hardware re-run.*

## 1. gemm_iq4xs.cuh can_handle stub — FIXED (was `return false`)

**File:** `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh:87-90`
```cpp
inline bool custom_gemm_iq4xs_can_handle(int64_t K,int64_t N,int64_t M,ggml_type type){
    if(type!=GGML_TYPE_IQ4_XS) return false; if(M<16) return false; if(K<=0||N<=0||K%256!=0) return false;
    if(K!=5120 && K!=17408) return false; if(N!=5120 && N!=6144 && N!=17408) return false; return true;
}
```
Prior stub `return false` (line 88 blocker in 07-VERIFICATION.md) disabled WMMA dispatch even when WMMA kernel was vendored. Now real shape gate matching `wmma_ok` (M>=16 generic, K%256==0, K/N whitelist for Qwen 5120/17408) and `gemm_iq4xs_wmma_stream_gpu_cuh` dispatch at line 91-93. `empty.cuh` retains stub intentionally for OFF fallback (not a bug). `gemv_iq4xs.cuh:113` already real (`M==1` gate) — unchanged.

**Patch:** `patches/0001-gfx1100-mul-mat-custom.patch` regenerated via `git -C llama.cpp diff bb4caa75` now 356 lines, 8 files, 277 insertions, contains real gate. Verified: no `return false` in gemm can_handle.

## 2. gemv_iq4xs.cuh layout — VERIFIED

**File:** `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh` (120 lines, vendored from `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`)
- `__launch_bounds__(256,4) __attribute__((amdgpu_flat_work_group_size(256,256)))` at line 77 — present.
- `__shared__ float sh_coop[32][33]` at line 83 — present (+33 padding, 3% overhead).
- DP4A: `__builtin_amdgcn_sudot4` + 3x `__builtin_amdgcn_perm` pairs in `coop_get_int_from_table16` — present.
- `ulong2` b128 16B load: `ulong2 q4_vec=*reinterpret_cast<const ulong2*>(bq4->qs+ib*16)` line 91 — present.
- GEMV is M==1 decode (`y[row]` reduction, not GEMM `X[gm*K+gk]`). GEMM file `gemm_iq4xs.cuh` correctly uses `X[gm*K+gk]` at line 56 (`X[gm*K+gk]` vs prior bug `X[gk*M+gm]`) and `Y[m*N+n]` at lines 41/68 per STATE.md fix — preserved.

**Patch also includes gemv fix:** `hipStreamCaptureStatus` guard + `hipStreamSynchronize` after `gemv_prequantized_cuh` for graph-capture safety.

## 3. kernels/CMakeLists.txt HIP_PATH find_package — VERIFIED

**File:** `kernels/CMakeLists.txt:16-17`
```cmake
list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}" "/opt/rocm")
find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")
```
String check `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` PASS with no `/opt/rocm` hardcode in `find_package` (Windows-native REQ-WIN-07). `CMAKE_PREFIX_PATH` retains `/opt/rocm` fallback for WSL2 but does not affect Windows search.

## 4. build_windows.bat — VERIFIED (all grep gates PASS, not executed without HIP SDK)

**File:** `build_windows.bat` (5857B)
- `HIP_PATH` default `C:\Program Files\AMD\ROCm\6.4`, env override, `if not exist "%HIP_PATH%\bin\clang++.exe"` guard
- `where clang++.exe && clang++.exe --offload-arch=gfx1100 --version` (22.0.0git-like)
- `where ninja` with error explaining `cl` cannot compile `__builtin_amdgcn_*`
- `cmake -S . -B build-windows -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe" -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang++.exe"`
- `curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -> 200` + `findstr "choices"` (port 8000) with `MODEL_PATH` guard
- No `cl` for `.hip` — only in comments/errors
- Not executed on this host (no HIP SDK) — documented as human verification on Windows bare-metal per 07-VERIFICATION.md.

## 5. .gitattributes eol=lf — VERIFIED

**File:** `.gitattributes`
```
*.patch eol=lf
*.patch text eol=lf
*.hip text eol=lf
*.cuh text eol=lf
```
+ comment documenting `core.autocrlf=false` + `git -C llama.cpp diff bb4caa75` provenance. `git config --local core.autocrlf` is `false`.

## 6. Python file count — 40 deferred to Phase 8

**Command:** `find . -name "*.py" ! -path "./llama.cpp/*" | wc -l` -> **40** (not 0 until Phase 8 prune).

List (sorted):
- benchmarks/bin/calibrate.py, profile_matrix.py, profile_workload.py, publish_matrix.py, run_model_gate.py, run_op_gate.py, run_prompts.py, run_session.py
- benchmarks/host/hwinfo_daemon.py, thermal_watchdog.py
- benchmarks/lib/fingerprint.py, guard.py, llabench.py, parse_profile.py, preflight.py, store.py, toast.py
- benchmarks/results/phase7/race.py (interleaved A,B,A,B --repeats 10, offline-only, will be pruned)
- benchmarks/tests/fixtures/gen_llabench_jsonl.py, gen_rss_trace.py, gen_shmem_snapshot.py
- benchmarks/tests/test_bottleneck_profiling.py, test_demo_iq4xs_dequant.py, test_fixture.py, test_guard_fixtures.py, test_journal_crash.py, test_llabench_wrapper.py, test_manifest.py, test_matrix_assembly.py, test_model_gate.py, test_op_gate.py, test_preflight.py, test_repro_gate.py, test_shmem_digest.py
- benchmarks/tools/run_kernel_bench.py, benchmarks/vulkan/run_session_vulkan.py
- tools/ask_model.py, dump_gguf_fixtures.py, dump_matmul_fixtures.py, swizzle_iq4xs.py

**Policy:** Not deleted in Phase 7 — Phase 7 retains harness offline-only. **Deferred to Phase 8** (`08-refactor-windows-native`) which prunes to `find ... ==0` (pure C++/HIP + CMake + .bat, <=2 langs). Per ROADMAP Phase 8 execution phase and 07-VERIFICATION.md deferred table.

## Grep verification (copy-paste)

```bash
grep -q "GGML_CUDA_ENABLE_CUSTOM_GFX1100" llama.cpp/ggml/CMakeLists.txt
grep -q "sh_coop.*33" llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh
grep -q "X\[gm\*K+gk\]" llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh
grep -q "Y\[m\*N" llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh
grep -q 'find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")' kernels/CMakeLists.txt
grep -q "HIP_PATH" build_windows.bat && grep -q "clang++.exe.*--offload-arch=gfx1100" build_windows.bat && grep -q "\-G Ninja" build_windows.bat && grep -q "curl.*8000.*chat/completions" build_windows.bat
grep -q "^\*.patch eol=lf" .gitattributes
wc -l patches/0001-gfx1100-mul-mat-custom.patch  # expect 356
find . -name "*.py" ! -path "./llama.cpp/*" | wc -l  # expect 40 (Phase8 prunes to 0)
```
