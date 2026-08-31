# Phase 7 Plan 01 Summary: Windows-Native Toolchain & Quilt Patch Hygiene (REQ-WIN-07)

**Execution Date:** 2026-08-31  
**Status:** Completed  
**Requirements Addressed:** REQ-WIN-07, REQ-PERF-07, REQ-STAT-07

## Key Accomplishments

1. **Hardened `build_windows.bat`:**
   - Enforced quoting on all `%HIP_PATH%` references to handle paths with spaces (e.g. `C:\Program Files\AMD\ROCm\6.4`).
   - Added validation for `%HIP_PATH%\bin\clang++.exe` and `ninja` with clear installation guidance.
   - Enforced `-G Ninja` with `-DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe"` and `-DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang++.exe"`, ensuring MSVC `cl` is never invoked for `.hip` device intrinsics (`__builtin_amdgcn_sudot4`, `__builtin_amdgcn_perm`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`).
   - Added automated `git config --global --add safe.directory "%CD%"` to avoid dubious ownership issues across shared checkouts.
   - Added `MODEL_PATH` validation and `localhost:8000/v1/chat/completions` smoke test harness.

2. **CMake HIP Discovery:**
   - In `kernels/CMakeLists.txt`, updated `find_package` to `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip" "/opt/rocm/lib/cmake/hip")`, enabling Windows-native HIP SDK discovery first with `/opt/rocm` fallback for WSL2.

3. **Git Hygiene & Patch Regeneration:**
   - Verified `.gitattributes` enforces `*.patch eol=lf` and repository `core.autocrlf=false`.
   - Verified `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` has real `custom_gemm_iq4xs_can_handle` dispatch gate (not stub `return false`).
   - Regenerated `patches/0001-gfx1100-mul-mat-custom.patch` (357 lines, 8 files, 276 insertions) via `git -C llama.cpp diff bb4caa75` and verified `git apply --check` passes cleanly against pinned upstream `bb4caa75`.

## Verification Evidence

- `grep -F '"%HIP_PATH%\bin\clang++.exe"' build_windows.bat`: PASS
- `grep -q 'where ninja' build_windows.bat && grep -q '\-G Ninja' build_windows.bat`: PASS
- `grep -q 'find_package(hip.*HIP_PATH' kernels/CMakeLists.txt`: PASS
- `git check-attr eol -- patches/0001-gfx1100-mul-mat-custom.patch | grep lf`: PASS
- `git apply --check patches/0001-gfx1100-mul-mat-custom.patch`: PASS (clean apply on `bb4caa75`)
