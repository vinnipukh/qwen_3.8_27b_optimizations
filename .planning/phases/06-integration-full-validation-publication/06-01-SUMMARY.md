# Summary 06-01: Empty-Flag Plumbing Proof (INTEG-01)

**Phase:** 6-Integration, Full Validation & Publication
**Plan:** 06-01
**Status:** COMPLETE
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Plumbing Switch Wire-up:**
   - Added `option(GGML_CUDA_ENABLE_CUSTOM_GFX1100 "ggml: enable custom gfx1100 IQ4_XS kernels" OFF)` to `ggml/CMakeLists.txt`.
   - Wired compile definitions and header inclusion in `ggml/src/ggml-hip/CMakeLists.txt` behind the `GGML_CUDA_ENABLE_CUSTOM_GFX1100` condition.
   - Implemented `ggml/src/ggml-cuda/custom_gfx1100/empty.cuh` stub containing no-op `can_handle` predicates (returning `false`) and `dispatch` stubs (returning `hipErrorNotSupported`).
   - Guarded dispatch hook intercepts in `ggml/src/ggml-cuda/mmvq.cu` and `ggml/src/ggml-cuda/mmq.cu` strictly within `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)` so that the OFF configuration compiles bit-identical to stock upstream without missing symbol errors (thermos H-4).

2. **Dual-Tree Compilation & Verification:**
   - **`build-stock` (`-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF`):**
     - Full build targeting gfx1100 completed with zero errors.
     - `test-backend-ops test -o MUL_MAT -b ROCm0`: **1193/1193 PASS**.
   - **`build-custom-empty` (`-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON`):**
     - Full build targeting gfx1100 completed with zero errors.
     - `test-backend-ops test -o MUL_MAT -b ROCm0`: **1193/1193 PASS**.

## Verification Criteria

- [x] `GGML_CUDA_ENABLE_CUSTOM_GFX1100` CMake option present and defaults to `OFF`.
- [x] `build-stock` (OFF) compiles clean on gfx1100 and passes `test-backend-ops` (1193/1193).
- [x] `build-custom-empty` (ON + stub) compiles clean on gfx1100 and passes `test-backend-ops` (1193/1193).
- [x] Stock execution path is 100% unregressed.
