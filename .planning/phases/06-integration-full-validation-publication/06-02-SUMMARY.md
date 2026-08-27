# Summary 06-02: Winner Patches Behind Switch (INTEG-01)

**Phase:** 6-Integration, Full Validation & Publication
**Plan:** 06-02
**Status:** COMPLETE
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Winner In-Tree Integration:**
   - Vendored custom `gemv_iq4xs.cuh` (decode M=1: 128-bit aligned loads, 8-thread cooperative per row dequant, shared-memory reduction, Wave32) and `gemm_iq4xs.cuh` (prefill M>=16: TILE_M=16 weight reuse, aligned vector loads, RDNA3 Wave32 WMMA hardware matrix cores via `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`).
   - Resolved GGML tensor convention layouts: `src1` `[K, M]` (coordinate $(k, m)$ at `m * K + k`) and `dst` `[N, M]` (coordinate $(n, m)$ at `m * N + n`).
   - Integrated dispatch hooks in `ggml/src/ggml-cuda/mmvq.cu` (`custom_gemv_iq4xs_can_handle` + `custom_gemv_iq4xs_dispatch`) and `ggml/src/ggml-cuda/mmq.cu` (`custom_gemm_iq4xs_can_handle` + `custom_gemm_iq4xs_dispatch`).
   - All `can_handle` and dispatch invocations are strictly guarded inside `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)` to ensure the OFF configuration compiles bit-identical to upstream.

2. **Patch Generation & Provenance (Thermos H-4):**
   - Generated `patches/0001-gfx1100-mul-mat-custom.patch` directly via `git diff` against pinned upstream commit `bb4caa7540188872173c44d161602d9271386413`.
   - Verified that `git apply --check` passes cleanly against a pristine clone of `bb4caa75`.
   - Removed the hand-written provisional design specification (`patches/phase5_mul_mat_custom.patch`).

3. **In-Tree Compilation & Verification:**
   - `build-custom` (`-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON`) compiled with zero errors across all targets (`test-backend-ops`, `llama-cli`, `llama-bench`, `llama-perplexity`).
   - `test-backend-ops test -o MUL_MAT -b ROCm0`: **1193/1193 PASS**.
   - `llama-bench` on Qwen3.8-27B IQ4_XS: **PASS** (`pp128` and `tg32`, full 99-layer GPU offload on RX 7900 XT).

## Artifacts

- `patches/0001-gfx1100-mul-mat-custom.patch`
- `tools/gemv_iq4xs.cuh`
- `tools/gemm_iq4xs.cuh`
