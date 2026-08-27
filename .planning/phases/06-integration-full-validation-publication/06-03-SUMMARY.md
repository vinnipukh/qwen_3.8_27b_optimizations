# Summary 06-03: Baseline-Preservation Guard (INTEG-01, Rule #3)

**Phase:** 6-Integration, Full Validation & Publication
**Plan:** 06-03
**Status:** COMPLETE
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Stock Baseline Rebuild & Validation:**
   - Configured and compiled `build-stock` (`-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF`) targeting gfx1100.
   - Executed `run_op_gate.py` on the stock build with 0 errors across 4,243 supported operations (including 1,193 `MUL_MAT` tests and all core hybrid architecture ops).
   - Saved fresh timestamped op-gate artifact to `benchmarks/results/phase6/op_gate_stock_20260827.json` (preserving historical gate files per thermos H-3).

2. **Custom Build Parity Validation:**
   - Executed `run_op_gate.py` on `build-custom` (`-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON`) with identical 0 errors across 4,243 supported cases.
   - Saved fresh artifact to `benchmarks/results/phase6/op_gate_custom_20260827.json`.

3. **Baseline Integrity & Environment Version Verification:**
   - Verified that `baseline/binaries/v0.2.0-bb4caa75/` binaries are preserved with intact sha256 checksums (`llama-bench`, `llama-cli`, `llama-perplexity`, `test-backend-ops`).
   - Verified frozen WSL environment snapshot reference (`E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar`, ROCm 7.2.1, librocdxg 1.2.2, Adrenalin 26.2.2).

## Verification Criteria

- [x] Stock build (`build-stock`) compiles cleanly from the same tree when `GGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF`.
- [x] `run_op_gate.py` passes with 0 errors on `build-stock` and output is written to a new timestamped file.
- [x] `run_op_gate.py` passes with 0 errors on `build-custom`.
- [x] Pinned baseline binaries under `baseline/binaries/` remain intact and unchanged.
