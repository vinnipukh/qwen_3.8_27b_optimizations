---
gsd_state_version: 1.0
current_phase: 7
current_phase_name: Hybrid DP4A & WMMA Matrix Core Optimization
status: planned
stopped_at: Phase 7 planned — 4 plans drafted (07-01 real stock DP4A comparator, 07-02 cooperative Wave32 DP4A GEMV, 07-03 streaming WMMA GEMM, 07-04 e2e A/B verification).
last_updated: "2026-08-25T23:59:00.000Z"
last_activity: 2026-08-25
last_activity_desc: Phase 7 planned to fuse Q8_1 integer activation quantization and RDNA3 hardware matrix cores to beat real production stock llama.cpp end-to-end.
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 28
  completed_plans: 24
  percent: 86
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Beat stock llama.cpp HIP on at least one important Qwen3.8-27B workload on the RX 7900 XT with a custom gfx1100 kernel, within agreed numerical tolerance — measured, reproducible, bisectable.

**Current focus:** Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization (beating real production stock end-to-end).

## Current Position

Phase: 7 (Hybrid DP4A & WMMA Matrix Core Optimization) — PLANNED
Status: Planned — 4 plans defined in `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/`

Progress: [████████░░] 86% (6 of 7 phases)

## What Phase 6 Delivered

- **Thermos Remediation & Kernel Guard Hardening (06-05):**
  - Eliminated barrier divergence in GEMV and WMMA GEMM workgroups.
  - Replaced unaligned 16-byte `uint4*` pointer casting with 8-byte aligned `uint64_t` pairs.
  - Implemented correct RDNA3 Wave32 WMMA 16x16x16 lane and fragment layout (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`), eliminating uninitialized LDS reads.
  - Added WMMA gate-passing test case (`wmma_gate_pass_5120_1024_512`) -> PASS `cosine=1.000000`.
  - Extracted common test/bench harness utilities (`kernels/common/matmul_test_util.h`).
  - Standardized fixture manifests (`manifest_dequant.json` and `manifest_matmul.json`) with deterministic SHA256 seeds.
- **Empty-Flag Switch Plumbing Proof (06-01, INTEG-01):**
  - Added `GGML_CUDA_ENABLE_CUSTOM_GFX1100` CMake option (default `OFF`).
  - Proved both `build-stock` (OFF) and `build-custom-empty` (ON + stub) compile cleanly targeting gfx1100 and pass op-gate (1193/1193 PASS).
- **Winner In-Tree Integration & Patch Provenance (06-02, INTEG-01):**
  - Vendored custom `custom_gfx1100/{gemv_iq4xs.cuh, gemm_iq4xs.cuh}` with exact GGML tensor convention layouts (`[K, M]` and `[N, M]`).
  - Generated quilt patch `patches/0001-gfx1100-mul-mat-custom.patch` via `git diff` against pinned upstream `bb4caa75`, verified with `git apply --check`.
  - Guarded all dispatch code inside `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)`.
- **Baseline-Preservation Guard (06-03, INTEG-01, Rule #3):**
  - Verified `build-stock` rebuilds cleanly from same tree with 0 errors across 4,243 supported ops (`benchmarks/results/phase6/op_gate_stock_20260827.json`).
  - Verified `baseline/binaries/v0.2.0-bb4caa75/` binaries are preserved with intact sha256 checksums.
- **Publication Package & Release Hygiene (06-04, PUB-01):**
  - `docs/PUBLICATION.md` published covering exact build commands, hardware and environment versions, model provenance, methodology, raw data, kernel sources, and known limitations.
  - `CHANGELOG.md` created with full release history.
  - `NOTICE` attribution and `LICENSE` (Apache 2.0) delivered.
  - Tagged `v1.0.0-gfx1100`.

## Next Step

Execute Phase 7:
1. `07-01`: Build direct microbenchmark comparator against real upstream DP4A `vec_dot_iq4_xs_q8_1`.
2. `07-02`: Author cooperative Wave32 DP4A GEMV kernel (`impl_gemv_dp4a_gfx1100.hip`) for decode.
3. `07-03`: Author streaming WMMA matrix core GEMM kernel (`impl_gemm_wmma_stream.hip`) for prefill.
4. `07-04`: Update quilt patch and execute paired end-to-end benchmark in `llama-bench`.
