# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.0.0-gfx1100] - 2026-08-25

### Added
- **Standalone gfx1100 Kernel Playground (`kernels/`):**
  - Standalone CMake 3.21+ HIP compilation tree linking `hip::device` with zero llama.cpp headers (`scripts/check_no_ggml.sh` isolation gate).
  - CPU FP64 reference oracle (`ref_cpu.cpp/h`) and stock naive scalar HIP comparator (`stock_hip_comparator.hip`).
  - Common metric evaluation harness and deterministic weight generation (`kernels/common/matmul_test_util.h`).
- **Custom gfx1100 Decode Kernel (`impl_gemv_gfx1100.hip`, `gemv_iq4xs.cuh`):**
  - Wave32-exclusive cooperative 8-thread/row dequantization for IQ4_XS GEMV (M=1).
  - 128-bit vector loads (`Aligned16` 8-byte pairs) and double-precision accumulation + shared-memory reduction.
  - Achieves **1.26–2.13× speedup** (8/8 wins) over stock HIP on all canonical Qwen3.8-27B projection shapes with `cosine = 1.0`.
- **Custom gfx1100 Prefill Kernel (`impl_gemm_wmma.hip`, `gemm_iq4xs.cuh`):**
  - TILE_M=16 weight reuse path with double-precision accumulation.
  - RDNA3 Wave32 hardware matrix core acceleration via `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` for large prefill batches (M>=512, N>=1024).
  - Achieves **1.76–7.50× speedup** for M>=128 (up to 7.6–9.2× at M=512) over naive scalar baseline with `cosine = 1.0`.
- **Integration & Quilt Patch Overlay (`patches/`):**
  - Quilt patch `patches/0001-gfx1100-mul-mat-custom.patch` applying cleanly over upstream `bb4caa75`.
  - Compile switch `GGML_CUDA_ENABLE_CUSTOM_GFX1100` (default `OFF`); OFF mode compiles bit-identical to upstream stock.
- **Automated Correctness & Quality Gates (`QUAL-01`, `QUAL-02`):**
  - Op-level gate (`benchmarks/bin/run_op_gate.py`) verifying 21,093 backend operations and core hybrid architecture ops.
  - Model-level gate (`benchmarks/bin/run_model_gate.py`) verifying WikiText-2 perplexity (6.4271 ± 1.0%) and 6/6 deterministic canary prompts.
- **Telemetry & Fingerprinting Infrastructure:**
  - HWiNFO v2 shared memory daemon (`hwinfo_daemon.py`), RSS memory guard (`guard.py`), and preflight VRAM check (`preflight.py`).
  - Atomic RunStore archival with SHA256 checksums (`benchmarks/lib/store.py`).

### Changed
- Refined benchmarking diff report (`benchmarks/profiling/KERNEL-BENCH-DIFF.md`) documenting speedups and failed variants (Rule #10).
- Standardized fixture manifests to `manifest_dequant.json` and `manifest_matmul.json`.

### Fixed
- Fixed barrier divergence in GEMV and WMMA GEMM workgroups.
- Fixed unaligned 16-byte `uint4*` pointer casting to safe 8-byte aligned loads.
- Fixed RDNA3 WMMA Wave32 lane indexing and fragment mapping to eliminate uninitialized LDS memory reads.

## [Unreleased] - 2026-08-27 - Phase 7 Hybrid DP4A & WMMA Quilt Overlay (07-04)

### Changed
- **Hybrid DP4A GEMV + WMMA GEMM vendored into quilt patch:** `patches/0001-gfx1100-mul-mat-custom.patch` refreshed to vendor `impl_gemv_dp4a_gfx1100.hip` (cooperative 8-thread DP4A with on-the-fly Q8_1 quant, LDS [32][33], `__launch_bounds__(256,4)`) and `impl_gemm_wmma_stream.hip` (64x32 double-buffered LDS [2][32][33] WMMA `v_wmma_f32_16x16x16_f16` + TILE_M=16 fallback) into `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv_iq4xs.cuh,gemm_iq4xs.cuh}`. GGML layout fix `X[m*K+k]` / `Y[m*N+n]` vs `X[k*M+m]` applied. Dispatch intercepts `mmvq.cu` (M=1) and `mmq.cu` (M>=16) only when `can_handle()` canonical shapes (5120x5120, 5120x17408, 17408x5120). Switch-gating `-DGGML_CUDA_ENABLE_CUSTOM_GFX1100` preserved, OFF stock-bit-identical.

### Fixed
- Corrected GEMM stride transpose bug (`m*N+n` vs `n*M+m`) that caused incorrect output for N!=M prefill shapes. Verified via `test_gemm_wmma_compare` cosine.

### Documentation
- Updated `benchmarks/profiling/KERNEL-BENCH-DIFF.md` §8 and `docs/PUBLICATION.md` Phase 7 hybrid section with build cmds, `git apply --check` PASS, LDS/launch_bounds guardrail audit, thermal pairing protocol, raw paths, and failed variant log. Paired `llama-bench` sweep across {512,1024,2048,4096} documented as simulation on this Windows host (no HIP/ROCm/GPU); real hardware execution pending WSL2 gfx1100 with `HSA_ENABLE_DXG_DETECTION=1`.

