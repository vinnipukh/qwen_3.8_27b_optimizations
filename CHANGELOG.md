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

## [v1.1.0-gfx1100] - 2026-08-31 - Phase 7 Hybrid DP4A & WMMA Optimization Complete

### Added
- **True Upstream DP4A Comparator (`real_stock_dp4a_comparator.hip`):**
  - Implements exact upstream `vec_dot_iq4_xs_q8_1` + `quantize_row_q8_1` integer pipeline via `__builtin_amdgcn_sudot4` + 6x `__builtin_amdgcn_perm` LUT.
  - Achieves **5.53–6.24× speedup** over naive float baseline on canonical Qwen3.8-27B projection shapes (87.8–99.3 µs vs 548 µs, $N=10$).
- **Cooperative Wave32 DP4A GEMV & XOR Variants (`impl_gemv_dp4a_gfx1100.hip`, `gemv_variant_xor.cuh`):**
  - Cooperative 8-thread/row superblock processing with Q8_1 on-the-fly quantization.
  - Dual compiled variants: `+33` padded LDS vs XOR preshuffle $x' = (y \pmod{4}) \oplus x$.
- **High-Yield Streaming WMMA GEMM Variants (`impl_gemm_wmma_stream.hip`, `impl_gemm_lut_iq4xs.hip`):**
  - 5 compiled variants: 64x32 P2+33 double-buffered, 64x32 P4 XOR quad-buffered, 64x64 B-stationary, LUT $\mu=4$ half-baked scale, and 128x32.
  - LLVM IR verified for `amdgcn.sudot4` (9 occurrences) and `amdgcn.wmma` (3 occurrences).
- **$N=15$ LLM QA Hardware Test Suite (`tools/run_llm_qa_n15.py`, `llm_qa_N15.json`, `llm_qa_stock_N15.json`):**
  - 15 consecutive greedy generation runs on RX 7900 XT (`gfx1100`) comparing Custom (`5c6b397`) vs Stock (`bb4caa7`).
  - Delivers **$+1.2\%$ generation speedup** ($36.38$ vs $35.95\text{ tok/s}$), **$+2.0\%$ prompt speedup** ($150.37$ vs $147.39\text{ tok/s}$), **$-821\text{ ms}$ lower request latency** ($19.05$ vs $19.87\text{ s}$), and **$45\%$ lower generation variance** ($\sigma = 0.61$ vs $1.12$).
- **Windows-Native Build Toolchain (`build_windows.bat`):**
  - Quoted `%HIP_PATH%` validation, `-G Ninja` enforcement with `%HIP_PATH%\bin\clang++.exe`, git `safe.directory` setup, model guards, and curl smoke harness.

### Changed
- **Quilt Patch (`patches/0001-gfx1100-mul-mat-custom.patch`):** 357 lines / 276 insertions over pinned `bb4caa75`, verified clean apply on both Windows and Linux/WSL2 with LF line endings. Real dispatch gate `custom_gemm_iq4xs_can_handle` restored.
- **Statistical Rigour:** All performance claims strictly enforced at $N \ge 10$ ($N \ge 15$ for LLM QA); single-run reporting banned across all documentation.

