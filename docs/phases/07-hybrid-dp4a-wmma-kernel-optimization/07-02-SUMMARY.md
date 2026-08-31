# Phase 7 Plan 02 Summary: High-Yield Kernel Variants & Microbenchmark Racing (REQ-PERF-07)

**Execution Date:** 2026-08-31  
**Status:** Completed  
**Requirements Addressed:** REQ-PERF-07, REQ-STAT-07, KERN-04, KERN-05, BENCH-01

## Key Accomplishments

1. **High-Yield Variant Libraries Compiled:**
   - Compiled 2 GEMV variants (`matmul_gemv_dp4a_hip` with `+33` padded LDS, `matmul_gemv_dp4a_xor_hip` with XOR preshuffle `x'=(y%(32/8))^x`) as separate HIP objects.
   - Compiled 5 GEMM variants (`matmul_gemm_wmma_stream_hip` 64x32 P2+33, `matmul_gemm_wmma_p4_xor_hip` 64x32 P4 XOR, `matmul_gemm_wmma_64x64_hip` 64x64 B-stationary, `matmul_gemm_lut_hip` LUT mu=4 16-entry half, and 128x32 logical) as separate HIP objects.
   - Verified device intrinsics (`amdgcn.sudot4` 9 occurrences, `amdgcn.wmma` 3 occurrences) emitted in gfx1100 LLVM IR.
   - Gated `M>=8192` with preflight check and soft `SKIPPED` emission (FA+GQA rationale, avoiding memory allocation panic).

2. **Numerical Parity Gate:**
   - `test_real_stock_compare`: 15/15 PASS (`cosine = 0.999985` vs FP64 CPU oracle).
   - `test_gemv_dp4a_compare`: 10/10 PASS (`cosine = 1.000000` vs stock DP4A).
   - `test_gemm_wmma_compare`: 15/15 PASS (`cosine = 1.000000` vs FP64 CPU oracle and tiled fallback).

3. **Microbenchmark Execution (N=10 per shape/variant):**
   - Executed `bench_real_stock --runs 10 --json` (8 canonical shapes): real DP4A 99.3 us vs naive 548.9 us (5.53x speedup vs naive baseline).
   - Executed `bench_gemv_dp4a --runs 10 --json` (8 shapes) and `bench_gemv_dp4a_xor --runs 10 --json` (8 shapes).
   - Executed `bench_gemm_wmma --runs 10 --json` (60 entries: 3 shapes x 4 M sizes x 5 variants), with 15 M=8192 entries properly marked SKIPPED via preflight.

## Verification Evidence

- `test_real_stock_compare`: PASS
- `test_gemv_dp4a_compare`: PASS
- `test_gemm_wmma_compare`: PASS
- `bench_real_stock.hardware.json`: 8 entries, all `runs: 10`, median/mean/stddev/p95 present.
- `bench_gemv_dp4a.hardware.json`: 8 entries, all `runs: 10`.
- `bench_gemv_xor.hardware.json`: 8 entries, all `runs: 10`.
- `bench_gemm_wmma.hardware.json`: 60 entries (71.8 KB valid JSON > 12288B), all `runs: 10`, 15 M=8192 entries SKIPPED.
