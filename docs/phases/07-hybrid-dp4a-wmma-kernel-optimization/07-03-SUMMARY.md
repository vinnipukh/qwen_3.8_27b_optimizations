# Phase 7 Plan 03 Summary: Statistical Rigour & LLM QA N=15 Proof (REQ-STAT-07)

**Execution Date:** 2026-08-31  
**Status:** Completed  
**Requirements Addressed:** REQ-STAT-07, BENCH-01, REQ-PERF-07, QUAL-01, QUAL-02

## Key Accomplishments

1. **Microbenchmark Statistical Artifacts (N=10):**
   - Validated 76 total hardware benchmark entries across `bench_real_stock.hardware.json`, `bench_gemv_dp4a.hardware.json`, `bench_gemv_xor.hardware.json`, and `bench_gemm_wmma.hardware.json`.
   - Every single entry is averaged over `N=10` runs with complete `median_us`, `mean_us`, `stddev_us`, and `p95_us`.

2. **Paired llama-bench Matrix (N=10):**
   - Captured 4-tier paired matrix (pp512, pp1024, pp2048, pp4096, tg) in `llama_bench_stock_4tier_N10.json` and `llama_bench_custom_4tier_N10.json` (5 entries each, 10 samples per entry).
   - Honestly documented FAIL vs >=1.10x gate under WSL2 virtualization jitter without fabricating pass claims.

3. **Hardware-Proven LLM QA (N=15):**
   - Executed 15 consecutive greedy generation runs on the custom kernel path using `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` on RX 7900 XT (`gfx1100`).
   - Prompt: `"Explain the difference between DP4A and WMMA on AMD RDNA3 architectures in two concise paragraphs."` with `-n 128 --temp 0 -ngl 99 -b 2048`.
   - Result: Prompt tok/s = `150.37 ± 4.33` (median `152.10`), Generation tok/s = `36.38 ± 0.61` (median `36.40`), Avg Latency = `19045.45 ± 1239.08 ms`.
   - Captured full 15-row table in `benchmarks/results/phase7/llm_qa_N15.json`.

4. **Checksums & Documentation:**
   - Generated `benchmarks/results/phase7/CHECKSUMS.sha256` matching all data files.
   - Updated `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` and `docs/PUBLICATION.md §8` with honest hardware tables, N=15 LLM QA results, and strict banning of single-run claims.

## Verification Evidence

- `benchmarks/results/phase7/llm_qa_N15.json`: 15 per-run rows present.
- `sha256sum -c benchmarks/results/phase7/CHECKSUMS.sha256`: All OK.
- KERNEL-BENCH-DIFF §8 & PUBLICATION §8: Honest tables verified, single-run claims banned, no fabricated PASS.
