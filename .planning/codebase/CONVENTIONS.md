<!-- refreshed: 2026-08-25 -->
# Coding & Methodology Conventions

**Analysis Date:** 2026-08-25 (Updated Phase 6 / v1.0.0-gfx1100)

## 1. Six Binding Methodology Rules

1. **Benchmark Before Optimize:** No kernel authoring proceeds without a fingerprinted stock baseline and quantitative bottleneck attribution.
2. **One Change at a Time:** Discrete, bisectable modifications.
3. **Keep the Stock Baseline Forever:** Stock binaries and builds are archived (`baseline/binaries/`) and never destroyed or overwritten.
4. **Enforce Prefill (pp) vs Decode (tg) Split:** Prefill ($M \gg 1$) and Decode ($M = 1$) metrics are measured and reported separately; blended token rates are prohibited.
5. **Numerical Correctness Gates First:** Candidate kernels must pass CPU reference oracle gates (`cosine \ge 0.999`, `max_rel \le 1e-3`) before benchmarking.
6. **Publish Failures Like Wins (Rule #10):** Slower shapes, regressions, and failed architectural variants are logged in `KERNEL-BENCH-DIFF.md` rather than pruned.

## 2. Kernel Playground Design Conventions

Every kernel in `kernels/` follows the **Op Quartet** pattern:
1. `ref_cpu.h/cpp`: Pure CPU double-precision reference oracle (zero external library dependencies).
2. `impl*.hip`: Standalone HIP kernel implementation targeting `gfx1100` Wave32 with zero `ggml` or `llama` headers.
3. `test_compare.cpp`: Numerical comparison binary asserting `cosine \ge 0.999` and `max_rel \le 1e-3` against the CPU oracle.
4. `bench_sweep.cpp`: Microbenchmark binary using `hipEvent_t` timers emitting structured JSON timing tables to stdout.

## 3. HIP Kernel Coding Standards

- **Wavefront Model:** Wave32 exclusive for `gfx1100`. Avoid hardcoded Wave64 assumption.
- **Resource Constraints:** Always annotate kernels with `__launch_bounds__(256, 4)` and `__attribute__((amdgpu_flat_work_group_size(256, 256)))` to enforce $\le 96$ VGPRs per thread.
- **Vector Loads:** Use 8-byte aligned `uint64_t[2]` or struct types when accessing quantized super-blocks; avoid unaligned 16-byte `uint4*` pointer casting.
- **Barrier Discipline:** Never place early `return` statements before `__syncthreads()`. Use conditional thread masking to maintain uniform barrier execution.
- **Error Checking:** Wrap all runtime HIP API calls in `HIP_CHECK(...)`.

## 4. Python Benchmark Harness Standards

- **Strict Typing:** All modules in `benchmarks/lib/` and `benchmarks/bin/` use `from __future__ import annotations` and explicit type annotations.
- **Standard Library Core:** Modules in `benchmarks/lib/` avoid heavy third-party dependencies to allow unit testing without GPU access.
- **Append-Only RunStore:** Every benchmark run produces a unique directory (`benchmarks/results/<timestamp>_<label>/`) with fsynced `rows.jsonl` and `CHECKSUMS.sha256`.
- **Verdict Vocabulary:** Standardized guard verdicts: `OK`, `FAILED:suspected-spill`, `FAILED:preflight-oom`, `FAILED:thermal-abort`, `REVIEW:repeat-deviation`.
