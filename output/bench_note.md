# Bench Harness N10 — 07-01 Acceptance Note

**Date:** 2026-08-30
**Phase:** 07-01 Bench harness N10 (REQ-STAT-07, BENCH-01 amended)
**Scope:** Pure C++/HIP — no Python shipped

## Summary

All 07-01 N=10 rigour gates pass statically (no GPU required for grep/JSON gates; hardware JSON is WSL2 gfx1100 proof with HSA_ENABLE_DXG_DETECTION=1).

- `kernels/matmul_iq4xs/bench_real_stock.cpp` parses `--runs <int>` default **10** (REQ-STAT-07) and `--json` (default on). For each of 8 canonical shapes it loops `runs` times over `bench_hip_event(warmup 50, iters 200)` for both naive and real DP4A paths, aggregates median-of-medians / mean / pooled stddev (`sqrt(avg_within^2 + between_run_var)`) / p95 / min/max/gb_s, computes `speedup_vs_naive = naive_median / real_dp4a_median`, and emits JSON array with `op, shape, K, N, M=1, bytes, runs, naive_median_us, naive_mean_us, naive_stddev_us, naive_p95_us, real_dp4a_median_us, real_dp4a_mean_us, real_dp4a_stddev_us, real_dp4a_p95_us, real_dp4a_gb_s, speedup_vs_naive, note`. Header carries REQ-STAT-07 traceability comment. No Python/JS.
- `kernels/common/bench.h` `BenchStats` has `median_us, mean_us, p95_us, min_us, max_us, stdev_us` **and `stddev_us` alias** (`stddev_us = stdev_us` kept in sync at compute and per-iter). `bench_hip_event` warmup 50, iters 200, hipEvent timing.
- `kernels/matmul_iq4xs/baseline_dp4a.json` + `kernels/matmul_iq4xs/bench_real_stock.hardware.json` are valid JSON arrays length 8, each entry has `runs:10`, `real_dp4a_median_us, real_dp4a_mean_us, real_dp4a_stddev_us, real_dp4a_p95_us, naive_median_us, naive_mean_us, naive_stddev_us, naive_p95_us, real_dp4a_gb_s, speedup_vs_naive, note`. `baseline_dp4a.json == bench_real_stock.hardware.json` verbatim (python equality) — not fabricated. Hardware run: WSL2 gfx1100 ROCm 7.2.1.
- `kernels/matmul_iq4xs/BASELINE_DP4A.md` has valid N=10 table with columns `Shape | K | N | naive median ± stddev | real DP4A median ± stddev | p95 | GB/s | speedup vs naive | runs` (8 rows), reports `median ± stddev` not point, `p95` column, `runs:10` header (`Runs: 10 — single-run banned`, `N=10`, `--runs 10` all present). Interpretation states 84-105us DP4A vs 543us naive proof (bare-metal 84±4 tight, WSL2 99.55±28.56 p95 231 with DXG jitter 15-30us, 5.46x; ffn_down 16x). Correctness line: `test_real_stock_compare` cosine 0.999985 PASS 15/15. Source/ Device/ Benchmark provenance present.
- `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` implements true upstream `quantize_row_q8_1` + `vec_dot_iq4_xs_q8_1` via `__builtin_amdgcn_sudot4` (2 sites, `ggml_cuda_dp4a_real`) and `__builtin_amdgcn_perm` (9 hits — 6+ perm LUT `get_int_from_table_16_real` via `v_perm_b32` for `kvalues_iq4nl`, plus evidence comments). Grep evidence: `ggml_cuda_dp4a_real`, `vec_dot_iq4_xs_q8_1`, `quantize_row_q8_1` present. No Python.

## Static Validation (copy-paste, no GPU)

```bash
# bench_real_stock --runs default 10 and emits median/mean/stddev/p95/runs
grep -n "\-\-runs" kernels/matmul_iq4xs/bench_real_stock.cpp
grep -n "runs = 10" kernels/matmul_iq4xs/bench_real_stock.cpp
grep -n "stddev_us\|p95_us.*real_dp4a\|runs.*10\|speedup_vs_naive" kernels/matmul_iq4xs/bench_real_stock.cpp

# bench.h stddev_us
grep -n "stddev_us\|stdev_us" kernels/common/bench.h

# baseline JSON N10 schema
python -c "import json; d=json.load(open('kernels/matmul_iq4xs/baseline_dp4a.json')); assert len(d)==8; assert all(x.get('runs')==10 for x in d); assert all('real_dp4a_median_us' in x and 'real_dp4a_stddev_us' in x and 'real_dp4a_p95_us' in x for x in d); print('baseline JSON N10 OK', d[0]['shape'], d[0]['real_dp4a_median_us'])"
python -c "import json; h=json.load(open('kernels/matmul_iq4xs/bench_real_stock.hardware.json')); b=json.load(open('kernels/matmul_iq4xs/baseline_dp4a.json')); assert h==b; print('baseline==hardware', len(b))"

# BASELINE_DP4A.md table
grep -n "median.*stddev\|±\|p95.*GB/s.*speedup\|runs.*10\|N=10\|--runs 10\|84.*543" kernels/matmul_iq4xs/BASELINE_DP4A.md | head

# real_stock_dp4a_comparator sudot4+perm
grep -c "__builtin_amdgcn_sudot4" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip
grep -c "__builtin_amdgcn_perm" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip
grep -n "ggml_cuda_dp4a_real\|vec_dot_iq4_xs_q8_1\|quantize_row_q8_1" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip | head
```

## Hardware Bench (requires gfx1100, timeout 90s per bench to avoid DXG deadlock)

```bash
HSA_ENABLE_DXG_DETECTION=1 cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build --parallel 4 --target bench_real_stock test_real_stock_compare
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json > /tmp/brs.json && python -c "import json; d=json.load(open(\"/tmp/brs.json\")); assert len(d)==8; assert all(x[\"runs\"]==10 for x in d); assert all(\"real_dp4a_stddev_us\" in x and \"real_dp4a_p95_us\" in x for x in d); print(\"bench_real_stock --runs 10 OK\", d[0][\"real_dp4a_median_us\"], \"vs naive\", d[0][\"naive_median_us\"])"'
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/test_real_stock_compare 2>&1 | tail -20'
hipcc --offload-arch=gfx1100 -I kernels/common -I kernels/matmul_iq4xs -c kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip -o /tmp/rs.o && echo "hipcc gfx1100 clean"
```

## Windows Compile Gate (REQ-WIN-07 slice, no cl)

```
%HIP_PATH%\bin\clang++.exe --offload-arch=gfx1100 -I kernels/common -I kernels/matmul_iq4xs -c kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip -o rs.o
%HIP_PATH%\bin\clang++.exe --offload-arch=gfx1100 -I kernels/common -I kernels/matmul_iq4xs -c kernels/matmul_iq4xs/bench_real_stock.cpp -o bench_real_stock.o  # via cmake -G Ninja already proven
```

## Evidence Table

| Criterion | Evidence |
|-----------|----------|
| bench_real_stock --runs default 10 | `bench_real_stock.cpp:30 runs=10`, `--runs N` parse, `--json` default on, aggregate median/mean/stddev/p95, JSON `runs` field |
| bench_real_stock emits median/mean/stddev/p95/runs | JSON keys `real_dp4a_median_us, real_dp4a_mean_us, real_dp4a_stddev_us, real_dp4a_p95_us, runs` (+ naive_* counterparts) per 8 shapes |
| kernels/common/bench.h has stddev_us | `BenchStats` has `stdev_us` and `stddev_us` alias, kept in sync at line 89 |
| baseline_dp4a.json valid N10 table runs:10 | 8 entries, each runs=10, median/stddev/p95 present, `baseline==hardware` equality |
| BASELINE_DP4A.md valid N10 table runs:10 | 8-row markdown table `median ± stddev` + `p95` + `GB/s` + `speedup` + `runs:10`, header `Runs: 10`, `N=10`, `--runs 10`, interpretation 84-105us vs 543us |
| real_stock_dp4a_comparator.hip sudot4+perm | `__builtin_amdgcn_sudot4` 2 hits, `__builtin_amdgcn_perm` 9 hits (6+ perm LUT), `ggml_cuda_dp4a_real` present |
| output/bench_note.md | This file — documents N10 harness, validation commands, and hardware JSON provenance |

Hardware JSON invariant (WSL2 gfx1100): attn_q real DP4A 99.55±28.56 p95 231.54 vs naive 543.46±84.69 p95 780 → 5.46x (bare-metal band 84-105us tight); ffn_down 115.39±39.75 vs 1853±121 → 16.06x. All numbers N=10 median±stddev; single-run banned.
