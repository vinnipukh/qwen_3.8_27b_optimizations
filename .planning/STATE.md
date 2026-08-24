---
gsd_state_version: 1.0
current_phase: 4
current_phase_name: Kernel Playground Scaffold
status: ready_for_planning
stopped_at: Phase 3 execution complete (all 4 plans executed and verified)
last_updated: "2026-08-24T21:30:00.000Z"
last_activity: 2026-08-24
last_activity_desc: Phase 3 complete. Two-tier quality gates armed, profiling table published, and Optimization Target #1 (MUL_MAT) designated.
progress:
  total_phases: 6
  completed_phases: 3
  total_plans: 12
  completed_plans: 12
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Beat stock llama.cpp HIP on at least one important Qwen3.8-27B workload on the RX 7900 XT with a custom gfx1100 kernel, within agreed numerical tolerance — measured, reproducible, bisectable.

**Current focus:** Phase 4 Kernel Playground Scaffold — ready for planning and implementation.

## Current Position

Phase: 4 (Kernel Playground Scaffold) — READY FOR PLANNING
Status: In Progress
Last activity: 2026-08-24 — Phase 3 execution complete

Progress: [█████░░░░░] 50% (3 of 6 phases)

## What Phase 3 proved & delivered

- **Op-Level Correctness Gate (`benchmarks/bin/run_op_gate.py`, QUAL-01):** Executed `test-backend-ops test -b ROCm0` across 21,093 cases (0 errors) on gfx1100 with explicit PASS assertions on `GATED_DELTA_NET`, `SOLVE_TRI`, `SSM_CONV`, `SSM_SCAN`, `FLASH_ATTN_EXT`, `MUL_MAT` (`benchmarks/results/phase3/op_gate.json`).
- **Model-Level Quality Gate & Golden Baseline (`benchmarks/bin/run_model_gate.py`, QUAL-02):** Pinned WikiText-2 PPL reference at **6.4271 +/- 0.04103** ($\pm 1\%$ interval $[6.3628, 6.4914]$) and 6/6 deterministic prompt canary store (`benchmarks/golden/stock_baseline_golden.json`, `benchmarks/results/phase3/model_gate.json`).
- **High-Precision Graph Evaluation Profiler (`benchmarks/bin/eval_profiler`, PROF-01):** Standalone C++ profiler instrumenting `ggml_backend_sched_set_eval_callback` with microsecond-resolution per-node timing across prefill and decode.
- **Dispatch Overhead & HIP Graphs Audit (`benchmarks/profiling/dispatch_overhead_report.md`):** Proved HIP graphs deliver **+19%** decode throughput speedup (70.5 t/s vs 59.3 t/s) by batching 520 kernel launches per step across WSL2 DXG.
- **Ranked Bottleneck Table & Optimization Target #1 (`benchmarks/profiling/BOTTLENECK-TABLE.md`, `bottleneck_summary.json`, PROF-02):** Profiled canonical shapes S1–S4. Formally designated `MUL_MAT` (quantized IQ4_XS GEMV/GEMM) as **Optimization Target #1** (31.12% cumulative GPU wall time; 50.89% prefill, 30.04% decode).
- **Test Suite Green:** 47 unit and integration tests passing (`benchmarks/tests/`).

## Next Step

Phase 4: Kernel Playground Scaffold (CPU reference → standalone gfx1100 HIP kernel → numerical comparison → microbenchmark sweep pipeline outside llama.cpp).
