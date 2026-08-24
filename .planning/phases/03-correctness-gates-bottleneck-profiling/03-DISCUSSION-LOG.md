# Phase 3: Correctness Gates & Bottleneck Profiling - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-24
**Phase:** 3-Correctness Gates & Bottleneck Profiling
**Areas discussed:** Op-Level Correctness Gate; Model-Level Quality Gate & Determinism; WSL2 Profiling Strategy & Upstream Practices; Workload Shapes & Bottleneck Ranking Table; Execution Protocol & Safety Integration.

---

## Op-Level Correctness Gate (QUAL-01)

| Option | Description | Selected |
|--------|-------------|----------|
| A) Full Suite (All 128 ops) | Run all 128 operations in `test-backend-ops test -b ROCm0` with strict checks on hybrid ops | ✓ |
| B) Target Ops Only | Only run the ~10 operations relevant to Qwen hybrid architecture | |

**User's choice:** Option A (Full Suite).
**Notes:** Full suite takes ~15 seconds, guarantees complete backend sanity without missing edge cases. Zero-tolerance failure policy adopted.

---

## Model-Level Quality Gate & Determinism (QUAL-02)

| Topic | Key Question / Exploration | Selected Resolution |
|-------|----------------------------|---------------------|
| Determinism in LLMs | User raised Horace He's research (*"Defeating Nondeterminism in LLM Inference"*) on whether LLMs are truly deterministic | Explored floating-point non-associativity across different kernels vs exact same binary. |
| Test Robustness | How to prevent false-negative test failures from autoregressive token branching? | Adopted 3-Layer Hierarchy: (1) WikiText-2 Perplexity (ctx=2048, ±1% tolerance); (2) Short Early-Token Canary (16–32 tokens on 6 corpus prompts); (3) Logit Cosine Similarity (>0.999). |
| Context Length | Perplexity context length | 2048 (Community standard on WikiText-2). |

**User's choice:** Approved the 3-layer approach.
**Notes:** Research confirmed that perplexity is teacher-forced and completely immune to the autoregressive butterfly effect, providing the bedrock model-level guarantee.

---

## WSL2 Profiling & Attribution Strategy (PROF-01)

| Exploration | Findings | Selected Strategy |
|-------------|----------|-------------------|
| `rocprofv3` live probe | Probed live inside WSL2; failed with missing `/sys/class/kfd` (DXG virtualization limitation) | Confirmed Rung (a) probe fails as predicted by research. |
| Web research on upstream practices | Investigated llama.cpp PR #21138 (*Multi-Backend Profiler*), `examples/eval-callback`, and ROCm dispatch issues (#20292, #20218) | Locked Rung (b): High-precision graph evaluation and op timers via `ggml_backend_sched_set_eval_callback` and `--perf`. Rung (c) (Native Linux) remains dormant contingency. |

**User's choice:** Approved Rung (b) baseline profiler.
**Notes:** Upstream core developers routinely use internal graph eval callbacks because they run with microsecond precision across all backends without kernel virtualization issues.

---

## Workload Shapes & Bottleneck Ranking Table (PROF-02)

| Item | Specification | Selected |
|------|---------------|----------|
| Workload Shapes | 4 canonical shapes: 128/128, 128/1024, 4096/128, 4096/1024 | ✓ |
| Dispatch Audit | Evaluate `GGML_HIP_GRAPHS=ON` vs `OFF` to isolate CPU launch overhead from GPU compute | ✓ |
| Optimization Target #1 | Empirical ranking in `BOTTLENECK-TABLE.md` formally names Target #1 for Phase 4/5 | ✓ |

**User's choice:** Approved.
**Notes:** Shapes guarantee balanced prefill and decode coverage across conversational and long-context agentic workloads.

---

## Execution Protocol & Safety Integration

- Re-uses Phase 2 HWiNFO 95 °C thermal watchdog and 3-signal VRAM spill guard.
- Windows toast notifications enabled for gate completions and guard alerts.
- Checksums recorded for all Phase 3 artifacts in `benchmarks/profiling/`.

---

## Claude's Discretion

- Golden baseline JSON schema
- Profiler bridge script structure
- Dataset management in `benchmarks/data/`

---

## Deferred Ideas

- Native Linux instruction-level ISA profiling (deferred unless Rung b fails to identify root cause).
- Autotuning sweeps for kernel block sizes (v2 requirement).
