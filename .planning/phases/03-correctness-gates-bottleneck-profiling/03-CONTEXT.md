# Phase 3: Correctness Gates & Bottleneck Profiling - Context

**Gathered:** 2026-08-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver two-tier automated correctness gates (op-level mathematical unit tests and model-level perplexity/logit quality checks) and execute workload profiling across 4 canonical inference shapes on the locked `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` model on the AMD Radeon RX 7900 XT (gfx1100). Build a ranked bottleneck attribution table that formally names Optimization Target #1 before any custom kernel authoring in Phases 4 & 5.

Requirements in scope: QUAL-01, QUAL-02, PROF-01, PROF-02.
</domain>

<decisions>
## Implementation Decisions

### Op-Level Correctness Gate (QUAL-01)
- **D3-01:** Full 128-operation suite executed via `test-backend-ops test -b ROCm0` in a scripted wrapper (`benchmarks/bin/run_op_gate.py`). Zero-tolerance failure policy: any non-passing operation produces exit code `1`, writes a failure artifact, and blocks downstream benchmark acceptance.
- **D3-02:** Strict explicit assertions on Qwen 3.8 hybrid architecture operations: `GATED_DELTA_NET`, `SOLVE_TRI`, `SSM_CONV`, `SSM_SCAN`, `FLASH_ATTN_EXT`, and `MUL_MAT` (NMSE $< 10^{-7}$ against FP64 CPU reference).

### Model-Level Quality Gate (QUAL-02)
- **D3-03:** 3-layer verification hierarchy avoiding autoregressive token-divergence false-negatives (grounded in Horace He's batch-invariance research):
  1. **Perplexity Gate:** WikiText-2 test split (`benchmarks/data/wiki.test.raw`) at context 2048 via `llama-perplexity`. Acceptance threshold: $\pm 1\%$ of stock baseline reference ($7.1583 \pm 0.07$, acceptable range `7.087` to `7.230`).
  2. **Early-Token Canary Gate:** 6 deterministic prompt files from Phase 2 evaluated for exact token match on the first 32 tokens of structured prompts.
  3. **Logit Similarity Gate:** Logit cosine similarity $> 0.999$ and top-1 probability divergence verified on fixed prompt completion boundaries.
- **D3-04:** Golden baselines saved to `benchmarks/golden/stock_baseline_golden.json` capturing stock token streams and SHA256 hashes.

### WSL2 Profiling & Attribution Strategy (PROF-01)
- **D3-05:** 3-rung profiling ladder resolved:
  - **Rung (a) (Probe):** Live probe executed on stock ROCm 7.2.1 under WSL2. As anticipated by research, `rocprofv3` aborts due to missing `/sys/class/kfd` in DXG paravirtualization (upstream PR #7016 unmerged and RDNA 3.5 specific).
  - **Rung (b) (Locked Baseline):** High-precision graph evaluation and op timers via `llama.cpp` / `ggml` instrumentation (`eval-callback` / `ggml_backend_sched_set_eval_callback` and `--perf`). Provides microsecond-accurate breakdown across prefill and decode phases without hardware counter dependencies.
  - **Rung (c) (Contingency):** Native-Linux dual-boot profiling session remains dormant fallback if instruction-level PM counter attribution is ever required.
- **D3-06:** Scripted profiler (`benchmarks/bin/profile_workload.py`) and parser (`benchmarks/lib/parse_profile.py`) capture and aggregate GPU op latencies.

### Workload Shapes & Bottleneck Ranking Table (PROF-02)
- **D3-07:** 4 canonical workload shapes evaluated:
  1. *Short Prompt / Short Gen:* $p=128, n=128$ (Interactive chat / single turn)
  2. *Short Prompt / Long Gen:* $p=128, n=1024$ (Code generation / creative writing)
  3. *Long Prompt / Short Gen:* $p=4096, n=128$ (Document QA / summarization)
  4. *Long Prompt / Long Gen:* $p=4096, n=1024$ (Agentic refactoring / multi-turn)
- **D3-08:** Dispatch overhead audit evaluating `GGML_HIP_GRAPHS=ON` vs `OFF` to isolate CPU kernel launch latency from GPU execution time.
- **D3-09:** Publication of `benchmarks/profiling/BOTTLENECK-TABLE.md` mapping op runtime percentages, classifying bound types (Memory Bandwidth, Compute/Register Bound, Dispatch Latency Bound), and designating **Optimization Target #1** for Phase 4 & Phase 5.

### Execution & Safety Integration
- **D3-10:** Re-use of Phase 2 HWiNFO 95 °C thermal watchdog and 3-signal VRAM spill guard during profiling and gate runs. Windows toast notifications fire on gate completion or guard trips.

### Claude's Discretion
- Exact format and schema of `benchmarks/golden/stock_baseline_golden.json`.
- Scripting language and CLI argument structure of `run_op_gate.py` and `run_model_gate.py`.
- Internal implementation details of the graph op profiler callback bridge.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning docs
- `.planning/ROADMAP.md` — Phase 3 success criteria, WSL2 risk notes, binding methodology rules
- `.planning/REQUIREMENTS.md` §Correctness & Profiling — QUAL-01, QUAL-02, PROF-01, PROF-02 authoritative text
- `.planning/phases/02-benchmark-harness-baseline-matrix/02-CONTEXT.md` — Phase 2 established harness, store, guard, and telemetry infrastructure
- `benchmarks/environment/versions.txt` — locked version pairings and environment parameters

### External research & upstream references
- Horace He (Thinking Machines Lab), *"Defeating Nondeterminism in LLM Inference"* (2025) — batch invariance, floating-point non-associativity, and cross-kernel verification
- llama.cpp PR #21138 (*Multi-backend profiler*) & `examples/eval-callback` — low-overhead op timing via `ggml_backend_sched_set_eval_callback`
- llama.cpp Issue #20292 (*Qwen 3.5 CPU bound on rocm / terrible performance*) — dispatch overhead audit & `rocprof --stats` findings on RDNA3
- llama.cpp Issue #20218 & Buun PR — RDNA3 `gated_delta_net` VGPR register spill and `__launch_bounds__` analysis
- `tests/test-backend-ops.cpp` @ `bb4caa75` — mathematical op verification suite, NMSE error formulas, and CPU reference comparison

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `baseline/binaries/v0.2.0-bb4caa75/` — stock binaries (`test-backend-ops`, `llama-perplexity`, `llama-cli`, `llama-bench`)
- `models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` — locked 15.31 GB model artifact
- `benchmarks/host/hwinfo_daemon.py` — Windows-side HWiNFO shared-memory telemetry daemon & thermal kill-switch
- `benchmarks/lib/guard.py` — 3-signal VRAM spill guard
- `benchmarks/prompts/` — 6 deterministic prompt corpus files (code, reasoning, prose)

### Established Patterns
- Environment variables: `export HSA_ENABLE_DXG_DETECTION=1`, `export LD_LIBRARY_PATH=/root/llama.cpp/build-ci/bin:$LD_LIBRARY_PATH`
- Headless invocation: `setsid` + `--simple-io` + `--single-turn` + explicit `-c` + `--no-mmap`
- Windows interop: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root bash -c "..."`

</code_context>

<specifics>
## Specific Ideas

- Ensure WikiText-2 test dataset is downloaded once and verified with SHA256 in `benchmarks/data/`.
- Ensure golden baselines capture logits alongside tokens to enable early detection of precision degradation.
- Table output in `BOTTLENECK-TABLE.md` must clearly separate prefill vs decode percentages as mandated by Rule 4.

</specifics>

<deferred>
## Deferred Ideas

- Deep instruction-level ISA profiling via native Linux session (deferred unless Rung b fails to identify root cause).
- Autotuning sweeps for kernel block sizes (v2 requirement).

</deferred>

---

*Phase: 3-Correctness Gates & Bottleneck Profiling*
*Context gathered: 2026-08-24*
