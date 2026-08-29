# v1 Requirements

**Project:** gfx1100-Tuned Inference Path for Qwen3.8-27B on RX 7900 XT
**Mode:** Horizontal layers (parallel-plan friendly)
**Primary artifact:** `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB, sha256 53adc4bb…) — see `.planning/research/MODEL-DECISION.md`

## v1 Requirements

### Environment & Baseline

- [x] **ENV-01**: WSL2 ROCm stack validated — `rocminfo` enumerates gfx1100, HIP runtime functional under Adrenalin 26.2.2-for-WSL2 + ROCm 7.2.1 guest
- [x] **ENV-02**: Stock llama.cpp HIP build pinned to one commit targeting gfx1100; `llama-cli`, `llama-bench`, `llama-perplexity`, `test-backend-ops` all built
- [x] **ENV-03**: Model runs fully on GPU at baseline — Gated DeltaNet + gated-attention paths with zero CPU fallback (hybrid-arch coverage gate)
- [x] **ENV-04**: IQ4_XS artifact downloaded, sha256-verified, provenance (repo, commit, imatrix, quant metadata) recorded in `models/README.md`

### Benchmarking

- [x] **BENCH-01**: Reproducible benchmark harness wrapping llama-bench — fixed workload profiles, enforced pp/tg split, warmup + ≥3 repeats, machine-readable output
- [x] **BENCH-02**: Every result row fingerprinted — llama.cpp commit, ROCm/driver versions, GGUF sha256, clocks/temps via Windows-side telemetry
- [x] **BENCH-03**: VRAM ledger per run incl. process-RSS guard defeating the WSL2 silent-overcommit failure mode; fail-fast allocation policy (no retry loops), supervised synthetic overcommit test, crash-resilient result journal
- [x] **BENCH-04**: Baseline matrix published: pp/tg × context {4k, 8k, 16k, 32k} × flash-attn {on, off}, plus a stock-Vulkan comparator arm at its own pinned commit (GDN coverage verified there first); every claim names its backend; 32k tier gated by empirical free-VRAM pre-flight with expected-FAIL path under WSL2

### Correctness

- [x] **QUAL-01**: Op-level gate — `test-backend-ops` green required before any performance claim is accepted (`benchmarks/bin/run_op_gate.py`, 21,093 cases, 0 errors, core ops asserted)
- [x] **QUAL-02**: Model-level gate — wikitext-2 perplexity within ±1% of stock baseline (6.4271±0.04103) AND fixed-prompt golden outputs (greedy decode) verified (`benchmarks/bin/run_model_gate.py`, `stock_baseline_golden.json`)

### Profiling

- [x] **PROF-01**: Kernel-level attribution strategy resolved via standalone C++ evaluation callback profiler (`benchmarks/bin/eval_profiler`) instrumenting `ggml_backend_sched_set_eval_callback`; dispatch latency evaluated with HIP graph audit (`benchmarks/profiling/dispatch_overhead_report.md`)
- [x] **PROF-02**: Ranked bottleneck table mapping top kernels → ggml ops across 4 workload shapes (S1–S4); `MUL_MAT` formally designated as Optimization Target #1 (31.12% cumulative GPU time; `BOTTLENECK-TABLE.md`, `bottleneck_summary.json`)

### Kernels & Integration

- [x] **KERN-01**: Kernel playground scaffold operating the full pipeline: CPU reference → HIP implementation → numerical comparison → microbenchmark (demo op `dequant_iq4_xs` only per owner lock D4-00-1; vendored `block_iq4_xs.h` D4-00-2; reuse `benchmarks/lib/store.py` D4-00-3; tight gate max_abs 1e-5/mean 1e-6/cosine 0.99999 +10× D4-00-4; templated `WARP_SIZE` D4-00-5 — see `.planning/phases/04-kernel-playground-scaffold/04-CONTEXT.md`) — **DONE 2026-08-25**
- [x] **KERN-02**: Custom gfx1100 `MUL_MAT` IQ4_XS GEMV (Wave32, 128-bit uint4, 8-thread/row) + GEMM (TILE_M=16 + WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, B_lds[2][32][33]) in `kernels/matmul_iq4xs/` — numerically correct `cosine=1.0, max_rel=0` vs FP64 `ref_cpu` (10/10 GEMV, 11/11 GEMM) — **DONE 2026-08-25**
- [x] **KERN-03**: Microbenchmark win: GEMV 8/8 1.26–2.13× vs stock HIP (512–54 GB/s), GEMM 7/9 1.47–7.50× for M≥128 (6.7× @512 WMMA), 2× M=16 losses logged per Rule #10; unified 32× sweep archived via `RunStore` (`kernels_mul_mat_iq4xs*`); provisional `patches/phase5_mul_mat_custom.patch` (ON/OFF) ready for e2e A/B with QUAL-01/02 green — **DONE 2026-08-25**
- [x] **INTEG-01**: Winning kernel integrated behind ON/OFF compile/runtime switch via quilt patch series over pinned upstream; stock baseline build remains permanently available — **DONE 2026-08-25**
- [x] **PUB-01**: Final deliverable published: complete stock-vs-optimized result matrix, raw data, methodology, known limitations — **DONE 2026-08-25**

### Phase 7 — Hybrid + Windows + 10% + 10× Rigour (added 2026-08-28, owner 3 wishes + deep-research report `output/deep-research/1000t-s-at-8k-gfx1100.md`)

Owner must-have outputs for Phase 7 — these supersede the earlier “> stock” phrasing (now quantified) and the WSL2-only dev assumption. Plans 07-01..07-04 MUST satisfy all four; verifier gates on them.

- [ ] **REQ-WIN-07 — Windows-native, ≤2 language runtimes**: Repo builds & runs natively on **Windows 11 + VS Build Tools + AMD HIP SDK** (no WSL2 required for build or run) with **pure C++/HIP + minimal CMake + `build_windows.bat`**. No Python/JS/`3+` language-server stack remains in the shipped tree — `find -name "*.py" ! -path "./llama.cpp/*" == 0` after prune, and `build_windows.bat` uses `clang++.exe --offload-arch=gfx1100` (not `cl`) via `HIP_PATH`. Verified by `build_windows.bat` clean compile + `llama-server.exe` serves `curl http://127.0.0.1:8000/v1/chat/completions → 200` on gfx1100. *Extends REQ-WIN-01..04 from Phase 8 into Phase 7 as must-have output #1; Phase 8 remains the pruning/refactor phase that lands it.*
- [ ] **REQ-PERF-07 — ≥10% end-to-end uplift vs stock llama.cpp**: Paired `llama-bench` A/B (stock `OFF` vs custom `ON`, same `bb4caa75`, `-ngl 99`, `-b 2048`) shows **custom `median tok/s` ≥1.10× stock** at every measured tier `{512,1024,2048,4096}` *and* at `8192` (if VRAM pre-flight passes) for **both** `pp (prefill)` and `tg (decode)` split — not just microbench vs naive. Requirement fails if either `pp` or `tg` uplift `<10%` at any tier, or if variance (`stddev` over `N=10`) makes `mean−1σ` fall below `1.10×`. Microbench `>1.2×` vs real DP4A (`vec_dot_iq4_xs_q8_1`) is a *necessary* but not sufficient condition; end-to-end `llama-bench` is the gate.
- [ ] **REQ-STAT-07 — 10× averaged (15× for LLM QA)**: Every quantitative claim in Phase 7 is **averaged over `N≥10` repetitions** (microbench `bench_*` → `median_us` + `mean` + `p95` + `stddev` over `10` runs; `llama-bench` → `10` runs per tier per build in **one thermal window**; quality gates → `10` repeats of `test-backend-ops`/`run_model_gate` where stochastic). **LLM question tests** (e.g., asking the model via the custom kernel through `llama-cli`/`llama-server`) are **repeated `N≥15` times** with prompt-fixed, temp-0, and results presented as `avg tok/s` + `avg latency` + `stddev` + `per-run table` (+ `median` for tok/s). Single-run numbers are **never** reported as a claim; only the averaged row is the verdict. *Amends BENCH-01*: `≥3` repeats → `≥10` (≥15 for LLM QA) for Phase 7 onward.
- [ ] **BENCH-01 (amended 2026-08-28)**: For Phase 7 onward `≥10` repeats with `pp/tg` split, warmup, and `RunStore` + `CHECKSUMS`; LLM QA `≥15`. Single-number claims banned.

### Context Scaling — 256k (added 2026-08-22, owner request)

Feasibility + math: `.planning/research/CONTEXT-SCALING.md`. Model natively supports 262,144 ctx; only 16/64 layers carry KV (~64 KiB/token f16); 48 DeltaNet layers hold constant state.

- [ ] **CTX-01**: Measured context×KV-scheme VRAM ledger under the WSL2 DXG budget — llama.cpp startup breakdown at {32k, 64k, 128k} × {f16, q8_0, q4_K} KV; replaces MEDIUM-confidence estimates; extends BENCH-03 ledger format
- [ ] **CTX-02**: KV quantization quality gate — wikitext-2 ppl + golden-output deltas per KV scheme vs fp16 baseline, accepted only within QUAL-02 tolerance; publish scheme×context×quality matrix
- [ ] **CTX-03**: No-tiering ceiling verdict — best-effort max-context fit attempt with q4_K KV (+ optional IQ3-class weight comparator arm) on both HIP/WSL and Vulkan/native arms; publish measured max stable context per backend
- [ ] **CTX-04**: Host-tiered KV prototype — hot window in VRAM / cold prefix in host RAM for the 16 attention layers; correctness via QUAL gates unchanged (pure data movement); probe librocdxg pinned-memory knob first, pageable fallback documented; target ≥128k stable under WSL2
- [ ] **CTX-05**: Long-context prefill strategy — chunked-prefill timing curve 32k→256k, persistent prompt-cache across runs, and DeltaNet-state checkpoint resume at semantic anchors for agentic context edits (FreeToken §3.1 analog)

## v2 Requirements (deferred)

- Autotuning loop (tile/workgroup/LDS sweeps → tuned-config registry)
- Shape/batch sweep matrices beyond BENCH-04 basics
- rocWMMA experiments (A/B flag only; known wave-cooperative regressions risk)
- Full hierarchical KV paging (attention-pattern-aware cold-tier recall) beyond CTX-04's recency/static policies
- 256k as hard v1 exit criterion if CTX-01/02 show quality-budget conflicts — owner to confirm after CTX-01 lands
- MTP speculative-decoding benchmark dimension + draft-model experiments
- KV-cache optimization phase (de-prioritized: hybrid arch KV ≈64 KiB/token est.)
- Additional quant comparators (Q4_K_M comparator optional in v1; Q6_K/Q8_0 deferred)
- Coding-capability eval (HumanEval-style) for the locked artifact — recommended before relying on it as a daily coding agent (artifact ships with zero generative/code evals; Heretic maintainer flags down_proj ablation as potentially intelligence-damaging)
- Qwen graph/tensor-level report deep-dive (Phase-10 style)

## Out of Scope

- Complete inference engine from scratch — builds on llama.cpp by explicit decision
- Multi-GPU / tensor parallelism — single-card project
- Vision pipeline (mmproj) — capability preserved, not optimized
- Serving UI / deployment tooling beyond llama-server smoke use
- HauhauCS patched runtime or custom K_P quants as baseline (contaminates measurement; eval-only variant)
- ~~Native Windows HIP SDK as build target~~ → **IN SCOPE since 2026-08-28** (REQ-WIN-07, Phase 7/8); Turkish output-quality evals; Q5_K_M+ quants on 20 GB card

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| ENV-03 | Phase 1 | Complete |
| ENV-04 | Phase 1 | Complete |
| BENCH-01 | Phase 2 | Complete |
| BENCH-02 | Phase 2 | Complete |
| BENCH-03 | Phase 2 | Complete |
| BENCH-04 | Phase 2 | Complete |
| QUAL-01 | Phase 3 | Complete |
| QUAL-02 | Phase 3 | Complete |
| PROF-01 | Phase 3 | Complete |
| PROF-02 | Phase 3 | Complete |
| KERN-01 | Phase 4 | Complete (2026-08-25) |
| KERN-02 | Phase 5 | Complete (2026-08-25) |
| KERN-03 | Phase 5 | Complete (2026-08-25) |
| INTEG-01 | Phase 6 | Complete (2026-08-25) |
| PUB-01 | Phase 6 | Complete (2026-08-25) |
| REQ-WIN-07 | Phase 7 (must-have #1) + Phase 8 | — must-have #1: Windows-native ≤2 langs (`build_windows.bat` + `llama-server.exe` @ `localhost:8000`) |
| REQ-PERF-07 | Phase 7 (must-have #2) | — must-have #2: `≥1.10×` stock `pp`+`tg` at `{512,1024,2048,4096,8192}` (median over `N=10` per tier, `mean−1σ ≥1.10×`) |
| REQ-STAT-07 | Phase 7 (must-have #3) | — must-have #3: every test `N≥10` averaged (`median`/`mean`/`stddev`), LLM QA `N≥15` presented |
| BENCH-01 (amended) | Phase 7 | `≥10` repeats (`≥15` LLM QA), `pp/tg` split, `RunStore`+`CHECKSUMS` |

---
*Created 2026-08-21 after research phase; owner approved v1 scope in-session.*
