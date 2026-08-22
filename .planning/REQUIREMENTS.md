# v1 Requirements

**Project:** gfx1100-Tuned Inference Path for Qwen3.8-27B on RX 7900 XT
**Mode:** Horizontal layers (parallel-plan friendly)
**Primary artifact:** `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB, sha256 53adc4bb…) — see `.planning/research/MODEL-DECISION.md`

## v1 Requirements

### Environment & Baseline
- [ ] **ENV-01**: WSL2 ROCm stack validated — `rocminfo` enumerates gfx1100, HIP runtime functional under Adrenalin 26.2.2-for-WSL2 + ROCm 7.2.1 guest
- [ ] **ENV-02**: Stock llama.cpp HIP build pinned to one commit targeting gfx1100; `llama-cli`, `llama-bench`, `llama-perplexity`, `test-backend-ops` all built
- [ ] **ENV-03**: Model runs fully on GPU at baseline — Gated DeltaNet + gated-attention paths with zero CPU fallback (hybrid-arch coverage gate)
- [ ] **ENV-04**: IQ4_XS artifact downloaded, sha256-verified, provenance (repo, commit, imatrix, quant metadata) recorded in `models/README.md`

### Benchmarking
- [ ] **BENCH-01**: Reproducible benchmark harness wrapping llama-bench — fixed workload profiles, enforced pp/tg split, warmup + ≥3 repeats, machine-readable output
- [ ] **BENCH-02**: Every result row fingerprinted — llama.cpp commit, ROCm/driver versions, GGUF sha256, clocks/temps via Windows-side telemetry
- [ ] **BENCH-03**: VRAM ledger per run incl. process-RSS guard defeating the WSL2 silent-overcommit failure mode; fail-fast allocation policy (no retry loops), supervised synthetic overcommit test, crash-resilient result journal
- [ ] **BENCH-04**: Baseline matrix published: pp/tg × context {4k, 8k, 16k, 32k} × flash-attn {on, off}, plus a stock-Vulkan comparator arm at its own pinned commit (GDN coverage verified there first); every claim names its backend; 32k tier gated by empirical free-VRAM pre-flight with expected-FAIL path under WSL2

### Correctness
- [ ] **QUAL-01**: Op-level gate — `test-backend-ops` green required before any performance claim is accepted
- [ ] **QUAL-02**: Model-level gate — wikitext-2 perplexity within ±1% of published 7.1583±0.25 AND fixed-prompt golden outputs (greedy decode) unchanged within tolerance

### Profiling
- [ ] **PROF-01**: Kernel-level attribution strategy resolved (binding gate in Phase 3; Phase 1 runs a non-binding rocprofv3 feasibility probe) — llama.cpp op-timers are the planned baseline with counter-less attribution pre-authorized; working rocprofv3/DXG capture is upside; native-Linux profiling session is the sanctioned escalation if timer-based attribution proves insufficient
- [ ] **PROF-02**: Ranked bottleneck table mapping top kernels → ggml ops across 4 workload shapes (short/long prompt × short/long generation)

### Kernels & Integration
- [ ] **KERN-01**: Kernel playground scaffold operating the full pipeline: CPU reference → HIP implementation → numerical comparison → microbenchmark
- [ ] **KERN-02**: First custom gfx1100 kernel attacks the #1 profiled bottleneck (expected candidates: Gated DeltaNet scan or IQ4_XS quantized matmul); numerically correct vs reference within tolerance
- [ ] **KERN-03**: Winning kernel beats stock in microbenchmark AND end-to-end A/B with correctness gates intact
- [ ] **INTEG-01**: Winning kernel integrated behind ON/OFF compile/runtime switch via quilt patch series over pinned upstream; stock baseline build remains permanently available
- [ ] **PUB-01**: Final deliverable published: complete stock-vs-optimized result matrix, raw data, methodology, known limitations

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
- Native Windows HIP SDK as build target; Turkish output-quality evals; Q5_K_M+ quants on 20 GB card

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Pending |
| ENV-02 | Phase 1 | Pending |
| ENV-03 | Phase 1 | Pending |
| ENV-04 | Phase 1 | Pending |
| BENCH-01 | Phase 2 | Pending |
| BENCH-02 | Phase 2 | Pending |
| BENCH-03 | Phase 2 | Pending |
| BENCH-04 | Phase 2 | Pending |
| QUAL-01 | Phase 3 | Pending |
| QUAL-02 | Phase 3 | Pending |
| PROF-01 | Phase 3 | Pending |
| PROF-02 | Phase 3 | Pending |
| KERN-01 | Phase 4 | Pending |
| KERN-02 | Phase 5 | Pending |
| KERN-03 | Phase 5 | Pending |
| INTEG-01 | Phase 6 | Pending |
| PUB-01 | Phase 6 | Pending |

---
*Created 2026-08-21 after research phase; owner approved v1 scope in-session.*
