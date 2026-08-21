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
- [ ] **BENCH-03**: VRAM ledger per run incl. process-RSS guard defeating the WSL2 silent-overcommit failure mode
- [ ] **BENCH-04**: Baseline matrix published: pp/tg × context {4k, 8k, 16k, 32k} × flash-attn {on, off}

### Correctness
- [ ] **QUAL-01**: Op-level gate — `test-backend-ops` green required before any performance claim is accepted
- [ ] **QUAL-02**: Model-level gate — wikitext-2 perplexity within ±1% of published 7.1583±0.25 AND fixed-prompt golden outputs (greedy decode) unchanged within tolerance

### Profiling
- [ ] **PROF-01**: rocprofv3 kernel tracing produces usable per-kernel wall times under WSL2 (Phase-1 exit criterion; failure triggers native-Linux contingency per defined triggers)
- [ ] **PROF-02**: Ranked bottleneck table mapping top kernels → ggml ops across 4 workload shapes (short/long prompt × short/long generation)

### Kernels & Integration
- [ ] **KERN-01**: Kernel playground scaffold operating the full pipeline: CPU reference → HIP implementation → numerical comparison → microbenchmark
- [ ] **KERN-02**: First custom gfx1100 kernel attacks the #1 profiled bottleneck (expected candidates: Gated DeltaNet scan or IQ4_XS quantized matmul); numerically correct vs reference within tolerance
- [ ] **KERN-03**: Winning kernel beats stock in microbenchmark AND end-to-end A/B with correctness gates intact
- [ ] **INTEG-01**: Winning kernel integrated behind ON/OFF compile/runtime switch via quilt patch series over pinned upstream; stock baseline build remains permanently available
- [ ] **PUB-01**: Final deliverable published: complete stock-vs-optimized result matrix, raw data, methodology, known limitations

## v2 Requirements (deferred)

- Autotuning loop (tile/workgroup/LDS sweeps → tuned-config registry)
- Shape/batch sweep matrices beyond BENCH-04 basics
- rocWMMA experiments (A/B flag only; known wave-cooperative regressions risk)
- MTP speculative-decoding benchmark dimension + draft-model experiments
- KV-cache optimization phase (de-prioritized: hybrid arch KV ≈64 KiB/token est.)
- Additional quant comparators (Q4_K_M comparator optional in v1; Q6_K/Q8_0 deferred)
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
| (filled by roadmap) | | |

---
*Created 2026-08-21 after research phase; owner approved v1 scope in-session.*
