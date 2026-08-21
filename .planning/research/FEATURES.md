# Feature Landscape

**Domain:** GPU LLM-inference optimization — a gfx1100 (RDNA3)-tuned inference path for a ~27–32B-class model, starting from stock llama.cpp HIP backend
**Researched:** 2026 (current session; ROCm 7.2.x-era ROCDXG/WSL2 stack)
**Overall feature confidence:** MEDIUM-HIGH — core workflow features HIGH; platform/profiling specifics MEDIUM-to-LOW (see Sources & session limitation note)

> **Scope decisions locked with supervisor (this session):**
> 1. **Model ID discrepancy:** "Qwen3.8-27B" matches no public release. Canonical candidates: **Qwen3-32B (dense)** and **Qwen3-30B-A3B (MoE)**. Final selection is PENDING-USER; features below are model-agnostic with **[DENSE]** / **[MOE]** deltas where they diverge.
> 2. **Environment:** WSL2 primary (ROCm 7.2.x-era ROCDXG stack); native Linux = contingency fallback only; native Windows HIP SDK = out of scope (footnote in Anti-Features).
> 3. **VRAM envelope:** 20 GB card. At Q4_K_M both candidates leave **<1.5 GB** for KV cache + compute buffers (dense ≈19.76 GB; MoE ≈18.6 GB weights-only) → VRAM budgeting is mandatory table stakes, not optional polish.

---

## Table Stakes

Features without which the project cannot execute its own methodology (baseline first, profile before optimizing, one bottleneck at a time). Missing any of these makes later "improvements" unprovable.

| # | Feature | Why Expected | Complexity | Confidence |
|---|---------|--------------|------------|------------|
| TS1 | **Locked baseline artifact set** — stock llama.cpp HIP build at a pinned commit; binary, build flags, and run logs archived and tagged; never silently rebuilt | Every claimed win is a delta against this. Losing/rebuilding the baseline invalidates the whole comparison chain | Low | HIGH |
| TS2 | **Environment/version fingerprint on every result row** — llama.cpp commit, ROCm release, ROCDXG/WSL kernel + driver versions, GPU model/clocks/thermals (rocm-smi), GGUF file SHA256 + quant type, compile defs, all `GGML_CUDA_*` env vars | Un-fingerprinted results are unreproducible noise; WSL2 adds kernel/driver layers that silently change perf | Low-Med | HIGH |
| TS3 | **Reproducible benchmark harness** wrapping `llama-bench` — fixed seeds, fixed prompt sets, warmup pass, ≥5 timed repeats, machine-readable output (JSON/CSV), env fingerprint embedded per row | Single-run numbers on a desktop GPU are dominated by variance (clocks, thermals); repeats + statistics are the minimum credible unit | Medium | HIGH (tool exists; wrapper is project code) |
| TS4 | **Prefill/decode split metrics enforced** — every benchmark records prompt-processing (pp) and text-generation (tg) separately; blended single numbers banned in all artifacts | Optimization levers differ completely (prefill = GEMM/compute-bound; decode = memory-bandwidth-bound); a blended number hides which path you improved | Low | HIGH |
| TS5 | **Correctness gate: perplexity + KL divergence** — `llama-perplexity` on a pinned corpus slice; for any kernel/backend change, KL divergence vs baseline logits; acceptance thresholds (propose: relative ppl drift ≤0.5%, zero NaN/garbage outputs) | Speed without output fidelity is a bug, not an optimization. The gate converts "feels faster" into "faster AND equivalent" | Medium | HIGH (tooling exists; thresholds are project-set defaults) |
| TS6 | **Golden-output regression tests** — pinned prompts, greedy decode, exact token-stream + top-k logit snapshots compared after every change | Catches silent numerical corruption that short perplexity runs can miss; cheap CI-style safety net for a fork-heavy workflow | Low-Med | MEDIUM |
| TS7 | **Per-quant VRAM budget tracker + OOM boundary map** — measured weights+KV+compute-buffer totals per (quant × context) combo against the 20 GB ceiling; explicit headroom margin rule | Both candidate models at Q4_K_M leave <1.5 GB headroom (supervisor-verified); naive context-length choices will OOM. Data also arbitrates the PENDING-USER model choice | Low | HIGH |
| TS8 | **Context-length × KV-cache tradeoff matrix** — n_ctx ∈ {4k, 8k, 16k, 32k} × K/V dtype {f16, q8_0, q4_0} × flash-attn on/off; records t/s, VRAM, and ppl impact of KV quantization | With <1.5 GB headroom at Q4_K_M/f16-KV, context length and KV quant are forced tradeoffs, not free settings; KV-quant correctness must be gated like any change | Low-Med | HIGH need; MEDIUM exact values |
| TS9 | **Quant-sweep benchmark suite** — benchmark every candidate quant that fits budget (Q3_K_M, Q4_K_S, Q4_K_M, IQ4_XS, …), each point paired with its perplexity number | Produces the speed-vs-quality frontier that decides dense-vs-MoE and quant level with data instead of vibes; feeds the model-choice decision | Medium | HIGH (sizes supervisor-verified; sweep is straightforward) |
| TS10 | **Profiling hooks** — scripted `rocprof`/`rocprofv3` captures with saved traces; must auto-detect WSL2 limitations and fall back to the native-Linux contingency path | Methodology literally cannot proceed past baseline without profile-before-optimize; kernel timings are the input to bottleneck ranking | Med-High | MEDIUM overall — **LOW for WSL2 profiling reliability specifically** (historically limited; verify at phase start) |
| TS11 | **Append-only result store + run-ID convention** — `results/<date>_<run-id>/config.json + metrics.csv + logs/`; immutability discipline | Post-hoc edited results destroy trust in every conclusion; append-only makes regression analysis and published matrices possible | Low | HIGH |
| TS12 | **Run protocol document (determinism discipline)** — warmup rules, repeat counts, thermal soak policy, clock/thermal recording (consumer RDNA3 can't reliably lock clocks → record, don't control), same-shell/env checklist | Makes TS3's "reproducible" claim actually hold across days/weeks of work | Low | MEDIUM |

## Differentiators

Not expected of such projects, but what separates a credible optimization campaign from a pile of anecdotes.

| # | Feature | Value Proposition | Complexity | Confidence |
|---|---------|-------------------|------------|------------|
| D1 | **Autotuning loop** — declarative knob sweeps (compile defs e.g. `-DGGML_CUDA_F16`, FA-all-quants builds; runtime knobs: n_batch/n_ubatch, thread counts, CUDA-graph toggles, KV dtypes), each candidate validated through the TS5 correctness gate, emitting a ranked best-config registry | Turns one-off tinkering into a searchable optimum; the correctness gate keeps the tuner honest | High | MEDIUM |
| D2 | **Shape/batch sweep matrices** — systematic batch × context × quant × FA grids beyond llama-bench defaults | Reveals which optimizations help which workload regime (interactive chat vs long-context batch) | Medium | MEDIUM |
| D3 | **Published result matrices committed to the repo** — versioned markdown+CSV tables at every phase gate; the project's shareable deliverable | External verifiability; lets others reproduce on their own 7900-series cards; forces tidy bookkeeping | Low-Med | HIGH |
| D4 | **Kernel-level hotspot attribution** — parse profiler output, map to ggml ops (mul_mat, softmax, rope, norms, dequant), rank by time; names THE bottleneck for the next phase | Operationalizes "one bottleneck at a time"; prevents optimizing things that don't matter | Med-High | MEDIUM (depends on TS10 working under WSL2) |
| D5 | **rocWMMA / WMMA microbenchmarks for gfx1100 GEMM paths** — targeted mul_mat-shape experiments using RDNA3's matrix-core instructions | Potentially the biggest single-kernel win if profiles show GEMM dominance — **[DENSE]** payoff highest (uniform large matmuls) | High | LOW-MEDIUM (rocWMMA gfx1100 coverage must be re-verified at phase start; go/no-go decided by D4) |
| D6 | **Regression gate / trend tracker** — phase fails if >X% perf regression at equal correctness vs prior best; start as plain CSV-diff, dashboards later if ever | Prevents "two steps forward, one step back" drift across 18 phases | Medium | MEDIUM |
| D7 | **[MOE] Expert-routing instrumentation + CPU-offload strategy sweeps** — per-expert activation histograms; `--n-cpu-moe` / tensor-override sweeps exploiting MoE's low active-parameter count | MoE hotspots (routing, expert gather/scatter, host-device traffic) are invisible to dense-style profiling; offload sweeps can buy large VRAM headroom | Med-High | MEDIUM — only relevant if Qwen3-30B-A3B is selected |
| D8 | **Speculative-decoding evaluation harness** — draft/target pairs, acceptance-rate-adjusted tg reporting | Decode-bandwidth-bound RDNA3 cards benefit disproportionately; evaluate before ever implementing | High | MEDIUM-LOW priority — defer until D4 proves decode-bound plateau |

## Anti-Features

Explicitly NOT building. Saying no is part of the feature plan.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|--------------------|
| **Custom inference engine / full fork-and-rewrite** | Rewrite risk swamps a research project; loses upstream fixes weekly; 18 phases die in month one | Tracked local patch overlay (patch series in repo) + upstream PRs for anything generalizable |
| **Multi-GPU / tensor parallelism** | Single 20 GB card is the locked scope; TP adds synchronization nondeterminism and doubles the debugging surface | Squeeze one device: quant choice, KV quant, layer/CPU offload ([MOE]: `--n-cpu-moe`) |
| **Serving stack / API / chat UI features** | Not research outputs; server plumbing consumes phases without producing evidence | Use `llama-cli`/`llama-server` minimally as eval harnesses only |
| **Native Windows HIP SDK support** | Locked by supervisor scoping; dual-platform maintenance halves iteration speed | WSL2 primary; native Linux contingency; footnote the Windows path, nothing more |
| **Inventing new quant formats** | Invalidates all comparability with ecosystem results; llama.cpp's quant zoo already spans the frontier | Standard quant sweep (TS9) over existing GGUF types |
| **Network/distributed inference (RPC)** | Scope creep orthogonal to kernel optimization | None — explicitly out |
| **Persistent monitoring dashboards/services** | Operational bloat; static artifacts answer the actual questions | Scheduled runs appending to the CSV result store (TS11) |
| **Overclock/undervolt tuning persistence** | Hardware risk, non-reproducible across silicon, confounds every measurement | Record clocks/thermals per run (TS2/TS12); never tune them |
| **Training/fine-tuning tooling** | Different domain entirely | None |
| **Early speculative-decoding implementation** | Premature before decode-bound plateau is proven; big implementation cost | Evaluation harness only (D8), deferred |

## Feature Dependencies

```
TS1 Baseline preservation ──────→ (everything; root dependency)
TS2 Version fingerprint ────→ TS3 Benchmark harness (fingerprints embedded per row)
TS3 Harness ────→ TS4 pp/tg split · TS8 ctx×KV matrix · TS9 quant sweep · D1 autotune · D2 shape sweeps
TS5 Correctness gate ────→ acceptance of ANY optimization · D1 autotune · D6 regression gate
TS7 VRAM budget ────→ TS8 ctx×KV matrix · TS9 quant sweep · all offload strategies
TS10 Profiling hooks ────→ D4 hotspot attribution ────→ D1 tuning targets · D5 rocWMMA go/no-go
TS11 Result store ────→ D3 published matrices · D6 regression tracker
D7 [MOE] instrumentation ────→ requires model selection = Qwen3-30B-A3B
Critical chain: TS1→TS3→TS5→TS10→D4  (methodology spine; everything else hangs off it)
```

## MVP Recommendation

Prioritize (maps cleanly onto the original 18-phase methodology):
1. **TS1 + TS2 + TS11 + TS12** — baseline preserved, logged, immutable (early phases).
2. **TS3 + TS4 + TS5 + TS6** — reproducible harness with pp/tg split and the correctness gate; nothing else is meaningful until this exists.
3. **TS7 + TS9 + TS8** — VRAM budgets, quant sweep, ctx×KV matrices → these generate the data that settles the PENDING-USER dense-vs-MoE decision.
4. **TS10 + D4** — profiling works end-to-end and produces ranked bottlenecks (methodology's engine).
5. **First differentiator: D1 autotuning**, publishing via **D3**.

Defer: **D5** rocWMMA (until D4 proves GEMM-bound), **D7** MoE instrumentation (until model selected), **D8** speculative decoding, fancy **D6** tracking (CSV-diff first).

## Sources

- **Supervisor-relayed web verification (this session, HIGH confidence):** candidate model GGUF sizes — Qwen3-32B: Q4_K_M 19.76 GB, Q4_K_S 18.77 GB, Q3_K_M ≈16 GB; Qwen3-30B-A3B: Q4_K_M ≈18.6 GB, IQ4_XS ≈16.5 GB; both leave <1.5 GB KV/runtime headroom at Q4_K_M on 20 GB. WSL2-primary environment decision; ROCm 7.2.x-era ROCDXG stack.
- **Session limitation (disclosed):** this researcher instance had no direct web-search/bash/write tools; the research-plan provider seam could not be executed. Items marked MEDIUM/LOW above rest on established upstream-documentation knowledge and carry an **unverified-web-this-session** tag. MUST be re-verified at phase start: `llama-bench` flag surface (`-r` repeats, `-p`/`-n` split, `-o` json/csv/md), `llama-perplexity` KL-divergence mode, **rocprof-v3 behavior under WSL2 (LOW confidence — known historical pain point; contingency = native Linux)**, rocWMMA gfx1100 kernel coverage (LOW-MEDIUM), RDNA3 WMMA/matrix-core availability (architecturally documented, HIGH).
- **Verification protocol note:** Phase 0 must pin the actual llama.cpp commit and ROCm release before any benchmarking — this requirement IS the table-stakes feature set above.
