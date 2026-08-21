# Project Research Summary

**Project:** qwen_3.8_27b_optimizations — gfx1100-Tuned Inference for a Qwen3 27B-Class Model on RX 7900 XT
**Domain:** GPU LLM inference optimization (RDNA3 / HIP / ROCm kernel-level work on stock llama.cpp, developed under WSL2)
**Researched:** 2026-08-21
**Confidence:** HIGH overall (core stack, architecture, and pitfalls verified against official AMD docs, llama.cpp source, and first-party HuggingFace API data; named MEDIUM pockets listed in Confidence Assessment)

> **Standing scoping decisions carried through all four research files:**
> 1. **"Qwen3.8-27B" does not exist.** Canonical candidates locked: **Qwen3-32B (dense)** vs **Qwen3-30B-A3B (MoE)** — final pick **PENDING-USER**, to be closed with measured smoke-test data in the first execution phase. All artifacts below are model-agnostic with [DENSE]/[MOE] deltas where they diverge.
> 2. **Environment:** WSL2 primary (Adrenalin 26.2.2-for-WSL2 driver + ROCm 7.2.1 guest via librocdxg). Native Linux is a contingency triggered by defined conditions, never gated on up front.
> 3. **VRAM envelope:** 20 GB. Planning-doc correction: dense Qwen3-32B **Q4_K_M is 19.76 GB and does NOT fit** (PROJECT.md's "~16–17 GB" assumption is wrong); realistic dense quants are IQ4_XS (17.69 GB) or Q3_K_M (~16 GB). The MoE candidate fits properly at Q4-class. Update PROJECT.md at requirements definition.

---

## Executive Summary

This project is a **kernel-level optimization campaign against stock llama.cpp's HIP backend**, not an engine build — and all four research files independently converge on how experts run such campaigns: freeze a stock baseline forever, fingerprint every measurement with environment metadata, profile real workloads before touching any code, develop kernels in a standalone playground (CPU reference → HIP implementation → numerical compare → microbenchmark), and integrate winners back behind ON/OFF flags via a quilt patch series over a pinned upstream checkout. The architecture is a hub-and-spoke around an append-only, fingerprinted result store; nothing reaches runtime integration except through a two-tier validation gate (op-level tolerance, then model-level perplexity/KL-divergence/golden outputs). The deliverable is a published, reproducible delta-versus-stock — which makes measurement infrastructure, not kernel cleverness, the actual foundation of the project.

**Recommended approach:** WSL2 is viable as the *primary* dev environment on the new production ROCDXG stack (Adrenalin 26.2.2-for-WSL2 + ROCm 7.2.1 guest): building/running llama.cpp HIP, rocprofv3 kernel tracing over `/dev/dxg`, and hipEvent microbenchmarks all work today. Guest-side `rocm-smi`/`amd-smi` do **not** work (telemetry must come from Windows-side tooling from day one), hardware-counter depth is newly-shipped and must be validated as a Phase-1 exit criterion, and a silent VRAM-overcommit failure mode means every benchmark run needs an RSS guard. Build single-target (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1100`, Release, Ninja+ccache), pin the llama.cpp commit, and keep a native-Linux dual-boot as a dormant fallback with explicit trigger conditions.

**Key risks:** (a) the model-choice fork itself — dense Qwen3-32B only fits at IQ4_XS/Q3-class quant with knife-edge headroom and slow (~25–35 tok/s estimated) benchmark loops, while MoE Qwen3-30B-A3B fits comfortably at Q4_K_S/IQ4_XS with context headroom and ~3× faster decode loops; research recommends the **MoE candidate as primary research vehicle** but both are cheap enough to smoke-test before deciding; (b) WSL2 platform traps (silent VRAM spill, no guest telemetry, free-VRAM APIs reporting 1.5–3 GiB high, driver↔ROCm coupling breaking the stack on routine Windows updates) — all cheaply mitigated if baked into Phase-1 gates and the harness; (c) RDNA3-specific kernel traps — wave-size assumptions smuggled from CUDA (wave32 vs wave64), rocWMMA being wave-cooperative rather than CUDA-tensor-core-like (documented −41% regressions when assumed-default), and rocBLAS Tensile coverage gaps — all neutralized by the measure-everything playground discipline the roadmap already mandates.

---

## Key Findings

### Recommended Stack

The stack is fully specified in `.planning/research/STACK.md`. Verdict: **production-supported WSL2 path exists right now** — no preview packages, no hacks.

**Core technologies:**

- **Windows host: Adrenalin Edition 26.2.2 "for WSL2"** — introduces production ROCDXG/librocdxg compute path; replaces deprecated roc4wsl packaging — HIGH confidence
- **Guest: ROCm 7.2.1 on Ubuntu 24.04 LTS (22.04 also supported)** via the librocdxg GitHub Quickstart — user-mode only, no dkms/kernel modules; `/dev/dxg` is the interface, `/dev/kfd` will never appear — HIGH
- **llama.cpp (ggml-org), pinned commit** with `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release` + Ninja + ccache — single-target gfx1100 build; do NOT set `HSA_OVERRIDE_GFX_VERSION`; `AMDGPU_TARGETS` is a legacy alias — HIGH
- **rocprofv3 / rocprofiler-sdk** — primary profiler; WSL2 gfx11-family tracing/counters landed upstream (rocm-systems PR #7016) — works but MEDIUM confidence until validated on this hardware (Phase-1 gate)
- **llama-bench / llama-perplexity / test-backend-ops** — benchmark + correctness harnesses, built from the tree; llama.cpp's startup VRAM breakdown becomes the primary VRAM accounting tool under WSL2
- **Windows-side telemetry** (Adrenalin overlay / Task Manager / HWiNFO) — mandatory substitute for the non-functional guest-side `rocm-smi`
- **rocWMMA** — A/B experiment flag only (`-DGGML_HIP_ROCWMMA_FATTN=ON`), never a default; rocBLAS/hipBLAS as measured comparison arms, not assumptions

**Model artifacts (first-party HuggingFace API sizes, HIGH confidence):**

| Candidate | Quant | Size | Fits 20 GB full-GPU? |
|---|---|---|---|
| Qwen3-32B (dense) | Q4_K_M | 19.76 GB | ❌ Never — do not download |
| Qwen3-32B (dense) | Q4_K_S | 18.77 GB | ⚠️ Marginal (~20.3 GB @ 4k f16 KV) |
| Qwen3-32B (dense) | IQ4_XS | 17.69 GB | ✅ Borderline — the working dense quant |
| Qwen3-32B (dense) | Q3_K_M | ~15.97 GB | ✅ Comfortable; quality step down |
| Qwen3-30B-A3B (MoE) | Q4_K_M | 18.56 GB | ✅ At 4k f16; q8_0 KV beyond |
| Qwen3-30B-A3B (MoE) | Q4_K_S | 17.46 GB | ✅ Good working quant |
| Qwen3-30B-A3B (MoE) | IQ4_XS | 16.38 GB | ✅✅ Best headroom (16–32k ctx) |
| Either | Q5_K_M | 21.7–23.2 GB | ❌ Drop from baseline matrix |

Q5_K_M appears in the original roadmap "if VRAM permits" — **it does not permit; remove it from the baseline plan.**

**Model recommendation status: PENDING-USER.** Research recommendation: **Qwen3-30B-A3B @ Q4_K_S/IQ4_XS as primary vehicle** (proper VRAM fit at Q4-class, context headroom for Phases 6–7 experiments, faster benchmark iterations), Qwen3-32B @ IQ4_XS/Q3_K_M as dense comparator only (purer bottleneck attribution, but knife-edge memory and slow loops). Downloading both families' working quants (~70 GB disk) is recommended so Milestone-2-style matrices can cover both architectures and the pick closes on data.

### Expected Features

From `.planning/research/FEATURES.md`. The feature frame: features exist to make claims *provable*.

**Must have (table stakes — TS1–TS12):**

- Locked baseline artifact set (pinned-commit binaries archived, never rebuilt) + environment/version fingerprint on every result row — TS1, TS2
- Reproduducible benchmark harness (warmup, ≥5 repeats, JSON/CSV) with **pp/tg split enforced everywhere; blended tok/s banned** — TS3, TS4
- Correctness gate: perplexity + KL-divergence vs baseline logits + golden-output regression tests; thresholds proposed (rel. ppl drift ≤0.5%, zero NaN/garbage) — TS5, TS6
- Per-config VRAM budget tracker + OOM boundary map; ctx×KV-cache tradeoff matrix; quant sweep paired with perplexity — TS7, TS8, TS9
- Profiling hooks with auto-detection of WSL2 limits and documented fallback — TS10
- Append-only result store + run-ID convention; written run protocol (thermal pairing, repeat counts, clock recording — record-don't-control on consumer RDNA3) — TS11, TS12

**Should have (differentiators):** autotuning loop with correctness-gated candidates (D1); shape/batch sweep matrices (D2); published result matrices committed to repo (D3); kernel-level hotspot attribution feeding "one bottleneck at a time" (D4); rocWMMA microbenchmarks gated on D4 evidence (D5); regression gate CSV-diff (D6); [MOE] expert-routing instrumentation + `--n-cpu-moe` sweeps (D7); speculative-decoding evaluation-only harness (D8).

**Anti-features (explicit no's):** custom engine or drifting hard fork; multi-GPU/tensor parallelism; serving stacks/chat UIs; native Windows HIP SDK; new quant formats; network/RPC inference; monitoring dashboards; overclock persistence; training tooling; early speculative-decoding *implementation*.

**MVP order (from FEATURES.md):** TS1/2/11/12 → TS3/4/5/6 → TS7/9/8 → TS10+D4 → D1+D3. Defer D5, D7, D8 until evidence demands them. Critical dependency chain: **TS1→TS3→TS5→TS10→D4** (the methodology spine).

### Architecture Approach

From `.planning/research/ARCHITECTURE.md`. Shape: **hub-and-spoke around an append-only, fingerprinted result store**, with two rules visible in the design itself — nothing integrates without passing the validation gate, and the stock baseline is a frozen artifact no component may modify.

**Major components:**

1. **C1 Baseline Builder** — env fingerprinting; pinned stock llama.cpp HIP builds; produces frozen reference binaries under `baseline/`
2. **C2 Benchmark Harness** — fixed workload profiles, warmup/repeats, pp/tg split, VRAM breakdown, fingerprints embedded per row
3. **C3 Profiler Bridge** — deliberately the *thinnest* component (rocprofv3 wrappers + parser → kernel→%runtime table) so it alone relocates to native Linux if counters fail; degrades gracefully to timing-only attribution
4. **C4 Kernel Playground** — per-op quartet (`ref_cpu.cpp` / `impl_gfx1100.hip` / `test_compare.cpp` / `bench_sweep.cpp`), **forbidden from including llama.cpp headers**; fixtures dumped from real GGUF tensors
5. **C5 Validation Suite** — op-level tolerances (per-bit-width constants à la `test-quantize-fns`) + model-level ppl/KL/golden streams
6. **C6 Runtime Integration Layer** — vendored pinned upstream + quilt patch series + `GGML_GFX1100_CUSTOM=ON/OFF` flags; **two binaries from the same tree**; first patch is empty flag plumbing proving bit-identical OFF behavior
7. **C7 Quantization Pipeline + Result Store** — imatrix/calibration corpora; cross-cutting immutable store keyed by run-ID

**Key patterns:** Reference-Implementation Ladder (every op, always); Patch Overlay + Flag-Gated Selection (anti-fork — rejected alternatives verified: out-of-tree backend plugins have wrong granularity, permanent forks have proven maintenance gravity per ik_llama.cpp); Fingerprinted Append-Only Store; Shape-Keyed Autotune Registry (only after ≥2 kernels need it). **Prefill(M≫1)/decode(M≈1) split is a first-class dimension in every record from day one.**

**Build-order insight for the roadmapper:** the only serial choke points are C1 and the C5 model-tier suite. Playground scaffolding (C4 core) is **model-independent and can start during the ~17 GB weight download**, materially shortening the critical path.

### Critical Pitfalls

Top pitfalls from `.planning/research/PITFALLS.md` (each critical item corroborated by ≥2 independent sources or AMD docs):

1. **Silent VRAM overcommit under WSL2** — allocation failures spill to system RAM instead of erroring; throughput collapses 5–10× and the WSL VM eventually dies while tokens still flow. *Prevention:* RSS guard in the harness (process RSS ≈ model size max), logged every run; conservative `-c` sizing; treat any RAM-climbing run as failed.
2. **No GPU telemetry inside WSL2** — `rocm-smi` AND `amd-smi` are both unavailable (UKI limitation; amd-smi not implemented in ROCDXG). *Prevention:* stand up Windows-side sensor capture (HWiNFO/Adrenalin) before the benchmark suite; the original roadmap's `rocm-smi.txt` artifact cannot be produced as written — document the substitution.
3. **Free-VRAM reporting lies twice** — `hipMemGetInfo` over-reports (excludes host desktop allocations) AND the DXG path reports ~1.5–3 GB less than native Windows. On a card with <1.5–2 GB headroom, automated decisions from reported free VRAM will be wrong. *Prevention:* never allocate from reported numbers; probe empirically; maintain a measured VRAM ledger from llama.cpp buffer logs.
4. **Profiling under WSL2 is brand-new and counter-incomplete** — the whole profile-before-optimize methodology depends on it. *Prevention:* explicit Phase-1 exit criterion (rocprofv3 must produce per-kernel timings + ≥1 hardware counter on a real workload); pre-agreed fallback ladder: (a) rocprofv3/DXG → (b) llama.cpp built-in timing + manual region timers → (c) native-Linux contingency.
5. **Driver↔ROCm coupling fragility** — a routine Windows Adrenalin auto-update silently invalidates the matched pair and breaks the stack mid-project. *Prevention:* pause driver updates; pin versions in `benchmarks/environment/`; `wsl --export` snapshot immediately after Phase 1 succeeds; re-run env gate after any forced change.
6. **RDNA3 kernel traps** (cluster): wave-size assumptions from CUDA (template on `warpSize`, never literal 32/64; benchmark wave32 vs wave64 per kernel); WMMA is wave-cooperative, not CUDA-tensor-core-like (rocWMMA FA showed −41% long-context prefill regression on gfx1151; MMA often loses to VALU at small M); rocBLAS Tensile coverage gaps (missing-arch symptoms are warm-up crashes, not graceful fallback); upstream perf regressions landing between builds (pin ONE commit + archived binary); dense-Q4_K_M-on-20GB budget trap (OOM arrives at the first long prompt, not model load — set `-c` explicitly, always); [MOE] offload tuning is **non-monotonic** (sweep `--n-cpu-moe`, plot it, size `.wslconfig` RAM); benchmark sins (single runs swing 12–18% thermally; pp/tg must be paired within one thermal window).

---

## Implications for Roadmap

Based on combined research, suggested phase structure — **8 phases that merge the original 18-phase roadmap and its 7 milestones** (per user requirement) with what the research changed. Original-phase references in brackets preserve traceability.

### Phase 1: Environment Validation & Model Selection Gate
**Rationale:** Everything gates on a working gfx1100 toolchain; WSL2 feasibility is itself the first exit criterion, and the PENDING-USER model choice must close on measured data before any downstream phase plans concretely.
**Delivers:** validated ROCDXG stack (rocminfo sees gfx1100 → llama-bench full-GPU smoke → rocprofv3 kernel timeline → measured free-VRAM actuals, in that order); pinned llama.cpp commit + frozen baseline binaries; pinned-version doc + `wsl --export` snapshot; Windows-side telemetry pipeline producing sensor CSV alongside a bench run; smoke-load tests of BOTH model candidates at target context; RSS-guarded run logging.
**Addresses:** TS1, TS2 foundations; closes the model-selection gap.
**Avoids:** Pitfalls 1, 2, 3, 4, 5, 9, 10 — all of which are Phase-1-baked mitigations per PITFALLS.md.
[Original Phases 1–2]

### Phase 2: Measurement Infrastructure
**Rationale:** Nothing else produces credible numbers until the harness, store, and gates exist; pure scripting, low effort, high leverage.
**Delivers:** llama-bench wrapper (fixed profiles, warmup, ≥5 reps, pp/tg split, fingerprints embedded); append-only `results/<date>_<run-id>/` store; run-protocol doc (thermal pairing, repeat policy); correctness gate suite (ppl/KL/golden) proven against the stock baseline.
**Addresses:** TS3, TS4, TS5, TS6, TS11, TS12.
**Avoids:** Pitfall 12 entirely (harness-enforced, not discipline-enforced).
[Original Phase 1.4 + methodology infra from Phase 14/15 pulled early]

### Phase 3: VRAM Ledger & Quant Frontier
**Rationale:** With <2 GB true headroom, memory budgets arbitrate every later choice (context length, KV dtype, quant level, even model choice).
**Delivers:** measured weights/KV/compute/overhead ledger per (quant × ctx × KV-dtype); OOM boundary map; quant-sweep results paired with perplexity (speed-vs-quality frontier); ctx×KV tradeoff matrix.
**Addresses:** TS7, TS8, TS9.
**Avoids:** Pitfalls 1, 3, 10 (ledger replaces lying APIs).
[Original Phase 1.3 + Phase 9 groundwork + Phase 11 partial]

### Phase 4: Profiling & Bottleneck Ranking
**Rationale:** Profile-before-optimize is the methodology's engine; the bottleneck table names THE next target and provides go/no-go data for later rocWMMA work.
**Delivers:** rocprofv3 captures across workload profiles (short/long prompt × short/long generation); kernel→%runtime→bound-type table; dispatch audit showing which ops route to rocBLAS vs ggml MMQ/mmv kernels; explicit verdict on WSL2 counter usability (triggers native-Linux contingency if failed).
**Addresses:** TS10, D4.
**Avoids:** Pitfalls 4, 8 (fallback ladder pre-agreed; rocBLAS coverage checked, not assumed).
**Research flag:** see below — profiler reliability under WSL2 is the project's softest technical spot.
[Original Phase 3]

### Phase 5: Kernel Playground & First Custom GEMM
**Rationale:** The highest-value optimization stage per both the original roadmap and the stack research (quantized matmul dominates); the playground quartet makes miscompiles debuggable in minutes instead of inside 17-GB-model runs. Scaffolding is model-independent — start it during weight downloads.
**Delivers:** quartet layout for dequant/matmul ops; fused Q4_K dequant+matmul experiments (tile sizes, LDS staging, vector widths, wave-level reductions); **separate prefill/decode kernel paths selected by M-regime from day one**; rocBLAS comparison arm per shape; first custom kernel beating stock in a microbenchmark with archived diff evidence across shapes.
**Addresses:** core of the Core Value proposition; D5 precursor measurements.
**Avoids:** Pitfalls 6, 8; anti-patterns 1 (developing in-runtime) and 4 (one universal kernel).
[Original Phases 4–6]

### Phase 6: Decode Path, Attention & KV Cache Strategy
**Rationale:** After GEMM, decode (M≈1) and attention dominate; KV strategy converts kernel wins into usable context on a 20 GB card. These three are coupled (attention shape determines KV behavior).
**Delivers:** M≈1 vec-kernel optimization; QKᵀ/softmax/PV experiments with fa-on/off × batch-size grids measured on this card; optional rocWMMA path strictly behind a flag with MMA-vs-VALU per-shape verdict; KV quant sweep (q8_0/q4_0) with ppl impact gated; max-context envelope established end-to-end (largest planned prompt tested, no spill).
**Addresses:** D5 (gated), TS8 completion.
**Avoids:** Pitfalls 6, 7, 10 (the classic "loads fine at 4k, dies at 16k" trap).
[Original Phases 7–9]

### Phase 7: Runtime Integration & Autotuning
**Rationale:** Independent winners become a coherent optimized runtime only through disciplined integration; the patch-overlay pattern keeps delta-from-stock reviewable and bisectable forever.
**Delivers:** empty-flag plumbing proven bit-identical, then each winner as one additive patch behind `GGML_GFX1100_CUSTOM`-style flags; paired baseline/optimized binaries from the same tree; shape-keyed autotune registry (`autotune/gfx1100.json`); end-to-end before/after benchmark matrix; upstream-rebase ritual documented.
**Addresses:** D1, D6, D3 (first publishable matrices).
**Avoids:** anti-patterns 2 (hard fork) and 5 (casual re-baselining); Pitfall 9.
[Original Phases 12–13 + 14 enforcement]

### Phase 8: Model Specialization, Full Validation & Publication
**Rationale:** Generic kernels first, model-specific wins second — specialization only pays once profiles identify what actually dominates Qwen runtime; publication is the project's terminal deliverable and requires the complete matrix, not the best number.
**Delivers:** [MOE] expert-routing instrumentation + plotted `--n-cpu-moe` sweeps (or [DENSE] graph specialization per tensor table); imatrix/calibration experiments; final numerical-validation pass (max-abs/rel/cosine per op + full-model same-seed diffs); Profiles A–D result matrix with temps/power columns from Windows telemetry; packaged repo with exact build commands, raw data, kernel source, known limitations.
**Addresses:** D2, D3, D7 (conditional), D8 evaluation-only (deferred unless evidence).
**Avoids:** Pitfalls 11, 12; UX pitfalls (publishing best-number-only).
[Original Phases 10, 11, 15–17; Phase 18 stretch goals remain explicitly out of scope]

### Phase Ordering Rationale

- **Dependency-honoring:** C1 → (∥ C2+store, C4 scaffolding) → C5-model-tier → C3 → first-kernel → C6 loop. The research's critical-path insight — playground scaffolding overlaps the model download — argues against serializing "environment → benchmarks → profiling → kernels."
- **Evidence-first grouping:** Phases 2–4 produce *no* optimizations but make every subsequent claim provable; FEATURES.md's spine (TS1→TS3→TS5→TS10→D4) maps directly onto Phases 1–4.
- **Pitfall-driven placement:** every WSL2 platform pitfall is neutralized in Phases 1–3 where it's cheapest; every RDNA3 kernel pitfall lands exactly where the kernel work happens (Phases 5–6).
- **The model-choice fork stays open through Phase 1 only** — both candidates' working quants get downloaded (~70 GB disk) and smoke-tested, then the decision closes on measured load/throughput/context data rather than desk analysis.

### Research Flags

Phases likely needing deeper research during planning (`/gsd-plan-phase --research-phase`):

- **Phase 1:** ROCm-on-WSL install specifics shift across releases (librocdxg quickstart is the live source of truth; package names/set moved across 7.x) — verify current steps at planning time; also design the profiler-validation protocol precisely.
- **Phase 4:** rocprofv3-under-WSL2 practical reliability is MEDIUM confidence (newly shipped via PR #7016); plan the fallback ladder and its decision criteria explicitly.
- **Phase 6:** rocWMMA gfx1100 coverage/performance on this specific card is unverified (support matrix says gfx1100/wave32 ✓, but shipped regressions exist on sibling targets) — needs an empirical mini-matrix; FlashAttention on/off behavior on RDNA3 is likewise a measured variable, not a default.
- **Phase 8 (if MoE chosen):** `--n-cpu-moe`/`-ot exps=CPU` semantics, non-monotonic offload behavior, and `.wslconfig` RAM sizing interact — worth targeted research before sweeping.

Phases with standard patterns (skip deep research):

- **Phase 2:** llama-bench wrapping and result-store conventions are well-documented; just verify flag surfaces at the pinned commit.
- **Phase 5:** the playground quartet and ggml's per-op dispatch structure (`ggml_cuda_mul_mat` selection points) were verified against llama.cpp source — established pattern.
- **Phase 7:** quilt-patch-overlay integration has multiple documented precedents (mesh-llam, ik_llama.cpp history) and the hook locations are known.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | Official AMD ROCm-on-WSL docs + compatibility matrices; first-party HF API GGUF sizes; llama.cpp build docs. Individual MEDIUM items: rocprofv3-under-WSL2 (newly shipped), rocWMMA perf on this card |
| Features | **MEDIUM-HIGH** | Workflow/measurement features HIGH (tooling verified in-tree); platform-specific feature values (exact bench flags, KL-mode availability) tagged for re-verification at the pinned commit |
| Architecture | **HIGH** | Cross-checked against llama.cpp source (dispatch points, test suites), ROCm tooling docs, and ≥2 independent kernel-project precedents |
| Pitfalls | **HIGH** | Each critical pitfall corroborated by ≥2 independent sources or AMD official docs; community numbers used for magnitudes only |

**Overall confidence: HIGH** — unusual for greenfield research, reflecting that this domain has extensive public documentation, active upstream issue trackers, and deterministic VRAM arithmetic. The honest residual risk is concentrated in one place: *how well brand-new WSL2 profiling software behaves on this specific machine*, which is a Phase-1 measurement question, not a planning unknown.

### Gaps to Address

- **Final model pick (PENDING-USER)** — close in Phase 1 via smoke-load tests of both candidates at target context; research recommends MoE-primary but the decision deserves measured data on this card.
- **PROJECT.md factual correction needed** — "~16–17 GB for Q4_K_M" is wrong: dense Q4_K_M = 19.76 GB (doesn't fit); MoE Q4_K_M = 18.56 GB (fits, short ctx). Fix during requirements definition.
- **rocprofv3-under-WSL2 reliability** — validate day 1; fallback ladder pre-agreed so a failure costs hours, not the methodology.
- **Actual free-VRAM deficit under DXG on THIS machine** (~1.5–3 GB reported elsewhere) — measure in Phase 1 before finalizing quant choices.
- **rocWMMA gfx1100 coverage + FA behavior on RDNA3** — empirical A/B matrix in Phase 6; never assume MMA wins.
- **Dense decode-rate estimate (~25–35 tok/s)** — bandwidth-arithmetic estimate only; measure before committing to dense.
- **llama.cpp tool flag surfaces at the pinned commit** (`llama-bench -r/-p/-n/-o`, perplexity KL mode) — verify once, record in harness docs.
- **Exact ROCm guest package set for WSL2** — follow the live librocdxg quickstart at execution; don't memorize package lists.

---

## Sources

Aggregated from the four research files; full per-claim citations live in each file's Sources section.

### Primary (HIGH confidence)
- AMD ROCm documentation: Radeon-WSL how-to (ROCDXG, Adrenalin 26.2.2 + ROCm 7.2.1), WSL compatibility matrices (Ubuntu 22.04/24.04, kernel 5.15), WSL limitations (rocm-smi unsupported/UKI), rocprofv3 usage, HIP porting guide (warpSize rules), rocWMMA API guide (gfx1100/wave32), GPUOpen RDNA3-WMMA articles
- llama.cpp repository: `docs/build.md` (gfx1100 example, HIP flags), `ggml/src/ggml-hip` / `ggml-cuda` sources (op-dispatch points), `tests/test-backend-ops.cpp`, `tests/test-quantize-fns.cpp`, `tools/perplexity` (KL-divergence PRs #5076/#6936), backend dynamic-loading PR #10469
- HuggingFace API (queried 2026-08-21): exact GGUF blob sizes for Qwen/Qwen3-32B-GGUF, Qwen/Qwen3-30B-A3B-GGUF, unsloth + bartowski variants; Qwen3 `config.json` dims for KV/token arithmetic
- ROCm/librocdxg GitHub quickstart; rocprofiler-sdk WSL2 `/dev/dxg` support (rocm-systems PR #7016); CUTLASS Profiler docs; ROCm hipBench

### Secondary (MEDIUM confidence)
- llama.cpp issue trackers: #20934 (ROCm-vs-Vulkan decode, wave-size discrepancy), #10439 (FA slower on ROCm at bs>1), #24437/#13110 (rocWMMA FATTN default-off, −41% gfx1151, compile failures), #19580 (ROCm 6.4.4 rocWMMA breakage), #20839 (TensileLibrary arch gap → warmup crash), #23999 (DXG free-VRAM anomaly), #22583 (WSL2 VRAM overcommit → RAM exhaustion), #11495/#11758 (ggml wave-size assumptions, deprecated macro), #20647 (upstream perf regression)
- ROCm issues: #6389 (hipMemGetInfo over-report), #4459 (driver-update breakage); librocdxg#6 (amd-smi unimplemented)
- Precedents: ik_llama.cpp (fork-cost evidence), mesh-llam fork-overlay practice, KernelBench correctness-grading conventions, Modular verify.py tolerance defaults, vLLM kernel test patterns

### Tertiary (LOW confidence — magnitudes only, re-measure in-project)
- Community benchmark methodology (CraftRigs thermal-swing 12–18%, repetition guidance); aiweekly DXG free-VRAM deficit report (~3 GB, single source); Qwen3-32B VRAM footprint aggregators

---
*Research completed: 2026-08-21*
*Ready for roadmap: yes*
