# Roadmap: Qwen3.8-27B on RX 7900 XT — gfx1100 Inference Optimization

## Overview

Six phases take this project from a bare Windows/WSL2 machine to a published, independently benchmarked gfx1100-tuned inference path for the locked `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` artifact (15.31 GB, sha256 `53adc4bb…`). Per owner mandate, this roadmap **merges** the original 18-phase methodology plan (`.planning/reference/ROADMAP-original.md`) and its first-seven-milestones sequence into the GSD phase structure — nothing was discarded: deferred original phases (attention/KV tuning, autotuning, imatrix/calibration, stretch goals) land in the v2 list (see REQUIREMENTS.md), and every retained element is traceable in the Merge Map below. The original execution order — ROCm setup → llama.cpp baseline → benchmark suite → profiler → top bottleneck → standalone HIP kernel → numerical validation → microbenchmark → integration → end-to-end benchmark — is preserved exactly.

Phases 1–4 deliberately produce **zero optimizations**: they build measurement and validation infrastructure so every later claim, including failures, is provable. First optimization code appears in Phase 5 against the single bottleneck Phase 3 ranks #1 (expected: Gated DeltaNet scan or fused IQ4_XS dequant+matmul — the hybrid qwen35 architecture's prime custom-kernel frontier; its tiny ~64 KiB/token KV cache is why the original Phase 9 is v2-deferred).

## Methodology Rules (binding for every phase)

Inherited verbatim from the original roadmap; planners and executors apply them automatically:

1. **Benchmark before optimizing.**
2. **One optimization at a time.**
3. **Keep a stock baseline forever** — never destroy it, never casually rebuild it.
4. **Test prefill (M≫1) and decode (M≈1) separately** — blended tok/s banned everywhere.
5. **Measure VRAM as carefully as throughput.**
6. **Do not assume CUDA optimization ideas map directly to RDNA3** (wave32/wave64, rocWMMA semantics, rocBLAS coverage gaps).
7. Prefer fused memory-efficient kernels **when they actually win**.
8. **Keep correctness tests next to every kernel.**
9. **Record compiler/ROCm/driver versions** with every result.
10. **Publish failed experiments too.**
11. **MANDATORY TIMEOUTS ON EVERY BASH COMMAND:** Every bash command and harness subprocess call MUST specify an explicit, bounded timeout to prevent hangs.

## Merge Map: original plan → GSD phases

| Original element | GSD destination |
|---|---|
| Orig Phase 1.1–1.2 (verify gfx1100, build reference llama.cpp) · Milestone 1 | **Phase 1** |
| Orig Phase 1.3–1.4 (baseline model + metrics) · Milestone 2 (reproducible bench scripts) | **Phase 2** |
| Orig Phase 2 (understand existing GPU path) | **Phase 3** (source study feeds kernel→op attribution; spillover into Phase 5 kernel authoring) |
| Orig Phase 3 (profile before optimizing) · Milestone 3 (top-3 kernels by wall time) | **Phase 3** |
| Orig Phase 14 (numerical validation) | Cross-cutting: gates armed in **Phase 3**, enforced per-kernel in **Phases 4–6** |
| Orig Phase 4 (HIP kernel playground) | **Phase 4** |
| Orig Phases 5–7 (dequant, quantized GEMM, decode-specific paths) · Milestones 4–5 | **Phase 5** (first winning kernel attacks profiled #1; prefill/decode split as design goal from day one) |
| Orig Phase 10 (Qwen-specific specialization) | **Phase 5** (light graph-aware report landed as `docs/QWEN-GRAPH.md`); deep tensor dump → v2 |
| Orig Phase 12 (runtime integration behind flags) · Milestone 6 | **Phase 6** |
| Orig Phases 15–16 (e2e suite, compare against real baselines) · Milestone 7 | **Phase 2** (harness/profiles built early) + **Phase 6** (final matrix) |
| Orig Phase 17 (package & publish) | **Phase 6** |
| Orig Phases 8–9 (attention experiments, KV cache) | v2 deferred (hybrid arch: KV ≈64 KiB/token est.) |
| Orig Phase 11 (imatrix/calibration quant experiments) | v2 deferred (artifact pre-imatrix-quantized) |
| Orig Phase 13 (autotuning) | v2 deferred |
| Orig Phase 18 (stretch: sampler, MTP spec-decode, multi-GPU, GGUF format, persistent kernels) | v2 / Out of Scope |

**Execution order:** numeric order 1 → 2 → 3 → 4 → 5 → 6, with one sanctioned overlap: Phase 4 scaffolding is model-independent and may run concurrently with Phases 2–3 (research critical-path finding — start it during the model-weight download window).

## Phases

- [x] **Phase 1: Environment Validation & Stock Baseline** - Validated WSL2 ROCm/HIP stack, pinned stock llama.cpp gfx1100 build, locked IQ4_XS artifact running fully on GPU (completed 2026-08-23)
- [x] **Phase 2: Benchmark Harness & Baseline Matrix** - Fingerprinted reproducible pp/tg-split harness with RSS-guarded VRAM ledger publishing the stock baseline matrix (completed 2026-08-23)
- [x] **Phase 3: Correctness Gates & Bottleneck Profiling** - Two-tier quality gates armed; rocprofv3/eval profiler validated over dxg under WSL2; ranked bottleneck table names Target #1 (completed 2026-08-24)
- [x] **Phase 4: Kernel Playground Scaffold** - Standalone CPU-reference → HIP → numerical-compare → microbenchmark pipeline operating outside llama.cpp (completed 2026-08-25)
- [x] **Phase 5: First Custom Kernel (Bottleneck Attack)** - GEMV 1.26–2.13× / GEMM 1.7–7.5× vs stock HIP (cosine 1.0), 30/32 wins, WMMA `v_wmma` in disasm, provisional quilt patch (completed 2026-08-25)
- [x] **Phase 6: Integration, Full Validation & Publication** - Winners behind ON/OFF flags via quilt patches; stock baseline intact forever; complete before/after matrix published (completed 2026-08-25)
- [x] **Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization — RE-SCOPED 2026-08-28, REPLANNED 2026-08-30** - Fuse Q8_1 + RDNA3 WMMA to beat **real production stock llama.cpp by ≥10% end-to-end** in `llama-bench`, **Windows-native (≤2 langs, no Python/JS servers, `build_windows.bat`)**, and **10× averaged (15× LLM QA)** — ways-to-achieve in `output/deep-research/phase7-3must-haves-exhaustive.md` + `docs/PUBLICATION.md §8`. (Status: 3 closure plans — 07-01 REQ-WIN-07 Windows ≤2 langs, 07-02 REQ-PERF-07 ≥1.10×, 07-03 REQ-STAT-07 N≥10/15; bare-metal N=10 committed `d414c552`/`6e46d2e`: 87.8µs DP4A 6.24× PASS, gemv 0.976 FAIL, gemm M1024 1.08 avg/1.89 peak, llama-bench n=10 512/1024/2048/4096 pp 1.079/0.996/1.003/0.978 all FAIL, tg 0.993 — *not yet* the required +10%; HIP SDK install pending.)
- [ ] **Phase 8: Refactor to Windows Native (now landing the Phase 7 must-have #1)** - Strip bloat to pure C++/HIP, prune kernels to IQ4_XS winners, provide foolproof build_windows.bat for gfx1100 via HIP SDK, and serve llama-server.exe at localhost:8000 — **Phase 7 now declares this as mandatory; Phase 8 is the execution phase that closes REQ-WIN-07.**

## Phase Details

### Phase 1: Environment Validation & Stock Baseline

**Mode**: standard (horizontal layers)
**Goal**: A validated WSL2 ROCm/HIP toolchain drives stock llama.cpp with the locked IQ4_XS artifact fully resident on the gfx1100 GPU, establishing the permanent, never-regressing reference baseline.
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04
**Success Criteria** (what must be TRUE):

  1. Inside WSL2, `rocminfo` enumerates the RX 7900 XT as `gfx1100` and a minimal HIP smoke program executes on that device successfully (ENV-01).
  2. From one pinned llama.cpp commit, `llama-cli`, `llama-bench`, `llama-perplexity`, and `test-backend-ops` all build targeting gfx1100 and run (ENV-02).
  3. The sha256-verified IQ4_XS GGUF generates coherent tokens with all tensors on GPU — startup logs show zero CPU fallback across both Gated DeltaNet and gated full-attention layer paths (ENV-03).
  4. `models/README.md` and `benchmarks/environment/` record full provenance: artifact repo/commit/imatrix/quant metadata, sha256 verification result, ROCm/driver/toolchain versions (ENV-04).

**Key deliverables**: pinned-commit stock binaries archived under `baseline/`; environment fingerprint files; `wsl --export` snapshot taken immediately after validation passes; measured free-VRAM actuals vs DXG-reported numbers.
**Original-roadmap lineage**: orig Phases 1.1–1.2; Milestone 1 ("stock llama.cpp running on gfx1100").
**WSL2 risk notes**: This phase IS the platform kill-gate — if `/dev/dxg` passthrough or HIP runtime fails, the pre-agreed native-Linux contingency triggers *before any other work*. Pin driver↔ROCm pairing (Adrenalin 26.2.2-for-WSL2 + ROCm 7.2.1 guest); pause Windows Adrenalin auto-updates; snapshot WSL after success. Recommended (non-binding): an early rocprofv3 feasibility probe here informs contingency planning early — the binding PROF-01 gate itself lives in Phase 3. Never trust reported free VRAM (~1.5–3 GB deficit documented on sibling setups): probe empirically.
**Plans**: TBD (indicative: 3 plans — stack install/validation; pinned build + archived binaries; artifact download/sha256/provenance + snapshot)
**Parallelization**: Stack installation must complete first; then artifact download/provenance and pinned-build plans can run concurrently as subagents (download is I/O-bound, build is CPU-bound). Snapshot step is serial-last.

### Phase 2: Benchmark Harness & Baseline Matrix

**Mode**: standard (horizontal layers)
**Goal**: Reproducible, fingerprinted measurement infrastructure produces the published stock baseline matrix, so every future claim has a fixed reference to beat or regress against.
**Depends on**: Phase 1
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04
**Success Criteria** (what must be TRUE):

  1. Re-running the harness on an unchanged system reproduces results within stated variance, emitting machine-readable rows with enforced pp/tg split, warmup, and ≥3 repeats over fixed workload profiles (BENCH-01).
  2. Any result row traces to its exact environment: llama.cpp commit, ROCm/driver versions, GGUF sha256, plus clock/temperature readings captured from Windows-side telemetry (BENCH-02).
  3. Every run carries a VRAM ledger plus process-RSS guard output; a synthetic RAM-climbing run (simulated overcommit) is detected and marked FAILED rather than reported as a valid number (BENCH-03).
  4. The published baseline matrix spans pp/tg × context {4k, 8k, 16k, 32k} × flash-attn {on, off} for the locked IQ4_XS artifact, PLUS a stock-Vulkan comparator arm at its own pinned commit (GDN coverage verified there before comparison) — every performance claim names its backend (BENCH-04). The 32k tier is gated by an empirical free-VRAM pre-flight; under WSL2's ~1.5–3 GB DXG deficit it carries an expected-FAIL path (or a documented cap at 24k with rationale).

**Key deliverables**: `benchmarks/` harness scripts; append-only fingerprinted result store (`results/<date>_<run-id>/`); written run protocol (thermal pairing within one window, repeat policy, record-don't-control clocks); crash-resilient result journal with fail-fast OOM policy (allocation caps, no retry loops — repeated GPU-OOM under WSL2 can hard-crash the host via Hyper-V per microsoft/WSL#40732 pattern; synthetic overcommit test runs supervised only); published stock baseline matrix.
**Original-roadmap lineage**: orig Phase 1.4 baseline metrics pulled forward as infrastructure; Milestone 2; feeds orig Phases 15–16 profiles.
**WSL2 risk notes**: Guest-side `rocm-smi`/`amd-smi` do not work under ROCDXG — Windows-side telemetry (HWiNFO/Adrenalin overlay → sensor CSV) is mandatory from day one; document the substitution from the original roadmap's `rocm-smi.txt` artifact. RSS guard exists specifically to defeat the silent VRAM-overcommit failure mode (spill to system RAM = 5–10× throughput collapse while tokens still flow). Set `-c` explicitly, always — OOM arrives at first long prompt, not model load. Fingerprint librocdxg#60 (ROCr busy-spin ~2 cores per GPU context under WSL2) during calibration; consider taskset isolation for decode-latency measurements. Verify `.wslconfig` memory settings don't clamp the ROCm pool (ROCm#6022 class bug).
**Plans**: 5 plans (executed and verified) — 02-01 (wrapper & corpus), 02-02 (fingerprint & telemetry), 02-03 (store, guard & preflight), 02-04 (matrix orchestration & calibration), 02-05 (Vulkan comparator arm & coverage gate)
**Parallelization**: Harness wrapper, fingerprinting/telemetry pipeline, and RSS-guard ledger are three independent plans runnable concurrently via subagents; baseline-matrix execution waits for all three.

### Phase 3: Correctness Gates & Bottleneck Profiling

**Mode**: standard (horizontal layers)
**Goal**: Two-tier quality gates stand guard over all future claims, and profiling of real workloads names the single highest-value optimization target before any optimization code is written.
**Depends on**: Phase 2
**Requirements**: QUAL-01, QUAL-02, PROF-01, PROF-02
**Success Criteria** (what must be TRUE):

  1. `test-backend-ops` runs green on the stock build, and the workflow enforces that any red run blocks acceptance of a performance claim (QUAL-01).
  2. `llama-perplexity` on wikitext-2 lands within ±1% of the published 7.1583±0.25, and greedy-decode golden outputs for fixed prompts match recorded baselines within tolerance (QUAL-02).
  3. Kernel-level attribution strategy resolved: rocprofv3-over-dxg feasibility probed and outcome recorded — per-kernel timings via llama.cpp op-timers/region timers are the PLANNED baseline (counter-less attribution pre-authorized); if rocprofv3/DXG works on gfx1100, treat as bonus; if attribution proves insufficient, the native-Linux profiling-session decision is recorded with rationale (PROF-01).
  4. A ranked bottleneck table maps top kernels to ggml ops (% runtime, bound type) across four workload shapes (short/long prompt × short/long generation) and explicitly names optimization target #1 (PROF-02).

**Key deliverables**: wired-in correctness gate suite; rocprofv3 wrapper + parser; kernel→%runtime→bound-type table; go/no-go verdict on WSL2 profiling depth; source-study notes mapping ggml dispatch points (orig Phase 2 folded here).
**Original-roadmap lineage**: orig Phase 2 (understand GPU path) + Phase 3 (profile first); Milestone 3; orig Phase 14 enforcement begins here.
**WSL2 risk notes**: PROF-01 is the project's softest technical spot — rocprofiler-sdk DXG support is an UNMERGED PR scoped to RDNA 3.5 iGPUs (rocm-systems PR #7016), and librocdxg officially lists profiling as unsupported; plan around llama.cpp timers from day one and treat any working rocprofv3 capture as upside. Ladder: (a) rocprofv3/DXG [upside] → (b) llama.cpp timing + region timers [planned baseline] → (c) native-Linux contingency session [sanctioned escalation for deep attribution]. Do not assume attention dominates — measure; low-impact ops are explicitly out of scope for optimization.
**Plans**: TBD (indicative: 3–4 plans — op-level gate wiring; model-level gate suite; profiler bridge + captures; bottleneck-table analysis)
**Parallelization**: QUAL op-gate, QUAL model-gate, and profiler-bridge plans are mutually independent and fully concurrent; table analysis depends on profiler capture output only.

### Phase 4: Kernel Playground Scaffold

**Mode**: standard (horizontal layers)
**Goal**: A standalone HIP kernel development pipeline operates end-to-end outside llama.cpp — CPU reference → HIP implementation → numerical comparison → microbenchmark — making miscompiles debuggable in minutes instead of inside 15-GB-model runs.
**Depends on**: Phase 1 (toolchain only — sanctioned overlap: may run concurrently with Phases 2–3)
**Requirements**: KERN-01
**Success Criteria** (what must be TRUE):

  1. The playground directory builds standalone HIP executables for gfx1100 with zero llama.cpp headers included, using tensor fixtures dumped from real GGUF weights (KERN-01).
  2. A candidate op traverses the full quartet pipeline (`ref_cpu.cpp` → `impl_gfx1100.hip` → `test_compare.cpp` → `bench_sweep.cpp`) producing recorded error metrics (max-abs / mean-abs / relative / cosine) and timing tables (KERN-01).
  3. A deliberately broken implementation is caught red by the comparison stage, and its corrected version passes green — proving the pipeline discriminates, not just runs (KERN-01).

**Key deliverables**: `kernels/` scaffold (common/dequant/matmul/benchmarks); fixture dumper from GGUF tensors; per-op quartet template; worked example `dequant_iq4_xs` op traversing the full pipeline with recorded error tables.
**Original-roadmap lineage**: orig Phase 4 (kernel playground), including its debugging rationale.
**RDNA3 risk notes**: Template kernels on `warpSize` — never literal 32/64; benchmark wave32 vs wave64 per kernel later. No CUDA assumptions (rule 6). Owner locks (2026-08-25): dequant-only demo, vendored `block_iq4_xs.h` copy, reuse `benchmarks/lib/store.py`, tight gate 1e-5/1e-6/0.99999 +10× discrimination, templated `WARP_SIZE` — see `04-CONTEXT.md` D4-00-1..5.
**Plans**: 3 plans (executed and verified) — 04-01 (scaffold + standalone gfx1100 build, vendored `block_iq4_xs.h` + `WARP_SIZE` templating), 04-02 (GGUF + synthetic fixture dumper), 04-03 (demo `dequant_iq4_xs` walkthrough with negative test) — see `.planning/phases/04-kernel-playground-scaffold/`
**Parallelization**: 04-01 ∥ 04-02, 04-03 depends on both stubs. Whole phase is model-independent and may run concurrently with Phases 2–3 (sanctioned overlap, now complete; toolchain proven `bb4caa75`).

### Phase 5: First Custom Kernel (Bottleneck Attack)

**Mode**: standard (horizontal layers)
**Goal**: A custom gfx1100 kernel attacking the #1 profiled bottleneck is numerically correct against reference, beats the stock llama.cpp kernel in shape-swept microbenchmarks, and the win survives end-to-end contact with the full model.
**Depends on**: Phase 3 (bottleneck ranking names the target), Phase 4 (pipeline exists), Phase 2 (microbenchmark/A-B conventions)
**Requirements**: KERN-02, KERN-03
**Success Criteria** (what must be TRUE):

  1. The implemented kernel targets entry #1 of the Phase 3 bottleneck table (expected candidates: Gated DeltaNet scan or fused IQ4_XS dequant+matmul) and passes numerical comparison against the CPU reference within agreed tolerances (KERN-02).
  2. In head-to-head microbenchmarks across the relevant shape sweep — prefill (M≫1) and decode (M≈1) measured and reported separately — the custom kernel beats the stock llama.cpp kernel on HIP, with the stock-Vulkan comparator reported alongside (a win over HIP that loses to plain Vulkan is recorded as such) (KERN-03).

**Pre-kernel gate**: before authoring any kernel, evaluate graph-capture/fusion (GGML_HIP_GRAPHS=ON probe; dispatch-overhead hypothesis per llama.cpp#20292 — 99% launch overhead on RDNA4) so we attack the right layer. First implementation hypothesis on record: community VGPR-spill fix (`__launch_bounds__`, +24% GDN decode on XTX, buun PR #20).

  3. An end-to-end A/B through the real model (provisional wiring acceptable) shows the speedup survives the full runtime with QUAL-01/QUAL-02 gates green (KERN-03).
  4. Lost shapes and regressed variants are recorded in the results store beside the wins (rule 10: publish failed experiments).

**Key deliverables**: winning kernel source + correctness tests co-located; archived microbenchmark diffs across shapes; e2e A/B evidence; explicit prefill-path and decode-path variants (separate paths are a design goal, not an afterthought).
**Original-roadmap lineage**: orig Phases 5–7 (fused dequant+matmul, quantized GEMM, decode-specific split); Milestones 4–5; orig Phase 10 target-selection discipline.
**RDNA3 risk notes**: One optimization at a time (rule 2) — no parallel variant chaos; read-only benchmark sweeps may parallelize, code changes serialize. Include a rocBLAS comparison arm per shape (large-M only — small-M decode routes through ggml MMVQ kernels, verified in ggml-cuda.cu dispatch); check Tensile coverage before trusting warmup crashes as kernel bugs. hipBLASLt is EXCLUDED as an arm (AMD excludes gfx1100; incomplete TensileLite). MMA/rocWMMA stays out of v1 (flag-only experiment deferred to v2).
**Plans**: 4 plans drafted — 05-01 (target study & stock reference extraction), 05-02 (custom GEMV for decode M=1), 05-03 (custom WMMA GEMM for prefill M>>1), 05-04 (shape-sweep microbenchmarks & e2e A/B integration) — see `.planning/phases/05-first-custom-kernel-bottleneck-attack/`
**Parallelization**: Strictly serial by rule 2 through implementation/correctness; microbenchmark sweep variants are parallelizable as measurement-only subagent jobs once the kernel compiles; e2e A/B runs only after the winner is selected.

### Phase 6: Integration, Full Validation & Publication

**Mode**: standard (horizontal layers)
**Goal**: Winning kernels live permanently behind switchable flags as additive quilt patches over the pinned upstream, the stock baseline remains intact forever, and the complete before/after result matrix plus methodology is published — failures included.
**Depends on**: Phase 5
**Requirements**: INTEG-01, PUB-01
**Success Criteria** (what must be TRUE):

  1. Custom kernels enable/disable via a documented compile/runtime ON/OFF switch implemented as additive patches over the pinned upstream; OFF-mode behavior is equivalent to stock, proven by an empty-flag plumbing patch first (INTEG-01).
  2. The pristine stock baseline remains buildable and runnable from the same tree at any time — demonstrated by rebuilding it and re-running a baseline sanity bench mid-phase (rule 3) (INTEG-01).
  3. The final deliverable is published with the complete stock-vs-optimized matrix (pp/tg × context × flash-attn), raw data files, exact build commands, all recorded versions, benchmark methodology, kernel source, known limitations, and failed-experiment log (PUB-01).

**Key deliverables**: patch series (`patches/*.patch`) + paired baseline/optimized binaries from one tree; final e2e result matrix with temps/power columns from Windows telemetry; packaged repo layout per orig Phase 17; publication writeup; **release hygiene** (LICENSE Apache 2.0 + NOTICE for Qwen/base, `README.md` license section, `models/README.md` provenance, stray build artifact cleanup `test_wmma*`, `benchmarks/results/tmp_*` removal, `check_no_ggml.sh` PASS, `CHANGELOG`/`v1.0.0-gfx1100` tag per `config.json`).
**Original-roadmap lineage**: orig Phases 12 (integration), 15–16 (e2e suite + honest comparison), 17 (packaging/publish); Milestones 6–7. "Publish the complete matrix, not just the best number."
**Integration risk notes**: Anti-fork discipline — quilt overlay keeps delta-from-stock reviewable and bisectable; no drifting hard fork. Re-run the env version gate after any forced driver/ROCm change (Pitfall: silent pairing breakage). Final numbers must come from paired runs within one thermal window.
**Plans**: 5 plans — 06-01 empty-flag plumbing proof (OFF=stock bit-identical, compiles clean); 06-02 winner patches behind `GGML_CUDA_ENABLE_CUSTOM_GFX1100` switch (real `git diff`-generated, `git apply --check` verified); 06-03 baseline-preservation guard (rebuild stock + `run_op_gate` + bench sanity + env version gate); 06-04 publication package + paired final matrix (pp/tg × {4096,8192,16384} × flash-attn {on,off} × HIP stock/custom + Vulkan arm, smoke-first gate, thermal pairing, variance published, `RunStore` + `CHECKSUMS.sha256`, full 145-chunk QUAL-02, hygiene incl. LICENSE/NOTICE/CHANGELOG/tag); 06-05 thermos remediation (gitignore staging trap, deterministic fixture seed, WMMA coverage + uninitialized-LDS fix, barrier/misaligned-load guards, dead code, doc drift) — see `.planning/phases/06-integration-full-validation-publication/`
**Parallelization**: Flag-plumbing/patch work and documentation/publication drafting are concurrent subagent-friendly tracks; the final matrix execution is serial and last (thermal pairing requirement).

## Progress

**Execution Order:** Phases execute in numeric order 1 → 2 → 3 → 4 → 5 → 6 (sanctioned overlap: Phase 4 may proceed concurrently with Phases 2–3).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Environment Validation & Stock Baseline | 4/3 | Complete    | 2026-08-23 |
| 2. Benchmark Harness & Baseline Matrix | 5/5 | Complete    | 2026-08-23 |
| 3. Correctness Gates & Bottleneck Profiling | 4/4 | Complete    | 2026-08-24 |
| 4. Kernel Playground Scaffold | 3/3 | Complete    | 2026-08-25 |
| 5. First Custom Kernel (Bottleneck Attack) | 4/4 | Complete    | 2026-08-25 |
| 6. Integration, Full Validation & Publication | 5/5 | Complete    | 2026-08-25 |
| 7. Hybrid DP4A & WMMA Matrix Core Optimization | 3/3 replanned | In Progress (0/3 must-haves closed) | 2026-08-30 (07-01..03 closure replan) |

## Coverage Validation

All 17 v1 requirements mapped to exactly one phase:

| Category | Requirements | Phase |
|---|---|---|
| Environment & Baseline | ENV-01..04 | 1 |
| Benchmarking | BENCH-01..04 | 2 |
| Correctness | QUAL-01..02 | 3 |
| Profiling | PROF-01..02 | 3 |
| Kernels | KERN-01 | 4 |
| Kernels | KERN-02..03 | 5 |
| Integration | INTEG-01 | 6 |
### Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization — RE-SCOPED 2026-08-28 (owner 3 wishes + deep-research `output/deep-research/1000t-s-at-8k-gfx1100.md`)

**Mode**: standard (horizontal layers)
**Goal**: Fuse `Q8_1` integer activation quantization and RDNA3 hardware matrix cores (`v_dot4_i32_i8` / `v_wmma`) with our Wave32 cooperative workgroups to **beat real production stock llama.cpp by ≥10% end-to-end in `llama-bench`, Windows-native (≤2 langs), and with 10× (15× LLM QA) statistical rigour** — addressing the deep-research findings: `8k` quadratic cliff, `800 GB/s` roof, `KV≈128 KiB/tok` budget (`GQA`), WSL2 DXG jitter/lack of `rocprofv3`.
**Depends on**: Phase 6
**Requirements**: KERN-04 (Hybrid DP4A GEMV), KERN-05 (WMMA Streaming GEMM), INTEG-02 (End-to-End Uplift) **plus new must-haves 2026-08-28: REQ-WIN-07 (Windows-native ≤2 langs), REQ-PERF-07 (≥1.10× pp+tg), REQ-STAT-07 (N≥10, LLM QA N≥15), BENCH-01 amended**
**Success Criteria** (all must be TRUE; `≥10` / `≥15` averaging is a gate, not a suggestion):
  1. Microbenchmark comparator in `kernels/matmul_iq4xs/` tests directly against real upstream `vec_dot_iq4_xs_q8_1` DP4A implementation — reported as `median` + `mean` + `stddev` + `p95` over **`N=10`** runs per shape (REQ-STAT-07).
  2. Custom cooperative 8-thread DP4A GEMV kernel (`impl_gemv_dp4a_gfx1100.hip`) achieves **>1.2× median** over stock MMVQ in `N=10` microbenchmarks and **>38 t/s decode** in `llama-bench` **`N=10` averaged** (not single-run).
  3. Custom WMMA streaming GEMM kernel (`impl_gemm_wmma_stream.hip`) executes hardware matrix cores on quantized inputs achieving **>950 t/s prefill** in `llama-bench` **`N=10` averaged**.
  4. Quilt patch `patches/0001-gfx1100-mul-mat-custom.patch` updated; `QUAL-01` op-gate (0 errors) and `QUAL-02` model-gate (PPL 6.4271) remain green — **each gate reported over `N=10` runs** (or `N≥10` for stochastic canaries).
  5. **[NEW — must-have #1, REQ-WIN-07]** Repo builds & runs **natively on Windows 11** via `build_windows.bat` (`HIP_PATH` + `clang++.exe --offload-arch=gfx1100`, `-G Ninja`) with **pure C++/HIP + CMake only** (`find -name "*.py" ! -path "./llama.cpp/*" == 0`), and `build-windows/bin/llama-server.exe` serves `curl http://127.0.0.1:8000/v1/chat/completions → 200` on gfx1100. No `3+` language-server stack.
  6. **[NEW — must-have #2, REQ-PERF-07]** Paired `llama-bench` A/B (`stock OFF` vs `custom ON`, same `bb4caa75`, `-ngl 99 -b 2048`) at **`{512,1024,2048,4096,8192}`** (if VRAM pre-flight passes) shows **custom median `tok/s` ≥1.10× stock** for **both** `pp` and `tg` at every tier, with `mean−1σ ≥1.10×` over `N=10` in one thermal window (microbench `>1.2×` alone is not sufficient). Current `808→849 pp4096` (+5.1%) does **not** yet pass.
  7. **[NEW — must-have #3, REQ-STAT-07]** All Phase 7 numbers are **`N≥10` averaged** (microbench + `llama-bench` + quality gates) with `median`/`mean`/`stddev`/`p95` tables, and every **LLM question test via the custom kernel is `N≥15`** (`temp=0`, fixed prompt, `avg tok/s` + `avg latency` + `stddev` + per-run table). Single-run claims are rejected.

**Plans**: 3 closure plans in `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/` — **replanned 2026-08-30: old 07-01..07-04 deleted, one closure plan per must-have; all 3 carry the required objectives and the ways to achieve them (see `output/deep-research/phase7-3must-haves-exhaustive.md` + `docs/PUBLICATION.md §8`)**:
- 07-01: **REQ-WIN-07 Windows-native ≤2 langs closure** — `build_windows.bat` HIP_PATH quoting + `find_package(hip via HIP_PATH)` + `-G Ninja` (not `cl`) + `safe.directory`/`*.patch eol=lf` + prune `py 40→0` + `llama-server.exe :8000 → 200` on gfx1100 (HIP SDK 6.4 install = blocking human gate). Ways: `winget install Ninja CMake`, `Verify clang++.exe --offload-arch=gfx1100 --version`, `Phase 8` prune per `08-refactor-windows-native/08-01..04`.
- 07-02: **REQ-PERF-07 `≥1.10×` pp+tg closure** — race 5 GEMM variants (`64x32 P2+33 / P4_XOR 0% / 64x64 B-stationary 64× reuse / LUT μ=4 / 128x32`) + 2 GEMV (`+33` vs `XOR`) as distinct OBJECTs, `soft HIP_CHECK` + `weak` ODR + `8192 SKIPPED`, `P=4 XOR` + `b128` + `16x64 swizzle` + `sched_barrier 0x0080/0x0008` + `VGPR ≤64 →16 waves`, `race.py --repeats 10 A,B,A,B` interleaved, `bench --runs 10`, paired `llama-bench N=10` one thermal window (`hwinfo 1Hz` + `watchdog 90C` + `RunStore+CHECKSUMS`); current honest N=10 `512 pp 1.079x FAIL`, `M1024 1.08 avg 1.89 peak` — ways per `07-RESEARCH.md` high-yield + MARLIN/SmoothQuant α=0.5.
- 07-03: **REQ-STAT-07 N≥10/15 closure** — `bench_* --runs 10` 45/45 valid JSON `median/mean/stddev/p95`, `llama-bench -r 10` 4-tier N=10 ×5 entries, LLM QA **N=15** `temp=0` fixed prompt `-n 128` per-run 15-row table, `QUAL-01/02` N=10 green, `race.py` interleaved, honest `1.05-1.07 FAIL` tables in `KERNEL-BENCH-DIFF.md §8`/`PUBLICATION.md` (single-run banned).

| Publication | PUB-01 | 6 |
| Hybrid DP4A Optimization (re-scoped 2026-08-28) | KERN-04, KERN-05, INTEG-02 **+ REQ-WIN-07, REQ-PERF-07, REQ-STAT-07, BENCH-01 amended** | 7 |
| Windows Native Refactor (now Phase 7 must-have #1 landing) | REQ-WIN-01..04 (Phase 7 REQ-WIN-07 is the umbrella) | 8 |

**Mapped: 24/24 ✓ — no orphans, no duplicates.**

### Phase 8: Refactor to Windows Native

**Mode**: standard (horizontal layers)
**Goal**: Repository builds and runs natively on Windows 11 for RX 7900 XT gfx1100 via AMD HIP SDK + VS Build Tools — bloat-free pure C++/HIP + minimal CMake, llama-server.exe serves OpenAI API at localhost:8000.
**Depends on**: Phase 7
**Requirements**: REQ-WIN-01 (Eliminate bloat & unnecessary languages), REQ-WIN-02 (Retain core kernels only), REQ-WIN-03 (Windows build pipeline), REQ-WIN-04 (Output & usage)
**Success Criteria**:
  1. No Python/JS remains (no benchmarks/, freetoken-rocm-probe/qstar.mjs, .planning phases 01-07, JSON/JSONL harnesses); `find -name "*.py" ! -path "./llama.cpp/*"` == 0.
  2. `kernels/` pruned to essentials: `block_iq4_xs.h`, `hip_helpers.h`, `impl_gemv_dp4a_gfx1100.hip`, `impl_gemm_wmma_stream.hip` (+ minimal bench/util/ref_cpu), patch clean `git apply --check` PASS.
  3. `build_windows.bat` compiles clean on Windows 11 + VS Build Tools + HIP SDK via `clang++.exe --offload-arch=gfx1100` (not cl) with Ninja, intrinsics `__builtin_amdgcn_sudot4/perm/wmma` compile without MSVC errors.
  4. `build-windows/bin/llama-server.exe` runs and serves `curl http://127.0.0.1:8000/v1/chat/completions` → 200 with `choices[0].message.content`; kernels still pass cosine >=0.999; tree is `kernels/`, `patches/`, `CMakeLists.txt`, `build_windows.bat`.

**Plans**: 4 plans in `.planning/phases/08-refactor-windows-native/`:
- 08-01: Inventory & deletion allowlist (REQ-WIN-01)
- 08-02: Minimal kernels/CMake prune (REQ-WIN-02)
- 08-03: build_windows.bat + Windows HIP toolchain validation (REQ-WIN-03)
- 08-04: Patch integration smoke + server run verification (REQ-WIN-04)

**Risks**: HIP SDK Windows vs WSL path divergence (HIP_PATH vs /opt/rocm), clang vs msbuild (Ninja mandated), gfx1100 vs amdgpu-arch, intrinsic portability, llama-server vs llama-cli naming, Windows model path handling, VRAM 20GB, driver 32.0.31041.1004.

Plans:
- [x] 07-01-PLAN.md — REQ-WIN-07 Windows-native ≤2 langs closure (`build_windows.bat` HIP_PATH `-G Ninja`, `py 40→0`, `:8000 → 200`)
- [x] 07-02-PLAN.md — REQ-PERF-07 `≥1.10×` closure (5 GEMM + 2 GEMV variants, P=4 XOR/b128/swizzle/LUT, `race.py --repeats 10`, paired `llama-bench N=10` thermal-paired)
- [x] 07-03-PLAN.md — REQ-STAT-07 `N≥10/15` closure (`bench 45/45`, `llama-bench N=10×5`, LLM QA N=15, QUAL-01/02 N=10, honest FAIL tables)
- [ ] 08-01-PLAN.md — Inventory & bloat deletion (REQ-WIN-01)
- [ ] 08-02-PLAN.md — Minimal kernels & root CMake (REQ-WIN-02)
- [ ] 08-03-PLAN.md — build_windows.bat + HIP toolchain probe (REQ-WIN-03)
- [ ] 08-04-PLAN.md — Patch smoke + llama-server at localhost:8000 (REQ-WIN-04)
