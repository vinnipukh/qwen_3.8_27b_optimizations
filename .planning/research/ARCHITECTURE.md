# Architecture Research

**Domain:** GPU kernel optimization project structure — gfx1100 (RX 7900 XT) tuned inference path built around stock llama.cpp HIP, developed under WSL2
**Researched:** 2026-08-21
**Overall confidence:** HIGH for component/data-flow/integration patterns (verified against llama.cpp source, ROCm tooling docs, and multiple independent kernel-project precedents); MEDIUM for WSL2-specific profiling reliability (ROCDXG stack is newly production-grade)

> **Model-agnosticity note (locked scoping):** Final model pick (Qwen3-32B dense vs Qwen3-30B-A3B MoE) is PENDING-USER. Architecture below is model-agnostic; **[DENSE]**/**[MOE]** markers appear where structure diverges. Nothing in the component layout changes with the pick; only benchmark-matrix width and one optional playground kernel family do.

---

## Standard Architecture

### System Overview

The architecture is a **hub-and-spoke around an append-only result store**, with two strict rules visible in the diagram itself: (1) nothing reaches the Runtime Integration layer except through the Validation gate, and (2) the stock baseline build is a frozen artifact that no component may modify.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     ARTIFACT INGEST (one-time per model)                   │
│   HF Hub ──► convert_hf_to_gguf ──► GGUF store (SHA256-pinned)             │
│                        │                    │                             │
│                        ▼                    ▼                             │
│              Quantization Pipeline      Tensor Dumper (fixtures)          │
└────────────┬────────────────────────────────┬─────────────────────────────┘
             │ quantized GGUFs                │ real weight-block fixtures
             ▼                                ▼
┌──────────────────────────┐    ┌─────────────────────────────────────────┐
│  C1 BASELINE BUILDER     │    │  C4 KERNEL PLAYGROUND                   │
│  env capture, pinned     │    │  per-op: CPU ref → HIP impl →           │
│  stock llama.cpp builds  │    │  compare → microbench (hipBench-style)  │
└────────────┬─────────────┘    └───────────────┬─────────────────────────┘
             │ baseline binaries                │ winners (kernel + config)
             ▼                                  ▼
┌──────────────────────────┐    ┌─────────────────────────────────────────┐
│  C2 BENCHMARK HARNESS    │    │  C5 VALIDATION SUITE                    │
│  profiles, repeats,      │◄───┤  op-level gates ← + → model-level gates │
│  pp/tg split, VRAM       │    │  (ppl / KL-div / golden outputs)        │
└────────────┬─────────────┘    └───────────────▲─────────────────────────┘
             │ every run                        │ PASS required
             ▼                                  │
┌───────────────────────────────────────────────┴──────────────────────────┐
│                      RESULT STORE (append-only, fingerprinted)            │
└────────────▲───────────────────────────────────────────────────────────────┘
             │ queries (bottleneck tables, regressions, publish matrices)
┌────────────┴─────────────┐    ┌─────────────────────────────────────────┐
│  C3 PROFILER BRIDGE      │    │  C6 RUNTIME INTEGRATION LAYER           │
│  rocprofv3 wrappers →    │    │  pinned upstream + quilt patches +      │
│  kernel→%time table      │───►│  GGML_GFX1100_CUSTOM=ON/OFF flags       │
└──────────────────────────┘    │  → baseline binary + optimized binary   │
                                └─────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **C1 Baseline Builder** | Environment fingerprinting (rocminfo, hipconfig, rocm-smi, versions); building *stock* llama.cpp HIP at a pinned upstream SHA targeting `gfx1100`; producing the never-rebuilt reference binary set | Thin shell/CMake scripting around a vendored llama.cpp checkout; outputs tagged artifacts under `baseline/` |
| **C2 Benchmark Harness** | Fixed workload profiles (chat/coding/long-ctx/long-gen); warmup + ≥N repeats; separate prompt-tok/s and gen-tok/s; VRAM breakdown (weights/KV/compute buffers); embeds full env fingerprint in every row | Wrapper scripts around `llama-bench` / `llama-cli`; JSON/CSV emission into result store |
| **C3 Profiler Bridge** | Wrapping `rocprofv3` captures of real workloads; parsing traces into a kernel→%runtime→bound-type bottleneck table; ranking candidates for the playground | Scripted rocprofiler-sdk invocations + a small parser; degrades to timing-only attribution when counters unavailable |
| **C4 Kernel Playground** | Standalone per-op development of custom kernels, isolated from the runtime: CPU golden reference → HIP implementation → numerical comparison → shape-swept microbenchmark | Plain HIP/C++ executables per op (`kernels/<op>/`), optionally using ROCm's `hipBench` axis-sweep library; fixtures from real GGUF tensors |
| **C5 Validation Suite** | Two tiers: **op-level** (max-abs / rel-error / cosine similarity vs CPU ref, per-bit-width tolerance constants) and **model-level** (perplexity, KL-divergence vs baseline logits, deterministic golden token streams) | Reuse llama.cpp's own patterns: `test-backend-ops.cpp`-style comparisons, `llama-perplexity --kl-divergence`, pinned-prompt greedy-decode snapshots |
| **C6 Runtime Integration Layer** | Applying validated winners to llama.cpp behind compile-time switches without destroying the stock path; producing paired baseline+optimized binaries from the same tree | Vendored pinned upstream + quilt patch series + `GGML_GFX1100_CUSTOM`-style flags |
| **C7 Quantization Pipeline** | Calibration datasets, imatrix generation, quant variants (Q4_K_S/M, Q5_K_M ± imatrix); **[MOE]:** expert-offload/tensor-override sweeps | llama.cpp `llama-imatrix` / `llama-quantize`; calibration corpus management |
| **Result Store** (cross-cutting) | Append-only, run-ID-addressed records of every benchmark, profile, and validation run; source of all published matrices | `results/<date>_<run-id>/{config.json,metrics.csv,logs/}`; never edited, only appended |

### What Talks to What (Boundary Contracts)

| Edge | Contract | Notes |
|------|----------|-------|
| Ingest → C4 | Real weight-block fixtures dumped from pinned GGUFs | Playground tests realistic data distributions, not just random tensors — catches Q4_K superblock-scale edge cases synthetic data misses |
| C4 → C5(op) | Kernel passes tolerance gate *before* any perf claim | Gate ordering is enforced by the playground's own driver, not developer discipline |
| C4 → C6 | Only "winners": kernel source + tuned config + passing op-level report | Losers are archived in result store (publish failed experiments too) |
| C6 → C5(model) | Integrated build must pass ppl/KL/golden gates vs baseline binary | This is the integration test; no phase-gate without it |
| C5(model) → C2 | Gated builds become benchmarkable | Harness refuses un-gated builds (config check) |
| C2 → Result Store | Every run appends a fingerprinted record | Fingerprints: llama.cpp SHA, ROCm/driver/WSL versions, GGUF SHA256, kernel version tag, clocks/thermals |
| C3 → Roadmap process | Bottleneck table names THE next target | Enforces "one optimization at a time"; C3 output is the input to prioritization, not C4 curiosity |

---

## Recommended Project Structure

```
qwen-gfx1100/
├── llama.cpp/                  # vendored upstream @ pinned SHA (git remote added, never pushed to)
├── patches/                    # quilt series applied over llama.cpp (see Pattern 2)
│   ├── 0001-add-gfx1100-q4k-fused-dequant-mm.patch
│   ├── 0002-flag-guard-custom-kernel-selection.patch
│   └── series                  # ordered; each patch = one optimization
├── kernels/                    # standalone playground (no llama.cpp includes allowed)
│   ├── common/                 # HIP error checks, timers, tensor IO, RNG, arg parsing
│   ├── dequant/
│   │   ├── ref_cpu.cpp         # golden scalar reference
│   │   ├── impl_gfx1100.hip    # device implementation(s)
│   │   ├── test_compare.cpp    # tolerance gate runner
│   │   └── bench_sweep.cpp     # shape/config sweep microbenchmark
│   ├── matmul/                 # same quartet layout
│   ├── attention/
│   └── kv/
├── tools/
│   ├── tensor_dump/            # extract real GGUF weight blocks → playground fixtures
│   └── apply_patches.sh        # rebuild baseline & optimized trees deterministically
├── bench/
│   ├── profiles/               # chat/coding/longctx/longgen definitions
│   ├── run_bench.sh            # warmup/repeats/pp-tg split/fingerprint wrapper
│   └── parse_rocprof.py        # profiler bridge parser
├── eval/
│   ├── golden/                 # pinned prompts + expected greedy token streams
│   ├── calibration/            # general/coding/reasoning/mixed corpora
│   └── run_ppl_kl.sh
├── models/README.md            # exact GGUF repos/files/SHAs (written Phase 1)
├── baseline/                   # FROZEN stock binaries + build logs (tagged, never rebuilt)
├── results/                    # append-only store: <date>_<run-id>/...
└── autotune/gfx1100.json       # shape → winning tile/wg config registry (later phases)
```

### Structure Rationale

- **`kernels/` forbids llama.cpp includes:** the whole point of the playground is debuggability — a failing kernel must be reproducible in a 200-line standalone binary, not inside a 5-minute model load. Dependency direction is one-way: integration copies *out* of the playground, never in.
- **`patches/` instead of committed edits to `llama.cpp/`:** makes the delta-from-stock always explicit, reviewable, and revertible; upstream bumps become `git fetch && rebase series`, and a failed bump costs hours, not weeks.
- **`baseline/` as frozen artifacts, not a build recipe:** the original roadmap's "keep a stock baseline forever" rule is only enforceable if the baseline is a stored binary + its build log, immune to accidental rebuilds.
- **`results/` append-only with fingerprints:** every published number traces to a complete environment record; WSL2 adds kernel/driver layers that silently change performance, so the fingerprint is mandatory, not decorative.

---

## Architectural Patterns

### Pattern 1: Reference-Implementation Ladder (CPU ref → HIP → compare → microbench)

**What:** Every kernel exists four times conceptually: a trivially-correct CPU scalar reference, the optimized HIP implementation, an automated comparison harness, and a swept microbenchmark — in that order of creation.
**When to use:** Always, for every op. This is the near-universal convention across kernel projects (KernelBench's correctness-vs-reference grading, llama.cpp's own `test-backend-ops.cpp` comparing every op against the CPU backend, multi-backend teaching repos with shared CPU references).
**Trade-offs:** Costs ~20% extra code up front; repays it by making miscompiles (LDS bank conflicts, wave-mode bugs, race conditions) debuggable in minutes instead of inside a full-model run where they masquerade as "quality loss."

```cpp
// kernels/matmul/test_compare.cpp — the gate, not a nicety
metrics m = compare(ref_cpu_out, hip_out);           // max_abs, rel_err, cosine_sim
assert_within(m, tolerances_for(GGML_TYPE_Q4_K));    // per-bit-width constants,
exit(m.worst > T ? EXIT_INCORRECT : EXIT_OK);        // mirroring test-quantize-fns.cpp
```

### Pattern 2: Patch Overlay + Flag-Gated Selection (anti-fork integration)

**What:** Keep llama.cpp as a **vendored pinned-SHA checkout**; express every customization as an ordered quilt patch series; guard each custom kernel's dispatch hook behind a compile-time macro (`GGML_GFX1100_CUSTOM_KERNELS=ON/OFF`). Build **two binaries from the same tree**: stock-flagged (re-verifiable against frozen baseline) and custom-flagged.
**When to use:** From day one — even before the first custom kernel, the *first* patch should be the empty flag plumbing, proving ON/OFF produce bit-identical behavior.
**Trade-offs:** Rebase friction on upstream bumps (mitigated by keeping patches small, single-purpose, mostly *additive files* + one guarded selection line); slightly more ceremony than naive in-tree hacking. Vastly cheaper than a drifting hard fork.

**Why not the alternatives (verified):**
- **Out-of-tree backend plugin (`GGML_BACKEND_DL`, PR #10469):** the dynamic-loading mechanism genuinely exists — backends build as shared libs and register via `ggml_backend_load_all()`. But it registers *additional* backends competing at buffer/schedule level; selectively overriding single ops *inside* the built-in HIP backend still requires modifying that backend, plus custom buffer-type plumbing for weights. High complexity, wrong granularity for a single-GPU project. Keep as a footnote alternative.
- **Permanent hard fork:** proven viable at scale by ik_llama.cpp (custom quants/kernels carried for years), but that project's own history — including eventually migrating hosting — illustrates the maintenance gravity. Wrong choice when the deliverable is "delta vs stock," because the delta becomes the codebase.

**Where the hook lives (verified against source):** ggml's CUDA/HIP backend dispatches per-op — e.g. `ggml_cuda_mul_mat` selects MMQ vs hipBLAS by type/device, `mul_mat_vec_q` serves decode-sized batches — with one file per op family compiled through HIP. A custom kernel therefore integrates as: new `.cu` file (compiled via HIP) + a few guarded lines in the existing selection function. Minimal, well-understood surface.

### Pattern 3: Fingerprinted Append-Only Result Store

**What:** Every benchmark/profile/validation run appends an immutable record addressed by run ID, embedding the full environment fingerprint (llama.cpp SHA, ROCm release, ROCDXG/WSL kernel + Windows driver versions, GGUF SHA256, kernel tag, clocks/thermals).
**When to use:** First week. It is cheap (a directory convention + a JSON emitter) and everything downstream — regression detection, published matrices, bisecting a mysterious slowdown — depends on it.
**Trade-offs:** None material. Skipping it is how projects end up unable to explain their own headline numbers.

### Pattern 4: Shape-Keyed Autotune Registry

**What:** Sweep results collapse into a lookup file (`autotune/gfx1100.json`): problem shape (M×N×K, quant type) → winning config (tile sizes, workgroup, LDS usage, vector width). Runtime dispatch consults the registry; CUTLASS Profiler's CSV + best-kernel-for-fixed-shape mode is the canonical design precedent.
**When to use:** After ≥2 kernels have non-obvious winning configs. Premature here = premature optimization of the tuner itself.
**Trade-offs:** Registry can go stale vs kernel changes; mitigate by storing the kernel-tag that produced each entry.

---

## Data Flow

### Weight/Data Flow

```
HF Hub weights (fp16/bf16 safetensors)
    ↓ convert_hf_to_gguf.py
GGUF master (SHA256-pinned, models/README.md)
    ├─► llama-quantize (+imatrix from calibration corpora) ─► Q4_K_S/M, Q5_K_M variants
    ├─► stock baseline builds ─► frozen baseline/ binaries
    ├─► tensor_dump ─► kernels/*/fixtures/*.bin (real superblocks/scales/blocks)
    └─► [MOE] expert-shape audit ─► routing/expert-count metadata for offload sweeps
```

### Kernel Development Loop (per bottleneck, repeated)

```
Profiler bridge: kernel X = top %runtime
    ↓
Playground: CPU ref (correct by construction) → HIP impl → compare gate → sweep microbench
    ↓ winner beats stock kernel in microbench (same shapes, same hardware, event timing)
Patch N: add kernel file + guarded dispatch hook (flag default OFF)
    ↓
Validation suite: op-level ✓ already passed → model-level ppl/KL/golden vs baseline
    ↓ PASS
Benchmark harness: full profiles, pp/tg separately, ≥N repeats → result store
    ↓ no regression beyond tolerance?
Patch stays; update autotune registry. Else: drop patch, archive failure in results/.
```

### Integration Loop (upstream bump, occasional)

```
git fetch upstream → rebase patches/series onto new pinned SHA
    ↓ conflicts? resolve keeping patches minimal/additive
Rebuild stock-flagged binary → verify bit-identical golden outputs vs frozen baseline
    ↓
Rebuild custom-flagged binary → full validation + benchmark pass
    ↓
Re-tag frozen baseline ONLY if security forces it (recorded, announced)
```

### Key Data Flows

1. **Correctness flows forward, never sideways:** a number in the result store is only meaningful if the run that produced it was gated by validation — the dependency chain ref → gate → integrate → validate → bench is linear and mandatory.
2. **Profiling closes the loop:** benchmark results alone say *what* got faster; profiler bridge says *what to attack next*. Both write to the same store so bottleneck tables cite measured baselines.
3. **Prefill/decode is a first-class dimension:** every record (bench row, microbench point, validation case) is keyed by M-regime (M≈1 decode vs large-M prefill). Blended numbers are structurally impossible in this store.

---

## Build Order (Dependency-Honoring)

Each step below depends only on steps above it; parallelizable tracks marked ∥.

| Order | Component | Depends On | Rationale |
|-------|-----------|------------|-----------|
| 1 | **C1 Environment + Baseline Builder** | — | Everything gates on a working gfx1100 toolchain; WSL2 feasibility is itself the first exit criterion. Produces frozen baseline + env fingerprints. |
| 2a ∥ | **C2 Benchmark Harness + Result Store** | C1 | Can't trust any future number without it; needed to characterize baseline quants. Low effort, pure scripting. |
| 2b ∥ | **C4 Playground Core Scaffolding** (common/ + compare driver + first trivial op, e.g. dequant) | C1 (toolchain only, no model) | Model-independent! Dequant/microbench work can start while ~17 GB of weights download. This overlap shortens the critical path materially. |
| 3 | **C5 Validation Suite, model tier** (golden outputs + ppl/KL vs baseline) | C1, C2 | Must exist *before* the first integrated change, or the first integration has nothing to be tested against. |
| 4 | **C3 Profiler Bridge** | C1, C2 (runnable workloads) | Profile-before-optimize requires real workloads running; produces the bottleneck table that picks playground targets. Verify rocprofv3-on-WSL2 HERE (see contingency section). |
| 5 | **C7 Quantization Pipeline** | C1 | Needed early enough to arbitrate dense-vs-MoE and quant choice with measured ppl/VRAM; feeds C2 matrices. |
| 6 | **C4 first serious kernel** (fused Q4_K dequant+matmul per bottleneck table) | 2b, 4 | Targets chosen by data, not folklore. |
| 7 | **C6 Runtime Integration Layer** (first real patch + flags) | 3, 6 | Scaffold the empty flag plumbing during 2b (cheap), but first *contentful* patch waits for a validated winner. |
| 8 | Loop: profile → playground → patch → validate → bench (attention, KV, scheduling…) | all above | Steady state; each iteration identical shape. |

**Critical-path insight:** the only serial choke points are C1 and the C5-model-tier suite (step 3). Everything else overlaps. The roadmap should let playground scaffolding proceed during model download/baseline characterization rather than serializing "environment → benchmarks → profiling → kernels."

---

## Scaling Considerations

Not user-scaling — workload-regime scaling (the axis that actually stresses this architecture) and project-growth scaling:

| Regime / Stage | Architecture Adjustments |
|----------------|--------------------------|
| **Decode (M≈1)** | Memory-bandwidth-bound; playground microbenches must include vec-kernel shapes (mul_mat_vec_q territory); result store keys these rows distinctly |
| **Prefill (large M)** | Compute-bound GEMM regime; shape sweeps dominate; hipBLAS comparison arm mandatory (sometimes rocBLAS wins — measure, don't assume) |
| **Long context (16–32K)** | KV cache + attention paths stress VRAM budgeting (<1.5 GB headroom at Q4_K_M); VRAM breakdown fields in every bench row become load-bearing |
| **[MOE] variant** | Adds expert-routing traffic + gather/scatter kernels to profiles; C2 gains offload-sweep configurations (`--n-cpu-moe`-style tensor overrides); playground gains an optional MoE kernel family. Dense components unchanged. |
| **1 kernel → many kernels** | Autotune registry (Pattern 4) prevents dispatch-table sprawl; per-op quartet layout scales linearly |
| **Upstream churn** | Patch discipline (additive-first) determines whether bumps cost an afternoon or a rewrite; this is the project's actual long-term scaling risk |

### Scaling Priorities

1. **First thing to break:** WSL2 profiling fidelity (counters/PM sampling under /dev/dxg). Mitigation designed-in: C3 is a thin, swappable wrapper with a native-Linux twin.
2. **Second:** patch-series drift as kernels accumulate. Mitigation: additive-only patches, one optimization per patch, flag-per-optimization.

---

## Anti-Patterns

### Anti-Pattern 1: Developing kernels inside the runtime
**What people do:** Edit ggml source, rebuild llama.cpp, load the 17 GB model, squint at outputs.
**Why it's wrong:** A 5-minute iteration loop turns 50 experiments/day into 5; failures inside the full graph are unattributable.
**Do this instead:** The playground quartet (Pattern 1); runtime only ever sees validated winners.

### Anti-Pattern 2: Drifting hard fork
**What people do:** Clone llama.cpp, commit custom kernels directly to a private main branch, merge upstream "eventually."
**Why it's wrong:** Delta-from-stock becomes unknowable; the "beat stock" claim degenerates into "beat our own stale snapshot." ik_llama.cpp demonstrates survival is possible but expensive.
**Do this instead:** Pinned-SHA vendor + quilt patches + flags (Pattern 2).

### Anti-Pattern 3: Validating after integrating ("it's faster, ship it")
**What people do:** Microbench win → flip the flag on permanently → spot-check a chat reply.
**Why it's wrong:** Quantized-kernel bugs are silent quality erosion — plausible text, subtly degraded; perplexity drift accumulates invisibly across stacked optimizations.
**Do this instead:** Hard gate order: op-level tolerance → model-level ppl/KL/golden → then benchmark. Store enforces via config check.

### Anti-Pattern 4: One universal kernel
**What people do:** Tune one matmul config that averages well across M.
**Why it's wrong:** M≈1 decode and large-M prefill sit in opposite regimes (bandwidth- vs compute-bound); an average is worse than either specialist, and RDNA3 (wave32/64 dual mode, LDS characteristics, no CUDA-style cp.async) punishes transplanted-CUDA intuitions.
**Do this instead:** Two dispatch paths selected by M-regime from day one; sweeps always stratified by regime.

### Anti-Pattern 5: Rebuilding the baseline casually
**What people do:** `git pull && rebuild` the stock build mid-project "to stay fresh."
**Why it's wrong:** Every historical comparison silently invalidates; the never-regressing property is defined against a specific binary.
**Do this instead:** Frozen `baseline/` artifacts + explicit, recorded re-baselining ritual (Integration Loop above).

---

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **ROCm/HIP toolchain (WSL2)** | Direct cmake builds `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100`; AMD docs list an explicit gfx1100 example | Versions fingerprinted everywhere; ROCDXG production support arrived with Adrenalin 26.2.2 + ROCm 7.2.1 — pin exactly these |
| **rocprofv3 / rocprofiler-sdk** | C3 thin wrappers (env-detecting) | Works on WSL2 gfx11 via `/dev/dxg` paravirtualization (`/dev/kfd` absent); treat as MEDIUM-reliability until proven in Phase-1-style smoke tests |
| **Radeon GPU Profiler** | **None under this constraint** | RGP does not support the ROCm/Linux stack (upstream issue #84); it targets the Windows PAL stack. Instruction-level insight is structurally out of reach under WSL2-only + no-native-Windows-HIP constraints |
| **hipBLAS/rocBLAS** | Comparison arm in every matmul microbench | Sometimes the vendor library wins; the harness must make losing to rocBLAS a *measured* outcome, not an assumption either direction |
| **ROCm hipBench (optional)** | Microbench sweep engine for playground | Official C++17 axis-sweep library; adopt if it fits, else the ~300-line custom sweep driver is fully sufficient |
| **Hugging Face Hub** | One-time scripted downloads with SHA256 recording | Weights arrive at execution start, per project decision |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Playground ↔ Integration | One-way copy: kernel source + config + report | Never the reverse; playground stays dependency-free |
| Validation ↔ Everything | Gate service: callers submit artifacts, receive PASS/FAIL + metrics | Same tolerances file used op-level and (scaled) model-level |
| Benchmark ↔ Result Store | Append-only writes, read-only analytics | No component mutates history |
| Profiler ↔ Prioritization | Bottleneck table documents | Human-in-the-loop decision, informed by data |

---

## WSL2 vs Native-Linux Contingency Boundary (Profiling Component)

**Verified state of the world:**
- Production ROCm-on-WSL2 ships via **ROCDXG (librocdxg)** with **Adrenalin 26.2.2 + ROCm 7.2.1**; the GPU appears through `/dev/dxg` (no `/dev/kfd`).
- **rocprofv3-over-/dev/dxg for gfx11-family compute/profiling is implemented upstream** (rocprofiler-sdk PR: agent enumeration, counter registration, HIP launch tracing under WSL2). Supervisor-provided facts independently confirmed against the ROCm/rocm-systems sources.
- **RGP is unavailable on ANY Linux path** (ROCm stack unsupported by design), so the WSL2-vs-native choice does *not* change RGP availability — it changes counter-collection robustness and tool maturity.

**Architectural consequence:** C3 (Profiler Bridge) is deliberately the *thinnest* component — shell wrappers + a parser — so the entire component relocates to native-Linux (dual-boot contingency) without touching any other component. HIP kernel sources, playground, validation, and benchmark harness are byte-identical across environments; only C3 carries environment-sensitive logic:

```
C3 wrapper pseudo-logic:
if /dev/kfd present  → native path (full counter set)
elif /dev/dxg present → WSL2 path (rocprofv3 via rocprofiler-sdk; verify counter coverage)
else                  → timing-only degradation (hipEvent attribution, coarse but honest)
```

**Degradation policy:** if WSL2 counter data proves incomplete or flaky, fall back *for profiling sessions only* to native Linux; never let profiling gaps silently turn into "we optimized what we could see." Timing-only attribution (which kernel eats wall time) remains available under WSL2 even when hardware-counter depth does not — sufficient for bottleneck *ranking*, insufficient for memory-analysis depth. Flag this as a Phase-scoped verification item.

---

## Sources

**Primary / official (HIGH confidence):**
- llama.cpp `tests/test-backend-ops.cpp` (cross-backend op validation + perf mode) — github.com/ggml-org/llama.cpp
- llama.cpp `tests/test-quantize-fns.cpp` (per-bit-width max-error constants) — ibid.
- llama.cpp `tools/perplexity` README + KL-divergence PR #5076 + KLD scoreboard PR #6936 — ibid.
- GGML backend dynamic loading PR #10469 (`GGML_BACKEND_DL`, `ggml_backend_load_all`) + `ggml-backend-reg.cpp` registry — ibid.
- ggml-cuda structure: `mmq.cu/.cuh` dispatch-by-type, mul_mat_vec_q tiling PR #5434, MMQ-vs-cuBLAS default PR #8075 (notes RDNA3 FP16-tensor-core exception), backend-split refactor discussion #22975 — ibid.
- CUTLASS Profiler docs (M/N/K sweeps, CSV output, best-kernel-for-fixed-shape, heuristics) — docs.nvidia.com/cutlass
- ROCm hipBench (HIP kernel benchmarking, parameter axes) — github.com/ROCm/hipBench
- rocprofiler-sdk WSL2 /dev/dxg support PR (ROCm/rocm-systems #7016); ROCDXG WSL guide (rocmdocs.amd.com — Adrenalin 26.2.2 + ROCm 7.2.1); rocprofv3 docs — AMD
- RGP ROCm-stack non-support — GPUOpen-Tools/radeon_gpu_profiler issue #84; RGP known-issues manual

**Precedents / community (MEDIUM-HIGH):**
- ik_llama.cpp — long-lived custom-kernel/quant fork (fork-cost evidence) — github.com/ikawrakow/ik_llama.cpp
- mesh-llm `LLAMA_CPP_FORK.md` — pinned-SHA + documented small-commit-overlay practice — github.com/Mesh-LLM/mesh-llm
- qvac `merging_strategy.md` (in ggml-org history) — branch-based fork sync strategy
- KernelBench / KernelBench_X — correctness-vs-reference + perf grading conventions; tensormux `write-kernel-test-plan` skill — coverage dimensions for kernel test plans
- Modular `verify.py` accuracy harness — tol/cosine/KL combined metric defaults (rel 1e-3, abs 1e-4, cos 1e-3, KL 1e-3)
- vLLM kernel test patterns — quantized-GEMM test conventions

**Session limitation note:** web-search synthesis was used rather than Context7 doc pulls (provider unavailable in session); all architecture-critical claims were cross-checked against ≥2 independent sources including primary repository/code sources, per verification protocol. WSL2 profiling *reliability in practice* remains MEDIUM confidence — it is newly-production software and is flagged for phase-scoped verification regardless.

---
*Architecture research for: gfx1100 kernel-optimization project structure (Qwen 27B-class on RX 7900 XT, WSL2)*
*Researched: 2026-08-21*
