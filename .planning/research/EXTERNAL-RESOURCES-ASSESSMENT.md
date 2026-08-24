# External Resources Assessment — amd/skills, Hyperloom, ROCm SDK, hip-rocm skill

> **UPDATE 2026-08-23:** `magpie-kernel-evaluator` and `rocm-doctor` have since been
> installed verbatim under `.agents/skills/` with local `PROJECT-NOTES.md` caveats;
> provenance in `skills-lock.json`. See §1 for their sanctioned-use boundaries.

*Evaluated 2026-08-23, between Phase 1 completion and Phase 2 planning. Sources inspected directly:
amd/skills repo cloned & read (SKILL.mds), Hyperloom repo cloned & read (README, compatibility.rst),
rocm.docs.amd.com component listings, mohitmishra786 hip-rocm SKILL.md fetched verbatim
(copy archived at `.planning/tmp/hip-rocm-SKILL.md`).*

## TL;DR verdict table

| Resource | Runs here? | Verdict | Where it maps |
|---|---|---|---|
| [amd/skills](https://github.com/amd/skills) | Partially | **Selective mining only** — don't adopt wholesale | Phase 4/5 conventions; PROF-01 contingency |
| [AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom) | **No** (MI3xx/vLLM/SGLang only) | Methodology reading, not tooling | Mirrors our Phase 2→6 loop |
| [ROCm Core SDK page](https://www.amd.com/en/products/software/rocm/sdk.html) | Already installed | **Bookmark as docs hub** — zero install action | Phase 3 (rocprofv3), Phase 5 (rocBLAS), v2 (rocWMMA) |
| skills.sh hip-rocm (mohitmishra786) | N/A (knowledge skill) | **Skip** — superseded by internal research, carries gfx942 bias | None |

---

## 1. github.com/amd/skills — agent-skill catalog (Claude Code/Cursor/Codex format)

Catalog reviewed: `local-ai-use`, `local-ai-app-integration`, `lemonade-router-builder`,
`serving-llms-on-instinct`, `serving-llms-on-epyc`, `hyperloom-workload-optimizer` (planned),
`rocm-doctor` (staging), `magpie-kernel-evaluator`, `tracelens-analysis-orchestrator`.
Most target Instinct/EPYC/Ryzen-AI stacks — out of scope for us. Three touch our phases:

### magpie-kernel-evaluator — most relevant item
- `magpie analyze` (single kernel: correctness testcase + perf profile) and
  `magpie compare` (rank ≥2 variants with explicit baseline, identical
  inputs/tolerances/warmup/iterations) for standalone **HIP** kernels.
- This is nearly a 1:1 external analogue of our Phase 4 KERN-01 quartet
  (`ref_cpu → impl_gfx1100 → test_compare → bench_sweep`) and Phase 5 head-to-head
  microbenchmarks. Its stated discipline ("reject candidates that fail correctness
  before considering performance rankings"; "record exact config, model revision,
  commit") matches our BENCH/KERN requirements.
- Caveats: Magpie is validated on Instinct + vLLM/SGLang; the skill itself warns to treat
  unlisted hardware/ROCm as *unverified until tested*. Its profiling legs depend on exactly
  what our PROF-01 identifies as the soft spot under DXG.
- **Action:** use as design prior art when writing the Phase 4 harness conventions;
  optionally probe `magpie analyze --type hip` inside the playground in Phase 4 as upside,
  never as dependency.

### rocm-doctor (staging) — contingency-only
- Diagnoses ROCm/HIP/PyTorch/**llama.cpp** failures against a closed misconfiguration
  catalog via the `rocm` CLI.
- **Its own scope gate excludes WSL2 explicitly** ("stop and decline"). Our frozen env is
  WSL2 → unusable day-to-day.
- **Action:** bookmark for the sanctioned PROF-01 ladder step (c) — if profiling depth
  forces a native-Linux session, run rocm-doctor there first.

### tracelens-analysis-orchestrator — not applicable
- Consumes PyTorch/Kineto traces from torch profiler. llama.cpp emits none; our attribution
  plan is llama.cpp op-timers (+ rocprofv3-if-DXG-works). Skip; its parallel-subagent report
  structure is anyway what GSD plans already give us.

## 2. AMD-AGI/Hyperloom — autonomous agentic inference optimizer

From README + `docs/compatibility.rst`: requires **MI300X / MI325X / MI355X**, native
Ubuntu 22.04/24.04, ROCm 7.2.x, **vLLM/SGLang**, Claude backend. Components: TraceLens
(profiling brain, roofline targets), Magpie/IntelliKit (trace capture), Arbor (search tree),
GEAK (multi-agent HIP/Triton kernel optimizer), MIT-licensed.

- **Cannot execute on this project**: wrong GPU class (CDNA3 datacenter vs RDNA3 consumer),
  wrong passthrough (native vs WSL2/DXG), wrong runtime (vLLM/SGLang vs llama.cpp).
- Value is methodological: its loop (trace → roofline → ranked bottlenecks → candidate
  kernels → correctness gate → re-benchmark → report) is the agentic version of our binding
  roadmap rules 1–10. Reading its optimization-loop doc before Phase 3 profiling design may
  sharpen the bottleneck-table format.
- GEAK's kernel-optimization knowledge base could be consulted in Phase 5 — but its patterns
  are CDNA/MFMA-flavored; transplanting them to gfx1100 violates roadmap rule 6 without
  re-validation.
- **Action:** optional background reading (`docs/conceptual/optimization-loop.md`). No
  installation, no integration.

## 3. ROCm Core SDK page (amd.com → rocm.docs.amd.com)

Marketing/product hub; canonical content lives at
`rocm.docs.amd.com/en/latest/components/core.html`. We already have ROCm 7.2.1 installed and
frozen in the Phase 1 snapshot — there is nothing to install, and the env must not drift.

Per-component doc entry points that map onto our remaining phases:
- **rocprofiler-sdk / rocprofv3** → Phase 3 PROF-01 feasibility probe (note: legacy
  rocprof/rocprofv2 officially deprecated; rocprofv3 is the supported path — matches our plan)
- **rocBLAS** → Phase 5 large-M comparison arm (roadmap-sanctioned)
- **hipBLASLt** → documented as excluded for us (no gfx1100 support) — docs confirm why
- **rocWMMA** → v2 flag-only experiment; MMA semantics differ from CDNA MFMA
- **ROCgdb, hipcc/clang** → Phase 4 playground debugging/build

Note: ROCm 7.14 transitions packaging to TheRock modular builds — irrelevant while our env is
frozen; do **not** upgrade mid-project (rule: re-run env version gate after any forced change).
- **Action:** add these doc URLs to the reference list used by Phase 3/5 planners. No other action.

## 4. skills.sh/mohitmishra786/low-level-dev-skills/hip-rocm

Generic HIP/ROCm primer (~230 lines): ROCm install, minimal vector_add, HIPIFY API-mapping
table, hipcc flag cheatsheet, rocprof/rocgdb basics, MI300X notes (wavefront 64, MFMA,
HBM ~5.3 TB/s), NVIDIA↔ROCm library map, common-problems table.

Assessment against our needs:
- Oriented to **CUDA→HIP porting** and **gfx942/gfx90a defaults** (`export
  AMDGPU_TARGETS=gfx942`) — wrong arch for us; an agent following it verbatim compiles for
  the wrong ISA. Exactly the failure mode roadmap rule 6 guards against.
- Zero coverage of: WSL2/DXG quirks, RDNA3 wave32-vs-wave64 duality (it hardcodes "wavefront
  = 64"), rocWMMA on RDNA3, quantized dequant/GEMM kernels, llama.cpp ggml dispatch — i.e.,
  all of our actual frontier (GDN scan, fused IQ4_XS dequant+matmul).
- Our internal `.planning/research/deep-research/` corpus (12 targeted scrapes: hip-windows-
  guide, GPUOpen wmma, chipsandcheese RDNA3, librocdxg, rocm-limitations…) is strictly deeper
  and project-correct.
- Copy archived at `.planning/tmp/hip-rocm-SKILL.md`; the common-problems table is the only
  mildly reusable bit.
- **Action:** skip installation. Do not let agents invoke it during kernel work.

---

## Consolidated recommendation for Phase 2+ planners

1. Nothing here changes the roadmap, the frozen environment, or Phase ordering.
2. Pull two ideas forward into Phase 4 design notes: Magpie's analyze/compare config
   discipline (baseline-explicit, tolerance-gated, warmup-identical) — cite as prior art.
3. Add ROCm component-doc URLs (rocprofiler-sdk, rocBLAS, rocWMMA, ROCgdb) to the planner
   reference set for Phases 3–5.
4. Park rocm-doctor + native-Linux note against PROF-01 escalation path (Phase 3).
5. Treat Hyperloom optimization-loop doc as optional pre-Phase-3 reading.
