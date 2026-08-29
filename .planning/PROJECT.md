# Qwen3.8-27B on RX 7900 XT — gfx1100 Inference Optimization

## What This Is

A reproducible GPU-specific inference optimization project for running Qwen3.8-27B on an AMD Radeon RX 7900 XT (RDNA3 / gfx1100, 20 GB VRAM). It starts from stock llama.cpp with the HIP/ROCm backend and progressively replaces verified hot-path bottlenecks with custom HIP kernels — benchmarking every change against a never-regressing baseline. The end goal is an independently benchmarked gfx1100-tuned inference path with published performance/VRAM results.

## Core Value

Beat stock by ≥10% pp+tg on Windows ≤2 langs, N=10/15× averaged — **stock llama.cpp HIP by ≥10% pp+tg** on RX 7900 XT (gfx1100) with a custom gfx1100 kernel **that builds & runs natively on Windows 11 via HIP SDK (≤2 language runtimes, no Python/JS servers)** vs stock, while preserving model output quality — **measured `N≥10` (LLM QA `N≥15`) averaged, reproducible, bisectable** across `pp` (prefill) and `tg` (decode) at `{512,1024,2048,4096,8192}` (see REQ-PERF-07 / REQ-STAT-07 / REQ-WIN-07).

## Requirements

### Validated

- [x] Reproducible ROCm/HIP environment on RX 7900 XT (gfx1100) under WSL2, with recorded tool versions (ENV-01..04, Phase 1)
- [x] Stock llama.cpp HIP build targeting gfx1100 as permanent reference baseline (ENV-02, Phase 1)
- [x] Baseline benchmark suite for Qwen3.8-27B IQ4_XS covering prompt tok/s, generation tok/s, and VRAM breakdown with RSS guard (BENCH-01..04, Phase 2)
- [x] Fixed quality evaluation set (perplexity + deterministic prompt canary store + test-backend-ops) to guard numerical correctness (QUAL-01..02, Phase 3)
- [x] Profiling of real workloads producing a ranked bottleneck table (kernel → % runtime → bound type naming MUL_MAT Target #1) (PROF-01..02, Phase 3)
- [x] Standalone HIP kernel playground with CPU reference → HIP implementation → numerical comparison → microbenchmark pipeline with negative test (KERN-01, Phase 4)
- [x] First custom kernel attacking Optimization Target #1 (`MUL_MAT` IQ4_XS) — GEMV 1.26–2.13× / GEMM 1.7–7.5× vs stock, cosine 1.0 (KERN-02, KERN-03, Phase 5 completed 2026-08-25)
- [x] Runtime integration behind a switchable flag (custom kernels ON/OFF) without destroying the baseline (INTEG-01, Phase 6 completed 2026-08-25)
- [x] End-to-end before/after benchmark results published against stock with release hygiene and v1.0.0-gfx1100 tag (PUB-01, Phase 6 completed 2026-08-25)

### Active

- [ ] **Phase 7 must-haves (owner 2026-08-28 + deep-research report):**
  - **Windows-native, ≤2 language runtimes** — builds & runs on Windows 11 via HIP SDK + VS Build Tools with pure C++/HIP + minimal CMake + `.bat` (no Python/JS `3+` language servers; see REQ-WIN-07, Phase 7 Plan 07-04 + Phase 8).
  - **≥10% end-to-end uplift vs stock llama.cpp** — `llama-bench` `pp/tg` at `{512,1024,2048,4096,8192}` shows custom ≥1.10× stock (both `pp` and `tg`), not just microbench vs naive (REQ-PERF-07).
  - **Statistical rigour: 10× avg (15× for LLM QA)** — every perf/quality test in Phase 7 is `N≥10` (microbench `median`+`mean`+`stddev`) and every LLM question `N≥15` with averaged `tokens/s`, `PPL`, `canary` presented (REQ-STAT-07, BENCH-01 amended).
- [ ] Context scaling track: measured max-context ceiling per backend (32k→256k ladder) with KV quantization quality gates, host-tiered KV prototype for ≥128k under WSL2, and long-context prefill strategy — see REQUIREMENTS.md CTX-01..05

### Out of Scope

- Complete inference engine from scratch — project builds on llama.cpp by explicit decision
- Turkish output quality as an evaluation dimension — dropped by user decision
- Multi-GPU support — stretch goal only, not v1
- Custom sampler, speculative decoding, custom GGUF format, persistent-kernel scheduling — Phase 18 stretch goals; only after core milestones
- ~~Windows-native ROCm compute stack — dev/build happens in WSL2~~ → **IN SCOPE as of 2026-08-28**: Windows 11 HIP SDK native is now a Phase 7 must-have (REQ-WIN-07); WSL2 remains for profiling fallback only (see 08-CONTEXT.md).

### Added 2026-08-22

- **Context scaling to 256k** (owner request): in scope as a tracked requirement family (CTX-*), not a v1 exit criterion until CTX-01/02 establish the memory/quality budget. Rationale and math in `.planning/research/CONTEXT-SCALING.md`.

## Context

- Full original planning document preserved at `.planning/reference/ROADMAP-original.md` — 18 phases, 7 suggested milestones, profiling-first methodology, and project rules (benchmark before optimizing, one optimization at a time, keep stock baseline forever, test prefill and decode separately).
- Hardware: AMD Radeon RX 7900 XT, RDNA3, LLVM target `gfx1100`, 20 GB VRAM. ROCm docs list it as supported (Runtime + HIP SDK); llama.cpp HIP build has an explicit gfx1100 example.
- Reference runtime: llama.cpp ggml HIP backend reuses many CUDA kernel sources through HIP and links hipBLAS/rocBLAS — useful as both reference implementation and patch target.
- Model LOCKED: `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` → **Qwen3.8-27B-Uncensored-IQ4_XS.gguf** (15.31 GB, sha256 53adc4bb…, imatrix-quantized, MTP head retained) — chosen over Q4_K_M for context headroom on the 20 GB card. Full decision record: `.planning/research/MODEL-DECISION.md`. Base model is official `Qwen/Qwen3.8-27B` (earlier "model does not exist" research notes are superseded). Weights not yet downloaded; fetch at Phase-1 execution start with sha256 verification.
- Dev environment: WSL2 on the user's Windows machine (Hyper-V available). ROCm compute/profiling tooling is Linux-first; WSL2 GPU passthrough is the chosen path. Environment feasibility (rocminfo/rocprof inside WSL2) is itself a Phase 1 validation gate.
- Architecture (corrects original roadmap assumptions): hybrid linear attention — 48 Gated DeltaNet layers + 16 gated full-attention layers (qwen35 arch), tiny KV cache (~64 KiB/token est.), native MTP speculative-decoding head. Consequences: Phase 9 (KV cache) de-prioritized; Gated DeltaNet HIP/gfx1100 kernel coverage is the critical Phase-1 unknown and the prime custom-kernel frontier; spec-decode (MTP) becomes a benchmark dimension and decode-path accelerator.
- Key technical tension to respect: decode runs at M≈1 while prefill has large M — separate prefill/decode kernel paths are a design goal, not an afterthought.
- Success criteria from original doc: no correctness regression beyond tolerance, measurable speedup or VRAM reduction enabling larger context, publishable complete result matrix (not just best number).

## Constraints

- **Hardware**: single RX 7900 XT, 20 GB VRAM — locked IQ4_XS artifact is 15.31 GB, leaving ~4–5 GB for KV + buffers (hybrid arch KV ≈64 KiB/token f16 est.)
- **Tech stack**: Windows HIP SDK native (`HIP_PATH` + `clang++.exe --offload-arch=gfx1100` + Ninja + `build_windows.bat`) as **primary** build/run path (REQ-WIN-07, ≤2 langs); WSL2 + ROCm/HIP on Linux as **profiling fallback only**; llama.cpp as reference runtime, HIP kernels compiled for gfx1100
- **Methodology**: profile before optimizing; every optimization switchable; correctness tests next to every kernel; record compiler/ROCm/driver versions with every result; statistical rigour `N≥10` averaged (`median`+`mean`+`stddev`+`p95`) and LLM QA `N≥15` averaged (`avg tok/s`+per-run table) — single-run claims banned (REQ-STAT-07 / BENCH-01 amended; REQ-PERF-07 `≥1.10× pp+tg` at `{512,1024,2048,4096,8192}`)
- **Execution Rule**: MANDATORY TIMEOUTS + STEP-UP + statistical rigour —
  1. EVERY bash command and subprocess execution MUST specify an explicit, bounded timeout (e.g. 60–90s, max 300s).
  2. Device pre-flight (`rocminfo`) required before heavy runs to guard against DXG deadlocks.
  3. Step-up verification (CPU `-ngl 0` → partial GPU `-ngl 10` → full GPU `-ngl 99`) for all new binaries to prevent silent TDR stalls and WSL memory thrashing.
  4. Statistical rigour `N=10` (`N=15` LLM QA) with `median`/`mean`/`stddev`/`p95` and `mean−1σ ≥1.10×` gate — `N=1` claims rejected (REQ-STAT-07).
- **Environment risk**: WSL2 ROCm support validated (ROCm 7.2.1) but WSL2 is now profiling fallback only — Windows HIP SDK native (`HIP_PATH`) is primary (REQ-WIN-07); if Windows HIP SDK probe fails, WSL2 remains for bare-metal `N=10` re-bench

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build on llama.cpp instead of custom engine | Reuse working HIP backend; replace one bottleneck at a time; always have a baseline | ✓ Done (frozen @ bb4caa75) |
| Dev environment: WSL2 (Hyper-V available) | User's machine is Windows; ROCm tooling is Linux-first | ✓ Done (ROCm 7.2.1 cleared) |
| Drop Turkish output quality evals | User decision — not part of actual workload | ✓ Done |
| Archive original ROADMAP.md to .planning/reference/ | Preserved as source material; GSD roadmap merges its phases/milestones | ✓ Done |
| Primary artifact: JonathanColetti IQ4_XS (15.31 GB), context-headroom rationale | User locked 2026-08-21 after repo comparison vs HauhauCS Aggressive (patched-runtime/custom-quant variants rejected for baseline contamination) | ✓ Done (sha256 53adc4bb…) |
| GSD roadmap merges original 18 phases + 7 milestones | User explicitly requested merge, not replacement | ✓ Done |
| 2026-08-28 Re-scope Phase 7 — Windows/10%/10× + high-yield LDS/XOR/P=4/b128/swizzle/LUT/Smooth | Deep-research `1000t@8k` cliff + 3 PDFs + high-yield keywords synthesis (800 GB/s roof, KV≈128 KiB/tok, WSL2 DXG jitter/lie, rocprof blind) → 3 new must-haves REQ-WIN-07/REQ-PERF-07/REQ-STAT-07; high-yield levers: `+33` vs `XOR preshuffle` LDS, `P=4` pipeline + `sched_barrier`, `b128`/`float4` coalescing, `16×64` swizzle, `LUT μ=4`, `SmoothQuant α=0.5` | RE-SCOPED (07-RESEARCH.md 65K HIGH confidence, 07-01..07-04 re-planned wave 1→2‖2→3) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

### 2026-08-28/29 Re-scope Note

- **07-RESEARCH.md 65K HIGH confidence** (65,750 bytes, HIGH confidence per `07-RESEARCH.md:2`; 40+ URLs, 28 slices) — deep-research `1000t@8k` cliff + 3 PDFs + high-yield keywords synthesis underpinning the re-scope.
- **07-01..07-04 re-planned wave 1→2‖2→3** — 07-01 (real DP4A comparator, `N=10`) wave 1, 07-02‖07-03 (GEMV DP4A ‖ WMMA GEMM) wave 2 parallel, 07-04 (quilt + paired A/B + Windows gate) wave 3; amended with `N=10`/`N=15` rigour, `≥1.10× pp+tg` gate, and Windows `build_windows.bat` gate (see `07-CONTEXT.md` Re-scope 2026-08-28 + `07-RESEARCH.md`).

---
*Last updated: 2026-08-29 — RE-SCOPED Phase 7 (Windows ≤2 langs REQ-WIN-07, ≥10% REQ-PERF-07, 10×/15× REQ-STAT-07)*
