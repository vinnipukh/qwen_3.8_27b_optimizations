# Qwen3.8-27B on RX 7900 XT — gfx1100 Inference Optimization

## What This Is

A reproducible GPU-specific inference optimization project for running Qwen3.8-27B on an AMD Radeon RX 7900 XT (RDNA3 / gfx1100, 20 GB VRAM). It starts from stock llama.cpp with the HIP/ROCm backend and progressively replaces verified hot-path bottlenecks with custom HIP kernels — benchmarking every change against a never-regressing baseline. The end goal is an independently benchmarked gfx1100-tuned inference path with published performance/VRAM results.

## Core Value

Beat stock llama.cpp HIP on at least one important Qwen3.8-27B workload on the RX 7900 XT with a custom gfx1100 kernel, while preserving model output quality within agreed numerical tolerance — measured, reproducible, and bisectable.

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

- [ ] Context scaling track: measured max-context ceiling per backend (32k→256k ladder) with KV quantization quality gates, host-tiered KV prototype for ≥128k under WSL2, and long-context prefill strategy — see REQUIREMENTS.md CTX-01..05

### Out of Scope

- Complete inference engine from scratch — project builds on llama.cpp by explicit decision
- Turkish output quality as an evaluation dimension — dropped by user decision
- Multi-GPU support — stretch goal only, not v1
- Custom sampler, speculative decoding, custom GGUF format, persistent-kernel scheduling — Phase 18 stretch goals; only after core milestones
- Windows-native ROCm compute stack — dev/build happens in WSL2

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
- **Tech stack**: ROCm/HIP on Linux (WSL2), llama.cpp as reference runtime, HIP kernels compiled for gfx1100
- **Methodology**: profile before optimizing; every optimization switchable; correctness tests next to every kernel; record compiler/ROCm/driver versions with every result
- **Execution Rule**: MANDATORY TIMEOUTS & STEP-UP DISCIPLINE —
  1. EVERY bash command and subprocess execution MUST specify an explicit, bounded timeout (e.g. 60–90s, max 300s).
  2. Device pre-flight (`rocminfo`) required before heavy runs to guard against DXG deadlocks.
  3. Step-up verification (CPU `-ngl 0` → partial GPU `-ngl 10` → full GPU `-ngl 99`) for all new binaries to prevent silent TDR stalls and WSL memory thrashing.
- **Environment risk**: WSL2 ROCm support must be validated first — if passthrough/profiling tools fail, fall back to native Linux before any kernel work

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build on llama.cpp instead of custom engine | Reuse working HIP backend; replace one bottleneck at a time; always have a baseline | ✓ Done (frozen @ bb4caa75) |
| Dev environment: WSL2 (Hyper-V available) | User's machine is Windows; ROCm tooling is Linux-first | ✓ Done (ROCm 7.2.1 cleared) |
| Drop Turkish output quality evals | User decision — not part of actual workload | ✓ Done |
| Archive original ROADMAP.md to .planning/reference/ | Preserved as source material; GSD roadmap merges its phases/milestones | ✓ Done |
| Primary artifact: JonathanColetti IQ4_XS (15.31 GB), context-headroom rationale | User locked 2026-08-21 after repo comparison vs HauhauCS Aggressive (patched-runtime/custom-quant variants rejected for baseline contamination) | ✓ Done (sha256 53adc4bb…) |
| GSD roadmap merges original 18 phases + 7 milestones | User explicitly requested merge, not replacement | ✓ Done |

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

---
*Last updated: 2026-08-21 after initialization*
