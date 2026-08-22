# FreeToken Adaptation Study — What We Steal, What We Don't

**Date:** 2026-08-22 · **Trigger:** owner directive "steal as much from that project as you can"
**Sources:** arXiv 2608.16157v1 (paper) · github.com/FlashML-org/FreeToken (code, Apache-2.0)
**Verdict:** Not adoptable as a runtime — CUDA-only (driver r580+/CUDA 13/nvcc JIT, zero ROCm paths),
MoE-specific machinery (expert offload/cache/CPU co-exec), and our dense IQ4_XS artifact fits fully
in VRAM. Adopted instead as **methodology**: benchmark rubric, bandwidth-measurement discipline,
and design patterns for the CTX-* host-tiering work.

## Already absorbed (prior session)

- DeltaNet-state checkpoint resume at semantic anchors → **CTX-05** ("FreeToken §3.1 analog").

## Now adopted (this study)

| # | FreeToken idea | Where it lands | What changes |
|---|---|---|---|
| 1 | §5 eval rubric: per-turn TTFT distribution incl. **worst-case**; decode-rate **retention** as context grows; multi-turn agentic workload shapes. Key finding: single-shot benchmarks overstate real performance (baselines lose 31%+ by W2; tails cross client timeouts at 150 s+) | **BENCH-05** (new, Phase 2 harness) | Harness gains a stability arm: multi-turn context-growth runs reporting TTFT mean + max per turn, and decode tok/s retention % vs the single-turn baseline. Final claims require it, not just pp/tg singles |
| 2 | **Measured-bandwidth discipline**: all decisions from B_P/B_H profiled *on deployed tensor shapes*, never spec sheets (their Table 1 columns are measurements, not datasheet numbers) | Methodology rule for PROF-02 interpretation + prerequisite for CTX-04 policy; adopt their Table-1 column set into `benchmarks/environment/` fingerprint (PCIe GB/s, host-stream GB/s, measured on our shapes) | Environment fingerprint gains two measured columns; any tiering/prefetch claim must cite measured bandwidths |
| 3 | **q⋆ residual-bandwidth split** (Eq. 2–4): concurrent branches balanced at q⋆ = m·B_P/B_H; GPU and CPU partial outputs merged *exactly* | **CTX-04** design option: cold-prefix attention need not be pure data movement — CPU can compute partial attention over host-resident cold KV while GPU does the hot window, exact merge preserves QUAL gates unchanged | Turns CTX-04 from "fetch everything" into a measured fetch-vs-local-compute policy; also covers prefetch-depth choice |
| 4 | **Full-layer double buffering** (prefill transfer hidden behind compute; costs 19–26% throughput when disabled) | **CTX-05** chunked-prefill: overlap next chunk's cold-KV fetch with current chunk's compute | Chunked-prefill timing curve gets a ±double-buffer comparator arm |
| 5 | **Locality-measure-first caching** (Fig 4b replay methodology: identical access traces replayed against candidate policies — LRU vs static vs prefill-frozen — at equal capacity) | **CTX-04** hot-window sizing: record KV-page access traces, replay against {recency, static-head, hybrid} window policies before committing | Window policy becomes evidence-based, not assumed |
| 6 | **Kernel cross-reference**: their `csrc/gguf/` vendors llama.cpp's `dequantize.cuh`/`mmvq.cuh`/`mmq.cuh`/`vecdotq.cuh` | Phase 3 source-study reading list; side-by-side reference for Phase 5 IQ4_XS dequant+matmul work (license-compatible) | Study aid only |
| 7 | **Graceful-degradation ladder** (pure-CPU backend when DMA/pinning unavailable) | Pattern endorsement for existing PROF-01 / CTX-04 ladders (probe pinned-memory knob → pageable fallback) | No change; confirms approach |

## Evaluated and rejected (with reasons)

- **Runtime adoption / port**: full kernel-layer rewrite (Triton-NVIDIA patterns, FlashInfer, CUDA Graph
  capture semantics, pynccl); WSL2 undermines pinned-DMA assumptions that q⋆ depends on; single 20 GB
  card cannot demonstrate the system's value proposition (frontier-scale MoE offload).
- **Expert LRU cache / expert banks / FTW format**: dense model — no experts exist.
- **MXFP4/NVFP4 paths**: RDNA3 has no FP4 hardware; irrelevant to IQ4_XS GGUF path.
- **Fast bootstrap (pin-after-fill)**: nice-to-have; out of scope (no serving daemon).
- **Elastic runtime VRAM reconfiguration**: llama.cpp cannot rebuild budgets mid-run without restart;
  nearest useful residue (budget curves informing startup `-c` choices) already covered by CTX-01.

## Honest limitations of the steal

Their numbers come from dual-channel consumer hosts + rented servers capped to emulate them; our
WSL2/DXG stack shifts every absolute number. We steal the *measurement structure*, never their
constants. All q⋆/window-policy adoptions remain hypotheses until gated through BENCH-05 /
QUAL-02 like everything else.
