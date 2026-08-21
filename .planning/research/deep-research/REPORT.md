# Deep Research: Validating the gfx1100 Qwen3.8-27B Optimization Plan

**Workflow:** firecrawl-deep-research · **Depth:** thorough · **Date:** 2026-08-21
**Method:** 9 Firecrawl search sweeps → 54 unique sources → 12 targeted scrapes (9 full-text primary sources captured) → 4 independent analyst passes (platform, model/llama.cpp, kernel feasibility, red team) cross-examining ROADMAP.md bets against evidence.
**Raw corpus:** `raw/` (search JSONs, scrapes, EVIDENCE-INDEX.md); analyst outputs: `analyst-verdicts-raw.md`

## Executive Summary

The plan's foundations survive adversarial review. The WSL2-primary platform bet is production-real (AMD's own docs + active librocdxg repo list the RX 7900 XT / gfx1100 explicitly), the locked IQ4_XS artifact checks out (KV math verified exactly against a runtime log: 64 KiB/token; zero community complaints; independent positive quality reports), and llama.cpp's HIP path demonstrably runs the hybrid qwen35 architecture fully on GPU with no CPU fallback at current builds. The custom-kernel opportunity is real but now carries calibrated expectations: config-level wins run ≈5–15%, heroic large-M GEMM work reached ≈30–60% in a rigorous third-party writeup, and decode (M≈1) is memory-bandwidth-bound with modest ceilings.

Four material corrections emerged. First and most important: **the roadmap's primary profiling assumption is contradicted** — rocprofiler-sdk's WSL2/DXG support is an *unmerged* PR scoped to RDNA 3.5 iGPUs, and librocdxg officially states profiling is unsupported; bottleneck attribution should be planned around llama.cpp op-timers from day one, with one native-Linux profiling session as the sanctioned escalation. Second, **a stock-Vulkan comparator arm is missing from BENCH-04**: multiple gfx1100 reports show Vulkan beating HIP on token generation by 15–20% for some models, so "beats stock" must always name which backend. Third, the **WSL2 VRAM deficit (~1.5–3 GB) makes BENCH-04's 32k context tier likely OOM/silent-overcommit**; it needs an empirical free-VRAM gate and an expected-fail path. Fourth, Phase 5's default thesis ("author a faster DeltaNet scan kernel") may attack the wrong layer — public evidence suggests the hybrid arch on HIP can be **dispatch/launch-overhead-bound**, so graph-capture/fusion evaluation must precede kernel authoring (though a concrete +24%-decode VGPR-spill fix from the community hands us a strong first hypothesis either way).

Net verdict: proceed, with the five amendments below folded into ROADMAP/REQUIREMENTS before Phase 1 execution.

## Key Findings

1. **Platform bet CONFIRMED** — ROCm 7.2.1 guest via librocdxg is production-supported on Adrenalin ≥26.2.2 for RX 7900 XT/gfx1100; repo active (v1.2.2, Aug 2026). [rocm.docs.amd.com howto_wsl; github.com/ROCm/librocdxg]
2. **PROF-01 rung (a) CONTRADICTED** — rocm-systems PR #7016 (dxg profiling) is open/unmerged and targets RDNA 3.5; librocdxg README: profiling/debugging "not supported." Rung (b) llama.cpp timers must become the planned baseline, not the fallback. [github.com/ROCm/rocm-systems/pull/7016; raw/scrapes/07]
3. **Guest telemetry dead CONFIRMED** — amd-smi unimplemented (librocdxg#6), rocm-smi architecturally unsupported; Windows-side CSV pipeline stays mandatory. [librocdxg/issues/6; AMD limitations doc]
4. **VRAM deficit CONFIRMED at upper bound** — measured −2.9 GB on XTX under ROCm 7.2.4 ⇒ assume ~17 GB usable on our 20 GB card until Phase 1 probes empirically; 32k ctx (weights 14.26 GiB + KV 2.0 GiB @32k + buffers) likely exceeds it. [aiweekly alert; raw/scrapes/05]
5. **ENV-03 (full-GPU hybrid execution) CONFIRMED with a pin condition** — GATED_DELTA_NET fused kernel compiles and runs on HIP without fallback; the scary "incoherent output" issue #20651 was a Vulkan multi-GPU split bug, fixed by PR #20518 (build b8394). Pin commit ≥ that lineage. [llama.cpp#20651, #20354]
6. **HIP-vs-Vulkan gap PARTIAL/real** — gfx1100 reports: Vulkan tg128 167–177 vs ROCm 129–144 t/s (7B); dense 27B-class reversed in one measurement. Direction is workload-dependent ⇒ comparator arm required, never assume. [llama.cpp#20934; aguyintech.com]
7. **MTP spec-decode on RDNA3 CONFIRMED possible** (merged upstream PR #22673; community 75 tok/s on XTX) but with OOM reports under ROCm (+~5 GB VRAM surge) — v2 deferral validated. [llama.cpp#22673, #23244; reddit 1t72ul7]
8. **Kernel-opportunity framing corrected** — small-M decode routes through ggml MMVQ kernels, *not* rocBLAS (verified in ggml-cuda.cu dispatch); there is no rocBLAS gap at M≈1. Real headroom: beat ggml MMQ/MMVQ per shape; large-M rocBLAS margins up to +60% (may have shrunk). [ggml-cuda.cu; seb-v writeup; llama.cpp#23349]
9. **Phase-5 target intel** — community VGPR-spill fix (`__launch_bounds__`) gave +24% decode on XTX for the GDN kernel; separately #20292 shows 99% dispatch-overhead (RDNA4) ⇒ evaluate GGML_HIP_GRAPHS/graph-capture first. [buun-llama-cpp#20; llama.cpp#20292, #20218]
10. **rocWMMA nuance** — API real (2× FP16/clock theoretical) but accepts only fp16/bf16/iu8/iu4 — **no IQ4_XS format**; the cited −41% regression was FlashAttention-prefill on gfx1151, not GEMM/gfx1100. v2 flag-only deferral stands; fix the citation. [gpuopen wmma_on_rdna3; llama.cpp#24437]

## Contrarian Views And Risks

- **Red team, strongest hit:** published numbers will carry an unquantified WSL2 offset vs native-Linux scoreboards (no public AMD WSL-vs-native throughput comparison exists). Mitigation: one-day native A/B during Phase 1 exit.
- **Crash severity:** repeated GPU OOM under WSL2 can panic Hyper-V → host BSOD (documented on NVIDIA; same GPU-PV layer). Harness needs fail-fast allocation caps, supervised synthetic tests, crash-resilient result journal.
- **CPU-steal risk:** librocdxg#60 — ROCr busy-spins ~2 cores per GPU context under WSL2; fingerprint and consider taskset isolation in Phase 2.
- **Moving targets:** librocdxg deprecation into rocr-runtime in progress; upstream continuously tunes RDNA3 paths (pinned-commit wins decay); driver↔ROCm decoupling claimed by AMD but contradicted by field reports — keep pin-and-freeze discipline.
- **End-goal honesty:** the locked uncensored artifact has **zero formal generative/code evals**; Heretic's maintainer suspects down_proj-ablation may cost intelligence. If the local coding agent goal matters, add a v2 HumanEval-style eval requirement.
- **Thin evidence acknowledged:** wave32-vs-64 GEMM delta unmeasured on Navi 31; no WMMA-vs-VALU data on quantized shapes; GDN scan kernels have zero public precedent (novel territory = both risk and opportunity).

## Open Questions

1. Does PR #7016 land (and cover gfx1100 dGPU) before Phase 3? Re-check at kickoff — determines HW-counter availability.
2. What is *our card's* actual DXG VRAM deficit and free-VRAM truth? (Phase 1 empirical probe.)
3. Is qwen35-on-HIP dispatch-bound on gfx1100 like #20292 shows on RDNA4? Decides Phase 5 thesis.
4. Does GGML_HIP_GRAPHS=ON restore prefill perf on our stack (#20218)? Cheap Phase-1/2 experiment.
5. Vulkan GDN coverage at our pinned commit — apples-to-apples comparator or excluded arm?

## Recommended Amendments (before roadmap approval)

| # | Amendment | Where |
|---|-----------|-------|
| A1 | Reword PROF-01: rung (b) op-timers = planned baseline; counter-less outcome pre-authorized; optional native-Linux profiling session sanctioned | ROADMAP Ph3, REQUIREMENTS |
| A2 | Add stock-Vulkan comparator arm to BENCH-04 (verify GDN coverage at its own pin); every "beats stock" claim names the backend | REQUIREMENTS BENCH-04, Ph2/Ph6 |
| A3 | BENCH-04 32k tier: empirical free-VRAM pre-flight; expected-FAIL path under WSL2 (or cap 24k with rationale) | ROADMAP Ph2 |
| A4 | Phase 5: insert graph-capture/fusion evaluation step before kernel authoring; record buun VGPR-spill fix as first hypothesis | ROADMAP Ph5 |
| A5 | Harness: fail-fast OOM policy, supervised overcommit test, crash-resilient journal; fingerprint librocdxg#60 busy-spin | ROADMAP Ph2 risk notes |
| A6 | Doc hygiene: fix "#7016 newly shipped", PROF-01 phase-attribution conflict, −41% rocWMMA citation, exclude hipBLASLt arms, note .wslconfig pool-clamp check | ROADMAP/REQUIREMENTS/PITFALLS/MODEL-DECISION |
| A7 | v2 candidate: coding-eval requirement (HumanEval-style) if local pi-agent goal is firm | REQUIREMENTS v2 |

## Sources

Full index: `EVIDENCE-INDEX.md` (54 sources). Primary full-text scrapes in `raw/scrapes/`: llama.cpp#20651 (43 KB), HIP-performance discussion #15021 (135 KB), microsoft/WSL#40732, AMD WSL how-to, librocdxg README, chipsandcheese RDNA3 microbench, gpuopen WMMA guide, AI Weekly DXG-VRAM analysis. Key external citations inline above.

## Rerun Inputs

workflow: firecrawl-deep-research
topic: stress-test gfx1100/Qwen3.8-27B WSL2 optimization roadmap assumptions
depth: thorough
output: markdown
