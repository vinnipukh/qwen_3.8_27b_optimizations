
########## undefined ##########
All six bets cross-examined against the four scrapes plus live sources. Findings below.

## VERDICTS

- BET: ROCm 7.2.1 guest via librocdxg works production-grade on Adrenalin 26.2.2+ under WSL2 for gfx1100 | VERDICT: CONFIRMED | KEY-EVIDENCE: raw/scrapes/06-amd-wsl-howto.md ("production support … Adrenalin 26.2.2 + ROCm 7.2.1") + raw/scrapes/07-librocdxg.md (compat matrix lists RX 7900 XT; expected `rocminfo` output is literally gfx1100; repo active, v1.2.2 Aug 2026) | IMPLICATION: none for Phase 1 kill-gate design; but "production-grade" excludes profiler/debugger and full amd-smi — ENV gates unaffected.
- BET: rocprofv3 over /dev/dxg yields usable per-kernel timings on gfx11 (PROF-01 ladder) | VERDICT: PARTIAL | KEY-EVIDENCE: https://github.com/ROCm/rocm-systems/pull/7016 (WSL2 dxg profiling support — still **open/unmerged**, targets "RDNA 3.5", last updated 2026-06-19) vs raw/scrapes/07-librocdxg.md ("Debugging/Profiling: ROCm-profiler, Debugger are not supported") | IMPLICATION: stock ROCm 7.2.1 almost certainly lands on rung (b) llama.cpp timers; only ≥1 HW counter requirement is at risk — pre-decide whether counter-less rung-b satisfies PROF-01 or triggers contingency; also fix ROADMAP wording that PR #7016 is "newly shipped" — it is not merged. Note REQUIREMENTS.md calls PROF-01 a "Phase-1 exit criterion" while ROADMAP binds it in Phase 3 — reconcile before planning.
- BET: guest rocm-smi/amd-smi non-functional; Windows telemetry mandatory | VERDICT: CONFIRMED (minor overstatement) | KEY-EVIDENCE: https://github.com/ROCm/librocdxg/issues/6 (amdsmi unimplemented) + https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.2/docs/limitations.html (rocm-smi unsupported, UKI architecture) | IMPLICATION: none — but amd-smi is emerging as "limited features" in librocdxg develop branch (+ community amd-smi-wsl shim); treat any future guest telemetry as untrusted until validated, keep HWiNFO/Adrenalin CSV pipeline primary.
- BET: DXG path loses ~1.5–3 GB VRAM vs native | VERDICT: CONFIRMED (upper end) | KEY-EVIDENCE: raw/scrapes/05-dxg-vram-loss.md (24,136→21,191 MiB = −2.9 GB on RX 7900 XTX, ROCm 7.2.4) | IMPLICATION: on the 20 GB card assume ~17 GB usable ⇒ MODEL-DECISION's 18.3–18.8 GB @32k estimate does NOT fit fully resident; Phase 1 must empirically measure free VRAM and Phase 2 should pre-plan the 32k BENCH-04 tier as likely partial-offload or cap context (e.g., 24k) with rationale recorded.
- BET: silent VRAM-overcommit spill to system RAM occurs; RSS guards defeat it; WSL#40732 severity | VERDICT: PARTIAL | KEY-EVIDENCE: raw/scrapes/04-wsl-oom-panic.md (consecutive GPU OOM → Hyper-V kernel panic → Windows BSOD; **NVIDIA RTX 5090/CUDA**, issue still open, logs attached) | IMPLICATION: BSOD severity is real on the shared GPU-PV layer but unproven for AMD/dxg; direct spill-collapse evidence for ROCm is thin. Keep RSS guards (cheap); ADD a fail-fast OOM policy to harness plans — never loop allocation retries (that pattern is the panic trigger); synthetic overcommit test from BENCH-03 should run supervised, never unattended.
- BET: driver↔ROCm pairing fragility under Windows auto-updates | VERDICT: PARTIAL | KEY-EVIDENCE: raw/scrapes/06-amd-wsl-howto.md (AMD claims "no version-pair constraints… no risk of driver update silently breaking") CONTRA raw/scrapes/05-dxg-vram-loss.md (fragility warning re Adrenalin >26.5.2) + 07-librocdxg.md versioned compat matrix + HSA_ENABLE_DXG_DETECTION semantics changing across ROCk releases | IMPLICATION: decoupling is real but not absolute — keep pause-auto-updates + pinned pair + Phase-6 re-run of env version gate exactly as planned.

## NEW-RISKS (not in roadmap)

- librocdxg#60 (https://github.com/ROCm/librocdxg/issues/60): ROCr AsyncEventsLoop busy-spins ~2 CPU cores for the lifetime of ANY GPU context under WSL2 — reproduced on RX 7900 XTX, Adrenalin 26.6.1, ROCm 7.2.4. Steals CPU from llama.cpp threads and may distort decode-latency/timing measurements; fingerprint it in Phase 2, consider taskset isolation.
- PR #7016 merge status is a schedule dependency: if it merges into a later ROCm only, HW-counter profiling means upgrading the guest stack mid-project → re-run full env gate (ties into bet-6 discipline).
- librocdxg VRAM-mapping bug class exists (https://github.com/ROCm/ROCm/issues/6022, Strix Halo: pool size bound by .wslconfig RAM instead of VRAM) — verify `.wslconfig` memory settings don't clamp the dGPU ROCm pool in Phase 1.
- Pinned host memory is disabled-by-default under WSL (librocdxg README, via vLLM UVA failure note) — probably harmless for fully-offloaded single-GPU llama.cpp, but log it as a known deviation if any host-pinning warnings appear.
- Related unfixed GPU-PV overhead issues (WSL#40401, Blackwell 16 GiB driver overhead) show the VM layer itself carries per-vendor surprises; treat any "only N GB usable" report as platform-specific until measured locally.

## CONFIDENCE-NOTES

VRAM-deficit rests on ONE documented data point (XTX/24 GB, ROCm 7.2.4); scaling to 20 GB card and to the pinned 7.2.1 pairing is extrapolated — Phase 1 empirical probe is authoritative. Spill-to-RAM collapse (5–10× figure) lacks a direct ROCm/dxg citation — treat magnitude as hypothesis, guard anyway. PR #7016 status (open) is from search metadata dated 2026-06-19; re-check at Phase 3 start. AMD's "no coupling" claim vs community fragility reports conflict; both are marketing-vs-field, resolved cheaply by the pin-and-freeze policy. WSL#40732 is NVIDIA-only evidence applied to a shared layer — directionally valid, not proof.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Cross-examination delivered inline: 6 bets verdicts (CONFIRMED x3, PARTIAL x3) each keyed to specific files (raw/scrapes/04,05,06,07) or URLs (rocm-systems#7016, librocdxg#6/#60, ROCm#6022, AMD limitations docs), plus 5 new risks and residual-confidence notes; no file writes performed per contract"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read EVIDENCE-INDEX.md, scrapes 04/05/06/07, ROADMAP.md, REQUIREMENTS.md, MODEL-DECISION.md",
      "result": "passed",
      "summary": "All 7 required inputs loaded successfully"
    },
    {
      "command": "web_search: rocprofiler-sdk dxg PR status + librocdxg tracker + amd-smi/rocm-smi WSL limitations",
      "result": "passed",
      "summary": "Gap-fill confirmed: PR #7016 open/unmerged; amd-smi unimplemented (issue #6); rocm-smi architecturally unsupported; new busy-spin issue #60"
    }
  ],
  "validationOutput": [
    "Every roadmap bet mapped to at least one scrape-file path or external URL",
    "Verdict distribution: 3 CONFIRMED / 3 PARTIAL / 0 CONTRADICTED / 0 UNCERTAIN"
  ],
  "residualRisks": [
    "DXG VRAM deficit has a single data point (RX 7900 XTX); 20 GB-card value unmeasured until Phase 1",
    "Spill-to-RAM 5-10x collapse magnitude unverified on ROCm/dxg; guard retained as cheap insurance",
    "PR #7016 merge timeline unknown; PROF-01 counter coverage may require mid-project guest-stack upgrade"
  ],
  "noStagedFiles": true,
  "diffSummary": "No files changed - analysis-only deliverable returned inline per output_contract",
  "reviewFindings": [
    "blocker: none",
    "major: .planning/ROADMAP.md Phase 3 states PR #7016 is 'newly shipped' but it is open/unmerged (https://github.com/ROCm/rocm-systems/pull/7016)",
    "major: .planning/REQUIREMENTS.md PROF-01 labels itself 'Phase-1 exit criterion' while ROADMAP binds PROF-01 gate in Phase 3 - reconcile",
    "moderate: ROADMAP BENCH-04 32k-context tier likely infeasible fully-resident given ~17 GB post-DXG budget vs 15.31 GB weights + ~2 GB KV@32k + buffers",
    "minor: ROADMAP claim 'guest rocm-smi/amd-smi do not work' slightly overstated - amd-smi has limited-feature support emerging in librocdxg develop"
  ],
  "manualNotes": "Verdicts, new risks, and confidence notes are in the response body above (contract forbids writing files). Highest-value roadmap edits: (1) pre-authorize counter-less rung-(b) outcome for PROF-01, (2) add fail-fast OOM policy to BENCH-03 plans citing WSL#40732 BSOD pattern, (3) plan 32k-tier fallback before Phase 2 execution.",
  "notes": ""
}
```

########## undefined ##########
All four bets now have solid evidence. Composing final cross-examination report.

## VERDICTS

**BET 1 — ENV-03: Gated DeltaNet + gated attention run fully on HIP/gfx1100, zero CPU fallback** | VERDICT: **CONFIRMED** (issue does not contradict it) | KEY-EVIDENCE: `raw/scrapes/01-issue20651.md` (llama.cpp #20651) + github.com/ggml-org/llama.cpp/issues/20354 | IMPLICATION: #20651 was a **Vulkan multi-GPU** (XTX+RTX4080S pipeline-split) bug — single-GPU XTX was coherent even on the broken build, and OP confirmed fix via PR #20518 in build b8394 (Mar 18, 2026). Separately, #20354 states the HIP path cross-compiles the fused `GGML_OP_GATED_DELTA_NET` CUDA kernel and runs it **on GPU with no CPU fallback**; a third-party recipe (smeltcore.com) and buun-llama-cpp PR #20 both run Qwen3.5/3.8-27B end-to-end on gfx1100 HIP. Phase 1 unchanged; just ensure pinned commit ≥ b8394 lineage and keep the startup-log fallback check + golden-output gates (QUAL-02) — they're empirically motivated: #20598/#20545 show fresh qwen35 code still regresses per-build.

**BET 2 — HIP on RDNA3 can be slower than Vulkan** | VERDICT: **PARTIAL** (true in general, workload-dependent, possibly reversed for dense 27B) | KEY-EVIDENCE: github.com/ggml-org/llama.cpp/issues/20934 + aguyintech.com/rocm-vs-vulkan-on-the-rx-7900-xtx/ + disc15021 lines 2152, 2693, 2869 | IMPLICATION: #20934 (open, gfx1100, Mar 2026, extensive testing) shows Vulkan tg128 ~167–177 vs ROCm ~129–144 (−15–20%) on LLaMA-7B, no workaround found (ROCm stuck wave32, VMM off). BUT aguyintech measured dense Qwen3.6-27B **favoring ROCm** (34.6 vs 28.1 t/s; MoE favored Vulkan), disc15021 shows rocWMMA FATTN giving large PP wins on RDNA3 HIP, and gfx1201/gfx1151 head-to-heads show −8…−25% tg gaps narrowing with model size. Phase 2/6 should add a cheap **Vulkan comparator arm** to BENCH-04 (currently HIP-only) so results are contextualizable; do not assume HIP dominance for any workload shape.

**BET 3 — MTP/spec-decode works on RDNA3/HIP (~75 tok/s on 7900 XTX)** | VERDICT: **CONFIRMED with material caveats** | KEY-EVIDENCE: reddit.com/r/ROCm/comments/1t72ul7 + github.com/ggml-org/llama.cpp/pull/22673 + issues/23244 | IMPLICATION: MTP support is merged upstream (#22673, `--spec-type draft-mtp`, May 2026, ships in ROCm 7.2 release binaries — backend-agnostic); 75 tok/s on XTX is real but used a repo-indicated custom branch + Q4_K_M. Hard caveats: #23244 "Always OOM with mtp model + rocm" on 7900 XTX; lcz.me reports ROCm+MTP VRAM surging ~5 GB at conversation start (max 8k ctx); Strix Halo data point: MTP −15~−21% on ROCm vs Vulkan. v2 deferral is correct; when attempted, budget several GB extra VRAM, cap context short, and benchmark stock-PR first.

**BET 4 — IQ4_XS 15.31 GB fits 20 GB card with ~64 KiB/token tiny-KV assumption** | VERDICT: **CONFIRMED** (math exact; fit caveat) | KEY-EVIDENCE: `raw/scrapes/01-issue20651.md` runtime log + `.planning/research/MODEL-DECISION.md` | IMPLICATION: KV math verified two ways — config (16 full-attn layers × 4 KV heads × 256K + 256V × f16 = 32+32 KiB/token) and empirically: scrape-01 log shows KV 1008 MiB / 16128 cells = **exactly 64 KiB/cell** (K 504 + V 504 MiB f16). Fit: weights 14.26 GiB + KV@32k 2.0 GiB + fixed recurrent state ~0.14–0.6 GiB + compute ~1–1.5 GiB ≈ 17.9–19.3 GiB — fits native Linux tightly, but under WSL2 the DXG deficit (~1.5–3 GB, scrapes 05/AI-weekly) makes **32k likely OOM/silent-overcommit and 16k tighter than "comfortable"**. BENCH-04's 32k arm needs a pre-flight empirical free-VRAM probe and an expected-FAIL path under WSL2 (RSS guard already covers detection).

## NEW-RISKS (not in roadmap)

- **qwen35-specific HIP prefill collapse**: #20218 reports severe pp degradation on HIP/ROCm vs Vulkan at all context depths (GGML_HIP_GRAPHS=ON partially restores) — threatens Phase-3 bottleneck ranking assumptions and strengthens the case for testing that flag in Phase 1/2.
- **Stock HIP GDN kernel known-slow with identified root cause**: #20354 (gfx1151) + buun-llama-cpp PR #20: VGPR spilling from 2-blocks/SM occupancy; `__launch_bounds__` fix gave **+24% decode** on 7900 XTX. Confirms Gated DeltaNet scan as prime Phase-5 target and hands us a concrete first hypothesis — but also means "beat stock" may be easy while absolute numbers stay behind Vulkan.
- **WSL2/Windows ROCm stability churn on qwen35**: #20545 (infinite wait, WSL2+ROCm 7.2, 35B-A3B), #20598 (Windows segfault regression b8354) — reinforces Phase-1 kill-gate and snapshot discipline.
- **Vulkan comparator validity**: #20354 says Vulkan lacked a GATED_DELTA_NET compute shader (CPU fallback) while the later b8373 log in scrape 01 shows fused GDN "enabled" on Vulkan — support landed mid-window; a Vulkan arm must re-verify GDN coverage at its own pinned commit or its numbers are apples-to-oranges.

## CONFIDENCE-NOTES

Evidence is strong on correctness/fallback (multiple independent end-to-end gfx1100 HIP reports) and on KV math (exact runtime-log match). Thin/conflicting elsewhere: (a) HIP-vs-Vulkan gap magnitudes vary −25%…+23% by model, arch, ROCm version, and flags — no number transfers cleanly to qwen3.8-27B hybrid on a 7900 XT; treat all as directional until Phase 2 measures both backends; (b) #20651's root cause is inferred from a regression window + fix-by-PR, never formally diagnosed, and the issue sat "bug-unconfirmed"; (c) IQ4_XS quality rests on publisher-reported PPL (7.1583±0.25 vs f16 7.1557) not yet independently reproduced — QUAL-02 is the real gate; (d) MTP figures come from custom-branch/pre-merge-era posts; stock-MTP-on-HIP throughput is unquantified; (e) #20292 ("Qwen 3.5 CPU bound on rocm", gfx1201, b8235) directly conflicts with #20354's "no CPU fallback on HIP" — likely version-dependent; resolve empirically at the pinned commit via startup logs.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Concrete findings delivered above with file paths (.planning/research/deep-research/raw/scrapes/01-issue20651.md, 03-disc15021.md lines 87-121/2152/2693/2820-2869, EVIDENCE-INDEX.md, ROADMAP.md, REQUIREMENTS.md, MODEL-DECISION.md), corroborating URLs (llama.cpp issues #20651, #20934, #20354, #20218, #23244, PR #22673, aguyintech.com, reddit 1t72ul7), severity via verdicts (CONFIRMED/PARTIAL), and implications per phase (Phase 1 pin >=b8394, Phase 2/6 Vulkan comparator arm + GGML_HIP_GRAPHS=ON probe, BENCH-04 32k expected-fail path, v2 MTP VRAM budgeting)."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read .planning/research/deep-research/EVIDENCE-INDEX.md; read scrapes 01,02,03,10,11,12; grep targeted sections of 03-disc15021.md",
      "result": "passed",
      "summary": "Evidence pack + project docs loaded; disc15021 scoreboard, HIP-vs-Vulkan head-to-heads, and rocWMMA sections extracted"
    },
    {
      "command": "web_search (exa x4, brave attempt failed-no-key) + fetch_content answer-mode on issue #20934 and aguyintech.com",
      "result": "passed",
      "summary": "Corroborated HIP<Vulkan tg gap on gfx1100 (#20934: 167-177 vs 129-144 t/s), dense-model reversal, MTP merge status and OOM caveats"
    }
  ],
  "validationOutput": [
    "KV math verified exactly: 16 layers x 4 KV heads x 256(K)+256(V) x 2B = 64 KiB/token; matches scrape-01 runtime log (1008 MiB / 16128 cells, K 504 + V 504 MiB f16)",
    "VRAM envelope recomputed: ~17.9-19.3 GiB @32k vs 20 GiB card minus WSL2 DXG deficit -> 32k marginal/unlikely under WSL2, <=16k feasible"
  ],
  "residualRisks": [
    "HIP-vs-Vulkan deltas are model/version/flag-dependent; no transferable number for qwen3.8-27B hybrid on 7900 XT until dual-backend measurement",
    "#20292 'CPU bound on rocm' conflicts with #20354 'no CPU fallback on HIP'; must be resolved at the pinned commit via startup logs",
    "IQ4_XS quality attested only by publisher PPL until QUAL-02 reproduces it",
    "Stock MTP-on-HIP throughput unquantified (all public numbers predate or bypass PR #22673)"
  ],
  "noStagedFiles": true,
  "diffSummary": "No files written or modified (analysis-only task per output contract)",
  "reviewFindings": [
    "no blockers",
    "finding: ROADMAP BENCH-04 matrix lacks a Vulkan comparator arm despite documented HIP tg regression on gfx1100 (#20934) - recommend adding as optional comparator in Phase 2/6",
    "finding: MODEL-DECISION 'comfortable <=16k' understates WSL2 DXG VRAM erosion; 32k arm should carry an expected-FAIL path"
  ],
  "manualNotes": "Issue #20651 examined line-by-line: Vulkan-only multi-GPU split bug, single-GPU XTX coherent, fixed by PR #20518 (b8394); it neither implicates HIP nor single-GPU correctness. Strongest actionable intel for Phase 5: buun-llama-cpp PR #20's GDN VGPR-spill fix (+24% decode on XTX) and #20218's GGML_HIP_GRAPHS=ON partial fix for HIP prefill degradation.",
  "notes": ""
}
```

########## undefined ##########
All four bets now have primary-source evidence. Final cross-examination:

## VERDICTS

**BET 1** — BET: wave32/wave64 handling matters on gfx1100; kernels should template on warpSize | VERDICT: CONFIRMED | KEY-EVIDENCE: scrapes/08-chipsandcheese-rdna3.md + https://gpuopen.com/learn/wmma_on_rdna3/ | IMPLICATION: None for phases — but note evidence is behavioral, not quantitative: C&C shows VOPD dual-issue is wave32-centric while wave64 gets "natural" dual-issue without compiler smarts, plus FP64 halved; gpuopen documents *different fragment VGPR layouts and separate intrinsics per mode* (C/D = 8 VGPRs wave32 vs 4 wave64). rocminfo confirms gfx1100 runs Wave Size 32 by default. Template-on-warpSize (Phase 4 rule) is correct; magnitude of wave32-vs-64 GEMM delta remains unmeasured → keep the planned Phase-4 both-modes benchmark.

**BET 2** — BET: rocWMMA on RDNA3: documented API exists, known regressions when misused (−41%), realistic upside for GEMM shapes | VERDICT: PARTIAL | KEY-EVIDENCE: https://gpuopen.com/learn/wmma_on_rdna3/ (API, HIGH); .planning/research/PITFALLS.md Pitfall 7 citing llama.cpp #24437 | IMPLICATION: API + theoretical 2× FP16/clock/CU confirmed; but the −41% figure is **FlashAttention-prefill on gfx1151**, not GEMM/gfx1100 — don't cite it as a GEMM risk. Upside caveats the roadmap should carry into v2 notes: WMMA accepts only f16/bf16/iu8/iu4 contiguous inputs with fixed 16×16×16 tiles and mandatory half-wave lane replication — **no K-quant/IQ4_XS format**, so quantized GEMM needs dequant-to-fp16 first (erodes the theoretical win), and at M≈1 decode matrix cores rarely help (latency/memory-bound). Real RDNA3 FA reports (scrapes/03-disc15021.md) show large PP gains on 7900 XTX with rocWMMA ON but strongly config-dependent results on RDNA4, sometimes slower. Roadmap already defers rocWMMA to v2 flag-only — correct call; just fix the attribution of the −41% number.

**BET 3** — BET: rocBLAS coverage gaps for small-M decode GEMMs leave a custom-kernel opening | VERDICT: PARTIAL (framing wrong, opportunity real) | KEY-EVIDENCE: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu (`use_mul_mat_vec_q` gates src1->ne[1] <= MMVQ_MAX_BATCH_SIZE to custom kernels) | IMPLICATION: llama.cpp never routes small-M quantized decode through rocBLAS — it uses its own MMVQ kernels, so there is no "rocBLAS gap" to exploit at M≈1; the competitor is ggml MMVQ. Tensile coverage for gfx1100 discrete exists (TensileLibrary_lazy_gfx1100.dat, #20839 context); gaps hit gfx1103 iGPU and old ROCm. Opportunity is real elsewhere: rocBLAS leaves big margins at large-M compute-bound shapes (seb-v: +60% headroom on FP32 SGEMM), and ggml's RDNA3 launch configs were still suboptimal until recently (PR #23349). Phase 5 should frame target #1 as "beat ggml MMQ/MMVQ or rocBLAS per shape", keeping the planned rocBLAS comparison arm; drop any thought of a hipBLASLt arm (see new risks).

**BET 4** — BET: realistic speedup ranges keep Phase-5 honest | VERDICT: CONFIRMED with calibration | KEY-EVIDENCE: https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html + https://github.com/ggml-org/llama.cpp/pull/23349 | IMPLICATION: Honest ranges from real projects: ~10% tg gain from a mere MMVQ warp-count retune (upstream PR #23349, "10%+ improvement in tg speeds"); +9.8% over rocBLAS achievable in plain HIP C++ (LDS padding/CU-mode/register tiling); +60% (30.5→49 TFLOPS, 4096³ FP32) only reached with hand-edited GCN ISA, explicitly declared non-scalable/unmaintainable. Phase-5 success criteria ("beats stock") stay realistic; expectations doc should say: config-level wins ≈5–15%, heroic large-M GEMM ≈30–60%, decode M≈1 bounded by memory bandwidth (modest ceilings).

## NEW-RISKS
- **HIP compiler fragility on RDNA3**: tiny C++ changes drastically change ISA; register spills to scratch silently kill kernels; needs `launch_bounds`/`--save-temps`/RGP ISA inspection discipline (https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html). Not in roadmap — belongs in Phase 4 playground practices.
- **hipBLASLt is a dead end on gfx1100**: AMD excludes gfx archs due to incomplete TensileLite support (https://github.com/ROCm/rocm-libraries/issues/7026; distro packages lacked gfx1100 kernels entirely, https://github.com/ROCm/hipBLASLt/issues/831).
- **rocWMMA −41% evidence is off-target** (gfx1151 FATTN, not gfx1100 GEMM) — risk that v2 planning over-generalizes a real but shape/arch-specific regression (PITFALLS.md:134, llama.cpp #24437).

## CONFIDENCE-NOTES
Thin spots: (a) no public wave32-vs-wave64 GEMM microbenchmark on Navi 31 — bet-1 magnitude unknown; (b) zero public gfx1100 WMMA-vs-VALU A/B on quantized-GEMM shapes — rocWMMA upside remains speculative; (c) speedup-range calibration rests on one rigorous writeup (FP32 SGEMM, Windows ROCm 6.2.4) plus one upstream PR — nothing IQ4_XS-specific, and nothing at all on Gated DeltaNet scan kernels (novel territory, expect Phase-5 deep-study); (d) seb-v numbers predate current rocBLAS releases, so the +60% headroom may have shrunk. Doc drift noted: PITFALLS Pitfall 7 still cites "Phase 8/12", which no longer exist post-merge.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Four verdicts returned with file paths (.planning/research/deep-research/raw/scrapes/08-chipsandcheese-rdna3.md, 09-gpuopen-wmma.md, 03-disc15021.md, .planning/research/PITFALLS.md:134, STACK.md:28, ROADMAP.md Phase 4/5 risk notes) plus severity-tagged NEW-RISKS with primary-source URLs (hipBLASLt#831, rocm-libraries#7026, seb-v writeup, llama.cpp PR #23349, source-verified mul_mat dispatch)"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "grep/find over .planning/research/** (EVIDENCE-INDEX, scrapes 03/08/09, PITFALLS/STACK/SUMMARY)",
      "result": "passed",
      "summary": "Located -41% claim provenance (llama.cpp #24437, gfx1151 FATTN) and all wave-size/rocWMMA assertions"
    },
    {
      "command": "web_search x6 + fetch_content (gpuopen guide, hipBLASLt#831, llama.cpp#20839/#16799/#23349, seb-v blog, ggml-cuda.cu source)",
      "result": "passed",
      "summary": "Verified bets 1-4 against primary sources; established MMVQ-not-rocBLAS decode routing and realistic speedup ranges"
    }
  ],
  "validationOutput": [
    "Bet1 CONFIRMED, Bet2 PARTIAL (-41% is FATTN/gfx1151-specific), Bet3 PARTIAL (decode routes through ggml MMVQ, not rocBLAS), Bet4 CONFIRMED (~10% config-level, up to +60% large-M with ISA hand-tuning)"
  ],
  "residualRisks": [
    "No public wave32-vs-wave64 GEMM microbenchmark on Navi 31; mode-delta magnitude unquantified",
    "No gfx1100 WMMA-vs-VALU A/B on quantized GEMM shapes; rocWMMA upside for IQ4_XS remains speculative",
    "Gated DeltaNet scan kernel performance has zero public precedent; Phase-5 target selection could land on unresearched territory",
    "seb-v rocBLAS-headroom data predates current rocBLAS; +60% ceiling may be stale",
    ".planning/research/PITFALLS.md Pitfall 7 references retired Phase 8/12 numbering (doc drift)"
  ],
  "noStagedFiles": true,
  "diffSummary": "Read-only research task; no files created or modified",
  "reviewFindings": [
    "no blockers",
    "medium: PITFALLS.md:134 attributes -41% generically to rocWMMA misuse; evidence is FlashAttention prefill on gfx1151 (llama.cpp #24437) - reframe before v2 planning",
    "medium: ROADMAP Phase-5 'rocBLAS coverage gaps' framing slightly off - small-M decode competes against ggml MMVQ kernels (verified in ggml-cuda.cu dispatch), not rocBLAS",
    "low: hipBLASLt must not appear as a baseline/comparison arm on gfx1100 (excluded by AMD, incomplete TensileLite)"
  ],
  "manualNotes": "Evidence pack JSON result files under raw/*.json are empty shells (results n=0); EVIDENCE-INDEX.md source lists and scrapes/*.md remain the usable corpus. Verification leaned on fresh web fetches of primary sources."
}
```

########## undefined ##########
## VERDICTS

**BET:** WSL2-primary is a sound platform for a benchmark-sensitive kernel-optimization project | **VERDICT:** PARTIAL | **KEY-EVIDENCE:** scrapes/05-dxg-vram-loss.md (github.com/ggml-org/llama.cpp/issues/23999: ~3GB VRAM loss vs native Windows; throughput WSL-vs-native "entirely absent") + zenvanriel.com/ai-engineer-blog/wsl2-falls-short-local-ai-development/ (+~1GB tax) | **IMPLICATION:** Phase 1/2 fine as kill-gated, but published numbers will have an unquantified systematic offset vs every native-Linux scoreboard (e.g., disc15021 RX 7900 XT tg128=116 native). Add a one-day native-Linux A/B (dual-boot) to Phase 1 exit to measure the delta once; make the rocprofv3 feasibility probe binding, not "recommended".

**BET:** rocprofv3-over-DXG is "newly shipped" and will produce usable per-kernel times by Phase 3 (PROF-01 rung a) | **VERDICT:** CONTRADICTED | **KEY-EVIDENCE:** github.com/ROCm/rocm-systems/pull/7016 — still OPEN, titled/targeted at **RDNA 3.5 (gfx1150 iGPU)**, verified only on Radeon 890M w/ ROCm 6.4.2, requires self-built hsakmt headers ("every released ROCm package ships headers in that state, 7.x included"); scrapes/07-librocdxg.md README: "Debugging/Profiling… are not supported", known GRBM-zero-counters issue | **IMPLICATION:** Demote rung (a) to "unlikely on gfx1100"; plan bottleneck table around llama.cpp op-timers (rung b) from day one; decide early whether kernel-level attribution needs a native-Linux profiling session while e2e stays on WSL2.

**BET:** 6-phase timeline survives rocprof-dxg immaturity | **VERDICT:** PARTIAL | **KEY-EVIDENCE:** github.com/ggml-org/llama.cpp/issues/20292 — qwen35 on HIP needed `rocprof --stats` on native Linux to diagnose; without profiling, Phase 5's "attack ranked #1" degenerates to guesswork | **IMPLICATION:** Similar projects die on tooling, not kernels; if PROF-01 collapses to timers, insert a decision gate: either borrow native-Linux time for attribution or rescope Phase 5 to timer-ranked targets. Timeline itself (infra-first ordering) is well-designed.

**BET:** IQ4_XS JonathanColetti artifact is a safe locked baseline | **VERDICT:** CONFIRMED (with caveats) | **KEY-EVIDENCE:** huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF + discussions/2 (independent M3 Max report: no measurable quality loss incl. coding; MTP accept preserved). No complaints found anywhere; aguyintech.com even reports IQ4_XS beating Q4_K_M per-VRAM on both backends | **IMPLICATION:** None for phases; see NEW-RISKS for the coding-eval hole relevant to the owner's end-goal, outside this project's scope.

**BET:** Custom HIP kernels beating stock llama.cpp-HIP is realistic (HIP = right substrate) | **VERDICT:** PARTIAL | **KEY-EVIDENCE:** github.com/ggml-org/llama.cpp/issues/20934 (gfx1100: ROCm tg 129–144 vs Vulkan 167–177 t/s) + reddit.com/r/ROCm/comments/1s1vo37 + issues/20292 (DeltaNet HIP pp 10× slower, 99% kernel-dispatch overhead, ~271µs/dispatch; GGML_HIP_GRAPHS made it worse); counterpoint: aguyintech.com (dense→ROCm wins) and scrapes/03-disc15021.md scoreboard | **IMPLICATION:** Two changes: (1) BENCH-04 must add a **stock-Vulkan comparator arm** — a kernel that beats stock HIP can still lose to plain stock Vulkan on tg; (2) #20292 suggests bottleneck #1 may be dispatch/launch overhead, which a faster kernel won't fix — Phase 5 should evaluate graph-capture/fusion strategy *before* authoring, and KERN-03's "beats stock" should name which backend.

## NEW-RISKS (not in roadmap)
- **Gated DeltaNet on HIP may be dispatch-bound, not kernel-bound** — 14,977 dispatches, 99% overhead (github.com/ggml-org/llama.cpp/issues/20292, gfx1201); invalidates "write a faster scan kernel" as default Phase 5 thesis.
- **librocdxg repo deprecation-in-progress** — rocdxg moving into rocr-runtime, DxgAbiCheck removed via rocm-systems#10034 (referenced inside PR #7016); Phase 1's pinning target may shift underfoot.
- **WSL2 GPU-OOM → Hyper-V panic → host BSOD** (github.com/microsoft/WSL/issues/40732, scrapes/04-wsl-oom-panic.md): repeated allocation failures during 32k matrix sweeps can hard-crash the machine; harness needs allocation caps + crash-resilient result journal.
- **32k-context row of BENCH-04 may be physically unreachable**: 18.3–18.8GB estimate minus 1.5–3GB WSL2 VRAM deficit exceeds usable memory; verify in Phase 1 before promising the matrix.
- **Upstream optimizes RDNA3 continuously** (mmq/rocBLAS switching tunes, e.g. llama.cpp#18537; rocWMMA fattn PP gains in disc15021) — pinned-commit wins decay in relevance between Phase 5 and publication.
- **Owner end-goal (coding agent)**: artifact author concedes *no generative/code evals* ran; Heretic maintainer suspects MLP down_proj ablation — which this model used — is "far more damaging to model intelligence" (github.com/p-e-w/heretic/issues/202).

## CONFIDENCE-NOTES
- No public AMD WSL2-vs-native throughput comparison exists (05-dxg-vram-loss.md states absence); WSL "timing instability" is inferential (VRAM tax, GPU-PV crash anecdote) — NVIDIA-side reports suggest near-native compute, so contamination risk is moderate, not proven. HIGH confidence on VRAM deficit, LOW-MEDIUM on jitter claims.
- #20292 is gfx1201/RDNA4; dispatch-overhead mechanism plausibly transfers to gfx1100 but is unverified on-card.
- JonathanColetti positive-quality report is informal (n≈40 requests, MLX/Q8, Apple silicon) — not IQ4_XS-on-AMD; thin either way.
- PR #7016 status is a moving target; recheck at Phase 1 kickoff. IQ4_XS-specific AMD instability: no evidence found either direction.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "5 roadmap bets adjudicated with concrete citations to .planning/research/deep-research/raw/scrapes/ (01,03,04,05,07) plus 10 external URLs; severities carried via VERDICT labels and implications; all five assigned attack vectors covered"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "read EVIDENCE-INDEX.md, ROADMAP.md, REQUIREMENTS.md, MODEL-DECISION.md + grep/sed over raw/scrapes/*.md",
      "result": "passed",
      "summary": "Evidence pack and project docs ingested; key claims extracted from 8 scrapes"
    },
    {
      "command": "web_search x6 + fetch_content x8 (PR #7016, issues #20934/#20292/#20651/#40732, HF model card + discussion, aguyintech, zenvanriel, heretic#202)",
      "result": "passed",
      "summary": "Adversarial searches across all 5 attack vectors returned citable primary sources"
    }
  ],
  "validationOutput": [
    "Attack 1: PARTIAL - VRAM deficit confirmed (3GB/1GB sources), timing-jitter unproven, native A/B absent",
    "Attack 2: CONTRADICTED - rocprofiler-sdk DXG support is an open PR scoped to RDNA3.5, not shipped for gfx1100",
    "Attack 3: CONFIRMED - no complaints found; independent positive benchmark; caveats on code-eval absence",
    "Attack 4: PARTIAL - HIP loses to Vulkan on tg in multiple gfx1100 reports; DeltaNet dispatch overhead may be real bottleneck",
    "Attack 5: risk confirmed - zero formal coding evals on artifact; heretic maintainer flags down_proj ablation damage"
  ],
  "residualRisks": [
    "No quantitative WSL2-vs-native AMD throughput data exists; offset remains unknown until Phase 1 native A/B",
    "#20292 mechanism verified on RDNA4 only",
    "PR #7016 / librocdxg deprecation status may change before Phase 1 execution"
  ],
  "noStagedFiles": true,
  "diffSummary": "Read-only red-team analysis; no files created or modified per task contract",
  "reviewFindings": [
    "blocker: none",
    "major: ROADMAP.md Phase 3 PROF-01 rung (a) premised on 'newly shipped' DXG profiling that is actually an unmerged RDNA3.5-only PR (rocm-systems#7016)",
    "major: REQUIREMENTS.md BENCH-04 lacks a stock-Vulkan comparator arm despite public gfx1100 evidence of Vulkan tg superiority (llama.cpp#20934)",
    "moderate: ROADMAP.md Phase 5 default target assumption (kernel efficiency) contradicted by dispatch-overhead evidence (llama.cpp#20292)",
    "minor: MODEL-DECISION.md VRAM envelope omits WSL2 1.5-3GB deficit interaction with the 32k BENCH-04 row",
    "minor: librocdxg pinning in Phase 1 exposed to repo-deprecation drift (rocm-systems#10034)"
  ],
  "manualNotes": "Task instructed no file writes; all findings delivered inline above. Owner end-goal (local coding agent) depends on coding capability never measured for the locked artifact - recommend a v2 requirement for HumanEval-style eval before relying on it agentically."
}
```
