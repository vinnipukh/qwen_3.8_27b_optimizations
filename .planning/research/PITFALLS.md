# Pitfalls Research

**Domain:** RDNA3/gfx1100 LLM inference optimization (llama.cpp HIP/ROCm) under WSL2, 27B-class models on 20 GB VRAM
**Researched:** 2026-08-21
**Confidence:** HIGH for pitfall identification (each critical pitfall corroborated by ≥2 independent sources or by AMD/ROCm official docs); individual claims tagged MEDIUM where single-source. Orchestrator-verified session facts cited as HIGH.

> Context anchor: dev happens **inside WSL2** via the ROCDXG/librocdxg path (Adrenalin 26.2.2 + ROCm 7.2.1, Ubuntu 22.04/24.04 guests). Native Linux is contingency-only. Final model candidate is PENDING-USER between **Qwen3-32B (dense)** and **Qwen3-30B-A3B (MoE)** — both Q4_K_M variants leave **<1.5 GB** headroom on the 20 GB card. Every VRAM-related pitfall below is amplified by that fact.

---

## Critical Pitfalls

### Pitfall 1: WSL2 VRAM overcommit silently spills to system RAM instead of erroring

**What goes wrong:**
Under WSL2 with the legacy allocator, when a HIP allocation exceeds available VRAM, the runtime does **not** return an out-of-memory error — it silently spills VRAM pages to system RAM. The process keeps "working": system RAM climbs, throughput collapses (PCIe round-trips), threads freeze intermittently, and eventually the whole WSL2 VM exhausts RAM and crashes (observed on RX 7900 XTX / gfx1100, llama.cpp#22583).

**Why it happens:**
The DXG paravirtualized memory path has different overcommit semantics than native KFD. Native Linux HIP returns `hipErrorOutOfMemory`; the WSL2 path pages out instead. Developers trained on native behavior never suspect allocation succeeded-in-name-only.

**How to avoid:**
- Treat any run where RSS of the llama.cpp process grows toward model size as failed, even if tokens are produced.
- Gate every benchmark run on a memory check: process RSS ≈ model-file-size-at-most, plus known KV/buffer footprint. Log both RSS and reported VRAM each run.
- Budget conservatively: with <1.5 GB nominal headroom (Q4_K_M candidates), assume real usable headroom is smaller still (see Pitfall 3).
- Prefer explicit `-c` (context) sizing and measured KV quantization over "let it fit and see."

**Warning signs:** tg throughput drops 5–10× mid-run; `free -g` inside WSL dropping; thrashing sounds; llama.cpp logs showing normal allocations while Windows-side RAM usage explodes.

**Phase to address:** Phase 1 (environment validation gate) and Phase 9 (KV/memory budget experiments). Must be baked into the benchmark harness before any optimization claim.

---

### Pitfall 2: No GPU telemetry exists inside WSL2 — rocm-smi AND amd-smi are both unavailable

**What goes wrong:**
`rocm-smi` is architecturally unsupported under WSL (UKI limitation: no active compute processes, no GPU utilization, no modifiable state per AMD docs). Worse, **amd-smi has not been implemented in ROCDXG** either (librocdxg issue #6: "AMD SMI support has not been implemented in ROCDXG"). There is no supported way to read temperature, power draw, clocks, fan speed, or true VRAM usage from inside the WSL2 guest.

**Why it happens:**
Both tools depend on kernel interfaces (/dev/kfd-based UKI) that don't exist in the DXG paravirtualization path.

**How to avoid:**
- Build the telemetry layer **Windows-side from day one**: AMD Adrenalin overlay/API, HWiNFO64 shared-memory reader, or a small PowerShell script sampling GPU sensors, writing CSV aligned by timestamp with benchmark runs.
- The original roadmap's `benchmarks/environment/rocm-smi.txt` artifact cannot be produced as specified — substitute Windows-side sensor capture and document the substitution.
- Record clocks/temp/power for every published number (original roadmap Phase 15 requires temperature/power columns; this pitfall means those come from Windows tooling, not ROCm).

**Warning signs:** scripts calling `rocm-smi`/`amd-smi` failing with driver-init errors; someone "fixing" benchmarks by simply omitting thermal data.

**Phase to address:** Phase 1 (environment baseline). This is a hard blocker for the benchmark suite design (Phase 15) if discovered late.

---

### Pitfall 3: Free-VRAM reporting lies twice — plan budgets against measured failure, not reported numbers

**What goes wrong:**
Two independent inaccuracies stack:
1. `hipMemGetInfo` / `hsa_agent_get_info(MEMORY_AVAIL)` **over-reports free VRAM** because KFD AVAILABLE_MEMORY excludes DRM graphics allocations (desktop compositor, browser, games on the Windows host) — gap can be multiple GiB (ROCm#6389).
2. Under ROCDXG/WSL2, the DXG connector reports roughly **~3 GB less free VRAM than native Windows** sees on the same card (llama.cpp#23999, RX 7900 XTX observed). Direction and magnitude vary with driver version.

Net effect: on a card where Q4_K_M weights alone leave <1.5 GB, *any* automated decision based on reported free VRAM ("it says 4 GB free, allocate KV for 16k ctx") will be wrong, in either direction.

**Why it happens:**
Paravirtualized reporting through dxg plus host-side desktop VRAM consumption invisible to the guest.

**How to avoid:**
- Never make allocation decisions from `hipMemGetInfo` in this project. Instead: probe empirically — attempt the allocation, catch failure, back off.
- Close host-side VRAM consumers before measurement sessions; treat "free VRAM" as an unstable quantity across reboots/driver updates.
- Maintain a measured VRAM ledger per config: `weights + KV(ctx, kv-quant) + compute buffer(batch) + runtime overhead`, filled in from llama.cpp's own buffer-size log lines (it prints every backend buffer allocation), cross-checked once against Windows-side totals.

**Warning signs:** same command line OOMs after a reboot with a browser open; reported-free minus actually-allocatable differs by GiBs.

**Phase to address:** Phase 1 (record the discrepancy explicitly in environment docs) and Phase 9 (VRAM ledger methodology).

---

### Pitfall 4: Profiling under WSL2 is brand-new, minimally validated, and counter-incomplete — validate before planning around it

**What goes wrong:**
The entire roadmap's Phase 3 ("profile before optimizing") assumes working profilers. Under WSL2 there is no /dev/kfd; rocprofv3 compute profiling only recently landed via rocprofiler-sdk over /dev/dxg for gfx11-family, with limited validation. Hardware PM counters may be missing or unreliable; Radeon GPU Profiler (RGP/RGA) toolchain assumes KFD-era interfaces and may not attach. A project that discovers at Phase 3 that its bottleneck table can't be built has lost its methodology.

**Why it happens:**
ROCm profiling tooling was built for native Linux/KFD; DXG support is a recent addition shipping ahead of its ecosystem.

**How to avoid:**
- Make profiler validation an **explicit Phase 1 exit criterion**: run rocprofv3 on a trivial HIP kernel and a llama.cpp workload; verify you get per-kernel timings AND at least one hardware counter (e.g., VALU utilization, LDS bank conflict rate).
- Prepare the fallback ladder documented in advance: (a) rocprofv3/DXG counters → (b) llama.cpp built-in timing (`--timing`, per-op verbose output) + manual kernel-region timers in custom code → (c) native-Linux dual-boot contingency (already the recorded fallback).
- Do not let Phase 3 begin until (a) or (b) demonstrably produces a bottleneck table.

**Warning signs:** `rocprofv3` segfaults or produces empty CSVs; counters present but values obviously bogus (0% or >100% utilizations); RGP can't see the guest workloads.

**Phase to address:** Phase 1 gate; directly determines whether Phase 3 runs in WSL2 or triggers the native-Linux contingency.

---

### Pitfall 5: Windows Adrenalin ↔ ROCm guest version coupling — a routine driver update breaks the whole stack mid-project

**What goes wrong:**
The production ROCDXG path requires matched pairs: Adrenalin 26.2.2 driver + ROCm 7.2.1 guest. Windows aggressively auto-updates GPU drivers. A silent Adrenalin upgrade invalidates the guest ROCm install: device disappears from `rocminfo`, or works-but-corrupt results/perf appear (documented repeatedly: Adrenalin updates breaking ROCm acceleration, ROCm#4459).

**Why it happens:** the user-mode ROCDXG library ABI-matches a specific kernel-mode driver; Windows Update doesn't know WSL compute depends on it.

**How to avoid:**
- Pause driver updates via Group Policy / Adrenalin "driver-only" discipline; document pinned versions (`Adrenalin 26.2.2 + ROCm 7.2.1 + WSL kernel`) in `benchmarks/environment/` and re-record on any forced change.
- Snapshot the configured WSL distro (`wsl --export`) right after Phase 1 succeeds — instant restore after breakage.
- Re-run the Phase 1 environment gate script after ANY driver change before trusting further benchmarks.

**Warning signs:** `rocminfo` shows CPU only; `HSA_ENABLE_DXG_DETECTION=1` suddenly required again; performance shifts ±20% with no code change.

**Phase to address:** Phase 1 (version pinning + snapshot); ongoing rule for Phases 2–17.

---

### Pitfall 6: Wave-size assumptions smuggled in from CUDA (wave32 vs wave64)

**What goes wrong:**
CUDA kernels hard-code warp width 32: shuffle masks, ballot patterns, reduction loops (`for (offset=16; …)`), and tile geometry all assume 32 lanes. RDNA3 HIP defaults to wave64, while llama.cpp's ggml-CUDA/HIP layer historically assumes 32 "all over the place" (PR#11495) and its ROCm path reports `Wave Size: 32` while the **Vulkan backend on the same card uses 64** (issue#20934). Ported or newly written kernels that mix these assumptions produce wrong reductions (correctness bug) or half-efficiency shuffles (perf bug). Additionally `__AMDGCN_WAVEFRONT_SIZE` is deprecated upstream — code relying on it will break at a future ROCm bump.

**Why it happens:** muscle memory from CUDA; the same physical GPU reports different widths per backend, which reads like a lie.

**How to avoid:**
- Template custom kernels on wave size or query `warpSize` at runtime (HIP porting guide requirement); never literal 32/64.
- For wave-level primitives use `__shfl_*_sync` family sized by `warpSize`, and design reductions as `warpSize`-generic loops.
- Decide and record the target wave mode per kernel: on RDNA3, wave32 gives double the waves-per-SIMD and is usually right for latency-bound decode kernels; wave64 halves scheduling overhead for wide tiles. Benchmark both in the Phase 4 playground rather than assuming.
- Fix the compiler flag set: avoid deprecated macros; pass explicit wave size where launch bounds require.

**Warning signs:** microbench correctness checks passing at one tile shape and failing at another; perf cliff exactly at workgroup-size changes; `__shfl_xor` masks written as constants.

**Phase to address:** Phases 4–6 (kernel playground, dequant, GEMM) — establish the wave-size-generic kernel skeleton before any hot-path kernel lands.

---

### Pitfall 7: Naive WMMA / tensor-core assumptions — RDNA3 matrix cores are NOT CUDA tensor cores

**What goes wrong:**
Porting `wmma::`/`mma.sync` logic 1:1 fails: RDNA3 WMMA instructions are **wave-cooperative** (one instruction covers the whole wavefront, 16×16×16 tile, fragments distributed differently than CUDA's per-thread fragment ownership). Practical fallout already shipped upstream: llama.cpp's rocWMMA FlashAttention (`GGML_HIP_ROCWMMA_FATTN`) defaults OFF, showed a **−41% prefill regression at long context on gfx1151** (#24437), fails to compile on some targets with certain ROCm versions (#13110), and plain FA on ROCm **loses** to non-FA at batch>1 (#10439: pp 678→375, tg 169→87 @ bs16 on 7900 XTX).

**Why it happens:** marketing equivalence of "matrix cores ≈ tensor cores"; fragment layout differences are invisible until numerics/perf go sideways.

**How to avoid:**
- Treat WMMA as a Phase 8 experiment behind flags, always benchmarked against a pure-VALU (no-MMA) implementation on the exact decode/prefill shapes; never assume MMA wins on RDNA3 — at small M (decode) it often doesn't.
- Pin rocWMMA/ROCm versions together; check the rocWMMA support matrix for gfx1100 before designing around it.
- When integrating attention changes, benchmark fa on/off × bs1/bs8 separately (the regression regime is batch-dependent).

**Warning signs:** attention kernel faster at 4k prefill but slower at long context; compile errors only on certain `GPU_TARGETS`; quality gate failures from fragment misinterpretation (garbage output, not just noise).

**Phase to address:** Phase 8 (attention) primarily; flag hygiene rules belong in Phase 12 (runtime integration).

---

### Pitfall 8: rocBLAS/Tensile coverage gaps and blind library fallbacks

**What goes wrong:**
The HIP backend links hipBLAS/rocBLAS; llama.cpp calls into it for certain GEMM shapes. rocBLAS dispatches from a pre-generated TensileLibrary **per GPU architecture**: when an arch's kernels are missing, builds crash at model warm-up (gfx1103 case, #20839) — and for covered archs, the selected kernel for unusual shapes (M=1 decode GEMMs, K-quant-dequantized fp16 GEMMs) may be far from optimal. Assuming "rocBLAS handles my shapes well because gfx1100 is officially supported" is unfounded for the specific M∈{1..32} sweep this project cares about.

**Why it happens:** Tensile libraries are generated offline for anticipated shapes; consumer RDNA3 coverage prioritizes large-M training-like shapes.

**How to avoid:**
- Early (Phase 3): profile which ops actually route to rocBLAS vs ggml's own MMQ/mmv kernels for Qwen shapes; record the dispatch table.
- In the Phase 6 shape sweep, include rocBLAS/hipBLASLt as a *measured competitor*, not a fallback assumption; the roadmap already asks "when does a custom kernel beat rocBLAS?" — answer with data per (M,N,K).
- Verify TensileLibrary contains gfx1100 entries for the encountered dtypes at first build; missing-kernel symptoms are crashes at warm-up, not graceful fallback.

**Warning signs:** crash during "warmup" phase after model load; perf cliffs between adjacent M values indicating kernel-selection discontinuities.

**Phase to address:** Phases 2–3 (understand existing GPU path; profiling) and Phase 6 (GEMM sweep).

---

### Pitfall 9: llama.cpp upstream drift — regressions land between builds; unpinned baselines rot

**What goes wrong:**
llama.cpp moves fast and perf regressions land on master regularly (e.g., commit 617db24 dropped gfx1103 generation ~18→13 tok/s; ROCm-vs-Vulkan decode regressions tracked in #20934). If the stock baseline is "whatever master was this week," week-over-week comparisons conflate your optimization's effect with upstream drift, and bisection becomes impossible.

**Why it happens:** thousands of commits/month across backends; RDNA3 is not the primary CI target.

**How to avoid:**
- Pin ONE llama.cpp commit for the whole project; archive its built binaries (`baseline/` directory) permanently — the original roadmap's "keep a stock baseline forever" rule needs this concrete mechanism.
- Rebase intentionally, at phase boundaries only, re-running the full Phase 15 suite on both old and new pins before switching.
- Record build hash + ROCm + driver with every result row (already a roadmap rule; enforce in the harness so it can't be skipped).

**Warning signs:** baseline numbers drifting without local changes; issue trackers confirming a regression in your pinned range.

**Phase to address:** Phase 1/2 (baseline establishment); enforced throughout via harness.

---

### Pitfall 10: Dense-Q4_K_M-on-20GB budget trap — OOM arrives at the first long prompt, not at model load

**What goes wrong:**
Qwen3-32B Q4_K_M weights ≈19.76 GB on a 20 GB card: model load itself barely fits or fails, and even if trimmed quantizations (IQ4_XS ~18 GB, Q4_K_S ~18.77 GB) load, llama.cpp additionally allocates: KV cache (**defaulting to the model's max context**, not your usage), compute/output buffers scaling with batch×context for prefill, and ~0.8–1.2 GB runtime overhead. Community data for this exact model class: Q4_K_M wants ~22–23 GB total at 4k context. The classic failure: loads fine at short context, then the first 8k-token prompt spikes the compute buffer and OOMs — or worse, hits Pitfall 1's silent spill. Meanwhile Q5_K_M (~23.4 GB weights) categorically does not fit dense.

**Why it happens:** people budget "file size + a bit"; llama.cpp's default context allocation and prefill-time buffer growth are easy to overlook; WSL2 hides the failure mode (Pitfall 1).

**How to avoid:**
- Set `-c` explicitly, always; start at 4096 and grow only with measured headroom.
- Plan per-config ledgers: for each quant × ctx × kv-quant combination, record `weights / KV / compute / overhead` from llama.cpp log lines before benchmarking.
- Accept upfront: dense 32B ⇒ IQ4_XS/Q4_K_S + KV q8_0 + modest ctx is the realistic envelope; Q4_K_M dense likely needs CPU-offload of some layers (and then "GPU-resident" claims are void).
- Test the largest planned prompt length in Phase 1 smoke tests, not Phase 9.

**Warning signs:** load succeeds at `-c 4096` but dies at `-c 16384`; peak VRAM within 200 MB of capacity in steady state; tg degrading over a session (spill onset).

**Phase to address:** Phase 1 (model selection smoke tests — feeds the PENDING-USER dense-vs-MoE decision) and Phase 9 (KV strategy).

---

### Pitfall 11: MoE offload tuning (if 30B-A3B is chosen) is non-monotonic — more CPU offload can be FASTER

**What goes wrong:**
For Qwen3-30B-A3B (~18.6 GB Q4_K_M, ~16.5 GB IQ4_XS), full-GPU placement still wants a 24 GB card. The standard remedy is `--cpu-moe` / `--n-cpu-moe N` / `-ot exps=CPU` to park routed-expert weights in system RAM. The trap: intuition says offloading more should monotonically slow things down, but measured behavior is frequently **non-monotonic** — moving more experts to CPU improves throughput (better overlap, less VRAM pressure, no spill), and the optimum N depends on RAM speed, context, and batch. Teams tune by intuition, pick a bad N, and conclude "the MoE path is slow."

**Why it happens:** routed experts activate only 3.3B params/token; PCIe transfers for cold experts overlap with GPU compute, so the naive mental model (CPU=slow) mispredicts.

**How to avoid:**
- Sweep `--n-cpu-moe` ∈ {0, 8, 16, 24, 48(≈all)} × {pp, tg} at fixed ctx; plot, don't spot-check.
- Ensure WSL2 `.wslconfig` grants enough RAM for experts + page cache (experts ~10–14 GB CPU-side) — WSL's default RAM cap silently throttles this path.
- Remember MoE changes the kernel-target landscape: grouped/expert GEMMs at small M dominate decode; a dense-model matmul win may not transfer (affects Phase 10 specialization).

**Warning signs:** tg insensitive or inversely sensitive to n-cpu-moe changes; WSL OOM-killing processes; expert-heavy layers absent from GPU profiles.

**Phase to address:** Phase 1 (if MoE chosen: establish offload baseline immediately) and Phase 10 (Qwen-specific graph work).

---

### Pitfall 12: Benchmark methodology sins — combined tok/s, single runs, thermal drift, avg_ts misreading

**What goes wrong:**
(a) Reporting one blended tok/s hides that prefill (compute-bound) and decode (memory-bandwidth-bound) have opposite bottlenecks — an optimization can win one and lose the other (the roadmap mandates separate paths precisely because of this).
(b) Single-run numbers on RDNA3 swing **12–18%** between cold and thermally-stabilized states; boost-clock decay over sustained inference makes "first run wins" artifacts.
(c) llama-bench's `avg_ts` in pg tests includes prompt tokens — misreading it inflates apparent generation speed.
(d) Publishing best-of-N without variance, or comparing runs taken hours apart at different clock states.

**Why it happens:** tok/s feels like one number; thermal state is invisible from inside WSL2 (Pitfall 2 compounds this).

**How to avoid:**
- Always report pp512 and tg128 (plus pg for realistic mixes) separately via llama-bench with `-r` repetitions and std-dev; discard nothing silently.
- Standardize thermal protocol: fixed warm-up run(s), fixed room/fan conditions, interleave A/B variants back-to-back within the same thermal window (paired comparison), randomize order across repeats.
- Pull temps/clocks/power from the Windows-side telemetry pipeline (Pitfall 2) and store alongside results; flag any sample where clock dropped below threshold.
- For correctness-sensitive A/B, fix seed + greedy decoding and diff outputs, don't eyeball vibes.

**Warning signs:** improvements that vanish on second run; "speedups" only visible in cold-start runs; pp and tg moving in opposite directions being reported as one metric.

**Phase to address:** Phase 1 (harness design) and Phase 15 (formal suite); paired-comparison discipline applies to every phase from Milestone 2 onward.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skipping the archived baseline binary, keeping only "the commit hash" | Less disk (binaries are ~100s of MB) | Baseline unbuildable later after upstream API churn; bisection dead | Never — disk is cheap |
| Hard-coding wave size 32 in playground kernels | Faster iteration early | Silent rewrite of every kernel when wave64 tested | Only in throwaway scratch files |
| Reading free VRAM from `hipMemGetInfo` for automation | One-line code | Wrong decisions on this card/driver combo (Pitfall 3) | Never — use empirical probing |
| Not recording driver/ROCm/build per result row | Faster benchmarking loop | Results unreproducible after first driver update | Never |
| Optimizing before the profiler gate passes | Feeling of progress | Optimizing the wrong kernel; wasted phases | Never — this is the roadmap's core rule |
| Ignoring numerical-diff harness "until kernels get serious" | Delayed infra work | First custom kernel ships subtly wrong; debugging inside full model instead of unit tests | Never — CPU-ref→HIP→diff pipeline is cheap and mandatory |
| Building/serving models from `/mnt/c/...` | Files visible on both OSes | 9P filesystem: glacial model load + build times | Only for one-off file exchange |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| WSL2 ↔ Windows driver | Letting Windows Update bump Adrenalin mid-project | Pin 26.2.2; snapshot distro post-setup; re-run env gate after any change |
| ROCm guest install | Mixing ROCm versions / using the old preview WSL instructions vs current ROCDXG flow | Follow the 7.2.1 Radeon WSL how-to exactly (repo packages, `HSA_ENABLE_DXG_DETECTION=1`, Ubuntu 22.04/24.04); record every step |
| llama.cpp build flags | Building without explicit `-DGPU_TARGETS=gfx1100` (generic target → suboptimal ISA) or toggling `GGML_HIP_ROCWMMA_FATTN` casually | Fixed CMake invocation scripted; flags recorded per binary; FA variants treated as separate configs |
| KV cache options | Assuming `-ctk q8_0` etc. work for every model/attn combo; enabling `-fa` "because it saves memory" without perf measurement | Measure fa on/off × kv-quant grid on the actual pinned build (FA changes both memory AND speed on ROCm — sometimes negatively, #10439) |
| Windows-side telemetry | Assuming ROCm tools expose sensors in WSL2 | Stand up HWiNFO/Adrenalin-based capture before benchmark suite work |
| Model download | Trusting third-party re-quant repos; wrong repo-id for the PENDING-USER model choice | Download from pinned official/barabanov-style reputable GGUF sources; record SHA/repo/commit in models/README.md |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| One universal GEMM for prefill+decode | Decode tok/s worse than stock despite "optimized" kernel; or vice versa | Separate M≈1 and M≫1 paths (roadmap Phase 7); autotune per shape bucket | Immediately at the other regime |
| Assuming MMA beats VALU on RDNA3 | Custom WMMA kernel loses to simple dot-product kernel at M=1 | Benchmark VALU-only vs MMA per shape (Phase 6 questions list) | Small-M decode shapes |
| Occupancy collapse from VGPR pressure | Kernel slower than its naive version after "optimization" (unrolled/vectorized) | Check `--save-temps`/occupancy metrics; cap registers deliberately; measure waves-per-CU | After aggressive unrolling/vector-width increases |
| LDS bank conflicts in tiled staging | Tile-size sweeps show plateau far below bandwidth roofline | Check Bank Conflict Rate in rocprofiler-compute; pad/swizzle LDS layouts | Any tiled dequant/matmul staging |
| Chasing prefill wins while decode dominates real usage | Great pp512 deltas, imperceptible chat improvement | Weight optimizations by target workload profile (Profile A–D mix) | Reporting/publication stage |
| Ignoring PCIe transfer costs in hybrid CPU/GPU splits | Offloaded configs far slower than modeled; surprise cliffs at ctx growth | Profile host-device copies explicitly; keep KV + attention + shared embeddings GPU-side | MoE offload configs, partial -ngl |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Downloading GGUFs from unvetted mirrors | Corrupted/backdoored weights; silent quality bugs blamed on kernels | Official/reputable sources only; checksum + metadata pinning in models/README.md |
| Running WSL with default user in admin groups / credential forwarding while experimenting | Expanded blast radius of any guest compromise | Standard WSL2 hygiene; no need for elevated guest for ROCm work beyond documented udev/rules steps |
| Ignoring model license/compliance in published results | License violation on redistribution of quats/results | Qwen3 family is Apache-2.0 (commercial OK) — record license alongside artifacts anyway |

*(This domain is offline local inference; attack surface is mainly supply chain, above.)*

## UX Pitfalls

*(Adapted: "users" here = consumers of the published benchmark results and future-you six months from now.)*

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Publishing only the best-number config | Others can't reproduce; credibility loss | Publish the full matrix (roadmap Phase 16 rule) incl. failed experiments |
| Results without env metadata (driver/ROCm/clocks/temps) | Numbers unverifiable; useless after any update | Harness enforces metadata columns per row |
| Blended "tok/s" headline | Readers optimize the wrong thing for their workload | Lead with pp/tg split + workload-profile framing |
| Docs describing WSL2 setup that worked "once" | Followers hit Pitfall 1/2/5 with no map | Document pinned versions, snapshot/restore procedure, known WSL2 limitations openly |

## "Looks Done But Isn't" Checklist

- [ ] **Environment phase "done":** often missing telemetry pipeline (Windows-side) and profiler-gate proof — verify rocprofv3 produced a real counter dump on a model workload, and a sensor CSV exists
- [ ] **Baseline "done":** often missing archived baseline *binary* + recorded flags/hashes — verify you can rerun the exact stock build after upstream rebases
- [ ] **Custom kernel "done":** often missing numerical-diff evidence vs CPU reference across shapes (not just one), and wave-size-generic testing — verify diff harness output archived
- [ ] **Speedup "claimed":** often missing paired thermal-window comparison + repetitions + std-dev — verify ≥N interleaved runs with variance reported
- [ ] **VRAM reduction "achieved":** often measured via lying free-VRAM APIs — verify ledger built from llama.cpp buffer logs + end-to-end max-context smoke test passed (no Pitfall-1 spill)
- [ ] **Model selection "decided":** often missing measured load test of BOTH candidates at target context on THIS card/driver — verify dense-IQ4 and MoE-Q4_K_S smoke runs logged before the PENDING-USER call closes

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Driver update broke stack (5) | LOW | Restore WSL snapshot; reinstall matched driver pair; re-run env gate |
| Profiler unusable in WSL2 (4) | MED | Drop to llama.cpp-timing tier; escalate to native-Linux contingency per recorded decision tree |
| Silent VRAM spill corrupted benchmark series (1) | LOW–MED | Identify affected runs via RSS log; re-run affected cells under tighter `-c`/kv-quant |
| Wrong model downloaded / doesn't fit (10) | HIGH (bandwidth + time) | Re-select per smoke-test matrix; this is why download waits for research confirmation |
| Wave-size correctness bug shipped into runtime (6) | MED | Bisect via ENABLE_CUSTOM_KERNEL flag; unit diffs localize offending kernel quickly if harness exists |
| Upstream rebase regressed perf silently (9) | MED | Compare archived baseline vs new pin; bisect upstream; stay on old pin until resolved |
| Thermal-variance invalidated a comparison (12) | LOW | Re-run paired within one thermal window; add clock guard to harness |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1 VRAM overcommit spill | Phase 1 gate + harness | RSS logging present; max-ctx smoke test passes without RAM growth |
| 2 No WSL telemetry | Phase 1 | Windows-side sensor CSV captured alongside a benchmark run |
| 3 VRAM reporting lies | Phase 1 + Phase 9 | Ledger populated from llama.cpp logs; discrepancy vs reported free documented once |
| 4 Profiler gate | Phase 1 exit criterion | rocprofv3 counter dump on real workload OR documented fallback decision |
| 5 Driver coupling | Phase 1 + standing rule | Pinned-version doc + WSL snapshot file exists; gate re-run log after updates |
| 6 Wave size | Phases 4–6 | Diff harness passes at both wave modes; no literal 32/64 in kernel sources (grep gate) |
| 7 WMMA assumptions | Phase 8 (+12 flags) | MMA-vs-VALU per-shape table exists; fa on/off × bs grid measured |
| 8 rocBLAS coverage | Phases 2–3, 6 | Dispatch/profiling table shows what routes to rocBLAS; warm-up crash absence confirmed |
| 9 Upstream drift | Phases 1–2, standing | Pinned commit + archived binary reproducible; rebase protocol followed at boundaries |
| 10 VRAM budget | Phase 1 (+9) | Per-quant/ctx ledger; largest-planned-prompt smoke test passed |
| 11 MoE offload tuning | Phase 1/10 (if MoE) | n-cpu-moe sweep plotted; .wslconfig RAM sized |
| 12 Benchmark method | Phase 1 harness + Phase 15 | Harness enforces pp/tg split, `-r` reps, std-dev, metadata columns, thermal pairing |

## Sources

**Official documentation (HIGH confidence):**
- ROCm Radeon WSL how-to & support matrices (ROCDXG production support, Adrenalin 26.2.2 + ROCm 7.2.1, HSA_ENABLE_DXG_DETECTION) — rocm.docs.amd.com (radeon-ryzen 7.2.x/latest)
- ROCm "Limitations and recommended settings" (WSL: rocm-smi unsupported; lower-than-native inference perf; WSL overhead noted) — rocm.docs.amd.com 6.3.4/6.4.2
- HIP porting guide & language extensions (warpSize rules) — rocm.docs.amd.com / rocmdocs.amd.com
- GPUOpen: "How to accelerate AI applications on RDNA3 using WMMA"; "Register pressure in AMD CDNA2 GPUs"
- rocprofiler-compute docs (LDS bank-conflict metrics; RDNA Speed-of-Light)
- rocWMMA programmer guide/API reference

**Issue trackers & PRs (MEDIUM confidence unless cross-corroborated):**
- llama.cpp#20934 (ROCm<Vulkan decode on 7900 XTX; wave-size backend discrepancy), #10439 (FA slower on ROCm, esp. bs>1), #12032 + #24437 + #13110 (rocWMMA FATTN: default-OFF, −41% prefill gfx1151, gfx1201 breakage), #20839 (TensileLibrary arch gap → warmup crash), #20647 (upstream regression 617db24), #23999 (DXG free-VRAM anomaly), #22583 (WSL2 VRAM overcommit → RAM exhaustion), #11758 (deprecated wavefront macro), #11495/#11519 (ggml wave-size assumptions/selectable mmv), #18506/#7885/#5993 (gradual VRAM growth / prefill OOM patterns)
- ROCm#6389 (hipMemGetInfo over-reports free VRAM), ROCm#6022/#4018 (WSL memory mapping/UKI limits), ROCm#4459 (driver update broke ROCm), librocdxg#6 (amd-smi NOT implemented in ROCDXG), ROCm/ROCm#4722 (ROCm perf anomaly gfx1100)
- PR#15077/#11397 (--n-cpu-moe / -ot exps=CPU semantics), huggingface.co MoE offload guide

**Community/methodology (LOW–MEDIUM, used for magnitudes only):**
- CraftRigs llama-bench methodology (thermal swing 12–18%; repetitions guidance); TechnoLynx & RunLocalAI benchmark-methodology pieces; TensorRT-LLM perf-benchmarking doc (clock-management practice); localmodel.run / canitrun.dev / hardwarepedia (Qwen3-32B VRAM footprints, consistent with orchestrator-verified GGUF sizes); TechSpot/Igor's Lab/Jeff Geerling (7900-series reference thermal defects — XTX MBA mostly, XT less affected)

---
*Pitfalls research for: RDNA3/gfx1100 LLM inference optimization (llama.cpp HIP/ROCm, WSL2, 27B-class on 20 GB)*
*Researched: 2026-08-21*
