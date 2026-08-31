# Phase 2: Benchmark Harness & Baseline Matrix - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-23
**Phase:** 2-Benchmark Harness & Baseline Matrix
**Areas discussed:** Windows telemetry bridge; Stock-Vulkan comparator arm; Harness architecture & workload profiles; Result store & fingerprint schema; RSS guard & overcommit defense; Matrix execution protocol & thermal fail-safe
**Research mode:** Owner mandated web research before every area's questions (overrides `research_before_questions: false`)

---

## Windows Telemetry Bridge

| Option | Description | Selected |
|--------|-------------|----------|
| A) HWiNFO shmem reader daemon | Programmatic per-run capture, ~50 LOC, HWiNFO must run | |
| B) Manual HWiNFO CSV | Zero code, manual start/stop each session | |
| C) LibreHardwareMonitor headless | Fully scriptable, weaker RDNA3 sensor coverage | |
| D) Hybrid fallback | A primary + B documented contingency | ✓ |

**User's choice:** D
**Notes:** Field set locked: core/mem clock, edge+hotspot temp, board power, fan %, util, VRAM used. Research basis: HWiNFO Shared Memory v2 interface + existing open-source readers; WSL clock tracks host via Hyper-V integration services.

## Stock-Vulkan Comparator Arm

| Question | Options | Selected |
|----------|---------|----------|
| Placement | A) Native Windows separate tree / B) defer arm / C) decide | A ✓ |
| Pin | A) same bb4caa75 / B) newest release | A ✓ |
| Coverage gate | A) accept 6-part revised gate / B) trim / C) decide | Research first → then A ✓ |

**User's choice:** A, A, and for the gate: "do more research" → research found `gated_delta_net.comp`/`solve_tri.comp`/`dequant_iq4_xs.comp` present at pin, no dedicated iq4_xs vec-dot shader (generic `mul_mat_vecq` path), and upstream's official `test-backend-ops support --output csv` matrix mechanism → six-part gate accepted.
**Notes:** llama.cpp#20934 validated comparator motivation (ROCm −15–25% tg vs Vulkan on gfx1100, wave32 vs wave64). WSL2 dzn ruled out via WSLg issues #1340/#1254.

## Harness Architecture & Workload Profiles

| Option | Description | Selected |
|--------|-------------|----------|
| 3.1 Two-layer harness | llama-bench wrapper + real-prompt runner provisioning Phase-3 shapes | ✓ |
| 3.2 Cell semantics | `-p C` AND `-pg C,128` per tier (empty-context tg banned from matrix) | ✓ |
| 3.3 Repeats | `-r 5`, mean±stdev, ±5% reproducibility gate | ✓ |
| 3.4 Corpus | ~6 deterministic files short/long × code/prose, greedy only | ✓ |

**User's choice:** A to all.
**Notes:** Research caught the `-pg` trap: plain `-n` measures empty-context decode regardless of tier.

## Result Store & Fingerprint Schema

| Option | Description | Selected |
|--------|-------------|----------|
| 4.1 Location | in-repo `benchmarks/results/` | ✓ |
| 4.2 Retention | raw per-repeat rows forever | ✓ |
| 4.3 Corrections | supersede model, checksums at close | ✓ |

**User's choice:** A to all.
**Notes:** Schema grounded in run-manifest pattern + fsync'd JSONL WAL practice from research.

## RSS Guard & Overcommit Defense

| Option | Description | Selected |
|--------|-------------|----------|
| 5.1 Signals | A) three signals (guest RSS / shared-GPU-mem / throughput collapse) | ✓ |
| 5.2 Thresholds | A) empirical calibration session first | ✓ |
| 5.3 Validation | A) fixture traces + one supervised live near-OOM | ✓ |
| 5.4 Notifications | C) toast on trips/FAILED + end-of-session summary ping | ✓ |

**User's choice:** a a a c (owner recommendation accepted).
**Notes:** Owner asked for fuller explanation mid-area, then added: "You can use normal ram when needed. But it would be good if I got a windows notification about it." → spill tolerated but always labeled FAILED + toast channel became first-class. WSL#11050/#40732 grounded the spill mechanism.

## Matrix Execution Protocol & Thermal Fail-Safe

| Option | Description | Selected |
|--------|-------------|----------|
| 6.1 Sessions | A) one session per backend arm, telemetry-flagged drift | ✓ |
| 6.2 Clocks | A) fully stock confirmed | ✓ |
| 6.3 32k gate | A) attempt under guard with expected-fail path | ✓ |
| 6.4 Ordering | A) fixed deterministic ascending order | ✓ |

**User's choice:** all A — with two follow-up requirements:
1. "can you control my fans in any fail safe situation?" → answered NO (methodology rule + firmware protection is superior backstop); owner accepted ("ok dont controll it").
2. "can you make a limit like if my gpu ever goes above 95 degrees close all?" → YES: Windows-side watcher kills benchmark process at junction ≥95 °C, row = FAILED:thermal-abort, toast fires.
3. Owner exception clause: "if we have to reach 95 degrees to do a one time job (if it has to be once to finish the project) thats alright" → supervised-override path, individually sanctioned and logged.

**Notes:** Kill-switch lives Windows-side so observer survives guest wedging; escalation ladder kill-process → optional wsl --terminate discussed, tier (i) chosen as default.

## Claude's Discretion

- HWiNFO shmem reader implementation choice
- Prompt-corpus file contents within locked constraints
- Pre-flight estimation method
- Calibration session structure
- Toast implementation mechanism

## Deferred Ideas

None — discussion stayed within phase scope.
