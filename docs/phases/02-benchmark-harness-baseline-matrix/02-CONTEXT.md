# Phase 2: Benchmark Harness & Baseline Matrix - Context

**Gathered:** 2026-08-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver reproducible, fingerprinted measurement infrastructure for the pinned stock build: a two-layer benchmark harness (llama-bench wrapper + real-prompt runner) with Windows-side telemetry, a three-signal VRAM-spill guard with thermal fail-safe, an append-only result store, and the published stock baseline matrix (pp/tg × ctx {4k,8k,16k,32k} × fa {on,off}) on HIP plus a stock-Vulkan comparator arm. Zero optimization work.

Requirements in scope: BENCH-01, BENCH-02, BENCH-03, BENCH-04.
</domain>

<decisions>
## Implementation Decisions

### Windows telemetry bridge
- **D2-01:** Hybrid fallback design — primary capture = HWiNFO shared-memory reader daemon (small Windows-side script polling HWiNFO's documented shmem interface), triggered per run window by the harness; documented manual HWiNFO CSV logging is the contingency if shared memory misbehaves. Mandatory fields locked: GPU core clock, mem clock, edge temp, hotspot/junction temp, board power draw, fan %, GPU util, VRAM used — plus shared-GPU-memory (added by D2-12). Time-joining via UTC wall-clock on both sides; per-run telemetry slices land inside the run directory; before/after clock-skew check each session.

### Stock-Vulkan comparator arm
- **D2-02:** Arm lives native-Windows in a separate source tree (WSL2 dzn/Vulkan-on-D3D12 is unusable for compute). Runs against the Adrenalin Vulkan runtime; VS2022 + Vulkan SDK (glslc) toolchain.
- **D2-03:** Pin = same `bb4caa75` as the HIP baseline — identical code, only backend differs (matches llama.cpp#20934 methodology).
- **D2-04:** Six-part GDN/coverage gate, all required BEFORE any Vulkan number publishes:
  1. Static shader inventory at pin (DONE during discussion, evidence archived): `gated_delta_net.comp`, `solve_tri.comp`, `dequant_iq4_xs.comp` present at `bb4caa75`. Known nuance to carry into results: no dedicated `mul_mat_vec_iq4_xs` vec-dot shader exists at this commit — IQ4_XS decode routes through generic `mul_mat_vecq`.
  2. `test-backend-ops support --output csv` executed on BOTH backends, both CSVs archived; `GATED_DELTA_NET`, `SOLVE_TRI`, `SSM_CONV`, `SSM_SCAN` rows must be ✅ on Vulkan.
  3. Full `test-backend-ops` green on the Vulkan backend.
  4. Verbose startup residency: all 132 tensor layers on Vulkan device, 0 on CPU.
  5. Coherent greedy-decode smoke test on fixed prompt.
  6. Any 🟡 partial-support row or CPU-fallback recorded verbatim in the result row.

### Harness architecture & workload profiles
- **D2-05:** Two-layer harness. Layer 1 = llama-bench wrapper producing the canonical matrix (community-comparable, `-o jsonl`). Layer 2 = thin real-prompt runner (`llama-cli --single-turn`, greedy, fixed corpus) provisioning Phase-3's four workload shapes now.
- **D2-06:** Cell semantics: each context tier C ∈ {4k, 8k, 16k, 32k} runs BOTH `-p C` (pure prefill) AND `-pg C,128` (decode-at-context-C). Plain `-n` empty-context tg is banned from the matrix — it would mislabel context-free numbers as tiered ones.
- **D2-07:** Repeats `-r 5` (upstream default), warmup default-on; publish mean ± stdev per cell; reproducibility gate = re-run variance within ±5% (BENCH-01 evidence).
- **D2-08:** Real-prompt corpus: fixed deterministic files under `benchmarks/prompts/`, ~6 files covering short/long × code/prose; exact file list at planning time; greedy decode only.

### Result store & fingerprint schema
- **D2-09:** Store lives in-repo at `benchmarks/results/<YYYYMMDD_HHMMSS>_<run-id>/` containing `manifest.json`, `rows.jsonl` (fsync per row), `telemetry/*.csv`, `logs/*.txt`, `CHECKSUMS.sha256` written at run close. Append-only enforced socially+verifiably via checksums.
- **D2-10:** Fingerprint (manifest) fields: harness git-rev; llama.cpp commit+tag+binary sha256+build flags; backend arm (HIP|Vulkan); ROCm 7.2.1; librocdxg 1.2.2; guest kernel + WSL kernel + Windows build; Adrenalin driver version; `.wslconfig` sha256 (memory=28GB); model sha256/HF rev; telemetry mode (`shmem`|`manual-fallback`|`absent`); UTC start/end + skew-check result; tune-state confirmation (stock).
- **D2-11:** Retention: raw per-repeat rows kept forever; aggregates computed at publish time. Corrections use supersede model: new run references old run-id, both remain visible.

### RSS guard & overcommit defense (BENCH-03)
- **D2-12:** Three-signal detector, any trip ⇒ row = `FAILED:suspected-spill`, excluded from published matrix: (1) guest `/proc/<pid>/status` VmRSS/VmSwap polled @1 Hz; (2) Windows shared-GPU-memory climb during steady decode (from D2-01 telemetry); (3) >2× intra-cell repeat-to-repeat throughput deviation flagged for review.
- **D2-13:** Thresholds derived empirically at a calibration session (healthy-run profiles across context tiers → margins above measured reality, e.g. ~1.5× steady-state), then written into the run protocol. No guessed hard-coded caps.
- **D2-14:** Guard validation = fixture traces (synthetic spiked RSS series asserted to produce FAILED verdicts; permanent regression test) + ONE supervised deliberate near-OOM run at calibration (candidates: 32k/fa-off, oversized batch) to prove clean-fail path or live trip.
- **D2-15:** Owner accepts system-RAM use when needed — spill is not prevented, it is *labeled*. Notifications: Windows toast (via powershell.exe interop) fires on any guard trip or FAILED cell, PLUS one end-of-session summary ping ("N OK / M FAILED").

### Matrix execution protocol
- **D2-16:** One session per backend arm, starting cool/idle; continuous telemetry throughout; rows flagged when clocks deviate >5% from session median or hotspot sustains ≥105 °C; `llama-bench -D` cooldown gaps between heavy tiers. Batch cells into few invocations (model loads once per invocation).
- **D2-17:** Clock/tune policy: fully stock confirmed — no Adrenalin tuning profiles, stock power limit; re-verified at execution start (feeds manifest tune-state field).
- **D2-18:** 32k tier: attempt full 32k under guard. Pre-flight allocation estimate first; predicted-over-budget ⇒ row lands `FAILED:preflight-oom` (documented expected-fail, still published); probe-passing runs execute guarded like any cell.
- **D2-19:** Fixed deterministic cell order (ctx ascending, fa off→on), identical across the project lifetime for cross-session comparability.

### Thermal fail-safe
- **D2-20:** NO software fan control — firmware protection (throttle ~110 °C junction, hardware shutdown beyond) is the real backstop; harness records fans, never drives them (record-don't-control methodology). Windows-side watcher arms automatically every session: junction ≥95 °C ⇒ kill current benchmark process via wsl.exe interop (last resort `wsl --terminate`), mark row `FAILED:thermal-abort`, toast the owner. Owner-sanctioned exception: a one-time job that genuinely must reach 95 °C may proceed ONLY as a supervised-override — individually sanctioned, logged in the protocol.

### Claude's Discretion
- Exact HWiNFO shmem reader implementation (PowerShell vs Python vs compiled helper)
- Exact prompt-corpus file contents (within D2-08 constraints)
- Pre-flight estimation method (startup probe vs buffer math)
- Calibration session structure details
- Toast implementation mechanism (BurntToast vs raw XML)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning docs
- `docs/ROADMAP.md` — Phase 2 success criteria, WSL2 risk notes, binding methodology rules (§Methodology Rules)
- `docs/REQUIREMENTS.md` §Benchmarking — BENCH-01..04 authoritative text
- `docs/reference/ROADMAP-original.md` — original methodology rules (binding, inherited)
- `docs/phases/01-environment-validation-stock-baseline/01-CONTEXT.md` — carried decisions D-01..D-07 (root-only, frozen ROCm 7.2.1/Adrenalin pairing, binary archive policy, model logistics)
- `docs/phases/01-environment-validation-stock-baseline/01-SUMMARY.md` — open items carried into this phase: elevated registry command pending owner; `.wslconfig` must join fingerprint; v0.2.0 lacks "offloaded N/M" line → verbose per-layer lines are residency evidence
- `docs/research/deep-research/REPORT.md` — HIP-vs-Vulkan decode gap evidence; librocdxg#60 ROCr busy-spin fingerprint note; WSL#40732 crash pattern behind fail-fast policy
- `benchmarks/environment/versions.txt` + `benchmarks/environment/vram-probe.txt` — frozen version pairing and free-VRAM actuals feeding pre-flight math

### External research references (consulted during discussion, 2026-08-23)
- llama.cpp#20934 — ROCm vs Vulkan tg gap on gfx1100 (15–25%), wave32-vs-wave64 observation; comparator-arm justification
- llama.cpp PR #18102 — Vulkan DeltaNet/SOLVE_TRI implementation era; confirms op maturity at pin
- `tools/llama-bench/README.md` @ bb4caa75 — `-o jsonl`, `-pg` combo semantics, `-r`, `-D/--delay`, warmup defaults
- Upstream `docs/ops.md` pipeline — `test-backend-ops support --output csv` as official per-backend op matrix mechanism
- microsoft/WSL#11050 (dxg sysmem-fallback ignored → silent RAM backing) and #40732 (hard-crash pattern) — guard rationale
- ML.ENERGY thermally-stable profiling; NVIDIA CUTLASS measurement guidelines — session/thermal protocol practice
- HWiNFO Shared Memory v2 spec + open-source readers (Hwinfo.SharedMemory.Net, hwinfo-telegraf-plugin) — telemetry daemon basis

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `benchmarks/environment/*` — established env-fingerprint artifact pattern (plain-text captures + README pointers) that manifest.json formalizes
- `baseline/binaries/v0.2.0-bb4caa75/` — archived stock binaries are THE harness target executables (never rebuilt casually, rule 3)
- Guest-side paths: `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (canonical copy), source tree at `/root/llama.cpp`

### Established Patterns
- Headless invocation: `setsid` + `--simple-io` + `--single-turn` + explicit `-c` + `--no-mmap` (Phase-1 proven)
- In-repo artifacts while repo < 750 MB (D-06 precedent extends naturally to small JSONL results)
- Verbose per-layer device-assignment lines as residency proof (v0.2.0 has no offload summary line)

### Integration Points
- Windows↔WSL interop (`powershell.exe` callable from guest; `wsl.exe` callable from host) — telemetry trigger path and kill-switch path both depend on it
- `.wslconfig` memory=28GB — REQUIRED state, hash enters every manifest
- Phase-1 snapshot `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` — recovery point if environment gates ever fail

</code_context>

<specifics>
## Specific Ideas

- Owner priority stated twice: notifications matter ("it would be good if I got a windows notification about it") — toast channel is a first-class deliverable, not garnish.
- Owner safety stance: hardware/firmware protection trusted; software fan control explicitly rejected; 95 °C watchdog always armed; supervised-override exception only for jobs that cannot finish otherwise.
- Comparator honesty emphasized throughout: every performance claim names its backend; empty-context tg banned; partial-support rows recorded verbatim.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Pending owner action carried from Phase 1, not new: elevated registry command for driver-update pause; see `benchmarks/environment/versions.txt`.)

</deferred>

---

*Phase: 2-Benchmark Harness & Baseline Matrix*
*Context gathered: 2026-08-23*
