# Phase 2: Benchmark Harness & Baseline Matrix - Research

**Researched:** 2026-08-23
**Domain:** llama.cpp benchmark instrumentation (pinned v0.2.0 @ bb4caa75), WSL2↔Windows interop telemetry, VRAM-spill guarding, append-only result stores
**Confidence:** HIGH (core harness semantics verified against the pinned source tree this session; external telemetry interfaces CITED from spec mirrors)

## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Windows telemetry bridge
- **D2-01:** Hybrid fallback design — primary capture = HWiNFO shared-memory reader daemon (small Windows-side script polling HWiNFO's documented shmem interface), triggered per run window by the harness; documented manual HWiNFO CSV logging is the contingency if shared memory misbehaves. Mandatory fields locked: GPU core clock, mem clock, edge temp, hotspot/junction temp, board power draw, fan %, GPU util, VRAM used — plus shared-GPU-memory (added by D2-12). Time-joining via UTC wall-clock on both sides; per-run telemetry slices land inside the run directory; before/after clock-skew check each session.
- #### Stock-Vulkan comparator arm
- **D2-02:** Arm lives native-Windows in a separate source tree (WSL2 dzn/Vulkan-on-D3D12 is unusable for compute). Runs against the Adrenalin Vulkan runtime; VS2022 + Vulkan SDK (glslc) toolchain.
- **D2-03:** Pin = same `bb4caa75` as the HIP baseline — identical code, only backend differs (matches llama.cpp#20934 methodology).
- **D2-04:** Six-part GDN/coverage gate, all required BEFORE any Vulkan number publishes: (1) static shader inventory at pin (DONE, evidence archived); (2) `test-backend-ops support --output csv` executed on BOTH backends, both CSVs archived; `GATED_DELTA_NET`, `SOLVE_TRI`, `SSM_CONV`, `SSM_SCAN` rows ✅ on Vulkan; (3) full `test-backend-ops` green on Vulkan; (4) verbose startup residency: all 132 tensor layers on Vulkan device, 0 on CPU; (5) coherent greedy-decode smoke test on fixed prompt; (6) any 🟡 partial-support row or CPU-fallback recorded verbatim in the result row.
- #### Harness architecture & workload profiles
- **D2-05:** Two-layer harness. Layer 1 = llama-bench wrapper producing the canonical matrix (`-o jsonl`). Layer 2 = thin real-prompt runner (`llama-cli --single-turn`, greedy, fixed corpus) provisioning Phase-3's four workload shapes now.
- **D2-06:** Each context tier C ∈ {4k, 8k, 16k, 32k} runs BOTH `-p C` AND `-pg C,128`. Plain `-n` empty-context tg is banned from the matrix.
- **D2-07:** Repeats `-r 5`, warmup default-on; publish mean ± stdev per cell; reproducibility gate = re-run variance within ±5% (BENCH-01 evidence).
- **D2-08:** Real-prompt corpus: fixed deterministic files under `benchmarks/prompts/`, ~6 files covering short/long × code/prose; exact file list at planning time; greedy decode only.
- #### Result store & fingerprint schema
- **D2-09:** Store lives in-repo at `benchmarks/results/<YYYYMMDD_HHMMSS>_<run-id>/` containing `manifest.json`, `rows.jsonl` (fsync per row), `telemetry/*.csv`, `logs/*.txt`, `CHECKSUMS.sha256` written at run close. Append-only enforced socially+verifiably via checksums.
- **D2-10:** Fingerprint (manifest) fields: harness git-rev; llama.cpp commit+tag+binary sha256+build flags; backend arm (HIP|Vulkan); ROCm 7.2.1; librocdxg 1.2.2; guest kernel + WSL kernel + Windows build; Adrenalin driver version; `.wslconfig` sha256 (memory=28GB); model sha256/HF rev; telemetry mode (`shmem`|`manual-fallback`|`absent`); UTC start/end + skew-check result; tune-state confirmation (stock).
- **D2-11:** Retention: raw per-repeat rows kept forever; aggregates computed at publish time. Corrections use supersede model: new run references old run-id, both remain visible.
- #### RSS guard & overcommit defense (BENCH-03)
- **D2-12:** Three-signal detector, any trip ⇒ row = `FAILED:suspected-spill`: (1) guest `/proc/<pid>/status` VmRSS/VmSwap polled @1 Hz; (2) Windows shared-GPU-memory climb during steady decode; (3) >2× intra-cell repeat-to-repeat throughput deviation flagged for review.
- **D2-13:** Thresholds derived empirically at a calibration session (healthy-run profiles → margins above measured reality, e.g. ~1.5× steady-state), then written into the run protocol. No guessed hard-coded caps.
- **D2-14:** Guard validation = fixture traces (synthetic spiked RSS series asserted to produce FAILED verdicts; permanent regression test) + ONE supervised deliberate near-OOM run at calibration (candidates: 32k/fa-off, oversized batch).
- **D2-15:** Spill is *labeled*, not prevented. Notifications: Windows toast (via powershell.exe interop) fires on any guard trip or FAILED cell, PLUS one end-of-session summary ping ("N OK / M FAILED").
- #### Matrix execution protocol
- **D2-16:** One session per backend arm, starting cool/idle; continuous telemetry; rows flagged when clocks deviate >5% from session median or hotspot sustains ≥105 °C; `llama-bench` cooldown gaps between heavy tiers. Batch cells into few invocations (model loads once per invocation).
- **D2-17:** Clock/tune policy: fully stock confirmed — re-verified at execution start (feeds manifest tune-state field).
- **D2-18:** 32k tier: attempt full 32k under guard. Pre-flight allocation estimate first; predicted-over-budget ⇒ row lands `FAILED:preflight-oom` (documented expected-fail, still published); probe-passing runs execute guarded like any cell.
- **D2-19:** Fixed deterministic cell order (ctx ascending, fa off→on), identical across project lifetime.
- #### Thermal fail-safe
- **D2-20:** NO software fan control. Windows-side watcher arms automatically every session: junction ≥95 °C ⇒ kill current benchmark process via wsl.exe interop (last resort `wsl --terminate`), mark row `FAILED:thermal-abort`, toast the owner. Supervised-override exception only, individually sanctioned and logged.

### Claude's Discretion
- Exact HWiNFO shmem reader implementation (PowerShell vs Python vs compiled helper)
- Exact prompt-corpus file contents (within D2-08 constraints)
- Pre-flight estimation method (startup probe vs buffer math)
- Calibration session structure details
- Toast implementation mechanism (BurntToast vs raw XML)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope. (Pending owner action carried from Phase 1: elevated registry command for driver-update pause; see `benchmarks/environment/versions.txt`.)

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BENCH-01 | Reproducible harness wrapping llama-bench — fixed profiles, enforced pp/tg split, warmup + ≥3 repeats, machine-readable output | Verified llama-bench flag surface at pin (`-r 5` default, warmup default-on, `-o jsonl` with per-repetition `samples_ts` arrays, mean±stddev computed upstream); cell-enumeration semantics verified so wrapper can enforce D2-06/D2-07 exactly; ±5% gate compares session means |
| BENCH-02 | Every result row fingerprinted incl. clocks/temps from Windows-side telemetry | HWiNFO SM2 map layout captured verbatim (`Global\HWiNFO_SENS_SM2`, struct fields); manifest field list locked in D2-10; free-VRAM anchor measured (18245 MiB free of 20421) |
| BENCH-03 | VRAM ledger + RSS guard defeating silent overcommit; fail-fast policy; crash-resilient journal | Guard design mapped to three signals with concrete data sources (`/proc/<pid>/status` VmRSS/VmSwap; HWiNFO shared-GPU-memory reading; intra-cell deviation); jsonl printer flushes per row (journal survives hard abort); OOM exit semantics verified (`return 1`, remaining cells lost — motivates pre-flight gate) |
| BENCH-04 | Baseline matrix pp/tg × ctx {4k..32k} × fa {on,off} + stock-Vulkan comparator arm; 32k pre-flight gate | Cell count derived: 16 rows/arm (4 tiers × {pp@C, pg C,128} × {fa off,on}); Vulkan build path verified from pinned docs/build.md; shader inventory re-verified in pinned tree; test-backend-ops support CSV schema captured verbatim |
</phase_requirements>

## Summary

Phase 2 is pure measurement infrastructure around two already-built binaries. This session verified the entire llama-bench contract directly against the pinned tree at `/root/llama.cpp` (HEAD = `bb4caa7540188872173c44d161602d9271386413`, clean): its flag surface, its cell-enumeration algorithm, its JSONL output schema (including per-repetition sample arrays and per-row flush), its warmup cost, its failure modes, and the fact that **it has no `-c` flag** — context is always derived as `n_prompt + n_gen + n_depth`. It also surfaced two facts that contradict shorthand in the discussion notes: the cooldown flag is **`--delay`, not `-D`**, and the defaults inject **plain `pp 512` + empty-context `tg 128` cells unless the wrapper passes explicit `-p` and `-n 0`**. A wrapper that naively passes only `-pg C,128` would silently publish exactly the banned context-free rows.

The environment side is favorable and partially proven: guest tooling (python3 3.12.3, taskset, flock, sha256sum) and host Python 3.14 are present; powershell.exe interop works from the guest; the Windows→guest kill path (`wsl.exe -d Ubuntu-24.04 -u root -- sh -c 'kill -9 <pid>'`) was live-tested successfully; archived baseline binaries are byte-identical to `/root/llama.cpp/build-ci/bin/`; and DXG already reports usable free-VRAM numbers through llama.cpp's own device-init log (20421 MiB total / 18245 MiB free) — the empirical input for the 32k pre-flight without rocm-smi. HWiNFO's Shared Memory v2 interface is fully specified (map name, mutex, packed structs, signature bytes) and readable from Python or PowerShell on the host, with one operational constraint discovered: **non-Pro HWiNFO64 limits shared memory to 12 h/day**.

**Primary recommendation:** Build the harness as a small guest-side Python orchestrator (one module per concern: fingerprint, bench wrapper, prompt runner, guard, store) driving `build-ci/bin` binaries with explicit `-p C -n 0 -pg C,128 -fa off,on -r 5 -o jsonl -v --progress --delay N` invocations, paired with a host-side Python HWiNFO-shmem daemon (CSV fallback documented) that also hosts the thermal watchdog, communicating through files under the run directory and toasts through powershell.exe. Enrich each llama-bench JSONL row into `rows.jsonl` with guard verdict + telemetry-slice references, fsync per row, checksum at close.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Benchmark execution (llama-bench cells, prompt runner) | Guest (WSL2 Ubuntu) | — | Binaries are Linux ELF linked against guest libggml-hip.so; ROCm lives in guest |
| Result store writes / fsync / checksums | Guest | — | Same filesystem as producer; in-repo path via /mnt/e mount |
| Telemetry capture (clocks/temps/power/fans/shared-GPU-mem) | Windows host | Guest (time-join) | rocm-smi absent under ROCDXG; HWiNFO shmem is a Windows kernel object |
| Thermal watchdog + kill switch | Windows host | Guest (target process) | Junction-temp readings are host-side; kill crosses boundary via wsl.exe (verified) |
| Toast notifications | Windows host | Guest (trigger via interop) | Toast API is Windows-only; powershell.exe callable from guest |
| VRAM ledger (per-cell buffers) | Guest (parse llama.cpp logs) | Host telemetry cross-check | Buffer sizes printed by llama.cpp at load; DXG free-VRAM line is the empirical anchor |
| Fingerprinting (versions, hashes) | Guest | Host (.wslconfig hash, driver via WMI) | versions.txt pattern extends naturally; host bits fetched via powershell.exe |
| Vulkan comparator arm execution | Native Windows | — | Locked D2-02: dzn under WSL2 unusable for compute |

## Standard Stack

This phase installs **zero new package-manager dependencies**. Everything rides on tools already present and verified:

### Core
| Component | Version | Purpose | Provenance |
|---------|---------|--------------|------------|
| llama-bench (pinned) | v0.2.0 @ bb4caa75 | Canonical matrix producer (`-o jsonl`) | [VERIFIED: /root/llama.cpp HEAD + README + source] |
| llama-cli (pinned) | v0.2.0 @ bb4caa75 | Layer-2 real-prompt runner (`--single-turn`, greedy) | [VERIFIED: common/arg.cpp:1903-1908,2005] |
| test-backend-ops (pinned) | same commit | Backend support gate (`support --output csv`) | [VERIFIED: tests/test-backend-ops.cpp:10860-10922] |
| Python (guest) | 3.12.3 | Harness orchestrator, guard, store writer | [VERIFIED: `python3 --version` in guest] |
| Python (host) | 3.14.7 | HWiNFO shmem daemon + thermal watcher | [VERIFIED: `python --version` on Windows] |
| HWiNFO64 | installed by owner (version UNVERIFIED) | Sensor source via Shared Memory v2 | [CITED: hwinfo.com forum/spec mirrors] |
| powershell.exe | system (Win PS 5.1 assumed) | Interop bridge: toasts, kill path, host fingerprints | [VERIFIED: interop call succeeded this session] |

### Supporting
| Component | Version | Purpose | When to Use |
|---------|---------|--------------|-------------|
| taskset, stdbuf, flock | util-linux (present) | CPU isolation probe, unbuffered logs, session locking | librocdxg#60 busy-spin fingerprinting; concurrent-invocation protection |
| sha256sum / hashlib | present | CHECKSUMS.sha256, binary/model fingerprints | every manifest |
| pytest (guest) | install at Wave 0 [ASSUMED availability via pip] | Guard regression fixtures (D2-14 mandates a permanent test) | Validation Architecture below |
| BurntToast (optional) | latest [ASSUMED if chosen] | Prettier toasts | Only if raw WinRT XML deemed insufficient; zero-dep raw XML is the recommended default |
| Vulkan SDK (glslc) + VS2022 Build Tools | current [ASSUMED versions] | Vulkan comparator arm build | native-Windows arm only |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw WinRT toast XML via powershell.exe | BurntToast module | BurntToast needs module install + PS gallery; raw XML needs nothing and works today — pick BurntToast only for richer actions |
| Python shmem daemon | C# Hwinfo.SharedMemory.Net / Go hwinfo-go | Compiled helpers add a build step; Python mmap+struct is ~100 lines and matches host Python 3.14 already present |
| llama.cpp startup-log free-VRAM | custom hipMemGetInfo probe binary | Startup log already prints "(20421 MiB, 18245 MiB free)" — zero new code; probe binary is fallback if log parsing proves brittle |
| HWiNFO shmem | LibreHardwareMonitor | LHM lacks the documented stable shmem contract HWiNFO offers; keep as contingency only |

**Version verification:** All core component versions above were read from the machine or the pinned tree this session (see Sources). No registry installs occur; Package Legitimacy Gate not applicable beyond the note below.

## Package Legitimacy Audit

No external packages will be installed by this phase (stdlib-only Python + existing binaries). The only optional third-party item is the BurntToast PowerShell module — if the planner selects it, add a `checkpoint:human-verify` before `Install-Module`.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| (none — stdlib only) | — | — | — | — | — | N/A |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                    ┌────────────────────────── WINDOWS HOST ──────────────────────────┐
                    │                                                                   │
  HWiNFO64 ──shmem──▶ hwinfo_daemon.py          thermal_watchdog.py                     │
  (sensors)          │ 1 Hz poll                  │ junction ≥95°C                    │
                    │ telemetry/<cell>.csv        │                                   │
                    │                            ▼                                    │
                    │                     powershell.exe ──wsl.exe──▶ sh -c 'kill -9' ─┼──┐
                    │                     (toast on trips / summary)                  │  │
                    └──────────────▲───────────────────────────────────────────────────┘  │
                                   │ powershell.exe interop                               │
                                   │                                                      │
                    ┌──────────────┴────────────────────────── WSL2 GUEST ───────────────▼──┐
                    │                                                                        │
  prompt corpus ──▶ harness.py (orchestrator, holds PID file)                                │
                    │  1. fingerprint → manifest.json                                        │
                    │  2. pre-flight estimate (32k tier) → FAILED:preflight-oom?             │
                    │  3. per cell: spawn llama-bench / llama-cli (setsid, PID recorded)     │
                    │        stdout: -o jsonl (flush-per-row)   stderr: -v logs              │
                    │  4. rss_guard.py @1Hz: /proc/PID/status VmRSS+VmSwap                   │
                    │  5. enrich row (guard verdict, telemetry ref) → rows.jsonl + fsync     │
                    │  6. at close: CHECKSUMS.sha256, toast summary                          │
                    │                                                                        │
                    │  binaries: /root/llama.cpp/build-ci/bin/ (= archived baseline, cmp ✓)  │
                    │  model:    /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf (15.31 GB)  │
                    └────────────────────────────────────────────────────────────────────────┘
                                   │ results appended to
                                   ▼
              E:/Projects/qwen_3.8_27b_optimizations/benchmarks/results/<YYYYMMDD_HHMMSS>_<run-id>/
```

Trace: a cell starts → orchestrator writes PID file → host daemon slices telemetry to the run dir → guard polls RSS → row lands flushed in rows.jsonl → on trip/abort the watcher kills via wsl.exe and the row is marked FAILED — every arrow above was exercised or verified this session except the daemon itself (design-level).

### Recommended Project Structure
```
benchmarks/
├── environment/           # (exists) versions.txt, startup-log.txt, ...
├── prompts/               # D2-08 corpus: 6 deterministic files (short/long × code/prose)
├── bin/                   # harness entrypoints: run_session.py, run_prompts.py, calibrate.py
├── lib/                   # fingerprint.py, llabench.py (wrapper), guard.py, store.py, toast.py, preflight.py
├── tests/                 # guard fixture regressions (D2-14), wrapper unit tests
├── results/<YYYYMMDD_HHMMSS>_<run-id>/
│   ├── manifest.json      # D2-10 fingerprint
│   ├── rows.jsonl         # enriched rows, fsync-per-row
│   ├── telemetry/*.csv    # per-cell/host slices from shmem daemon or manual CSV
│   ├── logs/*.txt         # stderr captures per invocation
│   └── CHECKSUMS.sha256   # written at run close (D2-09)
└── RUNBOOK.md             # written protocol: thresholds from calibration, repeat policy, thermal pairing
```

### Pattern 1: Explicit-cell llama-bench invocations (the anti-trap pattern)
**What:** One invocation per context tier, with every default overridden.
**When to use:** Every matrix session (D2-16 batches cells into few invocations; D2-19 fixes order).
**Example:**
```bash
# Source: tools/llama-bench/llama-bench.cpp @ bb4caa75 (verified this session)
BIN=/root/llama.cpp/build-ci/bin/llama-bench
MODEL=/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf
for C in 4096 8192 16384 32768; do            # ctx ascending (D2-19)
  "$BIN" -m "$MODEL" \
    -p $C -n 0 -pg $C,128 \                    # -n 0 KILLS the default tg128 cell; -p overrides default 512
    -fa off,on \                               # fa off→on (D2-19)
    -r 5 --no-warmup=0 --delay 30 \            # NOTE: '--delay', NOT '-D' at this pin
    -ngl 99 -sm none -t <physical-cores> \     # match Phase-1 invocation shape
    -o jsonl -oe jsonl -v --progress \
    > rows.raw.jsonl 2> logs/bench_c${C}.txt   # stdout=results, stderr=logs+buffer lines
done
```
Internal enumeration order (verified): outer loops end `…flash_attn → … → depth → poll`, then instances emit as `[all -p values] → [all -n values] → [all -pg pairs]` — so within a tier: pp-off, pg-off, pp-on, pg-on... actually fa-major: all fa=off cells (pp then pg), then all fa=on cells (pp then pg). Deterministic ⇒ fine for D2-19 as long as tiers run in ascending order.

### Pattern 2: Crash-resilient enriched journal
```python
# after each cell completes (or is marked FAILED):
row = parse_jsonl_cell(rows_raw)               # includes avg_ts/stddev_ts AND samples_ts[] per rep
enriched = {**row,
  "backend_arm": "HIP",
  "guard": {"verdict": "OK|FAILED:suspected-spill", "vmrss_peak_kb": ..., "vmswap_peak_kb": ...},
  "telemetry_slice": "telemetry/cell_003.csv",
  "run_id": run_id}
with open("rows.jsonl","a") as f:
    f.write(json.dumps(enriched)+"\n"); f.flush(); os.fsync(f.fileno())
```
llama-bench's own jsonl printer already `fflush()`es each row (verified, llama-bench.cpp:1806) — even a SIGABRT mid-matrix preserves completed cells on disk.

### Anti-Patterns to Avoid
- **Passing only `-pg C,128`:** silently runs default `pp 512` AND banned empty-context `tg 128` cells (defaults `{512}/{128}` verified). Always pass `-p C -n 0`.
- **Using `-D` for cooldown:** unrecognized at this pin → argument error kills the invocation. Use `--delay`.
- **Assuming `-c` controls llama-bench context:** there is no such flag; ctx = p+n+d. Context control happens implicitly via `-p`/`-pg` (and explicitly via `-c` ONLY in llama-cli).
- **Parsing stdout for diagnostics:** stdout is pure jsonl; logs (incl. buffer sizes) only appear with `-v` and go to stderr.
- **Hard-coding HWiNFO sensor indices:** discover readings by label substring match at calibration (labels differ per board/locale); record the mapping in the run dir.
- **Retry loops around allocation failures:** forbidden (fail-fast policy; WSL#40732 hard-crash pattern). Pre-flight gate instead.
- **Launching benchmark from a dying session:** backgrounded guests processes die when their launching wsl.exe session exits; the orchestrator must stay alive for the whole run (or hold the VM alive), with PID file for the watcher.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Repeat statistics (mean±stddev, per-rep samples) | custom timing loop | llama-bench `-r 5 -o jsonl` | upstream computes avg/stddev AND emits `samples_ts[]` arrays; community-comparable |
| Prefill/decode split | blended measurement | `-p C` vs `-pg C,128` cells | methodology rule 4 enforced by construction |
| Per-op backend support matrix | hand-written op probes | `test-backend-ops support --output csv` | official mechanism (docs/ops.md pipeline), CSV schema verified |
| Hashing / checksums | custom hash code | sha256sum / hashlib | standard, auditable |
| Toast notifications | WinForms balloon hacks | raw WinRT XML via powershell.exe (AppId = PowerShell's own AUMID) or BurntToast | documented, zero-install path verified working via interop precedents |
| Free-VRAM probing | rocm-smi attempts (impossible) | llama.cpp device-init log line "(20421 MiB, 18245 MiB free)" | rocm-smi unsupported under ROCDXG; the log line is already empirical ground truth |
| Sensor reading | vendor DLLs | HWiNFO SM2 documented shared-memory map | stable public contract since v6.x, spec captured verbatim |

**Key insight:** llama-bench already implements 80% of BENCH-01's hard requirements correctly; the harness's real job is *constraint enforcement* (killing default cells, fixing order, gating VRAM) and *provenance attachment*, not re-measurement.

## Common Pitfalls

### Pitfall 1: Default-parameter contamination of the matrix
**What goes wrong:** wrapper omits `-n 0`/`-p` → extra `pp 512` and empty-context `tg 128` rows enter the canonical matrix.
**Why:** defaults `n_prompt={512}, n_gen={128}` are independent vectors, enumerated separately from `-pg`.
**How to avoid:** wrapper constructs argv programmatically; assert expected row count per invocation (4 cells/tier × fa = 4 rows... precisely: 2 test types × 2 fa = 4 rows per tier) before publishing.
**Warning signs:** rows with `n_gen=128 && n_prompt=0` or `n_prompt=512` in output.

### Pitfall 2: `-D` cooldown flag
**What goes wrong:** D2-16 shorthand `llama-bench -D` fails arg parsing at this pin (only `--delay` exists, verified line 1019).
**How to avoid:** use `--delay <seconds>`; note it sleeps *before each test*, so 30 s × 16 cells adds ~8 min/session — budget it.

### Pitfall 3: Warmup doubles prefill wall-clock
**What goes wrong:** session-time underestimate; owner sits waiting hours.
**Why:** warmup runs the FULL-size prompt pass (the reduced 32-token variant is commented out at this pin, verified lines 2351-2358). At ~111 t/s, 60 k pp tokens/tier-set × 2 (warmup+measure) dominates.
**How to avoid:** plan HIP session ≈ 1.5–2.5 h [ASSUMED arithmetic]; keep `--delay` modest; never disable warmup for published cells (D2-07 requires default-on).
**Warning signs:** first tier taking far longer than the measure-only estimate.

### Pitfall 4: OOM loses remaining cells mid-invocation
**What goes wrong:** ctx-creation failure returns 1 immediately; remaining instances of THAT invocation never run (verified `return 1` paths).
**How to avoid:** the pre-flight gate (D2-18) exists precisely for the 32k tier; keep one invocation per tier so blast radius ≤ 4 cells; rely on flush-per-row for completed cells.
**Warning signs:** fewer rows than expected in rows.raw.jsonl after a non-zero exit.

### Pitfall 5: HWiNFO shared-memory 12 h/day cap (non-Pro)
**What goes wrong:** shmem stops updating partway through a long day of sessions; telemetry mode silently degrades.
**How to avoid:** manifest records `telemetry-mode`; daemon asserts freshness (`poll_time` advancing); fall back to documented manual CSV logging (D2-01) when stale; schedule calibration + both arms within the daily budget (each session ~2–3 h fits comfortably).
**Warning signs:** `dwSignature=='DEAD'`, or `poll_time` frozen >2× `dwPollingPeriod`.

### Pitfall 6: Cross-boundary kill mistakes
**What goes wrong:** `wsl.exe -- kill <pid>` fails (kill is a shell builtin, not a binary); killing the wrong PID; killing a session whose parent exited already reap the target.
**How to avoid:** verified pattern: `wsl.exe -d Ubuntu-24.04 -u root -- sh -c 'kill -9 <pid>'`; orchestrator records the exact benchmark PID in a file the watcher reads; last resort `wsl --terminate` per D2-20.
**Warning signs:** watcher log showing "No such process" repeatedly.

### Pitfall 7: Silent spill mislabeled as slow decode
**What goes wrong:** DXG sysmem-fallback keeps tokens flowing at 5–10× collapse; number looks like a legitimate bad cell.
**Why:** microsoft/WSL#11050 — the disable-sysmem-fallback setting is ignored on some stacks [CITED: repo deep-research REPORT.md].
**How to avoid:** three-signal guard (D2-12) with calibrated thresholds (D2-13); VmSwap growth is an early tell (swap=16 GB configured).
**Warning signs:** tg cell throughput <50% of sibling fa/tier cell; VmRSS climbing during steady decode.

### Pitfall 8: Deprecated flags in the prompt-runner layer
**What goes wrong:** `--no-mmap` triggers deprecation warnings and couples oddly with other load flags at this pin.
**How to avoid:** prefer `--load-mode none` (verified value set: auto|none|mmap|mlock|mmap+mlock|dio); keep `setsid + --simple-io + --single-turn + explicit -c` from Phase-1 practice.

### Pitfall 9: CSV fallback parsing traps
**What goes wrong:** manual HWiNFO CSV parsed as UTF-8 → mojibake; column names shift between versions/locales.
**How to avoid:** HWiNFO CSVs are ISO-8859-1 encoded with 200+ columns [CITED: community parsers]; parser keys on header row, tolerates missing columns, records unmatched mandatory fields as telemetry-gap rather than fabricating zeros.

## Code Examples

### Exact JSONL row schema emitted by llama-bench at bb4caa75
```
// Source: tools/llama-bench/llama-bench.cpp get_fields() (verbatim, verified this session)
build_commit, build_number, cpu_info, gpu_info, backends, model_filename, model_type,
model_size, model_n_params, n_batch, n_ubatch, n_threads, cpu_mask, cpu_strict, poll,
type_k, type_v, n_gpu_layers, n_cpu_moe, split_mode, main_gpu, no_kv_offload, flash_attn,
devices, tensor_split, tensor_buft_overrides, load_mode, embeddings, no_op_offload,
no_host, fit_target, fit_min_ctx, n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns,
avg_ts, stddev_ts
// plus per-row arrays: "samples_ns": [...], "samples_ts": [...]  (raw per-repetition!)
// types: flash_attn INT(0/1); avg_ts/stddev_ts FLOAT; test_time STRING timestamp
```

### test-backend-ops support CSV columns
```
// Source: tests/test-backend-ops.cpp get_fields_csv() (verbatim, verified)
op_name, op_params, supported, error_message, test_mode, backend_reg_name, backend_name
// Usage: test-backend-ops support [-b <backend>] [--output console|sql|csv] [--list-ops]
// Gate ops: GATED_DELTA_NET, SOLVE_TRI, SSM_CONV, SSM_SCAN must be supported on Vulkan
```

### HWiNFO SM2 reader skeleton (host Python)
```python
# Source: hwisenssm2.h layout (official header mirrored in MintySensorMonitor; CITED)
import mmap, struct
MAP = r"Global\HWiNFO_SENS_SM2"
# header: sig(4s) ver rev poll_time(q) + offsets/sizes/counts (I×6) + poll_period(I)
# reading elem (pack=1): tType(i) idx(I) id(I) label[128] userLabel[128] unit[16] val,dmin,dmax,davg(5d)
# iterate readings; match labels by substring:
WANTED = ["core clock","memory clock","temperature","hot spot","board power",
          "fan","gpu utilization","gpu memory usage","shared gpu memory"]  # calibrate exact labels
# acquire Global\HWiNFO_SM2_MUTEX briefly around each snapshot; verify b"HWiS" signature;
# treat b"DEAD" as disabled-shared-memory.
```

### Greedy prompt-runner invocation (Layer 2)
```bash
# flags verified at pin: -st/--single-turn (arg.cpp:1903), --temp (2005),
# --load-mode none (2668-2705); headless pattern carried from Phase-1 evidence
setsid /root/llama.cpp/build-ci/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -c <tier> --temp 0 --single-turn --simple-io --load-mode none -ngl 99 \
  -p "$(cat benchmarks/prompts/short_code_01.txt)" -n 128 \
  > logs/prompt_short_code_01.out 2> logs/prompt_short_code_01.err &
echo $! > run/current.pid
```

## State of the Art

| Old Approach | Current Approach (at pin) | Impact for planner |
|--------------|--------------------------|--------------------|
| `--no-mmap` / `--mlock` | `--load-mode <auto|none|mmap|mlock|mmap+mlock|dio>` | use modern flag; old ones warn deprecated |
| `-D` delay short form (older llama.cpp) | `--delay <s>` only | update runbook wording |
| markdown/csv outputs only for stats | jsonl includes per-rep `samples_*` arrays | raw retention (D2-11) comes free from tool output |
| llama-bench `-c` (never existed) | ctx derived from p/n/d | matrix design already aligned (D2-06) |

**Deprecated/outdated:** `--mmap`/`--no-mmap`/`--direct-io` family (warn + translate internally); `AMDGPU_TARGETS` (legacy alias, irrelevant here — build already done).

## Runtime State Inventory

Not a rename/refactor/migration phase — omitted per protocol. (Greenfield infrastructure inside an established repo.)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | HWiNFO64 is (or will be) installed on the host with Shared Memory Support enabled; exact sensor labels on this card include hotspot + shared-GPU-memory entries | Environment Availability / telemetry daemon | Daemon falls back to `manual-fallback` CSV mode (already designed-in); calibration maps labels |
| A2 | Session wall-clock ≈1.5–2.5 h/arm (warmup-doubling arithmetic from 111 t/s pp) | Pitfalls 3 / protocol | Schedule slips; harmless, adjust plan |
| A3 | pytest installable in guest (pip) for guard regression tests | Validation Architecture | Fall back to stdlib unittest — fixtures matter, framework doesn't |
| A4 | BurntToast behavior/version if selected over raw XML | Toast discretion | Default path (raw WinRT XML) avoids this entirely |
| A5 | Adrenalin's Vulkan runtime handles this pin's shaders correctly on RX 7900 XT | Vulkan arm | Caught by the six-part D2-04 gate before any publish; arm simply reports partial-support honestly |
| A6 | HWiNFO CSV logging start/stop can be automated via CLI switches (`-l`, `-max_time`) | Manual fallback | Worst case: owner clicks the green [+] button manually (documented procedure) |
| A7 | `test_time` field is a UTC-ish timestamp suitable for time-joining (format string not fully read) | Time-joining | Daemon stamps rows with its own UTC clock anyway; skew check covers drift |
| A8 | libggml resolution: running archived copies directly works because they are byte-identical to build-ci/bin and loader picks up sibling .so files | Standard Stack | Set `LD_LIBRARY_PATH=/root/llama.cpp/build-ci/bin` explicitly in wrapper — cheap insurance regardless |

## Open Questions

**RESOLVED (phase-2 planning revision): Q1 via 02-02 `--label-map` + 02-04 calibration; Q2 via 02-03 documented buffer-math pre-flight choice; Q3 via new plan 02-05.**

1. **Exact HWiNFO sensor labels on this machine**
   - What we know: SM2 exposes label strings; mandatory field list is locked (D2-01).
   - What's unclear: precise labels for hotspot/shared-GPU-memory on this board/driver.
   - Recommendation: first calibration step = dump full reading inventory to `benchmarks/results/<calib>/telemetry/label-map.txt`; matcher uses that map thereafter.

2. **Pre-flight method choice (discretion area)**
   - Startup probe (llama-cli at target `-c`, parse buffer lines, ~1 model load) vs buffer math (weights 15.31 GB + KV ≈ 64 KiB/token f16 × C + compute buffer from prior tier observation) against the 18 245 MiB-free anchor.
   - Recommendation: buffer math for the go/no-go, startup probe to confirm at 24k boundary if capped; document choice in RUNBOOK.md.

3. **Vulkan arm llama-bench invocation parity**
   - What we know: same commit, same flags work on Windows build (`build/bin/Release/llama-bench`).
   - What's unclear: whether `-t` thread count and `--delay` need different values natively.
   - Recommendation: keep identical cell definitions (D2-19 comparability); threads = physical cores on both arms; record any deviation in manifest.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| WSL2 distro Ubuntu-24.04 (root) | everything guest-side | ✓ | verified live | — |
| ROCm 7.2.1 + librocdxg 1.2.2 | HIP arm | ✓ | frozen (versions.txt) | — |
| Pinned binaries + libs | harness target | ✓ | build-ci/bin == archive (cmp identical) | LD_LIBRARY_PATH insurance |
| Guest python3 | orchestrator/guard | ✓ | 3.12.3 | bash+awk (painful) |
| taskset/stdbuf/flock/sha256sum | isolation/logging/locking/hashes | ✓ | present | — |
| Host Python | shmem daemon/watcher | ✓ | 3.14.7 | PowerShell-only daemon (harder) |
| powershell.exe interop (both directions) | toasts, kill path, host fingerprints | ✓ | live-tested | none needed |
| wsl.exe kill path | thermal fail-safe | ✓ | live-tested (CROSS_KILL_OK) | `wsl --terminate` last resort |
| HWiNFO64 + Shared Memory Support | primary telemetry | ✗ UNVERIFIED on host | — | manual CSV fallback (D2-01) |
| pytest (guest) | guard regression suite | ✗ (install at Wave 0) | — | stdlib unittest |
| Vulkan SDK (glslc) + VS2022/MSVC on host | Vulkan arm build | ✗ UNVERIFIED | — | w64devkit path documented in-tree |
| HWiNFO 12h/day shmem budget | long session days | constraint, not absence | non-Pro assumed [ASSUMED] | spread sessions / CSV fallback |

**Missing dependencies with no fallback:** none blocking — HWiNFO absence degrades telemetry mode but the phase still completes via documented fallback.
**Missing dependencies with fallback:** HWiNFO (CSV), pytest (unittest), Vulkan SDK (w64devkit).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest on guest python3.12 [ASSUMED pip-available; else unittest] |
| Config file | none yet — Wave 0 creates `benchmarks/tests/` + `pytest.ini` |
| Quick run command | `python3 -m pytest benchmarks/tests/ -x -q` (guest, <30 s — pure fixtures, no GPU) |
| Full suite command | `python3 -m pytest benchmarks/tests/ -q` + `bash benchmarks/tests/smoke_matrix.sh` (1-tier dry run, GPU) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BENCH-01 | Wrapper produces exactly 4 rows/tier with correct p/n/fa fields; rejects contaminated defaults | unit (fixture jsonl parse) | `python3 -m pytest benchmarks/tests/test_llabench_wrapper.py -x` | ❌ Wave 0 |
| BENCH-01 | Re-run variance gate: |mean₁−mean₂|/mean ≤ 5% logic | unit (synthetic pairs) | `python3 -m pytest benchmarks/tests/test_repro_gate.py -x` | ❌ Wave 0 |
| BENCH-02 | Manifest contains all D2-10 fields non-empty; sha256 of binary/model/.wslconfig recomputable | unit + integration smoke | `python3 -m pytest benchmarks/tests/test_manifest.py -x` | ❌ Wave 0 |
| BENCH-02 | Shmem digest parser: fixture snapshot → mandatory-field dict; DEAD/stale signatures detected | unit (byte fixtures) | `python3 -m pytest benchmarks/tests/test_shmem_digest.py -x` | ❌ Wave 0 |
| BENCH-03 | Synthetic spiked RSS series ⇒ FAILED:suspected-spill (permanent regression, D2-14) | unit | `python3 -m pytest benchmarks/tests/test_guard_fixtures.py -x` | ❌ Wave 0 |
| BENCH-03 | rows.jsonl survives simulated SIGKILL (fsync-per-row proof) | unit (tmpdir) | `python3 -m pytest benchmarks/tests/test_journal_crash.py -x` | ❌ Wave 0 |
| BENCH-04 | Matrix assembler: 4 tiers × {pp,pg} × {off,on} ordering = D2-19; FAILED cells excluded-but-published | unit | `python3 -m pytest benchmarks/tests/test_matrix_assembly.py -x` | ❌ Wave 0 |
| BENCH-04 | Vulkan coverage gate: CSV contains ✅ GATED_DELTA_NET/SOLVE_TRI/SSM_CONV/SSM_SCAN | integration (GPU, Vulkan arm) | `bash benchmarks/tests/vulkan_gate.sh` (runs test-backend-ops support --output csv + grep) | ❌ Wave 0 (arm-dependent) |
| BENCH-03 | ONE supervised near-OOM live trip | manual-only | RUNBOOK §calibration — justified: deliberate OOM is destructive-by-design and owner-supervised (D2-14) | manual |

### Sampling Rate
- **Per task commit:** `python3 -m pytest benchmarks/tests/ -x -q`
- **Per wave merge:** full unit suite + 1-tier GPU dry run (4k tier only, ~3 min)
- **Phase gate:** full matrix sessions complete; reproducibility re-run within ±5%; all unit tests green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `benchmarks/tests/` scaffold + pytest.ini (or unittest discovery) — covers all unit rows above
- [ ] Fixture generators: synthetic llama-bench jsonl, spiked RSS traces, HWiNFO shmem byte snapshots, support-CSV samples
- [ ] `benchmarks/RUNBOOK.md` skeleton (thresholds filled at calibration)

## Security Domain

### Applicable ASVS Categories (Level 1, enforcement on)
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | local single-owner tooling; no auth surface |
| V3 Session Management | no | no sessions |
| V4 Access Control | yes (filesystem) | result store under repo; guest runs root-only per D-01 (accepted project posture); CHECKSUMS.sha256 provides tamper-evidence (D2-09) |
| V5 Input Validation | yes | validate PIDs as integers before interpolation into kill commands; strict parsers for llama-bench jsonl / HWiNFO CSV / support CSV (schema-verified above); reject unexpected row counts instead of coercing |
| V6 Cryptography | yes (hashing only) | sha256 via hashlib/sha256sum — never hand-rolled; no encryption needs |
| V7 Errors/Logging | light | FAIL reasons recorded verbatim in rows; logs retained per run dir |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Command/argument injection into powershell.exe/wsl.exe interop calls (toast text, PIDs) | Tampering/Elevation | pass args via `-File script.ps1 -Param` or base64-encoded command; XML-escape toast strings; integer-validate PIDs; never interpolate raw telemetry text into shells |
| Silent result tampering | Tampering | append-only + CHECKSUMS.sha256 at close; supersede-model corrections keep originals visible (D2-11) |
| Telemetry spoofing/staleness masquerading as data | Repudiation | daemon stamps UTC + verifies `poll_time` freshness; manifest records telemetry mode incl. `absent` |
| Resource-exhaustion harm to host (runaway benchmark) | DoS | pre-flight gate, fail-fast policy, thermal watchdog kill path (verified), session lock via flock |

## Sources

### Primary (HIGH confidence — read directly this session)
- Pinned tree `/root/llama.cpp` (HEAD `bb4caa75…`, clean): `tools/llama-bench/README.md`, `tools/llama-bench/llama-bench.cpp` (defaults block; `-pg` parse 582-592; instance builder 1320-1430; jsonl printer 1790-1807; log suppression 2224; delay 1019/2326; warmup 2351-2358; OOM returns 2306-2320), `common/arg.cpp` (1903, 2005, 2660-2705), `tests/test-backend-ops.cpp` (usage 10860+, CSV fields 1091-1101), `docs/build.md` (Vulkan section 397-470), `ggml/src/ggml-vulkan/vulkan-shaders/` inventory (168 shaders; gated_delta_net/solve_tri/ssm_conv/ssm_scan/dequant_iq4_xs present; no mul_mat_vec_iq4_xs)
- Repo artifacts: `benchmarks/environment/{versions.txt,llamacpp-pin.txt,startup-log.txt,vram-probe.txt}`, `baseline/binaries/v0.2.0-bb4caa75/` (ELF; cmp-identical to build-ci), `.planning/{ROADMAP,REQUIREMENTS,STATE}.md`, Phase-1 CONTEXT/SUMMARY, deep-research REPORT.md, `.claude/CLAUDE.md`
- Live probes: guest tooling versions; `.wslconfig` (memory=28GB, swap=16GB); interop calls both directions; cross-boundary kill (CROSS_KILL_OK); DXG free-VRAM line 20421/18245 MiB (startup-log.txt:1-2)

### Secondary (MEDIUM confidence)
- HWiNFO SM2 spec — official header mirrored at MintyMods/MintySensorMonitor (`hwisenssm2.h`), Seraksab/Hwinfo.SharedMemory.Net, MatthiasKunnen/hwinfo-go, zachstence/hwinfo-telegraf-plugin; 12 h/day non-Pro limit (pkg.go.dev plugin docs, ModBros FAQ)
- Toast patterns — Microsoft Learn desktop-toast quickstart + ToastNotificationManager API docs; GitHub30/toast-notification-examples (PowerShell AUMID); WSL toast precedents (ripleyeldridge/WSLNotifications, goropikari/win_notify, stuartleeks wsl-notify-send)
- HWiNFO CSV mechanics — bestware/xmg/lumion support articles; smoothfps walkthroughs; HWiNFO manual CLI switches; ISO-8859-1 note from weberjonathan/HWiNFO_Plotter

### Tertiary (LOW confidence)
- Wall-clock session estimates (arithmetic from measured 111.5 t/s pp) — verify against first calibration tier
- BurntToast specifics (not exercised; raw XML path is default)

## Metadata

**Confidence breakdown:**
- llama-bench/tool semantics: HIGH — read from pinned sources, not recalled
- Interop/kill/tooling availability: HIGH — live-probed this session
- Telemetry interfaces: MEDIUM — spec captured from authoritative mirrors; on-machine HWiNFO state unverified
- Protocol/timing estimates: LOW-MEDIUM — arithmetic on measured constants

**Research date:** 2026-08-23
**Valid until:** 2026-09-22 (pin-frozen stack; HWiNFO spec stable; revisit only if driver/ROCm pairing changes)
