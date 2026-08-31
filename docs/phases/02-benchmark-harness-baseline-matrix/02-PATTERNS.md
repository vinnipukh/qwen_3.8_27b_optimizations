# Phase 2: Benchmark Harness & Baseline Matrix - Pattern Map

**Mapped:** 2026-08-23
**Files analyzed:** 24 planned outputs (10 harness modules/scripts, 6-prompt corpus, 8 test files + scaffold, RUNBOOK, result-store artifacts)
**Analogs found:** 11 / 24 have a usable in-repo analog — this repo is nearly greenfield for executable code; the strongest patterns are *process/evidence* conventions from Phase 1, not Python code. Where no analog exists, the RESEARCH.md code examples are named explicitly as the source of truth.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `benchmarks/bin/run_session.py` | controller/orchestrator | batch + subprocess-spawn | `01-environment-validation-stock-baseline/01-03-PLAN.md` T4 task pattern (spawn→tee→assert predicates) | partial (process pattern only; no Python exists) |
| `benchmarks/bin/run_prompts.py` | controller | request-response (single-turn runs) | Phase-1 headless invocation (`setsid`+`--simple-io`+`--single-turn`), 01-03-SUMMARY.md:20-25 | partial |
| `benchmarks/lib/llabench.py` | service (tool wrapper) | batch (argv→jsonl parse) | no code analog — **use RESEARCH.md Pattern 1 verbatim flag semantics** | role-match to RESEARCH only |
| `benchmarks/lib/fingerprint.py` | utility (transform→JSON) | transform | `benchmarks/environment/versions.txt` + `llamacpp-pin.txt` | exact (field-for-field) |
| `benchmarks/lib/guard.py` | monitor/service | event-driven (1 Hz poll) | `benchmarks/environment/hipsmoke.cpp` (fail-fast exit-code style) | role-match |
| `benchmarks/lib/store.py` | persistence/model | file-I/O append-only | `models/README.md` (sha256-of-record discipline) + 01-03 hash-gate | role-match (checksum half exact; fsync journal greenfield → RESEARCH Pattern 2) |
| `benchmarks/lib/toast.py` | utility (interop bridge) | event-driven notification | no analog — greenfield (RESEARCH security rules apply) | none |
| `benchmarks/lib/preflight.py` | utility | transform (buffer math vs anchor) | `benchmarks/environment/vram-probe.txt` (verdict-line format + DXG anchor values) | role-match |
| `benchmarks/host/hwinfo_daemon.py` | service/daemon | streaming poll @1 Hz | no analog — greenfield (**use RESEARCH HWiNFO SM2 skeleton**) | none |
| `benchmarks/host/thermal_watchdog.py` | middleware/watchdog | event-driven (threshold trip) | no code analog — kill path string is live-tested evidence in 01/02 CONTEXT | none |
| `benchmarks/prompts/*.txt` (~6) | fixture data | static | `models/README.md` determinism-by-hash precedent | partial (naming/hashing convention only) |
| `benchmarks/results/<ts>_<run-id>/*` | generated artifacts | file-I/O append-only | `benchmarks/environment/*` capture conventions | partial |
| `benchmarks/tests/test_llabench_wrapper.py` … `test_matrix_assembly.py` (7 files) | test | unit (fixtures) | no test framework exists — Wave 0 greenfield | none |
| `benchmarks/tests/fixtures/*` generators | test fixture | synthetic data | no analog — greenfield | none |
| `benchmarks/tests/vulkan_gate.sh` (+ optional `smoke_matrix.sh`) | integration gate script | batch | `docs/TESTING.md` gate-doctrine + `scrape_sahibinden.sh` (only house bash) | role-match (doctrine exact, style partial) |
| `benchmarks/RUNBOOK.md` | protocol doc | static | `benchmarks/environment/vram-probe.txt` + `versions.txt` amendment style | exact (format) |
| `pytest.ini` (Wave 0 scaffold) | config | static | none — greenfield | none |
| Vulkan arm build/run records (`benchmarks/vulkan/…`) | config/provenance | static capture | `benchmarks/environment/llamacpp-pin.txt` | exact (record format) |

---

## Pattern Assignments

### `benchmarks/lib/fingerprint.py` (utility, transform)

**Analog:** `benchmarks/environment/versions.txt` (whole 18-line file) and `benchmarks/environment/llamacpp-pin.txt`

The manifest.json is these plain-text captures formalized into JSON. Copy the *key vocabulary and completeness bar* verbatim — every key below already has a proven capture method from Phase 1.

**Env-capture key:value style** (`versions.txt:1-8`):
```
ROCm: 7.2.1
librocdxg: 1.2.2 (rocdxg-roct, rocdxg-amd-smi-lib)
Kernel: guest-reported-in-hipconfig
Driver(frozen D-04): 32.0.31041.1004 (Adrenalin 26.10.41) - Windows-side verified via WMI
HIP-smoke: PASS (RESULT=1 ARCH=gfx1100 NAME=AMD Radeon RX 7900 XT, exit 0)
captured: 2026-08-22T11:23Z UTC
```

**Build-provenance keys** (`llamacpp-pin.txt:1-7`) — map 1:1 into manifest D2-10 fields:
```
pin-tag: v0.2.0
commit: bb4caa7540188872173c44d161602d9271386413
configure: -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF (+ccache)
source-tree: guest ext4 /root/llama.cpp (DrvFs git-lock incompatibility documented)
compiler: gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 / hipcc HIP version: 7.2.53211-e1a6bc5663
```

**Convention:** every capture ends with a UTC timestamp line (`captured: <ISO8601>Z UTC`). Manifest must carry the same UTC-stamp discipline plus a `telemetry-mode` field (`shmem|manual-fallback|absent`) mirroring how `versions.txt` degrades gracefully (`updates-paused: PENDING-OWNER-ACTION`).

---

### `benchmarks/lib/store.py` (persistence, append-only file-I/O)

**Analog (checksum half):** `models/README.md:5-13` — provenance-of-record with verification trail:
```markdown
| sha256 | `53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5` |
| Verified | 2026-08-22 — `sha256sum -c` OK in WSL guest (Phase 1, plan 01-03 T1) |
| HF revision | `dee0a3164d9e11bbbebf5b63f52ba99443d14fc3` (lastModified 2026-08-16) |
```

**Analog (hash-gate procedure):** `01-03-PLAN.md:36` — expected-hash file staged into guest via base64 pipe, then `sha256sum -c`; mismatch ⇒ delete + ONE retry ⇒ FAIL. Apply the same one-shot-then-fail semantics when verifying binary/model digests at session start (never silent re-hash loops).

**Journal half — no repo analog:** use RESEARCH.md Pattern 2 verbatim (append enriched row → `flush()` + `os.fsync(f.fileno())` per row). Note upstream already fflushes per row (`llama-bench.cpp:1806`), so crash-resilience is inherited for raw cells; store.py adds the enrichment layer.

**Checksum-at-close:** emit `CHECKSUMS.sha256` in `sha256sum -c` compatible format so any later audit replays the exact Phase-1 verification command (`01-03-SUMMARY.md:25`: "sha256sum -c → OK").

---

### `benchmarks/bin/run_session.py` (controller/orchestrator, batch + spawn)

**Analog:** Phase-1 task-execution pattern — `01-03-PLAN.md:61` (T4):
> run llama-cli …, tee full output to benchmarks/environment/startup-log.txt. OBJECTIVE PASS PREDICATES on startup-log.txt: (a) contains … (b) contains … (c) ZERO lines matching … (d) process exit code 0 AND …

Copy this *shape*: every spawned invocation (1) tees full stderr to a log file in the run dir, (2) is followed by explicit machine-checkable assertions (row-count == 4 per tier; zero banned-row signatures `n_prompt=512` or `n_gen=128 && n_prompt=0`), (3) records verdict even on failure. This is the anti-trap enforcement point for D2-06 (defaults `{512}/{128}` contamination).

**Spawn conventions from Phase 1 lessons** (`01-03-SUMMARY.md:20-25`):
```
2. llama-cli hang blocked in `n_tty_write` on dead PTY → headless runs need `setsid` + `--simple-io`
5. v0.2.0 interactive-mode default caused prompt-flood → use `--single-turn`
```
Apply to every guest-side child process: `setsid`, PID written to a PID file the host watcher reads (D2-20 kill path), stdout/stderr separated (stdout = pure jsonl, stderr = logs — RESEARCH anti-pattern "Parsing stdout for diagnostics").

---

### `benchmarks/bin/run_prompts.py` (controller, request-response)

**Analog:** same headless invocation family as above, updated per RESEARCH Pitfall 8: replace Phase-1's `--no-mmap` with `--load-mode none` (verified value set at pin); keep `setsid + --simple-io + --single-turn + explicit -c --temp 0`. Full command template in RESEARCH "Greedy prompt-runner invocation (Layer 2)" — copy it directly; there is no in-repo script to copy from.

Prompt text loads from `benchmarks/prompts/<file>.txt`; corpus determinism proven the `models/README.md` way — record each prompt file's sha256 in the run manifest.

---

### `benchmarks/lib/guard.py` (monitor, event-driven poll) & `benchmarks/lib/preflight.py`

**Analog (style):** `benchmarks/environment/hipsmoke.cpp:6-21` — the repo's established micro-tool idiom: tagged single-line output + distinct exit codes per failure class:
```cpp
if (hipMalloc(&d, 4) != hipFailure…) { printf("HIPMALLOC-FAIL\n"); return 1; }
…
printf("RESULT=%d ARCH=%s NAME=%s\n", h, prop.gcnArchName, prop.name);
return (h == 1) ? 0 : 2;
```
Adopt for guard/preflight verdicts: machine-tagged verdict strings (`OK`, `FAILED:suspected-spill`, `FAILED:preflight-oom`, `FAILED:thermal-abort`) that land verbatim in rows.jsonl, matching the locked D2-12/D2-18/D2-20 vocabulary. Fail-fast: no retry loops around allocation failures (RESEARCH anti-pattern).

**Analog (verdict-record format):** `vram-probe.txt:2-7`:
```
verdict: PASS
evidence: 132 layers assigned to ROCm0 (gfx1100); CPU-assigned layers: 0
throughput: prompt ~111.5 tok/s @53tok; decode ~33.5 tok/s (32 tok sample)
flags: -ngl 99 -c 2048 --no-mmap --single-turn; exit 0
```
This verdict/evidence/flags triple is the house format for any go/no-go record — reuse it for the pre-flight estimate record and guard-trip annotations. The free-VRAM empirical anchor lives at `startup-log.txt:1-2` ("(20421 MiB, 18245 MiB free)") — preflight math compares against this measured number, not spec sheet.

**/proc polling + thresholds:** greenfield — thresholds come from calibration (D2-13), never hard-coded guesses; RUNBOOK carries them post-calibration.

---

### `benchmarks/host/hwinfo_daemon.py` + `thermal_watchdog.py` (host-side services)

**No analog — greenfield.** No Windows-side scripts exist anywhere in the repo. Use RESEARCH.md "HWiNFO SM2 reader skeleton" as the implementation seed (~100 lines mmap+struct, label substring matching calibrated against a dumped label-map, `HWiS`/`DEAD` signature checks, mutex-brief snapshots, ISO-8859-1 for any CSV fallback parsing).

Kill-path string is *verified live evidence*, not a guess — carry verbatim (CONTEXT D2-20 + research): `wsl.exe -d Ubuntu-24.04 -u root -- sh -c 'kill -9 <pid>'`; last resort `wsl --terminate`. Security rule from RESEARCH: integer-validate PIDs before interpolating into any interop command.

**Interop quoting precedent:** `01-01-PLAN.md:47` — "Write /etc/profile.d/rocdxg.sh via base64 pipe (Git-bash mangles $VAR through wsl.exe args)". Any toast/kill argument passed across the boundary should prefer `-File script.ps1 -Param` form or base64 encoding over inline quoting.

---

### `benchmarks/tests/vulkan_gate.sh` (+ smoke scripts) — bash conventions

**Doctrine analog:** `docs/TESTING.md:9-14` — gates order performance claims:
```markdown
| Tier | Tool | When required |
| Op-level | `test-backend-ops` (ROCm0 backend) | Green **before any performance claim** is accepted |
```
Plus the governing sentence (line 6-7): *"A kernel or code change is not 'done' until every applicable gate passes and the numbers are published — failures are published too."* This is exactly D2-04 (six-part gate BEFORE any Vulkan publish) and D2-11/D2-18 (FAILED rows still land in the store).

**Style analog:** `scrape_sahibinden.sh` is the only existing bash in the repo. House idioms worth copying (lines 20-26):
```bash
for p in $(seq 1 22); do
  F="$OUT_DIR/page_$p.json"
  [ -s "$F" ] && grep -q '"c"' "$F" && { echo "skip $p"; continue; }   # resume-safe skip
  for attempt in 1 2 3 4; do                                           # bounded retries
    …
    echo "page $p OK ($ROWS rows) attempt=$attempt"                    # per-step status echoes
```
Resume-skip-on-existing-output and per-step status echoes fit gate scripts well (re-running `vulkan_gate.sh` after a partial failure shouldn't redo archived CSVs). Do NOT inherit its unquoted-variable looseness or missing `set -euo pipefail` for anything touching PIDs/paths; note it deliberately lacks strict mode.

CSV archive target mirrors Phase-1 precedent `test-backend-ops-phase1.txt` (raw tool output stored verbatim as evidence file).

---

### `benchmarks/prompts/*.txt` corpus

**Analog:** `models/README.md` determinism discipline — the model is gitignored but its identity is pinned by sha256 + HF rev in a tracked README. Same treatment: prompts are ordinary tracked files, but their sha256s enter each manifest so any corpus drift is detectable. No content analog exists (greenfield prose within D2-08 constraints).

### Tests (`benchmarks/tests/*`, `pytest.ini`, fixtures) — Wave 0

**No analog — greenfield.** Repo has zero test files/framework. Validation Architecture section of RESEARCH.md is the spec: framework choice (pytest, unittest fallback), the seven named unit/integration test files mapped to BENCH-01..04, quick-run commands, and the manual-only exception (supervised near-OOM). Fixture-first testing (synthetic spiked RSS series asserted → FAILED verdict) has no precedent here — follow RESEARCH directly.

### `benchmarks/RUNBOOK.md` + result-store artifacts

**Format analog:** `versions.txt` amendment-block style — decisions evolve in place with dated owner amendments rather than rewrites:
```
D-04 AMENDMENT (owner, 2026-08-22): scope reduced to no-SILENT-driver-updates.
Applied when shell elevated: …
Detection net: BENCH-02 driver fingerprint on every result row + ENV gates re-run on drift.
```
RUNBOOK threshold sections start as TODO placeholders filled at calibration (D2-13), using this dated-amendment style for post-calibration edits. Result-run directories inherit the `benchmarks/environment/*` habit: raw tool output kept verbatim alongside human-readable summary records.

### Modified files

- `.gitignore` — likely addition only if telemetry/logs threaten the D-06 750 MB in-repo budget (results are meant to be tracked while small; binaries/models precedent shows the ignore-+-README-pointer fallback).
- `docs/ARCHITECTURE.md` repository-layout tree gains `benchmarks/{bin,lib,host,prompts,tests,results}` entries (cosmetic, end of phase).

---

## Shared Patterns

### S1 — Evidence-capture artifact format
**Source:** `benchmarks/environment/versions.txt`, `vram-probe.txt`, `llamacpp-pin.txt`
**Apply to:** `manifest.json` field naming, guard/pre-flight verdict records, RUNBOOK, Vulkan arm provenance.
Key:value lines, UTC `captured:` stamp on everything, `verdict:/evidence:/flags:` triple for pass-fail records, dated amendment blocks instead of rewrites, graceful degradation states recorded (`PENDING-OWNER-ACTION`, future `telemetry-mode: manual-fallback`) rather than omitted.

### S2 — Hash-of-record discipline
**Source:** `models/README.md:5-13` + `01-03-PLAN.md:36` (base64-pipe staging, `sha256sum -c`, mismatch⇒one retry⇒FAIL)
**Apply to:** CHECKSUMS.sha256 generation, binary/model/.wslconfig/prompt-corpus digests in every manifest, Vulkan-vs-HIP binary identity proof (cmp/sha256 equality like Phase-1's archive==build-ci check).
Never hand-roll hashing (RESEARCH "Don't Hand-Roll").

### S3 — Objective pass predicates on captured logs
**Source:** `01-03-PLAN.md:61` T4 predicate list style
**Apply to:** wrapper row-count assertions (exactly 4 rows/tier; reject banned default-cell signatures), Vulkan residency gate (132 layers on device / 0 on CPU from verbose log), smoke-test coherence checks. Assertions are enumerated (a)(b)(c)(d)-style before the run, checked mechanically after.

### S4 — Headless guest-process invocation shape
**Source:** `01-03-SUMMARY.md:20-25` lessons; refined by RESEARCH Pitfall 8
**Apply to:** every guest-side spawn in `run_session.py` / `run_prompts.py`.
`setsid` detachment + PID file + `--simple-io` + `--single-turn` (interactive tools only) + explicit `-c` + `--load-mode none` (not deprecated `--no-mmap`). Orchestrator stays alive for the whole session (backgrounded children die with launching wsl.exe session).

### S5 — Tagged-output + fail-fast exit-code taxonomy
**Source:** `benchmarks/environment/hipsmoke.cpp`
**Apply to:** guard.py, preflight.py, gate scripts, orchestrator error handling.
Single-line tagged verdicts (`VERDICT=…`), distinct exit codes per failure class, no retries around allocation failures. Verdict strings match the locked vocabulary: `FAILED:suspected-spill`, `FAILED:preflight-oom`, `FAILED:thermal-abort`.

### S6 — Publish failures honestly (gate doctrine)
**Source:** `docs/TESTING.md:6-7` ("failures are published too") + `docs/TESTING.md:9-14` tier table
**Apply to:** matrix assembly and Vulkan gate ordering — op-level/backend-support gates precede any performance publish; FAILED/expected-fail rows are still appended to rows.jsonl with reason verbatim (D2-11 supersede model keeps originals visible).

### S7 — Cross-boundary interop safety
**Source:** `01-01-PLAN.md:47` (base64 pipe because Git-bash mangles `$VAR` through wsl.exe args) + RESEARCH ASVS V5 threat table
**Apply to:** toast.py, thermal watchdog kill path, host-fingerprint fetches. Prefer `-File x.ps1 -Param value` over interpolated inline commands; integer-validate PIDs; XML-escape user-visible toast text; never interpolate raw telemetry strings into shells.

---

## No Analog Found

Planner must pull these directly from RESEARCH.md (sections named) — inventing repo-based patterns for them would be fabrication:

| File | Role | Data Flow | Reason | Substitute source |
|------|------|-----------|--------|-------------------|
| `benchmarks/lib/llabench.py` | service | batch | No wrapper code exists; correctness lives in exact flag semantics | RESEARCH Pattern 1 + Anti-Patterns (Pitfalls 1–2) |
| `benchmarks/host/hwinfo_daemon.py` | service | streaming | First Windows-side script in repo | RESEARCH HWiNFO SM2 skeleton |
| `benchmarks/host/thermal_watchdog.py` | middleware | event-driven | ditto | RESEARCH diagram + verified kill-path string |
| `benchmarks/lib/toast.py` | utility | event-driven | No notification code exists | RESEARCH "Don't Hand-Roll" toast row (raw WinRT XML default) |
| `benchmarks/tests/**` + `pytest.ini` | test | unit | Zero test infrastructure in repo | RESEARCH Validation Architecture (full test map) |
| `benchmarks/prompts/*.txt` | fixture | static | No corpus precedent | D2-08 constraints only |
| fsync-per-row journal mechanics | persistence | append-only | No journaling code exists | RESEARCH Pattern 2 (verbatim) |

---

## Metadata

**Analog search scope:** `benchmarks/`, `baseline/`, `models/`, `docs/`, `src/` (empty), `docs/research/freetoken-probe/src/` (C++ probe, vendored toolchain — reviewed, rejected as analog: different language, unrelated domain, gitignored bins), root-level `scrape_sahibinden.sh`, `docs/phases/01-*` summaries/plans/checker-notes, `.agents/skills/` (rocm-doctor, magpie-kernel-evaluator, firecrawl-deep-research — generic tool packs, not code-convention sources), `.claude/CLAUDE.md` project instructions.
**Files scanned:** ~35 (all small text/doc files; binaries and 5k+ line logs excluded per scope guidance — `startup-log.txt` and `test-backend-ops-phase1.txt` cited from prior research line references only).
**Key finding:** the repo's transferable patterns are *evidence and process conventions* (capture format, hashing, predicates, headless spawn, gate doctrine) established in Phase 1 — all executable-code patterns for Phase 2 come from RESEARCH.md's pinned-source-verified examples.
**Pattern extraction date:** 2026-08-23
