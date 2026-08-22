# Phase 1 Plan Check — Checker Notes

**Checked:** 2026-08-22 · **Checker:** gsd-plan-checker · **Plans:** 01-01, 01-02, 01-03
**Method:** Goal-backward verification against ROADMAP.md Phase-1 success criteria, REQUIREMENTS.md ENV-01..04, CONTEXT.md decisions D-01..D-07, RESEARCH.md facts. Structure validated via `gsd-tools query verify.plan-structure`; WSL runnability probed live.

## VERDICT: NEEDS_FIXES

**Issues:** 2 blocker(s), 6 warning(s), 4 info

---

## Blockers (must fix before execution)

**B1. [task_completeness] All three plans use a non-conforming task schema — unexecutable by /gsd-execute-phase tooling**
- Plans: 01-01, 01-02, 01-03 (every task)
- Evidence: `verify.plan-structure` returns `"valid": false` for all three plans. Every task uses a non-standard `<commands>` element instead of `<action>`, and every task is missing `<files>` and `<done>`. Frontmatter is missing required fields `phase`, `plan`, `type`, and `must_haves` (a `## must_haves` prose section exists in the body, but the structured consumer reads frontmatter). Example error set (01-01): 9 errors, 10 warnings; identical pattern in 01-02 (9 errors) and 01-03 (9 errors, 12 warnings).
- Why blocker: the executor parses `<action>`/`<files>`/`<done>` and frontmatter `must_haves`. As written, tasks cannot be parsed — execution degrades to improvisation with no acceptance criteria recorded, and post-phase verification loses its machine-readable contract. This is precisely the planner-persona quality gap flagged when the orchestrator authored plans directly.
- Fix hint: convert each `<commands>` into `<action>` (keep the exact wsl.exe/powershell command strings — they are good), add `<files>` (paths touched) and `<done>` (measurable outcome) per task, and hoist `must_haves` (truths/artifacts/key_links) into YAML frontmatter plus add `phase: 1`, `plan: 01-0N`, `type: auto`.

**B2. [context_compliance] D-04 is half-delivered: no task pauses Windows Adrenalin auto-updates**
- Plans: none (missing everywhere)
- Evidence: D-04 (locked): "**Pause/hold** Windows Adrenalin auto-updates for the project duration; **record** the current driver version (26.10.41 / 32.0.31041.1004)". Recording is implemented (01-01 T5 writes the pairing into `versions.txt`). The pause mechanism appears in zero tasks/commands across all three plans. The ROADMAP Phase-1 risk notes likewise mandate pausing Adrenalin auto-updates.
- Why blocker: a forced mid-phase driver update invalidates the frozen pairing, re-triggers ENV gates, and silently voids every fingerprint/benchmark this phase produces — the decision's primary protection is absent while its cosmetic half ships.
- Fix hint: add a small pre-build task (Windows side): pause Adrenalin/driver auto-update (e.g., wushowhide/Group Policy/"Pause updates" + AMD "Notify me only"), record the applied setting alongside the frozen pairing in `versions.txt`.

---

## Warnings (should fix)

**W1. [verify-format] 01-01 Task T4 verify has a quoting bug that guarantees a false result**
- Plan: 01-01, task T4
- Evidence: verify instructs `bash -lc "echo $HSA_ENABLE_DXG_DETECTION"` with double quotes. Driven from the Windows host through bash, `$HSA_ENABLE_DXG_DETECTION` expands host-side (empty) before WSL sees it. Live probe reproduced this: double-quoted form prints an empty line (exit 0); correct escaped form prints the guest value. The check as written can never print `1`.
- Fix hint: single-quote or escape: `bash -lc 'echo "$HSA_ENABLE_DXG_DETECTION"'`.

**W2. [key_links] 01-03 Task T6 verify demands output no command produces**
- Plan: 01-03, task T6
- Evidence: verify requires "path+size appended to benchmarks/environment/versions.txt", but the `<commands>` block only creates the snapshot dir, terminates the distro, exports, and `ls`s — nothing appends to versions.txt.
- Fix hint: add the append command (`echo "snapshot: ubuntu-2404-rocm721-phase1.tar $(stat -c%s …)" >> …/versions.txt`) to the task commands.

**W3. [verification_derivation] ENV-03 gate verify is vague and carries flag risk (planner-quality gap)**
- Plan: 01-03, task T4
- Evidence: (a) "`grep -ciE "cpu.*buffer|offload" must be ~0 except ngl=99 statement`" — "~0" is not an objective threshold and the pattern will match the legitimate "offloaded 63/63 layers" line; needs explicit expected-lines/forbidden-lines enumeration (e.g., assert absence of `CPU buffer size` tensor-buffer lines, presence of full VRAM buffer total ≈ 15.31 GB). (b) "coherent tokens emitted" is subjective with no automated check (e.g., require ≥N ASCII tokens and exit code 0). (c) `--no-conversation` flag validity at pin v0.2.0 is unverified — if renamed upstream the gate hard-fails; pin the exact CLI flags or add fallback. (d) OOM ladder `-ngl {95,90,85…}` is described in verify only, with no concrete loop, and — critically — has **no D-02 branch**: if OOM is caused by the ROCm#6022-class `.wslconfig` RAM clamp, the plan silently accepts partial residency, violating the "fully resident" goal; before settling for reduced `-ngl`, check/apply evidence-driven `.wslconfig` tuning as D-02 authorizes.
- Fix hint: enumerate exact log-line assertions, replace "~0"/"coherent" with measurable predicates, and add the D-02 escalation step ahead of the -ngl descent.

**W4. [requirement_coverage] SC1's "minimal HIP smoke program executes on device" has no wave-1 task**
- Plans: 01-01 (gap), 01-02 (indirect)
- Evidence: ROADMAP SC1 = rocminfo enumerates gfx1100 **AND** a minimal HIP smoke program executes successfully. 01-01's T5 gate captures rocminfo/hipconfig only — first actual device-code execution happens one wave later (01-02 T4 `test-backend-ops`). If HIP runtime init fails under DXG, the platform kill-gate fires a full wave late.
- Fix hint: append a trivial device-execution proof to 01-01 T5 (e.g., compile+run a 5-line HIP hello or `rocminfo`+`hipRuntimeGetVersion` smoke binary) so the kill-gate is complete within wave 1.

**W5. [context_compliance] D-06 <750 MB enforcement is asserted but not implemented**
- Plan: 01-02, task T4
- Evidence: verify mentions "repo size still <750MB after add (du -sh check)" but no command performs the measurement, and no conditional branch implements the D-06 consequence (breach ⇒ move binaries outside git + gitignore path).
- Fix hint: execute the du check inside the task and state the breach branch explicitly.

**W6. [scope_sanity] Task counts above target: 5 / 5 / 6 tasks per plan (target 2–3)**
- Plans: all three
- Mitigation considered: each task is one atomic shell command and plans are ~4 KB each, so context-budget risk is low for this env-provisioning phase; classified warning rather than blocker on that basis, but consolidation is advisable during the B1 rewrite (e.g., fold 01-01 T1 apt-prereqs into T2; fold 01-03 T5 vram-probe record into T4).

---

## Info (non-blocking)

**I1.** SC2 "…and run": `llama-bench`/`llama-perplexity` are built but never executed in Phase 1; executability is evidenced via `test-backend-ops` (01-02 T4) and `llama-cli` (01-03 T4). Full runs belong to Phase 2 (BENCH-*) — acceptable; record the interpretation in SUMMARY.md.
**I2.** ROADMAP sanctions running artifact-download concurrently with the pinned build after stack install; 01-03 instead serializes behind 01-02. Safe (gate T4 needs binaries), just slower than designed — fine to keep.
**I3.** Snapshot precondition unstated: `wsl --export` writes a multi-GB tarball to `E:\wsl-snapshots` — verify free disk space in-task. Also note `.gitignore` already excludes `baseline/binaries/` entirely (untracked), while D-06's letter implies binaries sit *in* git until the 750 MB breach — clarify intent once.
**I4.** Version-string drift across artifacts: ROADMAP/REQUIREMENTS name "Adrenalin 26.2.2-for-WSL2"; CONTEXT/D-04 freezes the actually-installed 26.10.41 (32.0.31041.1004). Plans correctly follow D-04; consider a roadmap footnote to prevent future confusion.

---

## Dimension Summary

| Dimension | Result |
|---|---|
| 1 Requirement coverage | PASS with W4 (ENV-01 smoke gap; ENV-02/03/04 covered by tagged tasks with objective verifies) |
| 2 Task completeness | **FAIL (B1)** — schema non-conforming across all plans |
| 3 Dependency correctness | PASS — waves 1→2→3, depends_on valid, acyclic, snapshot serial-last |
| 3b Undeclared coupling | PASS — shared entity versions.txt created wave 1, appended wave 3, wave-ordered |
| 4 Key links | FAIL (W2) + wiring otherwise present (tee/redirects, cross-plan binary reuse declared via deps) |
| 5 Scope sanity | WARNING (W6) — 5/5/6 tasks, tiny footprint |
| 6 must_haves derivation | FAIL via B1 (frontmatter missing); body truths are user-observable where present |
| 7 Context compliance | **FAIL (B2)** — D-04 half-delivered; D-01/02/03/05/07 honored (D-06 → W5) |
| 7b Scope reduction | Covered by B2 (silent omission of pause action); no invented v1/v2 reductions found |
| 7c Arch tier compliance | SKIPPED (no responsibility map in RESEARCH.md) |
| 8 Nyquist | SKIPPED (RESEARCH.md has no Validation Architecture section) |
| 9 Cross-plan data contracts | PASS — versions.txt create-then-append is wave-ordered; models/*.gguf gitignored (verified live) |
| 10 CLAUDE.md compliance | PASS — stack guidance matches plan choices (Ubuntu 24.04, gfx1100 single-target, hipconfig capture) |
| 11 Research resolution | PASS — no Open Questions section remains unresolved |
| 12 Pattern compliance | SKIPPED (no PATTERNS.md) |
| Verify cmd format (#1478/79) | WARNING (W1 quoting); no swallowed-error comparisons found |
| Numeric claim authority (#1480) | PASS — byte size/sha256/pin SHA match MODEL-DECISION.md lock record; no stale-measure conflicts |

**Live probes performed:** distro reachable as root (`wsl.exe … -u root -- bash -lc 'echo WSL_OK && whoami'` → `WSL_OK`/`root`); T4 quoting defect reproduced (double-quote form prints empty line); `.gitignore` confirms `models/*.gguf` + `baseline/binaries/` excluded.

## Recommendation

2 blockers require revision before `/gsd-execute-phase`. Both are mechanical: (1) re-shape tasks to the standard `<action>/<files>/<done>` + frontmatter `must_haves` schema — keep the existing command strings, which are well-constructed; (2) add the D-04 Adrenalin update-pause task. Address W1–W5 in the same pass; they are small edits to tasks already being rewritten.
