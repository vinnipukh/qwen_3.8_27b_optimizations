---
phase: 2
slug: benchmark-harness-baseline-matrix
# status lifecycle: draft (seeded by plan-phase) → validated (set by validate-phase §6)
# audit-milestone §5.5 distinguishes NOT-VALIDATED (draft) from PARTIAL (validated + nyquist_compliant: false) (#2117)
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-23
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest on guest python3.12 (fallback: unittest if pip unavailable — Wave 0 confirms) |
| **Config file** | none — Wave 0 installs `benchmarks/tests/` + `pytest.ini` |
| **Quick run command** | `wsl -d Ubuntu-24.04 -u root -- sh -c 'cd /mnt/e/Projects/qwen_3.8_27b_optimizations && python3 -m pytest benchmarks/tests/ -x -q'` (pure fixtures, no GPU, <30 s) |
| **Full suite command** | quick suite + `bash benchmarks/tests/smoke_matrix.sh` (1-tier GPU dry run) |
| **Estimated runtime** | quick ~30 s · full ~5 min |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds (quick), ~5 min (wave gate)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-xx | 01 | 1 | BENCH-01 | — | wrapper emits 4 rows/tier with correct p/n/fa fields; rejects contaminated defaults | unit (fixture jsonl parse) | `pytest benchmarks/tests/test_llabench_wrapper.py -x` | ✅ | ✅ green |
| 02-01-xx | 01 | 1 | BENCH-01 | — | re-run variance gate \|mean₁−mean₂\|/mean ≤ 5% logic | unit (synthetic pairs) | `pytest benchmarks/tests/test_repro_gate.py -x` | ✅ | ✅ green |
| 02-02-xx | 02 | 1 | BENCH-02 | — | manifest contains all D2-10 fields non-empty; binary/model/.wslconfig sha256 recomputable | unit + smoke | `pytest benchmarks/tests/test_manifest.py -x` | ✅ | ✅ green |
| 02-02-xx | 02 | 1 | BENCH-02 | T-shmem | shmem digest parser: fixture snapshot → mandatory-field dict; DEAD/stale signatures detected | unit (byte fixtures) | `pytest benchmarks/tests/test_shmem_digest.py -x` | ✅ | ✅ green |
| 02-03-xx | 03 | 1 | BENCH-03 | — | spiked RSS series ⇒ FAILED:suspected-spill (D2-14 regression test) | unit | `pytest benchmarks/tests/test_guard_fixtures.py -x` | ✅ | ✅ green |
| 02-03-xx | 03 | 1 | BENCH-03 | — | rows.jsonl survives simulated SIGKILL (fsync-per-row proof) | unit (tmpdir) | `pytest benchmarks/tests/test_journal_crash.py -x` | ✅ | ✅ green |
| 02-04-xx | 04 | 2 | BENCH-04 | — | matrix assembler: 4 tiers × {pp,pg} × {fa off,on} in D2-19 order; FAILED cells excluded-but-published | unit | `pytest benchmarks/tests/test_matrix_assembly.py -x` | ✅ | ✅ green |
| 02-05-xx | 05 | 2 | BENCH-04 | — | Vulkan coverage gate: support-CSV contains ✅ GATED_DELTA_NET/SOLVE_TRI/SSM_CONV/SSM_SCAN | integration (GPU, Vulkan arm) | `bash benchmarks/tests/vulkan_gate.sh` | ✅ | ✅ green |
| calibration | 04 | last | BENCH-03 | — | ONE supervised near-OOM live trip | manual | RUNBOOK §calibration | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] `benchmarks/tests/` scaffold + `pytest.ini` (or unittest discovery fallback)
- [x] Fixture generators: synthetic llama-bench jsonl, spiked RSS traces, HWiNFO SM2 byte snapshots, support-CSV samples
- [x] `benchmarks/RUNBOOK.md` skeleton (threshold values filled at calibration session)

---

## Manual-Only Verifications

- Supervised near-OOM live guard trip (destructive-by-design, owner present, D2-14) — RUNBOOK documents procedure and expected outcomes (clean-fail path or live trip)
- HWiNFO sensor label map on this specific card (calibration step; labels vary by card/driver)
- Thermal kill-switch end-to-end rehearsal at calibration (deliberate threshold set low, e.g. 60 °C idle-safe, to prove abort path without real heat)

---

## Notes

- All GPU-dependent tests are integration-tier and gated behind Wave 0 fixtures so quick feedback never touches the GPU.
- Crash-resilience proof uses tmpdir SIGKILL simulation on the journal writer, not real benchmark runs.
- Vulkan gate requires the Windows-native arm built first (arm-dependent task ordering enforced in plans).
