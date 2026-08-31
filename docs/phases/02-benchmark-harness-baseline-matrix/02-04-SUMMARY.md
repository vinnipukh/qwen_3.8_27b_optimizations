# Plan 02-04 Summary: Baseline Session Orchestration, Empirical Calibration & Stock HIP Matrix

**Executed:** 2026-08-23
**Plan:** `02-04-PLAN.md` (Wave 2)
**Requirements Satisfied:** `BENCH-01`, `BENCH-03`, `BENCH-04`

---

## What Was Done

1. **Session Orchestration Engine (`benchmarks/bin/run_session.py`):**
   - Built end-to-end driver acquiring exclusive flock (`benchmarks/results/.session.lock`).
   - Integrated full pipeline: system manifest -> pre-flight checks -> setsid process launch -> live /proc RSS guard thread -> jsonl parsing and cell integrity verification -> VRAM ledger recording -> CHECKSUMS.sha256 generation -> toast alerts.
   - Enforced pre-flight 32k gate: skips allocation on over-budget tiers and cleanly publishes `FAILED:preflight-oom` rows.

2. **Empirical Calibration (`benchmarks/bin/calibrate.py`):**
   - Executed `rehearse-kill`: proved cross-boundary thermal watchdog kill path against dummy process with exit code -9 and toast alert.
   - Executed `labels`: established documented fallback mapping for HWiNFO sensor metrics.
   - Executed `profile`: derived empirical thresholds from healthy 4k/8k runs with 1.5x steady-state margin:
     - `vmrss_fail_kb`: 22,788,858 kB (21.73 GB)
     - `vmswap_fail_kb`: 524,288 kB (512 MB)
     - `gpu_shared_climb_mb_per_min`: 250.0 MB/min
     - Wrote `benchmarks/config/thresholds.json`.
   - Executed `near-oom`: supervised near-OOM test on tier 32768, validating pre-flight interception and safe failure recording without host/guest freeze.
   - Populated binding runbook sections in `benchmarks/RUNBOOK.md`.

3. **Stock HIP Baseline Matrix Execution & Publication:**
   - Executed full 16-cell HIP matrix (`20260823_164724_baseline_hip`): 12 OK cells across 4k/8k/16k and 4 pre-flight intercepted cells at 32k.
   - Executed reproducibility re-run (`20260823_170839_baseline_hip`) on tier 8192.
   - Created `benchmarks/bin/publish_matrix.py` and published `benchmarks/results/BASELINE-MATRIX.md` and `BASELINE-MATRIX.json`.
   - Created unit tests `benchmarks/tests/test_matrix_assembly.py`.

---

## Verification Evidence

- `test_matrix_assembly.py`: 3/3 unit tests passed.
- `smoke_matrix.sh`: Passed with all predicates verified (manifest, rows count, integrity, sha256sum).
- `BASELINE-MATRIX.md`: Published 16 cells (12 OK + 4 FAILED:preflight-oom) with exact mean ± stddev throughput values.
