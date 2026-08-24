<!-- generated-by: gsd-doc-writer -->

# Testing & Verification Doctrine

This project enforces strict correctness, reproducibility, and safety gates before any
performance claim or custom kernel optimization is accepted.

## Gate Hierarchy

```
Level 0: Unit & Guard Regression Tests (47 tests in benchmarks/tests/)
Level 1: Platform & Environment Gates (ENV-01..04, rocminfo, hipconfig, versions.txt)
Level 2: Constraint & Integrity Gates (BENCH-01, scan_banned_signatures, D2-19 ordering)
Level 3: VRAM & RSS Overcommit Gates (BENCH-03, preflight buffer check, three-signal guard)
Level 4: Vulkan Coverage Gate (D2-04, vulkan_gate.sh 6-part check)
Level 5: Numerical Correctness Gates (QUAL-01 test-backend-ops, QUAL-02 perplexity + canaries)
Level 6: Kernel Numerical Comparison (test_compare vs CPU golden ref, KERN-01)
```

## Running the Unit Test Suite

The unit test suite validates wrapper argument construction, reproducibility math,
system fingerprinting, HWiNFO shared memory decoding, thermal watchdog kill command
validation, forked SIGKILL journal crash resilience, VRAM pre-flight allocation logic,
matrix aggregation, op-level correctness gate parsing (QUAL-01), model-level quality gate
tolerances and golden canaries (QUAL-02), and bottleneck profiling parser logic (PROF-01/02).

From repo root in WSL2:

```bash
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q
```

All 47 unit tests run pure-CPU/fixture-driven and execute in under 25 seconds.

## Smoke Matrix Integration Test

`benchmarks/tests/smoke_matrix.sh` runs a complete end-to-end mini benchmark session
on the real GPU and asserts:
1. `manifest.json` exists with all non-empty D2-10 fingerprint fields.
2. `rows.jsonl` contains the exact expected row count with zero contaminating default signatures.
3. `CHECKSUMS.sha256` verifies with `sha256sum -c` exit code 0.

Run from repo root:
```bash
bash benchmarks/tests/smoke_matrix.sh
```

## Six-Part Vulkan Coverage Gate

`benchmarks/tests/vulkan_gate.sh` enforces the 6-part D2-04 requirement before any Vulkan
performance claim can be cited:
1. Static shader inventory check (GDN and IQ4_XS shaders present).
2. Operator support CSV comparison (`hip-support-comparator.csv` vs `vulkan-support.csv`).
3. Full `test-backend-ops` suite verification on Vulkan.
4. Tensor layer residency check (132/132 layers on GPU).
5. Coherent greedy-decode smoke test.
6. Fallback and partial-support audit.

Run from repo root:
```bash
bash benchmarks/tests/vulkan_gate.sh
```

## Safety & Crash Resilience Tests

* **SIGKILL Crash Resilience (`test_journal_crash.py`):** Simulates writer death via `os.kill(SIGKILL)` in a forked child process, proving that every `fsync()`-ed row survives without JSON corruption.
* **Tamper Evidence (`test_journal_crash.py`):** Asserts that modifying a single byte in any tracked file causes `sha256sum -c CHECKSUMS.sha256` to fail immediately.
* **Adversarial Guard Regression (`test_guard_fixtures.py`):** Asserts that spiked RSS, swap growth, and shared GPU memory leak traces reliably trigger `FAILED:suspected-spill` verdicts.
