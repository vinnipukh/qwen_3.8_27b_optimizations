# Plan 02-03 Summary: Append-Only Result Store, RSS Guard & VRAM Pre-Flight Gate

**Executed:** 2026-08-23
**Plan:** `02-03-PLAN.md` (Wave 1)
**Requirements Satisfied:** `BENCH-03`

---

## What Was Done

1. **Append-Only Result Store (`benchmarks/lib/store.py`):**
   - Implemented `RunStore` managing per-run directories (`manifest.json`, `rows.jsonl`, `vram_ledger.jsonl`, `logs/`, `telemetry/`).
   - Row appends execute `flush()` + `os.fsync(fileno())` per row to guarantee survival across unexpected aborts.
   - Built `write_checksums()` emitting `CHECKSUMS.sha256` in standard `sha256sum -c` format.
   - Built `supersede()` metadata management and `index_entry()` summary generator.

2. **Three-Signal VRAM Spill Guard (`benchmarks/lib/guard.py`):**
   - Implemented `poll_proc()` reading `/proc/<pid>/status` for VmRSS and VmSwap at 1 Hz.
   - Implemented `Thresholds.from_json()` loading empirical calibration config, automatically reverting to observe-only mode when thresholds are absent.
   - Built `evaluate()` verifying:
     - Signal 1: VmRSS / VmSwap spike crossing threshold.
     - Signal 2: Windows shared GPU memory steady climb.
     - Signal 3: Intra-cell repetition throughput deviation ratio (>2.0x).
   - Applied locked verdict vocabulary: `OK`, `FAILED:suspected-spill`, `REVIEW:repeat-deviation`, `FAILED:thermal-abort`, `FAILED:preflight-oom`.

3. **VRAM Pre-Flight Gate & Ledger (`benchmarks/lib/preflight.py`):**
   - Implemented `estimate_needed_mib()` modeling Qwen3.8-27B hybrid architecture (weights 14.6 GB + 64 KiB/tok KV + compute buffer).
   - Implemented `check()` against measured DXG free VRAM (18245 MiB anchor), correctly yielding PASS for 4k/8k/16k and `FAILED:preflight-oom` for 32k.
   - Built `parse_free_mib()`, `parse_buffer_lines()`, and `record_ledger()`.

4. **Adversarial & Crash Regression Testing:**
   - Built `test_journal_crash.py` with forked SIGKILL mid-stream verification and single-byte tamper detection.
   - Built `gen_rss_trace.py` and `test_guard_fixtures.py` verifying detection across spiked traces and absence of hard-coded thresholds.
   - Built `test_preflight.py` asserting tier math against the 18245 MiB anchor.

---

## Verification Evidence

- `test_journal_crash.py`, `test_guard_fixtures.py`, `test_preflight.py`: 14/14 unit tests passed.
- Full test suite: 32/32 tests passed in guest.
