# Plan 02-01 Summary: Benchmark Harness Core Engine & Corpus

**Executed:** 2026-08-23
**Plan:** `02-01-PLAN.md` (Wave 1)
**Requirements Satisfied:** `BENCH-01`

---

## What Was Done

1. **Wave-0 Test Scaffold:**
   - Configured `benchmarks/tests/pytest.ini` for pytest test discovery.
   - Built `benchmarks/tests/fixtures/gen_llabench_jsonl.py` supporting synthetic multi-tier llama-bench JSONL outputs with verified schemas and sample arrays.

2. **llama-bench Wrapper Engine (`benchmarks/lib/llabench.py`):**
   - Implemented `build_argv()` enforcing explicit `-p C`, `-n 0` (zeroing upstream default prompt/gen vectors), `-pg C,128`, `-fa <off,on>`, `-r 5`, `--delay 30`, `-ngl 99`, `-sm none`, and `-o/-oe jsonl`.
   - Built `enumerate_tiers()` generating strictly ascending context plans (`4096`, `8192`, `16384`, `32768`) and per-tier batch execution plans.
   - Built `assert_cell_integrity()` and `scan_banned_signatures()` raising `MatrixContaminationError` on upstream default leakage (`n_prompt=512` or `n_prompt=0, n_gen=128`).
   - Built `variance_pct()` and `repro_ok()` enforcing the ±5% reproducibility gate.

3. **Layer-2 Greedy Runner & Deterministic Corpus:**
   - Authored 6 deterministic prompt files under `benchmarks/prompts/`:
     - `short_code_01.txt` (810 chars)
     - `short_prose_01.txt` (627 chars)
     - `short_prose_02.txt` (584 chars)
     - `long_code_01.txt` (9234 chars)
     - `long_prose_01.txt` (8539 chars)
     - `long_prose_02.txt` (8244 chars)
   - Created `benchmarks/bin/run_prompts.py` supporting headless `llama-cli` execution (`setsid`, `--single-turn`, `--simple-io`, `--load-mode none`, `--temp 0`) with per-prompt fsynced JSONL journaling and prompt SHA-256 drift detection.
   - Verified end-to-end against real model and binary in WSL.

4. **Runbook Protocol:**
   - Created `benchmarks/RUNBOOK.md` establishing §session-protocol, §thresholds placeholders, §telemetry-modes, §thermal-policy, and §operational-ceiling.

---

## Verification Evidence

- `test_llabench_wrapper.py` & `test_repro_gate.py`: 8/8 unit tests passed.
- Real-binary tracer smoke: 2 rows produced on ROCm0 device (`pp` and `tg` at 1024 context, flash_attn off).
- Real prompt smoke: `short_prose_02.txt` executed at tier 2048 in 16.3s producing 2271 bytes output, recorded in `rows.jsonl`.
