# Coding Conventions

**Analysis Date:** 2026-08-25

## Naming Patterns

**Files:**
- Python test files: `test_<subject>.py` (e.g., `benchmarks/tests/test_op_gate.py`, `benchmarks/tests/test_preflight.py`)
- Python CLI entry points: `snake_case.py` in `benchmarks/bin/` (e.g., `run_session.py`, `run_model_gate.py`, `run_op_gate.py`)
- Python library modules: single-word or compound `snake_case.py` in `benchmarks/lib/` (e.g., `guard.py`, `store.py`, `fingerprint.py`, `preflight.py`)
- HIP kernel implementations: `impl_<variant>.hip` (e.g., `kernels/matmul_iq4xs/impl_gemm_wmma.hip`, `impl_gemv_gfx1100.hip`)
- CPU golden oracles: `ref_cpu.cpp` / `ref_cpu.h` per kernel directory
- Correctness harnesses: `test_<op>_compare.cpp` (e.g., `kernels/matmul_iq4xs/test_gemv_compare.cpp`)
- Microbenchmarks: `bench_<op>.cpp` (e.g., `kernels/matmul_iq4xs/bench_gemv.cpp`)
- Shell scripts: `check_no_ggml.sh`, `smoke_matrix.sh`, `vulkan_gate.sh` — lowercase snake_case

**Functions:**
- Python: `snake_case` for all functions and methods (`run_op_gate()`, `parse_free_mib()`, `estimate_needed_mib()`)
- Factory/classmethods: `from_json()` (`benchmarks/lib/guard.py:Thresholds.from_json`), `create()` (`benchmarks/lib/store.py:RunStore.create`)
- C++: `snake_case` for free functions (`gemv_iq4xs_cpu_ref()`, `compute_metrics()`, `gen_W()`), `PascalCase` for types (`Metrics`, `MatmulShape`, `RunStore` equivalent)
- Test functions always start with `test_` (enforced by pytest config)

**Variables:**
- Python: `snake_case`; module-level constants use `UPPER_SNAKE_CASE` (`DEFAULT_THRESHOLDS_PATH`, `STOCK_PPL_REFERENCE = 6.4271`, `KV_BYTES_PER_TOKEN_F16`, `VERDICT_SPILL` — see `benchmarks/lib/guard.py`, `benchmarks/lib/preflight.py`, `benchmarks/bin/run_model_gate.py`)
- C++ constants: `UPPER_SNAKE_CASE` (`CANONICAL_SHAPES`, `NUM_CANONICAL_SHAPES`, `QK_K` — `kernels/matmul_iq4xs/ref_cpu.h`)
- Short names acceptable inside dense numeric loops (`r`, `g`, `d`, `dot`, `nr`, `ng` in `test_gemv_compare.cpp` metric kernels)

**Types:**
- Python: `PascalCase` dataclasses with docstrings (`Thresholds`, `PreflightVerdict` in `benchmarks/lib/guard.py`, `benchmarks/lib/preflight.py`)
- Verdict/status strings are locked vocabulary constants, never inline literals at call sites (`VERDICT_OK = "OK"`, `VERDICT_SPILL = "FAILED:suspected-spill"` — `benchmarks/lib/guard.py:14-18`); import these constants instead of re-typing strings

## Code Style

**Formatting:**
- No formatter/linter config present (no `.prettierrc`, `pyproject.toml`, `setup.cfg`, ruff/black/flake8 configs detected). Match surrounding style manually.
- Python: PEP 8-ish, 4-space indent, double blank lines between top-level defs
- C++: 4-space indent; correctness harnesses tolerate very dense one-line bodies inside metric/fixture code — do not propagate that density to new code
- Line length is informal (~100 chars typical)

**Linting:**
- None configured. De-facto gates are the test suite plus `scripts/check_no_ggml.sh` (grep-based isolation gate banning any `#include` of ggml/llama headers under `kernels/`).

## Import Organization

**Order:**
1. `from __future__ import annotations` (all Python modules use this — see `benchmarks/lib/guard.py:7`)
2. Module docstring comes BEFORE the future import (docstring first line of file)
3. Standard library imports (`json`, `os`, `re`, `subprocess`, `hashlib`, `datetime`, `threading`, `time`)
4. Third-party (`pytest` in tests only)
5. First-party (`from benchmarks.lib import guard`, `from benchmarks.bin import run_op_gate`)

**Path Aliases:**
- No aliases. First-party imports always use full paths rooted at repo root: `from benchmarks.lib.guard import VERDICT_PREFLIGHT`. Run everything with `PYTHONPATH=.` from repo root.

## Error Handling

**Patterns:**
- **Verdict objects, not exceptions**, for gate logic: checks return structured results (`PreflightVerdict(verdict, evidence, flags)`, guard `evaluate()` verdicts) with status strings from the locked vocabulary — see `benchmarks/lib/preflight.py:22-26`
- Gate runners return result dicts (`res["status"] == "PASS"/"FAIL"`) AND persist them as JSON (`out_json` parameter, e.g., `benchmarks/results/phase3/op_gate.json`) so failures survive process exit
- **Fail-fast macros** in HIP/C++ code: `HIP_CHECK(ans)` wraps every HIP API call, prints `[HIP ERROR] file:line message (code N)` to stderr then `std::abort()` — defined once in `kernels/common/hip_helpers.h:9-15`; never call raw hipMalloc/hipMemcpy without it
- **Tolerant config loading**: `Thresholds.from_json()` returns `None` on absent/corrupt JSON and callers fall back to observe-only mode (`benchmarks/lib/guard.py:29-43`)
- Binary harnesses signal pass/fail via exit code (0/1) plus a final stdout marker line (`FINAL RESULT: PASS (8/8 passed)` / `FINAL: PASS`) that Python wrappers assert on (`benchmarks/tests/test_demo_iq4xs_dequant.py:20-23`)
- Shell scripts use `set -euo pipefail` and explicit `exit 0`/`exit 1` with an echo'd PASS/ERROR line (`scripts/check_no_ggml.sh`)

## Logging

**Framework:** Plain `print()` / `fprintf(stderr, ...)`. No `logging` module usage detected anywhere in `benchmarks/` or `tools/`.

**Patterns:**
- Progress lines go to stdout with bracketed prefixes: `[GEMV-CUSTOM] %-16s K=%5lld ...` (`kernels/matmul_iq4xs/test_gemv_compare.cpp:41`), `[HIP ERROR]` for fatal errors
- Structured results go to JSON files, not logs: `meta.json`, `rows.jsonl`, `op_gate.json`, `model_gate.json`
- User-facing alerts use Windows toast notifications via PowerShell interop (`benchmarks/lib/toast.py:build_toast_xml`) — no third-party notification deps
- Follow this split: human-readable progress → stdout; machine-readable outcomes → JSON artifacts

## Comments

**When to Comment:**
- Every module starts with a docstring stating purpose AND requirement IDs from the plan (`"""Op-Level Correctness Gate runner (QUAL-01, Plan 03-01)."""` — `benchmarks/bin/run_op_gate.py:2-7`). Trace new modules back to their plan/requirement ID the same way.
- Constants carry provenance comments: `WEIGHTS_BYTES = 15309039008  # models/README.md sha256-of-record artifact size` (`benchmarks/lib/preflight.py:13`)
- Regex parsers document accepted input formats in the docstring (`parse_free_mib` lists both matched patterns — `benchmarks/lib/preflight.py:30-36`)

**JSDoc/TSDoc:**
- Python: docstrings on every public function/dataclass, one-liners preferred
- C++: header comment blocks at top of each file describing role (`// ref_cpu.h — FP64 golden oracle for IQ4_XS GEMV/GEMM` — `kernels/matmul_iq4xs/ref_cpu.h:2`); no Doxygen tooling

## Function Design

**Size:** Single-purpose functions; gate runners expose one orchestrating function taking injectable parameters (`run_op_gate(mock_csv=..., mock_exit_code=..., out_json=...)`) so tests can stub subprocesses without monkeypatching.

**Parameters:** Keyword arguments with defaults pointing at module-level `DEFAULT_*` constants (`benchmarks/bin/run_model_gate.py:16-24`). Optional params typed `str | Path`, `list[str] | None`.

**Return Values:** Dicts with fixed keys (`status`, `total_cases`, `errors`, `core_ops_status`) or small dataclasses; never bare booleans for gate decisions.

## Module Design

**Exports:** One primary callable per bin script plus a `main()`/argparse entry guarded by `if __name__ == "__main__"` pattern; lib modules export plain functions and dataclasses.

**Barrel Files:** None used. Import directly from the module path.

## Domain-Specific Rules (Binding)

These are enforced project doctrine (`AGENTS.md`, `CONTRIBUTING.md`, `docs/DEVELOPMENT.md`) — treat as conventions:

- **Kernels stay standalone:** zero ggml/llama includes under `kernels/` — verified by `scripts/check_no_ggml.sh` (KERN-01). Build via `cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100`.
- **Every kernel ships a correctness harness next to it** comparing against the FP64 CPU oracle (`ref_cpu.cpp`) with metrics max_abs / mean_abs / max_rel / cosine.
- **Integration is additive quilt patches only** over pinned upstream commit `bb4caa75`, placed in `patches/*.patch`, behind OFF-by-default build flags (e.g., `GGML_CUDA_ENABLE_CUSTOM_GFX1100`). Never hard-fork.
- **Never report blended tok/s** — prefill (M>>1) and decode (M≈1) measured separately.
- **Record compiler/ROCm/driver versions with every benchmark result** via `benchmarks/lib/fingerprint.py`.
- **GPU work requires `HSA_ENABLE_DXG_DETECTION=1`** exported before execution (WSL2/DXG pipeline).
- Branches: `feat/<topic>` / `fix/<topic>` off `main`.

---

*Convention analysis: 2026-08-25*
