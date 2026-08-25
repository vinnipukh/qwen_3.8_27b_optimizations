# Testing Patterns

**Analysis Date:** 2026-08-25

## Test Framework

**Runner:**
- pytest (Python), 55 unit/regression tests in `benchmarks/tests/`
- Config: `benchmarks/tests/pytest.ini` — `testpaths = benchmarks/tests`, `python_files = test_*.py`, `addopts = -ra`
- C++/HIP correctness harnesses are standalone binaries driven by pytest wrappers or run directly

**Assertion Library:**
- Plain `assert` statements (pytest native)

**Run Commands:**
```bash
# From repo root in WSL2:
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q              # all 55 tests, pure-CPU, <30s
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_op_gate.py -v   # one module
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_model_gate.py -v
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_demo_iq4xs_dequant.py -v  # requires built kernels + GPU env

# Kernel correctness binaries (require HSA_ENABLE_DXG_DETECTION=1):
export HSA_ENABLE_DXG_DETECTION=1
./kernels/build/demo_iq4xs_dequant/demo_test          # GREEN: FINAL RESULT: PASS (8/8 passed)
./kernels/build/demo_iq4xs_dequant/demo_test_broken   # RED: must FAIL (>1000x error) — discrimination check
./kernels/build/matmul_iq4xs/matmul_test_baseline     # 16/16 PASS vs stock comparator
./kernels/build/matmul_iq4xs/test_gemv_compare        # 10/10 PASS
./kernels/build/matmul_iq4xs/test_gemm_compare        # 11/11 PASS

# Isolation gate:
bash scripts/check_no_ggml.sh                          # bans ggml/llama includes under kernels/
```

## Test File Organization

**Location:**
- Python tests: centralized in `benchmarks/tests/` (not co-located with `benchmarks/lib/`)
- Kernel correctness tests: co-located with kernel source (`kernels/<op>/test_compare.cpp`, `kernels/<op>/test_*_compare.cpp`)
- Shell smoke/gate scripts: `benchmarks/tests/smoke_matrix.sh`, `benchmarks/tests/vulkan_gate.sh`

**Naming:**
- `test_<subject>.py`; test functions `test_<behavior>()`

**Structure:**
```
benchmarks/tests/
├── pytest.ini                  # runner config
├── fixtures/
│   ├── gen_rss_trace.py        # synthetic RSS/swap trace generators
│   ├── gen_shmem_snapshot.py   # HWiNFO shared-memory snapshots
│   └── gen_llabench_jsonl.py   # synthetic llabench JSONL rows
├── test_op_gate.py             # QUAL-01: parses mock CSV of test-backend-ops output
├── test_model_gate.py          # QUAL-02: PPL tolerance + golden canaries
├── test_preflight.py           # VRAM preflight arithmetic + ledger roundtrip
├── test_guard_fixtures.py      # three-signal guard against synthetic traces
└── ...                         # manifest, matrix assembly, journal crash, repro gate, shmem digest,
                                # bottleneck profiling, llabench wrapper, fixture integrity
```

## Test Structure

**Suite Organization:**
```python
# benchmarks/tests/test_op_gate.py
import json
import pytest
from pathlib import Path
from benchmarks.bin import run_op_gate

MOCK_VALID_CSV = """ggml_cuda_init: found 1 ROCm devices
"backend_name","op_name",..."""

def test_op_gate_parsing_valid(tmp_path: Path):
    out_json = tmp_path / "op_gate.json"
    res = run_op_gate.run_op_gate(mock_csv=MOCK_VALID_CSV, mock_exit_code=0, out_json=out_json)
    assert res["status"] == "PASS"
    assert res["total_cases"] == 8
    assert out_json.exists()
```

**Patterns:**
- **Injectable seams over monkeypatching:** production functions accept `mock_csv=`, `mock_exit_code=` parameters so tests substitute subprocess output directly (`benchmarks/bin/run_op_gate.py`, `run_model_gate.py`). Follow this for any new wrapper around an external binary.
- **tmp_path everywhere:** every file-writing test uses pytest's `tmp_path` fixture; never write into real `benchmarks/results/` from a unit test.
- **Golden constants inline:** reference values frozen as module constants or literals (`FROZEN_FREE_MIB = 18245.0` in `test_preflight.py:12`; expected sha256 hex in `test_manifest.py`).
- **Both in-memory and persisted assertions:** assert the returned dict AND re-read the emitted JSON artifact to verify persistence.

**Teardown pattern:** none needed — tmp_path auto-cleans; RunStore directories created under tmp_path.

## Mocking

**Framework:** No unittest.mock usage. Mocking is done via parameter injection and synthetic fixture generators.

**Patterns:**
```python
# Parameter-injection mocking (preferred): benchmarks/bin/run_model_gate.py:52-58
def run_perplexity(..., mock_output: str | None = None, mock_returncode: int = 0) -> dict[str, Any]:
    ...

# Synthetic telemetry traces from generators: benchmarks/tests/fixtures/gen_rss_trace.py
rss = make_spiked_rss_profile(base_rss_kb=1000000, spike_rss_kb=3000000)
verdict = guard.evaluate(rss_profile=rss, thresholds=sample_thresholds)
assert verdict.verdict == guard.VERDICT_SPILL   # import locked constants, don't retype strings
```

**What to Mock:**
- External binary stdout/exit codes (llama-cli, test-backend-ops) via `mock_csv`/`mock_output`/`mock_exit_code` params
- Telemetry time series via fixture generators (`gen_rss_trace.py`, `gen_shmem_snapshot.py`, `gen_llabench_jsonl.py`)
- Thresholds via constructed dataclasses (`guard.Thresholds(...)` with explicit values, `test_guard_fixtures.py:13-19`)

**What NOT to Mock:**
- The CPU oracle math (`ref_cpu.cpp`) — it is the ground truth itself and is compiled HIP-free for fixture validation
- Filesystem JSON roundtrips through `RunStore` — exercised for real on tmp_path (`test_journal_crash.py` covers SIGKILL crash resilience)
- GPU binaries in the pytest suite that exercise them (`test_demo_iq4xs_dequant.py`) — these run the real binary via `subprocess.run` with `timeout=60..90` and `HSA_ENABLE_DXG_DETECTION=1` env copy

## Fixtures and Factories

**Test Data:**
```python
# Factory-style generator functions returning domain objects: gen_rss_trace.py
def make_healthy_profile(base_rss_kb: int = 1000000) -> RssProfile:
    profile = RssProfile(pid=1001)
    for i in range(10):
        profile.vmrss_series_kb.append(base_rss_kb + (i * 100))
        ...
    return profile
```
- C++ harnesses generate deterministic pseudo-random weights/tensors with seeded `std::mt19937 rng(seed)` where seed is derived from shape (`1000+K*17+N` — `kernels/matmul_iq4xs/test_gemv_compare.cpp:33`), making failures reproducible.

**Location:**
- Python generators: `benchmarks/tests/fixtures/gen_*.py` (imported as modules, not conftest plugins)
- pytest fixtures defined per-module via `@pytest.fixture` (e.g., `sample_thresholds`)
- Tensor fixtures dumped by `tools/dump_gguf_fixtures.py` / `tools/dump_matmul_fixtures.py` into `kernels/fixtures/`; integrity checked by `benchmarks/tests/test_fixture.py`

**conftest.py:** Not present. Shared helpers live in `fixtures/gen_*.py` modules.

## Coverage

**Requirements:** None enforced (no coverage tooling configured). The de-facto completeness bar is documented in `docs/TESTING.md` (gate hierarchy Levels 0–7).

**View Coverage:**
```bash
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q --tb=short   # no coverage plugin installed
```

## Test Types

**Unit Tests:**
- Pure-CPU logic: parsing regexes, threshold arithmetic, fingerprint hashing, matrix assembly, wrapper argument construction. All 55 run without a GPU in <30s.

**Integration Tests:**
- GPU kernel correctness: pytest modules shell out to built binaries (`test_demo_iq4xs_dequant.py` uses `subprocess.run([bin], capture_output=True, text=True, env=get_env(), timeout=60)`); require `cmake --build kernels/build` artifacts to exist first — tests assert binary presence with a clear "Run cmake build" message.
- End-to-end gates: `benchmarks/bin/run_op_gate.py` (QUAL-01, 21,093 backend ops) and `benchmarks/bin/run_model_gate.py` (QUAL-02, PPL 6.4271 ±1% + 6/6 greedy canaries) — these are real-GPU gates, not part of the fast suite.

**E2E Tests:**
- Session-level: `benchmarks/bin/run_session.py` full matrix runs; smoke path `benchmarks/tests/smoke_matrix.sh` (1024 ctx, 1 repeat). Vulkan coverage checked by `benchmarks/tests/vulkan_gate.sh`.

## Numerical Correctness Doctrine (kernel tests)

Every custom kernel is validated against the FP64 CPU oracle with this metric set (`compute_metrics()` in `kernels/matmul_iq4xs/test_gemv_compare.cpp:15-28`):

- Gate thresholds: `cosine >= 0.999 && max_rel <= 1e-3 && no NaN/Inf` (matmul family); tighter `max_abs < 1e-5 / mean_abs < 1e-6 / cosine > 0.99999` for the demo dequant kernel
- **Discrimination testing:** each kernel directory ships a deliberately broken variant (`demo_test_broken`, `impl_broken.hip`) and a test asserts the broken version fails by ≥10× margin — proving the harness can actually detect errors (`test_discrimination_metrics` in `benchmarks/tests/test_demo_iq4xs_dequant.py:41-62`)
- Harnesses print per-case metrics plus a final `FINAL RESULT: PASS/FAIL` line and exit 0/1

## Common Patterns

**Async Testing:** Not used — everything is synchronous subprocess calls with explicit timeouts.

**Error Testing:**
```python
def test_demo_broken_fails():
    res = subprocess.run([bin_path], capture_output=True, text=True, env=get_env(), timeout=60)
    assert res.returncode != 0, "demo_test_broken MUST fail with non-zero exit code"
    assert "FINAL RESULT: FAIL" in res.stdout

# Gate failure paths asserted via injected bad input:
res = run_op_gate.run_op_gate(mock_csv=MOCK_FAILING_OP_CSV, ...)
assert res["status"] == "FAIL"
assert "numerical mismatch" in res["errors"][0]["error"]
```

**Environment-sensitive tests:** helper `get_env()` copies `os.environ` and adds `HSA_ENABLE_DXG_DETECTION=1` (`test_demo_iq4xs_dequant.py:6-9`); reuse this helper for any new GPU-binary test.

---

*Testing analysis: 2026-08-25*
