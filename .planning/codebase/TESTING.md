# Testing Patterns

**Analysis Date:** 2026-08-30

Two parallel test stacks, mirroring the two ecosystems:
1. **pytest suite** in `benchmarks/tests/` — 55 unit/integration/meta tests, CPU-only, injectable GPU mocks (Level 0 harness tests)
2. **Standalone C++ correctness binaries** in `kernels/` — `test_*_compare.cpp` compiled by CMake, run against HIP, comparing GPU output to CPU reference oracles

Both feed three end-to-end quality gates: the header-isolation gate (`scripts/check_no_ggml.sh`), the op-level gate (`benchmarks/bin/run_op_gate.py`, QUAL-01), and the model-level gate (`benchmarks/bin/run_model_gate.py`, QUAL-02).

---

## Test Framework

**Runner:**
- pytest — config `benchmarks/tests/pytest.ini`:
  ```ini
  [pytest]
  testpaths = benchmarks/tests
  python_files = test_*.py
  python_classes = Test*
  python_functions = test_*
  addopts = -ra
  ```
- C++ kernel tests: plain executables (no gtest/Catch2), wired in `kernels/matmul_iq4xs/CMakeLists.txt`, `kernels/demo_iq4xs_dequant/CMakeLists.txt`, `kernels/template/CMakeLists.txt`.

**Assertion Library:**
- Python: built-in `assert` + `pytest.approx`, `pytest.raises(match=...)`.
- C++: hand-rolled bool accumulation — `compute_metrics()` (`kernels/common/matmul_test_util.h`) returns `pass = !bad && cosine >= 0.999`; test cases return `bool`; `main()` exits `0`/`1`.

**Run Commands:**
```bash
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q     # full unit suite (CPU only)
bash scripts/check_no_ggml.sh                            # header isolation gate
./kernels/build/matmul_iq4xs/test_gemv_compare           # C++ kernel gate (needs cmake build + HIP)
./kernels/build/matmul_iq4xs/test_gemm_wmma_compare
./kernels/build/demo_iq4xs_dequant/demo_test             # fixtures-based dequant gate
python3 benchmarks/bin/run_op_gate.py --runs 10          # QUAL-01 op gate, 0 errors ×10
python3 benchmarks/bin/run_model_gate.py --runs 10       # QUAL-02 model gate, PPL ±1% ×10 + 6/6 canaries
```
The C++ binaries require `HSA_ENABLE_DXG_DETECTION=1` in the environment on WSL2/Windows (see `benchmarks/tests/test_demo_iq4xs_dequant.py` `get_env()`).

---

## Test File Organization

**Location:**
- Python tests are **not co-located** — they live in a dedicated `benchmarks/tests/` tree, one file per module under test: `test_op_gate.py` → `benchmarks/bin/run_op_gate.py`, `test_preflight.py` → `benchmarks/lib/preflight.py`, `test_guard_fixtures.py` → `benchmarks/lib/guard.py`.
- Kernel tests **are co-located** with the code under test: `kernels/matmul_iq4xs/test_*_compare.cpp` next to the `impl*.hip` files they validate; `kernels/demo_iq4xs_dequant/test_compare.cpp` next to `impl.hip`/`impl_broken.hip`.

**Naming:**
- Python: `test_<module>.py`, functions `test_<behavior>` (e.g. `test_model_gate_ppl_out_of_bounds_fails`).
- C++: `test_<op>_compare.cpp`, functions `test_one`/`test_shape`/`test_gemv_shape_real`.

**Structure:**
```
benchmarks/tests/
├── pytest.ini
├── test_op_gate.py, test_model_gate.py, test_preflight.py,
│   test_guard_fixtures.py, test_llabench_wrapper.py, test_repro_gate.py,
│   test_journal_crash.py, test_shmem_digest.py, test_manifest.py,
│   test_fixture.py, test_matrix_assembly.py, test_bottleneck_profiling.py,
│   test_demo_iq4xs_dequant.py
└── fixtures/
    ├── gen_llabench_jsonl.py, gen_rss_trace.py, gen_shmem_snapshot.py
```

---

## Test Structure

**Python suite organization:** flat `test_*` functions with an optional scoped `@pytest.fixture` for shared objects:

```python
# benchmarks/tests/test_guard_fixtures.py
@pytest.fixture
def sample_thresholds():
    return guard.Thresholds(
        vmrss_fail_kb=2000000,
        vmswap_fail_kb=500000,
        gpu_shared_climb_mb_per_min=200.0,
        repeat_deviation_max_ratio=2.0,
    )

def test_guard_healthy_trace(sample_thresholds):
    rss = make_healthy_profile(base_rss_kb=1000000)
    ...
    assert verdict.verdict == guard.VERDICT_OK
    assert len(verdict.signals["tripped"]) == 0
```

Assertions are made against **locked constants/verdicts**, never magic strings: `guard.VERDICT_SPILL`, `run_op_gate.CORE_HYBRID_OPS`, `llabench.MatrixContaminationError`.

**C++ kernel suite organization:**

```cpp
// kernels/matmul_iq4xs/test_gemv_compare.cpp
bool test_one(const char* name, int64_t K, int64_t N, std::mt19937& rng) {
    gen_iq4xs_weights(W, K, N, 1000 + (uint32_t)K * 17 + (uint32_t)N);   // seeded weights
    gemv_iq4xs_cpu_ref(W.data(), x.data(), y_ref.data(), K, N);          // 1. CPU oracle
    HIP_CHECK(hipMalloc(...)); HIP_CHECK(hipMemcpy(...));                // 2. upload
    HIP_CHECK(gemv_iq4xs_stock_gpu(...)); HIP_CHECK(hipDeviceSynchronize());  // 3. stock comparator
    HIP_CHECK(gemv_iq4xs_gfx1100_gpu(...)); HIP_CHECK(hipDeviceSynchronize()); // 4. candidate kernel
    Metrics m_gfx = compute_metrics(y_ref, y_gfx);                       // 5. metric + pass verdict
    printf("... %s", m_gfx.pass ? "PASS" : "FAIL");
    return m_gfx.pass;
}
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937 rng(42);
    bool ok = true;
    for (int i = 0; i < NUM_CANONICAL_SHAPES; ++i)          // 8 canonical Qwen shapes, ref_cpu.h
        if (!test_one(CANONICAL_SHAPES[i].name, ...)) ok = false;
    if (!test_one("small_512", 512, 512, rng)) ok = false;  // synthetic edge cases
    printf("=== FINAL GEMV-CUSTOM: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
```

**Patterns:**
- Per-shape setup/compare/teardown in one function; memory always freed via `HIP_CHECK(hipFree(...))` before returning.
- Three-way comparisons are common: candidate vs CPU oracle AND vs stock implementation (e.g. `test_gemv_dp4a_compare.cpp` computes `m_coop_vs_ref`, `m_stock_vs_ref`, `m_coop_vs_stock`).
- **Thresholds are per-path, documented inline** — DP4A/Q8_1 quantization noise (~0.1–1%) vs the FP64 oracle is expected, so DP4A gates use `cosine >= 0.99` / `cosine >= 0.999` with an explanatory comment, while exact-math paths gate at `cosine >= 0.999` + `max_rel <= 1e-3` (`kernels/common/matmul_test_util.h`). The gate metric for quantization-affected paths is cosine, never max_rel ("max_rel on small values can be >>1e-3" — `test_gemv_dp4a_compare.cpp`).
- Failure diagnostics: on FAIL, dump first mismatching elements (`for i < 4: printf("  y[%zu] ref=... stock=... coop=...")`) and extra metric detail (`max_rel`, `mean_abs`).

**Teardown pattern:** explicit `HIP_CHECK(hipFree(...))` at end of each case; no RAII guard wrappers.

---

## Mocking

**Framework:** none (plain parameter injection — no `unittest.mock`/`monkeypatch`). Every gate function takes explicit `mock_*` arguments that short-circuit the subprocess path.

**Patterns:**

```python
# benchmarks/bin/run_op_gate.py
if mock_csv is not None:
    raw_csv = mock_csv
    exit_code = mock_exit_code
else:
    proc = subprocess.run(cmd, stdout=PIPE, stderr=PIPE, text=True, env=env)
    raw_csv = proc.stdout
    exit_code = proc.returncode
```

```python
# benchmarks/tests/test_op_gate.py
MOCK_VALID_CSV = """ggml_cuda_init: found 1 ROCm devices
"backend_name","op_name","op_params","test_mode","supported","error_message","backend_reg_name"
"ROCm0","GATED_DELTA_NET","type=f32,head_count=32","test","1","",""
..."""

def test_op_gate_parsing_valid(tmp_path: Path):
    res = run_op_gate.run_op_gate(mock_csv=MOCK_VALID_CSV, mock_exit_code=0, out_json=tmp_path / "op_gate.json")
    assert res["status"] == "PASS"
```

- Mock data lives as **module-level constants** in the test file (`MOCK_VALID_CSV`, `MOCK_FAILING_OP_CSV`, `MOCK_MISSING_CORE_OP_CSV`, `MOCK_GOLDEN`) — covers pass path, op-error path, missing-core-op path, non-zero exit path.
- Mock CSV payloads include the real `ggml_cuda_init` noise line to exercise the CSV line-filtering logic.

**What to Mock:** llama.cpp binary execution (`test-backend-ops` output CSV, `llama-perplexity` PPL strings, `llama-cli` generated text) and memory-polling taps (`RssProfile`, shared-GPU traces from generator functions).

**What NOT to Mock:** the parsing/verdict logic itself, file I/O side effects (real `tmp_path` files are written and re-read), and end-to-end result artifacts.

**Golden/live-artifact verification tests** (deliberately NOT mocked): `test_saved_op_gate_result_exists_and_passes` (`benchmarks/tests/test_op_gate.py`) and `test_saved_model_gate_and_golden_baseline` (`benchmarks/tests/test_model_gate.py`) assert that the committed result files (`benchmarks/results/phase3/op_gate.json`, `model_gate.json`, `benchmarks/golden/stock_baseline_golden.json`, `benchmarks/data/wiki.test.raw`) exist, are non-empty, and still PASS — the suite fails if a live gate result goes stale. NOTE: these are environment-coupled; they fail on a machine without the phase-3 artifacts.

---

## Fixtures and Factories

**Test Data — Python (pure functions, deterministic):**

```python
# benchmarks/tests/fixtures/gen_rss_trace.py
def make_spiked_rss_profile(base_rss_kb=1000000, spike_rss_kb=2000000):
    ...  # returns guard.RssProfile with deterministic series
```

- Fixture *generators* (not static files) for time-series/synthetic traces: `gen_rss_trace.py`, `gen_llabench_jsonl.py`, `gen_shmem_snapshot.py`. Parameterized builders (`make_healthy_profile(base)`, `make_tier_rows(tier=4096, gen=128, fa_seq=(0,1))`) make boundary cases explicit at call sites.
- **Location:** `benchmarks/tests/fixtures/`.

**Test Data — C++ (binary fixture files + manifests):**
- `kernels/fixtures/` — static `.bin` (raw `block_iq4_xs` payloads), `.f32.bin` (dequantized references), `.npz` sources, plus `manifest.json`, `manifest_dequant.json`, `manifest_matmul.json` recording `name`, `tensor_type`, `synthetic`, `case`, `n_blocks`, `block_size`, `raw_sha256`, source `commit`, `rocm` version, and paths.
- Fixture families: `synthetic_*` edge-case blocks (`synthetic_zero`, `synthetic_max_scale` ls=+31/−32, `synthetic_split_half`, `synthetic_nibble_extremes`, `synthetic_subblock_isolated`) + real model tensors (`blk_0_ffn_down_weight`, `blk_0_attn_gate_weight`, `token_embd_weight`).
- Default fixture list is hardcoded in `kernels/demo_iq4xs_dequant/test_compare.cpp`; `--fixture <path>` overrides it. `manifest.json` is the sha256-of-record — regenerate it when fixtures change (see `kernels/fixtures/README.md`).

**Golden baseline:**
- `benchmarks/golden/stock_baseline_golden.json` — recorded 2026-08-24: WikiText-2 PPL `6.4271 ± 0.04103` (145 chunks, ctx 2048) with `min_allowed_ppl = 6.3628`, `max_allowed_ppl = 6.4914` (±1.0%), plus 6 prompt canaries, each carrying `prompt_sha256`, `generated_text`, `output_sha256`, `returncode`.
- Recorded via `python3 benchmarks/bin/run_model_gate.py --record-golden`; evaluated as **exact SHA256 matches** on greedy (`--temp 0`) output.
- Golden PPL reference is ALSO hardcoded in `benchmarks/bin/run_model_gate.py` as `STOCK_PPL_REFERENCE = 6.4271` / `PPL_TOLERANCE_PCT = 1.0` — keep the two in sync when re-baselining.

---

## Coverage

**Requirements:** None enforced — no `--cov` config, no `.coveragerc`, no `[tool.coverage]` in `pyproject.toml` (no root `pyproject.toml` at all). The pytest `addopts = -ra` shows failures/skips but not coverage.

**View Coverage (if needed):**
```bash
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q --cov=benchmarks
```

Gap risk is concentrated in the C++ kernel binaries (no automated coverage) and the environment-coupled live-artifact tests.

---

## Test Types

**Unit Tests (CPU-only, 55 functions):**
- Pure logic: `test_preflight.py` (buffer math, `parse_free_mib`), `test_repro_gate.py` (`variance_pct`, `repro_ok` thresholds ±5%), `test_llabench_wrapper.py` (argv construction, `-p` before `-pg`, `--delay` never `-D`), `test_guard_fixtures.py` (three-signal verdict matrix incl. observe-only mode), `test_shmem_digest.py`, `test_matrix_assembly.py`, `test_bottleneck_profiling.py`, `test_manifest.py`, `test_fixture.py`.
- Gate parsers with mocked subprocess output: `test_op_gate.py` (5), `test_model_gate.py` (4).
- **Crash-resilience**: `test_journal_crash.py` spawns a child `multiprocessing.Process` that SIGKILLs itself mid-write and asserts the append-only `rows.jsonl` is intact; also tampers a file and asserts `sha256sum -c` fails.
- **Static meta-tests**: `test_no_hardcoded_fail_thresholds_in_guard_source` re-reads `benchmarks/lib/guard.py` and asserts no `\d{6,}` literals exist — the thresholds must come from `benchmarks/config/thresholds.json`.

**Negative / Mutant Tests (discrimination discipline):**
- `kernels/demo_iq4xs_dequant/CMakeLists.txt` builds `demo_test` (correct `impl.hip`) and `demo_test_broken` (same harness, `impl_broken.hip` + `TEST_BROKEN=1`).
- `benchmarks/tests/test_demo_iq4xs_dequant.py:test_discrimination_metrics` runs BOTH and asserts the broken kernel's `max_abs` on `blk_0_ffn_down_weight` is `> 1e-3` AND `>= 10x` the correct kernel's `< 1e-5`. A test suite that cannot distinguish correct from broken is treated as worthless.

**Integration Tests (subprocess against built binaries):**
- `benchmarks/tests/test_demo_iq4xs_dequant.py` executes `./kernels/build/demo_iq4xs_dequant/demo_test`, `demo_test_broken`, `demo_bench` with `subprocess.run(..., timeout=60/90, env=get_env())` and asserts exact stdout markers (`"FINAL RESULT: PASS (8/8 passed)"`) plus parsed JSON schema from `demo_bench` (`op == "dequant_iq4_xs"`, `median_us > 0`, `count == 200`, `warmup == 50`).
- These require a prior CMake build (`kernels/build/`) — they `assert os.path.exists(bin_path)` and fail with a build hint otherwise.

**E2E Gates (N=10 statistical rigour, REQ-STAT-07):**
- **QUAL-01** `run_op_gate.py --runs 10`: runs `test-backend-ops test -b ROCm0 --output csv` 10 times; aggregate verdict requires **0 errors in every run** (`total_errors == 0 and all(r["status"] == "PASS")`). Per-run files `op_gate.run<N>.json` + aggregate. Honest FAIL: one bad repeat fails the whole gate.
- **QUAL-02** `run_model_gate.py --runs 10`: 10 repeats of PPL-in-range (6.3628..6.4914) + 6/6 exact canaries; aggregate reports `median_ppl`, `mean_ppl`, `stdev_ppl` via `statistics`; verdict is `all(r["status"] == "PASS")`.
- **Header isolation** `scripts/check_no_ggml.sh`: grep-based gate (`set -euo pipefail`), `exit 1` on any match, `PASS` banner otherwise.

---

## Common Patterns

**Async/streaming timing tests (C++):** `bench_hip_event` in `kernels/common/bench.h` — warmup loop → `hipEventRecord` per iteration → sorted samples → `median/p95/min/max/mean/stdev`, with `stddev_us` aliased to `stdev_us` (REQ-STAT-07 field contract).

**Error-path testing (Python):**

```python
# benchmarks/tests/test_llabench_wrapper.py
with pytest.raises(llabench.MatrixContaminationError, match="Contaminated by upstream default"):
    llabench.assert_cell_integrity(bad_rows, [...])
```

**Float comparison:** `assert pytest.approx(llabench.variance_pct(100.0, 104.9), 1e-5) == 4.9`.

**Output-format verification:** assert exact expected strings in argv/JSON (`argv[idx_p] + 1 == "4096"`, `item["count"] == 200`) rather than loose substring checks; exact stdout markers (`"FINAL RESULT: PASS (8/8 passed)"`, `"FAIL" not in res.stdout`) — this is the honest-PASS/FAIL discipline (test binaries print `=== FINAL <NAME>: PASS|FAIL ===` and exit 0/1; tests assert BOTH the marker and the exit code).

**Environment coupling:** GPU-touching tests read `HSA_ENABLE_DXG_DETECTION=1` from `get_env()`; unit tests never touch GPU or subprocess binaries.

---

*Testing analysis: 2026-08-30*