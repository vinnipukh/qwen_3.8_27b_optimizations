# Coding Conventions

**Analysis Date:** 2026-08-30

This repo has two distinct ecosystems, each with its own conventions:
1. **C++ / HIP kernel playground** in `kernels/` (standalone GPU kernels, zero llama.cpp/ggml dependencies)
2. **Python benchmark harness** in `benchmarks/` (stdlib-only engine in `benchmarks/lib/`, CLI gates in `benchmarks/bin/`)

---

## C++ / HIP Conventions

### Naming Patterns

**Files:**
- `snake_case.cpp` / `snake_case.hip` / `snake_case.h` — e.g. `test_gemv_compare.cpp`, `ref_cpu.cpp`, `impl_gemv_dp4a_gfx1100.hip`, `stock_hip_comparator.hip`.
- Test binaries follow `test_<op>_compare.cpp` — e.g. `test_gemv_compare.cpp`, `test_gemm_wmma_compare.cpp`, `test_real_stock_compare.cpp`.
- Bench binaries follow `bench_<op>.cpp` — e.g. `bench_gemv.cpp`, `bench_gemm_wmma.cpp`.
- Shared headers live in `kernels/common/` (`bench.h`, `hip_helpers.h`, `matmul_test_util.h`, `block_iq4_xs.h`).

**HIP kernel entry functions:** `<op>_iq4xs_<variant>_<arch>_gpu`:
- `gemv_iq4xs_gfx1100_gpu` (`kernels/matmul_iq4xs/test_gemv_compare.cpp`)
- `gemv_iq4xs_dp4a_gfx1100_gpu`, `gemv_iq4xs_dp4a_gfx1100_prequantized_gpu` (`kernels/matmul_iq4xs/test_gemv_dp4a_compare.cpp`)
- `gemm_iq4xs_wmma_stream_gpu`, `gemm_iq4xs_stream_tiled_gpu`, `gemm_iq4xs_wmma_p4_xor_gpu`, `gemm_iq4xs_wmma_64x64_gpu`, `gemm_iq4xs_lut_gpu` (`kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp`)
- Stock comparators: `gemv_iq4xs_stock_gpu`, `gemm_iq4xs_stock_gpu`, `gemv_iq4xs_stock_dp4a_gpu`

**CPU reference oracles:** `<op>_iq4xs_cpu_ref` — `gemv_iq4xs_cpu_ref`, `gemm_iq4xs_cpu_ref`, `dequant_mat_iq4xs_cpu` (declared in `kernels/matmul_iq4xs/ref_cpu.h`).

**Buffers:**
- Device pointers: `d_` prefix (`d_W`, `d_x`, `d_y`, `dX`, `dY`, `d_src`, `d_dst`).
- Host vectors: `h_W`, `h_x`, or role-based names (`y_ref`, `y_gfx`, `y_stock`, `Y_tiled`).
- Dimensions: `int64_t K` (in-features), `int64_t N` (out-features/rows of W), `int64_t M` (batch rows of X/Y).

**Types:**
- `struct`/`class` in `UpperCamelCase` — `Metrics`, `BenchStats`, `CompareResult`, `MatmulShape`, `GemmCase`, `block_q8_1_coop`.
- Tensor dims are `int64_t`; byte counts are `size_t`; quantized payload fields are `uint8_t`/`uint16_t`.

### Code Style

**Formatting:**
- No automatic formatter configured (no `.clang-format`, `.editorconfig`, or pre-commit hook in the repo root). Style is manual and consistent.
- 4-space indentation, K&R braces (opening brace on the same line as the function/control statement).
- Long `printf` format strings split across lines with continuation.

**Linting:**
- No linting config. The only enforced static rule is the include-isolation gate: `scripts/check_no_ggml.sh` greps `kernels/` (excluding `build`) for `#include [<"]ggml|llama...` and fails the build check if any is found. **Every new kernel file must stay free of ggml/llama headers.**

### Import Organization

Headers are grouped, local-first: the block/type header and shared util headers, then stdlib.

```cpp
// kernels/matmul_iq4xs/test_gemv_compare.cpp
#include "ref_cpu.h"
#include "matmul_test_util.h"
#include <vector>
#include <cstdio>
#include <random>
```

HIP implementations add HIP headers last (`#include <hip/hip_fp16.h>`), plus any variant header first if used (`gemv_variant_xor.cuh` in `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`).

### Error Handling

**HIP error checking — unconditional abort macro:**
All HIP runtime calls (malloc, memcpy, launch, sync, event ops) are wrapped in `HIP_CHECK(...)` from `kernels/common/hip_helpers.h`, which prints the file/line/error-string and calls `std::abort()`:

```cpp
#define HIP_CHECK(ans) do { \
    hipError_t err = (ans); \
    if (err != hipSuccess) { \
        fprintf(stderr, "[HIP ERROR] %s:%d: %s (code %d)\n", __FILE__, __LINE__, hipGetErrorString(err), (int)err); \
        std::abort(); \
    } \
} while (0)
```

`HIP_EVENT_CHECK` is an alias (`kernels/common/hip_helpers.h`). Never swallow a HIP error or ignore a launch return code; an abort is the intended failure mode.

**Test verdict handling:**
- Each per-shape test function returns `bool` (true = PASS). `main()` ANDs all results and exits `0` on overall PASS, `1` on any FAIL. No partial-credit exit codes.
- NaN/Inf is a hard fail: `compute_metrics()` sets `bad = true` and `pass = false` when any element is NaN/Inf (`kernels/common/matmul_test_util.h`).

### Logging

**No logging library.** Plain `printf` to stdout, unbuffered:
- Every test/bench `main()` calls `setvbuf(stdout, nullptr, _IONBF, 0)` first so output streams live (e.g. `kernels/matmul_iq4xs/test_gemv_compare.cpp`).
- Section banners: `printf("=== FINAL GEMV-CUSTOM: %s ===\n", ok ? "PASS" : "FAIL");`
- Per-case lines use a `[TAG]` prefix: `[GEMV-CUSTOM]`, `[GEMV-DP4A-COOP]`, `[GEMM-WMMA-STREAM]`, `[FIXTURE %-28s]`.
- Bench executables emit a **JSON array to stdout** with `printf` per row, comma-separated (`kernels/matmul_iq4xs/bench_gemv.cpp`, consumed by `benchmarks/tools/run_kernel_bench.py`). Stderr is reserved for usage (`usage()` in `bench_gemm_wmma.cpp`) and pre-flight warnings.

### Comments

**File banner comments** on every test/impl/bench file: `// name.ext — one-line purpose`, then design detail.

- Test files document **pass criteria explicitly** in the banner and in `main()`:
  ```cpp
  // test_gemv_dp4a_compare.cpp — Validate cooperative DP4A GEMV vs CPU oracle and vs real stock DP4A
  // Pass criteria: cosine >=0.999 and max_rel <=1e-3 vs FP64 oracle ...
  ```
- Impl files carry dense architecture annotations: VGPR budget, LDS layout/padding, launch bounds, load widths, bit-exactness notes — e.g. the header block of `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`.
- Requirement IDs are quoted inline (`REQ-STAT-07`, `REQ-PERF-07`, `KERN-01`, `QUAL-01`, `D2-12`).

**JSDoc/TSDoc equivalent:** Doxygen-style comments are NOT used. For C++ the banner + inline section comments are the norm.

### Function Design

**Size:** test/bench case functions are small; one function per scenario. `test_*_compare.cpp` files use `test_one` / `test_shape` / `test_gemv_shape_real` (~20–30 lines each) and a thin `main()` that only enumerates shapes and aggregates booleans.

**Parameters:** explicit positional parameters only — `(const char* name, int64_t K, int64_t N, std::mt19937& rng)`. Kernels take `(const block_iq4_xs* d_W, const float* d_x, float* d_y, int64_t K, int64_t N, hipStream_t stream)` with `stream` last (default `0` / `s`).

**Return values:** `bool` for tests; `Metrics` (by value) for metric computations via `compute_metrics(ref, gpu)`; `BenchStats` from `bench_hip_event(...)` in `kernels/common/bench.h`.

**Bench harness pattern** (`kernels/common/bench.h`): pass the launch as a `std::function<void(hipStream_t)>` lambda:

```cpp
auto gfx_launch = [&](hipStream_t s){ HIP_CHECK(gemv_iq4xs_gfx1100_gpu(dW, dx, dy, K, N, s)); };
BenchStats gfx = bench_hip_event(gfx_launch, 0, 50, 200, total_bytes);
```

Defaults are warmup=50, iters=200. Never use a synthetic-time adjustment — variant races must use real compiled OBJECTs (documented in `bench_gemm_wmma.cpp` banner: "Jitter REMOVED").

**Determinism discipline:** all test data is seeded, never `rand()`:
- Shared RNG: `std::mt19937 rng(42)` in `main()`, threaded through every case.
- Per-family weight seeds are arithmetic offsets of K/N/M — e.g. `gen_iq4xs_weights(W, K, N, 1000 + K*17 + N)` (GEMV), `2000 + M*11 + K` (GEMM), `9000 + K*7919 + N` (DP4A), `12345 + K*7919 + N` (stock).
- Activations: `std::normal_distribution<float> g(0, 1)`.

### Module Design

**Op Quartet** — every op in `kernels/` ships four artifacts wired in `kernels/matmul_iq4xs/CMakeLists.txt`:
1. `ref_cpu.h`/`ref_cpu.cpp` — pure C++ CPU oracle, no HIP includes (`kernels/matmul_iq4xs/ref_cpu.h`)
2. `impl<variant>.hip` — standalone HIP kernel, no ggml/llama headers
3. `test_<op>_compare.cpp` — correctness gate vs the oracle (+ optional cross-check vs a second implementation)
4. `bench_<op>.cpp` — `bench_hip_event` microbenchmark emitting JSON

CMake wiring pattern: `add_library(<name>_hip OBJECT impl*.hip)` + `target_compile_definitions(... PRIVATE VARIANT_FLAG)` for variant objects, then `add_executable(test_* <test>.cpp $<TARGET_OBJECTS:...>)`. The `demo_iq4xs_dequant/CMakeLists.txt` shows the deliberate-broken-mutant pattern: the same `test_compare.cpp` compiled against `impl_broken.hip` with `TEST_BROKEN=1` for negative testing.

**Barrel files:** not used; headers are included directly.

---

## Python Conventions (`benchmarks/`)

### Naming Patterns

**Files:** `snake_case.py` — `run_op_gate.py`, `run_model_gate.py`, `preflight.py`, `guard.py`, `store.py`.

**Functions:** `snake_case`, verbs — `run_perplexity`, `run_prompt_canaries`, `evaluate_model_gate`, `parse_free_mib`, `sha256_file`. Private helpers get `_` prefix (`_read_proc_status_kb`, `_poll_proc_windows`).

**Variables:** `snake_case`. Paths are `pathlib.Path`; module-level constants are `UPPER_SNAKE` (`DEFAULT_BIN`, `PPL_TOLERANCE_PCT`, `CORE_HYBRID_OPS`, `VERDICT_SPILL`).

**Types:** `@dataclass` classes in `UpperCamelCase` — `Thresholds`, `RssProfile`, `GuardVerdict`, `PreflightVerdict` (`benchmarks/lib/guard.py`, `benchmarks/lib/preflight.py`).

### Code Style

**Formatting:** No formatter config in the repo root (no `ruff`, `black`, `flake8`, `pylint`, or `pyproject.toml` at top level; the `llama.cpp/` submodule has its own `.flake8` which does not apply). 4-space indent, ~110-char lines, PEP-8 ordering.

**Typing — mandatory and strict:**
- Every module in `benchmarks/lib/` and `benchmarks/bin/` starts with `from __future__ import annotations` and annotates all signatures: `dict[str, Any]`, `list[str] | None`, `str | Path`, `tuple[int, int]`.
- Return shapes are documented typed dicts for gate results, dataclasses for verdicts.

**Linting:** none enforced.

### Import Organization

Ordered groups, one blank line between: (1) `from __future__ import annotations`, (2) stdlib (alphabetical: `datetime`, `hashlib`, `json`, `os`, `re`, `subprocess`, `sys`, `typing`; `collections`/`dataclasses` where needed), (3) third-party (only `pytest` in tests), (4) local imports `from benchmarks.lib import ...` / `from benchmarks.bin import ...`.

```python
# benchmarks/bin/run_model_gate.py
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
```

**Path Aliases:** none; imports are absolute from repo root (`from benchmarks.lib.guard import VERDICT_PREFLIGHT`). Tests import fixtures from `benchmarks.tests.fixtures.gen_*`.

### Error Handling

- **Raise on missing preconditions** with actionable messages: `raise FileNotFoundError(f"Golden baseline file {golden_path} does not exist. Run with --record-golden first.")` (`benchmarks/bin/run_model_gate.py`).
- **Never fail silently in gates**: every stage returns a structured dict with `"status": "PASS"`/`"FAIL"` and evidence (`error_cases`, `core_ops_status`, `errors[:50]`). Overall verdict ANDs all stages: `overall_pass = (exit_code == 0) and (error_cases == 0) and core_all_pass and (total_cases > 0)`.
- **Locked verdict vocabulary** (never free-text): `VERDICT_OK = "OK"`, `VERDICT_SPILL = "FAILED:suspected-spill"`, `VERDICT_REVIEW = "REVIEW:repeat-deviation"`, `VERDICT_THERMAL = "FAILED:thermal-abort"`, `VERDICT_PREFLIGHT = "FAILED:preflight-oom"` (`benchmarks/lib/guard.py`).
- **Graceful degradation**: `Thresholds.from_json()` returns `None` when the config file is absent → guards run in observe-only mode.
- **Exit codes**: gates return `0` only when every run of an N=10 repeat passes; any error → `1` (honest FAIL discipline — a single failed repeat fails the gate, never averaged away).

### Logging

- `print()` to stdout with `[GATE-TAG]` prefixes: `[QUAL-01] run 2/10`, `[QUAL-02]`, plus module doc comment tags. No `logging` module.
- Every gate writes a **structured JSON result file** with `json.dump(result, f, indent=2)`, `encoding="utf-8"`, after `out_p.parent.mkdir(parents=True, exist_ok=True)` — e.g. `benchmarks/results/phase3/op_gate.json` and `model_gate.json`. For N=10 runs, per-run files are suffixed `.run<N>.json` plus one aggregate file.
- Long-running subprocess output is captured (`stdout=PIPE`), never streamed.

### Comments

**Docstrings on every module and public function** — module docstring first (purpose + requirement IDs), then one per function. Examples: `benchmarks/lib/guard.py`, `benchmarks/lib/store.py`, `benchmarks/lib/fingerprint.py`. Dataclass fields get inline comments for non-obvious units.

### Function Design

- One function per pipeline stage (pure functions where possible): `run_perplexity()` → `run_prompt_canaries()` → `evaluate_model_gate()` → `record_golden()` in `benchmarks/bin/run_model_gate.py`.
- **Every GPU/external dependency is injectable via `mock_*` parameters** (`mock_csv`, `mock_exit_code`, `mock_output`, `mock_returncode`, `mock_ppl`, `mock_canaries`, `mock_results`). This is the primary testability convention — the unit suite runs with zero GPU access.
- `main()` is a thin argparse wrapper; `if __name__ == "__main__": sys.exit(main())`.
- Constants are module-level `DEFAULT_*` so tests and CLI share one source of truth.

### Module Design

**Exports:** modules expose functions and constants; no `__init__.py` re-exports needed (namespace packages via `benchmarks/bin/`, `benchmarks/lib/` — imports are `from benchmarks.bin import run_op_gate`).

**Layer rule:**
- `benchmarks/lib/` — stdlib-only engine modules (CRITICAL: unit-testable without GPU). No third-party imports.
- `benchmarks/bin/` — CLI gate runners that shell out to llama.cpp test binaries.
- `benchmarks/tests/fixtures/` — synthetic trace generators (`gen_llabench_jsonl.py`, `gen_rss_trace.py`, `gen_shmem_snapshot.py`).
- Hardware/env snapshots live as committed text under `benchmarks/environment/` (`rocminfo.txt`, `versions.txt`), never regenerated per test run.

**Shebang + future-import on all executables:** `#!/usr/bin/env python3` then `from __future__ import annotations`.

---

*Convention analysis: 2026-08-30*