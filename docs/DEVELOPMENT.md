<!-- generated-by: gsd-doc-writer -->

# Developer Guide

Workflows and conventions for developing custom gfx1100 kernels, running test suites,
and executing reproducible benchmark sessions.

## Working with WSL2 & Git

The repo root lives on the Windows host filesystem (`E:\Projects\qwen_3.8_27b_optimizations`),
accessible in the WSL2 guest at `/mnt/e/Projects/qwen_3.8_27b_optimizations`.

**Rule:** Python harness code and test scripts can be executed directly from `/mnt/e`. However,
C++ source trees and compilation builds (e.g. `/root/llama.cpp`) must remain on the **guest ext4**
filesystem because DrvFs exhibits file-locking incompatibilities with git and cmake.

## Running the Test Suite

The test suite contains 55 unit and regression tests covering wrapper constraints,
reproducibility math, fingerprint manifests, HWiNFO shared memory parsing, thermal watchdog
kill command construction, SIGKILL crash resilience, pre-flight allocation, matrix assembly,
op-level correctness gates (QUAL-01), model-level quality gates (QUAL-02), bottleneck profiling (PROF-01/02),
tensor fixture integrity (`test_fixture.py`), and kernel playground numerical comparison / discrimination (`test_demo_iq4xs_dequant.py`).

From repo root in WSL2:

```bash
# Run complete test suite:
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q

# Run specific test module:
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_demo_iq4xs_dequant.py -v
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_fixture.py -v
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_op_gate.py -v
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_model_gate.py -v
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_bottleneck_profiling.py -v
```

## Running Benchmark Sessions

All benchmark sessions are orchestrated through `benchmarks/bin/run_session.py`:

```bash
# Saturated matrix session (4k, 8k, 16k, 32k):
python3 benchmarks/bin/run_session.py --tiers 4096 8192 16384 32768 --repeats 5 --delay 10

# Fast single-tier reproducibility check:
python3 benchmarks/bin/run_session.py --tiers 8192 --repeats 5 --delay 10

# Smoke test (tiny 1024 context, 1 repeat, 0s delay):
python3 benchmarks/bin/run_session.py --smoke
```

Each session:
1. Acquires `benchmarks/results/.session.lock`.
2. Creates an append-only run directory `benchmarks/results/<timestamp>_<label>/`.
3. Runs the pre-flight check against free VRAM.
4. Executes the pinned binary with live background `/proc` RSS monitoring.
5. Fsyncs every row to `rows.jsonl`.
6. Closes with `CHECKSUMS.sha256` and dispatches a Windows toast notification.

## Running the Layer-2 Prompt Runner

To evaluate greedy token generation over the 6 deterministic prompt files in `benchmarks/prompts/`:

```bash
python3 benchmarks/bin/run_prompts.py --tier 4096 --gen 128
```

## Running Calibration

To discover sensor labels, derive guard thresholds, or test near-OOM safety:

```bash
# Discover HWiNFO sensor labels:
python3 benchmarks/bin/calibrate.py labels

# Rehearse thermal watchdog kill path on dummy process:
python3 benchmarks/bin/calibrate.py rehearse-kill

# Profile healthy runs (4k/8k) and write benchmarks/config/thresholds.json:
python3 benchmarks/bin/calibrate.py profile

# Supervised near-OOM verification on tier 32768:
python3 benchmarks/bin/calibrate.py near-oom
```

## Publishing Matrix Reports

To aggregate one or more benchmark sessions and publish `BASELINE-MATRIX.md`:

```bash
python3 benchmarks/bin/publish_matrix.py \
  benchmarks/results/20260823_164724_baseline_hip \
  --repro-run benchmarks/results/20260823_170839_baseline_hip
```

## Building and Running the Kernel Playground (Phases 4–5)

See complete hardware ISA and kernel reference library at [`.planning/reference/GPU-KERNEL-RESOURCES.md`](../.planning/reference/GPU-KERNEL-RESOURCES.md)
and Phase 4 design at [`.planning/phases/04-kernel-playground-scaffold/04-CONTEXT.md`](../.planning/phases/04-kernel-playground-scaffold/04-CONTEXT.md).

The playground operates completely **outside llama.cpp** (KERN-01):

1. Author CPU golden reference in `kernels/<op>/ref_cpu.cpp` (pure C++17, no HIP, vendored `kernels/common/block_iq4_xs.h` — never include `ggml.h`; verified by `bash scripts/check_no_ggml.sh`).
2. Implement HIP kernel targeting `gfx1100` in `kernels/<op>/impl.hip` (`template<int WarpSize>` + `__launch_bounds__(256,4)`, bench both wave32 and wave64).
3. Validate numerical tolerance (`kernels/<op>/test_compare.cpp`) against CPU reference — tight gate for `dequant_iq4_xs` is max_abs < 1e-5 / mean < 1e-6 / cosine > 0.99999; broken variant must fail by ≥10×.
4. Run microbenchmarks (`kernels/<op>/bench_sweep.cpp` via `kernels/common/bench.h`: 50 warmup / 200 measure via `hipEvent_t`, median/p95/min/max/stdev) comparing against stock; `bench_sweep.json` is archived via `benchmarks/tools/run_kernel_bench.py` and `benchmarks/lib/store.py` with fingerprint (commit `bb4caa75`, ROCm 7.2.1, artifact `53adc4bb…`).
5. Integrate winning kernels behind ON/OFF compile flags via quilt patches (Phase 6).

Build the playground standalone:

```bash
cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build
```

Run demo op tests and benchmark:

```bash
export HSA_ENABLE_DXG_DETECTION=1

# Correct implementation (passes GREEN):
./kernels/build/demo_iq4xs_dequant/demo_test

# Deliberately broken implementation (fails RED with >1000x error):
./kernels/build/demo_iq4xs_dequant/demo_test_broken

# Microbenchmark sweep across wave32 and wave64:
./kernels/build/demo_iq4xs_dequant/demo_bench

# Archive fingerprinted benchmark run:
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op demo_iq4xs_dequant
```

Fixtures are generated via `python3 tools/dump_gguf_fixtures.py` combining real GGUF `block_iq4_xs` blocks (136B/256 weights) and deterministic synthetic edge cases (zero, min/max scale, nibble extremes, split-half boundary).

## Building and Running Matmul Kernels (Phase 5)

Phase 5 extends the playground with `kernels/matmul_iq4xs/` — a standalone IQ4_XS `MUL_MAT` GEMV/GEMM implementation for `gfx1100`. It provides a FP64 CPU golden oracle (`kernels/matmul_iq4xs/ref_cpu.cpp` / `ref_cpu.h` exposing `gemv_iq4xs_cpu_ref` / `gemm_iq4xs_cpu_ref` over 8 canonical Qwen3.8-27B shapes), a stock HIP comparator (`stock_hip_comparator.hip`), and custom kernels (`impl_gemv_gfx1100.hip` — 8 threads/row, 128-bit `uint4` loads, `double` accumulate; `impl_gemm_wmma.hip` — `TILE_M=16` tiled fallback + WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` with `B_lds[2][32][33]` double-buffer). Reference: [`.planning/phases/05-first-custom-kernel-bottleneck-attack/05-CONTEXT.md`](../.planning/phases/05-first-custom-kernel-bottleneck-attack/05-CONTEXT.md).

### Build

`kernels/CMakeLists.txt` conditionally adds the op via `if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/matmul_iq4xs/CMakeLists.txt") add_subdirectory(matmul_iq4xs) endif()` — so the standard playground configure handles `matmul_iq4xs` automatically:

```bash
cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build
```

The shorthand form also works when defaults are acceptable:

```bash
cmake -S kernels -B kernels/build
cmake --build kernels/build
```

This produces `kernels/build/matmul_iq4xs/` targets: `libmatmul_ref_cpu.a`, `matmul_test_baseline`, `test_gemv_compare`, `test_gemm_compare`, `bench_gemv`, `bench_gemm`, `bench_matmul` (unified sweep). The top-level `matmul_stock_test` alias depends on `matmul_test_baseline`.

### Dumping Fixtures

Real-weight fixtures are extracted from the IQ4_XS GGUF via `tools/dump_matmul_fixtures.py`. It maps the 8 canonical shapes (`attn_q/k/v/gate/out`, `ffn_gate/up/down`) to GGUF tensors (e.g. `blk.0.ffn_gate.weight`, `blk.0.ffn_down.weight`), falls back to deterministic synthetic `block_iq4_xs` blocks when no GGUF is present, and generates Gaussian `X` (`seed 42`) plus `Y_ref` via `gguf-py` `dequantize()` for each `M`:

```bash
# All canonical shapes, all M tiers (default --ms 1 16 128 512) to kernels/fixtures/:
python tools/dump_matmul_fixtures.py --model models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf --out kernels/fixtures --ms 1 16 128 512

# Subset of shapes:
python tools/dump_matmul_fixtures.py --model models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf --out kernels/fixtures --shapes ffn_gate ffn_down --ms 1 16 512

# Using defaults (model path above, out kernels/fixtures):
python tools/dump_matmul_fixtures.py
```

Outputs per shape/M to `kernels/fixtures/`:
- `matmul_<name>_<K>x<N>_W.bin` / `matmul_<name>_<K>x<N>_W.npz` — raw `block_iq4_xs` weight blocks (`W_sha256` in manifest)
- `matmul_<name>_<K>x<N>_M<M>.npz` — `W_raw`, `X`/`x`, `Y_ref`/`y_ref`, `K`/`N`/`M` metadata
- `matmul_<name>_<K>x<N>_M<M>_X.bin` and `matmul_<name>_<K>x<N>_M<M>_Y_ref.bin` / `y_ref.bin` — raw `f32` binaries for HIP tests
- `manifest_matmul.json` — array of entries with `W_source`, `W_sha256`, `X_shape`, `Y_shape`, `dtype: IQ4_XS`

Existing fixtures under `kernels/fixtures/` (e.g. `matmul_ffn_gate_5120x17408_M16.npz`) were produced this way.

### Running Correctness Tests

All binaries require `HSA_ENABLE_DXG_DETECTION=1` under WSL2:

```bash
export HSA_ENABLE_DXG_DETECTION=1

# 05-01 stock-vs-oracle baseline — 8 canonical GEMV (M=1) + 8 small GEMM (M=16/128) = 16 cases:
./kernels/build/matmul_iq4xs/matmul_test_baseline
# Expected: 16/16 PASS, cosine 1.000000, max_abs 0, max_rel 0, 0 NaN/Inf

# 05-02 custom GEMV (gfx1100, 8 threads/row, uint4) vs CPU oracle — 8 canonical + 2 small = 10 cases:
./kernels/build/matmul_iq4xs/test_gemv_compare
# Expected: 10/10 PASS, cosine 1.000000 (gate cosine >= 0.999, max_rel <= 1e-3)

# 05-03 custom GEMM (tiled TILE_M=16 + WMMA) vs CPU oracle — 11 cases (small 512/1024, truncated 5120/17408, WMMA-aligned):
./kernels/build/matmul_iq4xs/test_gemm_compare
# Expected: 11/11 PASS, cosine 1.000000 (same gate)

# Via CTest (if configured):
ctest --test-dir kernels/build -R matmul -V
```

**Numerical gate:** Every case computes `cosine = dot(ref,gpu)/(||ref||·||gpu||)`, `max_rel`, `max_abs`, and `has_nan_or_inf`. Pass requires `cosine >= 0.999`, `max_rel <= 1e-3`, and no `NaN`/`Inf`. The stock comparator and both custom kernels achieve `cosine 1.0` (exact `double` accumulation matching `ref_cpu` FP64) with `max_abs 0` on the checked shapes — satisfying the tight Phase 5 gate (stricter than the Phase 4 `dequant_iq4_xs` cosine > 0.99999 gate which applies to dequant only).

### Running Microbenchmarks

Each bench binary emits a JSON array to stdout (parsed by `benchmarks/tools/run_kernel_bench.py`). All use `kernels/common/bench.h` (`50` warmup / `200` measure via `hipEvent_t`, median/p95/min/max/stdev) and report `stock_median_us` vs `gfx1100_median_us`, `gb_s`, and `speedup`:

```bash
export HSA_ENABLE_DXG_DETECTION=1

# GEMV sweep (M=1, 8 canonical shapes, wave32):
./kernels/build/matmul_iq4xs/bench_gemv
# GEMM sweep (tiled + WMMA paths, M in {16,32,64,128,512}):
./kernels/build/matmul_iq4xs/bench_gemm
# Unified sweep (GEMV + GEMM):
./kernels/build/matmul_iq4xs/bench_matmul
```

Example JSON record (one per shape):

```json
{
  "op": "gemv_iq4xs",
  "shape": "ffn_gate",
  "K": 5120,
  "N": 17408,
  "M": 1,
  "stock_median_us": 123.456,
  "gfx1100_median_us": 67.890,
  "speedup": 1.818,
  "winner": "gfx1100"
}
```

### Archiving Benchmark Runs

Fingerprinted archiving uses `benchmarks/lib/store.py:RunStore` (same as Phase 4). Two options:

**1. Via `benchmarks/tools/run_kernel_bench.py` (preferred):**

```bash
# Archive GEMV sweep (default demo op — override --bin/--op for matmul):
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op matmul_gemv --bin kernels/build/matmul_iq4xs/bench_gemv
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op matmul_gemm --bin kernels/build/matmul_iq4xs/bench_gemm
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op matmul --bin kernels/build/matmul_iq4xs/bench_matmul

# Demo op default (for reference):
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op demo_iq4xs_dequant --bin kernels/build/demo_iq4xs_dequant/demo_bench
```

The tool runs the binary with `HSA_ENABLE_DXG_DETECTION=1`, parses JSON stdout, creates `benchmarks/results/kernels_<op>_<timestamp>/` via `RunStore.create`, writes `bench_sweep.json`, appends each row to `rows.jsonl` (fsynced), generates `manifest.json` via `benchmarks/lib/fingerprint.py:collect_manifest` (ROCm 7.2.1, `gfx1100`, commit `bb4caa75`, model `models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`), and finalizes `CHECKSUMS.sha256`.

**2. Manual `RunStore` (when custom post-processing is needed):**

```python
from benchmarks.lib.store import RunStore
import json, subprocess, os

env = os.environ.copy()
env["HSA_ENABLE_DXG_DETECTION"] = "1"
raw = subprocess.check_output(["kernels/build/matmul_iq4xs/bench_gemv"], env=env)
data = json.loads(raw)

store = RunStore.create(results_root="benchmarks/results", label="matmul_gemv")
(store.run_dir / "bench_sweep.json").write_text(json.dumps(data, indent=2))
for row in data:
    store.append_row(row)
store.write_checksums()
```

In either case, verify archived output with `sha256sum -c benchmarks/results/<run_id>/CHECKSUMS.sha256`.
