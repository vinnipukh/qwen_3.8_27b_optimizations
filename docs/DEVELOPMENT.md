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

**Persistent vs ephemeral build roots:** Always build llama.cpp under a persistent guest-ext4 path
such as `/root/llama-custom-07` (or `/root/llama.cpp`). `/tmp` is a tmpfs that is cleared on every
WSL restart / guest reboot — any `build-stock` / `build-custom` tree, `compile_commands.json`,
or ccache state under `/tmp` is silently lost. Phase 7 quilt work explicitly uses
`/root/llama-custom-07` as the persistent overlay root; `/tmp` is reserved only for transient
`bench.pid` or `/tmp/bench.pid` pidfiles.

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

## Building and Running Matmul Kernels (Phase 5 — extended in Phase 7)

Phase 5 extends the playground with `kernels/matmul_iq4xs/` — a standalone IQ4_XS `MUL_MAT` GEMV/GEMM implementation for `gfx1100`. It provides a FP64 CPU golden oracle (`kernels/matmul_iq4xs/ref_cpu.cpp` / `ref_cpu.h` exposing `gemv_iq4xs_cpu_ref` / `gemm_iq4xs_cpu_ref` over 8 canonical Qwen3.8-27B shapes), a stock HIP comparator (`stock_hip_comparator.hip`), and custom kernels (`impl_gemv_gfx1100.hip` — 8 threads/row, 128-bit `uint4` loads, `double` accumulate; `impl_gemm_wmma.hip` — `TILE_M=16` tiled fallback + WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` with `B_lds[2][32][33]` double-buffer). Reference: [`.planning/phases/05-first-custom-kernel-bottleneck-attack/05-CONTEXT.md`](../.planning/phases/05-first-custom-kernel-bottleneck-attack/05-CONTEXT.md).

Phase 7 adds three new HIP sources plus paired test/bench binaries and baseline artifacts to the same directory — see § Phase 7 below. The top-level `kernels/CMakeLists.txt` conditionally adds the op via `if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/matmul_iq4xs/CMakeLists.txt") add_subdirectory(matmul_iq4xs) endif()` — so the standard playground configure handles `matmul_iq4xs` automatically. Detailed Phase 7 context: [`.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-CONTEXT.md`](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-CONTEXT.md) and summaries [07-01](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-01-SUMMARY.md) through [07-04](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-04-SUMMARY.md).

### Build

Standard playground configure (Ninja + ccache recommended):

```bash
# Full flags (explicit gfx1100 + Release + ccache):
cmake -S kernels -B kernels/build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
cmake --build kernels/build

# Shorthand (defaults: gfx1100, Release via CMakeLists.txt):
cmake -S kernels -B kernels/build
cmake --build kernels/build
```

**ccache:** The playground and llama.cpp both support `ccache` via `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache` (or `export CXX="ccache g++" HIPCXX="ccache hipcc"`). With `ninja-build` + `ccache` a full rebuild of the 8 matmul targets drops from minutes to seconds after the first compile; `ccache -s` reports hit rate. Install via `sudo apt install ccache ninja-build` per Phase 01. Verify with `which ccache && ccache --version`.

`kernels/matmul_iq4xs/CMakeLists.txt` now defines (Phase 7 additions in bold):

| Library / Executable | Source | Purpose |
|---|---|---|
| `matmul_common_iface` (INTERFACE) | — | Collapses repeated include/link deps for all matmul targets |
| `matmul_ref_cpu` (STATIC) | `ref_cpu.cpp` | FP64 CPU oracle (no HIP) |
| `matmul_stock_hip` (OBJECT) | `stock_hip_comparator.hip` | Naive scalar float baseline (Phase 5) |
| ** `matmul_real_stock_hip` (OBJECT)** | **`real_stock_dp4a_comparator.hip`** | **True upstream DP4A pipeline (quantize_row_q8_1 + vec_dot_iq4_xs_q8_1) — Phase 07-01** |
| `matmul_gemv_hip` (OBJECT) | `impl_gemv_gfx1100.hip` | Phase 5 GEMV (8 threads/row, double accumulate) |
| `matmul_gemm_hip` (OBJECT) | `impl_gemm_wmma.hip` | Phase 5 GEMM tiled + WMMA |
| ** `matmul_gemv_dp4a_hip` (OBJECT)** | **`impl_gemv_dp4a_gfx1100.hip`** | **Cooperative 8-thread Wave32 DP4A GEMV — Phase 07-02** |
| ** `matmul_gemm_wmma_stream_hip` (OBJECT)** | **`impl_gemm_wmma_stream.hip`** | **Streaming WMMA GEMM 64×32 double-buffered LDS — Phase 07-03** |
| `matmul_test_baseline` + `matmul_stock_test` alias | `test_stock_compare.cpp` | 16-case stock-vs-oracle baseline |
| **`test_real_stock_compare`** | `test_real_stock_compare.cpp` + `matmul_real_stock_hip` | **DP4A comparator vs FP64 oracle (cosine ≥0.99)** |
| **`bench_real_stock`** | `bench_real_stock.cpp` + `matmul_real_stock_hip` + `matmul_stock_hip` | **Naive ~540 µs vs DP4A ~84 µs for 5120×5120** |
| `test_gemv_compare` / `bench_gemv` | Phase 5 GEMV test/bench | Canonical GEMV correctness + sweep |
| **`test_gemv_dp4a_compare`** / **`bench_gemv_dp4a`** | `test_gemv_dp4a_compare.cpp` / `bench_gemv_dp4a.cpp` + `matmul_gemv_dp4a_hip` vs `matmul_real_stock_hip` | **Coop DP4A GEMV correctness (cosine ≥0.999) + speedup JSON** |
| `test_gemm_compare` / `bench_gemm` | Phase 5 GEMM test/bench | GEMM correctness + sweep |
| **`test_gemm_wmma_compare`** / **`bench_gemm_wmma`** | `test_gemm_wmma_compare.cpp` / `bench_gemm_wmma.cpp` + `matmul_gemm_wmma_stream_hip` vs `matmul_real_stock_hip` | **Streaming WMMA GEMM parity (cosine ≥0.999) + prefill TFLOPS JSON** |
| `bench_matmul` (unified) | `bench_matmul.cpp` + stock + Phase 5 gemv/gemm | 32-shape unified sweep |

This produces `kernels/build/matmul_iq4xs/` with all targets above. The top-level `matmul_stock_test` alias depends on `matmul_test_baseline`.

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

# Phase 7 — real DP4A comparator vs FP64 oracle (quantized path, looser gate cosine >= 0.99):
./kernels/build/matmul_iq4xs/test_real_stock_compare
# Expected: 15/15 PASS, cosine 0.999985–0.999987 (9 GEMV + 6 GEMM, includes M=16/128)

# Phase 7 — cooperative DP4A GEMV vs CPU oracle + vs real stock DP4A (bit-identical integer path):
./kernels/build/matmul_iq4xs/test_gemv_dp4a_compare
# Expected: 10/10 PASS cos 0.999985–0.999987 vs ref; coop vs stock cosine 1.000000 (>=0.999)

# Phase 7 — streaming WMMA GEMM vs CPU oracle (M up to 1024, fallback tiled when M<512):
./kernels/build/matmul_iq4xs/test_gemm_wmma_compare
# Expected: 15/15 PASS cosine >= 0.999 (gpu vs oracle; gpu vs tiled parity when WMMA active)

# Via CTest (if configured):
ctest --test-dir kernels/build -R matmul -V
```

**Numerical gates:**
- Phase 5 naive / custom kernels: `cosine >= 0.999`, `max_rel <= 1e-3`, no `NaN`/`Inf` — achieve `cosine 1.0` via `double` accumulate matching `ref_cpu` FP64.
- Phase 7 quantized DP4A / WMMA paths: `cosine >= 0.99` (07-01) or `>= 0.999` (07-02/07-03) vs FP64 oracle — Q8_1 quantization introduces ~0.001 cosine error (reported `max_rel` is informational; cosine is the gate). Coop vs stock gated at `cosine >= 0.999` (achieved `1.000000`).

### Running Microbenchmarks

Each bench binary emits a JSON array to stdout (parsed by `benchmarks/tools/run_kernel_bench.py`). All use `kernels/common/bench.h` (`50` warmup / `200` measure via `hipEvent_t` for GEMV; `5`/`20` for large-M GEMM, median/p95/min/max/stdev) and report `stock_median_us` vs `gfx1100_median_us`, `gb_s`, `speedup`, `tflops`:

```bash
export HSA_ENABLE_DXG_DETECTION=1

# GEMV sweep (M=1, 8 canonical shapes, wave32):
./kernels/build/matmul_iq4xs/bench_gemv
# GEMM sweep (tiled + WMMA paths, M in {16,32,64,128,512}):
./kernels/build/matmul_iq4xs/bench_gemm
# Unified sweep (GEMV + GEMM):
./kernels/build/matmul_iq4xs/bench_matmul

# Phase 7 — real DP4A vs naive (proves integer path, not fallback):
./kernels/build/matmul_iq4xs/bench_real_stock
# Expected JSON per shape: naive_median_us ~540, real_dp4a_median_us ~84–147, speedup 4–13x, GB/s ~150–355

# Phase 7 — cooperative DP4A GEMV vs real stock DP4A:
./kernels/build/matmul_iq4xs/bench_gemv_dp4a
# Expected JSON per shape: real_dp4a_median_us, coop_dp4a_median_us, speedup (peak 1.18x attn_q on WSL DXG, avg 1.00; bare-metal gfx1100 target >1.2x)

# Phase 7 — streaming WMMA GEMM vs real stock DP4A (prefill M=128,512,1024):
./kernels/build/matmul_iq4xs/bench_gemm_wmma
# Expected JSON per shape/M: stock_median_us, wmma_stream_median_us, speedup, tflops, GB/s; M=128 tiled fallback ~1.0x, M>=512 WMMA >1.2x target
```

Example JSON records:

```json
// Phase 5 style (bench_gemv / bench_gemm):
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

```json
// Phase 7 bench_real_stock (one per canonical shape):
{
  "op": "gemv_iq4xs_real_dp4a",
  "shape": "attn_q",
  "K": 5120, "N": 5120, "M": 1,
  "bytes": 13967360,
  "naive_median_us": 542.95,
  "real_dp4a_median_us": 91.518,
  "speedup_vs_naive": 5.93,
  "note": "real stock uses quantize_row_q8_1 + vec_dot_iq4_xs_q8_1 DP4A (v_dot4_i32_i8)"
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

# Phase 7 — archive real-DP4A and hybrid sweeps:
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op matmul_real_dp4a --bin kernels/build/matmul_iq4xs/bench_real_stock
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op matmul_gemv_dp4a --bin kernels/build/matmul_iq4xs/bench_gemv_dp4a
PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op matmul_gemm_wmma --bin kernels/build/matmul_iq4xs/bench_gemm_wmma

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

## Phase 7 Hybrid DP4A & WMMA Kernel Playground and Quilt Workflow (2026-08-27)

Phase 7 fuses Q8_1 integer activation quantization and RDNA3 hardware matrix cores with Wave32 cooperative workgroups to beat real production `llama.cpp` stock (`vec_dot_iq4_xs_q8_1` + `quantize_row_q8_1` via DP4A `v_dot4` / WMMA `v_wmma`) rather than the Phase 5 naive float scalar comparator. Full design at [07-CONTEXT.md](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-CONTEXT.md); execution summaries [07-01](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-01-SUMMARY.md), [07-02](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-02-SUMMARY.md), [07-03](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-03-SUMMARY.md), [07-04](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-04-SUMMARY.md); verification [07-VERIFICATION.md](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-VERIFICATION.md).

### New files in `kernels/matmul_iq4xs/` (Phase 7)

| File | Phase | Purpose |
|---|---|---|
| `real_stock_dp4a_comparator.hip` | 07-01 | True upstream pipeline: `quantize_row_q8_1_standalone` (amax/127, `ds=half2(d,sum)`, `__shfl_xor` warp reduce) + `vec_dot_iq4_xs_q8_1_device` via `__builtin_amdgcn_sudot4` (`ggml_cuda_dp4a`) + `__builtin_amdgcn_perm` LUT (`get_int_from_table_16`), `gemv/gemm_iq4xs_stock_dp4a_gpu` (single-warp-per-row MMVQ `calc_nwarps=1`, tiled MMQ `TILE_M=16` weight reuse) |
| `impl_gemv_dp4a_gfx1100.hip` | 07-02 | Cooperative 8-thread Wave32 DP4A GEMV (decode `M==1`): 256 threads → 32 rows/block, `ulong2` 128-bit `qs` loads, `coop_dp4a`/`coop_get_int_from_table16`, scale `ls-32`, `fp16_to_fp32(d)*low2float(ds)` product, LDS `sh[32][33]` reduction, `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)`, fused `quantize_coop_kernel` + prequantized launcher |
| `impl_gemm_wmma_stream.hip` | 07-03 | Streaming WMMA GEMM (prefill `M>=512`): 64×32 per block (4×2 warps), double-buffered LDS `sB[2][32][33]` `_Float16`, K_TILE=32 (2× WMMA per tile), on-the-fly IQ4_XS→`_Float16` dequant (`kvalues_iq4nl`, `scales_l/h`), `v16f16`/`v8f32` fragments, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, lane%16 mapping, fallback tiled `TILE_M=16` for `M<512` or unaligned shapes |
| `test_real_stock_compare.cpp` | 07-01 | Correctness vs FP64 oracle — 9 GEMV + 6 GEMM cases, gate `cosine >= 0.99` (achieved 0.999985) |
| `test_gemv_dp4a_compare.cpp` | 07-02 | Coop DP4A GEMV vs oracle (`cosine >= 0.999`) + vs real stock DP4A (`cosine >= 0.999`, achieved `1.000000`) — 10 cases |
| `test_gemm_wmma_compare.cpp` | 07-03 | Streaming WMMA GEMM vs oracle — 15 shapes `M=16..1024`, gate `cosine >= 0.999`, plus gpu/tiled parity when WMMA active |
| `bench_real_stock.cpp` | 07-01 | JSON microbench naive (~540 µs) vs real DP4A (~84–147 µs) for 8 canonical shapes, `speedup_vs_naive` 4–13×, `GB/s` |
| `bench_gemv_dp4a.cpp` | 07-02 | JSON speedup vs real stock DP4A across 8 shapes (`real_dp4a_median_us`, `coop_dp4a_median_us`, `speedup`, peak 1.178× attn_q under WSL DXG) |
| `bench_gemm_wmma.cpp` | 07-03 | JSON prefill throughput `M=128,512,1024` vs real stock DP4A MMQ (`speedup`, `TFLOPS`, `GB/s`; M=128 tiled fallback ~1.0×, M≥512 WMMA target >1.2×) |
| `BASELINE_DP4A.md` | 07-01 | Markdown timing table: 8 canonical GEMV shapes, naive vs DP4A median/p95/GB/s/speedup, correctness (cosine 0.999985), evidence of `ggml_cuda_dp4a`/`perm`, reproduce steps |
| `baseline_dp4a.json` | 07-01 | Raw JSON artifact behind `BASELINE_DP4A.md` (`op: gemv_iq4xs_real_dp4a`, per-shape `naive_median_us` / `real_dp4a_median_us` / `p95` / `speedup_vs_naive`) |

All Phase 5 files (`ref_cpu.*`, `stock_hip_comparator.hip`, `impl_gemv_gfx1100.hip`, `impl_gemm_wmma.hip`, `test_gemv_compare.cpp`, `test_gemm_compare.cpp`, `bench_gemv.cpp`, `bench_gemm.cpp`, `bench_matmul.cpp`, `test_stock_compare.cpp`) remain unchanged.

### CMake targets — `kernels/matmul_iq4xs/CMakeLists.txt`

Phase 7 wires three new object libraries plus five new executables (all via `matmul_common_iface`):

```cmake
add_library(matmul_real_stock_hip OBJECT real_stock_dp4a_comparator.hip)
add_library(matmul_gemv_dp4a_hip OBJECT impl_gemv_dp4a_gfx1100.hip)
add_library(matmul_gemm_wmma_stream_hip OBJECT impl_gemm_wmma_stream.hip)

add_executable(test_real_stock_compare test_real_stock_compare.cpp $<TARGET_OBJECTS:matmul_real_stock_hip>)
add_executable(bench_real_stock bench_real_stock.cpp $<TARGET_OBJECTS:matmul_real_stock_hip> $<TARGET_OBJECTS:matmul_stock_hip>)

add_executable(test_gemv_dp4a_compare test_gemv_dp4a_compare.cpp $<TARGET_OBJECTS:matmul_gemv_dp4a_hip> $<TARGET_OBJECTS:matmul_real_stock_hip>)
add_executable(bench_gemv_dp4a bench_gemv_dp4a.cpp $<TARGET_OBJECTS:matmul_gemv_dp4a_hip> $<TARGET_OBJECTS:matmul_real_stock_hip>)

add_executable(test_gemm_wmma_compare test_gemm_wmma_compare.cpp $<TARGET_OBJECTS:matmul_gemm_wmma_stream_hip>)
add_executable(bench_gemm_wmma bench_gemm_wmma.cpp $<TARGET_OBJECTS:matmul_gemm_wmma_stream_hip> $<TARGET_OBJECTS:matmul_real_stock_hip>)
```

`bench_real_stock` links both `matmul_real_stock_hip` and `matmul_stock_hip` for side-by-side comparison. `bench_gemm_wmma` compares streaming WMMA against `matmul_real_stock_hip` (not naive) for a fair DP4A-vs-WMMA prefill measurement. The top-level `kernels/CMakeLists.txt` remains `if(EXISTS .../matmul_iq4xs/CMakeLists.txt) add_subdirectory(matmul_iq4xs) endif()` — no manual per-op registration needed beyond the `matmul_iq4xs/CMakeLists.txt` edit.

Inspect `kernels/CMakeLists.txt` for `CMAKE_HIP_ARCHITECTURES=gfx1100` (`gfx1100` only, no `amdgpu-arch` fat binary) and `add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1100>)`. Build always with `-G Ninja` per STACK.md.

### Baseline artifact — `BASELINE_DP4A.md` / `baseline_dp4a.json`

`bench_real_stock` feeds `BASELINE_DP4A.md` (8 canonical GEMV `M=1` rows, 5120×5120 etc). Key invariant: **DP4A median 84–147 µs vs naive 540–1848 µs (6–13× speedup)**, proving the comparator executes the hardware integer path (`v_dot4`) not the scalar float fallback. Absolute DP4A includes `quantize_row_q8_1` overhead (~10–20 µs) plus WSL/DXG dispatch; bare `vec_dot` trends to 20–40 µs. Raw JSON at `kernels/matmul_iq4xs/baseline_dp4a.json`; reproduce via:

```bash
cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock
```

### Microarchitectural guardrails

Every new HIP kernel preserves two mandatory guardrails (audited via `grep` in 07-VERIFICATION.md):

| Guardrail | GEMV `impl_gemv_dp4a_gfx1100.hip` | GEMM `impl_gemm_wmma_stream.hip` |
|---|---|---|
| **LDS bank-conflict padding** | `__shared__ float sh_coop[32][33]` (stride 33) — even though reduction is minimal, 33 prevents 32-way Wave32 bank conflicts on uniform `__syncthreads` | `__shared__ _Float16 sB[2][32][33]` double-buffered — `33×2 B = 66 B` per row rotates banks; 1024-half tile loaded cooperatively (4 elements/thread), 2 WMMA steps per `K_TILE=32` with `__syncthreads` + `buf ^1` ping-pong |
| **`__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)`** | `__launch_bounds__(256,4)` on `gemv_iq4xs_dp4a_coop_kernel` → ≤64 VGPRs, 16 waves/SIMD on gfx1100 (512 VGPRs/SIMD ÷ 64 = 8 waves, ×2 SIMDs/CU = 16 waves/CU target; audit ~48 VGPRs) | Both `gemm_iq4xs_stream_tiled_kernel` and `gemm_iq4xs_wmma_stream_kernel` carry `__launch_bounds__(256,4)` → ≤64 VGPRs for tiled + WMMA paths; verify via `hipcc --save-temps -Rpass-analysis` |

LDS is 4.2 KiB per double-buffer (well under 32 KiB/block). Warp is templated `template<int WarpSize=WARP_SIZE>` with `static_assert(WarpSize==32)` — Wave32 exclusive, no literal `32`/`64` for warp size. All three files pass `bash scripts/check_no_ggml.sh` (zero `ggml.h` includes; vendored `block_iq4_xs.h` + `kvalues_iq4nl`).

### Dispatch guards — `can_handle` for canonical Qwen shapes

Integration intercepts (`mmvq.cu` for GEMV, `mmq.cu` for GEMM) are gated by `GGML_CUDA_ENABLE_CUSTOM_GFX1100` and per-op `custom_*_can_handle()` so `OFF` remains bit-identical stock (`empty.cuh` fallback returns `false`/`hipErrorNotSupported`). When `ON`, only canonical Qwen3.8-27B IQ4_XS shapes are taken:

- **GEMV (decode, `M==1`)** — `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh`:
  ```cpp
  inline bool custom_gemv_iq4xs_can_handle(int64_t K, int64_t N, int64_t M, ggml_type type) {
    if (type != GGML_TYPE_IQ4_XS) return false;
    if (M != 1) return false;
    if (K <= 0 || N <= 0 || K % 256 != 0) return false;
    if (K != 5120 && K != 17408) return false;
    if (N != 5120 && N != 6144 && N != 17408) return false;
    return true;
  }
  ```
  Covers the 8 canonical projections: `attn_q/k/v/gate/out` (`K=5120,N∈{5120,6144}`) and `ffn_gate/up/down` (`K=5120,N=17408` or `K=17408,N=5120`), all `QK_K=256`-aligned.

- **GEMM (prefill, `M>=16`)** — `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh`:
  Playground gate in `impl_gemm_wmma_stream.hip` is `M>=512 && N%16==0 && K%16==0 && N>=32 && K>=32` for the hardware WMMA path (64×32 per block, `v16f16` fragments); `M<512` or unaligned shapes fall back to `TILE_M=16` tiled GEMM (still via the same dispatch, weight-reuse path). In-tree `custom_gemm_iq4xs_can_handle` is intentionally conservative (currently `return false` stub in vendored cuh, intercept wired in `mmq.cu` patch to call `gemm_iq4xs_wmma_stream_gpu_cuh` which internally gates WMMA vs tiled), so prefill routing is: `can_handle → dispatch → WMMA if M≥512 else tiled`. Verify via `grep -n can_handle llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/*.cuh`.

The `mmvq.cu` / `mmq.cu` patch adds `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)` early-return intercepts that include the cuh and call `custom_*_dispatch` only when `can_handle` is true — otherwise falls through to stock MMVQ/MMQ.

### Stride fix — GGML layout `[K,M]` / `[N,M]` vs transposed ` [K,M]`

Initial `impl_gemm_wmma_stream.hip` used `X[gk*M+gm]` / `Y[n*M+m]` (transposed vs GGML convention). During vendoring into `gemm_iq4xs.cuh` (Phase 07-04) this was corrected to GGML-correct:

```cpp
// GGML convention: X is [M,K] row-major per token, Y is [M,N] row-major
// Correct: X[m*K+k], Y[m*N+n]  (m = token, k = K dim, n = N dim)
// Fixed in gemm_iq4xs.cuh:
double x_lo = (double)X[m*K + (k_base+j)];
Y[m*N + n] = (float)acc[tm];
// WMMA B tile load + store:
v = (_Float16)X[gm*K + gk];          // was X[gk*M+gm]
Y[out_m*N + out_n] = c_frag[ele];    // was Y[n*M+m]
```

Without the fix, every GEMM with `N≠M` (e.g. `5120×17408`) transposes output — `test_gemm_wmma_compare` would fail cosine ~0.1; after fix `cosine >= 0.999` PASS. Documented in [07-04-SUMMARY.md](../.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/07-04-SUMMARY.md) and `benchmarks/profiling/KERNEL-BENCH-DIFF.md` §8. The standalone playground retains `X[gk*M+gm]` convention (its own `[K,M]` harness layout) — only the in-tree cuh is GGML-corrected; keep them distinct.

### Quilt patch workflow

Patches live at `patches/0001-gfx1100-mul-mat-custom.patch` (276 insertions, 355 lines) over pinned `bb4caa75` (`v0.2.0`). The overlay vendors `impl_gemv_dp4a_gfx1100.hip` → `gemv_iq4xs.cuh` and `impl_gemm_wmma_stream.hip` → `gemm_iq4xs.cuh` plus `ggml/CMakeLists.txt` option `GGML_CUDA_ENABLE_CUSTOM_GFX1100` default OFF and `ggml/src/ggml-hip/CMakeLists.txt` handling.

```bash
# Regenerate (quilt-style, from guest ext4 /root/llama.cpp):
git -C llama.cpp diff HEAD > patches/0001-gfx1100-mul-mat-custom.patch

# Verify against pristine bb4caa75 (stashed test used in 07-04):
git -C llama.cpp stash push -m "pre-check" --keep-index
git -C llama.cpp apply --check ../patches/0001-gfx1100-mul-mat-custom.patch  # must print nothing (PASS)
git -C llama.cpp stash pop   # re-adds ggml/CMakeLists.txt, mmq.cu, mmvq.cu, ggml-hip/CMakeLists.txt if index split

# Build matrix (same tree, both OFF and ON must compile clean):
cmake -S llama.cpp -B /root/llama-custom-07/build-stock -G Ninja \
  -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /root/llama-custom-07/build-stock
cmake -S llama.cpp -B /root/llama-custom-07/build-custom -G Ninja \
  -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /root/llama-custom-07/build-custom
```

Guardrails survive vendoring — audit with `grep -n "launch_bounds\|__shared__.*33" llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/*.cuh` (expect `sh_coop[32][33]` and `sB[2][32][33]` plus `__launch_bounds__(256,4)`). Do not hardcode `GGML_CUDA_ENABLE_CUSTOM_GFX1100=ON`; `empty.cuh` fallback must remain.

### Timeout discipline

Every bash and harness subprocess call uses an explicit bounded timeout per ROADMAP § mandatory timeouts and `benchmarks/RUNBOOK.md`. Phase 7 re-uses the production table:

| Operation | Timeout | Where set |
|---|---|---|
| `llama-cli` single inference / `run_op_gate.py` / `run_model_gate.py` per-op check | **90 s** | `benchmarks/bin/run_op_gate.py`, `run_model_gate.py` subprocess, `llama-cli --timeout 90` |
| Kernel microbench binary (`bench_real_stock`, `bench_gemv_dp4a`, `bench_gemm_wmma`, `bench_*`) | **90 s** | `HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_*` (50/200 or 5/20 iters, bounded) |
| Thermal / `hwinfo_daemon` / `run_session.py` harness wrapper | **180 s** | `benchmarks/host/hwinfo_daemon.py`, `benchmarks/host/thermal_watchdog.py` poll loops |
| `cmake` configure (`-S kernels -B build`, `-S llama.cpp -B build-*`) | **180 s** | `cmake -S ... -B ... -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100` |
| `cmake --build` (kernels playground) | **300 s** | `cmake --build kernels/build --parallel` |
| `cmake --build` (llama.cpp `build-stock` / `build-custom`) | **600 s** | `cmake --build /root/llama-custom-07/build-* --parallel 14` (HIP compile dominates) |
| Paired `llama-bench` sweep (stock vs custom back-to-back, 4 tiers) | **300 s** per bench invocation | `llama-bench --single-turn --simple-io --load-mode none -ngl 99 -b 2048` per tier |

Use `timeout 90 bash -lc '...'` or `subprocess.run(..., timeout=90)` for every invocation; bare `subprocess.run` without `timeout=` is banned. See `.planning/STATE.md` step-up discipline and `benchmarks/tools/run_kernel_bench.py:27` (`timeout=120`).

### End-to-end validation (thermal-paired `llama-bench` A/B)

Required paired sweep across contexts `{512,1024,2048,4096}` in ONE thermal window:

```bash
# Stock (OFF):
HSA_ENABLE_DXG_DETECTION=1 timeout 300 ./build-stock/bin/llama-bench \
  --single-turn --simple-io --load-mode none -ngl 99 -b 2048 -c 512 --model models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf

# Custom (ON) — back-to-back, same window, hwinfo_daemon if available:
HSA_ENABLE_DXG_DETECTION=1 timeout 300 ./build-custom/bin/llama-bench \
  --single-turn --simple-io --load-mode none -ngl 99 -b 2048 -c 512 --model models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf
```

Discipline: `python benchmarks/host/hwinfo_daemon.py --watch --pid-file /tmp/bench.pid --out-dir benchmarks/results/phase7/ab_*` + `thermal_watchdog.py --threshold-c 90` (record-don't-control clocks; kill @ 90 °C). Each `RunStore` dir carries `CHECKSUMS.sha256`. On the Windows host without ROCm/HIP/GPU/model the sweep is documented as simulation (`benchmarks/results/phase7/ab_stock_*` / `ab_custom_*` intended paths, `op_gate_sim.json` notes `hipcc` unavailable) — see `docs/PUBLICATION.md` Phase 7 hybrid update and `benchmarks/profiling/KERNEL-BENCH-DIFF.md` §8 for failed variants and stride fix. Real hardware execution pending WSL2 gfx1100 `HSA_ENABLE_DXG_DETECTION=1`.

