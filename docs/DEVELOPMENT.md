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

The test suite contains 47 unit and regression tests covering wrapper constraints,
reproducibility math, fingerprint manifests, HWiNFO shared memory parsing, thermal watchdog
kill command construction, SIGKILL crash resilience, pre-flight allocation, matrix assembly,
op-level correctness gates (QUAL-01), model-level quality gates (QUAL-02), and bottleneck profiling (PROF-01/02).

From repo root in WSL2:

```bash
# Run complete test suite:
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q

# Run specific test module:
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

## Adding Custom Kernels (Phases 4–5)

See complete hardware ISA and kernel reference library at [`.planning/reference/GPU-KERNEL-RESOURCES.md`](../.planning/reference/GPU-KERNEL-RESOURCES.md).

1. Author CPU golden reference in `src/ref/`.
2. Implement HIP kernel targeting `gfx1100` in `src/hip/`.
3. Validate numerical tolerance (`test_compare`) against CPU reference.
4. Run microbenchmarks (`bench_sweep`) comparing against stock HIP implementation.
5. Integrate winning kernels behind ON/OFF compile flags via quilt patches.
