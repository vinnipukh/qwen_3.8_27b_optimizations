<!-- generated-by: gsd-doc-writer -->

# Testing & Verification Doctrine

This project enforces strict correctness, reproducibility, and safety gates before any
performance claim or custom kernel optimization is accepted.

## Gate Hierarchy

```
Level 0: Unit & Guard Regression Tests (55 tests in benchmarks/tests/)
Level 1: Platform & Environment Gates (ENV-01..04, rocminfo, hipconfig, versions.txt)
Level 2: Constraint & Integrity Gates (BENCH-01, scan_banned_signatures, D2-19 ordering)
Level 3: VRAM & RSS Overcommit Gates (BENCH-03, preflight buffer check, three-signal guard)
Level 4: Vulkan Coverage Gate (D2-04, vulkan_gate.sh 6-part check)
Level 5: Numerical Correctness Gates — Phase 14 doctrine (per-op max_abs / mean_abs / relative error / cosine similarity; full-model same prompt / same seed / same sampling — see ROADMAP-original.md Phase 14) via QUAL-01 test-backend-ops and QUAL-02 perplexity + canaries; per-kernel thresholds: demo `dequant_iq4_xs` tight gate max_abs < 1e-5 / mean_abs < 1e-6 / cosine > 0.99999 (+10× broken discrimination) and matmul gate cosine >= 0.999 / max_rel <= 1e-3 / no NaN/Inf (test_gemv 10/10, test_gemm 11/11, cosine 1.0)
Level 6: Kernel Numerical Comparison (test_compare vs CPU golden ref, KERN-01 — demo `dequant_iq4_xs` tight gate max_abs < 1e-5 / mean < 1e-6 / cosine > 0.99999 +10× broken discrimination; see `.planning/phases/04-kernel-playground-scaffold/`)
Level 7: Matmul Numerical Comparison (test_gemv_compare, test_gemm_compare, cosine 1.0) — custom gfx1100 GEMV/GEMM vs CPU FP64 oracle (`kernels/matmul_iq4xs/ref_cpu.*`), gate cosine >=0.999 / max_rel <=1e-3 / no NaN/Inf (achieved cosine 1.0, max_abs 0 on all cases); see `.planning/phases/05-first-custom-kernel-bottleneck-attack/`
Level 8: Phase 7 DP4A & WMMA True-Stock Comparator Gate — Direct microbenchmark vs real upstream `vec_dot_iq4_xs_q8_1` DP4A implementation with fused `Q8_1` integer activation quantization.
```

## Running the Unit Test Suite

The unit test suite validates wrapper argument construction, reproducibility math,
system fingerprinting, HWiNFO shared memory decoding, thermal watchdog kill command
validation, forked SIGKILL journal crash resilience, VRAM pre-flight allocation logic,
matrix aggregation, op-level correctness gate parsing (QUAL-01), model-level quality gate
tolerances and golden canaries (QUAL-02), bottleneck profiling parser logic (PROF-01/02),
tensor fixture integrity (`test_fixture.py`), and kernel playground numerical comparison / discrimination (`test_demo_iq4xs_dequant.py`).

From repo root in WSL2:

```bash
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q
```

All 55 unit tests run pure-CPU/fixture-driven and execute in under 30 seconds.

## Matmul Kernel Tests

Phase 5 adds standalone HIP correctness and microbenchmark harnesses for IQ4_XS matmul
outside the Python unit suite. Binaries live in `kernels/build/matmul_iq4xs/` (built via
`cmake --build kernels/build --target matmul_test_baseline test_gemv_compare test_gemm_compare bench_gemv bench_gemm bench_matmul`).

### Correctness gates (cosine >=0.999, max_rel <=1e-3, no NaN/Inf — achieved cosine 1.0)

| Binary | Cases | What is tested | Result |
|---|---|---|---|
| `matmul_test_baseline` (`kernels/matmul_iq4xs/test_stock_compare.cpp`) | **16 cases** | Stock HIP comparator (`stock_hip_comparator.hip` naive per-row/per-element) vs CPU FP64 oracle (`ref_cpu.h` `gemv_iq4xs_cpu_ref` / `gemm_iq4xs_cpu_ref`) — 8 canonical GEMV shapes (M=1, `attn_q/k/v`, `attn_gate` 5120x6144, `attn_out`, `ffn_gate/up` 5120x17408, `ffn_down` 17408x5120) + 8 truncated GEMM shapes (M=16/128/64, truncated N to keep CPU feasible) | **16/16 PASS**, `cosine=1.000000`, `max_abs=0` |
| `test_gemv_compare` (`kernels/matmul_iq4xs/test_gemv_compare.cpp`) | **10 cases** | Custom gfx1100 GEMV (`impl_gemv_gfx1100.hip`, Wave32, 128-bit `uint4` qs loads, 8-thread/row cooperative + `__shfl_xor` reduction, double accumulate) vs CPU oracle and vs stock — 8 canonical shapes + 2 small edge (`small_512` 512x512, `small_1024` 1024x2048) | **10/10 PASS**, `cosine=1.0`, `max_rel=0`, zero NaN/Inf; stock also PASS |
| `test_gemm_compare` (`kernels/matmul_iq4xs/test_gemm_compare.cpp`) | **11 cases** | Custom WMMA/tiled GEMM (`impl_gemm_wmma.hip`, `TILE_M=16` tiled fallback with double `acc[16]` + WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` double-buffered `B_lds[2][32][33]` padded) vs CPU oracle and vs stock — `small_512` 512x512 M16/128, `med_1024` 1024x1024 M16/64, `ffn_gate_trunc` 5120x1024 M16/64, `ffn_down_trunc` 17408x512 M16, `attn_q_trunc` 5120x1024 M128, `wmma_5120_512_64`, `wmma_5120_1024_32`, `wmma_large_5120_512_512` | **11/11 PASS**, `cosine=1.0`, `max_rel=0`, `v_wmma_f32_16x16x16_f16` confirmed in `llvm-objdump --mcpu=gfx1100` disasm |

All three harnesses print per-shape `cosine`, `max_abs`, `mean_abs`, `max_rel` and a final `FINAL: PASS/FAIL` (exit code 0/1). They require a gfx1100 GPU at runtime but the CPU oracle (`ref_cpu.cpp`) compiles without HIP for fixture validation.

### Bench sweeps (microbenchmarks)

| Binary | Sweep | Iterations | Archival |
|---|---|---|---|
| `bench_gemv` (`kernels/matmul_iq4xs/bench_gemv.cpp`) | 8 canonical shapes, M=1, custom gfx1100 vs stock HIP, reports `median_us`, `p95_us`, `GB/s`, `speedup` | 50 warmup / 200 measure (`bench.h` `hipEvent` pairs) | `benchmarks/tools/run_kernel_bench.py --bin kernels/build/matmul_iq4xs/bench_gemv --op mul_mat_iq4xs_gemv` → `benchmarks/results/kernels_mul_mat_iq4xs_gemv_*/bench_sweep.json` |
| `bench_gemm` (`kernels/matmul_iq4xs/bench_gemm.cpp`) | 9 shapes (ffn_gate/ffn_down/attn_q x M 16/128/512), tiled+WMMA vs stock, reports `TFLOPS` (`2NMK/median`) | 5 warmup / 20 measure | `... --bin bench_gemm --op mul_mat_iq4xs_gemm` → `benchmarks/results/kernels_mul_mat_iq4xs_gemm_*/bench_sweep.json` |
| `bench_matmul` (`kernels/matmul_iq4xs/bench_matmul.cpp`) | Unified 32 shapes (8 canonical x M 1/16/128/512) | 5 warmup / 20 measure | `... --bin bench_matmul --op mul_mat_iq4xs` → `benchmarks/results/kernels_mul_mat_iq4xs_*/bench_sweep.json` |

Each `run_kernel_bench.py` invocation archives via `benchmarks/lib/store.py` `RunStore` (`bench_sweep.json` + `rows.jsonl` + `manifest.json` + `CHECKSUMS.sha256`). Head-to-head diff and per-shape tables are published in `benchmarks/profiling/KERNEL-BENCH-DIFF.md` (§2 GEMV 8/8 WIN 1.26-2.13x, §3 GEMM 7/9 WIN 1.47-7.50x for M>=128).

Failed / sub-optimal variants are logged per **Rule #10** alongside wins — see `benchmarks/profiling/KERNEL-BENCH-DIFF.md` §4 (e.g., GEMM tiled M=16 `ffn_down` 17408x5120 and `attn_q` 5120x5120 both **0.82x** losses — small-M LDS staging overhead vs stock L1 hits; early float-acc tiled variant failed `max_rel` 1e-2 and was fixed via `double acc[16]`; `v8f16` WMMA lane-mapping compile fail fixed to `v16f16`). Run `bench_sweep.json` entries with `speedup <1.0` are preserved, not hidden.

Fixture reference: `kernels/fixtures/manifest_matmul.json` (32 fixtures, 8 shapes x M {1,16,128,512}, seed 42, real `W_raw` from `models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`).

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
