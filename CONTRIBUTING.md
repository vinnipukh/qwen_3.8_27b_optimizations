<!-- generated-by: gsd-doc-writer -->
# Contributing

Thanks for contributing to `qwen_3.8_27b_optimizations` — custom gfx1100 kernels for Qwen3.8-27B IQ4_XS on RX 7900 XT via stock `llama.cpp` (HIP/ROCm).

> Development details: see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) and [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md).

## 1. Setup

All development happens in **WSL2 (Ubuntu 24.04, root user)**. The repo lives on the Windows host at `E:\Projects\qwen_3.8_27b_optimizations` (`/mnt/e/...` in WSL) — but C++ builds must stay on guest ext4 (DrvFs breaks `git`/`cmake` locks).

1. **WSL2 memory** — `C:\Users\<you>\.wslconfig` must contain:
   ```ini
   [wsl2]
   memory=28GB
   swap=16GB
   ```
   Then `wsl --shutdown`. 15 GB guest RAM causes DXG `ENOMEM` during VRAM allocation.

2. **ROCm** — guest ROCm **7.2.1** + `librocdxg` 1.2.2, with `HSA_ENABLE_DXG_DETECTION=1` exported via `/etc/profile.d/rocdxg.sh`. Verify: `rocminfo | grep gfx1100`.

3. **Stock baseline** — never rebuild casually. Pinned at `/root/llama.cpp` commit **`bb4caa75`** (llama.cpp v0.2.0), built:
   ```bash
   cmake -S . -B build-ci -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
   ```
   Archived under `baseline/binaries/v0.2.0-bb4caa75/`. Model on guest ext4: `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB, sha256 `53adc4bb…`).

4. **Kernel playground** — standalone, zero `llama.cpp` headers:
   ```bash
   cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
   cmake --build kernels/build
   export HSA_ENABLE_DXG_DETECTION=1
   ./kernels/build/demo_iq4xs_dequant/demo_test        # GREEN
   ./kernels/build/demo_iq4xs_dequant/demo_test_broken # RED (>1000x error)
   ./kernels/build/matmul_iq4xs/matmul_test_baseline   # 16/16 PASS
   ```

## 2. Methodology Rules

Binding for every change — from `.planning/ROADMAP.md` (inherited verbatim from the original 18-phase plan):

1. **Benchmark before optimizing** — no kernel without a measured baseline matrix.
2. **One optimization at a time** — serialize code changes; parallelize only read-only sweeps.
3. **Keep a stock baseline forever** — never destroy `bb4caa75`; OFF must be bit-identical to stock.
4. **Prefill (M>>1) and decode (M≈1) measured separately** — blended `tok/s` is banned.
5. **Measure VRAM as carefully as throughput** — RSS guard + `vram_ledger.jsonl` on every run.
6. **Do not assume CUDA ideas map to RDNA3** — wave32/wave64, `rocWMMA` semantics, `rocBLAS` gaps must be probed.

Also: keep correctness tests next to every kernel, record compiler/ROCm/driver versions with every result, and publish failed experiments.

## 3. Workflow

**GSD phases 1–6** (see `.planning/ROADMAP.md`): 1 Environment & Stock Baseline → 2 Benchmark Harness & Matrix → 3 Correctness Gates & Bottleneck Profiling → 4 Kernel Playground Scaffold → 5 First Custom Kernel (MUL_MAT) → 6 Integration & Publication. Phases 1–4 produced zero optimizations; Phase 5 delivered GEMV 1.26–2.13× / GEMM 1.76–7.50× (cosine 1.0).

- **Quilt patches** — all integration is additive patches over pinned upstream in `patches/*.patch` (e.g. `phase5_mul_mat_custom.patch` behind `GGML_CUDA_ENABLE_CUSTOM_GFX1100`, OFF default). No hard fork.
- **RunStore** — every benchmark run uses `benchmarks/lib/store.py:RunStore` + `benchmarks/lib/fingerprint.py` → append-only `benchmarks/results/<timestamp>_<label>/` with `rows.jsonl` (fsynced), `meta.json`, `CHECKSUMS.sha256`, and lock `benchmarks/results/.session.lock`. Publish via `benchmarks/bin/publish_matrix.py`.
- **Branch** — `main` is the default; use `feat/<topic>` or `fix/<topic>` branches. See `docs/DEVELOPMENT.md` for full conventions.

## 4. Testing

55 unit/regression tests live in `benchmarks/tests/`:

```bash
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q          # all 55
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_op_gate.py -v
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_model_gate.py -v
PYTHONPATH=. python3 -m pytest benchmarks/tests/test_demo_iq4xs_dequant.py -v
```

- **Kernel correctness** (require `HSA_ENABLE_DXG_DETECTION=1` on WSL2):
  ```bash
  ./kernels/build/matmul_iq4xs/test_gemv_compare   # 10/10 PASS
  ./kernels/build/matmul_iq4xs/test_gemm_compare   # 11/11 PASS
  ./kernels/build/matmul_iq4xs/bench_gemv           # vs stock, 50 warmup / 200 measure
  ./kernels/build/matmul_iq4xs/bench_gemm
  PYTHONPATH=. python3 benchmarks/tools/run_kernel_bench.py --op demo_iq4xs_dequant
  ```
- **Quality gates** — `benchmarks/bin/run_op_gate.py` (QUAL-01, 21 093 ops) and `benchmarks/bin/run_model_gate.py` (QUAL-02, PPL 6.4271 ±1%, 6/6 canaries) must stay green. `test-backend-ops` red blocks any perf claim.

Full harness docs: `docs/DEVELOPMENT.md`.

## 5. PR Checklist

Before requesting review, confirm:

- [ ] `bash scripts/check_no_ggml.sh` **PASS** — no `#include <ggml…>` / `#include "llama…"` in `kernels/` (vendored `kernels/common/block_iq4_xs.h` only)
- [ ] Kernels are `template<int WarpSize>` with `__launch_bounds__(256,4)` (both wave32 and wave64 benched; never literal `32`/`64`)
- [ ] CPU reference co-located (`ref_cpu.cpp`) and `test_compare` gates pass (dequant: max_abs < 1e-5 / mean < 1e-6 / cosine > 0.99999; matmul: cosine 1.0 / gate ≥ 0.999)
- [ ] Results are fingerprinted — `rows.jsonl` + `CHECKSUMS.sha256` via `RunStore`; pp/tg split; failures logged alongside wins
- [ ] Stock baseline intact — OFF/patch-disabled rebuild is bit-identical; `patches/` are additive over `bb4caa75`
- [ ] `PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q` — 55/55 green (or new tests added with `test_*.py` naming)
- [ ] License header preserved — project is **Apache 2.0** (see `LICENSE`); new files carry the same header

Report bugs via GitHub Issues with steps to reproduce, expected/actual behavior, and `rocminfo` + `hipconfig` output.
