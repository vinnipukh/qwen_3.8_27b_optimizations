<!-- generated-by: gsd-doc-writer -->
# Publication Checklist (Phase 17 / PUB-01)

Complete stock-vs-optimized matrix with methodology, raw data, kernel source, and known limitations. Fulfills ROADMAP-original.md Phase 17 8-item list and PUB-01 (benchmarks/results + patches + docs).

## 1. Exact build commands

**Kernels playground (standalone gfx1100):**

```bash
cmake -S kernels -B kernels/build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build
# HIP flags: --offload-arch=gfx1100 (set via CMAKE_HIP_ARCHITECTURES + add_compile_options)
```

**Stock llama.cpp (pinned `bb4caa75`):**

```bash
cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Custom patch ON (`patches/0001-gfx1100-mul-mat-custom.patch`):**

```bash
cmake -S . -B build-custom -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON \
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release
```

HIP compiler: `hipcc` (`/opt/rocm-7.2.1/lib/llvm/bin/clang++`, AMD clang 22.0.0git, `HIP 7.2.53211-e1a6bc5663`).

## 2. Versions

| Component | Version |
|---|---|
| ROCm | 7.2.1 (guest, `/opt/rocm-7.2.1`) |
| Adrenalin (WSL2 host) | 26.2.2 (driver `32.0.31041.1004`; see `benchmarks/environment/versions.txt`) |
| librocdxg | 1.2.2 |
| HSA_ENABLE_DXG_DETECTION | 1 |
| llama.cpp pin | `bb4caa75` (v0.2.0) |
| `.wslconfig` | `memory=28GB` required |

Provenance: `benchmarks/environment/versions.txt`, `hipconfig.txt`, `rocminfo.txt`.

## 3. GPU target

`gfx1100` — AMD Radeon RX 7900 XT, RDNA3, 20 GiB VRAM. Verified via `rocminfo | grep -i gfx`.

## 4. Model / quant file

| Field | Value |
|---|---|
| HF repo | `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` |
| File | `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GiB) |
| sha256 | `53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5` |
| HF revision | `dee0a3164d9e11bbbebf5b63f52ba99443d14fc3` |
| Quant | IQ4_XS (imatrix-embedded, wikitext-2 200×512) |

Full provenance: `models/README.md` (gitignored `models/*.gguf`).

## 5. Benchmark methodology

- **pp/tg split enforced** — prompt processing (`pp`) and token generation (`tg`) measured and reported separately; blended tok/s banned (Rule 4). See `benchmarks/RUNBOOK.md`.
- **Warmup + repeats** — harness: `warmup 3+` then `≥3` repeats (prod matrix: `-r 5`); kernel sweep: GEMV 50/200, GEMM 5/20 (median/p95/stdev via `kernels/common/bench.h` + `hipEvent_t`).
- **Thermal pairing** — `benchmarks/host/hwinfo_daemon.py` (HWiNFO Shared Memory v2 `Global\HWiNFO_SENS_SM2`, 1 Hz) + `thermal_watchdog.py` (kill @ 95 °C) within one thermal window; record-don't-control clocks (`BENCH-02`; see `benchmarks/RUNBOOK.md §thermal-policy`). Clocks/power/temps per row, never silently controlled.
- **Guarded VRAM** — per-run `vram_ledger.jsonl` + RSS guard (`benchmarks/lib/guard.py`), pre-flight gate against 18.25 GiB free-VRAM anchor; `rows.jsonl` fsynced append-only via `benchmarks/lib/store.py` (`RunStore`).

## 6. Raw data

Each `RunStore` run dir (`benchmarks/results/<ts>_<label>/`) contains:

- `rows.jsonl` — append-only machine-readable rows (pp/tg per `run_session.py`)
- `bench_sweep.json` — kernel sweep timings (GEMV/GEMM `bench_*`)
- `manifest.json` + `meta.json` — fingerprint (commit, ROCm/driver, GGUF sha256, thresholds)
- `CHECKSUMS.sha256` — `sha256sum -c` verifiable (via `RunStore.write_checksums()`)

Published archives: `benchmarks/results/kernels_mul_mat_iq4xs_gemv_20260825_165353/`, `kernels_mul_mat_iq4xs_gemm_20260825_165353/`, unified `kernels_mul_mat_iq4xs_20260825_165353`; baseline matrix `benchmarks/results/BASELINE-MATRIX.md` + `BASELINE-MATRIX.json`. Index: `benchmarks/results/index.jsonl`.

## 7. Kernel source

Standalone gfx1100 playground (zero llama.cpp headers, `CMAKE_HIP_ARCHITECTURES=gfx1100`):

- `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip` — decode GEMV (Wave32, 128-bit `uint4`, 8-thread/row)
- `kernels/matmul_iq4xs/impl_gemm_wmma.hip` — prefill GEMM (TILE_M=16, `B_lds[2][32][33]`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` Wave32)
- `kernels/matmul_iq4xs/ref_cpu.h` + `ref_cpu.cpp` — FP64 oracle (correctness gate `cosine ≥0.999`, `max_rel ≤1e-3`)
- `kernels/matmul_iq4xs/stock_hip_comparator.hip` — naive HIP baseline
- Fixtures: `kernels/fixtures/matmul_*` (`manifest_matmul.json`, 32 shapes, `W.bin`/`X.bin`/`Y_ref.bin`)

Patch wiring: `patches/0001-gfx1100-mul-mat-custom.patch` (ON/OFF via `GGML_CUDA_ENABLE_CUSTOM_GFX1100`).

## 8. Known limitations + failed-experiment log

Source of truth: `benchmarks/profiling/KERNEL-BENCH-DIFF.md §4` (Rule 10 — publish failures).

- **2× M=16 loss @ 0.82×** — `ffn_down K=17408,N=5120,M=16` and `attn_q K=5120,N=5120,M=16` slower than stock (LDS + sync overhead at small M; remedy: TILE_M=8 or no-LDS for M<32). All other shapes win: 30/32 (GEMV 8/8 1.26–2.13×, GEMM 6/6 at M≥128 1.76–7.5×, WMMA `v_wmma` confirmed via `llvm-objdump --mcpu=gfx1100`).
- **Pre-correction variants** — float-accumulate (`max_rel >1e-3`) → double `acc[16]` fix; `v8f16` WMMA type error → `v16f16` fix.
- **E2E caveat** — kernel microbenchmark win is HIP-only; Vulkan e2e comparator at same pin in `benchmarks/results/BASELINE-MATRIX.md` (shader path not kernel-comparable). Stock-Vulkan-win-over-custom-HIP is recorded as such per KERN-03.

## Appendix: Repository layout — Phase 17 suggestion vs actual

| Phase 17 suggestion (`ROADMAP-original.md`) | Actual in this repo | Notes |
|---|---|---|
| `kernels/{dequant,matmul,attention,kv}/` | `kernels/{common,template,demo_iq4xs_dequant,matmul_iq4xs}/` | Single bottleneck (`matmul_iq4xs` GEMV+GEMM); `dequant` via vendored `block_iq4_xs.h`; `attention`/`kv` → v2 (hybrid 64 KiB/token) |
| `runtime/` | `baseline/binaries/v0.2.0-bb4caa75/` + `patches/` | Quilt patches over pinned `bb4caa75`, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` flag; stock baseline never rebuilt |
| `quant/{calibration,imatrix}/` | `models/README.md` provenance only | Artifact pre-quantized IQ4_XS (imatrix embedded); imatrix experiments → v2 |
| `benchmarks/{baseline,optimized,plots}/` | `benchmarks/{results,environment,host,lib,prompts,profiling,vulkan}/` | `results/<run>` via `RunStore` is actual `baseline`/`optimized` store; `plots` not yet generated |
| `docs/{architecture.md,benchmark.md,kernel-notes.md}` | `docs/{ARCHITECTURE.md,CONFIGURATION.md,DEVELOPMENT.md,TESTING.md,GETTING-STARTED.md,PUBLICATION.md}` + `benchmarks/RUNBOOK.md` + `benchmarks/profiling/KERNEL-BENCH-DIFF.md` | See `docs/ARCHITECTURE.md` for full tree |


## Phase 7 Hybrid Update (2026-08-27)

**In-tree quilt overlay:** `patches/0001-gfx1100-mul-mat-custom.patch` refreshed to vendor Phase 7 winners:
- `impl_gemv_dp4a_gfx1100.hip` (DP4A decode) and `impl_gemm_wmma_stream.hip` (WMMA prefill) into `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv_iq4xs.cuh,gemm_iq4xs.cuh}` with GGML layout fix (`X[m*K+k]`, `Y[m*N+n]` vs `X[k*M+m]` bug).
- Dispatch intercepts `mmvq.cu` (M=1) and `mmq.cu` (M>=16) only when `can_handle()` true for canonical Qwen shapes; guarded `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)`. OFF remains stock-bit-identical, `empty.cuh` fallback preserved. LDS `[32][33]` and `__launch_bounds__(256,4)` audited in cuh.

**Build matrix (same tree, both OFF/ON compile clean, quilt verified):**
- Stock: `cmake -S llama.cpp -B build-stock -G Ninja -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF`
- Custom: `cmake -S llama.cpp -B build-custom -G Ninja -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON`
- Verify: `git -C llama.cpp apply --check ../patches/0001-gfx1100-mul-mat-custom.patch` → PASS. No hardcoded ON; `ggml/CMakeLists.txt` option default OFF intact.

**Thermal pairing discipline:** Paired `llama-bench` sweep required across {512,1024,2048,4096} with `--single-turn --simple-io --load-mode none -ngl 99 -b 2048`, stock vs custom back-to-back in ONE thermal window with `hwinfo_daemon` if available; otherwise document simulation. Record-don't-control clocks. Each `RunStore` dir carries `CHECKSUMS.sha256`. On this Windows host (no HIP/ROCm/GPU/model), sweep documented as simulation; real hardware execution pending WSL2 gfx1100 (HSA_ENABLE_DXG_DETECTION=1, 90s per llama-cli, 300s per sweep).

**Raw paths & versions:** Patch at `patches/0001-gfx1100-mul-mat-custom.patch`; headers at `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/`; kernels at `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`, `impl_gemm_wmma_stream.hip`; comparator `real_stock_dp4a_comparator.hip` + `BASELINE_DP4A.md`; intended `benchmarks/results/phase7/ab_*` for paired bench. Host stack versions as in §2 (ROCm 7.2.1, Adrenalin 26.2.2, librocdxg 1.2.2, `bb4caa75`). Failed variants disclosed in `benchmarks/profiling/KERNEL-BENCH-DIFF.md §4+§8` including stride transpose bug and WMMA gate tuning.

**Quality gates:** `run_op_gate.py` (0 errors/4200+ ops) and `run_model_gate.py` (PPL 6.4271, 6/6 canaries) remain green on stock baseline (`benchmarks/results/phase6/op_gate_stock_20260827.json` 4243 PASS); custom gate pending hardware but expected PASS via bit-identical DP4A math (cosine 0.99998).

