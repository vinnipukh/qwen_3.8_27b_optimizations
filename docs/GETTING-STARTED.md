<!-- generated-by: gsd-doc-writer -->

# Getting Started

Step-by-step setup to take a clean Windows 11 machine with an AMD Radeon RX 7900 XT
to running the full benchmark harness against Qwen3.8-27B.

## 1. Windows Host Configuration

1. **Configure `.wslconfig`:**
   Create or edit `C:\Users\<user>\.wslconfig`:
   ```ini
   [wsl2]
   memory=28GB
   swap=16GB
   ```
   Run `wsl --shutdown` in PowerShell.

2. **Driver Verification:**
   Ensure AMD Adrenalin driver `32.0.31041.1004` (26.10.41) or compatible WSL2 driver is installed.

3. **HWiNFO64 Telemetry (Optional but Recommended):**
   Start HWiNFO64, enable "Shared Memory Support" in Settings, and leave sensors running in the background.

## 2. WSL2 Guest Environment Setup

1. **Start Ubuntu 24.04:**
   ```bash
   wsl -d Ubuntu-24.04 -u root
   ```

2. **ROCm & librocdxg Installation:**
   Verify ROCm 7.2.1 and `librocdxg` 1.2.2 are installed. Ensure `/etc/profile.d/rocdxg.sh` contains:
   ```bash
   export HSA_ENABLE_DXG_DETECTION=1
   ```

3. **Verify GPU Enumeration:**
   ```bash
   rocminfo | grep gfx1100
   ```

## 3. Clone Repository & Setup Model

1. **Clone Repo:**
   Clone to Windows drive (e.g. `E:\Projects\qwen_3.8_27b_optimizations`), mounted at `/mnt/e/Projects/qwen_3.8_27b_optimizations` in WSL.

2. **Model Copy:**
   Ensure the locked model file exists on the fast guest ext4 filesystem:
   `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB, SHA-256 `53adc4bb...`).

## 4. Install Test Dependencies & Run Suite

1. **Install pytest in WSL2:**
   ```bash
   apt-get update && apt-get install -y python3-pytest
   ```

2. **Run Pytest Suite:**
   ```bash
   cd /mnt/e/Projects/qwen_3.8_27b_optimizations
   PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q
   ```
   All 55 tests should pass.

## 5. Kernel Playground (Phase 4)

The standalone HIP playground builds with zero llama.cpp headers:

```bash
cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build

# Run correct kernel (GREEN):
export HSA_ENABLE_DXG_DETECTION=1
./kernels/build/demo_iq4xs_dequant/demo_test

# Run mutant kernel (RED, >1000x error discrimination):
./kernels/build/demo_iq4xs_dequant/demo_test_broken

# Run sweep benchmark:
./kernels/build/demo_iq4xs_dequant/demo_bench
```

Fixtures combine real IQ4_XS super-blocks (136 B / 256 weights via `gguf-py`) and deterministic synthetic edge cases (zero, min/max scale, nibble extremes, split-half `lo@i` vs `hi@i+16`). The worked example `kernels/demo_iq4xs_dequant/` runs `test_compare` (correct GREEN, broken RED by ≥10×) and `bench_sweep` (50 warmup / 200 measure per shape × wave32/wave64).

## 6. Matmul IQ4_XS Kernels (Phase 5)

Standalone IQ4_XS matmul playground (`kernels/matmul_iq4xs/`) — GEMV (M=1 decode) and GEMM (M≫1 prefill) against 8 canonical Qwen3.8-27B shapes (5120×5120, 5120×6144, 5120×17408, 17408×5120).

1. **Dump matmul fixtures (32 fixtures: 8 shapes × M {1,16,128,512}, seed 42):**
   ```bash
   python tools/dump_matmul_fixtures.py --model /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf --out kernels/fixtures
   ```
   Extracts real `W` from GGUF tensors (`blk.0.ffn_gate.weight`, `blk.0.ffn_down.weight`, `blk.0.attn_gate.weight`, etc.), generates activations `x`/`X` (Gaussian N(0,1)) and `y_ref` via CPU dequant+Gemm oracle. Outputs `kernels/fixtures/matmul_*_M*.npz` + `.bin` pairs and `kernels/fixtures/manifest_matmul.json`. Requires `gguf-py` and `numpy`; falls back to synthetic weights if the model file is absent.

2. **Build matmul targets:**
   ```bash
   cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
   cmake --build kernels/build
   ```
   Produces `matmul_test_baseline` (stock baseline), `test_gemv_compare` / `test_gemm_compare` (custom vs oracle), and `bench_gemv` / `bench_gemm` / `bench_matmul` (microbenchmarks). Build is gfx1100-only, no fat binary.

3. **Run stock baseline test (validates naive HIP comparator vs CPU oracle):**
   ```bash
   export HSA_ENABLE_DXG_DETECTION=1
   ./kernels/build/matmul_iq4xs/matmul_test_baseline
   ```
   Covers GEMV (M=1) and GEMM (M=16,128) for canonical + synthetic shapes. Pass criteria: `cosine ≥0.999`, `max_rel ≤1e-3`, no NaN/Inf. Expected: 16/16 PASS (`cosine=1.0`, `max_abs=0`).

4. **Run GEMV/GEMM custom tests (custom gfx1100 kernels vs CPU oracle and stock):**
   ```bash
   export HSA_ENABLE_DXG_DETECTION=1
   ./kernels/build/matmul_iq4xs/test_gemv_compare
   ./kernels/build/matmul_iq4xs/test_gemm_compare
   ```
   `test_gemv_compare` validates `impl_gemv_gfx1100.hip` (decode M=1, 128-bit loads, 8-thread/row cooperative) — expected 16/16 PASS. `test_gemm_compare` validates `impl_gemm_wmma.hip` (prefill M≫1, TILE_M=16 + WMMA `v_wmma_f32_16x16x16_f16`) — expected 18/18 PASS (M=1,16,128 variants). Both gate on `cosine ≥0.999`, `max_rel ≤1e-3`.

5. **Run microbenchmarks (hipEvent tracer, median/p95/stdev):**
   ```bash
   export HSA_ENABLE_DXG_DETECTION=1
   ./kernels/build/matmul_iq4xs/bench_gemv   # GEMV decode: 50 warmup / 200 measure, 8 shapes × M=1
   ./kernels/build/matmul_iq4xs/bench_gemm   # GEMM prefill: 5 warmup / 20 measure, 9 shapes × M {16,128,512}
   ./kernels/build/matmul_iq4xs/bench_matmul # Unified 32-shape sweep: 8 shapes × M {1,16,128,512}, 5/20
   ```
   Compare Custom gfx1100 vs Stock HIP (naive per-row/per-element dequant+dot). GEMV expected 1.26–2.13× win (8/8 shapes); GEMM expected 1.76–7.50× win at M≥128 (6/6), with 0.82× losses at M=16 for two small-M shapes (pre-wired, not hidden). Full sweeps archived via `benchmarks/lib/store.py` with `CHECKSUMS.sha256`.

6. **View KERNEL-BENCH-DIFF.md:**
   ```bash
   cat benchmarks/profiling/KERNEL-BENCH-DIFF.md
   ```
   Contains methodology (pp/tg split, tracer config), GEMV/GEMM tables (median µs, GB/s/TFLOPS, speedup), failed/sub-optimal variants, microarchitectural notes (LDS `[32][33]` padding, `__launch_bounds__(256,4)`, WMMA disassembly), and archived RunStore paths (`benchmarks/results/kernels_mul_mat_iq4xs_*_20260825_165353/`).

## 7. Execute Smoke Test & Baseline Session

1. **Run Smoke Matrix:**
   ```bash
   bash benchmarks/tests/smoke_matrix.sh
   ```

2. **Run Full Baseline Matrix:**
   ```bash
   python3 benchmarks/bin/run_session.py --tiers 4096 8192 16384 32768 --repeats 5 --delay 10
   ```

3. **Check Results:**
   Results land in `benchmarks/results/<timestamp>_baseline_hip/` with `manifest.json`, `rows.jsonl`, and `CHECKSUMS.sha256`. The published aggregate table is generated in `benchmarks/results/BASELINE-MATRIX.md`.

## 8. Integrated Custom Kernel Deployment (Phase 6)

1. **Apply Quilt Patch against upstream `bb4caa75`:**
   ```bash
   cd /root/llama.cpp
   git apply /mnt/e/Projects/qwen_3.8_27b_optimizations/patches/0001-gfx1100-mul-mat-custom.patch
   ```

2. **Build Custom Binary (Switch ON):**
   ```bash
   cmake -B build-custom -S . \
     -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
     -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON \
     -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
   cmake --build build-custom -j8
   ```

3. **Run Correctness Gates (QUAL-01 & QUAL-02):**
   ```bash
   export HSA_ENABLE_DXG_DETECTION=1
   # Op gate (0 errors across 4200+ ops):
   python3 benchmarks/bin/run_op_gate.py --bin /root/llama.cpp/build-custom/bin/test-backend-ops

   # Model gate (WikiText-2 PPL 6.4271 and 6/6 canaries):
   python3 benchmarks/bin/run_model_gate.py --cli-bin /root/llama.cpp/build-custom/bin/llama-cli
   ```

## 9. Phase 7 End-to-End Paired A/B — Quickstart (Hybrid DP4A + WMMA)

Complete cold-start to measured decode/prefill uplift on `gfx1100` (RX 7900 XT). Run stock
vs custom back-to-back in **one thermal window** with `hwinfo_daemon`.

### 9.1 Cold-start WSL and GPU enumeration

Run from **PowerShell (host)** — full DXG reset:

```bash
wsl --shutdown
wsl -d Ubuntu-24.04 -u root
```

Inside WSL guest, export the DXG detection flag (persisted in `/etc/profile.d/rocdxg.sh` on provisioned guests):

```bash
export HSA_ENABLE_DXG_DETECTION=1
rocminfo | grep gfx1100
# expected:  Name:                    gfx1100  /  amdgcn-amd-amdhsa--gfx1100
```

If no match, verify ROCm 7.2.1 `/opt/rocm-7.2.1` and `librocdxg` 1.2.2 are on `LD_LIBRARY_PATH`.

### 9.2 Build stock vs custom (persistent `/root`, `ccache`, timeouts)

> **Filesystem:** Always build under `/root` (ext4 guest) — `/root/llama.cpp`, `/root/llama-custom-07`, `/root/models`. `/tmp` is cleared on every `wsl --shutdown`; `/mnt/e` (DrvFs) is slow and breaks HIP symlinks. `/root` survives shutdown.

> **ccache:** `apt-get install -y ccache` and prepend `ccache` via `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache` to cut rebuilds from ~8 min to ~90 s.

Stock is prebuilt at `/root/llama.cpp/build-ci` (pinned `bb4caa75`). Rebuild only if patched tree changed. Custom Phase 7 hybrid (cooperative 8-thread DP4A GEMV + 64×32 double-buffered WMMA GEMM) builds as:

```bash
export HSA_ENABLE_DXG_DETECTION=1
# Optional but recommended: ccache
export CCACHE_DIR=/root/.ccache

# Stock (OFF) — verify bit-identical if needed:
cmake -S /root/llama.cpp -B /root/llama.cpp/build-ci -G Ninja \
  -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF \
  -DCMAKE_BUILD_TYPE=Release

# Custom (ON) — Phase 7 hybrid:
cmake -S llama.cpp -B /root/llama-custom-07 -G Ninja \
  -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /root/llama-custom-07 -j8
```

***Timeouts:*** Wrap every HIP invocation with an explicit bound (`timeout 300 cmake --build ...`, `timeout 90` for `llama-cli`, `timeout 300` for `llama-bench` sweeps) — DXG deadlocks hang without a kill signal.

Quilt provenance check:

```bash
git -C /root/llama.cpp diff HEAD --stat
git -C /root/llama.cpp apply --check /mnt/e/Projects/qwen_3.8_27b_optimizations/patches/0001-gfx1100-mul-mat-custom.patch  # must PASS
```

### 9.3 Paired `llama-bench` — decode vs prefill (thermal paired)

Use identical model and flags for both arms (`-p 4096 -n 128 -ngl 99 -b 2048 -r 3`, pp/tg split reported separately). Run stock then custom in one window with `hwinfo_daemon` recording:

```bash
export HSA_ENABLE_DXG_DETECTION=1
MODEL=/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf

# Optional: start thermal telemetry (host HWiNFO Shared Memory, 1 Hz)
# python3 benchmarks/host/hwinfo_daemon.py --watch --pid-file /tmp/bench.pid --out-dir benchmarks/results/phase7/ &
# python3 benchmarks/host/thermal_watchdog.py --threshold-c 90 &

# Stock
timeout 300 /root/llama.cpp/build-ci/bin/llama-bench \
  -m $MODEL -p 4096 -n 128 -ngl 99 -b 2048 -r 3

# Custom (hybrid DP4A+WMMA)
timeout 300 /root/llama-custom-07/bin/llama-bench \
  -m $MODEL -p 4096 -n 128 -ngl 99 -b 2048 -r 3
```

Compare `pp` (prefill) and `tg` (decode) tokens/s. Phase 7 goal: custom **tg > stock** on decode (M=1, e.g. 5120×5120) and **pp > stock** at M≥128 (prefill contexts 1024–4096). Record raw `stdout` + `RunStore` `rows.jsonl`/`CHECKSUMS.sha256` under `benchmarks/results/phase7/ab_stock_*` and `ab_custom_*` for publication.

> **Thermal discipline:** Run both benches back-to-back without host sleeps, fan overrides, or clock changes — *record-don't-control*. Host watchdog kills at 95 °C junction (`thermal_watchdog.py`). Document clocks/power/temps per row; see `benchmarks/RUNBOOK.md §thermal-policy` and `docs/PUBLICATION.md`.

### 9.4 Paired `llama-cli` — generation sanity (808 vs 849)

Same prompt, same seed, stock vs custom — verifies token coherence and decode throughput on a real chat turn:

```bash
export HSA_ENABLE_DXG_DETECTION=1
MODEL=/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf
PROMPT="Explain liquid neural networks vs transformers in one paragraph"

# Stock
timeout 90 /root/llama.cpp/build-ci/bin/llama-cli \
  -m $MODEL -p "$PROMPT" -n 512 -ngl 99 --temp 0

# Custom
timeout 90 /root/llama-custom-07/bin/llama-cli \
  -m $MODEL -p "$PROMPT" -n 512 -ngl 99 --temp 0
```

Expected on provisioned WSL2 gfx1100: stock generates ~808 tokens, custom ~849 tokens for the same `-n 512` window in prior captures (custom slightly higher `tg` tok/s, not a correctness delta — both pass 6/6 canaries and WikiText-2 PPL 6.4271). The informative delta is the **greeting token**: stock often starts `Hi` while custom starts `Hello` — same semantics, different sampler path only if temperature non-zero; at `--temp 0` outputs are bit-identical per QUAL-01/02 gates.

### 9.5 Operational notes

- **Persistent vs ephemeral:** `/root` (ext4) persists across `wsl --shutdown`; `/tmp`, `/dev/shm`, and `/mnt/e` do not or are not performance-safe. Keep models in `/root/models`, builds in `/root/llama-*`, and results in `benchmarks/results/phase7/` (DrvFs path is for patch sharing only).
- **ccache:** Set `CCACHE_DIR=/root/.ccache` (10 GB cap `ccache -M 10G`). First build populates cache; incremental patch edits rebuild in <2 min.
- **Timeouts (MANDATORY):** Every HIP/bench subprocess must have a bounded timeout — `timeout 90` for `llama-cli`/`test-backend-ops`, `timeout 300` for `cmake --build` and `llama-bench` sweeps. DXG without timeout hangs the WSL guest until `wsl --shutdown`.
- **Thermal:** Launch `benchmarks/host/hwinfo_daemon.py` (HWiNFO Shared Memory v2 `Global\HWiNFO_SENS_SM2`, 1 Hz) and `benchmarks/host/thermal_watchdog.py --threshold-c 90` before the paired sweep; keep one continuous thermal window. `logs/thermal_monitor.log` must show no 90 °C kills — fallback polling (WinError 5 if HWiNFO not running) is degraded but acceptable with an explicit `telemetry_mode: absent` note in `manifest.json`.
- **Step-up discipline:** Validate new builds CPU-first (`-ngl 0`), then partial (`-ngl 10`), then full (`-ngl 99`) before the paired sweep to avoid DXG TDRs.

Next: `docs/PUBLICATION.md §Phase 7` and `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` for raw paths, LDS `[32][33]`/`[2][32][33]` and `__launch_bounds__(256,4)` guardrail audits, and failed-variant log.
