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
