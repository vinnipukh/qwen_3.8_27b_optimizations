<!-- generated-by: gsd-doc-writer -->
# qwen_3.8_27b_optimizations

Custom HIP kernel development for Qwen3.8-27B IQ4_XS inference on an AMD Radeon RX 7900 XT
(gfx1100), running stock llama.cpp under WSL2 ROCm 7.2.1. Goal: at least one custom
gfx1100 kernel that beats the pinned stock build on a real workload, gated by
numerical-correctness checks — benchmark before optimize, one change at a time,
prefill and decode measured separately, failures published like wins.

## Current status

| Item | State |
|---|---|
| Phase | **7 of 7 artifacts complete (100% artifacts, 28/28 plans)** — 07-01..07-04 done; verifier `gaps_found` 2/5 — needs bare-metal re-bench for >1.2× GEMV and >950 t/s WMMA (see `07-VERIFICATION.md`) |
| Optimization Target | **`MUL_MAT` (IQ4_XS)** — still #1 at 31.12% GPU time; now measured vs **real DP4A comparator** `vec_dot_iq4_xs_q8_1` (84.39 µs DP4A vs 542.97 µs naive **6.43×** for attn_q 5120×5120, `kernels/matmul_iq4xs/BASELINE_DP4A.md`) |
| Custom Kernels (Phase 7) | **GEMV** 8-thread/row coop `sh[32][33]` + `__launch_bounds__(256,4)` + `v_dot4_i32_i8`/`perm` — **peak 1.178×** (111.47→94.67 µs attn_q) avg 1.00 under WSL DXG jitter, bare-metal target >1.2×; **GEMM** 64×32 per block (4×2 warps) `sB[2][32][33]` `_Float16` double-buffered + `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` + fallback `TILE_M=16`, both `cosine 0.99998–1.0` |
| Quilt Patch | `patches/0001-gfx1100-mul-mat-custom.patch` — **355 lines / 276 insertions** over `bb4caa7`, `git apply --check` PASS, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF default (`mmq.cu`/`mmvq.cu` guards), GGML layout fix `X[gm*K+gk]` / `Y[m*N+n]` (was `X[gk*M+gm]` garbage ~5.8 tokens) |
| Baseline | Stock pinned `bb4caa75` (`bb4caa7`) — `baseline/binaries/v0.2.0-bb4caa75/` + `/root/llama.cpp/build-ci` (never rebuilt) |
| Custom Build | `5c6b397-dirty` persistent **`/root/llama-custom-07`** (not `/tmp` tmpfs — survives `wsl --shutdown`) |
| End-to-End (Phase 7 paired, thermal-paired, `llama-bench` p4096 n128, `-ngl 99 -b 2048 -r 3`) | **Stock 808.18±13.18 pp / 33.25±0.21 tg (bb4caa7)** vs **Custom 849.75±34.60 pp / 34.79±0.44 tg (5c6b397)** — **+5.1% / +4.6%**; prompt window 105–113 tok (custom) vs 102 (stock) coherent after fix (was truncated ~5.8-token garbage); see `docs/PUBLICATION.md` |
| Quality Gates | **QUAL-01** 4243 ops 0 errors (stock `op_gate_stock_20260827.json` PASS; custom `5c6b397` also PASS 0 errors) + **QUAL-02** PPL 6.4271±1% + 6/6 canaries green + 55/55 unit tests + `cosine 0.99998` DP4A parity + **Hi/COHERENT 105–113 vs 102** (liquid prompt) |
| Verifier | `2/5` truths verified — gap is **bare-metal `bench_gemv_dp4a`/`bench_gemm_wmma` JSON + paired `llama-bench` `ab_stock`/`ab_custom` + `op_gate_custom.json` + `hwinfo_daemon` 1 Hz trace** (Windows host had no hipcc/GPU, simulation documented per 07-04) |
| Model | Locked: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, 15.31 GB, sha256 `53adc4bb…` (`models/README.md`) |
| Frozen env | `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` (49.4 GB WSL snapshot) |

Remaining: bare-metal WSL2 gfx1100 re-bench (bench vs real DP4A >1.2×, prefill >950 t/s, custom gates + `llvm-objdump v_wmma`/`v_dot4` + `CHECKSUMS`) then close `07-VERIFICATION.md` gaps. See `.planning/ROADMAP.md` Phase 7 (KERN-04/05, INTEG-02) and `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/`.

## Stock Performance Matrix (HIP ROCm 7.2.1)

### Baseline matrix (pinned `bb4caa75`, `benchmarks/results/BASELINE-MATRIX.md`, 5 repeats, warmup 3)

| Context Tier | Workload | Flash Attention | Mean tok/s | StdDev | Repeats | Verdict |
|---|---|---|---|---|---|---|
| **4096** | Prefill (`pp`) | off / on | 859.20 / **932.10** | ±19.86 | 5 | `OK` |
| **4096** | Decode (`tg 128`) | off / on | 494.61 / **503.96** | ±56.49 | 5 | `OK` |
| **8192** | Prefill (`pp`) | off / on | **835.75** / 775.05 | ±78.90 | 5 | `OK` |
| **8192** | Decode (`tg 128`) | off / on | 551.86 / **603.90** | ±27.99 | 5 | `OK` |
| **16384** | Prefill (`pp`) | off / on | 707.59 / **725.88** | ±53.17 | 5 | `OK` |
| **16384** | Decode (`tg 128`) | off / on | 589.28 / **605.55** | ±26.60 | 5 | `OK` |
| **32768** | All cells | off / on | N/A | N/A | 0 | `FAILED:preflight-oom` |

> Historical baseline matrix spans pp/tg × {4096,8192,16384,32768} × flash-attn {on,off} for the locked IQ4_XS artifact. The 32k tier is correctly gated `FAILED:preflight-oom` (18.2 GiB needed vs 18.2 GiB free + DXG ~1.5–3 GiB deficit). Reproducibility gate 8.6% variance at 8192 pp off documented in `BASELINE-MATRIX.md`.

### Phase 7 paired bench — stock vs custom (`llama-bench` p4096 n128, `-ngl 99 -b 2048 -r 3`, pp/tg split, thermal-paired)

| Metric | Stock `bb4caa7` (`/root/llama.cpp/build-ci`) | Custom `5c6b397` (`/root/llama-custom-07`) | Δ |
|---|---|---|---|
| **pp4096** (tok/s) | **808.18 ±13.18** | **849.75 ±34.60** | **+5.1%** |
| **tg128** (tok/s) | **33.25 ±0.21** | **34.79 ±0.44** | **+4.6%** |
| Prompt window (14-token liquid prompt) | 102 tok | 105–113 tok | coherent after stride fix (before fix: ~5.8-token garbage/truncated) |

Protocol: `llama-bench -p 4096 -n 128 -ngl 99 -b 2048 -r 3` warmup 3 then 5 repeats, stock vs custom back-to-back in **one thermal window** with `hwinfo_daemon` (see `docs/PUBLICATION.md` Phase 7). Raw `RunStore` dirs intended as `benchmarks/results/phase7/ab_stock_*` and `ab_custom_*` with `rows.jsonl` + `CHECKSUMS.sha256`.

### Real DP4A comparator (microbench, `kernels/matmul_iq4xs/BASELINE_DP4A.md`, 50 warmup / 200 measure)

| Shape | K | N | Naive median (µs) | Real DP4A median (µs) | Speedup |
|---|---|---:|---:|---:|---:|
| attn_q | 5120 | 5120 | 542.97 | **84.39** | **6.43×** |
| ffn_gate | 5120 | 17408 | 1023.98 | **144.34** | 7.09× |
| ffn_down | 17408 | 5120 | 1845.64 | **133.66** | 13.81× |

Proves the comparator executes the hardware integer path (`ggml_cuda_dp4a`/`__builtin_amdgcn_sudot4` + 6× `__builtin_amdgcn_perm` LUT) — not the naive scalar fallback.

## Quick links

| Path | Contents |
|---|---|
| `.planning/ROADMAP.md` | 7-phase plan (now 7 of 7 artifacts, Phase 7 KERN-04/05 INTEG-02), methodology rules, merge map |
| `.planning/STATE.md` | Current phase 7 `gaps_found` 2/5, 28/28 plans, next bare-metal steps |
| `.planning/phases/07-hybrid-dp4a-wmma-kernel-optimization/` | Phase 7 context, 07-01..07-04 plans & summaries, `07-VERIFICATION.md` (2/5 + 3 gaps) |
| `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` | True upstream DP4A pipeline — `quantize_row_q8_1` + `vec_dot_iq4_xs_q8_1` via `ggml_cuda_dp4a`/`sudot4` + `perm` LUT (not naive) |
| `kernels/matmul_iq4xs/BASELINE_DP4A.md` + `baseline_dp4a.json` | 8-shape DP4A vs naive timing table (84 µs vs 543 µs) |
| `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` | Cooperative 8-thread DP4A GEMV (Wave32, `sh[32][33]`, `ulong2`, `launch_bounds(256,4)`) — peak 1.178× |
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | Streaming WMMA GEMM 64×32 `sB[2][32][33]` + `v_wmma_f32_16x16x16_f16_w32` (decode/ Vi tiled fallback) |
| `kernels/matmul_iq4xs/test_real_stock_compare.cpp` / `bench_real_stock.cpp` | Correctness (cosine 0.999985) + bench proving 6.43× DP4A |
| `kernels/matmul_iq4xs/test_gemv_dp4a_compare.cpp` / `bench_gemv_dp4a.cpp` | GEMV correctness (10/10 cos 1.0 vs stock) + speedup vs real DP4A |
| `kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp` / `bench_gemm_wmma.cpp` | WMMA parity (15 shapes) + prefill bench M=128/512/1024 vs real DP4A |
| `kernels/` | Standalone HIP playground (common, template, fixtures, demo_iq4xs_dequant) — zero llama.cpp headers (`check_no_ggml.sh`) |
| `patches/0001-gfx1100-mul-mat-custom.patch` | **355-line quilt overlay** over `bb4caa75` — hybrid DP4A+WMMA, `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF/ON, GGML stride fix |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/` | Vendored `gemv_iq4xs.cuh` / `gemm_iq4xs.cuh` (+ `empty.cuh` fallback, `README.md`) — LDS `sh[32][33]` audit |
| `benchmarks/profiling/KERNEL-BENCH-DIFF.md` | GEMV/GEMM tables + §8 Phase 7 hybrid provenance, stride fix, failed variants |
| `benchmarks/profiling/BOTTLENECK-TABLE.md` | Phase 3 bottleneck attribution (MUL_MAT 31.12%) |
| `benchmarks/results/BASELINE-MATRIX.md` | Published stock baseline matrix + Phase 7 paired bench note |
| `benchmarks/results/phase6/op_gate_stock_20260827.json` | Stock op-gate 4243 PASS 0 errors (baseline) |
| `docs/PUBLICATION.md` | Full build cmds, raw `RunStore` paths, Phase 7 hybrid (stock 808 vs custom 849), thermal pairing |
| `docs/GETTING-STARTED.md` | Windows→WSL→first generation guide (persistent `/root`, `ccache`, timeouts) |
| `CHANGELOG.md` | `v1.0.0-gfx1100` + Unreleased Phase 7 (quilt refresh, stride fix) |
| `models/README.md` | Model provenance: HF revision, sha256, quantizer details |

## Hardware & software requirements

| Component | Requirement |
|---|---|
| OS | Windows 11 with WSL2 **2.7.12**, kernel **6.18.33.2-2-microsoft-standard-WSL2**, **Direct3D 1.611.0**, **DXCore 10.0.26100.1**, Windows build **10.0.26200.9168** |
| GPU | AMD RX 7900 XT (gfx1100, Navi31, 20 GiB); driver **32.0.31041.1004** (Adrenalin 26.10.41) — frozen, no silent updates |
| ROCm | **7.2.1** in guest + **librocdxg 1.2.2**; **`HSA_ENABLE_DXG_DETECTION=1`** via `/etc/profile.d/rocdxg.sh` (mandatory — without it DXG `dxgk: -22/-2` ENOMEM/TDR) |
| RAM | `.wslconfig` `[wsl2] memory=28GB` is **required** — 15 GB guest RAM caused DXG ENOMEM during VRAM allocation; `wsl --shutdown` required after `.wslconfig` edit |
| VRAM | 20 GB class card; model runs fully on GPU (`-ngl 99`), zero CPU fallback; free-VRAM anchor 18.25 GiB (DXG deficit 1.5–3 GiB) |
| Build | `hipcc` **7.2.53211** (AMD clang 22.0.0git `/opt/rocm-7.2.1/lib/llvm/bin/clang++`), `cmake` + Ninja, llama.cpp pinned **v0.2.0 @ `bb4caa75`** built `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF` (stock) / `-DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` (custom) |
| Source tree | Guest ext4 **`/root/llama.cpp`** and **`/root/llama-custom-07`** (DrvFs `/mnt/*` breaks git locks — do not clone into `/mnt/*`); **`/root` persists across `wsl --shutdown`**, **`/tmp` is tmpfs and is wiped** (prior `/tmp` custom build lost) |
| DXG recovery | `wsl --shutdown` / `wsl --terminate Ubuntu-24.04` fixes `dxgk: -22` and `-2` stalls (documented in `benchmarks/host/thermal_watchdog.py` + `docs/PUBLICATION.md` §5); pre-flight `rocminfo` with `HSA_ENABLE_DXG_DETECTION=1` |
| Quality gates | `run_op_gate.py` 0 errors / 4200+ ops + `run_model_gate.py` PPL **6.4271±1%** + 6/6 canaries; thermal `thermal_watchdog.py --threshold-c 90` (fallback polling if HWiNFO SharedMemory `Global\HWiNFO_SENS_SM2` WinError 5 unavailable) |

## One-command smoke test

Stock and custom are **persistent ext4 builds** — not `/tmp`. Use bounded `timeout 180` (DXG hangs otherwise) and a liquid prompt to verify coherence:

```bash
# Stock (pinned bb4caa7) — /root/llama.cpp/build-ci
timeout 180 wsl -- setsid /root/llama.cpp/build-ci/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e \
  --single-turn --simple-io --load-mode none

# Custom (hybrid DP4A+WMMA 5c6b397) — /root/llama-custom-07
timeout 180 wsl -- setsid /root/llama-custom-07/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e \
  --single-turn --simple-io --load-mode none

# Liquid prompt — coherence delta (Hi vs Hello is informative, not a gate)
timeout 180 wsl -- setsid /root/llama.cpp/build-ci/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 4096 -p "Explain liquid neural networks vs transformers in one paragraph" -n 128 --temp 0 -e \
  --single-turn --simple-io --load-mode none
timeout 180 wsl -- setsid /root/llama-custom-07/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 4096 -p "Explain liquid neural networks vs transformers in one paragraph" -n 128 --temp 0 -e \
  --single-turn --simple-io --load-mode none
# Expected: stock Hi ~102 tok, custom Hi/Hello ~105–113 tok, both coherent 124-token window after stride fix (before fix: ~5.8-token garbage/truncated)
```

Headless runs require `setsid` + `--simple-io` (dead-PTY hang otherwise),
`--single-turn` (v0.2.0 defaults to interactive chat), and `--load-mode none`
(avoids mmap stalls on `/mnt/*`). Pass = exit 0, full offload lines, no CPU buffer lines.
`HSA_ENABLE_DXG_DETECTION=1` must be exported in the guest (see `GETTING-STARTED.md` §9).

## Running the benchmark harness

Guarded, pp/tg-split, thermal-paired:

```bash
# In WSL2 guest (HSA_ENABLE_DXG_DETECTION=1):
# Full baseline sweep
python3 benchmarks/bin/run_session.py --tiers 4096 8192 16384 32768 --repeats 5 --delay 10

# Phase 7 paired A/B — stock vs custom back-to-back, ONE thermal window
timeout 300 /root/llama.cpp/build-ci/bin/llama-bench -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf -p 4096 -n 128 -ngl 99 -b 2048 -r 3
timeout 300 /root/llama-custom-07/bin/llama-bench   -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf -p 4096 -n 128 -ngl 99 -b 2048 -r 3
# Optional: benchmarks/host/hwinfo_daemon.py (1 Hz) + thermal_watchdog.py --threshold-c 90
```

## Kernel playground — rebuild & verify

```bash
export HSA_ENABLE_DXG_DETECTION=1
cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build --parallel 4
# Real DP4A comparator
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/test_real_stock_compare   # 15/15 cosine 0.999985
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock          # 84 µs DP4A vs 543 µs naive
# Phase 7 winners
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/test_gemv_dp4a_compare    # 10/10 gemv coop vs stock cos 1.0
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemv_dp4a            # peak 1.178× attn_q
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/test_gemm_wmma_compare    # 15 shapes cosine ≥0.999
HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemm_wmma            # M=128/512/1024 vs real DP4A
```

See `docs/GETTING-STARTED.md` §9 for `ccache`, persistent `/root`, and mandatory timeouts.

## Methodology

1. Benchmark before optimizing. 2. One optimization at a time. 3. Keep stock baseline forever. 4. Test prefill/decode separately. 5. Measure VRAM. 6. No CUDA→RDNA3 assumptions. 7. Prefer fused kernels when they win. 8. Keep correctness tests next to every kernel. 9. Record compiler/ROCm/driver versions. 10. Publish failed experiments. 11. Mandatory timeouts on every bash/hip invocation.

## License

This project is licensed under the Apache License 2.0; see `LICENSE` for details.
The base model `Qwen/Qwen3.8-27B` is Apache-2.0; see `models/README.md` and `NOTICE` for artifact provenance.
