# Deep Research: Phase 7 Three Must-Haves — How to Fulfill REQ-WIN-07, REQ-PERF-07, REQ-STAT-07 on RX 7900 XT gfx1100

**Date:** 2026-08-30
**Depth:** Exhaustive (5 angles, 10+ queries, 25+ pages, playwright-cli + web_search)
**Topic:** Fulfilling Phase 7's 3 must-haves on Qwen3.8-27B IQ4_XS 15.3GB on RX 7900 XT gfx1100 under WSL2 ROCm 7.2.1 + Windows HIP SDK 6.4
**Sessions:** overview, technical, windows, market, contrarian (isolated --session)
**Engine:** playwright-cli (npx --package @playwright/cli) + web_search (brave/exa)

## Executive Summary

Phase 7 re-scoped 2026-08-28 adds three must-haves atop `KERN-04/05`:
- **REQ-WIN-07** Windows-native `≤2 langs` `build_windows.bat` `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja` `llama-server.exe :8000 ->200`
- **REQ-PERF-07** `≥1.10x pp+tg` at `{512,1024,2048,4096,8192}` `median ≥1.10x` `mean-1σ ≥1.10x` `N=10` one thermal window (prior `808->849 pp4096 1.051x +5.1%` **FAIL**)
- **REQ-STAT-07** `N≥10 median/mean/stddev/p95` `LLM QA N=15 temp=0 fixed prompt avg tok/s + per-run 15-row table` single-run banned

Bare-metal `WSL2 Ubuntu-24.04 -u root HSA_ENABLE_DXG_DETECTION=1 hipcc 7.2.1` re-bench `N=10 timeout 90/600 weak ODR` on `2026-08-30` proves **we CAN run them** but **0/3 fulfilled** honestly:
- `bench_real_stock 87.8us DP4A vs 548us naive 6.24x` **PASS**
- `bench_gemv +33 0.968 peak 1.148` / `XOR 0.976 peak 1.161 attn_gate` **<1.2x FAIL**
- `bench_gemm 19K valid JSON` `M128 12.5x` `M512 0.70 1.22 peak` `M1024 1.08 avg 1.89 peak` (`peak >1.2x` first, `avg <1.2x`) `M8192 SKIPPED` `stock 10 wmma 3`
- `llama-bench 4-tier N=10 600s` `512 pp 838 vs 904 1.079 0.847 FAIL` `1024 0.996 FAIL` `2048 1.003 FAIL` `4096 0.978 FAIL` `tg 34.8 vs 34.6 0.993 FAIL`

**Bottom line:** `1.10x` is **not a kernel trick** but a **system co-design** problem. `DP4A 9 TF` stock already saturates `800 GB/s`; `WMMA 1024 ops/CU` loses to `on-the-fly d*(ls-32)*kvalues` scalar `40k iters/thread` + `LDS[2][32][33]` `__syncthreads` `20` syncs/row + `WSL2 DXG 15-30us jitter` `stddev 21-26us` `p95 138-156us`. `8192` is `VRAM+grid` `SKIPPED` correctly (`15.3GB+128KiB/tok ->18.5GB/20GB` `BSOD WSL#40732`). `Windows` needs `HIP SDK 6.4` install not code fix. `N=15 LLM QA` is harness-ready not hardware-proven. **Fulfilling requires:** `Windows SDK install` + `fused CK/aiter Flash` + `offline 16x64 swizzle + P=4 XOR + 64x64 B-stationary + LUT mu=4 + W8A8 SmoothQuant 0.5` + `interleaved A,B,A,B` `hwinfo 1Hz + watchdog 90C` `RunStore+CHECKSUMS`.

## Key Findings

1. **ROCm DOES work on Windows — two stacks.** `HIP SDK` natively supports `Windows 11 22H2+` `RX 7900 XT gfx1100` via `C:\Program Files\AMD\ROCm\6.4\bin\clang++.exe --offload-arch=gfx1100` `cmake -DCMAKE_HIP_ARCHITECTURES=gfx1100 -G Ninja` (must **manually** `gfx1100`, `-G Ninja` not `Visual Studio` `cl` fails `__builtin_amdgcn_*`) [rocm.docs system-requirements `gfx1100` enabled](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html) [HIP SDK windows page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html). `WSL2` routes `HIP -> DXCore` via `librocdxg` `ROCDXG` `HSA_ENABLE_DXG_DETECTION=1` [librocdxg](https://github.com/ROCm/librocdxg/) [WSL howto](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html). `wsl --shutdown` 10s + `rocminfo | grep gfx1100` clears `dxgk -22/-2` `ENOMEM`. Project correctly splits: **WSL2 `ext4 /root/models` `mmap` for `15GB` benches** + **native Windows `build_windows.bat` `llama-server.exe :8000 ->200`**.

2. **GEMV 0.97 <1.2x is structural, not occupancy.** `__launch_bounds__(256,4) + amdgpu_flat_work_group_size(256,256) => VGPR 43 <=64 16 waves/SIMD` already **PASS** `hipcc --save-temps -Rpass-analysis` + `llvm-objdump --mcpu=gfx1100 | grep v_dot4` `8x v_dot4 + 24x v_perm` proof, `ulong2` `b128` `__builtin_assume_aligned(16)` `SWDEV-556587` `8x` fewer transactions. Yet `stock MMVQ calc_nwarps=1` single-warp-per-row already `v_dot4` — our `8-thread coop 32 rows/block` adds `LDS[32][33] +33 3%` vs `XOR x'=(y%(32/8))^x 0%` `CK TileWindow` but adds `__syncthreads` per `256` `20` syncs/row `5120` + `quantize_coop` `amax/127` `round` `half2 ds` `__shfl_xor` `16->1` overhead. `XOR 1.161` beats `+33 1.148` `+0.04` but `avg 0.976 <1.2x` due to `WSL2 DXG 15-30us jitter` `p95 138-156us` flattening `1.16 ->0.97`. Bare-metal `16 waves` is `VGPR` not `wave` limited.

3. **GEMM 0.70/1.08 avg <1.2x vs 9 TF DP4A, but M1024 peak 1.89 >1.2x is first PASS.** `WMMA 1024 ops/CU` (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` `v16f16` `v8f32` `lane%16` `half_wave`) vs `DP4A 512 ops/CU` `TILE_M=16 VDR=4` unrolled `9.3 TFLOPS` stock at `M128`. Our `64x32 P2+33` `sB[2][32][33]` `K_TILE=32 2x WMMA` `sched_barrier 0x0080 DS / 0x0008 WMMA` `GMEM->VGPR->LDS->VGPR->WMMA 4-stage` loses to `on-the-fly d*(ls-32)*kvalues_iq4nl` scalar `dl` per `ele` `40k` iters/thread (identical to `fix-p2` `24x slower` `0.04x` `17663us vs 721us`). `weak` ODR fix + `soft HIP_CHECK return` not abort + `M>=8192 SKIPPED` via top `hipMemGetInfo>2GB` preflight makes `19K valid JSON` but `ffn_gate 17408x512` still `hipError 9 grid overflow` `4352 blocks 1M threads` `8KB LDS`. `M1024 1.89 peak` proves `WMMA` can win when `B-stationary 64x64` reuse `64x` amortizes `dl`, but `avg 1.08 <1.2x` needs `offline 16x64 swizzle + P=4 XOR 0% + LUT mu=4`.

4. **8192 SKIPPED is correct, not a FAIL.** `Qwen 64L/8KV GQA` `KV 128 KiB/tok` `8k 1.0GB` `+15.3GB model =18.5GB/20GB` `800 GB/s` `naive 15.3 TB/s` `1000 tok/s` roof per `1000t-s-at-8k` report, `FA` makes `HBM O(N)` but `FLOPs O(N²)` `8k->32k 3-4x` not `16x`, `L3 96MB <<134MB tile`, `WSL2 .wslconfig 28GB` `DXGI 800GB lie` `3.48GB contiguous fail` `BSOD 3-5 OOMs` `microsoft/WSL#40732` `#40401`, `hipMalloc probe 10MB` no retry loops per `T-07-04-02`. Our `bench_gemm` top `M>=8192 SKIPPED` `VRAM>2GB` + `hipMalloc probe` `5x SKIPPED` per `M` is **required** per `07-04` `race.py` `VRAM preflight` + `allow-terminate` `wsl --shutdown` recovery. `REQ-PERF-07` says `if VRAM preflight fails SKIPPED with FA+GQA rationale` — we did.

5. **N=10/15 harness is ready, hardware is not.** `bench_* --runs 10 --json` `median/mean/stddev/p95` `BenchStats stddev_us alias` `runs:10` `p95` `speedup_vs_naive` `note` + `llama-bench -r 10 -o json` `avg_ts/stddev_ts/samples` `500` `Benchmark` `Table` `± stddev` + `race.py --repeats 10` `A,B,A,B not AAAA BBBB` `adelj88` `Genetic+RF tune.py --budget 100` `crowding` + `RunStore rows.jsonl fsynced + CHECKSUMS.sha256` `manifest.json` fingerprint `commit ROCm/driver GGUF sha256` `hwinfo_daemon 1Hz Global\HWiNFO_SENS_SM2 fallback CSV manual-fallback/absent` `thermal_watchdog 90C (RUNBOOK 95C kill)` `wsl --terminate` dual kill modes. `llama-bench 4-tier N=10 600s` `stock then custom` `6.8K JSON valid` `Python` `py` both `3.14.7` `where clang++.exe` not `which` `"%HIP_PATH%"` quoting `core.autocrlf=false` `*.patch eol=lf`. `N=15 LLM QA temp=0 fixed prompt -n 128 avg tok/s + latency + per-run 15-row table` is **harness-documented** in `KERNEL-BENCH-DIFF §8` `PUBLICATION §8` `race.py` but **no `llama-cli --temp 0 N=15` table** on `gfx1100` yet — `single-run banned` honored (`250 synthetic rows.jsonl` `median 1.05 FAIL` not claimed).

## Detailed Analysis

### REQ-WIN-07 Windows-native ≤2 langs — How to Fulfill

**Current:** `build_windows.bat 5857B` correct `HIP_PATH` `C:\Program Files\AMD\ROCm\6.4` env override `if not exist "%HIP_PATH%\bin\clang++.exe"` guard `PATH=%HIP_PATH%\bin` `where clang++.exe && clang++.exe --offload-arch=gfx1100 --version` `where ninja` `vs generator (cl) cannot compile __builtin_amdgcn_*` `cmake -S . -B build-windows -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe" -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang++.exe" -DHIP_PATH="%HIP_PATH%"` `cmake --build build-windows` `build-windows/bin/llama-server.exe --help` `curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}" -> %CURL_CODE% 200` `MODEL_PATH` guard `start /B llama-server -m "%MODEL_PATH%" --port 8000` `timeout /t 15` `wsl --shutdown` note. `kernels/CMakeLists.txt:17` `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` no `/opt/rocm` hardcode `CMAKE_PREFIX_PATH "$ENV{HIP_PATH}" "/opt/rocm"` fallback for `WSL2`. `git config --global safe.directory` `core.autocrlf=false` `*.patch eol=lf` `patches 356 lines 8 files 277 ins` `git -C llama.cpp diff bb4caa75`.

**Why not fulfilled:** `C:\Program Files\AMD\ROCm\6.4` **not installed** `where hipcc/clang++.exe not found` `INFO: Could not find files` `build-windows/bin/llama-server.exe MISSING` `find -name "*.py" ! -path "./llama.cpp/*" ==40` `benchmarks/` `race.py` `swizzle` `tools/` `output/` offline-only.

**Fulfilment steps (no new langs):**
1. `winget install Ninja-build.Ninja` `winget install Kitware.CMake` `VS Build Tools` already `cl` present but must **not** use `cl` for `.hip` — `where ninja` `where clang++.exe` must pass `HIP SDK 6.4` `https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html` `Verify: clang++.exe --offload-arch=gfx1100 --version` `22.0.0git`
2. `set HIP_PATH=C:\Program Files\AMD\ROCm\6.4` `build_windows.bat` `>build-windows\server.log 2>&1` `curl ->200` `choices[0].message.content` proves `gfx1100`
3. `Phase 8 prune` `08-refactor-windows-native` `4 plans 08-01 inventory & deletion allowlist` `08-02 minimal kernels/CMake prune` `08-03 build_windows.bat + HIP toolchain probe` `08-04 patch smoke + server run verification` to `find py ==0` `pure C++/HIP + CMake + .bat` `kernels/` `block_iq4_xs.h hip_helpers.h impl_gemv/gemm winners` `patches/` `CMakeLists.txt` `build_windows.bat` only `llama.cpp` excluded `benchmarks/` `docs/research/freetoken-probe/qstar.mjs` `.planning 01-07` `JSON/JSONL` `harnesses` `tools/swizzle` offline `bench --runs 10` `git apply --check` `PASS` `core.autocrlf=false`

### REQ-PERF-07 ≥1.10x pp+tg at 512..8192 N=10 median + mean-1σ ≥1.10x — How to Fulfill

**Current honest N=10 4-tier 600s `HSA_ENABLE_DXG_DETECTION=1`:** `512 pp 838.3 vs 904.5 1.079 0.847 FAIL` `1024 0.996 FAIL` `2048 1.003 FAIL` `4096 0.978 FAIL` `tg 34.8 vs 34.6 0.993 FAIL` `XOR 1.161` `M1024 peak 1.89` `M512 0.70` `M128 12.5x` due to `tiled` vs `WMMA` compare.

**Why not 1.10x:** `GEMV 0.97` vs `real DP4A 87us` `v_dot4` already; `GEMM scalar dequant` `d*(ls-32)*kvalues` per `ele` `40k` iters dominates `1024 ops/CU` not `512` enough to hide `8KB LDS` `sB[2][32][33] stride33 66B/row` `banks rotate` `ds_write_b128 8 phases 0~7...56~63` `ds_read_b128 4-way 0:3+20:23 75% BW` if not `+33` vs `XOR`. `16x64 swizzle` `128B` `b128 32 thr x4B ->8x16B` `__builtin_amdgcn_global_load_b128` `__builtin_assume_aligned(16)` `hipMalloc 256B aligned` not wired to `offline swizzle_iq4xs.py` `MARLIN 16x64 reshape + cp.async evict_first`. `T=64 ->64x reuse` `loads/out=K·(1/M+1/N) 2K/T` `gfx1100 96 CUs 64x64->128x128 at 8192` `B-stationary weight frag 8 VGPR` `A/B 8 VGPR fp16 / D 8 VGPR wave32` `≤64 VGPR ->16 waves/SIMD` `amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d` predicts `VGPR` before commit, `llvm-objdump --mcpu=gfx1100 | grep v_wmma`.

**Fulfilment high-yield synthesis (from 07-RESEARCH 28 innerText + 3 PDFs + CK Tile docs):**
- **LDS:** `32 banks×4B bank=(addr/4)%32 ds_write_b128 8 phases` `+33` `+3%` (our default) **or** `XOR preshuffle x'=(y%(K/8))⊕x 0%` (`CK TileWindow` `lds_bank_conflicts.html`) `P=4 sB[4][32][32] 8KB` `GMEM->VGPR->LDS->VGPR->WMMA 4-stage` `sched_barrier 0x0080 DS / 0x0008 WMMA` `buf &3` vs `buf^1` double-buffer `weak` ODR fix already.
- **GEMM tiling:** `256 thr 16x16 naive 512K loads vs ideal 32K 16x` `B-stationary` `offline 16x64 swizzle ->128B b128` `T=64 ->64x` `MI300 256x64->304 CUs` `gfx1100 64x64 128x128 at 8192` — our `64x32 P2+33` vs `P4 XOR` vs `64x64 P4_XOR` `128x32` `LUT mu=4` `W8A8 α=0.5` `race.py --repeats 10 A,B,A,B` `interleaved` `median` `mean-1σ` per tier per split `pp/tg` separately.
- **Variants to race:** `MARLIN P=4 16x64 swizzle B in registers` `LUT-GEMM BCQ Σαb μ=4 16-entry LUT 32B bake` `SmoothQuant s_j=max|X_j|^α/max|W_j|^{1-α} α=0.5 fused rmsnorm -> W8A8 INT8 WMMA` `adelj88/rocm_wmma_gemm tune.py Genetic+RF surrogate --budget 100 crowding + race.py --repeats 10` template for `N=10`.
- **Concrete 5-line diff proposal (keep WMMA M>=512, delegate M<512 to Phase5 tiled `gemm_iq4xs_tiled_gpu` proven `1.47-7.39x vs naive`):** `bench_gemm_wmma.cpp` fallback `launch_stream_tiled_gemm` should call `gemm_iq4xs_tiled_gpu` object from `impl_gemm_wmma.hip` for `M<512`, keep `gemm_iq4xs_wmma_stream_kernel` only `M>=512` where `1024 ops` can hide `sB` latency — already `wmma_ok M>=512 && N%16==0 && K%16==0 && N>=32` `BLOCK_N/M` `grid_x/y` `dim3 grid` `dim3 block 256` `hipLaunchKernelGGL`.

**Expected uplift:** `P=4 XOR +33->0% + b128 + 16x64 swizzle` `+13% pp 8192` per `KDB §8` `MARLIN` `T=64 64x reuse` `B-stationary` + `LUT mu=4` `vs inline d*ls*kvalues` `+5%` + `W8A8 α=0.5` `INT8 WMMA` `+8%` -> `1.10x` median `+` `mean-1σ` at `512..8192` if `8192` `SKIPPED` `FA+GQA` rationale holds even if `8192` gated `FAIL`.

### REQ-STAT-07 N≥10/15 Rigour — How to Fulfill

**Current:** `bench_real_stock/gemv/gemm --runs 10 default` `BenchStats median/mean/stddev/p95/min/max/gb_s` `hipEvent 50 warmup 200 iters x runs 10` `agg median-of-medians + pooled within+between var` `speedup_median` `speedup_mean_minus_1sigma` `GB/s` `TFLOPS` `8192 VRAM preflight >2GB + hipMalloc probe 10MB` `no retry loops` `T-07-04-02` `fflush` `setvbuf _IONBF` `incremental JSON` `timeout 90/180/600` `hipcc --offload-arch=gfx1100` clean `Windows HIP_PATH/bin/clang++.exe` probe. `llama-bench -r 10 -o json` `4-tier 512..4096 pp+tg` `stock then custom` `6.8K JSON valid` `avg_ts/stddev_ts/samples` `500` `median/mean/stddev/p95` per `KDB §8` `PUBLICATION §8` `BENCH-01 amended >=10`. **Missing:** `llama-cli --temp 0 fixed prompt -n 128` `N=15` `avg tok/s + latency + stddev + per-run 15-row table` `6/6 canaries` `PPL 6.4271±1%` `N=10` per gate `run_op_gate --runs 10 0 errors` `run_model_gate --runs 10` `PPL 6.4271` not `N=10` yet, `bench_gemm 45/45` only `15/15 attn_q` in `180s` `ffn_*` `17408` `hipError 9` `grid overflow` `soft return` now `SKIPPED` but not `45`.

**Fulfilment:** `race.py --repeats 10 interleaved A,B,A,B` `5 variants 64x32 P2+33 64x32 P4_XOR 64x64 P4_XOR 128x32 LUT_mu4` `TIERS 512..8192` `vram_preflight() >2GB + hipMalloc probe` `hwinfo_daemon 1Hz Global\HWiNFO_SENS_SM2 + watchdog 90C threaded` `RunStore rows.jsonl fsynced + CHECKSUMS.sha256 + manifest fingerprint` `llama-cli N=15` `temp=0` `fixed prompt` `X[gm*K+gk] Y[m*N+n]` `15-row` `avg tok/s + latency` `stddev` `per-run` table `KERNEL-BENCH-DIFF §8` `PUBLICATION` `race_note` `bench_note` `windows_python_help` `python`/`py` both `3.14.7` `where` not `which` `"%HIP_PATH%"` quoting `core.autocrlf=false` `*.patch eol=lf` `timeout 90` `HSA_ENABLE_DXG_DETECTION=1` `WSL2` `DXG -22` `wsl --shutdown` `10s` `rocminfo`.

## Contrarian Views And Risks

- **WSL2 may never hit 1.10x.** `WSL2 DXG` `800GB lie` `3.48GB contiguous fail` `BSOD 3-5 OOMs` `15-30us jitter` flattens `1.16->0.97` `p95 138-156us` `DXG -22 ENOMEM` `1.5-3GB deficit` `librocdxg` `ROCDXG` `rocprofv3 librocdxg unsupported 404 Instinct-only` `llvm-objdump` `v_dot4/v_wmma` `rocprof lds_bank_conflict 0` `WSL2 blind noted` `hwinfo WinError5 fallback polling` `no hwmon in WSL` `no daemon`. `Bare-metal Linux` `native` `8120` `96 CUs` `64x64` `128x128` `T=64` `B-stationary` `swizzle` `cp.async` may be required — project `contingency` `native-Linux` `ladder a rocprofv3/DXG upside -> b llama timers baseline -> c native-Linux contingency` `PROF-01` softest spot `unmerged PR #7016 RDNA3.5 iGPU` `librocdxg officially unsupported`.
- **WMMA scalar dequant is the bottleneck, not WMMA.** `dl = d*(ls-32)` `w = dl*kvalues` per `ele` `16` `B-stationary` still `dl` scalar `float` `half` `v16f16` `40k` iters `M=512` `blocks_per_row 20` `b 0..19` `ib 0..7` `j 0..15` `tm 0..15` `double acc[16]` vs `DP4A VDR=4` `v_dot4` `512 ops/CU` `tiled` `9 TF` `wmmma 1024` not `2x` enough if `dl` not `LUT` `μ=4 16-entry half 32B bake` `B in registers` `MARLIN` `Σαb` `SmoothQuant` `W8A8` `INT8 WMMA` `16x faster` — without `LUT` `WMMA` is `scalar+renders 0.70x`.
- **Windows HIP SDK is install, not code.** `HIP SDK 6.4` `22.0.0git` `gfx1100` `Release` `Ninja` `6.8K JSON valid` `where clang++.exe` `where ninja` `cl cannot compile __builtin_amdgcn_*` `find_package(hip via HIP_PATH)` `*.patch eol=lf` `git diff bb4caa75` `356 lines` `277 ins` `8 files` `5c6b397-dirty` but `C:\Program Files\AMD\ROCm\6.4` `not found` `build-windows/bin/llama-server.exe MISSING` `curl :8000 ->200` `choices[0].message.content` not `py` `40` until `Phase 8` prune `pure C++/HIP + CMake + .bat`.

## Open Questions

- Will `P=4 XOR 0% + b128 + 16x64 swizzle + 64x64 B-stationary + LUT mu=4` push `M=512` `0.70->1.25x` and `M=1024 1.08->1.30x` `avg >1.2x` and `llama-bench 1.079->1.15x` `mean-1σ >1.10x` on `WSL2` or need `bare-metal Linux` `rocprof` `lds_bank_conflict 0` `VGPR 43->64` `16 waves` `v_wmma` `v_dot4`?
- Will `SmoothQuant α=0.5` `W8A8 INT8 WMMA` `s_j` fused `rmsnorm` beat `W4A16 WMMA 1.89 peak` `M1024` or `W4` `4x` memory `vs 512 ops` `INT8 4x` `vs 1024` `2x` trade?
- Will `tune.py Genetic+RF --budget 100` find `128x32 8x2 warps M=8192 128 blocks` winner or `8192` stays `SKIPPED` `FA+GQA 18.5GB/20GB` `800 GiB lie` `BSOD`?
- Will `N=15 LLM QA temp=0 fixed prompt 15-row` `PPL 6.4271 ±1%` `6/6 canaries` stay green at `1.10x` `50x` `hipcc` `clang 22.0` `gfx1100` `Release` `Ninja` `15GB` `ext4 /root/models` `mmap` `DrvFs` `git-lock` `wsl --export` snapshot?

## Sources

**Official/Primary (rocm.docs, amd.com, llvm.org, CK):**
- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html — `gfx1100 RX 7900 XT` `HIP SDK Windows 11 22H2+` enabled `✅` `gfx1100` `RDNA3`
- https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html — `HIP SDK` `Windows` `C:/Program Files/AMD/ROCm` `6.4/7.1` `Core SDK` unified `WSL`
- https://github.com/ROCm/librocdxg — `ROCDXG` `HSA_ENABLE_DXG_DETECTION=1` `W` `HSA` `DXG` `dxgk -22/-2` `librocdxg 1.2.2`
- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html — `WSL howto` `Ubuntu 24.04` `/opt/rocm` `rocminfo`
- https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300x/workgroup.html — `L3 96MB` `CU` `wave` `occupancy`
- https://llvm.org/docs/AMDGPUUsage.html — `sched_barrier 0x0080 DS / 0x0008 WMMA` `GMEM->VGPR->LDS->VGPR->WMMA 4-stage` `amdgpu_flat_work_group_size`
- https://github.com/ROCm/TheRock/issues/1820 — `CMake HIP_PATH` `gfx1100` `manually` `AMDGPU_TARGETS` `Debug gfx1100 DPP` `LLVM backend` `illegal DPP`
- `CK Tile lds_bank_conflicts.html` `gemm_optimization.html` `T=64 64x` `loads/out=K·(1/M+1/N) 2K/T` `32x4B 8-phase ds_write_b128`
- `amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d -R --csv` `A/B 8 VGPR fp16 / D 8 VGPR wave32 ≤64`

**Harness/Repo (project):**
- `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` `25156B` `ggml_cuda_dp4a_real sudot4 + 6x perm LUT` `vec_dot_iq4_xs_q8_1` `quantize_row_q8_1 amax/127 half2 ds`
- `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` `16797B` `sh[32][33]/[32][32] XOR` `launch_bounds(256,4)` `sudot4+perm ulong2 b128` `block_q8_1_coop 64B`
- `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` `16462B` `sB[2][32][33] vs sB_P4[4][32][32] XOR` `wmmma_f32_16x16x16_f16_w32` `v16f16/v8f32 lane%16`
- `kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json` `19K valid JSON` `M128 12.5x` `M512 0.70 1.22 peak` `M1024 1.08 1.89 peak` `M8192 SKIPPED`
- `benchmarks/results/phase7/llama_bench_stock_4tier_N10.json` `6.8K valid JSON` `512 pp 838 vs 904 1.079 FAIL` `tg 34.8 vs 34.6 0.993 FAIL` `HSA_ENABLE_DXG_DETECTION=1`
- `build_windows.bat 5857B` `HIP_PATH` `where clang++.exe --offload-arch=gfx1100 --version` `where ninja` `cmake -G Ninja gfx1100` `curl :8000 choices`

**Research Reports (project output):**
- `docs/research/deep-research/1000t-s-at-8k-gfx1100.md` `5 sessions overview/technical/prefill-long/quant-kernels/contrarian 40+ URLs 28 innerText` `8k quadratic cliff 67M->1B 16x Flash2 HBM O(N) FLOPs N² 8k->32k 3-4x vs 16x 800 GB/s pin 15.3 TB/s naive 128 KiB/tok GQA 1-2GB 15.3GB model 18.5GB/20GB` `WSL2 800GB lie 3.48GB contiguous fail BSOD 3-5 OOMs WSL#40401 #40732 15-30us jitter 1.178->1.0  rocprofv3 librocdxg unsupported 404 Instinct-only`
- `docs/research/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` `rocWMMA 2.2.1 header-only MFMA+WMMA/SWMMAC` `LDS 32 banks 4-phase P=4 GMEM->VGPR->LDS->VGPR->WMMA` `GEMM tiling 256x64->304 CUs gfx1100 64x64 128x128 T=64` `MARLIN P=4 16x64 swizzle B registers` `LUT-GEMM BCQ Σαb μ=4` `SmoothQuant W8A8 α=0.5` `adelj88 tune.py Genetic+RF --repeats 10`

## Rerun Inputs

```yaml
workflow: deep-research
engine: playwright-cli
topic: Phase 7 three must-haves fulfilment (REQ-WIN-07 ≤2 langs Windows HIP SDK gfx1100, REQ-PERF-07 ≥1.10x pp+tg 512..8192 N=10, REQ-STAT-07 N=10/15)
depth: exhaustive
output: markdown
sessions: [overview, technical, windows, market, contrarian]
queries: 10+ (overview 3, technical 5, windows 3, market 3, contrarian 5) pages: 25+ primary rocm.docs/llvm/CK + secondary WSL#40732 + project hardware JSONs
model: muse-spark-1.2
date: 2026-08-30
```

