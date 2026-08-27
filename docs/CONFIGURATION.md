<!-- generated-by: gsd-doc-writer -->

# Configuration Reference

Every knob that matters for reproducing this project's environment. All values below are verified
against the frozen Phase 1 & Phase 2 environment.

## Windows side: `.wslconfig`

`C:\Users\<user>\.wslconfig`:

```ini
[wsl2]
memory=28GB
swap=16GB
```

**Required.** With the default ~15 GB guest RAM allocation, VRAM allocation fails with
DXG ENOMEM (`dmesg`: `dxgkio_create_allocation: -12`). At 28 GB the guest sees 27 GB and
full-model GPU residency succeeds (132/132 tensor layers offloaded). Apply, then run
`wsl --shutdown` before restarting the distro.

## Guest environment

| Setting | Location | Purpose |
|---|---|---|
| `HSA_ENABLE_DXG_DETECTION=1` | `/etc/profile.d/rocdxg.sh` | Required for ROCr to enumerate the GPU through the WSL2 DXG path |
| `LD_LIBRARY_PATH` | `/root/llama.cpp/build-ci/bin` | Ensures binaries load the pinned HIP GGML shared objects |

The distro is root-only Ubuntu 24.04; everything runs as root in the guest.

## llama.cpp build flags

Pin: **v0.2.0 @ bb4caa7540188872173c44d161602d9271386413**. Source tree lives guest-side at
`/root/llama.cpp` (DrvFs has a git-lock incompatibility; do not build from `/mnt/e`).

```bash
cmake -B build -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF
```

| Flag | Why |
|---|---|
| `-DGGML_HIP=ON` | HIP backend (ROCm) instead of CUDA |
| `-DGPU_TARGETS=gfx1100` | RX 7900 XT codegen only |
| `-DLLAMA_BUILD_SERVER=OFF` | Server unused; smaller build surface |
| `-DLLAMA_CURL=OFF` | No network dependency in binaries |

Compiler: gcc 13.3.0 / hipcc 7.2.53211-e1a6bc5663. Note: `amdgpu-install` usecase `wsl` is
invalid in the 30.30.x build — use `--usecase=rocm --no-dkms`.

## Standalone Kernel Playground Build Flags (Phase 4)

The standalone playground (`kernels/`) compiles outside llama.cpp with zero external headers:

```bash
cmake -S kernels -B kernels/build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
```

| Flag | Why |
|---|---|
| `-DCMAKE_HIP_ARCHITECTURES=gfx1100` | Pins RDNA3 gfx1100 codegen (no `amdgpu-arch` auto-probe) |
| `-DCMAKE_BUILD_TYPE=Release` | Enables `-O3 -DNDEBUG` optimizations for accurate microbenchmarks |
| `find_package(hip REQUIRED)` | Discovers ROCm 7.2.1 HIP runtime and links `hip::device` |
| `scripts/check_no_ggml.sh` | CI isolation gate verifying zero `ggml`/`llama` headers included |

Python dependencies for fixtures and test runners: `gguf` (`gguf-py`), `numpy`, `pytest`.

## Custom Kernel Integration Build Flags (Phase 6)

When integrating custom kernels into `llama.cpp` using `patches/0001-gfx1100-mul-mat-custom.patch`:

```bash
# Build custom optimized binary (ON):
cmake -B build-custom -S . \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON

# Build stock baseline binary (OFF):
cmake -B build-stock -S . \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON
```

| Flag | Why |
|---|---|
| `-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` | Enables custom `gemv_iq4xs.cuh` and `gemm_iq4xs.cuh` dispatch hooks |
| `-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF` | Compiles stock HIP implementation bit-identically |

### Compiler and CMake flags

```bash
# HIP kernel codegen — gfx1100 only, Wave32 native
hipcc --offload-arch=gfx1100 -mwavefrontsize32

# CMake (kernels/CMakeLists.txt)
cmake -S kernels -B kernels/build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
```

| Flag | Where | Why |
|---|---|---|
| `--offload-arch=gfx1100` | `kernels/CMakeLists.txt: add_compile_options($<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1100>)` | Pins ISA to RX 7900 XT; also emitted per-target in `build.ninja` as `FLAGS = -O3 -DNDEBUG --offload-arch=gfx1100` |
| `-mwavefrontsize32` | hipcc invocation for `.hip` files | Forces Wave32 mode on gfx1100 (RDNA3 native 32-wide wavefront); required for WMMA Wave32 intrinsics and shuffle semantics |
| `-DCMAKE_HIP_ARCHITECTURES=gfx1100` | CMake cache (`CMakeCache.txt: CMAKE_HIP_ARCHITECTURES:STRING=gfx1100`) | Prevents `amdgpu-arch` auto-probe; single-arch build |

### Occupancy and register pressure

Both kernels declare:

```cpp
__global__ __launch_bounds__(256, 4) __attribute__((amdgpu_flat_work_group_size(256,256)))
```

| Attribute | Effect |
|---|---|
| `__launch_bounds__(256, 4)` | Hint: 256 threads per block, minimum 4 blocks resident per CU; compiler caps VGPR allocation accordingly |
| `amdgpu_flat_work_group_size(256,256)` | Forces exactly 256 threads per workgroup (8 warps Wave32); no dynamic sizing |

Combined they constrain the compiler to **<=96 VGPRs** per thread, sustaining 4 concurrent blocks per CU
(1024 threads/CU occupancy) on gfx1100. Exceeding 96 VGPR spills or drops occupancy to 3 blocks.

Sources: `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip:14`, `kernels/matmul_iq4xs/impl_gemm_wmma.hip:31,107`.

### LDS layout (GEMM WMMA path)

```cpp
// kernels/matmul_iq4xs/impl_gemm_wmma.hip:143
__shared__ _Float16 B_lds[2][32][33]; // double-buffered, padded
```

* Dimensions: `[2]` ping-pong double buffer for K-tile overlap, `[32]` rows (K tile = 32 halfs), `[33]` columns (32 + 1 pad).
* Padding: `LDS_PAD 1` adds one half (2 bytes) per row → stride 66 bytes vs 64 bytes. With 32 banks × 4 B = 128 B per row,
  unpadded 32-wide column accesses alias the same bank (32-way conflict). Padding shifts each row by one bank
  (33 halfs × 2 B = 66 B ≡ 2 banks offset), eliminating 32-way LDS bank conflicts on column reads.
* Cooperative load: 256 threads load the 16×32 B tile (512 halfs) with 2 halfs per thread, then `__syncthreads()` before WMMA consume.

### WMMA intrinsic (GEMM)

```cpp
// kernels/matmul_iq4xs/impl_gemm_wmma.hip:4,246
c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag);
```

* Intrinsic: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` — 16×16×16 matrix multiply, FP16 inputs → FP32 accumulate, Wave32, 32-cycle latency on RDNA3.
* Types: `v16f16` fragments (`typedef _Float16 v16f16 __attribute__((ext_vector_type(16)))`) per lane, 32 lanes × 16 halfs = 512 halfs (duplicated layout covers 16×16 tile); accumulator `v8f32` (8 floats per lane, 32×8=256 = 16×16 FP32 outputs).
* A fragment is dequantized on-the-fly from IQ4_XS (`kvalues_iq4nl` + `scales_l`/`scales_h` + `d`) into registers before packing; B fragment stages through `B_lds` ping buffer.
* Fallback: `gemm_iq4xs_tiled_kernel` (same launch bounds) handles arbitrary M/N/K or `M < 512` / `N < 1024` shapes with `TILE_M=16` tiled FMA — still LDS-optimized.
* Gate in host launcher (`gemm_iq4xs_wmma_gpu`): `M %16==0 && N %16==0 && K %16==0 && M >=512 && N >=1024 && K >=16` else fallback.

### Numerical parity — double accumulation

Both GEMV and tiled GEMM paths accumulate in `double` before casting to `float` for CPU parity:

```cpp
// GEMV: kernels/matmul_iq4xs/impl_gemv_gfx1100.hip
double thread_sum = 0.0; // per-thread 8-sub-block sum, reduced via shared memory
// GEMM tiled fallback: kernels/matmul_iq4xs/impl_gemm_wmma.hip:55
double acc[16] = {0.0}; // TILE_M up to 16 — double for CPU parity (cosine >=0.999, max_rel <=1e-3)
```

Thresholds enforced by `kernels/matmul_iq4xs/test_gemv_compare.cpp` and `test_gemm_compare.cpp`: cosine similarity ≥0.999, max relative error ≤1e-3 vs `ref_cpu` double reference.

### Benchmark harness (`kernels/common/bench.h`)

```cpp
// kernels/common/bench.h:27
inline BenchStats bench_hip_event(
    std::function<void(hipStream_t)> launch,
    hipStream_t stream = 0,
    int warmup = 50,
    int iters = 200,
    size_t bytes_transferred = 0);
```

Default: `warmup=50`, `iters=200` — used by `bench_gemv.cpp` (M=1) for stable decode measurements (50 warmup + 200 timed `hipEvent` pairs, median/p95/min/max/mean/stdev + GB/s).

| Bench | Warmup | Iters | File | Why |
|---|---|---|---|---|
| GEMV (M=1) | 50 | 200 | `kernels/matmul_iq4xs/bench_gemv.cpp:48,51` | Small payload (~2–46 MB); high iteration count amortizes launch overhead and yields tight p95 |
| GEMM (M>>1) | 5 | 20 | `kernels/matmul_iq4xs/bench_gemm.cpp:50,54` and `bench_matmul.cpp:51-58` | Large payloads (W + K×M + N×M, up to ~90 MB) plus WMMA; fewer iters avoids CI timeout while still computing median/p95 and TFLOP/s (`flops=2·N·M·K`) |

All benches emit JSON to stdout parsed by `benchmarks/tools/run_kernel_bench.py` and report `median_us`, `p95_us`, `gb_s`, `tflops`, and `speedup` vs stock comparator (`stock_hip_comparator.hip`).

## Quant Comparator Scope (Phase 1.3 vs BENCH-04)

Stock baseline is **IQ4_XS locked** — `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB, sha256 `53adc4bb…`, see `models/README.md` and `.planning/research/MODEL-DECISION.md`) — chosen over the 16.81 GB `Q4_K_M` class to preserve headroom on the 20 GB RX 7900 XT (see `.planning/research/MODEL-DECISION.md` § VRAM envelope). This is the **only frozen baseline** for BENCH-04 (`benchmarks/results/BASELINE-MATRIX.md` — pp/tg × {4k, 8k, 16k, 32k} × flash-attn {on, off} plus stock-Vulkan comparator; `benchmarks/golden/stock_baseline_golden.json`).

`Q4_K_S` / `Q5_K_M` (and the 16.81 GB `Q4_K_M` in the same repo) are **optional v1 comparators per REQUIREMENTS.md § v2** (`Additional quant comparators — Q4_K_M comparator optional in v1; Q6_K/Q8_0 deferred`) — not baseline. They exist only to contextualize IQ4_XS numbers when VRAM permits; they do not define the frozen environment in `benchmarks/environment/` or the `benchmarks/config/thresholds.json` calibration derived from the IQ4_XS run (`20260823_163954_calibration_profile`).

`ROADMAP-original.md` Phase 1.3 ("Use at least Qwen3.8-27B Q4_K_M; Also test Q4_K_S / Q5_K_M if VRAM permits") is **methodology, not a frozen artifact list**. Do not read any Phase 1.3 table as implying `Q4_K_S` (or any `Q4_K_*` / `Q5_K_M`) is frozen — the operative lock is IQ4_XS above, and CONFIG must not imply otherwise.

## Benchmark & Guard Configuration

The benchmark harness relies on empirical thresholds derived from calibration (`benchmarks/config/thresholds.json`):

```json
{
  "vmrss_fail_kb": 22788858,
  "vmswap_fail_kb": 524288,
  "gpu_shared_climb_mb_per_min": 250.0,
  "repeat_deviation_max_ratio": 2.0,
  "derived_from": "20260823_163954_calibration_profile",
  "measured_peak_vmrss_kb": 15192572,
  "measured_peak_vmswap_kb": 0
}
```

| Parameter | Value | Purpose |
|---|---|---|
| `vmrss_fail_kb` | `22,788,858 kB` (21.7 GB) | 1.5x steady-state margin to catch host RAM spill |
| `vmswap_fail_kb` | `524,288 kB` (512 MB) | Detects swap growth before performance collapse |
| `gpu_shared_climb_mb_per_min` | `250.0 MB/min` | Flags sustained host shared GPU memory leaks |
| `repeat_deviation_max_ratio` | `2.0x` | Flags unstable intra-cell repeat throughputs for review |
| Thermal Abort | `95.0 °C` | Host watchdog kills process on junction temp overshoot |

## Runtime flags

Headless interactive runs need process isolation flags; omitting them hangs waiting on TTY input:

```bash
setsid llama-cli -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p 'Hello' -n 32 --temp 0 \
  --load-mode none --simple-io --single-turn -t <threads>
```

| Flag | Purpose |
|---|---|
| `-ngl 99` | Offload all layers to GPU (target: fully resident) |
| `-c N` | Context size — **set explicitly, always**; OOM arrives at first long prompt, not load |
| `--load-mode none` | Avoid mmap stalls; canonical model copy is guest-side `/root/models/` |
| `--simple-io` | Required for headless/non-TTY execution |
| `--single-turn` | One prompt, one completion — no interactive loop |
| `-t <threads>` | CPU thread count (note: ROCr busy-spin under WSL2 consumes ~1–2 cores per GPU context) |

## Frozen versions (Phase 1 & 2)

Do not upgrade any component silently — see D-04 below.

| Component | Version |
|---|---|
| ROCm (guest) | 7.2.1 |
| Windows driver | 32.0.31041.1004 (Adrenalin 26.10.41) |
| librocdxg | 1.2.2 |
| llama.cpp | v0.2.0 @ bb4caa7540188872173c44d161602d9271386413 |
| hipcc | 7.2.53211-e1a6bc5663 |
| OS | Ubuntu 24.04 (guest, root-only), WSL2 host |

Full fingerprint files: `benchmarks/environment/` (versions.txt, hipconfig.txt, rocminfo.txt, llamacpp-pin.txt).

## D-04 update policy

No silent driver updates. Scope (as amended): prevent *silent* updates so the ROCm/driver
pairing stays frozen; notification-only behavior is acceptable. Sanctioned mechanism (**PENDING — requires an elevated shell, owner action**):

```powershell
reg add HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate /v ExcludeWUDriversInQualityUpdate /t REG_DWORD /d 1 /f
```

Detection net if drift occurs anyway: every benchmark row carries a driver fingerprint, and the
environment version gates are re-run on any detected mismatch.
