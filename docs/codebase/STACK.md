<!-- refreshed: 2026-08-30 -->
# Technology Stack

**Analysis Date:** 2026-08-30

## Hardware Target

| Component | Specification | Details |
|-----------|--------------|---------|
| GPU | AMD Radeon RX 7900 XT | RDNA3 architecture (`gfx1100`), Navi 31, 20 GiB GDDR6, Wave32 native + HW WMMA matrix cores |
| VRAM | 20 GiB GDDR6 | 18.25 GiB free-VRAM DXG anchor (`benchmarks/lib/preflight.py`) |
| Host CPU | AMD Ryzen 7 5700X | 8 cores / 16 threads |
| Host Memory | 32 GB DDR4 | WSL2 `.wslconfig` `memory=28GB` is mandatory (15 GB guest RAM caused DXG ENOMEM) |
| Secondary target | Any `gfx1100`/`gfx1101`/`gfx1102`/`gfx1150`/`gfx1151` | DP4A fallback path in `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh` |

## Languages

**Primary:**
- **C++17** (plus HIP device extensions) — all custom GPU kernels and benchmark/compare executables in `kernels/`; llama.cpp C++ runtime (vendored `llama.cpp/`). Compiler flags: `-std=c++17`, `--offload-arch=gfx1100` (`kernels/CMakeLists.txt`)
- **HIP .hip / .cuh** — device code: `kernels/matmul_iq4xs/*.hip`, `kernels/demo_iq4xs_dequant/*.hip`, `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/*.cuh`
- **.cu** (CUDA-named but HIP-compiled) — llama.cpp ggml-cuda backend: `llama.cpp/ggml/src/ggml-cuda/mmq.cu`, `mmvq.cu` (dispatch hooks)

**Secondary:**
- **Python 3.12** (WSL2 guest) / 3.14 (this analysis host) — benchmark harness (`benchmarks/bin/*.py`, `benchmarks/lib/*.py`), fixture extraction (`tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py`), kernel swizzle generator (`tools/swizzle_iq4xs.py`). 40 Python files outside `llama.cpp/` — exceeds the ≤2-lang gate, deferral documented in `README.md` and `docs/PUBLICATION.md`
- **Windows batch (.bat)** — `build_windows.bat` (REQ-WIN-07 gate, pure C++/HIP + CMake + .bat)
- **Bash** — `scripts/check_no_ggml.sh` (KERN-01 isolation gate)
- **JavaScript (Node .mjs)** — `docs/research/freetoken-probe/src/qstar.mjs` (q* policy projector, offline)
- **Zig 0.16** — bundled portable compiler in `docs/research/freetoken-probe/tools/zig-x86_64-windows-0.16.0/` (compiler only, nothing installed system-wide)

## Runtime

**Environment:**
- **Primary (bench/experiment):** WSL2 Ubuntu 24.04 guest + ROCm 7.2.1 + `librocdxg` 1.2.2 (`/dev/dxg` DXG passthrough to Windows driver). Mandatory env: `HSA_ENABLE_DXG_DETECTION=1` (via `/etc/profile.d/rocdxg.sh`)
- **Target (07-01 closure):** Windows-native HIP SDK 6.4 (`C:\Program Files\AMD\ROCm\6.4`) — `build_windows.bat` detects via `HIP_PATH`; **not yet installed/executed on this host (pending human gate)**
- Host OS: Windows 11 (build 10.0.26200.9168), WSL2 2.7.12, driver 32.0.31041.1004 (Adrenalin 26.10.41, frozen)

**Compilers:**
- `hipcc` 7.2.53211-e1a6bc5663 (AMD clang 22.0.0git, `/opt/rocm-7.2.1/lib/llvm/bin/clang++`) in guest
- `gcc` 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) for host-side C++ in guest (`benchmarks/environment/llamacpp-pin.txt`)
- Windows target: `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100` with `-G Ninja` (NOT `cl` — `.hip` files use `__builtin_amdgcn_*` intrinsics cl cannot compile)

**Package Manager:**
- CMake 3.21+ (kernels), 3.14+ (llama.cpp) + Ninja generator; no package-manager lockfiles, no conan/vcpkg
- Python deps via pip in guest (numpy, pytest, gguf-py); no `requirements.txt` at repo root outside `llama.cpp/`

## Frameworks

**Core:**
- **llama.cpp v0.2.0 pinned at `bb4caa7540188872173c44d161602d9271386413`** (vendored git tree `llama.cpp/`; local HEAD `5c6b397` = pin + 1 custom commit "feat(gfx1100): hybrid DP4A GEMV + WMMA GEMM dispatch") — inference runtime, GGUF model format, `llama-bench`/`llama-cli`/`llama-server`/`llama-perplexity`/`test-backend-ops` executables
- **ggml HIP backend** (`llama.cpp/ggml/src/ggml-hip/`) — CUDA backend compiled for HIP; custom-kernel hook via `GGML_CUDA_ENABLE_CUSTOM_GFX1100` compile definition

**Testing / Validation:**
- Standalone CPU oracle + HIP comparators in `kernels/`: `ref_cpu.cpp`, `stock_hip_comparator.hip` (naive scalar), `real_stock_dp4a_comparator.hip` (true upstream DP4A pipeline via `ggml_cuda_dp4a`/`sudot4` + perm LUT)
- pytest (harness unit tests, `benchmarks/tests/`, 55 tests) — guest

**Build/Dev:**
- CMake + Ninja. See `kernels/CMakeLists.txt`, `llama.cpp/ggml/CMakeLists.txt`, `build_windows.bat`
- llvm/ROCm toolchain artifacts at repo root: `impl_gemv_dp4a_gfx1100-*.hipi/.bc/.out` (compiler intermediates, gitignored)

## Key Dependencies

**Critical:**
- **HIP runtime** (`hip::device` — `kernels/common/CMakeLists.txt`; `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")` in `kernels/CMakeLists.txt`) — `hipMalloc`, `hipLaunchKernelGGL`, `hipStreamIsCapturing` used in `custom_gfx1100/gemv_iq4xs.cuh`
- **`gguf-py`** (vendored `llama.cpp/gguf-py/`) — real-tensor fixture extraction in `tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py`
- **numpy** — fixture `.npz` generation, dequant math, iq4xs swizzle (`tools/swizzle_iq4xs.py`)
- **pytest** — benchmark harness tests (`benchmarks/tests/`)

**Infrastructure:**
- `librocdxg` 1.2.2 (DXG compute passthrough, guest)
- HWiNFO64 Shared Memory v2 (`Global\HWiNFO_SENS_SM2`) — telemetry bridge, Windows host
- rocprofv3 (`HSA_TOOLS_LIB` / `.rocprofv3/` output dir at repo root) — kernel profiling
- OpenCL.dll (runtime-loaded, no SDK) — `docs/research/freetoken-probe/src/bench_bp.cpp` PCIe bandwidth probe on Windows

## Configuration

**Environment:**
- `HSA_ENABLE_DXG_DETECTION=1` — mandatory in guest for all ROCm launches (`tools/ask_model.py`, runbook)
- `HIP_PATH` — Windows HIP SDK location (default `C:\Program Files\AMD\ROCm\6.4`), used by `build_windows.bat` and CMake `find_package(hip ... PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`
- `MODEL_PATH` — overridable GGUF path in `build_windows.bat` (default `models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf`)
- `.wslconfig` — `[wsl2] memory=28GB` required (frozen via fingerprint `benchmarks/lib/fingerprint.py` hashes `.wslconfig`)
- `LD_LIBRARY_PATH` — set to llama.cpp build `bin/` dir in `tools/ask_model.py`
- Benchmark path constants: `/root/llama.cpp/build-ci/bin/llama-bench` (stock pin) and `/root/llama-custom-07` (custom `5c6b397-dirty`), per `benchmarks/lib/llabench.py` + `README.md`

**Build (kernels playground):**
- `-DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100 -DAMDGPU_TARGETS=gfx1100` (no auto-detect), `-DCMAKE_BUILD_TYPE=Release` (`kernels/CMakeLists.txt`); `CMAKE_EXPORT_COMPILE_COMMANDS=ON`

**Build (llama.cpp, per `benchmarks/environment/llamacpp-pin.txt` + `build_windows.bat`):**
- Stock: `-G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF`
- Custom: `-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` (default **OFF** — bit-identical to upstream when OFF)
- Windows: `cmake -S . -B build-windows -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER/HIP_COMPILER="%HIP_PATH%\bin\clang++.exe" -DHIP_PATH=...`

## Kernel Device-Code Stack (gfx1100 specifics)

- **Intrinsics:** `__builtin_amdgcn_sudot4` (v_dot4_i32_i8 DP4A), `__builtin_amdgcn_perm` (v_perm_b32 LUT gather), `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (Wave32 WMMA), `__shfl_xor` warp reduce, `_Float16` vector types
- **Launch bounds:** `__launch_bounds__(256,4)` + `__attribute__((amdgpu_flat_work_group_size(256,256)))` in both vendored CUHs and source `.hip` files
- **Memory:** LDS double-buffer `sB[2][32][33]` (stride-33 padding kills bank conflicts), `sh_coop[32][33]` GEMV reduction, `ulong2` 16-byte loads of `block_iq4_xs` quants
- **Dispatch gates:** GEMV `custom_gemv_iq4xs_can_handle` → `M==1 && K∈{5120,17408} && N∈{5120,6144,17408} && K%256==0`; GEMM `custom_gemm_iq4xs_can_handle` → `M>=16 && K%256==0 && N%16==0` (`custom_gfx1100/*.cuh`)

## Platform Requirements

**Development:**
- Windows 11 + WSL2 (build/bench in guest ext4 `/root`, never `/mnt/*` — DrvFs breaks git locks)
- ROCm 7.2.1 guest + librocdxg 1.2.2; OR Windows HIP SDK 6.4 + VS Build Tools + Ninja (`winget install Ninja-build.Ninja`)
- ccache recommended (guest, per `docs/GETTING-STARTED.md`); frozen env snapshot `ubuntu-2404-rocm721-phase1.tar` (49.4 GB)

**Production (target deploy):**
- Windows-native `build-windows/bin/llama-server.exe` serving OpenAI-compatible API at `http://127.0.0.1:8000/v1/chat/completions` (REQ-WIN-07 smoke; `curl` HTTP 200 with `choices[0].message.content`)

---

*Stack analysis: 2026-08-30*