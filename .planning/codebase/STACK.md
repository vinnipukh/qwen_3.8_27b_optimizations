<!-- refreshed: 2026-08-25 -->
# Technology Stack

**Analysis Date:** 2026-08-25 (Updated Phase 6 / v1.0.0-gfx1100)

## Hardware Target

| Component | Specification | Details |
|-----------|--------------|---------|
| GPU | AMD Radeon RX 7900 XT | RDNA3 architecture (`gfx1100`), Navi 31, 84 CUs, 5376 Stream Processors |
| VRAM | 20 GiB GDDR6 | 320-bit bus width, ~800 GB/s peak bandwidth, 18.25 GiB free DXG anchor |
| Wavefront Size | Wave32 native | 32 threads per wave, dual-issue SIMD32 units, hardware WMMA matrix cores |
| Host CPU | AMD Ryzen 7 5700X | 8 cores / 16 threads |
| Host Memory | 32 GB DDR4 | `.wslconfig` configured with `memory=28GB` and `swap=16GB` |

## Software & Toolchain Stack

| Layer | Component | Version | Notes |
|-------|-----------|---------|-------|
| Host OS | Windows 11 Pro | Build 26100 | Runs HWiNFO64 Shared Memory v2 and thermal watchdog |
| Host Driver | AMD Adrenalin (WSL2) | 26.2.2 (32.0.31041.1004) | Frozen baseline — no unvetted updates |
| Guest OS | WSL2 Ubuntu 24.04 LTS | Kernel 5.15+ | Root-only execution environment |
| Passthrough | `librocdxg` | 1.2.2 | `/dev/dxg` compute adapter passthrough |
| GPU Runtime | AMD ROCm / HIP | 7.2.1 | `HSA_ENABLE_DXG_DETECTION=1` in `/etc/profile.d/rocdxg.sh` |
| HIP Compiler | `hipcc` (AMD Clang) | 7.2.53211-e1a6bc5663 | Clang 22.0.0git backend |
| Build System | CMake & Ninja | CMake 3.28+, Ninja 1.11+ | Dual builds: standalone `kernels/` and `llama.cpp` |
| Core Framework | `llama.cpp` | v0.2.0 (`bb4caa75`) | Pinned upstream commit |
| Scripting & Harness | Python | 3.12.3 | Standard library + NumPy + pytest |
| Test Runner | Pytest | 7.4.4 | 55 unit and integration tests |
| Model Quantization | GGUF IQ4_XS | 4.25 bpw | `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB) |

## Compiler Flags & Directives

### Standalone Kernel Playground (`kernels/`)
- Target: `-DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release`
- Directives: `__launch_bounds__(256, 4)`, `__attribute__((amdgpu_flat_work_group_size(256, 256)))`
- Target VGPR Limit: $\le 96$ VGPRs per thread (maintaining 16 waves/SIMD occupancy)
- Intrinsics: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (16x16x16 FP16$\to$FP32 Wave32)

### In-Tree llama.cpp Builds
- Baseline: `cmake -B build-stock -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF -DLLAMA_BUILD_SERVER=ON`
- Custom: `cmake -B build-custom -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DLLAMA_BUILD_SERVER=ON`
- Runtime flags: `--single-turn --simple-io --load-mode none -ngl 99`
