# GPU Kernel Development & Optimization Resources: AMD RDNA3 (gfx1100)

**Target Hardware:** AMD Radeon RX 7900 XT (`gfx1100`, RDNA3, 20 GiB VRAM)  
**Target Workload:** `MUL_MAT` (IQ4_XS Quantized Matrix Multiplication / Fused Dequant+GEMV/GEMM)  
**Stack:** ROCm 7.2.1 / HIP SDK / llama.cpp GGML HIP backend  

---

## 1. AMD Official Architecture & ISA References

| Resource | URL | Description & Role in Phase 4/5 |
|---|---|---|
| **AMD RDNA3 Shader ISA Reference** | [docs.amd.com RDNA3 ISA](https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0) | Complete opcode and machine instruction reference for `gfx1100` (dual-issue ALU, WMMA, VOPD, VGPR allocation). |
| **ROCm Documentation Hub** | [rocm.docs.amd.com](https://rocm.docs.amd.com/en/latest/) | Master ROCm reference manual covering libraries, compilers, and tools. |
| **ROCm on Radeon (Windows / WSL2)** | [ROCm Radeon Guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/howto_windows.html) | Guide for running and configuring ROCm on consumer Radeon GPUs under WSL2/Windows. |
| **ROCm GitHub Hub** | [github.com/ROCm](https://github.com/ROCm) | Source repositories for ROCm toolchains and drivers. |

---

## 2. HIP Programming, Performance & Occupancy Guides

| Resource | URL | Description & Role in Phase 4/5 |
|---|---|---|
| **HIP Programming Guide** | [rocm.docs.amd.com HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) | Core HIP API reference, kernel launching, memory management, and stream synchronization. |
| **HIP Performance Guidelines** | [HIP Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) | Best practices for loop unrolling, memory coalescing, vector types (`float4`, `half2`), and branch divergence. |
| **HIP Optimization & VGPRs** | [HIP Performance Optimization](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html) | Deep dive into Vector General Purpose Register (VGPR) pressure, `__launch_bounds__`, and occupancy ceilings. |
| **ROCm Official Examples** | [github.com/ROCm/ROCm-Examples](https://github.com/ROCm/ROCm-Examples) | Clean, standalone HIP sample kernels, reduction loops, and matrix math templates. |
| **ROCm HIPIFY Tool** | [github.com/ROCm/HIPIFY](https://github.com/ROCm/HIPIFY) | CUDA-to-HIP source conversion engine for translating CUDA kernel patterns to portable HIP code. |

---

## 3. Composable Kernel (CK) & High-Performance Tensor Libraries

| Resource | URL | Description & Role in Phase 4/5 |
|---|---|---|
| **Composable Kernel Documentation** | [Composable Kernel Docs](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/) | AMD's C++ templated kernel library for fused GEMM, blockwise reductions, and attention. |
| **Composable Kernel Repository** | [github.com/ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) | Production RDNA3 blockwise GEMM and real GEMM kernel implementations. |

---

## 4. GPUOpen Architecture Tuning & Live VGPR Analysis

| Resource | URL | Description & Role in Phase 4/5 |
|---|---|---|
| **GPUOpen: Large Thread Groups** | [GPUOpen Occupancy Optimization](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/) | Optimizing thread group sizing (wave32 vs wave64) and LDS memory bank layout on RDNA3. |
| **GPUOpen: Occupancy Explained** | [GPUOpen Occupancy Guide](https://gpuopen.com/learn/occupancy-explained/) | Architectural mechanics of SIMD execution units and wave-switching latency hiding. |
| **GPUOpen: Live VGPR Analysis (RGA)** | [GPUOpen Live VGPR Analysis](https://gpuopen.com/learn/live-vgpr-analysis-radeon-gpu-analyzer/) | Analyzing register lifetimes and preventing register spilling into high-latency scratch memory. |

---

## 5. RDNA3 Matrix Hardware & Quantized Kernel References

| Resource | URL | Description & Role in Phase 4/5 |
|---|---|---|
| **RDNA3 WMMA Operations** | [github.com/glovepost/wmma_ops](https://github.com/glovepost/wmma_ops) | Working wave matrix multiply accumulate (`v_wmma_f32_16x16x16_f16`, `v_wmma_i32_16x16x16_iu4`) intrinsics for `gfx1100`. |
| **ggml-cuda Reference Kernels** | [llama.cpp ggml-cuda Source](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda) | Reference stock implementation of `mmvq.cu` (quantized GEMV), `mmq.cu` (quantized GEMM), and `vecdotq.cuh` (IQ4_XS dequant dot products). |
| **IST-DASLab Marlin GEMM** | [github.com/IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) | State-of-the-art 4-bit FP16/INT4 fused GEMM matrix-vector kernel architecture. |

---

## 6. How These Map Directly to Phase 4 & Phase 5

1. **Phase 4 (Kernel Playground Scaffold):**
   - Use `ROCm-Examples` and `HIP Programming Guide` to build standalone `ref_cpu.cpp` $\to$ `impl_gfx1100.hip` quartet compiler harness without llama.cpp dependencies.
   - Use `ggml-cuda/vecdotq.cuh` and `ggml.h` to extract the exact `block_iq4_xs` bit-layout fixtures.
2. **Phase 5 (Custom `MUL_MAT` Kernel Attack):**
   - Apply `HIP Performance Optimization` and `GPUOpen Live VGPR Analysis` to tune `__launch_bounds__` for wave32 on `gfx1100` to eliminate VGPR register spilling.
   - Evaluate `wmma_ops` and `Composable Kernel` blockwise GEMM templates for fused `IQ4_XS` dequantization + dot product accumulation.
