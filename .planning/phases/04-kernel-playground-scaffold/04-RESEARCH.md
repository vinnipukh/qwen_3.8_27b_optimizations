# Phase 4: Kernel Playground Scaffold - Research

**Date:** 2026-08-25
**Scope:** KERN-01 standalone HIP playground outside llama.cpp (CPU reference → HIP → compare → microbench)
**Method:** 6 parallel subagents against all 15 resources in `@.planning/reference/GPU-KERNEL-RESOURCES.md` + re-verified `@.planning/research/EXTERNAL-RESOURCES-ASSESSMENT.md` + 8 direct fetches (performance_guidelines, occupancy-explained, vecdotq.cuh 49KB, composable_kernel, live-vgpr, etc.) + prior 50/200 bench + IQ4_XS deep dives

---

## 1. AMD Official Architecture & ISA (amd-isa)

**Canonical ISA:** `RDNA3 Shader ISA Reference Guide` PDF `rdna3-shader-instruction-set-architecture-feb-2023_0.pdf` (doc 70650). All §2.1 Wave32/Wave64, §3.3 Storage State (VGPR/SGPR/LDS), §7.4 VALU/VOPD, §7.9 WMMA cite this doc. Do not confuse with RDNA3.5/CDNA. Hub `rocm.docs.amd.com/reference/gpu-arch/index.html` links it; also at `www.amd.com/system/files/TechDocs/rdna3...pdf` and KHUB mirror.

* VOPD dual-issue is 8B `VOPD_X :: VOPD_Y` (two 2-src/1-dst VALU ops), peak 256 FLOPs/CU/cycle FP32 (512 packed FP16) when both slots fill. X/Y cannot share same VGPR bank in same operand position (3-read-port bank cache) and dests cannot be both even/odd; 3-src ops (`v_fma`) excluded. ChipsAndCheese microbench: LLVM rarely emits VOPD for FMA — only `v_dual_add_f32` reliably dual-issues. **Playground probe:** `hipcc -mwavefrontsize32` + `llvm-objdump --mcpu=gfx1100` to verify VOPD emission.
* WMMA on gfx1100 is **only** `16×16×16` `AMDGPU.Device.WMMA_RDNA3`: `f16*f16+f32->f32`, `bf16*bf16+f32->f32`, `i8*i8+i32->i32` (bf8/f8 gfx12 only). Lane-replicated `L%16` rows (lanes 0–15 == 16–31, 32-cycle). `rocWMMA` / `FlyDSL rdna3_f16_gemm.py` demonstrates double-buffered LDS ping-pong (128×128×32 swizzled, 4 warps).
* VGPR/SGPR/LDS: per-WGP 128 KB LDS (2×64 KB blocks, 32 banks×4B), arch 256 VGPRs, physical ~50% larger than GFX10 on `gfx11-full-vgprs` (gfx1100), allocation granularity +50% (LLVM `D134522`). Use `llvm-calc-occupancy --amdgpu-arch=gfx1100 --vgpr-count N --sgpr-count M --lds-size K` (`GCNSubtarget::getOccupancyWithNumVGPRs`). `HasNoDataDepHazard` removes SW `s_waitcnt` on gfx11.
* Wave32 vs Wave64 is per-kernel compile-time (`__attribute__((amdgpu_wavefront_size(32)))` / HLSL `WaveSize` / `VK_EXT_subgroup_size_control` / `-mwavefrontsize32`); Wave32 favours divergent/long-latency+VOPD, Wave64 favours ALU-heavy coalesced.
* ROCm hub `rocm.docs.amd.com` is canonical; Radeon WSL matrices list `gfx1100` (RX 7900 XTX/XT/GRE, PRO W7900) since ROCDXG 1.2.0 / ROCm 7.2.x, Ubuntu 22.04/24.04/26.04 + Windows 11, `hipcc --offload-arch=gfx1100` validated.
* WSL2 librocdxg shim over `/dev/dxg` + `libdxcore.so`: install Windows driver → ROCm apt → `librocdxg` (cmake with `WIN_SDK` or prebuilt `rocdxg-roct_*.deb`), `rocminfo` agent `gfx1100`, container `--device /dev/dxg -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so ...`, `HSA_ENABLE_DXG_DETECTION=1` <7.13 only.
* WSL2 constraints to design around: `hipMemGetInfo` ~3 GB under-report (`librocdxg#57`/`llama.cpp#23999`), Strix Halo UMA maps only `.wslconfig memory=` (`ROCm#6022`), `AsyncEventsLoop` 2-core spin (`#60`), no `rocprofiler/roctracer/rocgdb`, pinned/UVA `RuntimeError: UVA is not available` unless patched (`vLLM#41496`). Playground: headroom, `.wslconfig memory=32GB`, `hipMalloc`+`hipMemcpy` over UVA, gate profiler tests.

## 2. HIP Programming, Performance & Occupancy Guides (hip-guides)

* Standalone HIP build: `hipcc` driver over `amdclang++` (`-x hip`, `__HIP_PLATFORM_AMD__`), `hipconfig --cpp_config` for plain GCC, CMake 3.21+ `enable_language(HIP)` + `CMAKE_HIP_ARCHITECTURES=gfx1100` → `clang-offload-bundler` fatbins via `__hip_fatbin`, `hip::device` forces `-x hip` (guard `$<$<COMPILE_LANGUAGE:HIP>:>` for mixed targets, bug #2158), `make HIP_ARCHITECTURES=gfx1100`. WGP vs CU default mismatch `hiprtc` CU / `hipcc` WGP = 3× slowdown (#3374).
* Memory coalescing: 128B aligned bursts, `lane i → base+i` (`data[threadIdx.x+blockIdx.x*blockDim.x]`), stride>1 collapses bandwidth. Use `float4`/`float2`+`__align__(16)`, pad 2D widths to `warpSize`, batch `hipMemcpy`, `hipHostMalloc` pinned + `hipHostMallocMapped` APUs. `rocprofv3 --stats`.
* LDS bank conflicts: 32 banks×4B, `bank=(addr>>2)%32`, `data[tid*32]` stride 128B → all bank 0. Fix `data[32][33]` pad; broadcast free; `extern __shared__` dynamic size = 3rd launch arg.
* Divergence: EXEC mask serializes, `if(threadIdx.x<32)` cheap vs `if(data[tid]>t)` data-dependent, `[[likely]]`/`__builtin_expect`, predication/split, `lane_mask_t uint32_t` wave32 vs `uint64_t` wave64, `SQ_BRANCH` in rocprofv3.
* `__launch_bounds__(MAX_TPB, MIN_WARPS_PER_EU)` caps VGPR (e.g. `(256,4)` → `maxRegs=available/MIN_WARPS`), LDS not auto-reduced → fail if unsatisfiable; historic `amdgpu-waves-per-eu` bug (#2521), verify `--resource-usage`.
* Occupancy: `active/max per CU`, RDNA3 `warpSize=32`, 16 slots/SIMD, WGP=2CU×2SIMD=4SIMD, 7900 XTX 192 SIMD → 3072 waves to fill, `128/256` ideal blocks; Little's Law zero-overhead wave switch hides ~100s-cycle HBM latency; high occupancy can thrash caches — MFMA GEMMs often optimal at low occupancy. Profile `rocprofv3 --occupancy` / RGP Pipeline / PIX `WaveOccupancyLimiters` (VGPR/LDS/TGSize/Barriers 16 per SIMD-pair).
* VGPR pressure: 1536 VGPRs/SIMD, 120 VGPR wave32 → 12 waves (75%), critical limit 192 on gfx1100 defers spill → global spill 100× slower; chain `combine(compute_a(),compute_b())`, move `float tmp[100]` → `__shared__[blockDim.x][100]`, RGA Live VGPR + `--resource-usage`, `__launch_bounds__` cap.
* HIPIFY warnings: `hipify-clang` vs `hipify-perl`, `1<<lane` UB → `1ULL<<lane` per waveSize, `__CUDA_ARCH__`→`__HIP_ARCH_HAS_*`+`hipDeviceGetAttribute(WarpSize)`, `cuCtxCreate`→`HIP_UNSUPPORTED` (#2062), no dynamic parallelism.
* `ROCm/rocm-examples/HIP-Basic/llvm_ir_to_executable` canonical `HIP_ARCHITECTURES=gfx1100` pattern; hardware features table (wavefront 32, 1024 threads/block, 32 waves/CU, 256 VGPR/thread, matrix cores no vs CDNA).

## 3. Composable Kernel (ck-lib)

* 4 layers (Tile Operators → Kernel/Invoker → Instantiated → Client API); modern entry is CK Tile `UniversalGemmKernel<TilePartitioner,GemmPipeline,EpiloguePipeline>` (`RunGemm` static device entry). Legacy `ck::DeviceGemm_Xdl` still documented.
* Templated GEMM = 4 groups (dtype, layout Row/Col, element ops `AElementOp/BElementOp/CDEElementOp`, tunables `BlockSize/M/N/KPerBlock/M/N PerXDL/AK1/BK1`). Instantiation `MakeInvoker/MakeArgument/invoker.Run()` in `example/ck_tile/03_gemm/universal_gemm.cpp`, `19_gemm_multi_d` (multi-D fusion `E=f(A×B,D0…Dn)`), `01_fmha` fused attention; docs `Composable-Kernel-structure.html`, `optimizing-with-composable-kernel.html`.
* Blockwise reductions `block_reduce.hpp` (not cross-warp) + `Static Distributed Tensor` auto-distributed across waves; GEMM as BlockTile→Warp Tile→MFMA with `global→LDS→registers` pipeline — primitive to mimic for reductions/norms.
* CK Tile decoupled (`ck_tile/core.hpp` single header, namespace `ck_tile`): self-contained vs old CK, but full repo still Docker+`GPU_TARGETS`+`make -j` (~2 GB/thread), gated on `GPU_TARGETS`. Standalone HIP playground compiles with just `hipcc`, no CK.
* **Vendor-copy verdict: reference-only** (MIT permits but heavy, fast churn, `xdl` Instinct-only gfx90a/gfx942). Precedent `ROCm/aiter csrc/include/ck_tile_shim.h` vendors only shim when `ENABLE_CK==0`. Phase 4: cite tiling/policy, hand-roll minimal `universal_gemm`-like `__shared__`+`mfma`, prove `-ngl 0` step-up.
* Borrow: tile-partitioner/pipeline/epilogue decomposition, BlockSize/MPerBlock tunable table, 128-bit `AK1/BK1`, LDS `CShuffle` staging, epilogue fusion pattern.

## 4. GPUOpen Architecture Tuning & Live VGPR Analysis (gpuopen — 8 findings, see `gpuopen.md`)

* Wave32 native 32-wide 1-cycle, Wave64 emulated 2×32 higher VGPR/SGPR; control via HLSL `WaveSize` / Vulkan `subgroup_size_control`; template `warpSize` 32 for RDNA3 / 64 for GCN fallback; two PSOs.
* Occupancy math 16 slots/SIMD, 1536 VGPRs/SIMD: 120 VGPR wave32 → 12 waves; VGPR granularity rounds up (RGP pipeline tab reports `VGPRs to save for +1 wave`). `warpSize==64` ≈½ wave count.
* Large groups 256 vs 1024: 2×1024 needs ≤32 VGPRs/thread (65536÷2048) and ≤32 KiB LDS/group; AMD default 256; 1024 only for LDS island multi-pass (border 56%→13%). Expose `GROUP_SIZE=warpSize*N` keep `%64==0`.
* LDS 128 KiB/WGP 32 banks×4B (2×64 KB blocks), coalesced 256B/wave, pad `[32][33]`, SoA over AoS.
* Occupancy ≠ perf: memory-bound benefits, ALU-bound 100% VALU does not; high occupancy thrashes L1/L2; watch PIX `WaveOccupancyLimiters` (VGPR/LDS/TGSize/Barriers).
* RGA Live VGPR ≥2.7 for `gfx1100`: `rga -s vulkan --vert ... --isa isa.txt --livereg livereg.txt`, `: ^ v x` + `Maximum # VGPR used/allocated Y/256`, GUI pressure column; CI-gate allocated >96 VGPRs (≈16→12 threshold).
* VGPR reduction: `ByteAddressBuffer` SGPR scalar loads (typed→VGPR), `SV_GroupID` SGPRs, bools→SGPR lane mask, 32 bools→1 VGPR `countbits`, 16b+16b pack `f16tof32`, `min16float` 2-per-VGPR double-rate, hoist into `[loop]`, spill transient to LDS between barriers.

## 5. RDNA3 Matrix Hardware & Quantized Kernels (quant-kernels — 8 findings, see `quant-kernels.md`)

* wmma_ops → RDNA3 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, 32-cycle, 1024 FLOP/WGP/cycle, `ck/utility/amd_wmma.hpp` lists `__gfx1100__/__gfx1151__`, lane-replicated `L%16` fragments (GPUOpen prose "A holds column per lane" flagged wrong).
* ggml-cuda IQ4_XS path `vecdotq.cuh:get_int_from_table_16` (HIP `__builtin_amdgcn_perm` vs CUDA `__byte_perm`) → `mmvq.cu:get_vec_dot_q_cuda(IQ4_XS)=vec_dot_iq4_xs_q8_1` + `mmq.cuh/mma.cuh` with `VDR_IQ4_XS_Q8_1_MMVQ/MMQ` tiling `VDR*32`, `generate_cu_files.py:TYPES_MMQ` proves both MMVQ/MMQ instantiations; `hipEngine BENCHMARK.md` 50/200 evidence policy primary for `bench.h`.
* IQ4_XS 136B/256 fixed `d(2)+scales_h(2)+scales_l(4)+qs(128)` 4.25 bpw, 8×32 sub-blocks, `kvalues_iq4nl[16]=[-127..113]`; PR #5747, `Tensor-Encoding-Schemes` wiki, `oxillama iq4_xs.rs`.
* Dequant `w = d * (ls-32) * q`, `ls=(scales_h>>2*ib&3)<<4|(scales_l[ib/2]>>4*(ib%2)&0xF)`, `qs` low `2i` high `2i+1` split-half; `ggml-sycl/dequantize.hpp` + `ggml-quants.c` + AVX2 decoder 3-way verified.
* Matmul decomposition `global→shared` (padded/XOR-swizzled LDS) → registers → `dp4a`/`mma` → `wmma_f32_16x16x16_f16` (not CDNA `mfma`, `Can co-execute with VALU: False`).
* Marlin fused INT4 `M_s×N_sm` tiles (`N_sm` 64/128/256), 128-bit widest loads (32 INT4/thread → 1024/instr), 3-stage `cp.async`+register double-buffer, scale fused at Tensor-Core fetch; Phase 5 HIP equivalent `ds_read_b128`+`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`+`s_waitcnt`/`V_NOP` per ISA §7.9.1.
* Demo dequant-only confirmed: 136B determinism + `dequantize_row_iq4_xs` oracle + wmma_ops 41.3 TFLOPS validation gate — synthetic super-blocks suffice, no GGUF fixture required for isolation; `test_fragment_loading.py` pattern proves helpers testable alone.

## 6. External Resources Assessment Re-verified (external-assessment — 4 findings)

* **magpie-kernel-evaluator — Conditional yes** (hardware-agnostic HIP/PyTorch/Triton via `local/container/ray`, but matrix lists MI300X/MI325X/MI355X+ROCm 7.0–7.2 only → unverified on gfx1100 → mine discipline, don't depend). Discipline to copy: correctness-before-perf, `correctness.atol 1e-6 rtol 1e-5 equal_nan false`, identical `warmup/iterations/profiler`+`HIP_VISIBLE_DEVICES`, `compare --baseline 0` (`kernels[0]`), `testcase_command` required, `analyze_report.json` + `rocprof-compute` weighted `perf_weights` (MFMA_FLOPs 0.35), preflight `magpie --gpu-info`.
* **rocm-doctor — No, WSL2 hard gate** — SKILL verbatim *"If … WSL2 — then stop and decline"* + `rocm examine` `status: wsl/out_of_scope` on `/dev/dxg`. Do not wire into CI.
* **tracelens-analysis-orchestrator — Partial** — analysis offline yes, capture needs `rocprof-compute`/TraceLens; optional offline analyzer.
* **Hyperloom — Does NOT run here** — `compatibility.rst` min `MI300X/MI325X/MI355X` + `gfx942/gfx950` (`gpu_types.py` gfx1100→None, issue #1041), images `rocm/hyperloom:sglang-…-mi300x`. Reference loop only (Magpie→TraceLens→Arbor/GEAK roofline).
* **ROCm SDK marketing `amd.com/sdk.html` → canonical `rocm.docs.amd.com/en/latest/components/core.html`** (HIP, HIPIFY, LLVM, rocprof-compute/systems-profiler/SDK, ROCgdb, AMD SMI, rocminfo).

## Gaps to verify on RX 7900 XT before locking 04-01

- VOPD forbidden-pair matrix (ISA Table 7.x) — re-check downloaded PDF §§7.4–7.9.
- VGPR granule wave32 vs wave64 (4 vs 8 VGPR blocks) — `llvm-calc-occupancy --amdgpu-arch=gfx1100` + RGA `allocated vs used`.
- `librocdxg` 1.2.0→1.2.1 `dids.conf` delta — `git diff v1.2.0..v1.2.1`.
- `VDR_IQ4_XS_*` exact — local `grep -r VDR_IQ4 ggml/src/ggml-cuda`.
- WMMA 32-cycle / XOR-swizzle `s_waitcnt` vs LDS under 64KB block — profile `s_waitcnt` on decode kernel.
- Magpie `warmup_iterations/num_iterations` defaults — clone `AMD-AGI/Magpie` and parse `kernel_config.yaml.example`+`argparser`; verify `rocprof-compute` metric blocks on gfx1100+WSL2.
- No stable LDS bandwidth microbenchmark for RDNA3 to quantify bank-conflict cost — rely on PIX counters + iterative padding sweeps.
- `rocprofv3` counter list for gfx1100 RDNA3 (SQ_ACCUM_* CDNA-biased) — need live 7900 XT capture.

