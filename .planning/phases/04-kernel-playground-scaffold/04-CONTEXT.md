# Phase 4: Kernel Playground Scaffold - Context

**Gathered:** 2026-08-25
**Status:** Ready for planning
**Deep research:** 6 parallel subagents + 8 direct fetches covering all 15 resources in `@.planning/reference/GPU-KERNEL-RESOURCES.md` and re-verified `@.planning/research/EXTERNAL-RESOURCES-ASSESSMENT.md`

<domain>
## Phase Boundary

Build a standalone HIP kernel development pipeline that operates end-to-end **outside llama.cpp** — CPU reference → HIP implementation → numerical comparison → microbenchmark — making miscompiles debuggable in minutes instead of inside 15-GB model runs.

Requirements in scope: **KERN-01** only (sanctioned overlap: depends on Phase 1 toolchain only, may run concurrently with Phases 2–3 — both now complete).

**KERN-01 success criteria (binding):**

1. The playground directory builds standalone HIP executables for **gfx1100** with **zero llama.cpp headers** included, using tensor fixtures dumped from real GGUF weights.
2. A candidate op traverses the full quartet pipeline (`ref_cpu.cpp` → `impl_gfx1100.hip` → `test_compare.cpp` → `bench_sweep.cpp`) producing recorded error metrics (max-abs / mean-abs / relative / cosine) and timing tables.
3. A deliberately broken implementation is caught **RED** by the comparison stage, and its corrected version passes **GREEN** — proving the pipeline discriminates, not just runs.

Phase 4 produces **zero optimizations** — it is measurement/validation infrastructure so Phase 5's `MUL_MAT` attack is provable. First optimization code appears in Phase 5 against Target #1 `MUL_MAT` (31.12% cumulative GPU time, 50.89% prefill / 30.04% decode per `benchmarks/profiling/BOTTLENECK-TABLE.md`).
</domain>

<decisions>
## Implementation Decisions

### Locked by owner (5/5 A - 2026-08-25 step-by-step)
- **D4-00-1 — Demo op:** **A) `dequant_iq4_xs` only** (unpack `block_iq4_xs → float[256]`). Fused dequant+GEMV deferred to Phase 5. Isolates IQ4_XS bit-layout risk (scales, nibbles, split-half) and gives fastest bug-catch. Synthetic + real fixtures still cover `4096×4096` shapes for Phase 5.
- **D4-00-2 — Struct source:** **A) vendored copy** — `kernels/common/block_iq4_xs.h` copies `block_iq4_xs` + `QK_K=256` + `kvalues_iq4nl[16]` verbatim from `ggml/src/ggml-common.h @ bb4caa75` with Apache-2.0 attribution + `static_assert(sizeof(block_iq4_xs)==136)`. Gate: `rg -r "ggml|llama" kernels/` → 0 hits.
- **D4-00-3 — Result store:** **A) reuse `benchmarks/lib/store.py`** — `bench_sweep` emits fingerprinted JSON (commit, ROCm/driver, GGUF sha256 `53adc4bb…`, clocks/temps via Windows telemetry) into `benchmarks/results/kernels_*/` (or `kernels/results/` symlink for ergonomics). Mirrors Magpie discipline and Phase 2 BENCH-02.
- **D4-00-4 — PASS gate:** **Tight** `max_abs < 1e-5`, `mean_abs < 1e-6`, `cosine > 0.99999`, `max_rel < 1e-4` (for |ref|>1e-3). Broken must fail by **≥10×** on `max_abs`. Dequant is lossless unpack → ~1e-7 from fp16→fp32 rounding only; fused `MUL_MAT` in Phase 5 will loosen to `1e-3 / 0.999`.
- **D4-00-5 — Wave handling:** **A) templated `WARP_SIZE`** — `template<int WarpSize>` + `__launch_bounds__(256,4)` + `--save-temps` VGPR gate; bench both `wave32` (native 32-wide, 1-cycle on RDNA3) and `wave64` (emulated 2×32) per kernel. Never literal `32`/`64`; `lane_mask_t = uint32_t` for wave32 vs `uint64_t` for wave64.

### Scaffold & Standalone Build (Plan 04-01)
- **D4-01:** CMake 3.21+ `project(LANGUAGES HIP CXX)` + `CMAKE_HIP_ARCHITECTURES=gfx1100` (or `hip::device` legacy via `find_package(hip)` + `GPU_TARGETS=gfx1100`). `hipcc --offload-arch=gfx1100 -mwavefrontsize32` is the copy-paste build; verify `rocminfo | grep gfx1100` gate + `HSA_ENABLE_DXG_DETECTION=1` shim for ROCm <7.13. Top-level `kernels/CMakeLists.txt` + per-op `kernels/<op>/CMakeLists.txt` standalone like `ROCm/rocm-examples/HIP-Basic`. Ninja + ccache. WSL2 librocdxg constraints designed around: `hipMemGetInfo` ~3 GB under-report (`librocdxg#57`/`#23999`), `.wslconfig memory=32GB` bump, `hipMalloc`+`hipMemcpy` over UVA, gate `rocprof` tests to non-WSL.
- **D4-02:** `kernels/common/` owns `hip_helpers.h` (`HIP_CHECK`), `bench.h` (50 warmup / 200 measure via `hipEvent_t`, median/p95/min/max/stdev per hipEngine), `half_math.h` (fp16→fp32), `block_iq4_xs.h` (vendored). Zero llama headers. Per-op quartet skeleton under `kernels/template/`.
- **D4-03:** RDNA3 tuning levers baked in: VGPR budget `≤96` for 12-wave headroom (1536 VGPRs/SIMD), LDS 128 KiB/WGP 32 banks×4B (pad `[32][33]`), `__launch_bounds__` occupancy caps, `llvm-calc-occupancy` pre-check, RGA `--livereg` CI gate.

### Fixture Dumper (Plan 04-02)
- **D4-04:** `tools/dump_gguf_fixtures.py` via `gguf-py GGUFReader` (memmap, little-endian, `quant_shape_to_byte_shape` with `GGML_QUANT_SIZES[IQ4_XS]=(256,136)`) — dumps N tensors (e.g. `blk.0.attn_q.weight`, `blk.0.ffn_down.weight`) as raw `block_iq4_xs` bytes + dequantized `f32` reference + `.json` manifest (shape, n_blocks, sha256, commit). Also synthetic fixture path via `gguf.quants.quantize` / `quantize_row_iq4_xs_ref` fallback + deterministic edge cases (zero block, min/max scale `ls 0→-32 / 63→+31`, all nibbles 0/15, split-half boundary `lo@i vs hi@i+16`).
- **D4-05:** IQ4_XS layout locked 136B / 256 weights (8×32 sub-blocks): `d(2)+scales_h(2)+scales_l(4)+qs(128)`, 6-bit scale `ls = (scales_h>>2*ib &3)<<4 | (scales_l[ib/2]>>4*(ib%2)&0xF)` centred `ls-32`, codebook `kvalues_iq4nl[16]=[-127..113]`, `qs` low nibble `2i` high `2i+1` split-half across 128B. Validated 3-way (`ggml-quants.h` static_assert, `PR #5747`, `oxillama AVX2`).

### Demo-Op Walkthrough & Negative Test (Plan 04-03)
- **D4-06:** Worked example `kernels/demo_iq4xs_dequant/` traverses full quartet: `ref_cpu.cpp` (pure C++17 calls `dequantize_row_iq4_xs` FP64 oracle), `impl.hip` (correct `__global__ __launch_bounds__` warp-templated dequant), `impl_broken.hip` (one-bit bug: `ls_high<<3` or nibble swap), `test_compare.cpp` (asserts `max_abs`/`cosine`, correct GREEN / broken RED by ≥10×, NaN/Inf check), `bench_sweep.cpp` (wraps `hipEventRecord`, 50 warmup/200 measure, sweeps `M ∈ {1,16,128,512,4096}` tiling, reports `median/p95/min/max/stdev` + VGPR/LDS/GB/s per shape into `benchmarks/lib/store.py`). Both `wave32` and `wave64` instantiations benched.
- **D4-07:** Correctness gate mirrors Magpie discipline (correctness-before-perf, explicit `atol 1e-6 rtol 1e-5`, identical warmup/iterations, baseline-explicit `--baseline 0`). For pure dequant, host `dequantize_row_iq4_xs` is golden oracle without GPU; no matmul needed (wmma_ops 41.3 TFLOPS same validation).
- **D4-08:** Agentic coordination: 04-01 ∥ 04-02, 04-03 depends on both stubs. Whole phase is model-independent and was sanctioned to overlap Phases 2–3 (now complete, toolchain proven `bb4caa75`).

### Claude's Discretion
- Exact CMake target names, `CMakePresets.json` shape, `kernels/common` header split.
- Python CLI flags for `dump_gguf_fixtures.py` (`--tensor`, `--num-blocks`, `--output`).
- `bench_sweep` JSON schema field ordering, `bench.h` helper API, and per-shape `VDR`/`AK1/BK1` naming.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project planning docs
- `.planning/ROADMAP.md` §Phase 4 — KERN-01 success criteria, RDNA3 risk notes (warpSize templating), parallelization
- `.planning/REQUIREMENTS.md` §Kernels & Integration — KERN-01 authoritative text
- `.planning/STATE.md` — Phase 3 deliveries, Optimization Target #1 `MUL_MAT`, frozen env `bb4caa75` / ROCm 7.2.1 / `53adc4bb…`
- `benchmarks/profiling/BOTTLENECK-TABLE.md` + `bottleneck_summary.json` — Target #1 attribution (31.12% cumulative)
- `benchmarks/profiling/dispatch_overhead_report.md` — HIP graphs +19% decode, isolation of kernel time
- `.planning/phases/03-correctness-gates-bottleneck-profiling/03-CONTEXT.md` — prior gate/profiler patterns
- `benchmarks/lib/store.py`, `benchmarks/lib/fingerprint.py`, `benchmarks/lib/guard.py` — reuse for fingerprinted bench store

### Deep research (6 subagents + direct fetches, 2026-08-25)
- `amd-isa` — RDNA3 ISA 70650 (VOPD 8B dual-issue, WMMA 16×16×16 wave32 32-cycle, VGPR granule +50%, librocdxg WSL2 deltas #57/#60/#6022)
- `hip-guides` — Performance Guidelines (coalescing 128B, LDS 32-bank pad `[32][33]`, EXEC divergence, `__launch_bounds__`, `llvm-calc-occupancy`), HIP docs (CMAKE_HIP_ARCHITECTURES, WGP vs CU #3374, hip::device #2158)
- `ck-lib` — Composable Kernel Tile `UniversalGemmKernel` reference-only verdict (heavy, Instinct xdl, shim pattern `ck_tile_shim.h`)
- `gpuopen` — Occupancy (16 slots/SIMD, 1536 VGPRs/SIMD, RGP Pipeline, PIX WaveOccupancyLimiters), Live VGPR RGA `--livereg`, VGPR reduction playbook (ByteAddressBuffer, bit-pack, 16-bit)
- `quant-kernels` — `wmma_ops` lane-replicated fragments, `ggml-cuda vecdotq.cuh` HIP `__builtin_amdgcn_perm`, IQ4_XS 136B/8×32, Marlin 128-bit fused pipeline
- `external-assessment` — Magpie analyze/compare discipline (explicit `atol/rtol`, identical warmup/iterations, `--baseline 0`, `analyze_report.json`), rocm-doctor WSL2 decline, Hyperloom MI300-only (gfx942/gfx950)

### External resource hubs (from @.planning/reference/GPU-KERNEL-RESOURCES.md)
- **AMD ISA:** `docs.amd.com RDNA3 ISA` (70650), `rocm.docs.amd.com`, `ROCm on Radeon WSL2`, `github.com/ROCm`
- **HIP:** `HIP Programming Guide`, `HIP Performance Guidelines`, `HIP Performance Optimization`, `ROCm-Examples/HIP-Basic`, `ROCm/HIPIFY`
- **CK:** `Composable Kernel Docs`, `github.com/ROCm/composable_kernel`
- **GPUOpen:** `Large Thread Groups`, `Occupancy Explained`, `Live VGPR Analysis (RGA)`
- **Quant:** `wmma_ops`, `ggml-cuda mmvq/mmp/vecdotq`, `IST-DASLab/Marlin`
- **External assessment re-verified:** `amd/skills` magpie (mine), `rocm-doctor` (WSL2 out-of-scope), `Hyperloom` (MI300-only), `rocm.docs core.html` canonical

### Upstream / vendor sources
- `ggml/src/ggml-common.h @ bb4caa75` — `block_iq4_xs` struct, `QK_K 256`, `kvalues_iq4nl[16]`
- `ggml/src/ggml-quants.c` — `quantize_row_iq4_xs_ref` / `dequantize_row_iq4_xs` FP64 oracle
- `ggml/src/ggml-sycl/dequantize.hpp` — `dequantize_block_iq4_xs` tiling proof
- `ggml/src/ggml-cuda/vecdotq.cuh` + `mmvq.cu`/`mmq.cuh`+`mma.cuh` + `generate_cu_files.py:TYPES_MMQ` — VDR dispatch, `get_int_from_table_16` perm
- `gguf-py/gguf/gguf_reader.py`, `gguf-py/gguf/quants.py`, `gguf-py/gguf/constants.py: GGML_QUANT_SIZES`
- `oxillama-quant/simd/avx2/iq4_xs.rs` — exhaustive byte-layout table
- `hipEngine docs/BENCHMARK.md` — 50 warmup / 200 measure, median/p95/min/max/stdev evidence policy (primary for `bench.h`)
- `ROCm/ROCm-Examples HIP-Basic/hello_world/CMakeLists.txt` — canonical CMake HIP language pattern
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `baseline/binaries/v0.2.0-bb4caa75/` — stock `llama.cpp @ bb4caa75` HIP binaries (archived)
- `/root/llama.cpp` guest tree (ext4, not `/mnt/*`) — source for vendored `block_iq4_xs` copy + oracle
- `models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` — 15.31 GB locked artifact, sha256 `53adc4bb…` (see `models/README.md`)
- `models/README.md` — imatrix / quant provenance
- `benchmarks/lib/{store, fingerprint, guard, parse_profile, llabel, toast}.py` — store/fingerprint/guard reuse
- `benchmarks/bin/{run_session, run_op_gate, run_model_gate, profile_workload}.py` — harness conventions to mirror
- `benchmarks/environment/{versions.txt, hipconfig.txt, rocminfo.log, vram_probe.txt}` — fingerprint components
- `benchmarks/host/hwinfo_daemon.py` — Windows-side HWiNFO thermal telemetry (95 °C watchdog)
- `.agents/skills/magpie-kernel-evaluator/PROJECT-NOTES.md` — Magpie analyze/compare prior art, unverified on gfx1100

### Established Patterns
- Toolchain: `hipcc --offload-arch=gfx1100` / `CMAKE_HIP_ARCHITECTURES=gfx1100`, `hip::device` + `$<$<COMPILE_LANGUAGE:HIP>:>` guard, `HSA_ENABLE_DXG_DETECTION=1` for <7.13, `Ninja` + `ccache`
- Invocation: `setsid` + `--simple-io` + `--single-turn` + explicit `-c` + `--no-mmap`, `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root`
- Telemetry: guest `rocminfo` + Windows HWiNFO, `hipEvent_t` timing with `hipEventSynchronize` before `hipEventElapsedTime`
- Safety: `.wslconfig memory=32GB`, `hipMemGetInfo` tolerance (~3 GB), 3-signal VRAM guard, `llvm-calc-occupancy` budget

### Must Avoid
- Any `#include "ggml.h"` in `kernels/` (KERN-01 isolation break)
- Literal `32`/`64` warp assumptions — template on `warpSize`/`WARP_SIZE`
- `hipEventDisableTiming` when `hipEventElapsedTime` needed; `hipEventRecord` inside captured HIP graphs (ROCm unsupported)
- Fat binaries via `amdgpu-arch` auto-detect — pin `gfx1100` only
</code_context>

<specifics>
## Specific Ideas

- Fixture dumper CLI: `python tools/dump_gguf_fixtures.py --model models/...gguf --tensors blk.0.attn_q.weight --tensors blk.0.ffn_down.weight --num-blocks 8 --out kernels/fixtures/`
- Manifest fields: `{name, tensor_type: "IQ4_XS", shape, n_blocks, block_size:136, QK_K:256, sha256, commit:"bb4caa75", rocm:"7.2.1", artifact:"53adc4bb…"}`
- Demo `bench_sweep` sweeps `n_blocks ∈ {1,8,64,512}` + `wave32/64` both, emits `bench_sweep.json` with per-shape `median_us/p95/min/max/stdev` + `vgpr/lds/gb_s` + footer fingerprint via `store.py`
- Broken impl: `scales_h` shift `>> (2*i)` → `>> (2*i+1)` or `qs` lo/hi nibble swap `i vs i+16` — must trip `max_abs` to >0.5
- Wave templating: `template<int kWarp=WARP_SIZE> __global__ __launch_bounds__(256,4) void dequant_kernel(...) { int lane = threadIdx.x % kWarp; }`

</specifics>

<deferred>
## Deferred Ideas

- Full IQ4_XS fused `MUL_MAT` GEMV/GEMM `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` + XOR-swizzle LDS (Phase 5 — ck-lib Tile pipeline, Marlin 128-bit staged loads, dp4a/wm_mma lowering)
- CK Tile vendor shim `ck_tile_shim.h` (only if types needed, per aiter pattern) — keep playground standalone in Phase 4
- VOPD dual-issue `v_dual_add_f32` microprobe and `llvm-objdump --mcpu=gfx1100` disassembly check (Phase 5 optional)
- RGA 2.7 `gfx1100` Live VGPR GUI column + `rocprofv3 --occupancy` on toy 256-thread shared `[32][33]` kernel to lock VGPR numbers
- Native-Linux `rocprof-compute` WGP VALU 256 FLOPs/CU/cycle peak validation (WSL2 only wall-clock+disasm)
</deferred>

---

*Phase: 4-Kernel Playground Scaffold*
*Context gathered: 2026-08-25*
*Deep research: 6 subagents (amd-isa, hip-guides, ck-lib, gpuopen, quant-kernels, external-assessment) + 8 direct fetches*
*Owner locks: 5/5 A (dequant-only, vendored copy, reuse store, tight gate +10×, templated WARP_SIZE)*
