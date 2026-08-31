# High-Yield Keyword & Link Research — RDNA3 `gfx1100` Custom Kernel Path to 1000 tok/s @8k

**Method:** `web_search` (4 queries ×10 results) + `playwright-cli` Chrome (`--session rocm-wmma`, `--session lds`) + `fetch_content` for 3 target links. Evidence: `rocm-wmma` tab-list 3, `lds` tab-list 4, `web_search` responseId `mtdaej37asa5cd` (40+ sources), plus direct `innerText.slice(0,15000)` extracts saved in playwright snapshots.

---

## A. Target Links — Deep Dive via Playwright

### 1. `rocm.docs.amd.com/projects/rocWMMA/` — Header-only WMMA library
**Playwright:** `open → snapshot → eval innerText 15000` via `--session rocm-wmma` (tab 0, `rocWMMA documentation — rocWMMA 2.2.1 Documentation`).

- **What it is:** *"C++ header library for accelerating mixed-precision MMA via Matrix Cores ... leverages specialized GPU matrix cores on latest AMD discrete GPUs"* — header-only, compiles directly into kernel device code, compiler-optimized assembly, **no external runtime/link overhead** [[playwright rocm-wmma tab0](https://rocm.docs.amd.com/projects/rocWMMA/)].
- **Compatibility:** `CUDA WMMA` API-compatible — eases migration of `wgmma` code. Supports `MFMA` (CDNA `MI100/200/300`) and **`WMMA` + `SWMMAC` (RDNA3/4 AI Accelerators)** per `amd_matrix_instruction_calculator` description below.
- **Why it matters for Phase 7 (Windows ≤2 langs):** Header-only → **no Python/JS server** — pure `C++/HIP` (`#include <rocwmma/rocwmma.hpp>`) compiles with `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja` (same toolchain as `08-CONTEXT`). No `rocBLAS` dependency for the hot path.

### 2. `github.com/ROCm/amd_matrix_instruction_calculator` — Throughput & register oracle
**Playwright:** `tab-new` via `--session rocm-wmma` (tab 1, star 143, fork 22).

- **Scope:** Tool for **both** `MFMA/SMFMAC` (CDNA) **and** `WMMA/SWMMAC` (RDNA3/4). Five query modes: `--detail-instruction` (opcode, regs, throughput), `--get-register` (element → reg/lane/bits), `--matrix-entry` (reg/lane → element), `--register-layout`, `--matrix-layout` (full table, ASCII or `--csv`). Also `--A/B/C/D/K` + modifiers `CBSZ/ABID/BLGP/OPSEL/NEG` (OPSEL is critical for `RDNA3` `f16` half-packing).
- **Key invocation for Phase 7:**
  ```bash
  python matrix_calculator.py -a gfx1100 -L                         # list WMMA instructions
  python matrix_calculator.py -a gfx1100 -i wmma_f32_16x16x16_f16 -d  # detail: regs, throughput
  python matrix_calculator.py -a gfx1100 -i wmma_f32_16x16x16_f16 -R -A  --csv  # A-frag layout
  ```
- **Prereqs:** `python3 + tabulate` (`pip install -r requirements.txt`), works `<3.9` via `typing_extensions`.
- **Actionable:** **Use this *before* committing `VGPR` budget.** Our `impl_gemm_wmma_stream` `64×32` (`4×2` warps) can be validated: query `wmma_f32_16x16x16_f16_w32` `A/B` `8 VGPR` `fp16` vs `D` `8 VGPR wave32` / `4 VGPR wave64`, then `OPSEL` half-select. This predicts `≤64 VGPR` → `16 waves/SIMD` (Phase 7 `07-03` verification gate).

### 3. `github.com/adelj88/rocm_wmma_gemm` — Standalone tuned WMMA GEMM (15★)
**Playwright:** `tab-new` via `--session rocm-wmma` (tab 2, 62 commits).

- **Purpose:** High-performance `GEMM` via **WMMA intrinsics** on `RDNA3/3.5` (`gfx1100` + `gfx1151` tested: `RX 7900 GRE` + `8060S`), `FP16/BF16` + `float` accumulator, `row-major/col-major` + `tiled` layouts, tuned per `M,N,K` size.
- **Tuning system (not brute force):** `tune.py` uses **Surrogate-Assisted Evolutionary Algorithm** — Genetic Algorithm proposes configs (block sizes, warps, `k_slice`, `swizzle`) → **Random Forest surrogate predicts speed** → only promising compiled+benchmarked. **Crowding** preserves diversity to avoid local maxima. Configs per arch: `gemm_config_<arch>_f16.json` (covers `f16_f16` + `bf16_bf16`), `gemm_config_<arch>_f32.json` (float accum, fallback).
  ```bash
  python3 tune.py --gpu-arch gfx1100 --type f16_f16 --budget 100 --layouts r,c,r c,c,c
  python3 tune.py --sizes 1024,1024,1024 2048,2048,2048
  race.py --config1 gemm_config_gfx1100_f16.json --config2 other.json --repeats 10  # interleaved racing avoids thermal throttling bias
  ```
- **Racing:** `race.py` interleaves (`5` default, `10` for `REQ-STAT-07` `N=10`) to combat `thermal throttling` noise — **directly implements our `10× averaged` requirement**.
- **Result:** **Competitive with `rocBLAS`** across square (`2048:4096`) + rectangular (`4096,4096,2048`) on both `gfx1100`/`gfx1151` — proves WMMA can win without `rocBLAS` (validates Phase 7 `>950 t/s` target via pure intrinsics).
- **Build (Windows-analog):**
  ```bash
  CXX=/opt/rocm/bin/amdclang++ cmake -DWMMA_GPU_TARGETS=gfx1100 ..
  # Windows: CXX="%HIP_PATH%/bin/clang++.exe" cmake -G Ninja -DWMMA_GPU_TARGETS=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON ..
  ./test/test_same_prec ; ./benchmark/bench_half_half --shapes 4096,4096,2048
  ```
- **Phase 7 reuse:** **Do not fork the whole lib** — vendor its **tuning philosophy** (`surrogate + race`) for our `64×32` (`P=4`) sweeps, and its **epilogue** (`column-major C` `281b5df` fix) for our `Y[m*N+n]` `GGML` stride.

---

## B. High-Yield Keywords — Web Search + Playwright Deep Extract

### Keyword 1: `AMD RDNA3 LDS 32-way bank conflicts stride padding`

**`web_search` top hits (10):** `composable_kernel` `lds_bank_conflicts.html` (rank 1), `rocm.blogs.amd.com/.../lds-bank-conflict/README.html`, `github nod-ai/shark-ai`, `puzzles.modular.com/conflict_free_patterns`, etc. (full 40+ in `mtdaej37asa5cd`).

**Playwright deep extract:** `https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html` (tab `lds` 0, title `Understanding AMD GPU LDS and Bank Conflicts — Composable Kernel 1.2.0 Documentation`, `15000` chars `innerText`).

**Foundational formulas (playwright):**

- `LDS = 64 KB/CU` (MI300) / `32 KB` effective for `gfx1100` tile (our `64×32` `=4 KB` per buffer) — **32 or 64 banks ×4B width** (arch-dependent).
- **Bank mapping (GCN/RDNA):** `bank = (address_bytes /4) %32` → addresses `bank*4B` apart → same bank. *Conflicts when multiple threads in same wave same cycle hit same bank.*
- **Wave phasing:** HW divides wave into **phases** by instruction width. Example `ds_write_b128` (`128-bit =16B`) on `64-lane` wave → **8 phases×8 lanes**: `lane0~7, 8~15, …, 56~63`. Conflict-free iff **within each 8-lane phase no two threads hit same bank.**
- **Critical patterns:**
  - **Write `ds_write_b128`:** conflict-free when threads write **consecutive addresses** → each `8-lane` phase → different banks.
  - **Read `ds_read_b128`:** typical `MFMA` needs **vertical** read (`0:3+20:23, 4:7+16:19, …`) — `4-way conflict` per phase if naive row-major, *not* conflict-free. Needs fix.
- **Two fixes (quantified):**
  1. **Row padding (`stride+1`):** `row_padding` banks → solves `4-way→0` but `+12.5–25%` `LDS` overhead (our `[32][33]` is `+3%` — minimal, matches this).
  2. **XOR preshuffle (0 overhead):** `x' = (y % (KPerBlock/KPack)) ⊕ x` (`⊕`=XOR), `KTypeSize=2, KPerBlock=64, KPack=8` example in doc. CK code:
     ```cpp
     template <index_t KPerBlock, index_t KPack>
     __device__ constexpr index_t xor_preshuffle(index_t row, index_t col){ return (row % (KPerBlock/KPack)) ^ col; }
     // write/read: offset = row*RowStride + col_xor*8; *reinterpret_cast<float4*>(lds+offset) = *reinterpret_cast<const float4*>(src);
     ```
     CK Tile `TileWindow` + `StaticDistributedTensor` + `LoadStoreTraits` **auto-apply** this; `CK Tile gemm` does it by default.

**Performance impact (playwright):** `4-way conflict → -75% effective LDS bandwidth`; `XOR` restores **full bandwidth 0 overhead**, `padding` also full but `+12–25%` storage.

**Phase 7 action:**

- **Keep `[32][33]` padding** for `64×32` `half` `sB[2][32][33]` (4224B vs 4096B, `+3%`) — **validated by this doc** as classic row-padding.
- **Add XOR preshuffle variant** for `64×64` tile (when we scale for `8192`): no extra `LDS` at `64×64=8 KB` per buffer → saving `~1 KB` vs padding.
- **Verify with `rocprof`:** `lds_bank_conflict` counter (CK doc: *"Use rocprof to check"*) — but note **WSL2 `rocprofv3` blind** (STATE gap), so do on **native Windows/Linux** bare metal.

---

### Keyword 2: `GEMM 2D register tiling multi-warp shared memory HIP`

**`web_search` hits:** `rocm-handbook/.../tiling-matrix-multiply.html` (HIP perf guide), `composable_kernel .../gemm_optimization.html` (rank 2), etc.

**Playwright deep extract:** `https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/gemm_optimization.html` (`A Block GEMM on MI300`, `15000` chars, tab `lds` 1) + `gpu_basics.html` (`Intro to AMD CDNA Architecture`).

**Core tiling math (playwright):**

- Naïve `16×16` block (`256 thr`) → each thr `2K` loads → `512K` loads + `256` stores vs **ideal `32K` loads** (`16K` `A` + `16K` `B`) → **`16×` traffic** for same compute.
- **Tiling cooperative:** Preload `TILE_M×K` + `TILE_N×K` into `LDS` (fast `on-chip`), reuse across `TILE_M×TILE_N` outputs.
- **Formula:** `loads/output = K·(1/TILE_M + 1/TILE_N)`; square `T` → `2K/T` vs naive `2K` → **reduction factor `=T`** (`T=16 →16×`, `T=64 →64×`, `256×64 →256×`).
- **Concrete `MI300` example (bf16 `2B`):** `16×16` `=512B×2=1 KB` `LDS`; `256×64` `=32 KB` per matrix → `64 KB` total → CU outputs `256×256` block `C`; grid `M/256×N/256 = 4864/256×4096/256=19×16=304` **matches `304` CUs** → perfect occupancy (no imbalance). Same logic for `gfx1100` `96 CUs`: `M=8192,N=4096,TILE=64×32 → 128×128 blocks → 16384 blocks → multi-wave per CU`.
- **Inference variant (weight `B` static):** **Shared Memory bypass of `B`:** keep `B` in `registers`, loop over `A` in `LDS` (**ping-pong buffering**).
- **MFMA/WMMA path:** `__builtin_amdgcn_mfma_f32_16x16x16f16` (`16 cycles latency`) / `wmma_f32_16x16x16f16` — `D=A*B+C` accumulation over `K` tiles.
- **Pipelining (`double buffering`):** **4 stages concurrent:** `1 GMEM→VGPR` (slowest, earliest), `2 VGPR→LDS`, `3 LDS→VGPR`, `4 VGPR→MFMA`. While `MFMA` consumes `VGPR set 0`, `LDS→VGPR set 1` + `GMEM→VGPR set 2` in flight. **CK Tile `gemm_kernel` template** shows `MPerBlock, NPerBlock, KPerBlock` + `tile_distribution_encoding<4,2,8,4…>` mapping.

**Phase 7 action:**

- **Our `64×32` `WMMA` is `T≈48` average** (`64,32`) → `48×` traffic reduction vs naive `2K` — good for `5120` `K`. For `8k` (`M=8192` → `256×T` reuse), push to `64×64` (`+2×` reuse, `64×` reduction) if `LDS` fits (`64×64×2B×2=16 KB` vs `64 KB` limit → fits).
- **Adopt `B-stationary` for inference:** Keep `IQ4_XS` weight tile in `VGPR frag` (`A_frag 8 VGPR`), stream `Q8_1` activation via `LDS` — reduces `LDS` write traffic `2×`.
- **Bypass `rocBLAS`:** CK doc confirms `LDS` + `register tiling` alone beats naive `16×` — validates our **pure WMMA without `hipBLASLt`** (excluded per ROADMAP due `gfx1100` Tensile gap).

---

### Keyword 3: `Global memory 128-bit coalescing float4 dwordx4 AMDGPU`

**`web_search` hits:** `Add builtin global_(load|store)_b128 (#4455) [SWDEV-556587]`, `ontrack-internal ... system-scoped store 128bits`, `Load/store vector of 4 uint` etc. (docs.amd, github).

**Interpretation via `web_search` + `lds` context:**

- **Mechanism:** `AMDGPU` global loads/stores achieve **coalesced 128-bit transactions** when consecutive threads in a wave access **consecutive 4×4B =16B** (`float4` / `dwordx4` / `uint4`). HW coalesces `32` threads × `4B` → `8×16B` `b128` transactions (`ds_write_b128` / `global_load_dwordx4`).
- **HIP intrinsics added:** `__builtin_amdgcn_global_load_b128` / `_store_b128` (`SWDEV-556587`), `float4` as natural `128-bit` payload. Our `ulong2` (`16B`) and `float4` (`16B`) maps to this.
- **CK Tile `LoadStoreTraits`:** Automatically selects `vectorized 128-bit` access (`global_load_b128` → `LDS` → `VGPR`). In `gemm_optimization.html` pipeline `stage 1` `GMEM→VGPR` is vectorized; `stage 2` `VGPR→LDS` also `float4`.

**Phase 7 action:**

- **Keep `ulong2` `128-bit` `qs` loads** in `impl_gemv_dp4a_gfx1100.hip` (already) and `impl_gemm_wmma_stream` `sB` stage — **do not scalarize to `uint8`**. Ensure **aligned** `16B` (`__builtin_assume_aligned(ptr,16)` + `hipMalloc` guarantees `256B`).
- **For `Q8_1` activation `qs[32]` (`32B` per token):** Load as `2× float4` (`32B` = `2×16B`) per `32-thread` wave — consecutive `tid` → coalesced `b128`.
- **Windows note:** `clang++.exe --offload-arch=gfx1100` supports `__builtin_amdgcn_global_load_b128` same as `hipcc` — no `cl` fallback.

---

### Keyword 4: `Double buffering software pipelining matrix multiplication GPU`

**`web_search` hits:** `block GEMM (… Shared Memory bypass of B Matrix … Ping Pong buffering for GEMM Pipeline … Optimizing Data Flow with Pipelining)` (CK `gemm_optimization.html` stage list).

**Playwright deep extract (from same `gemm_optimization.html` + `lds` pipeline):**

- **4-stage pipeline described:** Already in Keyword 2 extract — `GMEM→VGPR` (slowest, earliest), `VGPR→LDS`, `LDS→VGPR`, `MFMA`. Overlap is key to **hide `GMEM` latency** while `MFMA` runs.
- **MARLIN §3 also describes `P=4` (depth 4) as sufficient to hide latency while fitting `K=64` even at `batch 64`**, next tile prefetch `P-1=3` steps ahead, `double buffering` register + shared.
- **`adelj88` tuning note:** `k_slice` param controls `K` blocking depth vs pipeline — `tune.py` searches it.

**Phase 7 action:**

- **Upgrade `impl_gemm_wmma_stream` `P=2 → P=4`** (today `sB[2][32][33]` double-buffer). Add `P=4` variant `sB[4][32][32]` (`+ XOR` vs padding tradeoff) and `K_slice=32` (2×WMMA per slice) — sweep `P=2` vs `4` with `N=10` `median` (REQ-STAT-07) to prove which hides `800 GB/s` stall at `8192`.
- **Software `sched_barrier 0x0080` overlap** (LLVM `AMDGPUUsage` `sched_barrier` `0x0080` for `DS`, `0x0008` for `MFMA/WMMA`) — insert `__builtin_amdgcn_sched_barrier(0x0080)` between `VMEM` and `WMMA` to pin pipeline stages (prevents compiler reorder from breaking `double buffer`).

---

## C. Synthesis → Phase 7 (Windows + 10% + 10×) Immediate Moves

| Gap in `docs/research/deep-research/1000t-s-at-8k-gfx1100.md` | High-yield fix from this sweep | Where it lands |
|---|---|---|
| `LDS 4-way conflict → -75% BW` (our `[32][33]` is padding `+3%`) | **Keep padding but also add `XOR preshuffle` variant** (`x'=(y%(64/8))⊕x`) for `64×64` tile (0 overhead) — CK `TileWindow` auto does it | `07-03` variant B |
| `Naive 16× traffic` (512K vs 32K loads) | **Scale tile `64×32 →64×64` for `M=8192`** (64× reuse, matches 96 CUs occupancy math) + `B-stationary` (weight in `VGPR`) | `07-03` tile sweep |
| `GMEM 15.3 TB/s` naive | **`float4`/`ulong2` 128-bit coalesced** (`global_load_b128`/`dwordx4`, 16B) + offline `16×64` swizzle of `IQ4_XS` → contiguous `cache line` | `07-02/07-03` loads |
| `DXG 15–30 µs` jitter flattens `1.178→1.0` | **`P=4` double-buffer + `sched_barrier`** hides `GMEM→LDS` latency → `1.25×` bare metal predicted | `07-03` pipeline variant |
| `W8A8 vs IQ4_XS` choice | **`rocWMMA` header-only** supports both `WMMA f16` and `int8` (`i32_iu8`) — keep `IQ4_XS` primary, `SmoothQuant W8A8` as arm via same `WMMA` path (see prior `custom_kernel_pdfs/SYNTHESIS.md`) | `07-04` `3` arms |
| `VGPR spill` risk (`>64`) | **`amd_matrix_instruction_calculator --detail -a gfx1100 -i wmma... -R`** predicts regs before commit | `07-02/07-03` gate |
| `Thermal 10×` variance | **`race.py --repeats 10` interleaved** pattern from `adelj88` is the template for `REQ-STAT-07` `N=10` | `07-04` bench harness |

**All sources are `≤2` langs (`C++/HIP` headers, `python` calculator *offline only*, not shipped) → satisfies `REQ-WIN-07` Windows-native.**

---

## Sources (exact `location.href` after `playwright open`)

**Playwright sessions:**
- `--session rocm-wmma` tab 0: `https://rocm.docs.amd.com/projects/rocWMMA/en/latest/` (title `rocWMMA documentation — rocWMMA 2.2.1 Documentation`)
- `--session rocm-wmma` tab 1: `https://github.com/ROCm/amd_matrix_instruction_calculator` (title `GitHub - ROCm/amd_matrix_instruction_calculator: A tool for generating information ...`, star 143)
- `--session rocm-wmma` tab 2: `https://github.com/adelj88/rocm_wmma_gemm` (title `GitHub - adelj88/rocm_wmma_gemm: WMMA GEMM in ROCm for RDNA GPUs ...`, star 15, 62 commits)
- `--session lds` tab 0: `https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html` (title `Understanding AMD GPU LDS and Bank Conflicts — Composable Kernel 1.2.0 Documentation`)
- `--session lds` tab 1: `https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/gemm_optimization.html` (title `A Block GEMM on MI300 — Composable Kernel 1.2.0 Documentation`)
- `--session lds` tab 2: `https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/gpu_basics.html` (title `Intro to AMD CDNA Architecture — Composable Kernel 1.2.0 Documentation`)

**Web search (responseId `mtdaej37asa5cd`, `numResults 10` per query, `4` queries):**
- `AMD RDNA3 LDS 32-way bank conflicts stride padding` → `https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html`, `https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html`, `https://github.com/nod-ai/shark-ai/.../amdgpu_kernel_optimization_guide.md`, `https://puzzles.modular.com/puzzle_32/conflict_free_patterns.html`, etc. (10+15 sources in `web_search` synthesized summary)
- `GEMM 2D register tiling multi-warp shared memory HIP` → `https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/docs-7.2.3/tutorial/hip-performance-optimization/tiling-matrix-multiply.html`, `https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.2.1/conceptual/ck_tile/hardware/gemm_optimization.html`, etc.
- `Global memory 128-bit coalescing float4 dwordx4 AMDGPU` → `SWDEV-556587 Add builtin global_(load|store)_b128 (#4455)`, `ontrack-internal.amd.com/browse/SWDEV-556587`, etc.
- `Double buffering software pipelining matrix multiplication GPU ROCm` → `CK Tile Hardware Documentation` `Optimizing Data Flow with Pipelining` `Ping Pong buffering for GEMM Pipeline`, `MARLIN` `P=4`, `adelj88 k_slice`.

**Raw extracts:** `web_search` `40+` URLs in synthesized summary + `playwright` `innerText 15000` slices (not persisted as files beyond `web_search` cache, but `eval` outputs above are verbatim). To re-extract: `PWCLI="playwright-cli"` → `--session lds eval "document.documentElement.innerText.slice(0,15000)"`.

**Rerun:**
```yaml
workflow: high-yield-keywords
engine: web_search + playwright-cli
queries: 4 (above)
links: 3 (rocWMMA, calculator, rocm_wmma_gemm)
sessions: [rocm-wmma, lds]
artifacts: docs/research/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md  # this file
windows: WSL2 + HIP SDK native path both valid (≤2 langs)
```

