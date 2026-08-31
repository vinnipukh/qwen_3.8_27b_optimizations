# Proposal: Compile Real Variant Objects forGEMM High-Yield Sweeps

**Date:** 2026-08-29
**File inspected:** `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` L141
**Context:** Phase 7 re-scoped high-yield: tile sweeps 64x32/64x64/128x32, P=2 vs P=4, XOR vs +33

## L141 Finding

```hip
// High-yield: P=4 variant would be __shared__ _Float16 sB[4][32][32] (+XOR) for GMEM->VGPR->LDS->VGPR->WMMA 4-stage overlap pinned with sched_barrier
__shared__ _Float16 sB[2][32][33];
// __shared__ _Float16 sB_P4[4][32][32]; // P=4 XOR variant alternative (kept as comment for gate, 16x64 swizzle companion)
```

- L141 is **comment-only**: documents `P=4` `sB[4][32][32]` XOR quad-buffer as future work.
- Real code at L142 is `sB[2][32][33]` P=2 double-buffer, +33 padding (Variant A).
- Comments at L5, L138, L161 also document `sB[4][32][32]` and `template<int TILE_M,int TILE_N>` sweeps, but no compiled objects exist.
- Header claims `template<int TILE_M, int TILE_N> (compile flag TILE_M/N)` but file has no such template; only `template<int TILE_M=16>` for fallback tiled path (L31) and hard-coded `BLOCK_N=64 BLOCK_M=32 K_TILE=32` (L126-128).

**Implication:** `bench_gemm_wmma.cpp` documents 5 variants (64x32_P2+33, 64x32_P4_XOR, 64x64_P4_XOR, 128x32, LUT_mu4) and `bench` documents `phases 1-5` with `--variant race`, but all 5 map to the **same** compiled kernel (P=2 64x32). Race would compare identical code.

## Proposal: Minimal Real Variant Objects (no scope widening beyond compiled variants)

**Goal:** Make each variant a real compiled object that builds clean via `hipcc --offload-arch=gfx1100` and is benchable via `--variant`, without redesigning WMMA algorithm.

### 1. File structure (additive, no deletion)

Keep `impl_gemm_wmma_stream.hip` as base (`matmul_gemm_wmma_stream_hip` = 64x32 P2+33, current). Add two small variant files that `#include` base logic with compile-time constants:

- `kernels/matmul_iq4xs/impl_gemm_wmma_64x32_p4_xor.hip`
  ```
  #define TILE_M 32  // keep 64x32 but show pattern
  #define TILE_N 32
  #define P 4
  #define USE_XOR 1
  #include "impl_gemm_wmma_stream.hip" // with #ifdef guards for sB[P][...]
  ```
  Real: replace `__shared__ _Float16 sB[2][32][33]` under `#ifdef USE_XOR` with `__shared__ _Float16 sB[4][32][32]` and XOR indexing `xor_preshuffle()` (reuse `gemv_variant_xor.cuh` logic for GEMM: `int x_xor = (y % (K_TILE/8)) ^ x`).

- `kernels/matmul_iq4xs/impl_gemm_wmma_64x64_p4_xor.hip`
  ```
  #define TILE_M 64
  #define TILE_N 32 // or 64x64 = BLOCK_N=64 BLOCK_M=64
  #define P 4
  #define USE_XOR 1
  // BLOCK_N=64 BLOCK_M=64, 16 warps per block -> need 512 threads or 2 blocks; propose 128x32 variant as 256 threads 8x2 warps still 64x64 tile via 2x4 warps with doubled LDS
  ```

- `kernels/matmul_iq4xs/impl_gemm_wmma_128x32.hip`
  ```
  #define BLOCK_N 128
  #define BLOCK_M 32
  // 256 threads 8 warps: 8x2 warps still viable, 128 blocks per row at M=8192
  ```

**Alternative single-file approach (lower diff):**
Add `#ifndef TILE_M` guards in `impl_gemm_wmma_stream.hip`:

```hip
#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef P
#define P 2
#endif
#if P==4
__shared__ _Float16 sB[4][32][32];
  #define XOR_IDX(y,x) ((y % (K_TILE/8)) ^ (x))
#else
__shared__ _Float16 sB[2][32][33];
  #define XOR_IDX(y,x) (x)
#endif
```

Then compile with `-DTILE_M=64 -DP=4 -DUSE_XOR` per variant.

### 2. CMake: one OBJECT per variant (additive)

In `kernels/matmul_iq4xs/CMakeLists.txt` (add, do not replace base):

```cmake
add_library(matmul_gemm_wmma_64x32_p4_xor_hip OBJECT impl_gemm_wmma_64x32_p4_xor.hip)
target_compile_definitions(matmul_gemm_wmma_64x32_p4_xor_hip PRIVATE TILE_M=32 TILE_N=32 P=4 USE_XOR)
target_link_libraries(matmul_gemm_wmma_64x32_p4_xor_hip PRIVATE matmul_common_iface)

add_library(matmul_gemm_wmma_64x64_p4_xor_hip OBJECT impl_gemm_wmma_64x64_p4_xor.hip)
target_compile_definitions(matmul_gemm_wmma_64x64_p4_xor_hip PRIVATE TILE_M=64 TILE_N=32 P=4 USE_XOR)
# similarly 128x32
```

Each must pass: `hipcc --offload-arch=gfx1100 -c -Rpass-analysis` → VGPR ≤64, `llvm-objdump --mcpu=gfx1100 | grep v_wmma`.

### 3. Bench wiring: --variant maps to object

In `bench_gemm_wmma.cpp`, replace synthetic jitter with real dispatch:

```cpp
if (variant=="64x32_P4_XOR") hipErr = gemm_iq4xs_wmma_64x32_p4_xor_gpu(...);
else if (variant=="64x64_P4_XOR") hipErr = gemm_iq4xs_wmma_64x64_p4_xor_gpu(...);
```

Each variant’s `gemm_iq4xs_wmma_*_gpu` is the same signature as `gemm_iq4xs_wmma_stream_gpu` but compiled with different `BLOCK_N/BLOCK_M/P`.

### 4. Verification per variant (must be run on gfx1100 bare-metal)

```
hipcc --offload-arch=gfx1100 -c impl_gemm_wmma_64x32_p4_xor.hip -o /tmp/p4.o -Rpass-analysis 2>&1 | grep VGPR
llvm-objdump --mcpu=gfx1100 /tmp/p4.o | grep v_wmma | head
./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json --variant 64x32_P4_XOR > /tmp/p4.json
```

Check `sched_barrier 0x0080` (DS) pinned before `ds_write` and `0x0008` before `v_wmma` via `llvm-objdump -d | grep s_barrier`.

### 5. What NOT to do (scope containment)

- Do **not** rewrite WMMA algorithm or dequant (keep `dl*(ls-32)*kvalues` + `v16f16 a_frag` + `sB` load).
- Do **not** add new Python/C++ harness beyond `--variant` string mapping.
- Do **not** change `bench`’s 5-variant table shape; keep same JSON fields `tile/P/banking`.
- LUT `impl_gemm_lut_iq4xs.hip` stays separate; not merged into this P-sweep.

### 6. Residual risk

- `128x32` (128 blocks at M=8192) may need `__launch_bounds__(512,2)` or 2-block tiling; if `BLOCK_N=128` exceeds `256` threads, fallback to `128x32` via `BLOCK_N=64` with `grid_x` doubled (128 blocks = `grid_x = N/64 =80` at N=5120, not 128; so `128x32` label is `BLOCK_M=32` not `BLOCK_N=128`; clarify naming to `128x32` = `M=128` tier not tile).
- `P=4` quad-buffer `sB[4][32][32]` doubles LDS to `8KB` vs `4KB`; still <32KB limit but VGPR pressure +1; verify `rocprof lds_bank_conflict 0` on bare-metal.

## Acceptance Check

- `grep -n "sB\[4\]\[32\]\[32\]" impl_gemm_wmma_64x32_p4_xor.hip` → 1 line (real, not comment)
- `cmake --build kernels/build --target matmul_gemm_wmma_64x32_p4_xor_hip` → `hipcc` clean
- `./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json --variant 64x32_P4_XOR | python3 -m json.tool` → valid JSON with `variant` field, `wmma_median_us` distinct from P2

---
*Saved as proposal, no source widening beyond variant objects.*
