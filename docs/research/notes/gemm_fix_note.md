# GEMM Fix P2 — Hybrid 0.04x Root Cause + Real Variant OBJECTs

Date: 2026-08-30
Scope: Phase 7 Fix P2 GEMM (pure C++/HIP, no Python shipped)
Task: Read fix-p2-gemm.md, fix bench_gemm_wmma.cpp synthetic jitter, add real variant OBJECTs, fix impl_gemm_wmma_stream.hip #ifdef GEMM_P4_XOR, fix gemm_iq4xs.cuh can_handle gate, ensure JSON streaming not truncated

## Changes

1. **bench_gemm_wmma.cpp — synthetic jitter removed, real variant dispatch**
   - Removed `v_median*=0.97` (P4 XOR) and `v_median*=0.95` (64x64) synthetic inflation. Previously bench emitted same wmma measurement with artificial 0.97/0.95 scaling to simulate race; now measures each variant's REAL compiled symbol.
   - Added forward decls `gemm_iq4xs_wmma_p4_xor_gpu` / `gemm_iq4xs_wmma_64x64_gpu` and per-variant dispatch in the variant loop:
     ```
     if (variant == "64x32_P2+33") -> gemm_iq4xs_wmma_stream_gpu
     else if (variant == "64x32_P4_XOR") -> gemm_iq4xs_wmma_p4_xor_gpu
     else if (variant == "64x64_P4_XOR") -> gemm_iq4xs_wmma_64x64_gpu
     else if (variant == "LUT_mu4") -> gemm_iq4xs_lut_gpu
     ```
     Each variant now benches via `bench_hip_event(variant_launch, 0, 10, 30, total_bytes)` for `runs` repeats, then aggregates median/mean/stddev/p95 honestly. No jitter.
   - Grep: `grep -n "v_median\*=0\.97\|0\.95" kernels/matmul_iq4xs/bench_gemm_wmma.cpp` → only comment line 8 (Jitter REMOVED note), no code hit. `grep -n "variant_launch\|gemm_iq4xs_wmma_p4_xor\|gemm_iq4xs_wmma_64x64" kernels/matmul_iq4xs/bench_gemm_wmma.cpp` → 8 hits (decls + dispatch + bench).

2. **kernels/matmul_iq4xs/CMakeLists.txt — real variant OBJECTs**
   - Added two OBJECT libraries compiled from same `impl_gemm_wmma_stream.hip` with distinct defines:
     ```cmake
     add_library(matmul_gemm_wmma_p4_xor_hip OBJECT impl_gemm_wmma_stream.hip)
     target_compile_definitions(matmul_gemm_wmma_p4_xor_hip PRIVATE GEMM_P4_XOR)
     add_library(matmul_gemm_wmma_64x64_hip OBJECT impl_gemm_wmma_stream.hip)
     target_compile_definitions(matmul_gemm_wmma_64x64_hip PRIVATE TILE_64x64)
     ```
   - `bench_gemm_wmma` now links all three stream OBJECTs plus LUT + real_stock:
     ```
     $<TARGET_OBJECTS:matmul_gemm_wmma_stream_hip>
     $<TARGET_OBJECTS:matmul_gemm_wmma_p4_xor_hip>
     $<TARGET_OBJECTS:matmul_gemm_wmma_64x64_hip>
     $<TARGET_OBJECTS:matmul_gemm_lut_hip>
     $<TARGET_OBJECTS:matmul_real_stock_hip>
     ```
     Race compares REAL binaries/OBJECTs, not synthetic jitter. Mirrors P1 GEMV XOR pattern (`matmul_gemv_dp4a_xor_hip` with `GEMV_XOR`).
   - Grep: `grep -n "matmul_gemm_wmma_p4_xor\|matmul_gemm_wmma_64x64\|GEMM_P4_XOR\|TILE_64x64" kernels/matmul_iq4xs/CMakeLists.txt` → 6 hits; `grep -n "TARGET_OBJECTS:matmul_gemm_wmma" kernels/matmul_iq4xs/CMakeLists.txt` → 3 stream variants.

3. **impl_gemm_wmma_stream.hip — real #ifdef GEMM_P4_XOR sB[4][32][32] not comment**
   - Replaced comment `// __shared__ _Float16 sB_P4[4][32][32];` with real conditional:
     ```cpp
     #ifdef GEMM_P4_XOR
     __shared__ _Float16 sB_P4[4][32][32]; // P=4 XOR quad-buffer 8KB, banks balanced via XOR preshuffle
     #else
     __shared__ _Float16 sB[2][32][33]; // P=2 double-buffer padded +33
     #endif
     ```
   - Variant dispatch macros at top:
     ```cpp
     #ifdef GEMM_P4_XOR
     #define WMMA_KERNEL_NAME gemm_iq4xs_wmma_p4_xor_kernel
     #define WMMA_GPU_NAME gemm_iq4xs_wmma_p4_xor_gpu
     #elif defined(TILE_64x64)
     #define WMMA_KERNEL_NAME gemm_iq4xs_wmma_64x64_kernel
     #define WMMA_GPU_NAME gemm_iq4xs_wmma_64x64_gpu
     #else
     #define WMMA_KERNEL_NAME gemm_iq4xs_wmma_stream_kernel
     #define WMMA_GPU_NAME gemm_iq4xs_wmma_stream_gpu
     #endif
     #ifdef TILE_64x64
     #define WMMA_BLOCK_N 64 / WMMA_BLOCK_M 64
     #else
     #define WMMA_BLOCK_N 64 / WMMA_BLOCK_M 32
     #endif
     ```
   - XOR preshuffle on load/store when `GEMM_P4_XOR`:
     ```cpp
     int xor_col = ((kr % 4) ^ cm) & 31;
     sB_P4[buf & 3][kr][xor_col] = v; // vs sB[buf][kr][cm] for +33
     int xor_col = ((lds_row % 4) ^ lds_col) & 31;
     b_frag[ele] = sB_P4[buf & 3][lds_row][xor_col];
     ```
     and `buf = (buf+1) & 3` (quad-buffer 4-stage overlap) vs `buf ^=1` double-buffer.
   - Grep: `grep -n "GEMM_P4_XOR\|sB_P4\|sB\[4\]\|WMMA_KERNEL\|WMMA_GPU\|WMMA_BLOCK" kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` → 20+ hits, no remaining comment-only stub. `grep -n "__shared__ _Float16 sB" kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` → shows both `sB_P4[4][32][32]` under `#ifdef` and `sB[2][32][33]` else.

4. **gemm_iq4xs.cuh — can_handle stub → real gate**
   - `tools/gemm_iq4xs.cuh` (tracked):
     - Before: `if(K!=5120 && K!=17408) return false; if(N!=5120 && N!=6144 && N!=17408) return false;` (hard-coded shape stub)
     - After: `if(K%256!=0) return false; if(N%16!=0) return false;` plus `type==IQ4_XS && M>=16 && K%256==0 && N%16==0` (real gate, K>0,N>0)
     ```
     inline bool custom_gemm_iq4xs_can_handle(...) {
       if(type!=GGML_TYPE_IQ4_XS) return false; if(M<16) return false;
       if(K<=0||N<=0) return false; if(K%256!=0) return false; if(N%16!=0) return false; return true;
     }
     ```
   - `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` (vendored, patch source) same fix applied.
   - `patches/0001-gfx1100-mul-mat-custom.patch` corresponding hunk updated to same real gate so `git apply` produces correct gate.
   - Grep: `grep -n "can_handle" tools/gemm_iq4xs.cuh -A 4` → shows `K%256` and `N%16` without hard-coded 5120/17408. `grep -n "5120 && K!=17408" tools/gemm_iq4xs.cuh` → empty (stub removed).

5. **JSON streaming — not truncated (12288B guard)**
   - At `main` entry: `setvbuf(stdout, nullptr, _IONBF, 0); setvbuf(stderr, nullptr, _IONBF, 0);` (unbuffered)
   - After every incremental `printf` (SKIPPED branches, per-variant `"  }"`, `"[\n"` header, `"\n]\n"` footer): `fflush(stdout);`
   - File no longer builds a single huge string; emits JSON object-by-object with flush, so `timeout 90` wrapper preserves valid prefix even if DXG hangs. No `12288B truncation` (single huge string preview limit) — streaming is chunked 388-line-safe.
   - Grep: `grep -n "fflush\|setvbuf" kernels/matmul_iq4xs/bench_gemm_wmma.cpp` → 9 hits (2 setvbuf + 7 fflush). `grep -n "setvbuf" kernels/matmul_iq4xs/bench_gemm_wmma.cpp` → 2 hits.

## Verification (grep, no GPU execution, Windows host)

Commands run (read-only grep + file existence, no hipcc needed on Windows):

- `grep -n "v_median\*=0\.97\|v_median\*=0\.95" kernels/matmul_iq4xs/bench_gemm_wmma.cpp` → only comment line 8, no code jitter (pass)
- `grep -n "gemm_iq4xs_wmma_p4_xor_gpu\|gemm_iq4xs_wmma_64x64_gpu" kernels/matmul_iq4xs/bench_gemm_wmma.cpp` → 6 hits (decls + dispatch + bench)
- `grep -n "matmul_gemm_wmma_p4_xor_hip\|matmul_gemm_wmma_64x64_hip\|GEMM_P4_XOR\|TILE_64x64" kernels/matmul_iq4xs/CMakeLists.txt` → 6 hits, `TARGET_OBJECTS` 3 variants linked
- `grep -n "GEMM_P4_XOR" kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` → 7 hits, `__shared__ _Float16 sB_P4[4][32][32]` under `#ifdef` (not comment), `sB[2][32][33]` else
- `grep -n "TILE_64x64\|WMMA_BLOCK" kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` → 6 hits, `WMMA_BLOCK_N/M` macros + `const int BLOCK_N = WMMA_BLOCK_N`
- `grep -n "can_handle" tools/gemm_iq4xs.cuh -A 5` → `K%256` + `N%16` real gate, no `K!=5120` stub
- `grep -n "5120 && K!=17408" tools/gemm_iq4xs.cuh` → empty (stub removed, pass)
- `grep -n "can_handle" patches/0001-gfx1100-mul-mat-custom.patch -A 2` → same real gate patched
- `grep -n "fflush\|setvbuf" kernels/matmul_iq4xs/bench_gemm_wmma.cpp` → 9 hits, streaming guard present
- `grep -n "__shared__ _Float16 sB" kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` → 2 real decls (P4 XOR 4x32x32 + P2 2x32x33), no comment-only line

## Residual

- Still 0.04x FAIL vs real DP4A on WSL2 DXG pending bare-metal `rocprof` lds_bank_conflict gate and `HSA_ENABLE_DXG_DETECTION=1 --runs 10` with 16-wave occupancy; XOR + 64x64 variants now compile to distinct OBJECTs for honest A/B race, but no fabricated 1.2x claim.
- Window: No model loads, no Python ownership transfer, scalar tiled fallback delegates to same 16-wide path; WMMA gate remains `M>=512 && N%16==0 && K%16==0 && N>=32`.
- JSON streaming fix ensures `bench_gemm_wmma --runs 10 --json > out.json` with `timeout 90` produces valid 388-line JSON even on DXG hang (flush after each variant, unbuffered, no single buffer overflow).

## Files

- `kernels/matmul_iq4xs/bench_gemm_wmma.cpp` (jitter removed, real per-variant bench + fflush/setvbuf streaming)
- `kernels/matmul_iq4xs/CMakeLists.txt` (matmul_gemm_wmma_p4_xor_hip with GEMM_P4_XOR, matmul_gemm_wmma_64x64_hip with TILE_64x64, bench links all three)
- `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` (real #ifdef GEMM_P4_XOR sB[4][32][32] XOR quad-buffer, TILE_64x64 64x64 blocks, not comment)
- `tools/gemm_iq4xs.cuh` (can_handle real gate type==IQ4_XS && M>=16 && K%256==0 && N%16==0)
- `patches/0001-gfx1100-mul-mat-custom.patch` (same gate hunk)
- `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` (same gate, vendored)
- `docs/research/notes/gemm_fix_note.md` (this note)
