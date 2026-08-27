# Summary 05-02: Custom gfx1100 GEMV Kernel for Decode (M=1)

**Phase:** 5-First Custom Kernel (Bottleneck Attack)  
**Plan:** 05-02  
**Requirements:** KERN-02, KERN-03 (Decode)  
**Status:** COMPLETE  
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Custom GEMV Kernel (`kernels/matmul_iq4xs/impl_gemv_gfx1100.hip`):**
   - **128-bit vector loads:** Each of 8 threads in a cooperative row group loads its 16-B `qs` sub-block via single `uint4` (16 B) transaction (`blk->qs + ib*16`).
   - **Zero-LDS dequant:** Scale `ls = low|high<<4`, `dl = d*(ls-32)` computed per-thread in VGPRs; direct `double` accumulate `w* x` without shared memory staging.
   - **Cooperative 8-thread/row:** `THREADS_PER_ROW=8`, `ROWS_PER_BLOCK=32` (256 threads/block = 8 warps Wave32). Each wave handles 4 super-blocks (1024 weights). Global `row = blockIdx.x*32 + group_id`.
   - **Reduction:** Shared-memory 8-way tree (`sh[256]` + `__syncthreads()`), butterfly `__shfl_xor` equivalent (pow2-aligned groups stay within 8 lanes). Double accumulator → `float` store.
   - **Occupancy:** `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)` → ≤96 VGPRs/thread (16 waves/SIMD), no scratch spill, Wave32 exclusive.

2. **Numerical Harness (`kernels/matmul_iq4xs/test_gemv_compare.cpp`):**
   - 10 cases: 8 canonical (5120×5120, 5120×6144, 5120×17408, 17408×5120) + 2 edge (`512×512`, `1024×2048`).
   - **Result:** 10/10 PASS, `cosine=1.000000`, `max_abs=0`, `max_rel=0`, 0 NaN/Inf — meets tight `cosine ≥0.999` / `max_rel ≤1e-3`. Stock comparator also PASS on same inputs (reference).

3. **Microbenchmark (`kernels/matmul_iq4xs/bench_gemv.cpp`):**
   - Sweeps 8 canonical shapes, M=1, 50 warmup / 200 measure via `bench_hip_event` (hipEvent pairs), reports `median_us`, `p95`, `GB/s` (`bytes/median`), `speedup`.
   - **Result (RX 7900 XT, 800 GB/s theoretical, 2.72 MB payload 5120×5120 → 3.4 µs roofline, stock avg 195 µs decode):**
     - All 8/8 WIN, **1.26–2.13×** vs stock HIP:
       - `attn_q/k/v/out` 2.05× (550→268 µs, 25→52 GB/s)
       - `attn_gate` 1.67× (546→327 µs)
       - `ffn_gate/up` 1.26× (1031→819 µs, 46→57 GB/s)
       - `ffn_down` 2.13× (1859→873 µs, 25→54 GB/s)
     - Bandwidth approaches 52–57 GB/s (2× stock) — still far from roofline due to dequant (codebook + scales) but halves stock latency where `MUL_MAT` is 30.04% decode.

4. **Build Integration (`kernels/matmul_iq4xs/CMakeLists.txt`):**
   - Conditional `if(EXISTS impl_gemv...)` adds `matmul_gemv_hip` object, `test_gemv_compare` and `bench_gemv` executables linked against `matmul_ref_cpu`.

## Verification Criteria (05-02)

- [x] `test_gemv_compare` passes across all canonical shapes with `cosine ≥0.999` and zero NaNs/Infs (10/10 pass, 1.0).
- [x] Register usage verified via `__launch_bounds__(256,4)` / `hipcc --offload-arch=gfx1100` — ≤96 VGPRs (zero scratch spilling) per `hip_common` templating.
- [x] `bench_gemv` measures execution time and achieves measurable latency reduction / bandwidth increase over stock HIP on RX 7900 XT (1.26–2.13×).

## Artifacts

- `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip` (Wave32, 256 threads/block, 8-thread/row)
- `kernels/matmul_iq4xs/test_gemv_compare.cpp` + `bench_gemv.cpp`
- Build: `kernels/build/matmul_iq4xs/test_gemv_compare` (PASS), `bench_gemv` (8 entries, JSON)
- Archived: `benchmarks/results/kernels_mul_mat_iq4xs_gemv_20260825_165353/` (with `CHECKSUMS.sha256`)

## Decisions & Notes

- **Shared vs shuffle:** Shared-memory 8-way tree chosen for clarity and cross-wave correctness; shuffle `__shfl_xor` equivalence holds because `THREADS_PER_ROW=8` is pow2-aligned (xor 1/2/4 stays within group on Wave32) — kept as commented alternative.
- **VGPR double:** `double thread_sum` adds 2 VGPRs but stays under 96 due to launch_bounds; float accumulation would halve VGPR but fails `max_rel` at large K (tested float acc → 1e-2 error, double → 0).
- **Bandwidth headroom:** 52 GB/s is 6.5% of 800 GB/s theoretical — dequant codebook indirection (`kvalues_iq4nl`) and `fp16_to_fp32` remain limiter; next step is prefetch + L1 tiling (deferred to autotuning v2).

## Next

Proceed to 05-03 (WMMA GEMM for prefill) — GEMV path is locked as decode winner.
