# Phase 5: First Custom Kernel (Bottleneck Attack) - Context

**Gathered:** 2026-08-25
**Status:** In Planning
**Target Hardware:** AMD Radeon RX 7900 XT (`gfx1100`, RDNA3, 20 GiB VRAM)
**Target Model:** `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (sha256 `53adc4bb…`)
**Target Operation:** `MUL_MAT` (Quantized IQ4_XS Matrix Multiplication, 31.12% cumulative GPU wall time)

<domain>
## Phase Boundary

Author and validate a custom gfx1100 kernel attacking **Optimization Target #1: `MUL_MAT` (IQ4_XS)**, the single largest bottleneck identified during Phase 3 profiling. The kernel must be numerically correct against the CPU reference, beat the stock `llama.cpp` HIP kernel in shape-swept microbenchmarks (with the stock-Vulkan comparator recorded alongside), and survive end-to-end integration into the full model.

Requirements in scope: **KERN-02** and **KERN-03**.

**Success Criteria (binding):**
1. **KERN-02:** The implemented kernel targets entry #1 of the Phase 3 bottleneck table (`MUL_MAT` on IQ4_XS) and passes numerical comparison against the CPU reference within agreed tolerances ($\text{cosine} \ge 0.999$, $\text{max\_rel} \le 10^{-3}$).
2. **KERN-03 (Microbenchmark):** In head-to-head microbenchmarks across canonical shapes — with prefill ($M \gg 1$) and decode ($M = 1$) measured and reported separately — the custom kernel beats stock `llama.cpp` HIP on gfx1100, with stock Vulkan reported alongside.
3. **KERN-03 (End-to-End A/B):** An end-to-end A/B through the real model (provisional wiring) shows the speedup survives full runtime execution with `QUAL-01` (Op-gate: 0 errors) and `QUAL-02` (Model-gate: WikiText-2 PPL within $\pm 1\%$ of $6.4271$) green.
4. **Publish Failures:** Lost shapes, regressed variants, and sub-optimal tuning attempts are recorded in the results store beside the wins (project rule #10).
</domain>

<decisions>
## Implementation Decisions

### 1. Separate Prefill ($M \gg 1$) and Decode ($M = 1$) Paths (Rule #4)
- **Decode Path (GEMV, $M=1$):** Memory-bandwidth bound. Target optimization: 128-bit aligned global loads (`uint4`), zero-LDS direct register accumulation, Wave32 butterfly shuffle reduction (`__shfl_xor`), `__launch_bounds__(256, 4)`.
- **Prefill Path (GEMM, $M \ge 16$):** Compute and tensor-throughput bound. Target optimization: RDNA3 hardware matrix cores via `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (16×16×16 FP16→FP32 in 32 cycles on Wave32), on-the-fly fragment dequantization into registers, LDS double-buffering with padded stride `[32][33]` to eliminate bank conflicts.

### 2. Canonical Matrix Dimensions in Qwen3.8-27B
- **Attention Projections:**
  - $Q, K, V$ Projections: $[K=5120, N=5120]$ and $[K=5120, N=6144]$
  - Output Projection: $[K=5120, N=5120]$
- **Feed-Forward Network (FFN) Projections:**
  - Gate / Up Projections: $[K=5120, N=17408]$
  - Down Projection: $[K=17408, N=5120]$

### 3. Numerical Tolerance Gate
- **Microbenchmark Gate (Standalone):**
  - $\text{cosine similarity} \ge 0.999$ (target $\ge 0.9999$).
  - $\text{max\_rel} \le 1.0 \times 10^{-3}$ (for values $|y_{\text{ref}}| > 10^{-3}$).
  - Zero NaNs or Infs allowed.
- **Model Quality Gate (Full Model):**
  - `QUAL-01`: `test-backend-ops test -b ROCm0` passes with 0 errors.
  - `QUAL-02`: WikiText-2 PPL within $\pm 1\%$ interval $[6.3628, 6.4914]$ of reference $6.4271$ + 6/6 golden canaries.

### 4. RDNA3 Microarchitectural Budget
- **Wavefront Size:** Wave32 exclusively (`-mwavefrontsize32`).
- **VGPR Limit:** $\le 96$ registers per thread (guarantees 16 wavefronts per SIMD / full occupancy).
- **LDS Allocation:** $\le 32$ KiB per threadblock.
- **Launch Bounds:** Explicit `__launch_bounds__(256, 4)` and `__attribute__((amdgpu_flat_work_group_size(256, 256)))`.
</decisions>

<canonical_refs>
## Canonical References

- `docs/ROADMAP.md` §Phase 5 — KERN-02, KERN-03 authoritative text and constraints.
- `docs/REQUIREMENTS.md` — Authoritative requirements traceability.
- `benchmarks/profiling/BOTTLENECK-TABLE.md` — Designation of `MUL_MAT` as Target #1 (31.12% total GPU time, 50.89% prefill, 30.04% decode).
- `benchmarks/profiling/dispatch_overhead_report.md` — HIP graph batching (+19% decode throughput) and kernel time isolation.
- `docs/reference/GPU-KERNEL-RESOURCES.md` — AMD RDNA3 ISA (70650), WMMA intrinsics, and GPUOpen tuning guides.
- `ggml/src/ggml-cuda/vecdotq.cuh` & `mmvq.cu` — Stock `llama.cpp` HIP implementation for `vec_dot_iq4_xs_q8_1`.
- `kernels/common/block_iq4_xs.h` — Vendored 136-byte IQ4_XS data layout from Phase 4.
</canonical_refs>
