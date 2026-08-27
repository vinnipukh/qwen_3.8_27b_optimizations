# Phase 5: Technical Research — Custom IQ4_XS GEMV & WMMA GEMM on gfx1100

**Gathered:** 2026-08-25
**Hardware Target:** AMD Radeon RX 7900 XT (`gfx1100`, RDNA3, Navi 31)

---

## 1. Stock `llama.cpp` HIP Implementation Analysis

### How Stock Implements `MUL_MAT` for IQ4_XS
In `llama.cpp` commit `bb4caa75`, `MUL_MAT` on quantized models routes through:
* **Decode ($M \le 4$):** `mul_mat_vec_q` in `ggml-cuda/mmvq.cu` using `vec_dot_iq4_xs_q8_1` in `ggml-cuda/vecdotq.cuh`.
* **Prefill ($M \ge 16$):** `mul_mat_q` in `ggml-cuda/mmq.cu`.

### Identified Inefficiencies on RDNA3 (`gfx1100`)
1. **Permutation Bottleneck in Codebook Lookup:**
   In `vecdotq.cuh:34`, lookup in `kvalues_iq4nl[16]` is performed via `get_int_from_table_16`:
   ```cpp
   uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
   uint32_t v_odd_low  = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);
   ```
   Each 4-bit lookup requires masking, shifting, and multiple `v_perm_b32` instructions. On RDNA3, `v_perm_b32` has higher instruction latency than simple vector arithmetic or direct constant broadcast.
2. **Intermediate `Q8_1` Quantization Overhead:**
   Stock MMVQ quantizes the activation vector $x$ to INT8 (`block_q8_1`) on the fly, then uses `dp4a` (`v_dot4_i32_i8`). This introduces:
   - Extra memory passes and quantization latency for activations.
   - Accumulation truncation error compared to direct FP32 accumulation from FP16 weights.
3. **Generic Warp Scheduling:**
   Upstream MMVQ contains dynamic branch ladders checking `table_id == MMVQ_PARAMETERS_RDNA3_0`, with hardcoded workgroup dimensions that leave CUs under-utilized for non-power-of-two matrix shapes like $[5120 \times 17408]$.

---

## 2. Decode Optimization: Vectorized 128-Bit GEMV ($M=1$)

### Memory Bandwidth Roofline on RX 7900 XT
* **Theoretical VRAM Bandwidth:** $800.0\text{ GB/s}$ (320-bit bus @ 20 Gbps GDDR6).
* **Workload Size:** For $K=5120, N=5120$, $W$ contains $20,000$ super-blocks = $2.72\text{ MB}$.
* **Theoretical Lower Bound Latency:**
  $$T_{\text{min}} = \frac{2.72\text{ MB}}{800\text{ GB/s}} = 3.40\text{ }\mu\text{s}$$
* **Stock Latency (Profiled in Phase 3):** $\approx 195.0\text{ }\mu\text{s}$ (including kernel launch & dispatch overhead).

### 128-Bit Vector Load Strategy
To approach the memory roofline:
* Super-block size is 136 bytes:
  $$\text{Payload} = 8 \times 16\text{ B (qs nibbles)} + 8\text{ B (scale headers)} = 136\text{ B}$$
* 8 threads in a Wave32 wavefront collaboratively process 1 super-block:
  * Each thread loads 16 bytes (`uint4` / `v_pk_mov_b32`) of `qs` in a single 128-bit memory transaction.
  * Thread 0 loads the 8-byte scale header (`d`, `scales_h`, `scales_l`) and broadcasts to the other 7 threads via `__shfl_sync` / `__shfl`.
  * All 32 weights of the sub-block are unpacked into VGPRs and multiplied by activation floats $x$.
  * Cross-thread reduction is completed via 3-step butterfly shuffle: `__shfl_xor(sum, 1)`, `__shfl_xor(sum, 2)`, `__shfl_xor(sum, 4)`.

---

## 3. Prefill Optimization: RDNA3 Hardware WMMA GEMM ($M \gg 1$)

### RDNA3 WMMA Microarchitecture
* **Intrinsic:** `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
* **Operation:**
  $$D_{16 \times 16} = A_{16 \times 16} \cdot B_{16 \times 16} + C_{16 \times 16}$$
* **Throughput:** 512 FLOPs per 32 clock cycles on Wave32 (16 FLOPs/cycle/SIMD32).
* **Matrix Fragment Register Layout:**
  Across the 32 threads in a Wave32 wavefront:
  * **Matrix A (16x16 FP16, column-major):** Each thread holds 8 FP16 values (packed as 4 `uint32_t` / `v4i32`).
  * **Matrix B (16x16 FP16, row-major):** Each thread holds 8 FP16 values (packed as 4 `uint32_t` / `v4i32`).
  * **Matrix C/D (16x16 FP32, row-major):** Each thread holds 8 FP32 values (in 8 separate 32-bit registers / `v8f32`).

### On-The-Fly Fragment Dequantization
Rather than executing a separate dequantization pass over the entire matrix into global VRAM, threads unroll 16 weights directly from the 136-byte IQ4_XS block into the 4 packed FP16 register pairs for Matrix A immediately before issuing `__builtin_amdgcn_wmma`.

### LDS Tiling and Bank Conflict Elimination
* RDNA3 Local Data Share (LDS) has **32 banks** of 4-byte (32-bit) words.
* If a tile of $32 \times 32$ FP16 values is laid out row-wise:
  $$\text{Row stride} = 32 \times 2\text{ bytes} = 64\text{ bytes} = 16\text{ banks}$$
  Accessing columns causes 2-way or 4-way bank conflicts.
* **Solution:** Pad the row allocation to `[32][33]` (or 66 bytes per row). Every row start shifts by 1 bank ($17 \pmod{32}$), ensuring all 32 lanes access distinct memory banks simultaneously with zero conflict stalls.

---

## 4. Register Allocation & Occupancy Bounds on gfx1100

| Parameter | Value | Constraint / Target |
|---|---|---|
| VGPRs per SIMD | 1536 | Max hardware capacity |
| Allocation Granularity | 8 registers | Multiples of 8 |
| Max Occupancy (16 waves/SIMD) | $\le 96$ VGPRs | Target limit for custom kernels |
| High Occupancy (12 waves/SIMD) | $\le 128$ VGPRs | Acceptable ceiling if unrolling warrants |
| LDS per CU | 64 KiB | $\le 32$ KiB used per threadblock |
| Launch Bounds | `__launch_bounds__(256, 4)` | 256 threads/block, min 4 blocks/CU |
