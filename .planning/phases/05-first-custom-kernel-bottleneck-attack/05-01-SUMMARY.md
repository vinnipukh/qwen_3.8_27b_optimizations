# Summary 05-01: Target Study, Dimension Mapping & Stock Reference Extraction

**Phase:** 5-First Custom Kernel (Bottleneck Attack)  
**Plan:** 05-01  
**Requirements:** KERN-02, KERN-03 (pre-kernel gate)  
**Status:** COMPLETE  
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Canonical Dimension Map (`kernels/matmul_iq4xs/ref_cpu.h`):**
   - Codified 8 Qwen3.8-27B projection shapes covering both attention and FFN paths:
     `attn_q/k/v` 5120×5120, `attn_gate` 5120×6144, `attn_out` 5120×5120, `ffn_gate/up` 5120×17408, `ffn_down` 17408×5120.
   - Verified against live GGUF tensor probes (`blk.0.ffn_gate 5120×17408 IQ4_XS 17408×2720`, `blk.0.ffn_down 17408×5120 IQ4_XS 5120×9248`, `blk.0.attn_gate 5120×6144`, `blk.3.attn_q 5120×12288` etc) — mappings handle fused `qkv` fallback via row/column slicing.
   - Gated by `BOTTLENECK-TABLE.md`: MUL_MAT #1 = 31.12% cumulative (50.89% prefill, 30.04% decode) — target study confirms IQ4_XS GEMV (M=1) and GEMM (M≫1) split per Rule #4.

2. **FP64 CPU Golden Oracle (`kernels/matmul_iq4xs/ref_cpu.cpp` + `ref_cpu.h`):**
   - Pure C++17, zero HIP, double-accumulated `gemv_iq4xs_cpu_ref` and `gemm_iq4xs_cpu_ref` plus `dequant_mat_iq4xs_cpu` helper.
   - Replicates `kvalues_iq4nl[16]` codebook, 6-bit scale `ls = low|high<<4`, `dl = d*(ls-32)`, split-half `qs` layout exactly as `block_iq4_xs.h` (136B, QK_K=256).
   - **Verification:** `libmatmul_ref_cpu.a` builds as CXX static lib without HIP (`CMAKE` `LINKER_LANGUAGE CXX`); synthetic and real GGUF tensors dequantize to `gguf-py` `dequantize()` within `cosine=1.0`.

3. **Fixture Extractor (`tools/dump_matmul_fixtures.py`):**
   - Generates Gaussian activations `x~N(0,1)` seed 42 and extracts contiguous `W_raw` slices from `models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (gguf-py `GGUFReader`, `GGMLQuantizationType.IQ4_XS`).
   - Emits 32 fixtures (8 shapes × M {1,16,128,512}) as `.npz` (`W_raw`, `x`/`X`, `y_ref`/`Y_ref`) + `.bin` mirrors + `manifest_matmul.json` with `W_sha256`, source tensor, `K/N/M`, seed.
   - **Verification:** `python tools/dump_matmul_fixtures.py --model ... --out kernels/fixtures` produced 64 `.npz` (including `W.npz`) + 96 `.bin` + `manifest_matmul.json` (32 entries). Required checks `ffn_gate 5120×17408` and `ffn_down 17408×5120` FOUND. Sizes verified: `W` 14–46 MB, `Y_ref` 20 KB–34 MB.

4. **Stock HIP Comparator (`kernels/matmul_iq4xs/stock_hip_comparator.hip`):**
   - Naive per-row (GEMV) and per-element (GEMM) kernels looping over `K/QK_K` blocks with scalar dequant (double accumulate) — faithful recreation of `vec_dot_iq4_xs_q8_1` logic without Q8 quant overhead, compiled as standalone HIP object.
   - `gemv_iq4xs_stock_gpu` (256 threads/block, `grid=(N+255)/256`) and `gemm_iq4xs_stock_gpu` (`grid=(N*M+255)/256`).

5. **CMake Scaffold (`kernels/matmul_iq4xs/CMakeLists.txt` + top-level `kernels/CMakeLists.txt`):**
   - `matmul_ref_cpu` static lib, `matmul_stock_hip` object, `matmul_test_baseline` executable alias `matmul_stock_test`.
   - Top-level `if(EXISTS .../matmul_iq4xs/CMakeLists.txt) add_subdirectory(matmul_iq4xs)`.

6. **Baseline Validation (`kernels/build/matmul_iq4xs/matmul_test_baseline`):**
   - Tests 8 GEMV canonical shapes + 8 GEMM truncated shapes (M {16,32,64,128}) vs CPU oracle, 16 cases.
   - **Result:** 16/16 PASS, `cosine=1.000000`, `max_abs=0`, `max_rel=0`, 0 NaN/Inf — satisfies `cosine ≥0.999` gate. Stock build `cmake --build kernels/build` green.

## Verification Criteria (05-01)

- [x] `ref_cpu.cpp` compiles without HIP and produces bit-exact reference outputs for mock blocks.
- [x] `tools/dump_matmul_fixtures.py` extracts canonical shapes (5120×5120 and 5120×17408) into `kernels/fixtures/matmul_*.npz`.
- [x] Stock HIP comparator compiles in standalone playground and matches CPU oracle within `cosine ≥0.999`.

## Artifacts

- `kernels/matmul_iq4xs/ref_cpu.h` / `ref_cpu.cpp`
- `kernels/matmul_iq4xs/stock_hip_comparator.hip`
- `tools/dump_matmul_fixtures.py`
- `kernels/matmul_iq4xs/CMakeLists.txt`
- `kernels/fixtures/matmul_*` (32× `.npz`, 96× `.bin`, `manifest_matmul.json`)
- `kernels/build/matmul_iq4xs/matmul_test_baseline` — 16/16 PASS log

## Decisions & Notes

- **Synthetic fallback for fused qkv:** `attn_q/k/v` use `blk.3.attn_q` / `blk.0.attn_gate` when `qkv` Q5_K not IQ4_XS; handled via `TENSOR_CANDIDATES` map with row+K slicing — still exercises canonical 5120×5120 GEMV.
- **GEMM CPU fast path:** Python fixture generator uses `gguf.quants.dequantize` + NumPy `W_mat @ X` (BLAS) for 32 fixtures; C++ test baseline validates truncated N to keep naive double-loop under 1s per case (16 cases, total <10s).
- **KERN-03 half achieved:** Baseline comparator archived for shape-swept diff in 05-04.

## Next

Proceed to 05-02 (custom GEMV) and 05-03 (custom WMMA GEMM) — both depend on this harness and reuse its fixtures/oracle.
