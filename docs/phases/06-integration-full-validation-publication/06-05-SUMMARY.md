# Summary 06-05: Thermos Review Remediation (Pre-Publication Gates)

**Phase:** 6-Integration, Full Validation & Publication
**Plan:** 06-05
**Status:** COMPLETE
**Date:** 2026-08-25

---

## What Was Accomplished

1. **Repo & Build Hygiene (Part A):**
   - Updated `.gitignore` to ignore all binary fixtures in `kernels/fixtures/*` while tracking `manifest*.json`, plus added patterns for compiler probe files (`/test_wmma*`, `*.bc`, `*.hipi`, `*.hipfb`, `*.out.resolution.txt`, `benchmarks/results/tmp_bench_*`).
   - Removed stray `test_wmma*` intermediate probe files from the repository root.
   - Updated `scripts/check_no_ggml.sh` to defensively exclude `kernels/build/` from grep inspections.
   - Hardened `tools/dump_matmul_fixtures.py` with deterministic SHA256-derived RNG seeds, removed non-Qwen leftovers from candidate tensor search, and deleted pass stubs/dead code.
   - Standardized manifest names (`manifest_dequant.json` and `manifest_matmul.json`) and verified `benchmarks/tests/test_fixture.py`.
   - Updated `benchmarks/tools/run_kernel_bench.py` fallback exception handling to mark unknown fields `"unknown"` instead of fabricating hardware provenance.
   - Streamlined `kernels/matmul_iq4xs/CMakeLists.txt` by removing conditional target wrappers and collapsing repeated include/link declarations into an INTERFACE library (`matmul_common_iface`).

2. **Kernel Guard Hardening & Validation (Part B):**
   - **Barrier Divergence (M-1):** In `impl_gemv_gfx1100.hip` and `impl_gemm_wmma.hip`, replaced early-return statements preceding barriers with thread masking so all threads in workgroups participate uniformly in `__syncthreads()`.
   - **Aligned Loads (M-2):** Replaced unaligned 16-byte `uint4*` pointer casting with 8-byte aligned `uint64_t` pairs (guaranteed 8-byte aligned at `offset 8 + ib*16`).
   - **WMMA Fragment Layout & LDS Bounds (H-1):** Implemented correct RDNA3 Wave32 WMMA 16x16x16 lane and fragment layout (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`), masked inactive threads, eliminated uninitialized LDS reads, and added the WMMA gate-passing test case (`wmma_gate_pass_5120_1024_512`).
   - **Shape & Dimension Guards (L-1/NIT):** Enforced `K % QK_K == 0` validation and error handling for invalid grid configurations.
   - **Deduplicated Harness:** Extracted `kernels/common/matmul_test_util.h` containing shared metric computation (`compute_metrics`) and deterministic synthetic weight generation (`gen_iq4xs_weights`).

3. **Verification Results:**
   - `matmul_test_baseline`: **16/16 PASS** (`cosine=1.000000`, `max_rel=0.00e+00`)
   - `test_gemv_compare`: **10/10 PASS** (`cosine=1.000000`, `max_rel=0.00e+00`)
   - `test_gemm_compare`: **11/11 PASS** (`cosine=1.000000` on all cases including `wmma_gate_pass_5120_1024_512`)
   - `scripts/check_no_ggml.sh`: **PASS** (zero ggml/llama includes in `kernels/`)
   - `pytest benchmarks/tests/`: **55/55 PASS**

## Artifacts

- `kernels/common/matmul_test_util.h`
- `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip`
- `kernels/matmul_iq4xs/impl_gemm_wmma.hip`
- `kernels/matmul_iq4xs/CMakeLists.txt`
- `kernels/fixtures/manifest_dequant.json`
- `benchmarks/results/phase3/model_gate_probe_chunks2.json`
- `tools/dump_matmul_fixtures.py`
- `.gitignore`
