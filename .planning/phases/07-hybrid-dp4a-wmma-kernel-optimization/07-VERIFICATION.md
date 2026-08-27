---
phase: 07-hybrid-dp4a-wmma-kernel-optimization
verified: 2026-08-27T19:50:00Z
status: gaps_found
score: 2/5 must-haves verified
behavior_unverified: 1
overrides_applied: 0
gaps:
  - truth: "Cooperative 8-thread DP4A GEMV (impl_gemv_dp4a_gfx1100.hip) achieves >1.2x over real stock vec_dot_iq4_xs_q8_1 DP4A and >38 t/s decode in llama-bench"
    status: failed
    reason: "Microbench vs real stock DP4A peaks at 1.178x (attn_q 94.67us vs 111.47us per 07-02 SUMMARY), average 1.00x across 8 shapes under WSL DXG; summary documents avg 1.00 peak 1.178 not >1.2x. Decode >38 t/s claim not measured — no llama-bench JSON produced on hardware; 07-04 explicitly documents simulation on Windows host without GPU."
    artifacts:
      - path: "kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip"
        issue: "Kernel present and wired but speedup target not met vs real DP4A comparator; needs bare-metal gfx1100 re-bench"
      - path: "benchmarks/results/phase7"
        issue: "Missing paired llama-bench A/B JSON (ab_stock / ab_custom) with pp/tg split; no decode tok/s evidence"
    missing:
      - "Bare-metal WSL2 gfx1100 re-bench: bench_gemv_dp4a median_us + speedup JSON proving >1.2x average or peak median >1.2 across canonical decode shapes"
      - "Paired llama-bench A/B JSON (stock vs custom, -ngl 99, contexts 512/1024/2048/4096, thermal-paired, RunStore + CHECKSUMS) showing decode tok/s >38 and > stock"
  - truth: "WMMA streaming GEMM (impl_gemm_wmma_stream.hip) 64x32 tiling with hardware wmma_f32_16x16x16_f16_w32 achieves >950 t/s prefill in llama-bench"
    status: failed
    reason: "Streaming WMMA kernel exists with correct tiling, double-buffered LDS, builtin present (grep verified), but prefill >950 t/s not measured. 07-03/07-04 document simulation only; bench_gemm_wmma microbench vs real stock not executed on this host (no hipcc/ROCm/GPU), and no llama-bench prefill tok/s JSON exists. Existing 6-7x speedups cited are vs naive stock_hip_comparator, not vs real DP4A MMQ at M>=128."
    artifacts:
      - path: "kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip"
        issue: "Artifact present and substantive (verified grep), but hardware throughput and e2e prefill tok/s unmeasured on this host"
      - path: "benchmarks/results/phase7"
        issue: "Missing: bench_gemm_wmma JSON vs real_stock_dp4a at M=128/512/1024 and paired llama-bench prefill tok/s"
    missing:
      - "HSA_ENABLE_DXG_DETECTION=1 ./bench_gemm_wmma JSON proving >1.2x vs real_stock_dp4a MMQ at M>=128 (spec) on gfx1100"
      - "Paired llama-bench prefill tok/s JSON (M>=128 contexts) showing >950 t/s and > stock, thermal-paired"
  - truth: "Quality gates QUAL-01 (0 errors) and QUAL-02 (PPL 6.4271) remain green on custom build, and thermal pairing discipline enforced with no 90C aborts"
    status: failed
    reason: "Stock op_gate PASS exists (phase6/op_gate_stock_20260827.json 4243 PASS 0 errors), but custom ON build gates were not executed on hardware — 07-04 documents simulated skip (no hipcc/binary/model on Windows host). No phase7 op_gate_custom.json or run_model_gate PPL JSON. Thermal monitor log shows expected WSL fallback (HWiNFO access denied, no hwmon) with polling continues and no kill @90C — not an abort, but documented as no reliable temp source, requiring WSL2 daemon verification."
    artifacts:
      - path: "benchmarks/results/phase6/op_gate_stock_20260827.json"
        issue: "Stock gate PASS verified; custom gate missing"
      - path: "logs/thermal_monitor.log"
        issue: "No 90C aborts (pass), but fallback mode due to HWiNFO SharedMemory access denied — record-don't-control discipline documented, not measured via hwinfo_daemon"
    missing:
      - "HSA_ENABLE_DXG_DETECTION=1 run_op_gate.py on build-custom (GGML_CUDA_ENABLE_CUSTOM_GFX1100=ON) — 0 errors JSON"
      - "run_model_gate.py PPL + 6/6 canaries on build-custom JSON"
      - "Real WSL2 gfx1100 paired-bench thermal log with hwinfo_daemon 1Hz readings and per-row clocks/temps"
behavior_unverified_items:
  - truth: "End-to-end uplift survives full runtime with QUAL gates green (custom decode > stock, prefill > stock)"
    test: "Run paired llama-bench A/B (stock OFF vs custom ON) back-to-back in one thermal window: --single-turn --simple-io --load-mode none -ngl 99 -b 2048 across {512,1024,2048,4096}, capture RunStore JSON + hwinfo_daemon CSV"
    expected: "Custom median decode >38 t/s and > stock; custom prefill >950 t/s and > stock; QUAL-01 0 errors and QUAL-02 PPL within 1% on custom build"
    why_human: "Requires WSL2 gfx1100 hardware, ROCm 7.2.1, model GGUF present, and thermal telemetry — not available on this Windows host (no hipcc, no /opt/rocm, no GPU); summaries declare simulation"
human_verification:
  - test: "Bare-metal microbench vs real stock DP4A"
    expected: "bench_real_stock, bench_gemv_dp4a, bench_gemm_wmma JSON median_us proving GEMV >1.2x and GEMM prefill >1.2x vs real DP4A (not vs naive)"
    why_human: "Needs WSL2 gfx1100 hardware; WSL DXG virtualization adds jitter flattening deltas (avg 1.0 on Windows trace)"
  - test: "Paired llama-bench A/B thermal-paired sweep"
    expected: "pp/tg split tok/s JSON in benchmarks/results/phase7/ab_stock_* and ab_custom_* with CHECKSUMS, hwinfo temps, variance proof"
    why_human: "Simulation-only on this host per 07-04 — real GPUs, model, and hwinfo daemon required"
  - test: "Custom ON build QUAL-01/02 gates"
    expected: "0 errors over 4200+ ops and PPL 6.4271±1% with 6 canaries green on build-custom"
    why_human: "Custom build not compiled/executed here (no hipcc)"
  - test: "VGPR / disasm audit"
    expected: "llvm-objdump --mcpu=gfx1100 shows v_dot4 (sudot4) in gemv and v_wmma_f32_16x16x16_f16 in gemm; hipcc --save-temps -Rpass-analysis reports <=64 VGPRs"
    why_human: "Needs gfx1100 build artifacts not produced on this Windows host (07-03 build marked not-run)"
---

# Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization Verification Report

**Phase Goal:** Fuse Q8_1 integer activation quantization and RDNA3 hardware matrix cores (v_dot4_i32_i8 / v_wmma) with Wave32 cooperative workgroups to outperform real production stock llama.cpp end-to-end in llama-bench.
**Verified:** 2026-08-27T19:50:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Real-stock DP4A comparator in kernels/matmul_iq4xs/ implements exact vec_dot_iq4_xs_q8_1 + quantize_row_q8_1 via ggml_cuda_dp4a / __builtin_amdgcn_perm, not naive float | ✓ VERIFIED | real_stock_dp4a_comparator.hip: grep ggml_cuda_dp4a_real + __builtin_amdgcn_sudot4 (L68,L73) and __builtin_amdgcn_perm x6 (L111-L118), vec_dot_iq4_xs_q8_1_device (L145) with sumi=ggml_cuda_dp4a, ls decode + d=half2float*low2float; BASELINE_DP4A.md baseline_dp4a.json 8 shapes 84us vs 540us 6.4x; test_real_stock_compare cosine 0.999985 PASS (summary) |
| 2 | Cooperative 8-thread DP4A GEMV (impl_gemv_dp4a_gfx1100.hip) achieves >1.2x over real stock DP4A and >38 t/s decode in llama-bench | ✗ FAILED | Present + wired (grep below) but speedup target not met: 07-02 bench_gemv_dp4a JSON peak 1.178x (attn_q 111.47→94.67us), avg 1.00 across 8 shapes under WSL DXG; >38 t/s decode never measured — 07-04 declares simulation, no benchmarks/results/phase7 llama-bench JSON. Kernel correct (10/10 PASS cos 0.999985 coop/stock 1.000) but throughput claim failed |
| 3 | WMMA streaming GEMM (impl_gemm_wmma_stream.hip) 64x32 tiling, double-buffered [2][32][33] LDS, wmma_f32_16x16x16_f16_w32 achieves >950 t/s prefill | ✗ FAILED | Present + substantive (grep verified) but prefill tok/s never measured. 07-03 bench_gemm_wmma vs real_stock not run on this host (no hipcc); existing 6-7x speedups in KERNEL-BENCH-DIFF.md §3 are vs naive scalar, not vs real DP4A MMQ. No benchmarks/results/phase7 prefill JSON; bare-metal execution required |
| 4 | Quilt patch patches/0001-gfx1100-mul-mat-custom.patch updated with GGML_CUDA_ENABLE_CUSTOM_GFX1100 OFF/ON gating intact, real git diff, git apply --check passes | ✓ VERIFIED | Patch 355 lines / 276 insertions over bb4caa75 (git log 5c6b397), grep GGML_CUDA_ENABLE_CUSTOM_GFX1100 in patch (+OFF default, +#if guards in mmq/mvq/cmakes), empty.cuh fallback preserved, LDS [32][33] + launch_bounds survive vendoring (grep in cuh), 07-04 reports git apply --check PASS via stash test |
| 5 | QUAL-01 op-gate (0 errors) and QUAL-02 model-gate (PPL 6.4271) remain green on custom build; thermal pairing discipline with no 90C aborts | ✗ FAILED | Stock op_gate PASS verified: benchmarks/results/phase6/op_gate_stock_20260827.json (4243 supported, 0 errors 06-03/07-04). Custom ON gates missing — 07-04 documents Windows host simulation skip (no hipcc/model). KERNEL-BENCH-DIFF §8 notes intended phase7 ab dirs not created (ls benchmarks/results/phase7 → missing). Thermal log shows no kills (pass on abort) but fallback polling due to WinError 5 access denied — hwinfo_daemon not exercised |

**Score:** 2/5 truths verified (2 present, behavior-unverified pending hardware)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` | True upstream DP4A pipeline (quantize+vec_dot) | ✓ VERIFIED | 25156 bytes, standalone HIP zero ggml headers, ggml_cuda_dp4a + perm present, quantize_row_q8_1_standalone + GEMV/MMQ kernels, check_no_ggml PASS per 07-03 summary |
| `kernels/matmul_iq4xs/BASELINE_DP4A.md` + `baseline_dp4a.json` | 8-shape timing table 20-40us bare / ~84us with quant vs 540us naive | ✓ VERIFIED | 8 canonical shapes, median_us 84-147us DP4A vs 543us naive, speedup 3.89-13.81x, cosine 0.999985; reproduce steps documented |
| `kernels/matmul_iq4xs/test_real_stock_compare.cpp` + `bench_real_stock.cpp` | Correctness + microbench vs real DP4A | ✓ VERIFIED | test 9 GEMV+6 GEMM cosine >=0.99 gate; bench reports naive vs real_dp4a median/p95 |
| `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` | Cooperative 8-thread Wave32 DP4A GEMV | ✓ VERIFIED (artifact) | 15186 bytes, 8-thread/row (ROWS_PER_BLOCK 32, THREADS_PER_ROW 8), LDS [32][33] (L184), launch_bounds(256,4)+amdgpu(256,256) (L163), DP4A sudot4+perm, ulong2 loads, correctness 10/10 PASS but speedup target not met |
| `kernels/matmul_iq4xs/test_gemv_dp4a_compare.cpp` + `bench_gemv_dp4a.cpp` | Correctness + speedup vs real DP4A | ✓ VERIFIED (artifact) | Test cosine >=0.999 vs ref + 1.000 vs stock; bench JSON speedup field present (peak 1.178) |
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | Streaming WMMA 64x32 double-buffered LDS WMMA | ✓ VERIFIED (artifact) | 13610 bytes, 64x32 per block 4x2 warps, LDS [2][32][33] _Float16 (L135), launch_bounds x2 kernels (L24 L105), wmma_f32_16x16x16_f16_w32 (L216), fallback TILE_M=16, GGML stride corrected |
| `kernels/matmul_iq4xs/test_gemm_wmma_compare.cpp` + `bench_gemm_wmma.cpp` | Parity + prefill bench vs real stock | ✓ VERIFIED (artifact) | 15 shapes cosine >=0.999 gate + gpu/tiled parity; bench M=128/512/1024 speedup+TFLOPS but not executed on this host |
| `kernels/matmul_iq4xs/CMakeLists.txt` | matmul_real_stock_hip + gemv_dp4a + wmma_stream targets | ✓ VERIFIED | 6 libraries + 7 executables wired; bench_real_stock links both stock+real objects |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh` | Vendored DP4A coop with LDS+launch_bounds | ✓ VERIFIED | sh_coop[32][33] (L83), launch_bounds+amdgpu (L77), sudot4+perm present, can_handle + dispatch M=1 |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` | Vendored WMMA stream with GGML layout fix | ✓ VERIFIED | sB[2][32][33] (L51), launch_bounds x2 (L25,L45), wmma builtin (L64), X[gm*K+gk] + Y[m*N+n] GGML fix, can_handle M>=16 |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/empty.cuh` + `README.md` | OFF fallback stub + provenance | ✓ VERIFIED | empty.cuh returns false/not-supported under guard; README notes OFF default |
| `llama.cpp/ggml/CMakeLists.txt` + `ggml-hip/CMakeLists.txt` | OFF/ON switch plumbing | ✓ VERIFIED | GGML_CUDA_ENABLE_CUSTOM_GFX1100 option OFF (L221), add_compile_definitions when ON |
| `patches/0001-gfx1100-mul-mat-custom.patch` | Quilt overlay over bb4caa75 | ✓ VERIFIED | Real git diff HEAD, dispatch intercepts in mmq/mvq, header count correct, reported apply --check PASS |
| `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` | Hybrid provenance + §8 update | ✓ VERIFIED | §8 exists with provenance, stride fix, failed variants, build cmds, raw paths |
| `docs/PUBLICATION.md` + `CHANGELOG.md` | Phase7 methodology + unreleased entry | ✓ VERIFIED | PUBLICATION §Phase7 hybrid, hygiene LICENSE/NOTICE intact per phase6; CHANGELOG unreleased 07-04 without false tok/s |
| `benchmarks/results/phase7/ab_*` + `op_gate_custom.json` | Paired bench + gates JSON | ✗ MISSING | Directory does not exist; 07-04 declares simulation — simulation != verification |
| `kernels/build/` | gfx1100 build artifacts | ⚠️ PARTIAL | Ninja build exists (07-01/07-02 built pre-Windows), but 07-03 build not-run on this host (no hipcc/ROCm) — files present cannot be disasm-verified here |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `kernels/matmul_iq4xs/CMakeLists.txt` | `real_stock_dp4a_comparator.hip` | matmul_real_stock_hip OBJECT + test/bench targets | ✓ WIRED | grep cmake shows library + executables linking correctly |
| `impl_gemv_dp4a_gfx1100.hip` | `test_gemv_dp4a_compare.cpp` / `bench_gemv_dp4a.cpp` | matmul_gemv_dp4a_hip object linked vs matmul_real_stock_hip for fair bench | ✓ WIRED | CMake targets include $<TARGET_OBJECTS:matmul_real_stock_hip> |
| `impl_gemm_wmma_stream.hip` | `bench_gemm_wmma.cpp` | bench_gemm_wmma links matmul_gemm_wmma_stream_hip vs matmul_real_stock_hip | ✓ WIRED | CMake target wires bench vs real stock (not vs naive) |
| `mmvq.cu` / `mmq.cu` | `gemv_iq4xs.cuh` / `gemm_iq4xs.cuh` | #if GGML_CUDA_ENABLE_CUSTOM_GFX1100 intercept can_handle+dispatch | ✓ WIRED | Patch adds includes + early-return dispatch guarded |
| `benchmarks/profiling/KERNEL-BENCH-DIFF.md` | `BASELINE_DP4A.md` | References baseline path | ✓ WIRED | §8 references BASELINE_DP4A.md + baseline_dp4a.json raw paths |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `real_stock_dp4a_comparator.hip` | y[row] via vec_dot_iq4_xs_q8_1 | quantize_row_q8_1_standalone (amax/127, ds half2) + ggml_cuda_dp4a/perm DP4A | ✓ FLOWING | Cosine 0.999985 vs FP64 oracle — integer path real |
| `impl_gemv_dp4a_gfx1100.hip` | y[row] via sh_coop reduction | quantize_coop (Q8_1) + coop_dp4a(sudot4/perm) + scale ls-32 * d*low2float | ✓ FLOWING (microbench) | coop/stock cos 1.000 — bit-identical DP4A; microbench measured but e2e not yet |
| `impl_gemm_wmma_stream.hip` | Y[m*N+n] via wmma | On-the-fly IQ4_XS->half dequant (d*(ls-32)*kvalues) into v16f16 + sB[2][32][33] B-tile from X[gm*K+gk] | ✓ FLOWING (structurally) | Stride fix applied; WMMA path gated M>=512, fallback tiled verified via 07-03 test harness (expected PASS on hardware) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| DP4A intrinsics present in comparator | grep ggml_cuda_dp4a/__builtin_amdgcn_* | 6 perm + 2 sudot4 + 8 dp4a calls | ✓ PASS |
| LDS padding [32][33] + [2][32][33] | grep __shared__.*33 | sh_coop[32][33] + sB[2][32][33] in hip and cuh x2 each | ✓ PASS |
| launch_bounds(256,4)+amdgpu(256,256) | grep launch_bounds/amdgpu | 4 hits across hip+cuh (GEMV 1, GEMM 2, vendored 2) | ✓ PASS |
| WMMA builtin | grep wmma_f32_16x16x16 | __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 in hip (L216) + cuh (L64) | ✓ PASS |
| Switch gating OFF/ON intact | grep GGML_CUDA_ENABLE_CUSTOM_GFX1100 | OFF default in CMakeLists + #if guards in 3 cuh + mmq/mvq/cmakes | ✓ PASS |
| Patch is real git diff | git -C llama.cpp log --oneline + diff stat | bb4caa75 base + 5c6b397 hybrid commit, 355 lines / 276 insertions | ✓ PASS (structural) |
| Thermal 90C aborts | cat logs/thermal_monitor.log | No kills; fallback polling (WinError 5 access denied) with 60s interval | ⚠️ PASS-NO-ABORT / FALLBACK |
| Paired llama-bench A/B JSON exists | ls benchmarks/results/phase7 | Missing — simulation documented | ✗ FAIL (expected per guardrail #4) |

### Probe Execution

No phase-declared probe scripts found under scripts/*/tests/probe-*.sh for this phase — skipped per Phase 7 (kernel build probes are `test_*` executables, verified via correctness gates).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| KERN-04 | ROADMAP Phase 7 | Hybrid DP4A GEMV 8-thread/row v_dot4 coop 8 DP4A, >1.2x vs real stock DP4A, >38 t/s decode | ⚠️ PARTIAL | Artifact PASS: impl_gemv_dp4a_gfx1100.hip has 8-thread coop, LDS [32][33], launch_bounds, sudot4+perm, correctness PASS cos 1.000 vs stock; GAP: speedup 1.178 peak <1.2 avg 1.00 (WSL DXG jitter) and no decode tok/s |
| KERN-05 | ROADMAP Phase 7 | WMMA streaming GEMM 64x32, double-buffered [2][32][33] LDS, wmma_f32_16x16x16_f16_w32, >950 t/s prefill | ⚠️ PARTIAL | Artifact PASS: impl_gemm_wmma_stream.hip has 64x32 4x2 warps, [2][32][33] half, wmma builtin, fallback TILE_M=16, GGML stride fixed; GAP: no bench vs real stock DP4A at M>=512 on hardware, no prefill tok/s |
| INTEG-02 | ROADMAP Phase 7 | Quilt patch updated with OFF/ON gating, paired llama-bench A/B protocol documented, gates green | ⚠️ PARTIAL | Patch PASS: real quilt 0001 over bb4caa75, gating intact, protocol documented (build-stock/build-custom, --single-turn -ngl 99 -b 2048, thermal window); GAP: protocol not executed — no JSON, no gates PASS on custom, simulation-only |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` | L215 | Q8_1 qs scalar int fallback (4B) vs 128-bit | ℹ️ Info | Documented residual — 36B struct misalignment prevents 16B q8 load; future 64B padded AQ could unlock 5-10% BW, not a blocker |
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | L77 | WMMA gate M>=512 limits coverage | ℹ️ Info | Intentional per spec — M=128 falls back to tiled TILE_M=16, still expected >1.2x but unverified vs real stock |
| `patches/0001-gfx1100-mul-mat-custom.patch` | header | compact vendoring (276 vs >500 lines) | ℹ️ Info | Intentional compact vendoring; full verbose impl remains in kernels/ for audit — not a stub |

No TBD/FIXME/XXX/HACK markers found in Phase 7 artifacts. No empty returns or hardcoded empties. No console.log-only handlers.

### Human Verification Required

4 items — automated checks pass for code correctness, but end-to-end performance requires WSL2 gfx1100 hardware:

### 1. Bare-metal microbench vs real stock DP4A
**Test:** HSA_ENABLE_DXG_DETECTION=1 ./bench_gemv_dp4a and ./bench_gemm_wmma (and ./bench_real_stock for baseline) under WSL2 gfx1100 ROCm 7.2.1
**Expected:** GEMV peak/avg speedup >1.2x vs real DP4A across 8 canonical shapes; GEMM M=512 speedup >1.2x (WMMA path) vs real DP4A MMQ at M=128/512
**Why human:** This Windows host has no hipcc/ROCm/GPU — WSL DXG virtualization jitter flattened GEMV delta to avg 1.00 peak 1.178; bare metal needed per 07-02 residual

### 2. Paired llama-bench A/B thermal-paired sweep
**Test:** Back-to-back stock OFF vs custom ON llama-bench --single-turn --simple-io --load-mode none -ngl 99 -b 2048 across {512,1024,2048,4096}, one thermal window with hwinfo_daemon
**Expected:** Custom decode tok/s >38 and > stock; custom prefill tok/s >950 and > stock; RunStore dirs benchmarks/results/phase7/ab_stock_* and ab_custom_* with CHECKSUMS + clocks/temps
**Why human:** 07-04 documents simulation only — real GPU + model GGUF + hwinfo share mem required, not available on Windows host

### 3. Custom ON QUAL-01/02 gates
**Test:** cmake -S llama.cpp -B build-custom -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON && cmake --build && run_op_gate.py + run_model_gate.py
**Expected:** 0 errors across 4200+ ops; PPL 6.4271±1% and 6/6 canaries PASS
**Why human:** Build-custom not compiled here; requires hijacking WSL2 ROCm build under 90s/300s timeouts per 07-04 protocol

### 4. VGPR + disasm gate
**Test:** hipcc --save-temps -Rpass-analysis --offload-arch=gfx1100 + llvm-objdump --mcpu=gfx1100 on impl_gemv_dp4a_gfx1100.hip.o / impl_gemm_wmma_stream.hip.o
**Expected:** <=64 VGPRs (16 waves/SIMD) and v_dot4 (sudot4) + v_wmma_f32_16x16x16_f16 in disasm
**Why human:** No build artifacts on this Windows host; 07-03 build marked not-run

### Gaps Summary

Phase 7 artifactual work is substantially complete and guardrail-compliant: real DP4A comparator is hardware-faithful (not naive), both hybrid kernels implement the mandated microarchitectural patterns (DP4A v_dot4 + perm, WMMA matrix cores, LDS 33-padding, launch_bounds 256,4, amdgpu 256), and the quilt patch is a real bisectable git diff with OFF/ON gating intact. What is MISSING is measurable proof of the phase goal — outrunning production stock end-to-end in llama-bench — because the host executing 07-04 had no ROCm/HIP/GPU/model and correctly refused to fabricate tok/s. Microbench delta vs real stock is close but short of target (1.178 not >1.2 under virtualization), and GEMM WMMA vs real stock was never benched on metal. Closing requires one WSL2 gfx1100 session: rebuild kernels + llama.cpp OFF/ON, run the 3 microbenches, run the 4-tier paired llama-bench sweep in one thermal window with hwinfo_daemon, and re-run the two quality gates on custom — then publish the JSON artifacts and update KERNEL-BENCH-DIFF §8 + PUBLICATION with measured medians.

---
_Verified: 2026-08-27T19:50:00Z_
_Verifier: Claude (gsd-verifier)_
