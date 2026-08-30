---
phase: 07-hybrid-dp4a-wmma-kernel-optimization
verified: 2026-08-29T14:45:00Z
# REPLAND 2026-08-30: plans 07-01..07-04 deleted; closure plans 07-01 (REQ-WIN-07 Windows), 07-02 (REQ-PERF-07 ≥1.10×), 07-03 (REQ-STAT-07 N≥10/15) carry objectives + ways-to-achieve. Bare-metal N=10 evidence committed d414c552/6e46d2e supersedes gaps below: bench_real_stock 6.24× PASS, gemv 0.976 FAIL, gemm M1024 peak 1.89× avg 1.08×, llama-bench 4-tier 1.079/0.996/1.003/0.978/tg 0.993 FAIL. 0/3 must-haves closed; ways in output/deep-research/phase7-3must-haves-exhaustive.md + docs/PUBLICATION.md §8.
status: gaps_found
score: 1/7 must-haves verified (pre-replan; see replan note)
behavior_unverified: 1
overrides_applied: 0
gaps:
  - truth: "GEMV cooperative 8-thread DP4A >1.2x median N=10 vs real vec_dot_iq4_xs_q8_1 DP4A (REQ-PERF-07 decode slice, REQ-STAT-07)"
    status: failed
    reason: "Cooperative GEMV kernel present and wired (sh[32][33] + launch_bounds + sudot4/perm + ulong2 b128), LDS/XOR helper present, but WSL2 gfx1100 hardware N=10 bench_gemv_dp4a.hardware.json shows 0.942 avg FAIL <1.2x. No shape >=1.2x (peak 1.048 ffn_up, attn_q 0.965). WSL2 DXG jitter 15-30us flattens delta; bare-metal 16 waves pending."
    artifacts:
      - path: "kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip"
        issue: "Present 16797B — sh[32][33] LDS + launch_bounds(256,4)+amdgpu 256,256 + sudot4+6x perm + ulong2 b128 present; XOR variant only as helper cuh not compiled as second OBJECT"
      - path: "kernels/matmul_iq4xs/gemv_variant_xor.cuh"
        issue: "Present 2108B — xor_preshuffle helpers present but gated via #ifdef GEMV_XOR not built as separate variant object"
      - path: "kernels/matmul_iq4xs/bench_gemv_dp4a.hardware.json"
        issue: "Valid JSON 6430B 8 entries runs:10 each — speedup_median 0.965/0.898/0.944/0.938/0.983/0.957/1.048/0.801 avg 0.942 ALL FAIL <1.2x, 0/8 mean-1sigma >=1.15x — honest FAIL, not fabricated"
    missing:
      - "Bare-metal WSL2 gfx1100 HSA_ENABLE_DXG_DETECTION=1 bench_gemv_dp4a --runs 10 --json with XOR vs +33 as two compiled OBJECTs (matmul_gemv_dp4a_hip + matmul_gemv_dp4a_xor_hip) interleaved to prove median >1.2x + mean-1sigma >1.15x + >38 t/s decode"
      - "hipcc --save-temps -Rpass-analysis VGPR <=64 (43 already) + llvm-objdump --mcpu=gfx1100 | grep v_dot4 + rocprof lds_bank_conflict 0 on bare-metal per variant"
  - truth: "WMMA streaming GEMM hardware wmma 64x32 [2][32][33] + P=4 XOR + tile sweeps (64x64/128x32, B-stationary, LUT mu=4, b128) >1.2x at M>=512 + >950 t/s prefill N=10"
    status: failed
    reason: "Core WMMA 64x32 P2 [2][32][33] + wmma_f32_16x16x16_f16_w32 + sched_barrier pinning present, but high-yield sweeps P=4/XOR/64x64/128x32 are comments not compiled objects. bench_gemm_wmma.hardware.json is TRUNCATED at 12288B (incomplete JSON, last object cuts at 'stock_dp'), partial data shows M128 0.042-0.044x (17619us vs 736us) and M512 0.57-0.60x (11847us vs 6754us) ALL FAIL <1.2x; custom_gemm can_handle stub disables dispatch entirely."
    artifacts:
      - path: "kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip"
        issue: "16462B — sB[2][32][33] + wmma builtin + sched_barrier 0x0080/0x0008 + v16f16/v8f32 lane%16 present; but sB[4][32][32] XOR is comment-only (__shared__ _Float16 sB_P4[4][32][32] line 143), no TILE_M/N template, no 64x64/128x32 compiled variant"
      - path: "kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip"
        issue: "7233B — LUT mu=4 secondary exists as separate file, not variant switch in main stream"
      - path: "kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json"
        issue: "TRUNCATED 12288B — incomplete JSON (JSONDecodeError unterminated string line 389 col 15, last chars 'winner: stock_dp'), partial valid entries show 0.042 M128 and 0.57 M512 FAIL; cannot be used as complete evidence until regenerated via timeout 90s bench"
      - path: "kernels/matmul_iq4xs/bench_gemm_wmma.cpp"
        issue: "CLI --runs 10 --json + per-variant table present but variants use synthetic jitter if vi==1 v_median*=0.97 / vi==2 *=0.95 — identical compiled code would still show 3% delta; not real per-object measurement"
      - path: "llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh"
        issue: "custom_gemm_iq4xs_can_handle stub return false line 88 disables WMMA dispatch — even if WMMA were faster, mmq.cu never calls dispatch"
    missing:
      - "Regenerate bench_gemm_wmma.hardware.json via timeout 90 HSA_ENABLE_DXG_DETECTION=1 bench_gemm_wmma --runs 10 --json on bare-metal (avoid 271s DXG deadlock), valid JSON with 3 shapes x 5 variants x 4 Ms, showing speedup_median >1.2x at M=512/1024"
      - "Compile real variant OBJECTs: matmul_gemm_wmma_64x32_p2_hip + matmul_gemm_wmma_64x32_p4_xor_hip (real sB[4][32][32] XOR + sched_barrier P=4) + matmul_gemm_wmma_64x64_p4_hip + matmul_gemm_lut_hip, remove synthetic jitter, expose distinct gemm_iq4xs_wmma_*_gpu symbols"
      - "Restore custom_gemm_iq4xs_can_handle to real gate (type==IQ4_XS && M>=16 && M>=512/W_16-aligned check mirroring impl GEMM) and expose b128 __builtin_amdgcn_global_load_b128 + assume_aligned + 16x64 swizzle wired"
      - "llvm-objdump --mcpu=gfx1100 | grep v_wmma + VGPR <=64 + rocprof lds_bank_conflict 0 per variant on bare-metal (WSL2 blind noted)"
  - truth: "Quilt patch git apply --check (real git diff bb4caa75, GGML_CUDA_ENABLE_CUSTOM_GFX1100 OFF/ON gating, no stub disabling dispatch)"
    status: failed
    reason: "Patch file on disk is now 355 lines (fixed from 30-line truncation) and core.autocrlf=false + *.patch eol=lf present, so git apply --check would pass, but patch retains gemm can_handle stub return false which disables WMMA path — patch is structurally complete yet functionally disables the prefill uplift it is supposed to enable."
    artifacts:
      - path: "patches/0001-gfx1100-mul-mat-custom.patch"
        issue: "355 lines / 276 insertions, 8 files via git -C llama.cpp diff bb4caa75 — now complete (was 30-line truncated before 07-04 fix). .gitattributes *.patch eol=lf correct."
      - path: "llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh"
        issue: "Vendored coop GEMV present with real can_handle (type!=IQ4_XS false; M!=1 false; K/N checks true) — PASS"
      - path: "llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh"
        issue: "Vendored WMMA stream present but can_handle stub return false line 88 — FAIL, disables GEMM dispatch"
      - path: "llama.cpp/ggml/src/ggml-cuda/mvq.cu"
        issue: "Guard #if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100) present"
      - path: "llama.cpp/ggml/src/ggml-cuda/mmq.cu"
        issue: "Guard present but dispatch never reached due to stub"
    missing:
      - "Regenerate patch after restoring gemm can_handle to real shape gate (M>=16 etc) and after adding real P=4/XOR variant objects, verify git -C llama.cpp apply --check on both WSL2 (/opt/rocm) and Windows (HIP_PATH) with core.autocrlf=false"
  - truth: "REQ-WIN-07 Windows build_windows.bat (HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja, no cl, llama-server.exe :8000 ->200, <=2 langs)"
    status: failed
    reason: "build_windows.bat exists and is code-correct per gate (HIP_PATH + clang++ --offload-arch=gfx1100 --version + -G Ninja + find_package(hip via HIP_PATH) + curl :8000 smoke), but Windows native build not executed on this host (no HIP SDK at C:/Program Files/AMD/ROCm/6.4, no clang++.exe, no build-windows/bin/llama-server.exe), and find -name *.py ! -path ./llama.cpp/* == 40 violates <=2 langs gate (deferred to Phase 8 prune)."
    artifacts:
      - path: "build_windows.bat"
        issue: "Present 5857B — correct HIP_PATH, where clang++.exe --offload-arch=gfx1100 --version, where ninja, GGML_HIP=ON + GGML_CUDA_ENABLE_CUSTOM_GFX1100=ON + -G Ninja, curl :8000 smoke, MODEL_PATH guard — not executed (no HIP SDK)"
      - path: "kernels/CMakeLists.txt"
        issue: "find_package(hip REQUIRED CONFIG PATHS \"$ENV{HIP_PATH}/lib/cmake/hip\") line 17 present — no /opt/rocm hardcode PASS at code level"
      - path: "build-windows/bin/llama-server.exe"
        issue: "MISSING — not built on this host; no curl :8000 200 evidence"
    missing:
      - "Windows 11 bare-metal execution of build_windows.bat (HIP SDK 6.4 + Ninja + VS Build Tools) producing build-windows/bin/llama-server.exe and curl http://127.0.0.1:8000/v1/chat/completions ->200 with choices[0].message.content on gfx1100"
      - "Phase 8 prune to satisfy find -name *.py ! -path ./llama.cpp/* ==0 (currently 40: benchmarks/ harness + race.py + swizzle + output/) — correctly offline-only but still counts until pruned; deferred to Phase 8"
  - truth: "REQ-PERF-07 >=1.10x pp+tg at {512,1024,2048,4096,8192} N=10 median and mean-1sigma >=1.10x thermal-paired"
    status: failed
    reason: "No paired llama-bench A/B N=10 JSON exists. Prior real 808->849 pp4096 is +5.1% (1.051x) FAILS gate. Synthetic rows.jsonl median 1.03-1.08x across 250 repeats (<1.10x) correctly labeled synthetic and FAILS gate — no fabricated 1.10x PASS."
    artifacts:
      - path: "benchmarks/results/phase7/rows.jsonl"
        issue: "250 synthetic repeats (timestamps 1787995716 future, random uniform jitter via race.py), median 1.05 FAILS 1.10x — harness shape correct, not hardware"
      - path: "benchmarks/results/phase7/CHECKSUMS.sha256"
        issue: "Single checksum 8d6a943a for synthetic rows.jsonl only; no ab_stock/ab_custom per-tier pp/tg split"
      - path: "benchmarks/profiling/KERNEL-BENCH-DIFF.md"
        issue: "§8 correctly labels HONEST synthetic vs hardware FAIL: 512 1.08x FAIL / 1024 1.07x FAIL / 4096 real 1.051x FAIL — does not fabricate 1.10x PASS"
      - path: "benchmarks/results/phase7/race.py"
        issue: "Interleaved --repeats 10 A,B,A,B + hwinfo_daemon 1Hz + thermal_watchdog 90C + VRAM preflight present at code level but hardware not executed"
    missing:
      - "Paired llama-bench A/B stock OFF vs custom ON (same bb4caa75, -ngl 99 -b 2048 --single-turn --simple-io --load-mode none -r 10 per tier per build, ONE thermal window, interleaved race.py --repeats 10 A,B,A,B, hwinfo_daemon 1Hz + thermal_watchdog 90C, RunStore rows.jsonl + CHECKSUMS.sha256, VRAM preflight >2GB + hipMalloc probe for 8192) proving median >=1.10x and mean-1sigma >=1.10x for pp+tg at 512..8192 (8192 SKIPPED with FA+GQA rationale if preflight fails)"
      - "High-yield variant that actually pushes 512/1024 over 1.10x on bare-metal (current BEST synthetic 1.08 <1.10 suggests 64x32 P2 alone insufficient; needs P=4+XOR+b128+16x64 swizzle with GEMM dispatch enabled)"
  - truth: "REQ-STAT-07 N>=10 rigour and LLM QA N=15 temp=0 are enforced end-to-end (bench_* --runs 10 + llama-bench N=10 per tier + 15x LLM QA per-run tables)"
    status: failed
    reason: "Harness now supports N>=10 (bench_* --runs 10 defaults, rows.jsonl 250 proves loop), but hardware N=10 thermal-paired execution and N=15 LLM QA not witnessed. bench_gemm truncated JSON breaks N=10 evidence chain; no llama-cli --temp 0 N=15 table exists. Single-run banned is honored (no fabricated single-run claims), but gates not yet closed."
    artifacts:
      - path: "kernels/matmul_iq4xs/bench_real_stock.cpp"
        issue: "--runs 10 default + BenchStats median/mean/stddev/p95 present and hardware JSON valid 8 entries runs:10 — PASS for this component"
      - path: "kernels/matmul_iq4xs/bench_gemv_dp4a.cpp"
        issue: "--runs 10 default + speedup_median + mean-1sigma present, hardware JSON valid 8 entries runs:10 — PASS"
      - path: "kernels/matmul_iq4xs/bench_gemm_wmma.cpp"
        issue: "--runs 10 default present but hardware JSON truncated at 12288B — no complete N=10 proof for GEMM"
      - path: "benchmarks/results/phase7/race.py"
        issue: "--repeats 10 interleaved + RunStore + CHECKSUMS present at code level; LLM QA N=15 documented but not run"
    missing:
      - "HSA_ENABLE_DXG_DETECTION=1 bench_gemm_wmma --runs 10 --json on bare-metal with timeout 90s producing valid JSON (replace truncated file) — proves median/mean/stddev/p95 per variant"
      - "Paired llama-bench N=10 per tier per split + llama-cli --temp 0 fixed prompt N=15 with per-run 15-row table (avg tok/s + avg latency + stddev) on gfx1100"
behavior_unverified_items:
  - truth: "bench_real_stock 99us vs 543us is true hardware median N=10 (not single-run, not naive)"
    test: "HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json — check baseline_dp4a.json median 99.547 ±28.56 vs naive 543.457"
    expected: "8 entries median 92-135us ± stddev with p95 160-298us, speedup 5.46x vs naive, real_stock_dp4a_comparator.hip grep shows ggml_cuda_dp4a + perm x6 not naive scalar"
    why_human: "Structure verified (bench has --runs 10, JSON has runs:10 + stddev/p95, comparator has sudot4+perm), but bare-metal hipEvent timing not re-witnessed on this Windows host; we do not fabricate — values quoted from bench_real_stock.hardware.json captured on WSL2 gfx1100"
human_verification:
  - test: "Bare-metal WSL2 gfx1100 N=10 microbench vs real DP4A + llama-bench A/B sweep"
    expected: "bench_real_stock 99us±28 vs 543us (already hardware), bench_gemv median >1.2x (>38 t/s decode), bench_gemm M>=512 >1.2x + >950 t/s prefill, paired llama-bench pp+tg >=1.10x median and mean-1sigma >=1.10x at 512..8192 (8192 conditional SKIPPED on VRAM preflight)"
    why_human: "Requires WSL2 Ubuntu-24.04 ROCm 7.2.1 gfx1100, hipcc --offload-arch=gfx1100, GGUF model, hwinfo_daemon+thermal_watchdog; Windows host has no hipcc/GPU — intentionally not fabricated here"
  - test: "Windows native build_windows.bat full execution + :8000 smoke"
    expected: "HIP SDK 6.4 at HIP_PATH, Ninja + clang++.exe --offload-arch=gfx1100 -G Ninja builds build-windows/bin/llama-server.exe and curl :8000 ->200 choices[0].message.content"
    why_human: "Needs Windows 11 + VS Build Tools + HIP SDK; this host has no HIP SDK at C:/Program Files/AMD/ROCm/6.4, so build not executable — code inspection only"
  - test: "rocprof lds_bank_conflict 0 + VGPR <=64 + llvm-objdump v_dot4/v_wmma per variant"
    expected: "rocprof shows 0 bank conflicts for winner, calculator predicts A/B/D 8 VGPR each <=64, llvm-objdump shows v_dot4 (GEMV) and v_wmma (GEMM)"
    why_human: "rocprofv3 librocdxg unsupported on WSL2 (404 Instinct-only), disasm needs .o from gfx1100 build"
deferred:
  - truth: "Repo pure C++/HIP <=2 langs (find -name *.py ! -path ./llama.cpp/* ==0)"
    addressed_in: "Phase 8"
    evidence: "Phase 8 goal: strip bloat to pure C++/HIP, prune to block_iq4_xs.h + hip_helpers.h + impl_gemv/gemm winners; Phase 7 correctly keeps benchmarks/ Python harness offline-only but still counts"
---

# Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization — RE-SCOPED 2026-08-28 Verification Report

**Phase Goal:** Fuse Q8_1 integer activation quantization and RDNA3 hardware matrix cores (v_dot4_i32_i8 / v_wmma) with Wave32 cooperative workgroups to **beat real production stock llama.cpp by ≥10% end-to-end in llama-bench, Windows-native (≤2 langs), and with 10× (15× LLM QA) statistical rigour**.

**Verified:** 2026-08-29T14:45:00Z on Windows 11 Git Bash (no ROCm/HIP/GPU/model — WSL2 gfx1100 hardware JSONs used, not fabricated). All GPU numbers below are *quoted from captured hardware JSONs*, not re-measured here.

**Status:** gaps_found
**Re-verification:** Yes — after honest execution of 07-01..07-04 (previous 07-VERIFICATION 2026-08-29 12:30 score 2/7; hardware JSONs now captured and examined, patch regenerated to 355 lines, GEMM truncated JSON discovered)

## Goal Achievement

### Observable Truths — 7 Re-Scoped Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | bench_real_stock N=10 median±stddev vs real DP4A 99us (not naive 543us, not single-run) | ✓ VERIFIED | `bench_real_stock.cpp` --runs 10 default, `bench_real_stock.hardware.json` 8 entries valid JSON each `runs:10` + `real_dp4a_median_us 99.547±28.56 p95 231.54` vs `naive 543.457±84.68 p95 780.64` = **5.46x** (ffn_down 115.39 vs 1853.56 =16.06x). `baseline_dp4a.json` copied verbatim (py equality). `real_stock_dp4a_comparator.hip` grep `ggml_cuda_dp4a_real` + `__builtin_amdgcn_sudot4` L73 + 6x `__builtin_amdgcn_perm` L111-118 + `vec_dot_iq4_xs_q8_1_device` L145 — **not naive**. `BASELINE_DP4A.md` N=10 table median±stddev+p95. Hardware quoted, not fabricated. |
| 2 | GEMV >1.2x median N=10 vs real vec_dot_iq4_xs_q8_1 DP4A (cooperative 8-thread Wave32) | ✗ FAILED | Present + wired: `impl_gemv_dp4a_gfx1100.hip` 16797B `launch_bounds(256,4)+amdgpu(256,256)` L174, `sh[32][33]` L184, `sudot4` L89 + `perm x6` L109-116, `ulong2` b128 16B L214, `gemv_variant_xor.cuh` XOR helper present. BUT hardware `bench_gemv_dp4a.hardware.json` (valid 6430B, 8 entries) shows **avg 0.942 <1.2x**: attn_q 0.965, attn_k 0.898, attn_v 0.944, attn_gate 0.938, attn_out 0.983, ffn_gate 0.957, **ffn_up 1.048 peak**, ffn_down 0.801. 0/8 >=1.2x, 0/8 mean-1sigma >=1.15x (mean-1sigma 0.402-0.614). Prior peak 1.178 collapsed under DXG jitter 15-30us p95 148-343us. |
| 3 | WMMA streaming GEMM hardware wmma (64x32 [2][32][33] + P=4 XOR + sweeps + b128, >1.2x at M>=512, >950 t/s) | ✗ FAILED | Core PASS: `impl_gemm_wmma_stream.hip` 16462B `sB[2][32][33]` L142, `wmmma_f32_16x16x16_f16_w32` L230, `sched_barrier 0x0080` L162 + `0x0008` L177, `__launch_bounds(256,4)+amdgpu(256,256)` x2, `v16f16/v8f32 lane%16`. BUT sweeps FAIL: `sB[4][32][32]` only comment L143, no TILE_M/N template. Hardware `bench_gemm_wmma.hardware.json` **TRUNCATED at 12288B** (JSONDecodeError line 389, last chars `"winner": "stock_dp`), partial entries prove FAIL: **M128 736us vs 17619us =0.042x** (24x slower), **M512 6754us vs 11847us =0.57x** (1.75x slower, 2.26 TFLOPS vs stock 3.97 TFLOPS), P4_XOR synthetic 0.588 only via `v_median*=0.97` jitter. `can_handle return false` disables dispatch. |
| 4 | Quilt patch git apply --check (real diff over bb4caa75, OFF/ON gating, no disabling stub) | ✗ FAILED | `llama.cpp` log `bb4caa75 -> 5c6b397` exists, `.gitattributes *.patch eol=lf` present, `ggml/CMakeLists.txt:221 OFF` + `mmvq.cu:3+112` + `mmq.cu:2+1278` guards verified, patch now **355 lines / 276 insertions 8 files** via `git -C llama.cpp diff bb4caa75` (was 30-line truncated before 07-04 fix) — `git apply --check` would PASS. BUT `gemm_iq4xs.cuh:88 can_handle return false` stub remains, disabling WMMA dispatch (gemv can_handle is correct M==1 gate). Patch is complete yet functionally gated off. |
| 5 | REQ-WIN-07 Windows build_windows.bat (HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja, not cl, :8000 ->200, <=2 langs) | ✗ FAILED | `build_windows.bat` 5857B code-correct: `HIP_PATH` L6, `clang++.exe --offload-arch=gfx1100 --version`, `where ninja` (errors if cl path), `cmake -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` with `HIP_PATH/bin/clang++.exe`, `curl :8000/v1/chat/completions ->200` smoke with MODEL_PATH guard. `kernels/CMakeLists.txt:17` HIP_PATH search. BUT not executed: no HIP SDK at `C:/Program Files/AMD/ROCm/6.4` on this host (Windows Git Bash, no hipcc), `build-windows/bin/llama-server.exe` MISSING, no curl 200. `find -name *.py ! -path ./llama.cpp/*` == **40** (benchmarks/ + race.py + swizzle + output/) — fails ==0 until Phase 8 prune (deferred). |
| 6 | REQ-PERF-07 >=1.10x pp+tg at {512,1024,2048,4096,8192} N=10 median and mean-1sigma >=1.10x thermal-paired | ✗ FAILED | Correctly does NOT fabricate: `KDB §8` + `PUBLICATION.md` + `rows.jsonl` all label synthetic and show FAIL — prior real `808->849 pp4096 1.051x +5.1% FAILS`, synthetic race median **1.05 FAILS 1.10x** per tier (512 1.08x FAIL / 1024 1.07x FAIL / 2048 1.08x FAIL / 4096 real 1.051x FAIL), 8192 conditional SKIPPED on VRAM preflight (15.3GB+128KiB/tok->18.5GB on 20GB, 800 GiB lie + BSOD risk). `race.py --repeats 10` interleaved A,B,A,B harness exists but no hardware ab_stock/ab_custom JSON produced. |
| 7 | REQ-STAT-07 N>=10 rigour (microbench + llama-bench N=10 median/mean/stddev/p95, LLM QA N=15 temp=0 per-run table, single-run banned) | ✗ FAILED | Harness supports N>=10: `bench_real_stock/gemv/gemm` all default `--runs 10` + BenchStats median/mean/stddev/p95 via bench.h; `bench_real_stock` and `bench_gemv` hardware JSONs valid N=10; `race.py --repeats 10` interleaved + RunStore+CHECKSUMS proven via rows.jsonl 250 lines. BUT hardware chain broken: `bench_gemm` JSON truncated (12288B), no paired `llama-bench N=10` per tier, no `llama-cli --temp 0 N=15` per-run 15-row table. `rows.jsonl` is synthetic random jitter (future ts 1787995716). Harnessed but unverified on hardware. |

**Score:** 1/7 truths verified (Truth 1 only; 0 verified for perf; 1 behavior_unverified item below)

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|--------------|----------|
| 1 | Repo pure C++/HIP <=2 langs (no Python/JS, find py ==0) | Phase 8 | Phase 8 goal: `08-refactor-windows-native` strips bloat to kernels/ + patches/ + build_windows.bat; Phase 7 correctly keeps benchmarks/ offline-only |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` | True upstream DP4A pipeline | ✓ VERIFIED | 25156B, `ggml_cuda_dp4a_real` via `sudot4` L73 + 6x `perm` L111-118 + `vec_dot_iq4_xs_q8_1_device` L145 + `ls decode d=half2float*low2float` — not naive |
| `kernels/matmul_iq4xs/bench_real_stock.cpp` | --runs 10 --json | ✓ VERIFIED | Parses `--runs` default 10 + `--json`, emits JSON `runs:10` + `median/mean/stddev/p95 + speedup_vs_naive` per 8 shapes |
| `kernels/matmul_iq4xs/bench_real_stock.hardware.json` | N=10 hardware proof | ✓ VERIFIED | Valid JSON 4578B, 8 objects each `runs:10` + `real_dp4a_median 99.547±28.56 p95 231.54` vs `naive 543.457±84.68 p95 780.64` + `speedup 5.46x` — **quoted, not fabricated** |
| `kernels/matmul_iq4xs/baseline_dp4a.json` | N=10 median±stddev table source | ✓ VERIFIED | 4578B, 8 entries copied verbatim from hardware JSON (py `baseline==hardware` equality) |
| `kernels/matmul_iq4xs/BASELINE_DP4A.md` | N=10 table + 84vs543 proof | ✓ VERIFIED | 62 lines, table `median ± stddev` + `p95` + `GB/s 130` + `speedup 5.46x`, `runs:10`, reproduce steps, wx2 jitter note |
| `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` | Coop 8-thread DP4A GEMV | ✓ VERIFIED (artifact) | 16797B, 8 thr/row (256->32 rows), `sh[32][33]` L184, `launch_bounds(256,4)+amdgpu(256,256)` L174, `sudot4+perm`, `ulong2` 16B L214 + `assume_aligned` |
| `kernels/matmul_iq4xs/gemv_variant_xor.cuh` | XOR preshuffle helper | ✓ VERIFIED | 2108B, `xor_preshuffle_32x33 (y%(32/8))^x` + `xor_preshuffle_32x32 (y%8)^x`, `#ifdef GEMV_XOR` gating — helper present |
| `kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` | --runs 10 --json vs real DP4A | ✓ VERIFIED (artifact) | `--runs 10` default, `speedup_median` + `mean-1sigma`, `variant` field, links vs `matmul_real_stock_hip` (not naive) |
| `kernels/matmul_iq4xs/bench_gemv_dp4a.hardware.json` | N=10 GEMV hardware proof | ✓ VERIFIED (structure) / ✗ FAIL perf | Valid JSON 6430B, 8 entries `runs:10` + `speedup_median 0.965...1.048 avg 0.942` ALL FAIL <1.2x — **honest, not fabricated** |
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | WMMA 64x32 streaming | ✓ VERIFIED (core) / ✗ variants | 16462B, `sB[2][32][33]` L142, `wmmma_f32_16x16x16_f16_w32` L230, `sched_barrier 0x0080` L162 + `0x0008` L177, `v16f16/v8f32 lane%16`, but P=4/XOR/64x64 only comments L143 |
| `kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip` | LUT mu=4 secondary | ✓ VERIFIED | 7233B, `lut_entry 16 half 32B`, B-stationary + LUT + `sB[2][32][33]` + barriers, `__launch_bounds` present |
| `kernels/matmul_iq4xs/bench_gemm_wmma.cpp` | --runs 10 --json per-variant | ✓ VERIFIED (code) / ⚠️ jitter | `--runs 10` + `--variant all`, 5 variants + `M={128,512,1024,8192}` + `TFLOPS_median` + `VRAM preflight >2GB + hipMalloc probe`, but `v_median*=0.97/0.95` synthetic jitter L170-171 means race compares near-identical code |
| `kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json` | N=10 GEMM hardware proof | ✗ TRUNCATED | 12288B, incomplete JSON (unterminated string, cuts at `"winner": "stock_dp`), partial valid shows **M128 0.042x 17619us vs 736us** and **M512 0.57x 11847us vs 6754us FAIL** — must be regenerated via timeout 90s |
| `kernels/matmul_iq4xs/CMakeLists.txt` | matmul targets | ✓ VERIFIED | 6 libs + executables, `matmul_real_stock_hip` OBJECT + bench/test linking correct, HIP_PATH line17 |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh` | Vendored GEMV coop | ✓ VERIFIED | `sh_coop[32][33]` + `launch_bounds` + `sudot4+perm` + real `can_handle (M==1, IQ4_XS)` line 114 true |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` | Vendored WMMA stream | ✗ STUBBED | `sB[2][32][33]` + `wmma` present but `custom_gemm_iq4xs_can_handle` line88 `return false` disables path |
| `patches/0001-gfx1100-mul-mat-custom.patch` | Quilt overlay | ✓ VERIFIED (structure) / ✗ gate | 355 lines / 276 insertions 8 files via `git -C llama.cpp diff bb4caa75` — complete, `.gitattributes eol=lf` correct, apply --check would PASS, but gemm stub inside |
| `build_windows.bat` | Windows native gate | ✓ VERIFIED (file) / ✗ not executed | 5857B correct HIP_PATH/clang++/gfx1100/-G Ninja/:8000 smoke, but `build-windows/bin/llama-server.exe` MISSING |
| `benchmarks/results/phase7/race.py` | race --repeats 10 | ✓ VERIFIED | 12096B, `--repeats 10` interleaved A,B,A,B (adelj88), 5 variants + TIERS 512..8192 + hwinfo_daemon+thermal_watchdog + VRAM preflight + RunStore+CHECKSUMS |
| `benchmarks/results/phase7/rows.jsonl` | N=10 RunStore | ⚠️ SYNTHETIC | 250 lines synthetic random jitter (future ts 1787995716), median 1.05 <1.10x — proves harness shape, not hardware |
| `tools/swizzle_iq4xs.py` | offline 16x64 swizzle | ✓ VERIFIED | 5113B, offline-only, py_compile PASS (offline-only correctly not shipped until Phase 8 prune) |
| `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` | Hybrid provenance | ✓ VERIFIED | §8 per-variant table + per-tier 1.10x verdict HONEST: all synthetic FAIL + hardware FAIL <1.10x + gemm stub note |
| `docs/PUBLICATION.md` | Phase7 methodology | ✓ VERIFIED | High-yield variant racing + N=10 synthetic vs hardware HONEST FAIL, no fabricated 1.10x PASS |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `kernels/matmul_iq4xs/CMakeLists.txt` | `real_stock_dp4a_comparator.hip` | `matmul_real_stock_hip OBJECT` | ✓ WIRED | cmake OBJECT links bench vs real DP4A (not vs naive) |
| `bench_gemv_dp4a.cpp` | `impl_gemv_dp4a_gfx1100.hip` vs `real_stock` | `matmul_gemv_dp4a_hip` vs `matmul_real_stock_hip` fair race | ✓ WIRED | Code links vs real DP4A 99us denominator |
| `bench_gemm_wmma.cpp` | `impl_gemm_wmma_stream.hip` vs `real_stock` | `matmul_gemm_wmma_stream_hip` vs `matmul_real_stock_hip` | ✓ WIRED (code) | Bench wires vs real DP4A, but truncated JSON breaks evidence |
| `mmvq.cu`/`mmq.cu` | `gemv_iq4xs.cuh`/`gemm_iq4xs.cuh` | `#if GGML_CUDA_ENABLE_CUSTOM_GFX1100` intercept | ✓ WIRED (code) / ✗ patch gate | Guards present, but gemm can_handle return false disables dispatch |
| `build_windows.bat` | `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100` | `-G Ninja + HIP_PATH` | ✓ WIRED (code) | Bat checks `where clang++.exe`, `where ninja`, errors if cl |
| `race.py` | `bench_* --runs 10 --json` + `llama-bench N=10` | `race --repeats 10` interleaved median pick | ✓ WIRED (harness) / ✗ hardware | Harness documents interleaving + RunStore+CHECKSUMS, no hardware JSON for llama-bench A/B |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `real_stock_dp4a_comparator.hip` | y[row] via vec_dot | `quantize_row_q8_1_standalone (amax/127, ds half2) + ggml_cuda_dp4a/perm DP4A` | ✓ FLOWING | Cosine 0.999985 vs FP64 oracle (claimed prior hardware, N=10) |
| `impl_gemv_dp4a_gfx1100.hip` | y[row] via sh_coop | `quantize_coop (Q8_1) + coop_dp4a(sudot4/perm) + scale ls-32*d*low2float` | ✓ FLOWING (struct) | coop/stock cos 1.000 claimed 07-02, bench shows 0.942x not >1.2x |
| `impl_gemm_wmma_stream.hip` | Y[m*N+n] via wmma | `On-the-fly IQ4_XS->half d*(ls-32)*kvalues -> v16f16 + sB[2][32][33] B from X[gm*K+gk]` | ✓ FLOWING (struct) | Stride fix X[gm*K+gk]/Y[m*N+n] present, WMMA gated M>=512 fallback TILE_M=16, but perf 0.042x at M128 slower than stock |

### Behavioral Spot-Checks — Windows Git Bash (no GPU) + Hardware JSON Evidence

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| DP4A intrinsics in comparator | grep ggml_cuda_dp4a/__builtin_amdgcn_perm | `sudot4` L73 + 6x `perm` L111-118 + `ggml_cuda_dp4a_real` L68 present | ✓ PASS |
| LDS padding [32][33] + [2][32][33] | grep __shared__.*33 | `sh[32][33]` gemv + `sB[2][32][33]` gemm each found | ✓ PASS |
| launch_bounds+amdgpu | grep launch_bounds\|amdgpu_flat | 4 hits across hip+cuh (GEMV 1, GEMM 2) | ✓ PASS |
| WMMA builtin | grep wmma_f32_16x16x16 | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` hip L230 + cuh present | ✓ PASS |
| Switch gating OFF/ON | grep GGML_CUDA_ENABLE_CUSTOM_GFX1100 | OFF CMakeLists L221 + guards in cuh+mmq/mvq present, but gemm can_handle stub false | ⚠️ PARTIAL |
| bench --runs 10 CLI | grep runs.*10 bench_*.cpp | All three have `int runs=10` default + `--runs` parse + `runs:10` JSON emit | ✓ PASS |
| Patch git log base | `git -C llama.cpp log --oneline -5` | `5c6b397 feat(gfx1100)` over `bb4caa75` — base pinned | ✓ PASS |
| Patch file completeness | `wc -l patches/0001...` | **355 lines** (expected 355) — FIXED from 30-line truncation | ✓ PASS (structural) |
| bench_real_stock.hardware.json valid | `py -c json.load` + runs==10 | 8 entries valid, each `runs:10` + `real 99.547±28.56 p95 231.54` vs `naive 543.457` + `speedup 5.46x` | ✓ PASS — quoted 99us vs 543us |
| bench_gemv_dp4a.hardware.json valid | `py json.load` + speedup_median | 8 entries valid, avg 0.942, peak 1.048 ffn_up, all <1.2x — **honest FAIL** | ✓ PASS (evidence of FAIL, not fabricated) |
| bench_gemm_wmma.hardware.json valid | `py json.load` | **JSONDecodeError truncated at 12288B** line 389 — **TRUNCATED**, partial M128 0.042x M512 0.57x both FAIL | ✗ FAIL — must regenerate with timeout 90s |
| build_windows.bat gates | grep HIP_PATH\|clang++.*gfx1100\|-G Ninja\|curl.*8000 | All patterns present | ✓ PASS (file) |
| find py <=2 langs | `find -name *.py ! -path ./llama.cpp/*` | **40** (benchmarks/ + race.py + swizzle + output/) — **FAIL until Phase 8 prune** (deferred) | ✗ FAIL (deferred) |
| rows.jsonl N=10 | `wc -l rows.jsonl` + `CHECKSUMS.sha256` | 250 lines synthetic, checksum 8d6a943a matches, median 1.05 <1.10x — **not hardware** | ⚠️ SYNTHETIC |
| Thermal 90C aborts | cat logs/thermal_monitor.log | No real WSL2 log on this Windows host; protocol documents hwinfo_daemon 1Hz + thermal_watchdog 90C | ? SKIP (bare-metal) |

### Probe Execution

No `scripts/*/tests/probe-*.sh` declared for Phase 7 — kernel probes are `bench_* --runs 10 --json` executables requiring `HSA_ENABLE_DXG_DETECTION=1` gfx1100 bare-metal; skipped on this Windows host per WSL2 blind. `bench_gemm_wmma` hung 271s previously — future runs must use `timeout 90`.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| KERN-04 | 07-02 | Hybrid DP4A GEMV 8-thread/row v_dot4 coop, >1.2x vs real stock DP4A, >38 t/s N=10 | ✗ FAIL | Artifact PASS (coop+LDS+perm+ulong2 correct, cosine 1.000 claimed), speed FAIL 0.942 avg <1.2x (hardware) |
| KERN-05 | 07-03 | WMMA streaming GEMM 64x32 double-buffered [2][32][33] LDS + wmma, tile sweeps, >950 t/s N=10 | ✗ FAIL | Core PASS (wmma+sB+barriers), sweeps FAIL (P=4/XOR only comments), perf 0.042x M128 /0.57x M512 FAIL, can_handle stub |
| INTEG-02 | 07-04 | Quilt patch ON/OFF gating, paired llama-bench A/B protocol + gates green | ⚠️ PARTIAL | Gating present + patch 355 lines, but gemm disabled + no bench JSON on custom gates green N=10 |
| REQ-WIN-07 | 07-01..04 | Windows-native build_windows.bat + :8000 + <=2 langs | ✗ FAIL | Bat correct but not executed, server missing, py 40 (Phase 8 deferred) |
| REQ-PERF-07 | 07-02..04 | >=1.10x pp+tg at {512,1024,2048,4096,8192} median+mean-1sigma N=10 | ✗ FAIL | Synthetic 1.05 FAIL + prior 1.051x FAIL — correctly not fabricated |
| REQ-STAT-07 | 07-01..04 | N>=10 median/mean/stddev/p95 + N=15 LLM QA rigour | ✗ FAIL | Code supports --runs 10, bench_gemv valid, bench_gemm truncated, no N=15 LLM QA table |
| BENCH-01 amended | 07-01 | >=10 repeats pp/tg split RunStore+CHECKSUMS | ⚠️ HARNESS-READY but not complete | bench.h median/p95 present, bench_gemm hardware JSON incomplete |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `kernels/matmul_iq4xs/bench_gemm_wmma.hardware.json` | 389 | Truncated JSON at 12288B buffer (12KB) — incomplete write | 🛑 BLOCKER | Evidence incomplete for WMMA; prevents verifying N=10 GEMM. Regenerate with timeout 90s (avoid DXG deadlock) |
| `llama.cpp/.../gemm_iq4xs.cuh` | 88 | `custom_gemm_iq4xs_can_handle` stub `return false` | 🛑 BLOCKER | Disables WMMA dispatch — all GEMM falls back to stock, nullifying prefill uplift; patch ships disabled code |
| `kernels/.../bench_gemm_wmma.cpp` | 170-171 | Synthetic `v_median*=0.97/0.95` jitter simulates race | ⚠️ Warning | Race would report winner even though only one variant compiled; inflates variant diversity |
| `kernels/.../impl_gemm_wmma_stream.hip` | 143 | `// __shared__ _Float16 sB_P4[4][32][32];` P=4 variant as comment | ⚠️ Warning | High-yield variant not benchable; race cannot pick true winner |
| `benchmarks/results/phase7/rows.jsonl` | 1 | Future timestamps + uniform random jitter | ℹ️ Info | Honestly synthetic — correctly not claimed as hardware |

No `TBD/FIXME/XXX/HACK` markers in Phase 7 artifacts (grep clean). Data flows verified (no hardcoded empty to render).

### Human Verification Required

End-to-end uplift requires WSL2 gfx1100 bare-metal + Windows HIP SDK:

#### 1. Bare-metal N=10 microbench vs real DP4A 99us (bench_real_stock already valid, bench_gemv/gemm re-bench needed)
**Test:** `HSA_ENABLE_DXG_DETECTION=1 cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 && timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemv_dp4a --runs 10 --json > bench_gemv_N10.json' && timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json > bench_gemm_N10.json' && py -c "import json; d=json.load(open('bench_gemv_N10.json')); print(sum(1 for x in d if x['speedup_median']>=1.2),'shapes >=1.2x')"`
**Expected:** GEMV median >1.2x vs real DP4A 99us in >=6/8 shapes + mean-1sigma >1.15x; GEMM M=512 >1.2x + TFLOPS_median > stock; bench_gemm JSON valid (>12288B, not truncated); `llvm-objdump --mcpu=gfx1100 | grep v_dot4/v_wmma` + `VGPR <=64`
**Why human:** This Windows host has no hipcc/GPU; previous WSL2 hardware showed 0.942 avg under DXG jitter 15-30us, and gemm JSON truncated at 12KB — need bare-metal re-bench with timeout 90s to avoid 271s deadlock

#### 2. Paired llama-bench A/B thermal-paired N=10 5-tier sweep + N=15 LLM QA
**Test:** `race.py --repeats 10` interleaved `A,B,A,B` (not `AAAA BBBB`) stock OFF vs custom ON at `{512,1024,2048,4096,8192}` with `-ngl 99 -b 2048 --single-turn --simple-io --load-mode none -r 10` per tier per build in ONE thermal window, `hwinfo_daemon 1Hz + thermal_watchdog 90C`, `RunStore rows.jsonl + CHECKSUMS`, `VRAM preflight >2GB + hipMalloc probe` for 8192, plus `llama-cli --temp 0 fixed-prompt N=15` per-run 15-row table (avg tok/s, avg latency, stddev)
**Expected:** Custom median >=1.10x stock for BOTH `pp` and `tg` at every tier, `mean-1sigma >=1.10x`, per-tier PASS; 8192 SKIPPED with FA+GQA rationale if preflight fails; N=15 QA shows avg tok/s+latency+per-run table
**Why human:** Requires GPU+model GGUF + hwinfo + one thermal window; this host has no GPU/model

#### 3. Windows native build_windows.bat bare-metal execution
**Test:** On Windows 11 + VS Build Tools + HIP SDK 6.4 (`HIP_PATH`), run `build_windows.bat` — verify `where clang++.exe && clang++.exe --offload-arch=gfx1100 --version` + `where ninja` + `cmake -G Ninja -DHIP_PATH ...` builds `build-windows/bin/llama-server.exe` and `curl http://127.0.0.1:8000/v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}" -> 200` with `choices[0].message.content`
**Why human:** This Git Bash has no HIP SDK at `C:/Program Files/AMD/ROCm/6.4`, so build not executable here; also proves `cl` cannot compile `__builtin_amdgcn*`

#### 4. Quality gates + disasm/VGPR/rocprof per variant
**Test:** `cmake --build build-custom -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` then `run_op_gate.py --runs 10` (0 errors) + `run_model_gate.py --runs 10` (PPL 6.4271±1% + 6/6 canaries) + `hipcc --save-temps -Rpass-analysis | grep VGPR` + `llvm-objdump --mcpu=gfx1100 | grep v_wmma/v_dot4` + `rocprof --metric lds_bank_conflict` (expect 0, WSL2 blind)
**Why human:** Custom ON gates not executed (no hipcc/model); rocprof blind on WSL2 DXG

### Gaps Summary

Phase 7 honest execution re-scoped 2026-08-28 deliberately reports FAILs without fabrication:

* **One win verified:** Real DP4A comparator is hardware-faithful — `bench_real_stock.hardware.json` valid 8 entries `runs:10` proves **99.55us ±28.56 vs naive 543us =5.46x** via `sudot4+6x perm` (not naive). This is the sole `✓ VERIFIED` truth (1/7). `gemv_variant_xor.cuh` and `tools/swizzle_iq4xs.py` exist as offline helpers.
* **What blocks close (5 gaps + 1 truncated artifact):** (1) **GEMV >1.2x not proven** — hardware avg 0.942 <1.2x (peak 1.048, attn_q 0.965); DXG jitter 15-30us p95 148-343us flattens uplift; needs bare-metal 16 waves + XOR compiled variant. (2) **WMMA not proven** — core 64x32 P2+wmmma present but **P=4/XOR/64x64/128x32 only comments**, not compiled; **bench_gemm_wmma.hardware.json truncated at 12288B** (must regenerate via `timeout 90`); partial M128 0.042x / M512 0.57x both FAIL <1.2x; **can_handle stub disables dispatch** (patch ships gated off). (3) **Patch structurally complete (355 lines) but functionally disabled** by stub. (4) **Windows native not executed** (no HIP SDK/binary/curl 200, py 40 — deferred to Phase 8). (5) **≥1.10x pp+tg gate FAIL** — prior 1.051x +5.1% and synthetic 1.05 FAIL <1.10x (correctly not fabricated). (6) **REQ-STAT-07 harness-ready but unverified** — bench_gemm N=10 evidence broken by truncation, no N=15 LLM QA table.
* **No GPU numbers fabricated** — every tok/s/µs above is quoted from captured hardware JSONs (`bench_real_stock` 99.55 vs 543, `bench_gemv` 0.965 avg 0.942, `bench_gemm` partial 736 vs 17619 =0.042). Prior verification complained patch was 30 lines — fixed to 355; new blocker is the 12KB truncation (buffer limit) and the stub.

---
_Verified: 2026-08-29T14:45:00Z_
_Verifier: Muse Spark (gsd-verifier) on Windows Git Bash — WSL2 gfx1100 hardware JSONs quoted, not re-measured_
_Env: Windows Git Bash (MINGW64_NT), no hipcc, no /opt/rocm, no GPU; WSL2 ROCm 7.2.1 gfx1100 required for behavioral proof. bench_gemm_wmma.hardware.json truncated at 12288B — regenerate with timeout 90s._

