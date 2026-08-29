---
phase: 07-hybrid-dp4a-wmma-kernel-optimization
verified: 2026-08-29T12:30:00Z
status: gaps_found
score: 2/7 must-haves verified
behavior_unverified: 1
overrides_applied: 0
gaps:
  - truth: "GEMV >1.2x median N=10 vs real vec_dot_iq4_xs_q8_1 DP4A (REQ-PERF-07 decode slice, REQ-STAT-07)"
    status: failed
    reason: "Cooperative GEMV kernel present and wired, but no hardware N=10 median proves >1.2x. Prior WSL hardware bench peaked 1.178x avg 1.00 under DXG jitter (07-02 SUMMARY). Current code adds XOR/b128 variants but no bench_gemv_dp4a --runs 10 --json output exists on this host; KERNEL-BENCH-DIFF §8 synthetic race median 1.085-1.18x is labeled synthetic/bare-metal pending, not hardware."
    artifacts:
      - path: "kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip"
        issue: "Present (15186B-16797B), LDS sh[32][33], launch_bounds 256,4+amdgpu 256,256, sudot4+perm, ulong2 b128 present; XOR variant only as helper cuh, not compiled as second object for race"
      - path: "kernels/matmul_iq4xs/bench_gemv_dp4a.cpp"
        issue: "CLI --runs 10 --json + speedup_median + mean-1sigma wires stock_dp4a vs coop, variant tag, but no emitted JSON on this Windows host (no hipcc/ROCm/GPU)"
      - path: "benchmarks/profiling/KERNEL-BENCH-DIFF.md"
        issue: "§8 table labeled synthetic race on Windows host, bare-metal pending; no real bench_gemv_dp4a JSON"
    missing:
      - "HSA_ENABLE_DXG_DETECTION=1 WSL2 gfx1100 bench_gemv_dp4a --runs 10 --json (N=10 median/mean/stddev/p95 per 8 shapes, vs real DP4A 84us denominator, showing >1.2x median and mean-1sigma >1.15x, winner XOR vs +33 picked)"
      - "llvm-objdump --mcpu=gfx1100 impl_gemv_dp4a_gfx1100.hip.o | grep v_dot4 + hipcc --save-temps -Rpass-analysis VGPR <=64 proof on gfx1100 (WSL2 bare-metal)"
  - truth: "Quilt patch git apply --check (real git diff HEAD over bb4caa75, GGML_CUDA_ENABLE_CUSTOM_GFX1100 OFF/ON gating intact)"
    status: failed
    reason: "Patch file on disk is truncated to 30 lines (vs expected ~355 lines / 276 insertions). Current diff only shows gemv/gemm can_handle stubs returning false, not the full vendored cuh + mmq/mvq + CMake gating. git apply --check would pass on the truncated file but patch does not represent the full quilt overlay."
    artifacts:
      - path: "patches/0001-gfx1100-mul-mat-custom.patch"
        issue: "30 lines on disk (expected 355 lines). Earlier 5c6b397 commit had full patch; current unstaged changes not regenerated into patch via git -C llama.cpp diff bb4caa75..HEAD"
      - path: "llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh"
        issue: "Vendored coop GEMV present but not captured in patch file"
      - path: "llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh"
        issue: "Vendored WMMA stream present but not captured; gemm can_handle currently stubbed to return false (disables WMMA path)"
    missing:
      - "Regenerate patches/0001-gfx1100-mul-mat-custom.patch via git -C llama.cpp diff bb4caa75 (or HEAD base) to include full 07-04 vendored winners (LDS [32][33]+launch_bounds+wmma+sudot4 with GGML fix X[gm*K+gk]/Y[m*N+n]) and verify git -C llama.cpp apply --check / git apply --check on both WSL2 (/opt/rocm) and Windows (core.autocrlf=false, *.patch eol=lf via .gitattributes)"
      - "Restore gemm can_handle to real shape gate (type==IQ4_XS && M>=16 etc) instead of unconditional return false"
  - truth: "REQ-WIN-07 Windows build_windows.bat (HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja, no cl, llama-server.exe :8000 ->200, <=2 langs)"
    status: failed
    reason: "build_windows.bat exists and is syntactically correct per gate (HIP_PATH, clang++ --offload-arch=gfx1100, -G Ninja, find_package hip via HIP_PATH, curl :8000 smoke). But Windows native build not executed on this host (no HIP SDK at C:/Program Files/AMD/ROCm/6.4, no clang++.exe, no build-windows/bin/llama-server.exe produced). <=2 langs gate fails: find -name \"*.py\" ! -path \"./llama.cpp/*\" == 40 (benchmarks/ Python harness + race.py + swizzle + output/ still present; Phase 8 prune pending, but Phase 7 now declares this must-have)."
    artifacts:
      - path: "build_windows.bat"
        issue: "Present (5857B), correct: HIP_PATH, where clang++.exe, clang++.exe --offload-arch=gfx1100 --version, -G Ninja, HIP_PATH/bin/clang++.exe for CXX/HIP, :8000 curl smoke, MODEL_PATH guard — but not executed (no HIP SDK on this Git Bash host)"
      - path: "kernels/CMakeLists.txt"
        issue: "find_package(hip REQUIRED CONFIG PATHS \"$ENV{HIP_PATH}/lib/cmake/hip\") present (line 17), no hardcoded /opt/rocm alone — PASS at code level"
      - path: "build-windows/bin/llama-server.exe"
        issue: "MISSING — not built on this host; no curl :8000 200 evidence"
    missing:
      - "Windows 11 bare-metal execution of build_windows.bat (HIP SDK 6.4, Ninja, VS Build Tools) producing build-windows/bin/llama-server.exe and curl http://127.0.0.1:8000/v1/chat/completions ->200 with choices[0].message.content on gfx1100 (or compile-gate + one llama-bench smoke tier proving toolchain parity)"
      - "Phase 8 prune to satisfy find -name \"*.py\" ! -path \"./llama.cpp/*\" ==0 (currently 40) — benchmarks/ Python harness is correctly offline-only but still counts until pruned; document as deferred to Phase 8 with explicit allowlist"
  - truth: "REQ-PERF-07 >=1.10x pp+tg at {512,1024,2048,4096,8192} N=10 median and mean-1sigma >=1.10x thermal-paired"
    status: failed
    reason: "No paired llama-bench A/B JSON exists. Prior real measurement 808->849 pp4096 is +5.1% (1.051x) FAILS gate (documented). Current synthetic race rows.jsonl median 1.085 across 250 synthetic repeats (<1.10x) and KBD §8 synthetic 512 1.08x FAIL / 1024 1.07x FAIL are labeled synthetic/bare-metal pending, not hardware. No ab_stock_*/ab_custom_* RunStore + CHECKSUMS with hwinfo_daemon 1Hz produced."
    artifacts:
      - path: "benchmarks/results/phase7/rows.jsonl"
        issue: "250 synthetic repeats with random uniform jitter, timestamps 1787995716 (future) — synthetic, not HSA_ENABLE_DXG_DETECTION=1 hardware; median 1.085 FAILS 1.10x"
      - path: "benchmarks/results/phase7/CHECKSUMS.sha256"
        issue: "Single checksum for synthetic rows.jsonl only; no ab_stock/ab_custom per-tier pp/tg split"
      - path: "benchmarks/profiling/KERNEL-BENCH-DIFF.md"
        issue: "§8 documents HONEST synthetic projection: 512 1.08x FAIL, 1024 1.09x FAIL, prior 808->849 +5.1% FAILS — correctly does not fabricate 1.10x"
      - path: "benchmarks/results/phase7/README.md"
        issue: "Notes winner 64x64_P4_XOR median 1.085 need >=1.10x — FAIL declared"
    missing:
      - "Paired llama-bench A/B stock OFF vs custom ON (same bb4caa75, -ngl 99 -b 2048 --single-turn --simple-io --load-mode none -r 10 per tier per build, ONE thermal window, interleaved race.py --repeats 10 A,B,A,B not AAAA BBBB, hwinfo_daemon 1Hz + thermal_watchdog 90C, RunStore rows.jsonl + CHECKSUMS.sha256, VRAM preflight >2GB + hipMalloc probe for 8192) proving median >=1.10x and mean-1sigma >=1.10x for both pp and tg at every tier 512..8192 (8192 SKIPPED with FA+GQA rationale if preflight fails)"
      - "High-yield variant that actually pushes 512/1024 over 1.10x on bare-metal (current BEST synthetic 1.085 <1.10x suggests 64x64 P4+XOR+b128+16x64 swizzle insufficient at 16 waves/SIMD; needs bare-metal tuning)"
  - truth: "WMMA tile sweep variants fully wired (64x32/64x64/128x32 + P=2 vs P=4 + XOR vs +33 + B-stationary + LUT mu=4, each benchable via --variant race)"
    status: failed
    reason: "Core WMMA 64x32 P=2 [2][32][33] + wmma_f32_16x16x16_f16_w32 + sched_barrier pinning is present and substantive, but high-yield sweeps are comments/docs, not compiled variants. P=4 sB[4][32][32] XOR is documented as future variant (comment only), template<TILE_M/TILE_N> sweep not present, 64x64/128x32 tilings not gated by compile flag. bench_gemm_wmma.cpp documents 5 variants but impl only implements one."
    artifacts:
      - path: "kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip"
        issue: "Has 64x32 P2 [2][32][33] + wmma builtin + sched_barrier 0x0080/0x0008 + b128 docs; but P=4 variant is comment __shared__ _Float16 sB[4][32][32] not compiled, no TILE_M/N template, no 64x64/128x32 gating"
      - path: "kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip"
        issue: "Present (7233B) LUT mu=4 secondary kernel present, but melded as separate file not as variant of main stream; still counts as LUT path"
      - path: "kernels/matmul_iq4xs/bench_gemm_wmma.cpp"
        issue: "Documents 5 variants (64x32_P2+33, P4_XOR, 64x64_P4_XOR, 128x32, LUT_mu4) with VRAM preflight, but impl only drives one codepath — race would compare identical code"
    missing:
      - "Compile-time variant gating in impl_gemm_wmma_stream.hip: #if TILE_M==64 etc or separate objects (matmul_gemm_wmma_64x64_hip etc) for 64x32/64x64/128x32, and real sB[4][32][32] XOR + sched_barrier P=4 path that builds clean via hipcc --offload-arch=gfx1100"
      - "llvm-objdump --mcpu=gfx1100 | grep v_wmma + hipcc --save-temps -Rpass-analysis VGPR <=64 per variant + rocprof lds_bank_conflict 0 on native bare-metal (WSL2 blind noted)"
behavior_unverified_items:
  - truth: "REQ-STAT-07 N>=10 rigour and LLM QA N=15 temp=0 are enforced end-to-end (bench_* --runs 10 + llama-bench N=10 per tier per split + 15x LLM QA with per-run tables)"
    test: "Inspect bench_* --runs 10 --json CLI defaults and run HSA_ENABLE_DXG_DETECTION=1 bench_real_stock/bench_gemv_dp4a/bench_gemm_wmma --runs 10 on WSL2 gfx1100; run paired llama-bench N=10 per tier per build in one thermal window with hwinfo_daemon 1Hz; run llama-cli --temp 0 fixed prompt N=15"
    expected: "Every number reported as median+mean+stddev+p95 (microbench + llama-bench) over N=10, winner picked by median N=10 interleaved A,B,A,B; LLM QA shows avg tok/s+avg latency+stddev+15-row per-run table; single-run claims absent"
    why_human: "Code now has --runs 10 wiring (verified via grep/bench.h), but this Windows Git Bash host has no ROCm/HIP/GPU/model (hipcc not found, /opt/rocm missing), so N=10 thermal-paired hardware execution and 15x LLM QA cannot be exercised here; rows.jsonl synthetic proves harness can emit N=10 but not that hardware numbers meet gate"
  - truth: "bench_real_stock 84us vs 543us DP4A is true hardware median N=10 (not single-run, not naive)"
    test: "HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json and check baseline_dp4a.json median 84.39 ±4.20 vs naive 543"
    expected: "median 84us ± stddev with p95, 6.4x speedup, real_stock_dp4a_comparator.hip grep shows ggml_cuda_dp4a + perm x6 not naive scalar"
    why_human: "Structure is correct (bench has --runs 10, JSON has runs:10 + stddev/p95, comparator has sudot4+perm grep), but bare-metal hipEvent timing not witnessed on this Windows host; we do NOT fabricate GPU numbers — values in baseline_dp4a.json are claimed, WSL2 DXG jitter note acknowledges 15-30us tax"
coincidental_reliance_items: []
human_verification:
  - test: "Bare-metal WSL2 gfx1100 N=10 microbench vs real DP4A (bench_real_stock, bench_gemv_dp4a, bench_gemm_wmma) + pair llama-bench A/B"
    expected: "bench_gemv_dp4a median speedup >1.2x (>38 t/s decode), bench_gemm_wmma >1.2x at M>=512 (>950 t/s prefill), paired llama-bench pp+tg >=1.10x median and mean-1sigma >=1.10x at 512..8192"
    why_human: "Requires WSL2 Ubuntu-24.04 ROCm 7.2.1 gfx1100 bare-metal, hipcc --offload-arch=gfx1100, model GGUF, hwinfo_daemon/thermal_watchdog; Windows Git Bash has no GPU/hipcc — intentionally not fabricated here"
  - test: "Windows native build_windows.bat full execution + :8000 smoke"
    expected: "HIP SDK 6.4 at HIP_PATH, Ninja + clang++.exe --offload-arch=gfx1100 -G Ninja builds build-windows/bin/llama-server.exe and curl :8000 ->200 choices[0].message.content"
    why_human: "Needs Windows 11 + VS Build Tools + HIP SDK installed; this host has no HIP SDK at C:/Program Files/AMD/ROCm/6.4 (path not found), so build not executable in this env — code inspection only"
  - test: "rocprof lds_bank_conflict 0 + calculator VGPR <=64 + llvm-objdump v_wmma/v_dot4 per variant"
    expected: "rocprof on native bare-metal shows lds_bank_conflict 0 for winner (WSL2 blind via librocdxg), calculator predicts A_frag 8/B 8/D 8 <=64 VGPR, llvm-objdump shows v_dot4 (GEMV) and v_wmma (GEMM)"
    why_human: "rocprofv3 is Instinct-only / 404 on WSL2 DXG; calculator offline-only; disasm needs .o from gfx1100 build"
---

# Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization — RE-SCOPED 2026-08-28 Verification Report

**Phase Goal:** Fuse Q8_1 integer activation quantization and RDNA3 hardware matrix cores (v_dot4_i32_i8 / v_wmma) with Wave32 cooperative workgroups to **beat real production stock llama.cpp by ≥10% end-to-end in llama-bench, Windows-native (≤2 langs), and with 10× (15× LLM QA) statistical rigour**.

**Verified:** 2026-08-29T12:30:00Z on Windows 11 Git Bash (no ROCm/HIP/GPU/model — WSL2 blind, `hipcc` not found, `/opt/rocm` missing). All GPU numbers below are *claimed* in-repo, **not** re-measured here; we do **not** fabricate tok/s.

**Status:** gaps_found
**Re-verification:** Yes — after gap closure attempt for 07-01..07-04 amended plans (previous 07-VERIFICATION 2026-08-27 score 2/5 on OLD 5-truth set; re-scoped 2026-08-28 to 7 truths, previous score maps to 2/7)

## Goal Achievement

### Observable Truths — 7 Re-Scoped Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | bench_real_stock N=10 median±stddev vs 84us DP4A (not naive 543us, not single-run) | ✓ VERIFIED (artifact) / ⚠️ bare-metal pending | `kernels/matmul_iq4xs/bench_real_stock.cpp` has `--runs 10` default (REQ-STAT-07), emits JSON with `runs:10`, `real_dp4a_median_us/mean/stddev/p95 + speedup_vs_naive`; `baseline_dp4a.json` has 8 entries each median 84.39±4.20..133±6.5 vs naive 543±8.2, `speedup 6.43x`; `BASELINE_DP4A.md` N=10 table median±stddev+p95 + 84vs543 proof; `real_stock_dp4a_comparator.hip` grep `ggml_cuda_dp4a_real` + `__builtin_amdgcn_sudot4` (L73) + 6x `__builtin_amdgcn_perm` (L111-118) + `vec_dot_iq4_xs_q8_1_device` (L145) + `ls decode d=half2float*low2float` — **not naive** (see bench.h BenchStats median/stddev/p95). Hardware N=10 not witnessed in this Windows env — **WSL2 bare-metal re-bench pending**, DXG jitter 15-30us noted |
| 2 | GEMV >1.2x median N=10 vs real vec_dot_iq4_xs_q8_1 DP4A (cooperative 8-thread Wave32) | ✗ FAILED | Present + wired: `impl_gemv_dp4a_gfx1100.hip` 16797B has `__launch_bounds__(256,4)+amdgpu_flat(256,256)` (L174), `__shared__ float sh[32][33]` (+33), `__builtin_amdgcn_sudot4` (L89) + `perm x6` (L109-116), `ulong2` b128 16B + `__builtin_assume_aligned`, `gemv_variant_xor.cuh` XOR helper `xor_preshuffle_32x33 (y%(32/8))^x` present; `bench_gemv_dp4a.cpp` has `--runs 10 --json + speedup_median + mean-1sigma` vs `gemv_iq4xs_stock_dp4a_gpu` (fair). **BUT** peak prior hardware 1.178x avg 1.00 <1.2x under WSL DXG (07-02 SUMMARY), no new `bench_gemv_dp4a --runs 10` JSON on this host (no hipcc/GPU), KBD §8 synthetic race median 1.085/1.12x is labeled synthetic, not hardware |
| 3 | WMMA streaming GEMM hardware wmma (64x32 double-buffered [2][32][33] LDS + wmma_f32_16x16x16_f16_w32) | ✓ VERIFIED (core) / ⚠️ variants incomplete | `impl_gemm_wmma_stream.hip` 16462B has `__launch_bounds__(256,4)+amdgpu(256,256)` x2 kernels (L30,L111), `__shared__ _Float16 sB[2][32][33]` (L142, +33 padded), `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (L230, wave32 OPSEL false lane%16), `sched_barrier 0x0080` (L162) + `0x0008` (L177), `v16f16/v8f32` + `lane%16 half_wave`, `K_TILE=32 2x WMMA per tile, coop 4 elem/thread` (L160-161), `b128 float4/ulong2 16B + assume_aligned + 16x64 swizzle` docs, `impl_gemm_lut_iq4xs.hip` LUT mu=4 16-entry half secondary present (7233B). **BUT** tile sweeps 64x64/128x32 and P=4 sB[4][32][32] XOR are **comments only** (L141), not compiled variants — fails full 07-03 sweep truth |
| 4 | Quilt patch git apply --check (real diff over bb4caa75, GGML_CUDA_ENABLE_CUSTOM_GFX1100 OFF/ON gating) | ✗ FAILED | `llama.cpp` log `bb4caa75 -> 5c6b397 feat(gfx1100)` exists, `.gitattributes` has `*.patch eol=lf` (core.autocrlf=false guard), `ggml/CMakeLists.txt:221 OFF` + `mmq.cu:3+112` + `mmvq.cu:2+1278` guards via `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)` verified, `CMakeLists.txt` has `find_package hip PATHS $HIP_PATH` (line17). **BUT** `patches/0001-gfx1100-mul-mat-custom.patch` on disk is **30 lines** (vs expected 355/276 insertions) — only gemm can_handle stub `return false` diff, not full vendored `custom_gfx1100/*.cuh` + mmq/mvq dispatch; unstaged changes `M` not regenerated. `git apply --check` would pass on truncated file but patch is **incomplete** |
| 5 | REQ-WIN-07 Windows build_windows.bat (HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja, not cl, :8000 ->200, <=2 langs) | ✗ FAILED | `build_windows.bat` exists 5857B and is **code-correct**: `HIP_PATH` (line6) + `clang++.exe --offload-arch=gfx1100 --version`, `where ninja` (errors if missing, notes `cl` cannot compile `__builtin_amdgcn*`), `cmake -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_CXX_COMPILER=%HIP_PATH%/bin/clang++.exe`, `find_package(hip REQUIRED CONFIG PATHS $HIP_PATH/lib/cmake/hip)` check, `build-windows/bin/llama-server.exe --help` + `curl :8000/v1/chat/completions ->200 choices` smoke with `MODEL_PATH` guard + `taskkill`. `kernels/CMakeLists.txt:15-17` has HIP_PATH search. **BUT** not executed: no HIP SDK at `C:/Program Files/AMD/ROCm/6.4` on this Git Bash, `build-windows/bin/llama-server.exe` MISSING, no curl 200 proof; `find -name *.py ! -path ./llama.cpp/*` == **40** (benchmarks/ Python + race.py + swizzle + output/) — fails `==0` until Phase 8 prune (correctly offline-only but still present) |
| 6 | REQ-PERF-07 >=1.10x pp+tg at {512,1024,2048,4096,8192} N=10 median and mean-1sigma >=1.10x thermal-paired | ✗ FAILED | Correctly **does not fabricate**: `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` + `docs/PUBLICATION.md` + `rows.jsonl` all label numbers synthetic/HONEST and show **FAIL**: prior real `808->849 pp4096 1.051x +5.1% FAILS` gate, synthetic race median `1.085 <1.10x` across 250 repeats, KBD synthetic 512 1.08x FAIL / 1024 1.07x FAIL / 2048 1.12x PASS but synthetic, 8192 conditional SKIPPED on VRAM preflight (FA+GQA 15.3GB+128KiB/tok ->18.5GB on 20GB, 3-5 OOMs->BSOD). `race.py --repeats 10` interleaved A,B,A,B (not AAAA BBBB) + `hwinfo_daemon 1Hz + thermal_watchdog 90C` + `RunStore+CHECKSUMS` harness exists but **no hardware ab_stock/ab_custom JSON** |
| 7 | REQ-STAT-07 N>=10 rigour (microbench + llama-bench N=10 median/mean/stddev/p95, LLM QA N=15 temp=0 per-run table, single-run banned) | ⚠️ PRESENT_BEHAVIOR_UNVERIFIED | Harness now **supports** N>=10: `bench_real_stock.cpp`, `bench_gemv_dp4a.cpp`, `bench_gemm_wmma.cpp` all default `--runs 10` + `BenchStats median/mean/stddev/p95` via `bench.h`; `BASELINE_DP4A.md` N=10 table `runs:10`; `bench_gemm_wmma.cpp` has VRAM preflight + `hipMalloc` probe + `TFLOPS_median`; `race.py` interleaved `--repeats 10` + `REPEATS` + `rows.jsonl` 250 lines proves N=10 plumbing; docs note `N=15 temp=0` fixed prompt `avg tok/s+latency+stddev+per-run table`. **BUT** hardware execution not witnessed in this Windows env (no GPU/hipcc/model) — `rows.jsonl` is synthetic random jitter (future timestamps 1787995716), no real `N=10` `llama-bench` or `N=15` `llama-cli` output. Counts as present+wired, behavior not exercised |

**Score:** 2/7 truths verified (Truth 1 artifact + Truth 3 core; 1 present behavior-unverified, 4 failed)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` | True upstream DP4A pipeline | ✓ VERIFIED | 25156B, `ggml_cuda_dp4a_real` via `sudot4` (L68-73) + 6x `perm` (L111-118) + `vec_dot_iq4_xs_q8_1_device` (L145) + `ls decode` |
| `kernels/matmul_iq4xs/bench_real_stock.cpp` | --runs 10 --json | ✓ VERIFIED | Parses `--runs` default 10, emits JSON array `runs:10` + `median/mean/stddev/p95 + GB/s + speedup_vs_naive` per 8 shapes, links `matmul_real_stock_hip` vs `matmul_stock_hip` fairly |
| `kernels/matmul_iq4xs/baseline_dp4a.json` | N=10 median±stddev 8 shapes | ✓ VERIFIED (structure) | 8 objects, each `runs:10` + `naive_median 543±8.2` vs `real 84.39±4.2` + `p95 114` + `GB/s 165` + `speedup 6.43x` — **claimed**, WSL blind, not re-measured here |
| `kernels/matmul_iq4xs/BASELINE_DP4A.md` | N=10 table + 84vs543 proof | ✓ VERIFIED | 62 lines, title N=10, table `median ± stddev` + `p95` + `GB/s` + `speedup`, interpretation `6.43x proves DP4A path`, Windows compile probe note, reproduce steps |
| `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` | Coop 8-thread DP4A GEMV | ✓ VERIFIED (artifact) | 16797B, 8 thr/row (256->32 rows), `sh[32][33]` L184, `launch_bounds(256,4)+amdgpu(256,256)` L174, `sudot4+perm`, `ulong2` 16B + `assume_aligned`, `gemv_variant_xor.cuh` XOR helper present |
| `kernels/matmul_iq4xs/gemv_variant_xor.cuh` | XOR preshuffle helper | ✓ VERIFIED | 2108B, `xor_preshuffle_32x33 (y%(32/8))^x` + `xor_preshuffle_32x32 (y%8)^x`, `#ifdef GEMV_XOR` gating, CK Tile refs |
| `kernels/matmul_iq4xs/bench_gemv_dp4a.cpp` | --runs 10 --json vs real DP4A | ✓ VERIFIED (artifact) | `--runs 10` default, `speedup_median` + `mean-1sigma`, `variant` field, links `matmul_gemv_dp4a_hip` vs `matmul_real_stock_hip` (not naive) |
| `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` | WMMA 64x32 streaming | ✓ VERIFIED (core) / ⚠️ variants | 16462B, `sB[2][32][33]` half L142, `wmmma_f32_16x16x16_f16_w32` L230, `sched_barrier 0x0080/0x0008` L162/L177, `v16f16/v8f32 lane%16`, but P=4/XOR/64x64/128x32 only as comments |
| `kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip` | LUT mu=4 secondary | ✓ VERIFIED | 7233B, `lut_entry 16 half 32B`, `B-stationary + LUT` + `sB[2][32][33]` + sched barriers, `__launch_bounds` present |
| `kernels/matmul_iq4xs/bench_gemm_wmma.cpp` | --runs 10 --json per-variant | ✓ VERIFIED (artifact) | `--runs 10` + `--variant all`, 5 variants table + `M={128,512,1024,8192}` + `TFLOPS_median` + `VRAM preflight >2GB + hipMalloc probe SKIPPED` logic, but only one impl path |
| `kernels/matmul_iq4xs/CMakeLists.txt` | matmul targets | ✓ VERIFIED | 6 libs + executables wired, `bench_real_stock` links both stock+real, HIP_PATH search line17 |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh` | Vendored GEMV coop | ✓ VERIFIED (file) | sh_coop[32][33] + launch_bounds + sudot4+perm present (not patch-covered) |
| `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh` | Vendored WMMA stream | ⚠️ STUBBED | sB[2][32][33] + wmma present, but `custom_gemm_iq4xs_can_handle` stubbed to `return false` (disables path) |
| `patches/0001-gfx1100-mul-mat-custom.patch` | Quilt overlay | ✗ STUB (30 lines) | Expected ~355 lines / 276 insertions over bb4caa75; currently only can_handle stub diff |
| `build_windows.bat` | Windows native gate | ✓ VERIFIED (file) / ✗ not executed | 5857B correct HIP_PATH/clang++/-G Ninja/:8000 smoke, but `build-windows/bin/llama-server.exe` MISSING, no curl 200 |
| `benchmarks/results/phase7/race.py` | race --repeats 10 | ✓ VERIFIED | 12096B, `--repeats 10` interleaved A,B,A,B (adelj88 pattern), 5 variants + `TIERS [512..8192]` + `hwinfo_daemon+thermal_watchdog` + `HSA_ENABLE_DXG_DETECTION=1` + median-1sigma gate |
| `benchmarks/results/phase7/rows.jsonl` | N=10 RunStore | ⚠️ SYNTHETIC | 250 lines synthetic random uniform jitter (future ts), median 1.085 <1.10x — proves harness shape, not hardware |
| `tools/swizzle_iq4xs.py` | offline 16x64 swizzle | ✓ VERIFIED | Exists, offline-only helper (correctly not shipped until Phase 8 prune notes it) |
| `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` | Hybrid provenance | ✓ VERIFIED | §8 exists with per-variant N=10 table + per-tier 1.10x verdict + synthetic HONEST note + stride fix + failed variants + produce commands |
| `docs/PUBLICATION.md` | Phase7 methodology | ✓ VERIFIED | Phase7 high-yield variant racing section + N=10 synthetic race + Windows gate note + hygiene |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `kernels/matmul_iq4xs/CMakeLists.txt` | `real_stock_dp4a_comparator.hip` | matmul_real_stock_hip OBJECT | ✓ WIRED | grep cmake library + bench linking |
| `bench_gemv_dp4a.cpp` | `impl_gemv_dp4a_gfx1100.hip` vs `real_stock` | `matmul_gemv_dp4a_hip` vs `matmul_real_stock_hip` fair | ✓ WIRED | CMake target wires bench vs real DP4A (not vs naive) |
| `bench_gemm_wmma.cpp` | `impl_gemm_wmma_stream.hip` vs `real_stock` | bench_gemm_wmma links `matmul_gemm_wmma_stream_hip` vs `matmul_real_stock_hip` | ✓ WIRED | Per-variant table + VRAM preflight wired |
| `mmvq.cu`/`mmq.cu` | `gemv_iq4xs.cuh`/`gemm_iq4xs.cuh` | `#if GGML_CUDA_ENABLE_CUSTOM_GFX1100` intercept | ✓ WIRED (code) / ✗ patch | Guards present in files, but `gemm can_handle` returns false disabling dispatch |
| `build_windows.bat` | `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100` | `-G Ninja + HIP_PATH` | ✓ WIRED (code) | Bat checks `where clang++.exe`, `where ninja`, errors if `cl` path |
| `race.py` | `bench_* --runs 10 --json` + `llama-bench N=10` | `race --repeats 10` interleaved picks winner by median N=10 | ✓ WIRED (harness) / ✗ hardware | Harness documents interleaving + RunStore+CHECKSUMS, but no hardware JSON yet |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `real_stock_dp4a_comparator.hip` | y[row] via vec_dot | `quantize_row_q8_1_standalone (amax/127, ds half2) + ggml_cuda_dp4a/perm DP4A` | ✓ FLOWING | Cosine 0.999985 vs FP64 oracle (claimed, prior hardware) |
| `impl_gemv_dp4a_gfx1100.hip` | y[row] via sh_coop | `quantize_coop (Q8_1) + coop_dp4a(sudot4/perm) + scale ls-32*d*low2float` | ✓ FLOWING (struct) | coop/stock cos 1.000 claimed 07-02, microbench measured peak 1.178 but not >1.2 |
| `impl_gemm_wmma_stream.hip` | Y[m*N+n] via wmma | `On-the-fly IQ4_XS->half dequant d*(ls-32)*kvalues -> v16f16 + sB[2][32][33] B from X[gm*K+gk]` | ✓ FLOWING (struct) | Stride fix applied via vendored cuh, WMMA gated M>=512 (fallback TILE_M=16) |

### Behavioral Spot-Checks — Windows Git Bash (no GPU)

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| DP4A intrinsics in comparator | grep ggml_cuda_dp4a/__builtin_amdgcn_perm| sB[2][32][33] validation / loops vs 10k lines where needed. | `sudot4` + 6x `perm` + `ggml_cuda_dp4a_real` present (L68,111-118,158) | ✓ PASS |
| LDS padding [32][33] + [2][32][33] | grep __shared__.*33 | `sh[32][33]` (gemv) + `sB[2][32][33]` (gemm) + cuh x2 each | ✓ PASS |
| launch_bounds+amdgpu | grep launch_bounds\|amdgpu_flat | 4 hits across hip+cuh (GEMV 1, GEMM 2) | ✓ PASS |
| WMMA builtin | grep wmma_f32_16x16x16 | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` in hip L230 + cuh | ✓ PASS |
| Switch gating OFF/ON | grep GGML_CUDA_ENABLE_CUSTOM_GFX1100 | OFF in CMakeLists L221, guards in cuh+mmq/mvq + gemm can_handle (but stubbed false) | ⚠️ PARTIAL |
| bench --runs 10 CLI | grep "runs.*10\|--runs" bench_*.cpp | All three have `int runs=10` default + `--runs` parse + `runs:10` in JSON emit | ✓ PASS |
| Patch git log base | `git -C llama.cpp log --oneline -5` | `5c6b397 feat(gfx1100)` over `bb4caa75` — base pinned | ✓ PASS |
| Patch file completeness | `wc -l patches/0001...` | **30 lines** (expected 355) — **TRUNCATED** | ✗ FAIL |
| build_windows.bat gates | grep HIP_PATH\|clang++.*gfx1100\|-G Ninja\|curl.*8000 | All patterns present in bat | ✓ PASS (file) |
| find py <=2 langs | `find -name *.py ! -path ./llama.cpp/*` | **40** files (benchmarks/, tools/, output/) — **FAIL until Phase 8 prune** | ✗ FAIL (deferred) |
| rows.jsonl N=10 | `wc -l rows.jsonl` + `CHECKSUMS.sha256` | 250 lines synthetic, checksum matches rows.jsonl only | ⚠️ SYNTHETIC |
| Thermal 90C aborts | cat logs/thermal_monitor.log (if exists) | No real WSL2 log on this Windows host; protocol documents `hwinfo_daemon 1Hz + thermal_watchdog 90C` | ? SKIP (WSL2 needed) |

### Probe Execution

No `scripts/*/tests/probe-*.sh` declared for Phase 7 — kernels probes are `test_*` executables (require `hipcc`/`HSA_ENABLE_DXG_DETECTION=1` gfx1100 bare-metal; skipped on this Windows host per WSL2 blind).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| KERN-04 | 07-02 | Hybrid DP4A GEMV 8-thread/row v_dot4 coop, >1.2x vs real stock DP4A, >38 t/s N=10 | ⚠️ PARTIAL | Artifact PASS (coop+LDS+perm+ulong2 correct, cosine 1.000 claimed), speed target FAIL (1.178 peak <1.2) |
| KERN-05 | 07-03 | WMMA streaming GEMM 64x32 double-buffered [2][32][33] LDS + wmma, tile sweeps, >950 t/s N=10 | ⚠️ PARTIAL | Core PASS (wmma+sB+barriers present), sweep variants FAIL (P=4/XOR/64x64 only comments) |
| INTEG-02 | 07-04 | Quilt patch ON/OFF gating, paired llama-bench A/B protocol + gates green | ⚠️ PARTIAL | Gating code present but patch truncated + gemm disabled + no bench JSON on custom |
| REQ-WIN-07 | 07-01..04 amended | Windows-native build_windows.bat + :8000 + <=2 langs | ✗ FAIL | Bat file correct but not executed, server missing, py count 40 |
| REQ-PERF-07 | 07-02..04 amended | >=1.10x pp+tg at {512..8192} median+mean-1sigma N=10 | ✗ FAIL | Synthetic shows <1.10x, prior 808->849 1.051x FAILS — correctly not fabricated |
| REQ-STAT-07 | 07-01..04 amended | N>=10 median/mean/stddev/p95 + N=15 LLM QA rigour | ⚠️ HARNESS-READY but hardware unverified | Code supports --runs 10, but hardware N=10/15 not executed here (WSL2 blind) |
| BENCH-01 amended | 07-01 | >=10 repeats pp/tg split RunStore+CHECKSUMS | ⚠️ WIRED but not run | bench.h median/p95 present, rows.jsonl synthetic only |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `patches/0001-gfx1100-mul-mat-custom.patch` | header | Truncated patch (30 lines vs 355) | 🛑 BLOCKER | Patch does not represent quilt; git apply --check on truncated file hides missing vendoring |
| `llama.cpp/.../gemm_iq4xs.cuh` | L88 | `custom_gemm_iq4xs_can_handle` stub `return false` | 🛑 BLOCKER | Disables WMMA dispatch — all GEMM would fall back to stock, nullifying prefill uplift |
| `kernels/.../impl_gemm_wmma_stream.hip` | L141 | P=4 variant as comment not code | ⚠️ Warning | High-yield variant not benchable; race compares identical code |
| `benchmarks/results/phase7/rows.jsonl` | 1 | Future timestamps + uniform random | ℹ️ Info | Honestly synthetic — correctly not claimed as hardware, but inflates progress appearance |

No `TBD/FIXME/XXX/HACK` markers in Phase 7 artifacts (grep clean). No hardcoded empty data flows to render (data flows verified).

### Human Verification Required

All automated checks that can run on Windows Git Bash pass at code level; end-to-end uplift requires WSL2 gfx1100 bare-metal:

#### 1. Bare-metal N=10 microbench vs real DP4A 84us
**Test:** `HSA_ENABLE_DXG_DETECTION=1 cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 && cmake --build kernels/build -j4 && ./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json > baseline_dp4a.json && ./kernels/build/matmul_iq4xs/bench_gemv_dp4a --runs 10 --json > bench_gemv_N10.json && ./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json > bench_gemm_N10.json` then `python3 -c "import json; d=json.load(open('bench_gemv_N10.json')); print([x['speedup_median'] for x in d])"`
**Expected:** GEMV median >1.2x vs real DP4A 84us across >=6/8 shapes (peak already 1.178 -> needs bare-metal 16 waves/SIMD), GEMM M=512 >1.2x + TFLOPS, baseline 84±stddev proven DP4A not naive
**Why human:** This Windows host has no hipcc/GPU; WSL2 DXG jitter 15-30us flattens 1.178->1.00 per RESEARCH; bare-metal needed to see true 1.2x and to generate llvm-objdump v_dot4/v_wmma + VGPR <=64

#### 2. Paired llama-bench A/B thermal-paired N=10 5-tier sweep
**Test:** `race.py --repeats 10` interleaved `A,B,A,B` (not `AAAA BBBB`) across `stock OFF` vs `custom ON` at `{512,1024,2048,4096,8192}` with `-ngl 99 -b 2048 --single-turn --simple-io --load-mode none -r 10` per tier per build in ONE thermal window, `hwinfo_daemon 1Hz + thermal_watchdog 90C`, `RunStore rows.jsonl + CHECKSUMS`, `VRAM preflight >2GB + hipMalloc probe` for 8192
**Expected:** Custom median >=1.10x stock for BOTH `pp` and `tg` at every tier, `mean-1sigma >=1.10x`, per-tier verdict PASS; 8192 SKIPPED with FA+GQA rationale if preflight fails (avoid 3-5 OOMs->BSOD)
**Why human:** Requires GPU+model GGUF + hwinfo share mem + one thermal window; this host has no GPU/model/hwinfo

#### 3. Windows native build_windows.bat bare-metal execution
**Test:** On Windows 11 + VS Build Tools + HIP SDK 6.4 (`HIP_PATH`), run `build_windows.bat` — verify `where clang++.exe && clang++.exe --offload-arch=gfx1100 --version` + `where ninja` + `cmake -G Ninja -DHIP_PATH ...` builds `build-windows/bin/llama-server.exe` and `curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}" -> 200` with `choices[0].message.content`
**Why human:** This Git Bash has no HIP SDK at `C:/Program Files/AMD/ROCm/6.4` (path missing), so build not executable here; also proves `cl` cannot compile `__builtin_amdgcn*` (Ninja+clang++ mandatory)

#### 4. Window 8 verification (Phase 8 landing REQ-WIN-07) | Winner
**Test:** `find . -name "*.py" -maxdepth 4 -not -path "./.venv/*" -not -path "./node_modules/*" -not -path "./.git/*"`

**Expected**: `find` should show 0 Python files except `tooling`/`pre-commit` and `find ./__pycache__` clean; validate `VCXPROJ` build logs show 0 errors from `llama-server>`.

**Why human :**: `find ... ! -path "./llama.cpp/*" ==0` currently 40 due to benchmarks/ Python harness (correctly offline-only) — Phase 8 is the execution phase that closes REQ-WIN-07 via prune; verify after `make clean` / `prune` shows 0 files left.

- **Risk if skipped**: Current `<=2` langs gate fails; shipping with `benchmarks/` Python would violate `REQ-WIN-07` (pure C++/HIP+CMake only).

#### 5. Quality gates + disasm/VGPR/rocprof per variant
**Test:** `HSA_ENABLE_DXG_DETECTION=1 cmake --build build-custom` (`-DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON`) then `run_op_gate.py --runs 10` (0 errors) + `run_model_gate.py --runs 10` (PPL 6.4271±1% + 6/6 canaries) + `hipcc --save-temps -Rpass-analysis | grep VGPR` + `llvm-objdump --mcpu=gfx1100 | grep v_wmma/v_dot4` + `rocprof --metric lds_bank_conflict` (expect 0 on winner, WSL2 blind documented)
**Why human:** Custom ON gates not executed here (no hipcc/model); rocprof blind on WSL2 DXG

### Gaps Summary

Phase 7 artifactual work is **substantially wired but not yet gated**:

* **Code-correct wins:** Real DP4A comparator is hardware-faithful (84us vs 543us, `sudot4+6x perm`, structure N=10), both hybrid kernels implement mandated microarchitectural patterns at the 64x32 P2 baseline (GEMV coop 8-thread/row LDS [32][33] `launch_bounds(256,4)` `sudot4+perm` `ulong2 b128`; GEMM WMMA `sB[2][32][33]` `wmmma_f32_16x16x16_f16_w32` `sched_barrier 0x0080/0x0008`), `build_windows.bat` is syntactically correct per REQ-WIN-07 (HIP_PATH+clang++/-G Ninja/:8000), `race.py --repeats 10` and `bench_* --runs 10` harness correctly implements `N=10` median/mean/stddev/p95 + interleaved thermal-bias kill and **honestly reports failure** (1.085 <1.10x, prior 808->849 1.051x FAILS).
* **What blocks Phase 7 close (5 gaps):** (1) GEMV >1.2x never breached on hardware (1.178 peak <1.2, synthetic only); (2) quilt patch truncated 30 vs 355 lines + gemm `can_handle` stubbed false disabling WMMA dispatch; (3) Windows native not executed (no HIP SDK/binary/curl 200, py count 40); (4) >=1.10x pp+tg at 5 tiers not proven — synthetic median 1.085 <1.10 and prior real 5.1% FAIL (correctly not fabricated); (5) high-yield tile sweeps P=4/XOR/64x64/128x32 are docs not compiled variants, so race cannot pick a true winner that pushes 512/1024 over 1.10. Infrastructure for `N=10/N=15` exists in code but hardware behavior unverified on this Windows host (intentionally not fabricated).
* **Deep-research implication:** `800 GB/s` roof + `KV ~128 KiB/tok GQA -> 8192 ~18.5GB on 20GB` + WSL2 `800 GiB` lie + BSOD risk + `rocprof` blind + `15-30us` DXG jitter are all gated by `8192 conditional SKIPPED` and `one thermal window hwinfo+watchdog` — the harness is correct; the **bare-metal WSL2 gfx1100 re-bench** is the sole closer (plus Windows `HIP_PATH` build). No GPU numbers were fabricated in this verification — every tok/s is either a prior real 808->849 or labeled synthetic.

---
_Verified: 2026-08-29T12:30:00Z_
_Verifier: Claude (gsd-verifier) on Windows Git Bash without ROCm/HIP — note WSL2 blind_
_Env: Windows Git Bash (MINGW64_NT), no hipcc, no /opt/rocm, no GPU; WSL2 Ubuntu-24.04 ROCm 7.2.1 gfx1100 required for behavioral proof_
