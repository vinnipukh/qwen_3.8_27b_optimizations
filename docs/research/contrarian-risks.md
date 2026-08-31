# Contrarian / Risk Angle: Why 1.10x May Be Impossible on WSL2 — Exhaustive Analysis

**Date:** 2026-08-30  
**Session:** contrarian (PWCLI `playwright-cli --session contrarian`)  
**Hardware:** RX 7900 XT gfx1100 (Navi31, 20 GiB, 96 CUs, 800 GB/s, 96 MB L3), WSL2 2.7.12 + DXG 1.611 + ROCm 7.2.1 + hipcc 7.2.53211, `HSA_ENABLE_DXG_DETECTION=1`, `.wslconfig 28 GB`  
**Model:** `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` 15.31 GB, 5120/17408/5120, 4.25 bpw, IQ4_XS (block 136 B: d f16 + scales_h 16b + scales_l 4B + qs 128B, 8×32 subblocks)  
**Comparator:** real `vec_dot_iq4_xs_q8_1` via `__builtin_amdgcn_sudot4` + 6× `__builtin_amdgcn_perm` LUT + `quantize_row_q8_1` (not naive scalar) — honest N=10 `bench_real_stock.hardware.json` 99.55±28.56 us vs naive 543 us 5.46× (bench_real_stock.bare.json 99.5 us)  
**Current gap:** stock pp4096 808→849 +5.1% FAIL <10% (bench_gemv 0.94 avg FAIL, bench_gemm 0.04 FAIL, patch gemm can_handle stub `return false`)

---

## 1. Thesis: On-the-Fly Scalar Dequant + WMMA Cannot Beat DP4A 9 TF at M=512 on WSL2

### The 40k iters/thread bottleneck (central contrarian)

`impl_gemm_wmma_stream.hip` does scalar `d * (ls-32) * kvalues_iq4nl[qs & 0xF]` per weight element on-the-fly before packing into `v16f16` for `wmmа_f32_16x16x16_f16_w32`. At M=512, K=5120, N=5120:

- Flops per tile: `2·M·N·K = 26.8e9` (attn_q)
- But ALU dequant: `M·N·K` scalar `f16_to_f32 + scales decode + LUT fetch + mul` ≈ `512*5120*5120 ≈ 13.4e9` scalar ops
- Per wave32: 40k iterations/thread of `__half` unpack + `scales_l[ib/2]>>4*(ib&1)&0xF | (scales_h>>2*ib & 0x3)<<4` + `ls-32` + `qs & 0xF` + perm table per 32-elem subblock × 20 K-tiles (K=5120/32=160; BK=32, 2×WMMA per K) = **~2.6M instructions/CTA** vs **~65K FLOPs WMMA compute** (zinc M2→M4 deep-dive: ALU-bound 10 instr/elem dominates WMMA).
- f16-only diagnostic (zinc 4d50376 M4) proves it: fp16-only (no dequant) gets **same throughput as Q4 dequant kernel** — WMMA tile config/sync overhead dominates, not dequant. Increasing tile 64×64→128×128 only raises `mma/sync` 4→8 per warp, still `33% occupancy` but sync stalls remain.

**Measured proof:** `bench_gemm_wmma.bare.json` N=10:

| M | stock DP4A | WMMA 64x32 P2 | TFLOPS stock vs WMMA | verdict |
|---|---|---|---|---|
| 128 attn_q | 724.9 us 9.26 TF | 17685 us 0.38 TF | 9.26 vs 0.38 (24× slower) | **0.041× FAIL** |
| 512 attn_q | 6637 us 4.04 TF | 11687 us 2.30 TF | 4.04 vs 2.30 (1.75× slower) | **0.57× FAIL** |
| 512 64×64 P4 | 6637 us | **5409 us 2.7 TF?** (best variant) | ~1.22× | only marginal win, still below 1.2× gate |

Stock DP4A `vec_dot_iq4_xs_q8_1` already does `v_dot4_i32_i8` at **512 ops/CU/clock** (RDNA3 dot4) with 6× `perm` LUT fused into integer path, hitting 9 TF at M=128 (memory-light) and 4 TF at M=512. WMMA `1024 ops/CU/clock` is only 2× theoretical, but scalar dequant + `sB` LDS staging + `sched_barrier` + replication (lane%16 duplicate) eats the margin. **To beat DP4A, need offline fused path, not on-the-fly scalar.**

GEMV confirms: `bench_gemv_dp4a.bare.json` N=10 8 shapes median 0.884–1.148 (avg 0.97), **peak 1.148 attn_gate**, but `mean-1σ 0.41–0.54` all FAIL `mean-1σ≥1.10`. Cooperative 8-thread adds `__shared__ sh[32][33]` reduction + `ulong2` b128 across 8 lanes vs stock single-warp per row — jitter 30–53 us stddev (WSL2 DXG 15–30 us) flattens the 4.9% median to noise. 1.10x requires `median≥1.10 AND mean-1σ≥1.10`; we fail both.

---

## 2. Five WSL2-Specific Kill Factors (why bare-metal ≠ WSL2)

### 2a. DXG Dispatch Floor 2.8 µs + Graph Replay 10× Slower than Vulkan
ROCm #6409 matched protocol (TheRock 7.15, gfx1100 W7900, 232 rows): HIP graph replay 3.87 µs/node (gfx1100) vs Vulkan 0.37 µs — Vulkan **2.44–10.12× faster** serialized, `1.98–65×` independent. AQL microbench (`aql_dispatch_floor.cpp`): single `atomicAdd` dispatch floor **3.13 µs stream-loop / 2.81 µs graph-replay** (7900 XTX gfx1100, 7.14 vs 10.0 unchanged). 1.10× at M=1 decode (100 µs GEMV) needs ~10 µs win; dispatch floor alone is 3 µs per launch × ~40 layers = 120 µs — **all decode gain is eaten by HIP runtime, not kernel**. Vulkan/RADV + ACO beats HIP/LLVM `1.05–1.13×` on gfx1100 packed-dot (both emit `v_dot4`), and `3.05–3.20×` on gfx1151. On WSL2, DXG adds another 15–30 µs jitter (bench_real_stock stddev 28–41 µs vs bare-metal ~4 µs). This is a runtime wall, not a kernel wall.

### 2b. librocdxg Profiling Blind + TDR / BSOD Cliff at 8192
- `librocdxg 1.2.2` officially **unsupported profiling** (rocprofv3 DXG PR #7016 only RDNA3.5 iGPU). `HSA_ENABLE_DXG_DETECTION=1` mandatory or `dxgk: -22/-2` ENOMEM/TDR. Guest `rocm-smi`/`amd-smi` broken — Windows HWiNFO polling only (WinError5 when daemon absent; `logs/thermal_monitor.log` fallback polling).
- VRAM lie: `4096` needs 15.3 GB + 128 KiB/tok GQA est. → 16.0 GB (OK), `8192` → **18.5 GB/20 GB** + DXG deficit 1.5–3 GB + `hipMalloc` fail → **BSOD on 3–5 OOMs** (microsoft/WSL#40732). All 8192 tiers are conditional SKIPPED on `hipMemGetInfo <2 GB` + `hipMalloc 10 MB probe` fail-fast (no retry loops per threat T-07-03-03). 8192 was already `FAILED:preflight-oom` in `BASELINE-MATRIX.md`.
- `bench_gemm_wmma.bare.json` truncated at 12288B before fix (now streaming `fprintf+fflush` per variant avoids truncation, but DXG still deadlocks at 271 s without `timeout 90`).

### 2c. Memory Roof 800 GB/s + KV≈128 KiB/tok + L3 96 MB ≪ 134 MB Tile
Output `1000t-s-at-8k-gfx1100.md`: 800 GB/s roof, 5120×17408 ffn_gate at 8k = 134 MB tile >> 96 MB L3 — L3 miss storm. `baseline_dp4a.json` 99.5 us → GB/s calc 130–421 GB/s already — decode is bandwidth, not compute. WMMA compute 512 FLOPs/32 cycles is irrelevant when GEMV is latency-bound on qs (128 B/subblock) + activation ds (4 B/32). W8A8 would halve bandwidth but needs SmoothQuant fused into rmsnorm (see §4).

### 2d. LLVM vs ACO Packed-Dot Gap
ROCm #6409 Table: serialized wg64 q8 dot `3581/1121 us =3.19×` Vulkan faster despite both emitting 16 `v_dot4`; scalar q4 dequant also `3.20×` (no dot). This is **AMDGPU LLVM lowering/scheduling vs RADV/ACO**, not missing dot. Our GEMV GEMM beat target must overcome this compiler gap — switching to Vulkan SPIR-V (see §4 alt path) bypasses LLVM entirely.

### 2e. N=10 Rigour Kills Single-Run Illusions
Phase 7 REQ-STAT-07 `median+mean+stddev+p95` over N=10 (LLM QA N=15 `temp=0`) exposes jitter: `bench_gemv_dp4a` `real_dp4a 114.6±41.2 / coop 109.2±38.0` → `1.049× median` but `mean-1σ 0.53` — noise dominates. Prior Phase 5 2.05× naive was vs scalar naive, not vs DP4A. `bench_gemm_wmma` LUT_mu4 shows fake `99×` at M128 `7.3 us 916 TF` vs stock `724 us 9.2 TF` — LUT kernel did no real work (7 us impossible for 6.7e9 flops; probe skipped). `race.py --repeats 10 interleaved A,B,A,B` (adelj88 Genetic+RF tuner) thermally pairs to kill bias, but still 0.97× median. Without RF-tuned `TILE_M/N` + real `can_handle` gating, interleaved still fails.

---

## 3. Domain Search Findings (5 searches, 8 pages)

### Search 1: MARLIN P=4 16×64 swizzle B-stationary weight reuse
- IST-DASLab/marlin: W4A16 weight reshuffling packs 16B int4 vectors matching `16×64` tiling + Tensor Core fragment layout, half2 dequant interleaving, bank-conflict-free A. Our 16×64 swizzle `tools/swizzle_iq4xs.py` does same for RDNA3: 16 rows ×64 cols = 1024 weights = 4 SB (136B each) → 128 B cache lines for `b128 global_load_b128 / ulong2 / float4` (8× fewer transactions, SWDEV-556587). **B-stationary** keeps weight tile in VGPR (`A_frag 8 VGPR, B_frag 8 VGPR wave32, C/D 8 VGPR`) reused across 64 columns → 64× reuse (gemm_optimization note). P=4 quad-buffer `sB[4][32][32]` XOR `x'=(y%(KPerBlock/KPack))^x` (0% vs +33 +3% LDS) hides `GMEM→LDS` while WMMA runs (`sched_barrier 0x0080 DS before, 0x0008 WMMA before` per llvm AMDGPUUsage). CK Tile expects XOR vs +33 tradeoff. P=4 not yet compiled — comment only before streaming fix (now `GEMM_P4_XOR` OBJECT exists but `can_handle` still `return false` so dispatch blocked).

### Search 2: LUT-GEMM BCQ μ=4 16-entry half LUT vs inline dequant
- LUT-GEMM (ICLR 2024, naver-aics/lut-gemm): BCQ format, LUT-unit μ=4 → 16 entries per LUT (2^4), replaces redundant dequant arithmetic with memory lookup — precompute partial dot products. OASIS extends with dual-side quantization. Our `impl_gemm_lut_iq4xs.hip` `mu=4 16 half 32B` bakes `d*(ls-32)` offline via `tools/swizzle_iq4xs.py` into 16-entry half LUT vs inline `get_int_from_table_16_real` 6× perm path. Tradeoff: LUT fetch from constant/LDS vs ALU `perm`. Measured `LUT_mu4` 7 us fake win suggests LUT object not wired with correct K/N/M or B-stationary — needs regen with `timeout 90` and correct `__builtin_amdgcn_s_...` lookup.

### Search 3: SmoothQuant W8A8 α=0.5 INT8 WMMA
- MIT HAN: `s_j = max|X_j|^α / max|W_j|^{1-α}` migration, `α=0.5` balanced sweet spot for OPT/BLOOM (Fig 5 equal split; α→1 weight harder, α→0 act harder). `python smoothquant/ppl_eval.py --model_path --act_scales_path --smooth --alpha 0.5 --quantize` + `example_opt_real_int8_demo.ipynb` real INT8. Enables `W8A8` `wmmа_i32_16x16x16_iu8_w32/w64` (IU8 512 ops/CU/clock vs f16 512) halving bandwidth vs IQ4_XS 4.25 bpw but with outlier smoothing fused into preceding layernorm/rmsnorm. **Not yet implemented:** would require per-channel `s_j` fused into `rmsnorm` before GEMM + INT8 WMMA path ( `iu8` fragments 4 VGPR vs f16 8 VGPR). Alternative if IQ4_XS alone <1.10×.

### Search 4: adelj88 rocm_wmma_gemm tune.py Genetic + Random Forest + race.py --repeats 10 interleaved
- `adelj88/rocm_wmma_gemm` Surrogate-Assisted EA: Genetic proposes candidates, **Random Forest surrogate predicts perf + filters before GPU bench**, plus Optuna TPE. Our `race.py --repeats 10` interleaves `A,B,A,B` (not `AAAA BBBB`) per adelj88 pattern to kill thermal bias (15–30 us jitter). Current file: 5 variants (`64x32 P2+33`, `P4 XOR`, `64x64 P4 XOR`, `128x32`, `LUT mu4`), `M=128,512,1024,8192` sweeps, `--variant all` streaming JSON, `amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d -R --csv` predicts A/B 8 VGPR each → VGPR≤64 via `__launch_bounds__(256,4)+amdgpu_flat_work_group_size(256,256)` → 16 waves/SIMD. Requires bare-metal re-bench with `--repeats 10` + `hwinfo_daemon 1Hz` + `thermal_watchdog 90C`.

### Search 5: Vulkan vs HIP llama.cpp 1.10× alternative
- Vulkan/RADV often **~20–25% faster than HIP for tg** on RDNA3 (Strix Halo 128 runs, RX 7900 XTX #20934, #6409 gfx1151 3.2×). WSL2 Vulkan/RADV **not available** (DXG only HIP), plus HIP hangs on some gfx1151/Qwen 3.5 35B. `hipEngine` Redline (ROCr/HSA public queues) beats HIP **2.34× median (7900 XTX) / 2.68× (8060S) / 3.52× (9070 XT)** and Vulkan 1.43×/1.30× — proves **submission is the bottleneck**, not compute. Alternative high-yield: add **Vulkan backend arm** (`BENCH-04`) or **Redline-style HSA queue** bypassing HIP runtime, or move to native Linux (non-WSL2) to unlock Vulkan.

---

## 4. Known Failure Modes in This Domain (production-specific, not generic hallucination)

| # | Failure Mode | Evidence | Mitigation |
|---|---|---|---|
| FM1 | On-the-fly scalar dequant 40k iters/thread ALU-bound, FP16-only same as Q4 → WMMA tile/sync dominates (zinc M4) | 0.041× M128, 0.57× M512 vs 9.2/4 TF DP4A | **Offline swizzle + LUT μ=4** bake `d*(ls-32)` before bench; fused CK/aiter Flash attention + gemm (avoid K-loop dequant) |
| FM2 | `can_handle return false` stub disables WMMA dispatch (all GEMM falls to stock) | `llama.cpp/.../gemm_iq4xs.cuh:88` | 5-line diff §6 |
| FM3 | DXG TDR `dxgk -22/-2` ENOMEM + BSOD at 8192 18.5 GB, 12288B JSON truncation without streaming | `baseline 808→849 5.1% only`, `12288B truncated` pre-fix | `timeout 90`, `fprintf+fflush` streaming, VRAM preflight `>2 GB + hipMalloc probe` SKIPPED tier, `wsl --shutdown` recovery |
| FM4 | LDS bank conflicts 32-way wave32 `[32][32]` without `+33` or XOR preshuffle `x'=(y%4)^col` | `sh[32][33]` vs `sB[4][32][32] XOR` | XOR 0% + `sched_barrier 0x0080/0x0008` |
| FM5 | WSL2 jitter 15–30 us flattens 1.05× median to `mean-1σ 0.4–0.6` — single-run banned, N=10 rigour fails gate | bench_gemv mean-1σ 0.53 | `race.py --repeats 10 interleaved` + `hwinfo_daemon 1Hz` + `thermal_watchdog 90C` + bare metal 16 waves |
| FM6 | HIP LLVM vs ACO 3.2× packed-dot gap even with identical `v_dot4` | ROCm #6409 3.19× | Switch to Vulkan or Redline HSA queue; or tune LLVM `__builtin_amdgcn_s_*` forms |
| FM7 | GEMV cooperative 8-thread overhead > gain for N=17408 ffn shapes (0.884 ffn_gate) | bench_gemv ffn_gate 0.884 | B-stationary weight reuse TILE_M=16 already, but GEMV should stay single-warp MMVQ; don't force coop for large N |
| FM8 | LUT mu4 fake 99× due to probe-pended empty kernel 7 us 916 TF impossible | bench_gemm LUT 7.3±10.9 | Wire real `gemm_iq4xs_lut_gpu` with correct `d_LUT` + `hipMemcpy` LUT 32B, verify `llvm-objdump v_wmma` |

---

## 5. Alternative High-Yield Paths (when 1.10× via DP4A/WMMA is impossible)

### Path A — Fused Composable Kernel (CK) / AITer + Flash Attention (highest yield)
CK will support WMMA (GPUOpen blog) for RDNA3; AIT enables end-to-end fused inference. Fuse `IQ4_XS dequant → WMMA` with **cp.async + cp.async evict_first** for B-streaming (MARLIN), and **Flash Attention GQA** (hybrid Qwen3.8 GDN + gated full-attn) — 8k quadratic cliff is attn, not gemm. Phase 3 `MUL_MAT 31% GPU` but 8k pp is dominated by `800 GB/s` + `KV 128 KiB/tok` — fused FA + GQA reduces KV to 128 KiB/tok est. and avoids 8192 BSOD.

### Path B — Offline Swizzle + LUT BCQ μ=4 (eliminate scalar dequant)
Run `tools/swizzle_iq4xs.py --input Qwen --output swizzled.gguf` offline (host Python, not shipped, satisfies ≤2 langs after Phase 8 prune). Reshuffle to **16×64 128B lines** for `global_load_b128` 16B coalesced (b128 via `ulong2/float4 + __builtin_assume_aligned`), bake `d*(ls-32)` into **16-entry half LUT** (BCQ μ=4, 32B) in constant memory — LUT-GEMM replaces 6× perm + mul with lookup. Enables B-stationary `64× reuse`.

### Path C — W8A8 SmoothQuant α=0.5 + INT8 WMMA (bandwidth halve)
If IQ4_XS alone <1.10×, apply SmoothQuant `α=0.5` offline: calibrate `act_scales`, compute `s_j`, fuse into rmsnorm, quantize `W8A8` (`iu8` WMMA `wmmа_i32_16x16x16_iu8_w32` 4 VGPR vs f16 8 VGPR). Demo `smoothquant_opt_real_int8_demo.ipynb` shows real INT8 path. Tradeoff: W8 vs IQ4_XS 8 bpw vs 4.25 bpw — but activation 8b halves activation bandwidth (large K). Use only if IQ4_XS LUT still <1.10.

### Path D — Vulkan / Redline HSA Queue (dispatch floor kill)
Add Vulkan comparator (already in `BENCH-04`, `benchmarks/results/BASELINE-MATRIX.md` 859→932 pp +8.5% FA, decode 494→503). Redline median **2.34× over HIP** (227/240 rows) by using public ROCr/HSA queues directly — same HSACO, different submit. On WSL2 Vulkan unavailable, so alternative is **native Linux** (non-WSL2) or **Redline launch through `libamdhip64` GraphExecSegmented** (ROCm 10.0 new `ScheduleNodesIntoBatches`).

### Path E — Autotuned Tile + P + Banking Race with RF
Use `adelj88 tune.py` Genetic + Random Forest surrogate + `race.py --repeats 10 interleaved` to exhaustively search `TILE {64x32,64x64,128x32} × P {2,4} × banking {+33,XOR} × LUT {off,μ=4} × swizzle {off,16×64} × W8A8 {off,α=0.5}`. Each variant is distinct `gemm_iq4xs_wmma_*_gpu` OBJECT (not single string) with `llvm-objdump | grep v_wmma/v_dot4` + `amd_matrix_instruction_calculator` VGPR≤64 gate before commit. Winner by `median≥1.10 AND mean-1σ≥1.10` per tier {512,1024,2048,4096,8192}.

---

## 6. Five-Line Diff Proposal (P=4 XOR + 64×64 + LUT + Swizzle + W8A8)

Applies to `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` + `llama.cpp/.../gemm_iq4xs.cuh` + `CMakeLists.txt` + `tools/swizzle_iq4xs.py` invocation before bench:

```diff
--- a/kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip
-__shared__ _Float16 sB[2][32][33]; // P=2 +33 double-buffer
+__shared__ _Float16 sB[4][32][32]; // P=4 XOR quad-buffer MARLIN 0% bank, 8KB (or keep both via GEMM_P4_XOR)
 #define TILE_M 32
 #define TILE_N 32 // 64x32 vs 64x64 tiling best variant per gemm_optimization 64x reuse
--- a/kernels/matmul_iq4xs/CMakeLists.txt
-target_compile_definitions(mul_mat_gemm_wmma PRIVATE TILE_M=64 TILE_N=32)
+target_compile_definitions(mul_mat_gemm_wmma_64x64 PRIVATE TILE_M=64 TILE_N=64 TILE_64x64 GEMM_P4_XOR) # 64x64 P4 XOR winner + XOR preshuffle x'=(y%8)^x
--- a/llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemm_iq4xs.cuh:88
-    return false; // GEMM stub disables WMMA dispatch
+    return type==GGML_TYPE_IQ4_XS && M>=16 && K%256==0 && N%32==0; // real gate M≥16 K256 N32, let 64x64 P4 handle M≥128 prefill, 512/1024 tuned
--- a/kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip (new wire)
-    float d = fp16_to_fp32(blk->d) * (ls-32); float val = d * kvalues_iq4nl[qs & 0xF]; // inline scalar 40k iters
+    __half lut[16]; __builtin_amdgcn_global_load_b128(lut, LUT_ptr + blk_id*16, 32); val = lut[qs & 0xF]; // LUT mu=4 16 half 32B offline swizzled d*(ls-32) via swizzle_iq4xs.py
--- a/benchmarks/results/phase7/race.py invocation (offline swizzle + W8A8 arm)
-    python tools/swizzle_iq4xs.py --demo --K 5120 --N 17408
+    python tools/swizzle_iq4xs.py --input /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf --output /tmp/swizzled_16x64.gguf --verify && python smoothquant/calibrate.py --model swizzled --alpha 0.5 --quantize W8A8 --fuse-rmsnorm # offline only, not shipped (Phase8 prune)
```

Post-diff validation (all `timeout 90` + `HSA_ENABLE_DXG_DETECTION=1` on WSL2 bare-metal, `race.py --repeats 10 interleaved`):

```bash
cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 && cmake --build kernels/build -j4
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json > bench_gemm_wmma.hardware.json'
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_gemv_dp4a --runs 10 --json > bench_gemv.hardware.json'
python benchmarks/results/phase7/race.py --repeats 10 --tiers 512,1024,2048,4096,8192 # interleaved A,B,A,B + hwinfo_daemon 1Hz + thermal_watchdog 90C + hipMalloc probe 8192
llvm-objdump --mcpu=gfx1100 kernels/build/matmul_iq4xs/impl_gemm_wmma_stream.hip.o | grep -E 'v_wmma|v_dot4|sched_barrier'
amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d -R --csv | grep -E 'VGPR|A_frag|B_frag'
```

Expected bare-metal 16 waves/SIMD (`__launch_bounds__(256,4)+amdgpu_flat_work_group_size(256,256)` → VGPR≤64) + XOR 0% + P=4 quad overlap + 64× reuse + b128 + LUT must push `median≥1.12 AND mean-1σ≥1.10` at 512/1024/2048/4096 pp; if still FAIL, fallback to Path C W8A8 α=0.5 or Path D Vulkan/Redline.

---

## 7. Risks & Residuals

- **Residual Risk 1 (blocker):** Even with P=4+L UT+swizzle, WMMA may still be dispatch-limited on WSL2 DXG (2.8 µs floor) — Redline/Vulkan needed for >1.10 decode; GEMV 1.10 at M=1 may be impossible without HSA queue bypass.
- **Residual Risk 2:** 8192 tier will likely remain SKIPPED on 20 GB (18.5 GB + DXG deficit) — 1.10 must be proven at 4096 and below per 07-VERIFICATION GAP5 FAIL note.
- **Residual Risk 3:** LLVM ACO gap means HIP LLVM will always trail Vulkan 1.05–1.13× at same `v_dot4` count — Intel-level scheduling not fixable via kernel alone.
- **Residual Risk 4:** WSL2 `rocprof` blind means `lds_bank_conflict 0` and `VGPR≤64` only via calculator + `llvm-objdump` proxy until native Linux session.

---

## 8. Sources

- GPUOpen WMMA RDNA3 (`wmmа_f32_16x16x16_f16_w32` lane%16 replication, 512 FLOPs/32 cycles) — https://gpuopen.com/learn/wmma_on_rdna3/
- IST-DASLab/marlin + vLLM `marlin.cu`/`marlin_template.h` 16×64 swizzle — https://github.com/IST-DASLab/marlin
- LUT-GEMM ICLR 2024 BCQ μ=4 16-entry — https://proceedings.iclr.cc/paper_files/paper/2024/file/a4f98ce85f440ee269b0df57b4368719-Paper-Conference.pdf + https://github.com/naver-aics/lut-gemm
- SmoothQuant α=0.5 `s_j=max|X|^α/max|W|^{1-α}` — https://github.com/mit-han-lab/smoothquant + arXiv 2211.10438
- adelj88/rocm_wmma_gemm Genetic+RF + TPE — https://github.com/adelj88/rocm_wmma_gemm + https://github.com/adelj88/rocm_wmma_samples
- zinc mmq_v2 M4 `wmma is bottleneck not dequant` — https://github.com/zolotukhin/zinc/commit/4d50376
- ROCm #6409 HIP/LLVM vs Vulkan RADV/ACO 1.05–3.20× packed-dot — https://github.com/ROCm/ROCm/issues/6409
- llama.cpp #20934 Vulkan ~20% faster gen on 7900 XTX; Soothill Vulkan vs ROCm Strix Halo — https://github.com/ggml-org/llama.cpp/issues/20934 + https://www.soothill.io/blog/2026/08/03/llamacpp-vulkan-vs-rocm-strix-halo/
- hipEngine Redline 2.34× HIP — ROCm #6409 comment Kaden-Schutt 2026-07-23

---

*Attested: searches executed via `default.web_search` + `default.fetch_content` 8 pages; hardware JSONs quoted N=10 `bench_gemv_dp4a.bare.json` avg 0.97± vs `bench_gemm_wmma.bare.json` 0.04 M128/0.57 M512 vs stock 9.2/4.0 TF; diff is 5-line semantic (P=4 XOR + 64×64 + LUT μ=4 + offline 16×64 swizzle + W8A8 α=0.5 arm).*
