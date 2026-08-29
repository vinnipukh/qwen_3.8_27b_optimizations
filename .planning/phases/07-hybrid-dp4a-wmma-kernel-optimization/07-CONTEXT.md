# Phase 7: Hybrid DP4A & Matrix Core Optimization — Context

**Date:** 2026-08-25
**Goal:** Outperform real production `llama.cpp` stock kernels on AMD Radeon RX 7900 XT (`gfx1100`) end-to-end by fusing `Q8_1` integer activation quantization and RDNA3 hardware matrix cores (`v_dot4_i32_i8` / `v_wmma`) with our Wave32 cooperative workgroup architecture.

## Background & Post-Mortem Findings

In Phases 4–6, custom kernels were benchmarked and proven against `stock_hip_comparator.hip` (a naive scalar float reference loop) achieving 2.13× GEMV and 9.27× GEMM microbenchmark speedups. However, real production `llama.cpp` uses:
1. **On-the-fly `Q8_1` Quantization:** Converts float activations to 8-bit integers, cutting activation memory traffic by $4\times$.
2. **Hardware DP4A SIMD (`v_dot4_i32_i8`):** Computes 4-way INT8 dot products in a single clock cycle.
3. **Multi-Warp Register Blocking in MMQ:** 4-way unrolled DP4A matrix multiplication.

Our initial Phase 5/6 kernel operated directly on unquantized 32-bit floats with double-precision accumulation to guarantee numerical parity (`cosine = 1.0`), which incurred a $4\times$ memory bandwidth penalty and lower arithmetic density than stock DP4A.

## Architectural Opportunities on gfx1100

1. **Decode Opportunity (GEMV $M=1$):**
   - Stock `mmvq.cu` restricts RDNA3 (`MMVQ_PARAMETERS_RDNA3_0`) to a single warp per row for IQ4_XS (`calc_nwarps` returns 1).
   - Our 8-thread cooperative decomposition (256 threads $\to$ 32 output rows per block) combined with DP4A instructions (`__dp4a` / `v_dot4_i32_i8`) can achieve higher occupancy and full memory coalescing over stock.
   - Expected Target: **>40 t/s** (vs Stock 34.8 t/s).

2. **Prefill Opportunity (GEMM $M \ge 128$):**
   - Stock `mmq.cu` runs integer DP4A on general shader ALUs (peak 512 ops/CU/clock).
   - RDNA3 Wave32 hardware WMMA matrix cores (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` or `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`) execute matrix multiplication directly in tensor registers at **1024 ops/CU/clock** ($2\times$ higher compute density).
   - Expected Target: **>1000 t/s** (vs Stock 880 t/s).

## Re-scope 2026-08-28 — Owner 3 Wishes + Deep-Research Report `output/deep-research/1000t-s-at-8k-gfx1100.md`

Exhaustive `playwright-cli` research (5 sessions: `overview`/`technical`/`prefill-long`/`quant-kernels`/`contrarian`, 40+ URLs, 28 `innerText` slices) found `8k` is a **quadratic cliff, not a linear scale**: `67M (8k) → 1B (32k)` per head `16×`, `Flash2` makes `HBM O(N)` but `FLOPs` stays `N²` → `8k→32k 3–4×` latency *with* Flash (50–73% GEMM on A100, 225 TFLOP/s) vs `16×` without, `7900 XT 800 GB/s` pin caps `1000 tok/s` (naive `15.3 TB/s` needed), `KV≈128 KiB/tok GQA` (Qwen 64L/8KV) → `8k≈1–2 GB` + `15.3 GB` = `18.5 GB` on `20 GB`, WSL2 adds `16 GiB` invisible `DXGI/Hyper-V` overhead where `80 GiB` reported vs `3.48 GiB` contiguous fails + `BSOD` after `3–5 OOMs` (`WSL #40401`/`#40732`), `15–30 µs` DXG jitter flattens `1.178×→1.0×`, `rocprofv3`+`librocdxg` unsupported (404s, Instinct-only). Prior `808→849 pp4096 +5.1%` is *expected* short of `10%` without fully-fused `CK`/`aiter` Flash + `WMMA 64×32 double-buffered [2][32][33]+33` + `GQA 4×` + `b≥2048 --flash-attn on`.

**Owner must-have outputs #1–#3 (now binding for Phase 7):**

1. **Windows-native, ≤2 langs (REQ-WIN-07)** — `build_windows.bat` via `HIP_PATH` + `clang++.exe --offload-arch=gfx1100` + `-G Ninja`, pure `C++/HIP` + `CMake` only (`find -name "*.py" ! -path "./llama.cpp/*" == 0`), `llama-server.exe` serves `localhost:8000` on `gfx1100` without `3+` language servers. Phase 8 is the *landing* phase that closes this.
2. **≥10% end-to-end uplift (REQ-PERF-07)** — `llama-bench` `pp+tg` at `{512,1024,2048,4096,8192}` (if VRAM pre-flight passes) must be `median ≥1.10× stock`, `mean−1σ ≥1.10×` over `N=10` thermal-paired runs — microbench `>1.2×` vs real DP4A alone is insufficient.
3. **10× averaged, 15× LLM QA (REQ-STAT-07)** — every perf/quality number in Phase 7 is `N≥10` (`median`+`mean`+`stddev`+`p95`), LLM question tests `N≥15` (`temp=0` fixed prompt, `avg tok/s`+`latency`+per-run table). Single-run claims banned; amends `BENCH-01` `≥3→≥10`.

These add `Success Criteria 5–7` in `ROADMAP.md` and new requirements `REQ-WIN-07`/`REQ-PERF-07`/`REQ-STAT-07` in `REQUIREMENTS.md`. All four plans below are amended to carry them.

## High-Yield Design Notes (added 2026-08-28 — `output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` + 3 PDFs)

- **rocWMMA:** Header-only `rocwmma/rocwmma.hpp` — no runtime, `CUDA WMMA`-compat, `MFMA` (CDNA) + `WMMA/SWMMAC` (RDNA3/4). `amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d` predicts `A_frag 8 VGPR / D 8 VGPR wave32` → `≤64 VGPR` gate before code lands.
- **LDS:** `32 banks×4B`, `bank=(addr/4)%32`, `ds_write_b128` `8 phases×8 lanes` (`0~7…56~63`) conflict-free iff consecutive; `ds_read_b128` vertical `0:3+20:23` etc. `4-way` → `-75%` BW. Fix: `+1` padding `[32][33]` `+3%` (our default) **or** `XOR preshuffle` `x'=(y%(K/8))⊕x` `0%` (CK `TileWindow`). `P=4` pipeline `GMEM→VGPR→LDS→VGPR→WMMA` hides `GMEM` while `WMMA` runs (`sched_barrier 0x0080/0x0008`).
- **GEMM tiling:** `256 thr` `16×16` naive `512K` loads vs ideal `32K` (`16×`), `loads/out=K·(1/M+1/N)` → `T=64→64×` reduction. `MI300` `256×64→304` CUs perfect occupancy; for `gfx1100` `96` CUs `64×64` → `128×128` blocks at `8192`. `B-stationary` (weight in `VGPR`, stream `A` via `LDS`/`ping-pong`) + `16×64` offline swizzle → `128B` `b128` coalesced (`float4`/`ulong2` `16B`, `32 thr×4B→8×16B`).
- **Tuning:** `adelj88/rocm_wmma_gemm` `tune.py` `Genetic + Random Forest surrogate` + `race.py --repeats 10` interleaved (thermal-bias kill) — template for `REQ-STAT-07` `10×`/`15×`.
- **Custom PDFs:** `MARLIN` `P=4` `16×64` swizzle `B` in registers; `LUT-GEMM` `BCQ` `Σαb` `μ=4` `16-entry LUT` eliminates inline dequant; `SmoothQuant` `s_j=max|X_j|^α/max|W_j|^{1-α}` `α=0.5` fused into `rmsnorm` enables `W8A8` `INT8 WMMA` arm.

## Plans in Phase 7 (amended 2026-08-28)

- **07-01:** Real Upstream Microbenchmark Comparator (Wire true `vec_dot_iq4_xs_q8_1` and `quantize_row_q8_1` into `kernels/matmul_iq4xs/`) — **now `N=10` averaged + `stddev` table, LLM QA rigour documented**.
- **07-02:** Custom Cooperative Wave32 DP4A GEMV Kernel (`impl_gemv_dp4a_gfx1100.hip`) — **must prove `>1.2× median` + `10×` table, `15×` LLM QA slice, Windows `hipcc` probe**.
- **07-03:** Custom Hardware WMMA Matrix Core GEMM Kernel (`impl_gemm_wmma_q8.hip`) — **must prove `10×` table + `>950 t/s` median, Windows `hipcc` probe, `64×32 [2][32][33]` audit**.
- **07-04:** Quilt Patch Integration & Paired End-to-End A/B Benchmark Verification (`llama-bench`) — **now adds Windows-native gate (`build_windows.bat` + `llama-server :8000`), `≥1.10× pp+tg` gate at `{512,1024,2048,4096,8192}`, and `10×/15×` rigour (thermal-paired `RunStore` + `CHECKSUMS` batch).**
