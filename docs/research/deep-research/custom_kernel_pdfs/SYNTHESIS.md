# Synthesis: Custom Kernel PDFs → Phase 7 Actionable for gfx1100 @ 8k, 10% Uplift, Windows, 10×/15×

**Inputs (3 PDFs in `E:/Downloads/custom_kernel/` → `docs/research/deep-research/custom_kernel_pdfs/*.txt`):**
- `3710848.3710871.txt` (118K, 993 lines) — **MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on LLMs** (Frantar et al., ISTA/ETH/Neural Magic, CCS Parallel Algorithms). *Batch 16–32 close to 4×, 64–128 graceful, 2.8× vLLM, async + pipelining.*
- `ICLR-2024-lut-gemm-...txt` (75K, 1230 lines) — **LUT-GEMM: Quantized Matrix Multiplication Based on LUTs for Efficient Inference in Large-Scale Generative LMs** (Naver AICs, ICLR 2024). *BCQ + LUT, no dequant, 2.1× vs OPTQ at 3-bit on OPT-175B single GPU.*
- `xiao23c.txt` (73K, 770 lines) — **SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs** (MIT Han Lab, ICML 2023). *W8A8, 1.56×, 2× mem, 530B in 1 node, α=0.5/0.75 smoothing.*

All extracted via `pdftotext -layout` (no truncation). Below is the **distilled kernel design language** for our `Phase 7: Hybrid DP4A & WMMA` re-scoped to `≥10%` + `Windows ≤2 langs` + `10×/15×`.

---

## 1. MARLIN — the batched prefill blueprint we are missing

**Thesis:** Weight-only 4-bit still wins at batch >>1 *iff* you stay memory-bound by overlapping everything. On Ampere `FLOP/Byte≈100–200 FP16`, 4-bit gives `~25–50 ops/weight` headroom → **batch 16–32 still gets ~4×**, 64–128 gradual fall, not cliff. Prior kernels die at `batch=4`.

**Core techniques (directly portable to RDNA3 `gfx1100` WMMA):**

| MARLIN on Ampere | gfx1100 translation (our `impl_gemm_wmma_stream`) | Action for Phase 7 |
|---|---|---|
| `cp.async` GMEM→SMEM (bypass RF) + `L1 evict_first`, `L2` overlapping, `cp.async` even for *synchronous* loads to avoid compiler reorder | `ds_read2b` / `global→LDS` async via `buffer_load` + `sched_barrier 0x0080` overlap, keep `L2 6MB` hot for `A` while `B` streams `GDDR6` | **Depth-4 pipeline**: prefetch `A_sm` & `B_sm` `P-1` steps ahead + double-buffer `SMEM→registers` (currently depth-2 `[2][32][33]`; push to 4). Requires `__syncthreads()` + static unroll (our `TILE_K=32=2×WMMA` already halves; extend to `4` and unroll `K_tiles/4`). |
| `16×64 tiles` reshuffled offline to be contiguous (weights static) → optimal `128B cache line` loads, `SM=64/128/256` wide tiles for `batch=N` | Same: preprocess `IQ4_XS` `block_iq4_xs.h` layout offline in `gguf-py` to `16×64` swizzled order; our `X[gm*K+gk]` stride fix is step 1, but still strided. Add `python tools/swizzle_iq4xs.py` that emits contiguous `16×64` `.bin` | **Offline swizzle** + `sB[2][32][33]` stays, `sA` for `Q8_1` activations also swizzled. |
| **Partition `C` into `M×sm`** (`sm∈{64,128,256}`) assigned per `SM`, `warp-level` `wa×wa` (`wa=16`=Tensor Core) → `log-reduction` | `64×32` per block (`4×2` warps) is undersized for `8k` (256 tiles). Scale to **`64×64` or `128×64`** for `M=8192` (today `256×32` tiles → 256 blocks). Keep `256 thr/block` but `J=128→256` (see `mmq-config-rdna3.cuh` `J=256` case). | **Tile up** for `8192`: test `64×64` vs `128×32` with `N=10` `median` (REQ-STAT-07). |
| `Grouped quantization` (`g=128`) — scale per group, reorganized as `16B vector`, dequant overlapped with `mma.sync` `16×16×8` | Our `IQ4_XS` already group-wise (`ls-32` per `QK_K/64=4` scales). Overlap `d*(ls-32)*kvalues` dequant with *previous* `WMMA` (pipelined), not inline. | **Dequant pipeline**: compute `A_frag` for tile `i+1` while `WMMA` on tile `i` runs. |
| `2.8× vLLM` E2E, `3.2× Sparse-MARLIN` | Our `llama.cpp` analogue is `llama-bench pp` at `b=2048`; Marlin proves `batch 16–32 ≠ OOM` | Validate `pp8192` with `b=2048` *batched* not just single-stream. |

**Key quote (MARLIN §3.1):** *"memory loading will dominate runtime as long as batch < opt≈50 … `opt` is where latency is neither memory nor compute bound … we need to hide `P-1` steps via `cp.async`"* — our `P=2` hides 1 step; at `8k` you need `P=4` to hide `LDS` fill.

**Why it matters for `1000 tok/s @8k`:** Deep-research said `800 GB/s` roof + `67M` head `×64` layers = bandwidth-bound without overlap. MARLIN proves you can stay bound up to `batch 32` (~`8k` tokens ≈ `batch 32 × 256`? not directly). Our `WMMA 512 FLOP/clock/CU` is useless if `LDS` fill stalls; `P=4` removes that stall.

---

## 2. LUT-GEMM — eliminate the dequant tax entirely

**Thesis:** `W4A16` (weight-only 4-bit, act FP16) today does `dequant→FP16 → matmul` (Fig 1b) — extra pass. `W8A8` does `quant→INT8 → matmul` — also quantization cost. Both pay. `LUT-GEMM` (Fig 1c) uses **BCQ**: `w ≈ Σ_i α_i b_i` (`b_i∈{-1,+1}^n`, `α∈R+`), any `q-bit` uniform *or* non-uniform maps to BCQ (proof §3.2, bias term extension), and computes `B·x` via **LUT of `2^μ` precomputed activation combos** (`μ=3` → `8` values). No bit-level memory, no dequant.

**Kernel shape (Fig 2–3):**

- Pack `B` column `11..44` as `b'00111111` etc., offline.
- For each `μ=3` activation sub-vector `[x1 x2 x3]`, precompute `8` sums `a0…a255` into `LUT1..LUT4` (shared). Then `y = Σ LUT[ index(B_col) ]` — just `index → load → add`.
- `group-wise` (`g=64/128`) gives `compression vs accuracy` knob (§3.3) — `g=128` sweet spot (OPT-175B 3-bit 2.1× vs OPTQ).

**Results:** `OPT-175B 3-bit BCQ` **2.1× token gen latency** vs `OPTQ` (which dequantizes) on *single GPU*; reduces `GPU count` for power (§6).

**Action for `gfx1100` `IQ4_XS`:**

- Our `kvalues_iq4nl[16]` LUT is *already* a mini-LUT-GEMM (non-uniform codebook). But we still do `get_int_from_table_16 → dp4a` inline (dequant+DP4A). **LUT-GEMM says: precompute `FP16` LUT for `μ=4` (16 values) per `Q8_1` activation tile, then `LD` from LUT by `qs[4]` nibble, *no* `scale*(ls-32)` multiply in hot loop** — move `d*(ls-32)` to LUT build (once per `8192` row, offline).
- **Group-wise knob:** `QK_K/64=4` scales today → try `g=32` (finer) vs `g=128` (coarser) and publish `PPL` vs `tok/s` curve (`CTX-02` style but for `IQ4_XS`).
- **BCQ proof** means we can support *both* `Q4_K_M` and `IQ4_XS` with same LUT kernel (just change `α, bias`) — future-proof.

**Caveat:** LUT needs `2^μ` shared entries (`μ=4→16` half values = `32B` per `LUT`); for `M=8192` this is fine. For `μ=8` → `256` entries = `512B` — still `LDS` friendly. Our `[2][32][33]` can host `4×LUT` easily.

---

## 3. SmoothQuant — make `W8A8` viable for `8k prefill` without accuracy loss

**Thesis:** Activations outliers (`>70` in `OPT-13B` Fig 4) make `W8A8` per-tensor `32%` accuracy (Table 1: `OPT-13B 33.0%` vs `65.6% FP16`; per-channel would be `65.6%` but *not* GEMM-compatible — scalings only on outer dims `T, Co` via `Y = diag(s)·(X_int8·W_int8)·diag(s)` Eq 2). Instead **migrate difficulty offline**: `Y = (X diag(s)^{-1})·(diag(s)W) = X̂·Ŵ` (Eq 3), choose `s_j = max|X_j|^α / max|W_j|^{1-α}` (Eq 4). `α=0.5` evenly splits (0.75 for `GLM-130B` with `30%` outlier channels). Smooths activation `>70→~1`, weight stays flat (Fig 4 after). **Mathematically equivalent**, fuse `diag(s)` into previous `LN`/`Linear` offline — no runtime.

**Settings `O1→O3` (Table 2):** `O1` `per-tensor→per-token dynamic` (safe), `O2` `per-tensor dynamic→per-tensor dynamic` (faster), `O3` `per-tensor static` (fastest, `1.56×` on `FasterTransformer` vs `1.51×` PyTorch, `1.96×` mem). `O3` holds `OPT-175B` `66.9% avg`=`66.9% FP16` on `7` zero-shot + `WikiText 10.99` (§5.2).

**Action for `gfx1100`:**

- **Our `Q8_1` activation quant (`block_q8_1 36B/32`) is *already* per-token dynamic** (`quantize_mmq_q8_1_cuda` per `M` row). SmoothQuant says: *smooth first, then per-token becomes per-tensor-friendly*. Add **offline `s` calibration** (`512` Pile sentences, like paper) → `smoothed` `X̂` → `Q8_1` error `↓` → `DP4A` `INT8` throughput `↑` (since `effective bits` `≈8` for all channels, not `2–3` for non-outlier channels due to `max≈70` outlier dominance).
- **Fuse `s` into `rmsnorm` weight**: `rmsnorm(x·diag(s)^{-1})` is cheap (one `÷s` per channel). No extra kernel.
- **Try `W8A8` as alternative to `IQ4_XS→FP16` for `8k prefill`:** `W8A8` halves `KV` `+` weight mem (`2×` saving), doubles `GEMM` `INT8` throughput (`CUTLASS INT8 GEMM` on `RDNA3` via `WMMA i32_iu8`). Evaluate `SmoothQuant-O2` vs our `IQ4_XS+DP4A` for `pp8192` — may beat `10%` gate more easily than `IQ4_XS` alone (since `Q8_1` `1.125 B/elem` vs `FP16 2B` is `1.7×`, `W8 1B` vs `IQ4 0.56B` is loss but compute `INT8` is `2×`).
- **Alpha sweep:** `α∈{0.5,0.6,0.75}` grid search on `Pile` val (paper did fast grid) — report `PPL` vs `tok/s` for `Qwen3.8-27B` (our `PPL 6.4271` gate). `0.5` is start.

**Perf note:** `INT8` halved mem → `800 GB/s` effective `1.6 TB/s` logical, directly attacks `8k` roof.

---

## 4. How the three combine for the `≥10% / 1000 tok/s @8k` + `Windows` + `10×/15×` gates

**Stack (in order of kernel pipeline):**

1. **Offline:** `SmoothQuant s` (`α=0.5`) calibrate `512` Pile → fuse `s` into `rmsnorm`/`Linear` → activation `X̂` outliers `70→1` → `Q8_1` per-token error `↓` → `INT8` GEMM accuracy `≈FP16` (no `O3` collapse).
2. **Offline weights:** Two parallel paths (keep `IQ4_XS` as primary for `20 GB`; add `W8` SmoothQuant as `comparator arm` for `pp8192`):
   - `IQ4_XS` `4.25 bpw`: swizzle `16×64` offline + `kvalues` → **LUT** build (`μ=4`, `16` FP16 per `LUT ×4`) → no inline `dequant`.
   - `W8` (`SmoothQuant Ŵ`): per-channel `INT8` (already `Q8_1` compatible) → `INT8` `WMMA` `i32` path.
3. **On-the-fly (prefill `M=8192`):** `quantize_row_q8_1_coop` (`8-thread coop`) → `Q8_1` `36B/32` → `LDS [2][32][33]` **depth-4 pipeline** (`cp.async`-style `ds_read` prefetch `P=4`, unrolled) → `dequant` (if not LUT) overlapped with previous `WMMA` → `WMMA 64×64` (or `64×32×4`) `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` / `i32_iu8` → `log-reduction` across `wa` (Algo 1 style) → `Y`.
4. **Decode `M=1`:** Keep `8-thread coop GEMV` `sudot4`+`perm` (single-warp `MMVQ` is bottleneck; MARLIN shows `batch 1` is still `memory bound` — our `coop` is right), but add `SmoothQuant` smoothing for `Q8_1` `amax` stability.

**Expected stack speedup vs `real DP4A` stock MMQ (not naive):**

- `SmoothQuant` alone `1.5×` (paper) → `1.2×` realistic on `gfx1100` `WMMA` vs `VALU DP4A`.
- `MARLIN P=4` overlap `1.15×` (hiding `LDS` fill `15–30 µs` DXG tax) — moves `peak 1.178× avg 1.00` (WSL jitter) → `1.25×` bare metal.
- `LUT` eliminates `dp4a` table lookup `+ scale` `~5%` (`1.05×`).
- **Multiplicative `≈1.5×` microbench** → translates to `≈1.12–1.15×` E2E `pp8192` (since attention `50–73%` GEMM with `Flash2` limits). Pushes `808→~950 pp4096` (`+17%`) → passes `≥1.10×` `pp+tg` at `{512…8192}` (`REQ-PERF-07`).

**Windows-native (`REQ-WIN-07`):** All three are **pure `C++/HIP`** (CUTLASS → `hipCUTLASS` analog is `rocWMMA`/`CK`, already header-only; LUT is `__shared__ half LUT[16]`; Smooth `s` is host Python *offline only*, not shipped). `build_windows.bat` (`HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja`) stays `≤2` langs.

**Statistical rigour (`REQ-STAT-07`):** Each technique gets its own `N=10` `median/mean/stddev/p95` table (`bench_gemv_dp4a --runs 10`, `bench_gemm_wmma --runs 10`, `bench_lut_gemm --runs 10`, `bench_smooth_w8a8 --runs 10`) + `15×` LLM QA (`llama-cli -p "Hi" -n 128 --temp 0` repeated `15×`, `avg tok/s` + per-run). Single-run `2.1×` / `1.56×` claims from papers are *not* our verdict; our `median−1σ ≥1.10×` is.

---

## 5. What to change in Phase 7 plans tomorrow (when you are ready)

- **07-02 (`GEMV`):** Add `SmoothQuant s` calibration script (`tools/calibrate_smooth.py` offline, not shipped) and `coop quantize` `s`-aware variant; add `P=4` prefetch unroll experiment as `variant B` (keep `P=2` as `A` for A/B).
- **07-03 (`GEMM`):** Add `LUT` variant (`impl_gemm_lut_iq4xs.hip` `μ=4` `16-entry LUT`) as second kernel under same `64×32` tile; add `MARLIN P=4` pipeline as `variant B`; tile sweeps `64×32` vs `64×64` vs `128×32` with `N=10` `median`.
- **07-04 (`A/B`):** Split `pp/tg` `≥1.10×` gate into **three comparator arms**: `stock DP4A` (real `vec_dot`), `IQ4_XS LUT+WMMA P=4`, `W8A8 SmoothQuant INT8 WMMA` — report which passes `10×` `pp+tg` at `8192` (if VRAM pre-flight `>2 GB` free, else `SKIPPED` with `FA` rationale). Add `build_windows.bat` `llama-server :8000` smoke with `15×` `curl` averaged.
- **07-CONTEXT.md:** Add `§ MARLIN/LUT/SmoothQuant` design notes + `α` sweep table placeholder.

**Not changing now** — awaiting your go-ahead after you review this synthesis + the `docs/research/deep-research/1000t-s-at-8k-gfx1100.md` cliff notes.

---

## 6. Open questions for you before we edit

1. **W8A8 vs IQ4_XS priority?** Keep `IQ4_XS 15.31 GB` as locked artifact (current) and treat `SmoothQuant W8A8` as *comparator arm only*, or allow `W8` as alternative for `8k` if it wins `≥10%`?
2. **LUT μ?** `μ=4` (`16` entries, `32B/LUT`) is safe; `μ=8` (`256` entries, `512B/LUT`) more compute saving but shared pressure — start `μ=4`?
3. **Offline swizzle?** Approve `tools/swizzle_iq4xs.py` that rewrites `IQ4_XS` layout to `16×64` contiguous (one-time, not runtime)?
4. **Commit now?** `9` files `M` for `re-scope` are still unstaged — commit before swizzle experiments or keep unstaged until `MARLIN/LUT/SmoothQuant` variant lands?

*Files:* `docs/research/deep-research/custom_kernel_pdfs/*.txt` (raw), `E:/Downloads/custom_kernel/*.pdf` (source), this synthesis (`docs/research/deep-research/custom_kernel_pdfs/SYNTHESIS.md`), plus `docs/research/deep-research/1000t-s-at-8k-gfx1100.md` (40-source cliff report).

