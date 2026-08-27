# Phase 4: Kernel Playground Scaffold - Discussion Log

**Date:** 2026-08-25
**Phase:** 4 (KERN-01) — Ready for planning → Planning

## Decision Log (owner step-by-step, 5/5 A)

### Q1/5 Demo op
**Q:** `dequant_iq4_xs` only vs fused `dequant+dot` GEMV vs both for the 04-03 walkthrough?
**Options:** A) dequant-only (isolates bit layout, fastest bug-catch), B) fused GEMV (closer to Target #1 MUL_MAT 31.12% wall time), C) both
**Owner:** **A** — dequant-only for 04-03, fused deferred to Phase 5. Edge fixtures still cover `4096×4096` shapes.
**Rationale recorded in 04-CONTEXT D4-00-1:** 136B determinism 3-way verified, `dequantize_row_iq4_xs` oracle exists (ggml-quants.c), wmma_ops same gate validated 41.3 TFLOPS — no matmul needed to isolate risk.

### Q2/5 Struct source
**Q:** How to get `block_iq4_xs` without llama.cpp headers in `kernels/`?
**Options:** A) vendored copy `kernels/common/block_iq4_xs.h` with `static_assert(136)` + attribution, B) git submodule `extern/ggml`, C) system `-I /root/llama.cpp`
**Owner:** **A** — vendored copy. Gate `rg -r "ggml|llama" kernels/` → 0.
**Reference:** `ggml/src/ggml-common.h @ bb4caa75`, `ggml/src/ggml-quants.h:140`, `GGML_QUANT_SIZES[IQ4_XS]=(256,136)`.

### Q3/5 Result store
**Q:** Reuse `benchmarks/lib/store.py` vs new `kernels/results/` loose CSV?
**Owner:** **A** — reuse `store.py` (fingerprinted `benchmarks/results/kernels_*/` + `kernels/results/` symlink). One truth, Phase 6 can join stock vs custom; mirrors Magpie `analyze_report.json` discipline.

### Q4/5 PASS gate
**Q:** Tight `max_abs 1e-5 / mean 1e-6 / cosine 0.99999` + 10× broken discrimination?
**Owner:** **yes** — tight for dequant (lossless unpack ~1e-7), will loosen to `1e-3 / 0.999` for fused `MUL_MAT` in Phase 5.
**Edge cases locked:** zero, min/max `ls 0→-32 / 63→+31`, all nibbles 0/15, split-half `lo@i vs hi@i+16`.

### Q5/5 Wave handling
**Q:** Templated `WARP_SIZE` vs hard-code 64?
**Owner:** **A** — templated `template<int WarpSize>` + `__launch_bounds__(256,4)` + `--save-temps` VGPR gate, bench both wave32 (native 32-wide 1-cycle) and wave64 (emulated 2×32) per kernel. Never literal `32`/`64`.

## Research Pass (pre-planning, owner-requested)

**Triggered by:** owner flag to mine `@.planning/reference/GPU-KERNEL-RESOURCES.md` (15 resources, §1–5) + re-verify `@.planning/research/EXTERNAL-RESOURCES-ASSESSMENT.md` before drafting.

**Method:**
- 6 parallel subagents (amd-isa, hip-guides, ck-lib, gpuopen, quant-kernels, external-assessment) + 8 direct `fetch_content` (performance_guidelines, occupancy-explained, vecdotq.cuh 49KB, composable_kernel, live-vgpr, etc.) + prior `scaffold`/`iq4xs`/`bench` trio.
- All 6 reports completed: `amd-isa.md`, `hip-guides.md`, `ck-lib.md`, `gpuopen.md`, `quant-kernels.md`, `external-assessment.md` under `subagent-artifacts/outputs/83332a43-...` + `faca2a56-...`.

**Key findings folded into CONTEXT:**
- RDNA3 ISA 70650 is canonical (VOPD 8B dual-issue, WMMA 16×16×16 wave32 32-cycle, VGPR granule +50% — amd-isa).
- Standalone HIP via `CMAKE_HIP_ARCHITECTURES=gfx1100` / `hip::device`, coalescing 128B, LDS 32-bank pad `[32][33]`, `__launch_bounds__` occupancy caps, WGP vs CU mismatch #3374 (hip-guides).
- CK Tile reference-only (heavy, Instinct xdl, shim pattern `ck_tile_shim.h`) — hand-roll minimal HIP in Phase 4 (ck-lib).
- Wave32 native vs Wave64 2-cycle, 16 slots/SIMD, 1536 VGPRs/SIMD, RGA `--livereg` (gpuopen).
- `wmma_ops` lane-replicated fragments, `vecdotq.cuh` HIP `__builtin_amdgcn_perm`, IQ4_XS 136B/8×32 + Marlin 128-bit pipeline as Phase 5 pre-read (quant-kernels).
- Magpie discipline (explicit `atol/rtol`, identical warmup/iterations, `--baseline 0`) mine; `rocm-doctor` WSL2 decline confirmed; Hyperloom MI300-only gfx942/g950 (external-assessment).

**Curator note:** One `web_search` curator session returned `0 results` (JS-rendered Perfetto/RGA pages timing out) — covered by `fetch_content` fallbacks and gpuopen agent; no gap.

## Verification

- All 5 locks recorded in `04-CONTEXT.md` D4-00-1..5 with citations.
- Deep research consolidated in `04-RESEARCH.md` (6 subagents + direct fetches, gaps listed).
- 3 plans drafted: `04-01-PLAN.md` (scaffold+build), `04-02-PLAN.md` (fixtures), `04-03-PLAN.md` (demo+negative test).

## Next Step

Execute plans 04-01 ∥ 04-02 → 04-03, then `subagent_wait --all` for build/test artifacts.

