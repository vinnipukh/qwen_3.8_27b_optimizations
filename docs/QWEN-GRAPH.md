<!-- generated-by: gsd-doc-writer -->
# Qwen3.8-27B Graph-Aware Report — Phase 10 Light (v2)

> One-page graph snapshot. Deep 64-layer tensor dump deferred to v2 per `.planning/ROADMAP.md` Merge Map | Orig Phase 10 → Phase 5 (graph-aware target choice); tensor-report deep-dive → v2.

## Model Snapshot

- **Artifact:** `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` — 15.31 GB, IQ4_XS, imatrix, sha256 `53adc4bb…` (`.planning/research/MODEL-DECISION.md`)
- **Base:** `Qwen/Qwen3.8-27B` (`qwen35` / `Qwen3_5ForConditionalGeneration`), Apache 2.0
- **Hybrid stack:** 64 text layers = **48× Gated DeltaNet** (SSM, conv_kernel 4, state_size 128, groups 16) + **16× gated full-attention** (full_attention_interval 4)
- **Dims:** hidden **5120**, FFN intermediate **17408**, 24 heads / 4 KV heads, key/value_length 256, ctx **262,144**, KV cache **~64 KiB/token** (f16 est.) — Phase 9 / attention tuning v2-deferred (tiny KV)
- **Graph cost:** ~497 `MUL_MAT` nodes/token (S1: 497 prefill, 63616/128 decode) across 64 layers; FFN dominates FLOPs/bytes.

## Canonical Projection Tensors (Top 8 — MUL_MAT payload)

All weights verified via `tools/dump_matmul_fixtures.py` + `kernels/matmul_iq4xs/ref_cpu.h` (gguf-py probe: `blk.0.ffn_gate 5120×17408 IQ4_XS`, `blk.0.ffn_down 17408×5120 IQ4_XS`, `blk.0.attn_gate 5120×6144`, `blk.3.attn_q 5120×12288 Q5_K fused` sliced to canonical).

| tensor_name | shape (K×N) | dtype | quant | %runtime | kernel | frequency |
|---|---|---|---|---|---|---|
| `blk.*.ffn_gate.weight` | 5120×17408 | IQ4_XS | 4.25 bpw (136 B/256) | **31.12% agg.** — 50.89% prefill / 30.04% decode | **MMVQ** (M=1) → `impl_gemv_gfx1100.hip` / **MMQ** (M≫1) → `impl_gemm_wmma.hip` | 64×/forward (1/layer) |
| `blk.*.ffn_up.weight` | 5120×17408 | IQ4_XS | 4.25 bpw | same agg. | MMVQ / MMQ (hipBLAS excluded gfx1100) | 64×/forward |
| `blk.*.ffn_down.weight` | 17408×5120 | IQ4_XS | 4.25 bpw | same agg. | MMVQ / MMQ-wrapped WMMA | 64×/forward |
| `blk.*.attn_gate.weight` | 5120×6144 | IQ4_XS | 4.25 bpw | same agg. | MMVQ / MMQ | 64× GDN proj (or 16× full-attn) |
| `blk.*.attn_q.weight` | 5120×5120* | IQ4_XS (*fused `attn_qkv` 5120×12288 Q5_K on blk.0, sliced) | 4.25 bpw / Q5_K fused | same agg. | MMVQ (decode) / MMQ | 16× full-attn layers |
| `blk.*.attn_k.weight` | 5120×5120 | IQ4_XS (fallback) | 4.25 bpw | same agg. | MMVQ | 16× |
| `blk.*.attn_v.weight` | 5120×5120 | IQ4_XS (fallback via `ssm_out`) | 4.25 bpw | same agg. | MMVQ | 16× |
| `blk.*.attn_output.weight` | 5120×5120 | IQ4_XS | 4.25 bpw | same agg. | MMVQ | 16× |

* `*` — `attn_q 5120×12288` in task = fused `blk.0.attn_qkv.weight` (Q5_K) row-sliced to canonical `5120×5120` IQ4_XS for the microbenchmark harness (`TENSOR_CANDIDATES` map). Per-tensor GGUF storage: `ffn_gate` 46 MB / `ffn_down` 46 MB / `5120×5120` 14 MB (W_raw, `manifest_matmul.json`).

**%runtime source:** `benchmarks/profiling/bottleneck_summary.json` + `BOTTLENECK-TABLE.md` — MUL_MAT #1 aggregate 88,033 ms. Decode 195–223 µs avg (S1 195.0 µs, S2 217.1 µs); prefill 425–1065 µs avg (S1 425.9 µs, S3 1064.7 µs). Bound: **Memory Bandwidth / Dequant** (IQ4_XS dequant + GEMV/GEMM). F32 activations (`x`/`X`) are `F32` (not quantized); KV cache `F32`/`F16` (q8_0 option >8k ctx).

## Why MUL_MAT is Target #1 (not GDN)

| Op | %total | %prefill | %decode | Bound | Vertex |
|---|---|---|---|---|---|
| **MUL_MAT** | **31.12%** (88,033 ms) | **50.89%** (7,451 ms) | **30.04%** (80,582 ms) | Dequant + memory BW | All 64 layers × 4–8 proj |
| GATED_DELTA_NET | 2.25% (6,377 ms) | 4.55% (667 ms) | 2.13% (5,710 ms) | Compute / Register | 48 layers only |

**13.8× total gap** (31.12/2.25), **11.2× prefill**, **14.1× decode**. GDN is rank #11; even perfect elimination recovers ≤2.3% — below the noise floor of dispatch overhead (HIP Graphs alone give +5–19% decode per `dispatch_overhead_report.md`). MUL_MAT spans **both** phases and scales with `K×N`: FFN `5120×17408` (89 M params each) dominates bytes (2.72 MB per `5120×5120`, ~9 MB per `5120×17408` W). Custom kernels prove headroom: **GEMV 1.26–2.13×** (decode) and **GEMM 1.7–7.5× at M≥128** (prefill WMMA `v_wmma_f32_16x16x16_f16` confirmed in disasm) vs naive scalar HIP baseline (`benchmarks/profiling/KERNEL-BENCH-DIFF.md`). GDN optimization is v2 (needs VGPR-spill fix `__launch_bounds__`; +24% community report, unproven on gfx1100 HIP vs CUDA-first kernels).

## Kernel Mapping (gfx1100)

- **Decode M=1 → MMVQ** (`ggml-cuda.cu` `vec_dot` path) → custom `gemv_gfx1100.hip` (uint4 loads, zero-LDS dequant, `__shfl_xor` reduction) wins 8/8 (1.26× FFN, 2.13× ffn_down).
- **Prefill M≫1 → MMQ** (tiled) → custom `gemm_wmma.hip` (`TILE_M=16` + WMMA 16×16×16) wins 6/6 at M≥128 (6.7–7.5× at M=512); 2 losses at M=16 (overhead amortized at target `M≥128`).
- **hipBLAS/rocBLAS:** excluded for decode (no small-M Tensile coverage), optional large-M comparator only; hipBLASLt excluded (no gfx1100).

## Sources

- `MODEL-DECISION.md` — 5120/17408/262k/64 KiB, 48+16 hybrid
- `BOTTLENECK-TABLE.md` + `bottleneck_summary.json` — 31.12% / 50.89% prefill / 30.04% decode, GDN 2.25%
- `ROADMAP.md` Merge Map — Phase 10 → Phase 5 + v2 deep-dive
- `ref_cpu.h` / `KERNEL-BENCH-DIFF.md` — 8 shapes, kernels, speedups

*Not a 64-layer dump — v2.*
