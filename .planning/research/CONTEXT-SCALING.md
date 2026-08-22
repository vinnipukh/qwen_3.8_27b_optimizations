# Context Scaling Research — 256k Target

**Date:** 2026-08-22 · **Trigger:** owner request to extend usable context toward 256k
**Feeds:** REQUIREMENTS.md CTX-* section, future phase planning

## Why 256k is plausible on this card at all

Qwen3.8-27B is a hybrid: **48 Gated DeltaNet layers** (recurrent state — constant
size per sequence, independent of context length) + **16 gated full-attention
layers** (full_attention_interval 4). Only the latter grow a KV cache.

KV bytes/token (verified against model config): 16 layers × 4 KV heads ×
256 head_dim × 2 (K+V) × 2 B (f16) = **65,536 B/token ≈ 64 KiB/token**.

Native trained context: **262,144** — rope/YaRN extension NOT required.

## The memory wall (IQ4_XS baseline, RX 7900 XT)

Budgets: ~20 GB native · **~17 GB under WSL2 DXG** (−2.9 GB measured on XTX,
assume until probed). Fixed costs: weights 15.31 GB + compute/runtime buffers
~1.0–1.5 GB ⇒ free-for-KV ≈ **0.5–1.5 GB (WSL)** / **~3 GB (native)**.

| Context | KV f16 | KV q8_0 | KV q4_K* |
|---|---|---|---|
| 32k | 2.0 GB | 1.0 GB | ~0.55 GB |
| 64k | 4.0 GB | 2.0 GB | ~1.1 GB |
| 128k | 8.0 GB | 4.0 GB | ~2.2 GB |
| **256k** | **16.0 GB** | **8.0 GB** | **~4.4 GB** |

*q4_K ≈ 4.5 bits/elem effective incl. scales; verify empirically (CTX-02).

Verdict per arm:
- **WSL2/HIP:** 32k fits only with quantized KV; 64k needs q4-class KV;
  128k/256k **cannot fit** by KV quantization alone.
- **Native/Vulkan:** same shape, ~1.5–2 GB more slack — still not 256k.
- **Weights lever:** dropping to an IQ3-class artifact (~12.5–13 GB) frees
  ~2.5 GB — combined with q4_K KV, 256k lands at ≈18.5–19 GB: marginal native
  fit, still over the WSL line. Quality gate (QUAL-02) must arbitrate.

⇒ **256k under WSL2 requires host-tiered KV** (hot window in VRAM, cold prefix
in system RAM). 32 GB host RAM comfortably holds a 4.4–8 GB cold prefix.

## Host-tiered KV design sketch (FreeToken-informed)

- Layer-scoped: only the 16 attention layers page; DeltaNet states stay resident
  (constant size, tens of MB total).
- Hot set = recent W tokens (e.g., 16–32k) + any pages touched by recent
  attention spans; cold prefix lives in pinned host memory as contiguous
  per-layer segment rings (append-only during decode — friendly to bulk DMA).
- Migration policy candidates: (a) naive recency ring, (b) attention-pattern-
  aware recall (needs FA span metadata), (c) static split tuned by benchmark.
- WSL2 blocker: **pinned host memory is disabled by default** (librocdxg).
  Probe order: (1) locate enabling knob/env var, (2) fall back to pageable
  transfers (works, slower background fills), (3) native-Vulkan arm uses
  non-blocked pinned path as comparator.
- Correctness: identical logits expected (pure data movement) — QUAL gates apply
  unchanged; performance risk is recall stalls on long-range attention into the
  cold tier.

## Prefill reality at 256k

Stock pp on gfx1100 will make 256k prompt ingestion expensive (minutes).
Required mitigations, in order: chunked-prefill timing curve (already ROADMAP
Profile C), persistent prompt cache across runs (`--prompt-cache`), and — for
agentic edit patterns — **DeltaNet state checkpoints at semantic anchors**
(thinking blocks / tool-call boundaries) so edits recompute only the suffix.
That last item is FreeToken §3.1 transplanted to this model's recurrent layers.

## Open decisions for owner

1. Accept IQ3-class weight downgrade for the 256k arm, or hold IQ4_XS and cap
   at 128k until host tiering lands?
2. Is 256k a v1 exit criterion, or v1 = measured ceiling + tiering prototype?
3. Priority vs MTP/speculative-decode track (both touch decode path benchmarks).
