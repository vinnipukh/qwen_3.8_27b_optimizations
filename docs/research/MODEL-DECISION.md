# Model Decision Record

**Status:** LOCKED by owner, 2026-08-21
**Supersedes:** the "model PENDING-USER / does-not-exist" status in STACK/SUMMARY research notes (those reflected the researchers' sources predating awareness of this release family).

## Locked artifact (primary)

`JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` → **Qwen3.8-27B-Uncensored-IQ4_XS.gguf**

- Size: **15,309,039,008 bytes (15.31 GB)** · sha256 `53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5`
- Chosen over Q4_K_M (16.81 GB) specifically to preserve context-length headroom on the 20 GB card
- Built with llama.cpp `a94d563ed`; imatrix embedded (wikitext-2, 200×512 chunks); MTP tensors verified present (65/65 blocks)
- Published wikitext-2 PPL 7.1583 ±0.25019 (vs f16 baseline 7.1557 — inside noise; quantization damage ruled out)

## Base model facts (verified from HF pages, 2026-08-21)

- Official base: `Qwen/Qwen3.8-27B`, Apache 2.0 — **the model name is real**; earlier research notes claiming otherwise are corrected here.
- Architecture: `qwen35` / `Qwen3_5ForConditionalGeneration` — HYBRID linear attention:
  - 64 text layers total: **48 Gated DeltaNet** (SSM-family, conv_kernel 4, state_size 128, groups 16) + **16 gated full-attention** layers (full_attention_interval 4)
  - Attention dims: 24 heads / **4 KV heads**, key_length 256, value_length 256
  - Consequence A: KV cache is small (~64 KiB/token f16 est.) — Phase 9 de-prioritized accordingly
  - Consequence B: **Gated DeltaNet kernels on HIP/gfx1100 are the critical unknown and prime custom-kernel target** — new-arch kernels typically land CUDA-first; Phase 1 must validate HIP coverage before anything else
- Native MTP/NextN head (1 layer) → speculative decoding is a first-class runtime feature (`--spec-type draft-mtp`, PR #22673+); relevant later as decode-path accelerator and benchmark dimension
- Vision-capable (mmproj/projector separate); not a project priority
- Context: 262,144 native

## Secondary / comparator artifacts (optional, not downloaded at start)

| File | Size | Role |
|---|---|---|
| JonathanColetti `Q4_K_M.gguf` | 16.81 GB | quality comparator quant |
| JonathanColetti `draft-Q8_0.gguf` | 3.16 GB | explicit-draft spec-decode experiments |
| HauhauCS Aggressive IQ4_XS | 15.71 GB | serving-variant eval only; NOT baseline (custom K_P recipe + patched-runtime FastMTP conflict with stock-baseline methodology) |

## VRAM envelope estimate (IQ4_XS on RX 7900 XT 20 GB)

weights 15.31 + KV @f16 (~64 KiB/token → ~2.0 GB @32k) + compute/runtime buffers ~1–1.5 GB ⇒ ~18.3–18.8 GB @32k native-Linux (tight), comfortable ≤16k. WSL2 DXG deficit measured −2.9 GB on XTX (assume ~17 GB usable on our card until probed) ⇒ @32k under WSL2 likely OOM/silent-overcommit; plan BENCH-04 accordingly (expected-fail path or 24k cap). Estimates tagged MEDIUM — verify against llama.cpp startup breakdown in Phase 1.

## Uncensoring provenance (for the record)

Heretic abliteration of official base; measured refusals 12/100 vs base 98/100; mean capability delta −0.5 (MMLU/ARC/HellaSwag/Winogrande); KL(first-token) 0.1191. Refusal-boundary behaviour near old refusal directions is the least stable property — irrelevant to kernel work but noted for eval design.
