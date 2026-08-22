# Model Artifact Provenance

**Primary artifact (LOCKED):**

| Field | Value |
|---|---|
| HF repo | `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` |
| File | `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` |
| Size | 15,309,039,008 bytes (15.31 GB) |
| sha256 | `53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5` |
| Verified | 2026-08-22 — `sha256sum -c` OK in WSL guest (Phase 1, plan 01-03 T1) |
| HF revision | `dee0a3164d9e11bbbebf5b63f52ba99443d14fc3` (lastModified 2026-08-16) |
| Quantizer | llama.cpp @ `a94d563ed` |
| Imatrix | embedded — wikitext-2, 200×512 chunks (`/workspace/gguf/Qwen3.8-27B-Uncensored-imatrix.dat` at build time) |
| Base model | `Qwen/Qwen3.8-27B`, Apache-2.0 |
| Architecture | `qwen35` hybrid: 64 layers = 48 Gated DeltaNet + 16 gated full-attention (interval 4); ctx 262,144 native; MTP/NextN head present (65/65 blocks) |
| Download URL | `https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` |

**Notes**
- Published wikitext-2 PPL 7.1583 ±0.25019 (f16 baseline 7.1557) — see `.planning/research/MODEL-DECISION.md`
- This file is gitignored (`models/*.gguf`); this README is the provenance of record
- Companion artifacts available in same repo if later needed: `noMTP-IQ4_XS` (15.08 GB), `draft-Q8_0` (3.16 GB, v2 spec-decode), `imatrix.dat`
- Heretic maintainer flags down_proj ablation as potentially intelligence-damaging; artifact ships with zero generative/code evals — coding-capability eval deferred to v2 (see REQUIREMENTS.md)
