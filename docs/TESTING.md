<!-- generated-by: gsd-doc-writer -->

# Testing & Quality Gates

Quality is enforced by two independent gates plus a measurement protocol. A kernel or code change
is not "done" until every applicable gate passes and the numbers are published — failures are
published too.

## Gate tiers

| Tier | Tool | When required |
|---|---|---|
| Op-level | `test-backend-ops` (ROCm0 backend) | Green **before any performance claim** is accepted |
| Model-level | `llama-perplexity` (wikitext-2) + golden-output decode | Before any change is declared a win end-to-end |

Both binaries exist as archived stock builds at `baseline/binaries/v0.2.0-bb4caa75/`.
Phase 1 evidence of a green op-level run: `benchmarks/environment/test-backend-ops-phase1.txt`.

## Op-level gate

`test-backend-ops` must pass on the ROCm0 backend for the current build before any benchmark row
from that build is considered valid. Run it from the guest-side source tree (`/root/llama.cpp`)
or against the archived binary.

## Model-level gate

Two checks, both against the frozen model
`models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (sha256-verified; gitignored):

1. **Perplexity**: wikitext-2 perplexity within ±1% of the published reference **7.1583**
   (published tolerance band 7.1583 ± 0.25).
2. **Golden outputs**: fixed-prompt greedy decode (`--temp 0`) must match recorded baseline
   outputs within tolerance.

Any deviation beyond tolerance invalidates the optimization, regardless of speed.

## Numerical validation quartet (kernel work)

Each candidate op traverses the full pipeline and records four error metrics:

```
ref_cpu.cpp  →  impl_gfx1100.hip  →  test_compare.cpp  →  bench_sweep.cpp
(CPU ref)       (HIP kernel)         (error metrics)      (timing sweep)
```

Recorded metrics per comparison: **max-abs**, **mean-abs**, **relative**, **cosine** error.
These tables are part of the deliverable, not internal scratch notes.

## Benchmark protocol rules (binding)

- **pp/tg split mandatory** — prompt processing (prefill, M≫1) and text generation (decode,
  M≈1) are measured and reported separately. Blended tok/s is banned everywhere.
- **Warmup runs** precede all timed measurements.
- **≥3 repeats** per measurement; report variance, not single-shot bests.
- **Fingerprint every result row**: llama.cpp commit, ROCm/driver versions, GGUF sha256,
  clocks/temps from Windows-side telemetry (guest `rocm-smi`/`amd-smi` do not work under
  ROCDXG).
- **VRAM ledger + RSS guard** per run: process-RSS monitoring exists to defeat WSL2's silent
  VRAM-overcommit failure mode (spill to system RAM = 5–10× throughput collapse while tokens
  still flow). Fail-fast allocation policy — no retry loops (repeated GPU-OOM under WSL2 can
  hard-crash the host). Set `-c` explicitly, always.
- **Thermal windows**: final compared numbers must come from paired runs within one thermal
  window. Clocks are recorded, not controlled.

## Status: planned vs running

| Component | Status |
|---|---|
| Archived stock binaries incl. `test-backend-ops` | ✅ Running (Phase 1) |
| Environment fingerprint files | ✅ Running (`benchmarks/environment/`) |
| Automated harness (llama-bench wrapper, fingerprinting pipeline, RSS-guarded ledger) | ⏳ Planned — lands in Phase 2 |
| Golden-output capture automation | ⏳ Planned — Phase 2 |
| Baseline matrix publication | ⏳ Planned — Phase 2 |

Until the Phase 2 harness lands, measurements follow the manual protocol above with results
teed into `benchmarks/` subdirectories.
